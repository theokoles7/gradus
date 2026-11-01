"""# scratch_reweight.py

A third, structurally different mechanism from the two families built
earlier this session:

  * The stride/local-pool family controls *which samples end up in a
    batch together* (composition).
  * The temperature/hybrid family controls *sampling probability* over
    the dataset, still ultimately deciding batch composition.

Every one of those, however well-designed, risks the same failure mode
this session spent a long time diagnosing: changing composition risks
changing within-batch gradient diversity, and gradient diversity turned
out to be the actual causal mechanism behind curriculum harm (confirmed
via the analytic per-sample gradient-norm instrumentation in
scratch_stride_diagnostic.py). Composition and diversity are coupled by
construction in all three prior mechanisms.

This one decouples them completely: batches are plain, fully shuffled,
exactly like the shuffled baseline — zero composition curriculum, batch
diversity is always genuinely shuffled-equivalent by construction, not
just empirically close to it. Instead, curriculum bias is applied to the
*loss*: each sample's contribution to the batch's gradient is weighted
by its own difficulty (this step's per-sample loss, which the training
loop is already computing — no separate scoring pass, no probe, no
custom sampler needed at all, the cheapest mechanism built this
session), w_i = exp(-zscore(loss_i)/tau). tau -> 0 concentrates nearly
all gradient signal on the easiest examples in each batch; tau ->
infinity recovers exactly standard unweighted training.

Related to, but distinct from, Katharopoulos & Fleuret's importance
sampling (cited in the prospectus's related work): that work reweights
to reduce gradient *variance* toward a fixed optimum; this reweights
toward difficulty bias, with diversity loss and estimate staleness as
the things being tested, not minimized. It's also a continuous,
always-shuffled generalization of self-paced learning (Kumar et al.
2010), which used a hard inclusion/exclusion rule on the learner's own
loss rather than a continuous weight.

Adaptivity: instead of the grad-var-triggered probe used throughout the
other two families (which would be structurally uninformative here,
since batch composition never varies, so measured/probe gradient
variance would always be close to 1 regardless of tau), tau is driven by
Kish's effective sample size of the weights themselves: eff_fraction =
(sum w)^2 / (batch_size * sum w^2), self-referential, no external probe
needed. Same validated one-way, multiplicative, force-ceiling-guaranteed
philosophy as the other controllers, triggered when eff_fraction (i.e.
how many samples are still meaningfully contributing) drops too low.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing   import Any, Dict, List, Optional

import numpy as np
import torch
from torch                  import device as t_device, no_grad, Tensor
from torch.nn.functional    import cross_entropy
from torch.optim            import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data       import DataLoader
from tqdm                   import tqdm

from gradus.datasets         import Dataset
from gradus.networks         import Network
from gradus.registration     import DATASET_REGISTRY, NETWORK_REGISTRY
from gradus.utilities        import determine_device, get_logger, set_seed


class ReweightController:
    """# Adaptive Loss-Reweighting Controller.

    Holds tau and adapts it from the weights' own effective sample size,
    no probe, no live scoring pass, no custom sampler. See module
    docstring for full rationale.
    """

    def __init__(
        self,
        initial_tau:           float = 1.0,
        tau_step:               float = 1.5,
        tau_ceiling:            float = 50.0,
        eff_threshold:          float = 0.5,
        eff_smooth_window:      int = 3,
        force_ceiling_epoch:    Optional[int] = None,
    ):
        self._tau_ =            initial_tau
        self._tau_step_ =       tau_step
        self._tau_ceiling_ =    tau_ceiling
        self._eff_threshold_ =      eff_threshold
        self._eff_smooth_window_ =  eff_smooth_window
        self._force_ceiling_epoch_ = force_ceiling_epoch

        self._eff_history_: List[float] = []

    def weights(self, per_sample_loss: Tensor) -> Tensor:
        """# Turn This Batch's Per-Sample Loss into Curriculum Weights.

        z-scored within the batch before applying tau, so tau's meaning
        stays roughly consistent across the whole run even as absolute
        loss magnitude shrinks dramatically over training (same reasoning
        used for the temperature sampler's z-scored difficulty).
        """
        with no_grad():
            mean = per_sample_loss.mean()
            std = per_sample_loss.std() + 1e-8
            z = (per_sample_loss - mean) / std
            logits = -z / self._tau_
            logits = logits - logits.max()
            w = logits.exp()
        return w

    def weighted_loss(self, per_sample_loss: Tensor, w: Tensor) -> Tensor:
        return (w * per_sample_loss).sum() / w.sum()

    @staticmethod
    def effective_fraction(w: Tensor) -> float:
        """# Kish's Effective Sample Size, Normalized to [1/n, 1]."""
        eff_n = (w.sum() ** 2) / (w ** 2).sum()
        return (eff_n / w.numel()).item()

    def adapt(self, mean_eff_fraction: float, val_accuracy: float) -> Dict[str, Any]:
        """# Update tau, One-Way (Only Increases — Weakens Bias)."""
        self._eff_history_.append(mean_eff_fraction)
        record: Dict[str, Any] = {
            "tau": self._tau_, "val_accuracy": val_accuracy, "mean_eff_fraction": mean_eff_fraction
        }

        next_epoch = len(self._eff_history_) + 1
        forcing_ceiling = (
            self._force_ceiling_epoch_ is not None and next_epoch >= self._force_ceiling_epoch_
        )
        record["forcing_ceiling"] = forcing_ceiling

        if forcing_ceiling and self._tau_ < self._tau_ceiling_:
            self._tau_ = min(self._tau_ceiling_, self._tau_ * self._tau_step_)
        elif len(self._eff_history_) >= self._eff_smooth_window_:
            recent = self._eff_history_[-self._eff_smooth_window_:]
            smoothed = sum(recent) / len(recent)
            record["smoothed_eff_fraction"] = smoothed
            if smoothed < self._eff_threshold_ and self._tau_ < self._tau_ceiling_:
                self._tau_ = min(self._tau_ceiling_, self._tau_ * self._tau_step_)

        record["next_tau"] = self._tau_
        return record


class NoisyLabelDataset(torch.utils.data.Dataset):
    """# Wraps a Dataset, Corrupting a Fixed Fraction of Training Labels.

    Motivated directly by the prospectus's own cited finding (Wu et al.
    2020): curriculum ordering shows little benefit over shuffling in
    standard supervised settings, with measurable gains appearing mainly
    under limited training budgets or label noise. Every mechanism this
    session was tested in the regime that finding says curriculum
    learning struggles to help (full budget, clean data) — this tests the
    regime it's actually expected to help in.

    noise_seed is deliberately a SEPARATE, fixed parameter from the
    training seed (same lesson learned from the stride family's
    sort-seed confound): every training seed must compare against the
    IDENTICAL corruption pattern, or seed-to-seed variance would
    conflate genuine training stochasticity with which specific samples
    happened to get corrupted.

    Does one full pass over the base dataset at construction to collect
    true labels (a one-time cost, not per-epoch) — correctness-safe
    regardless of the underlying Dataset implementation's internals.
    """

    def __init__(self, base_dataset: Any, num_classes: int, noise_rate: float, noise_seed: int):
        self._base_ = base_dataset
        n = len(base_dataset)
        rng = np.random.RandomState(noise_seed)
        noisy_mask = rng.rand(n) < noise_rate

        labels = np.array([base_dataset[i][1] for i in range(n)])
        for i in np.where(noisy_mask)[0]:
            true_label = labels[i]
            wrong = rng.randint(0, num_classes)
            while wrong == true_label:
                wrong = rng.randint(0, num_classes)
            labels[i] = wrong

        self._labels_ = labels
        self.actual_noise_rate = float(noisy_mask.mean())

    def __len__(self) -> int:
        return len(self._base_)

    def __getitem__(self, idx: int):
        image, _ = self._base_[idx]
        return image, int(self._labels_[idx])


def run(
    network_id:            str,
    dataset_id:             str,
    batch_size:             int =              128,
    epochs:                 int =              100,
    seed:                   int =              1,
    initial_tau:            float =            1.0,
    tau_step:               float =            1.5,
    tau_ceiling:            float =            50.0,
    eff_threshold:          float =            0.5,
    eff_smooth_window:      int =              3,
    force_ceiling_epoch:    Optional[int] =    None,
    unweighted:             bool =             False,
    label_noise_rate:       float =            0.0,
    label_noise_seed:       int =              1,
    device:                 str =              "auto",
    output_path:            str =              "results/reweight_experiment",
) -> Dict[str, Any]:
    """# Run Isolated Loss-Reweighting Curriculum Experiment."""
    logger = get_logger("reweight-curriculum-experiment")

    set_seed(seed = seed)
    resolved_device: t_device = determine_device(device = device)
    print(f"Device: {resolved_device}")

    dataset: Dataset = DATASET_REGISTRY.load_dataset(
        dataset_id = dataset_id,
        seed =        seed,
        epochs =      epochs,
        batch_size =  batch_size,
        metric =      None
    )

    network: Network = NETWORK_REGISTRY.load_network(
        network_id =  network_id,
        input_shape = dataset.input_shape,
        num_classes = dataset.num_classes
    ).to(resolved_device)

    train_data = dataset.train_data
    if label_noise_rate > 0.0:
        train_data = NoisyLabelDataset(
            base_dataset = train_data, num_classes = dataset.num_classes,
            noise_rate = label_noise_rate, noise_seed = label_noise_seed
        )
        print(f"Label noise injected: requested {label_noise_rate:.2%}, "
                    f"actual {train_data.actual_noise_rate:.2%} (noise_seed={label_noise_seed}, "
                    f"fixed independent of training seed={seed}). Test set unchanged.")

    # Batches are ALWAYS plain shuffled — this mechanism never touches
    # composition, only how much each sample's loss counts.
    train_loader = DataLoader(
        dataset =      train_data,
        batch_size =   batch_size,
        shuffle =      True,
        num_workers =  dataset._max_workers_,
        pin_memory =   True
    )
    test_loader = dataset.test_loader

    controller: Optional[ReweightController] = None
    if not unweighted:
        controller = ReweightController(
            initial_tau =        initial_tau,
            tau_step =           tau_step,
            tau_ceiling =        tau_ceiling,
            eff_threshold =      eff_threshold,
            eff_smooth_window =  eff_smooth_window,
            force_ceiling_epoch = force_ceiling_epoch
        )
        print(f"Loss-reweighting controller: initial_tau={initial_tau}, tau_step={tau_step}x, "
                    f"tau_ceiling={tau_ceiling}, eff_threshold={eff_threshold}.")
    else:
        print("Running unweighted reference (w=1 always — identical to plain shuffled training, "
                    "same code path as the weighted run otherwise).")

    optimizer = SGD(
        params =        network.parameters(),
        lr =            0.1,
        weight_decay =  5e-4,
        momentum =      0.9
    )
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    epoch_records: List[Dict[str, Any]] = []
    control_log:    List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):

        network.train()
        train_total_loss, train_correct, train_total = 0.0, 0, 0

        batch_loss_vars:       List[float] = []
        batch_grad_norm_means: List[float] = []
        batch_grad_norm_vars:  List[float] = []
        batch_eff_fractions:   List[float] = []

        tau_desc = f"{controller._tau_:.3f}" if controller else "unweighted"
        for samples, targets in tqdm(train_loader, desc = f"Epoch {epoch}/{epochs} [tau={tau_desc}]", leave = False):
            samples, targets = samples.to(resolved_device), targets.to(resolved_device)

            predictions:     Tensor = network(samples)
            per_sample_loss: Tensor = cross_entropy(input = predictions, target = targets, reduction = "none")

            if controller is not None:
                w = controller.weights(per_sample_loss)
                loss = controller.weighted_loss(per_sample_loss, w)
                batch_eff_fractions.append(ReweightController.effective_fraction(w))
            else:
                loss = per_sample_loss.mean()
                batch_eff_fractions.append(1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Diagnostic instrumentation (unweighted, measuring the true
            # underlying batch's composition diversity — always
            # shuffled-equivalent by construction here, kept for
            # continuity/comparison with the other two mechanism families).
            with no_grad():
                probs = predictions.softmax(dim = 1)
                one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
                sample_grad_norms = (probs - one_hot).norm(dim = 1)

            batch_loss_vars.append(per_sample_loss.var(unbiased = False).item())
            batch_grad_norm_means.append(sample_grad_norms.mean().item())
            batch_grad_norm_vars.append(sample_grad_norms.var(unbiased = False).item())

            train_total_loss += loss.item()
            _, predicted = predictions.max(dim = 1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        network.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with no_grad():
            for samples, targets in test_loader:
                samples, targets = samples.to(resolved_device), targets.to(resolved_device)
                predictions = network(samples)
                loss = cross_entropy(input = predictions, target = targets)
                val_total_loss += loss.item()
                _, predicted = predictions.max(dim = 1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)

        val_accuracy = round(val_correct / val_total, 4)
        mean_eff_fraction = sum(batch_eff_fractions) / len(batch_eff_fractions)

        if controller is not None:
            control_record = controller.adapt(mean_eff_fraction, val_accuracy)
            control_log.append({"epoch": epoch, **control_record})
            eff_note = ""
            if "smoothed_eff_fraction" in control_record:
                eff_note = f"  eff_fraction(smoothed)={control_record['smoothed_eff_fraction']:.3f}"
            print(f"Epoch {epoch:3}: tau={control_record['tau']:.3f} -> {control_record['next_tau']:.3f}{eff_note}")
        else:
            control_record = None

        record: Dict[str, Any] = {
            "epoch":                    epoch,
            "tau":                      control_record["tau"] if control_record else None,
            "mean_eff_fraction":        round(mean_eff_fraction, 4),
            "train_loss":               round(train_total_loss / len(train_loader), 4),
            "train_accuracy":           round(train_correct / train_total, 4),
            "val_loss":                 round(val_total_loss / len(test_loader), 4),
            "val_accuracy":             val_accuracy,
            "mean_batch_loss_var":      round(sum(batch_loss_vars) / len(batch_loss_vars), 6),
            "mean_batch_grad_norm":     round(sum(batch_grad_norm_means) / len(batch_grad_norm_means), 6),
            "mean_batch_grad_norm_var": round(sum(batch_grad_norm_vars) / len(batch_grad_norm_vars), 6),
        }
        epoch_records.append(record)

    result: Dict[str, Any] = {
        "config": {
            "network_id":         network_id,
            "dataset_id":         dataset_id,
            "batch_size":         batch_size,
            "epochs":             epochs,
            "seed":               seed,
            "initial_tau":        initial_tau,
            "tau_step":           tau_step,
            "tau_ceiling":        tau_ceiling,
            "eff_threshold":      eff_threshold,
            "eff_smooth_window":  eff_smooth_window,
            "force_ceiling_epoch": force_ceiling_epoch,
            "unweighted":         unweighted,
            "label_noise_rate":   label_noise_rate,
            "label_noise_seed":   label_noise_seed,
        },
        "epochs":      epoch_records,
        "control_log": control_log,
    }

    _save(result = result, output_path = output_path, seed = seed, unweighted = unweighted,
          initial_tau = initial_tau, eff_threshold = eff_threshold,
          force_ceiling_epoch = force_ceiling_epoch, label_noise_rate = label_noise_rate)

    return result


def _save(
    result: Dict[str, Any], output_path: str, seed: int, unweighted: bool = False,
    initial_tau: float = 1.0, eff_threshold: float = 0.5,
    force_ceiling_epoch: Optional[int] = None, label_noise_rate: float = 0.0
) -> None:
    """# Write Results to Disk. Distinctive filename tag avoids the
    seed-overwrite footgun documented throughout this session."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    noise_str = f"-noise{label_noise_rate}" if label_noise_rate > 0.0 else ""

    if unweighted:
        tag = f"seed-{seed}_unweighted{noise_str}"
    else:
        ceiling_str = f"-ceil{force_ceiling_epoch}" if force_ceiling_epoch else ""
        tag = f"seed-{seed}_reweight-init{initial_tau}-eff{eff_threshold}{ceiling_str}{noise_str}"

    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent = 2)

    print(f"Saved: {out_dir / f'{tag}.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Loss-reweighting curriculum experiment.")
    parser.add_argument("--network-id",   type = str,   default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-100")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 100)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument(
        "--initial-tau", type = float, default = 1.0,
        help = "Starting temperature for the loss weights. z-scored loss means tau around 1 gives "
               "meaningful bias without being near-degenerate."
    )
    parser.add_argument("--tau-step",     type = float, default = 1.5)
    parser.add_argument("--tau-ceiling",  type = float, default = 50.0)
    parser.add_argument(
        "--eff-threshold", type = float, default = 0.5,
        help = "Smoothed Kish effective-sample-size fraction below which tau is increased (bias "
               "weakened). Self-referential — no probe, no external reference needed, unlike the "
               "grad-var-triggered controllers used for the composition-based mechanisms."
    )
    parser.add_argument("--eff-smooth-window", type = int, default = 3)
    parser.add_argument(
        "--force-ceiling-epoch", type = int, default = None,
        help = "Safety net: from this epoch onward, force tau toward tau_ceiling every epoch "
               "regardless of the effective-fraction signal."
    )
    parser.add_argument(
        "--unweighted", action = "store_true",
        help = "Reference run: w=1 always, identical in effect to plain shuffled training, same "
               "code path otherwise (for a clean same-code comparison)."
    )
    parser.add_argument(
        "--label-noise-rate", type = float, default = 0.0,
        help = "Fraction of TRAINING labels randomly flipped to a wrong class (test set always "
               "stays clean). Default 0.0 = no noise. Motivated by the prospectus's own cited "
               "finding (Wu et al. 2020) that curriculum learning's benefits over shuffling appear "
               "mainly under limited training budgets or label noise, not the clean full-budget "
               "regime every other mechanism this session was tested in."
    )
    parser.add_argument(
        "--label-noise-seed", type = int, default = 1,
        help = "Fixed, separate from --seed: which specific samples get corrupted must stay "
               "identical across different training seeds, or seed-to-seed variance would "
               "conflate genuine training stochasticity with which samples happened to be "
               "corrupted (same lesson as the stride family's sort-seed confound)."
    )
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/reweight_experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        network_id =          args.network_id,
        dataset_id =           args.dataset_id,
        batch_size =           args.batch_size,
        epochs =               args.epochs,
        seed =                 args.seed,
        initial_tau =          args.initial_tau,
        tau_step =             args.tau_step,
        tau_ceiling =          args.tau_ceiling,
        eff_threshold =        args.eff_threshold,
        eff_smooth_window =    args.eff_smooth_window,
        force_ceiling_epoch =  args.force_ceiling_epoch,
        unweighted =           args.unweighted,
        label_noise_rate =     args.label_noise_rate,
        label_noise_seed =     args.label_noise_seed,
        device =               args.device,
        output_path =          args.output_path,
    )