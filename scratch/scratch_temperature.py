"""# scratch_temperature.py

Isolated experiment script for a continuous, temperature-parameterized
curriculum sampler, distinct in kind from the stride/homogeneity-ratchet
family built earlier this session (scratch_stride_diagnostic.py). That
family is a hard partitioning scheme: samples are sorted, chunked into
supergroups, and interleaved into batches via a discrete stride parameter
k. Every hard cliff and hard transition-cost found in that family (the
epoch-1 collapse under h_target=1.0, the epoch-71 instant-jump penalty,
the abrupt forced-floor cost) came from a discrete jump in batch
composition: k moving from one integer to another, h_target moving from
one fixed value to another.

This script implements the dissertation prospectus's actual proposed
mechanism instead (Section 4.3): examples are sampled with probability
proportional to exp(s_i / tau), where s_i is a difficulty-derived
usefulness score and tau is a temperature. As tau -> infinity, sampling
converges to uniform random shuffling; as tau -> 0, it converges to a
deterministic difficulty ordering. There is no discrete partition and no
hard jump for tau to suffer from: every sample has nonzero probability of
being drawn in every batch regardless of tau, so the entire
transition-cost problem diagnosed at length in the stride family may be
an artifact of that family's hard-partition design, not a fundamental
property of curriculum learning.

Score
-----
s_i uses the same live, per-sample analytic gradient norm validated in
scratch_stride_diagnostic.py's local-pool mode (softmax(logits) -
onehot(target), exact, forward-pass only, no backward needed), refreshed
every `refresh_epochs` so it never goes as stale as a static offline
score. Raw gradient norm is a *difficulty* signal (higher = harder). To
match this session's established "ascending" (easy-first) convention,
s_i = -zscore(difficulty), so easier examples get higher s_i and are
prioritized at low tau.

Adaptivity
----------
Same validated control philosophy as the grad-var-triggered stride
controller: a one-way ratchet driven by a live measured signal (this
epoch's batch gradient-norm variance vs. a plain-shuffled probe at the
same model state), not a fixed schedule or lagging validation accuracy.
The direction is inverted relative to the stride controller: tau only
ever *increases* (bias only ever weakens, never strengthens), multiplied
by tau_step each trigger, capped at tau_ceiling (an "effectively
uniform" temperature). A force-ceiling-epoch safety net guarantees tau
reaches tau_ceiling before training ends, for the same reason the stride
controller's force-floor guarantee existed: the ratio signal was
confirmed to become unreliable near convergence in that mechanism, and
there is no reason to assume this one is immune.

Does not modify gradus.curricula.Curriculum or any other framework file.
Loads network and dataset objects through the existing Gradus registries,
bypasses the framework's Curriculum entirely.
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
from torch.utils.data       import BatchSampler, DataLoader
from tqdm                   import tqdm

from gradus.datasets         import Dataset
from gradus.networks         import Network
from gradus.registration     import DATASET_REGISTRY, NETWORK_REGISTRY
from gradus.utilities        import determine_device, get_logger, set_seed


def compute_live_scores(network: Network, loader: DataLoader, device: t_device, n: int) -> np.ndarray:
    """# Forward-Pass-Only Live Difficulty Scores for Every Sample.

    Analytic per-sample output-layer gradient norm (softmax(logits) -
    onehot(target), exact, no backward pass needed), same signal validated
    throughout scratch_stride_diagnostic.py. Runs the network in eval()
    mode (restoring whatever mode it was in beforehand), over a
    shuffle=False loader so returned scores align index-for-index with
    the dataset's natural ordering. Shared between the live-scoring probe
    and the curriculum's own score refresh.
    """
    was_training = network.training
    network.eval()

    scores = np.empty(n, dtype = np.float32)
    ptr = 0
    with no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = targets.to(device)
            predictions = network(samples)
            probs = predictions.softmax(dim = 1)
            one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
            sample_grad_norms = (probs - one_hot).norm(dim = 1)
            k = sample_grad_norms.shape[0]
            scores[ptr:ptr + k] = sample_grad_norms.cpu().numpy()
            ptr += k

    if was_training:
        network.train()

    return scores


class TemperatureCurriculum(BatchSampler):
    """# Continuous Temperature-Parameterized Curriculum Sampler.

    Each refresh, computes live per-sample difficulty scores, z-scores and
    negates them (so higher = easier = prioritized, matching this
    session's "ascending" convention), and turns them into a sampling
    distribution via softmax(s_i / tau). Batches are drawn independently,
    each via weighted sampling without replacement within that batch, so
    across a full epoch some samples may appear in multiple batches and
    others in none; this is the standard temperature-sampler design (a
    distribution over the dataset, not a deterministic partition of it),
    distinct in kind from the stride family's exact one-pass-per-epoch
    coverage.
    """

    def __init__(
        self,
        batch_size:           int,
        seed:                 int,
        network:              Network,
        dataset:               Any,
        device:               t_device,
        initial_tau:           float = 2.0,
        tau_step:              float = 1.5,
        tau_ceiling:           float = 50.0,
        grad_var_threshold:    float = 0.85,
        grad_var_smooth_window: int = 3,
        grad_var_force_ceiling_epoch: Optional[int] = None,
        refresh_epochs:        int = 1,
    ):
        self._batch_size_ =    batch_size
        self._seed_ =          seed
        self._network_ =       network
        self._dataset_ =       dataset
        self._device_ =        device
        self._n_ =              len(dataset)

        self._tau_ =            initial_tau
        self._tau_step_ =       tau_step
        self._tau_ceiling_ =    tau_ceiling
        self._grad_var_threshold_ =     grad_var_threshold
        self._grad_var_smooth_window_ = grad_var_smooth_window
        self._grad_var_force_ceiling_epoch_ = grad_var_force_ceiling_epoch
        self._refresh_epochs_ = refresh_epochs

        self._rng_ =            np.random.RandomState(seed)
        self._score_loader_ =   DataLoader(
            dataset =     dataset,
            batch_size =  256,
            shuffle =     False,
            num_workers = 0,
            pin_memory =  True
        )

        self._val_acc_history_:       List[float] = []
        self._grad_var_ratio_history_: List[float] = []
        self._epoch_counter_:          int = 0

        self._probs_ = self._compute_probs()
        self._batches_ = self._draw_batches()

    def _compute_probs(self) -> np.ndarray:
        scores = compute_live_scores(self._network_, self._score_loader_, self._device_, self._n_)
        z = (scores - scores.mean()) / (scores.std() + 1e-8)
        s = -z  # higher s = easier = prioritized, matching this session's ascending convention
        # Subtract max before exponentiating for numerical stability; does
        # not change the resulting distribution (softmax is shift-invariant).
        logits = s / self._tau_
        logits = logits - logits.max()
        weights = np.exp(logits)
        return weights / weights.sum()

    def _draw_batches(self) -> List[List[int]]:
        batches_per_epoch = self._n_ // self._batch_size_
        batches: List[List[int]] = []
        for _ in range(batches_per_epoch):
            idx = self._rng_.choice(self._n_, size = self._batch_size_, replace = False, p = self._probs_)
            batches.append(idx.tolist())
        return batches

    def adapt(
        self, val_accuracy: float,
        measured_grad_var: Optional[float] = None,
        probe_grad_var: Optional[float] = None,
    ) -> Dict[str, Any]:
        """# Update tau from a Live Measured Signal, One-Way (Only Increases).

        Mirrors the stride family's grad-var-triggered controller exactly
        in philosophy, inverted in direction: instead of ratcheting a
        homogeneity target down toward more diversity, this ratchets tau
        up toward tau_ceiling (weaker bias), since higher tau IS more
        diversity here. Multiplicative step, since tau's effect on the
        distribution is exponential, not linear: a fixed additive step
        would mean something very different at tau=1 than at tau=20.
        """
        self._val_acc_history_.append(val_accuracy)
        record: Dict[str, Any] = {"tau": self._tau_, "val_accuracy": val_accuracy}

        if measured_grad_var is not None and probe_grad_var:
            ratio = measured_grad_var / probe_grad_var
            self._grad_var_ratio_history_.append(ratio)
            record["grad_var_ratio"] = ratio

        next_epoch = len(self._val_acc_history_) + 1
        forcing_ceiling = (
            self._grad_var_force_ceiling_epoch_ is not None
            and next_epoch >= self._grad_var_force_ceiling_epoch_
        )
        record["forcing_ceiling"] = forcing_ceiling

        if forcing_ceiling and self._tau_ < self._tau_ceiling_:
            self._tau_ = min(self._tau_ceiling_, self._tau_ * self._tau_step_)
        elif len(self._grad_var_ratio_history_) >= self._grad_var_smooth_window_:
            recent = self._grad_var_ratio_history_[-self._grad_var_smooth_window_:]
            smoothed_ratio = sum(recent) / len(recent)
            record["smoothed_grad_var_ratio"] = smoothed_ratio
            if smoothed_ratio < self._grad_var_threshold_ and self._tau_ < self._tau_ceiling_:
                self._tau_ = min(self._tau_ceiling_, self._tau_ * self._tau_step_)

        record["next_tau"] = self._tau_
        return record

    # DUNDERS ======================================================================================

    def __iter__(self):
        # DataLoader calls __iter__ once per epoch, giving a natural
        # once-per-epoch hook for the refresh, same pattern as local-pool
        # mode in the stride script.
        if self._epoch_counter_ % self._refresh_epochs_ == 0:
            self._probs_ = self._compute_probs()
            self._batches_ = self._draw_batches()
        else:
            # tau/scores unchanged this epoch, but batches should still be
            # redrawn fresh (this is a distribution, not a fixed partition).
            self._batches_ = self._draw_batches()
        self._epoch_counter_ += 1
        yield from self._batches_

    def __len__(self) -> int:
        return len(self._batches_)


def run(
    network_id:            str,
    dataset_id:             str,
    batch_size:             int =              128,
    epochs:                 int =              100,
    seed:                   int =              1,
    initial_tau:            float =            2.0,
    tau_step:               float =            1.5,
    tau_ceiling:            float =            50.0,
    grad_var_threshold:     float =            0.85,
    grad_var_smooth_window: int =              3,
    grad_var_probe_batches: int =              15,
    grad_var_force_ceiling_epoch: Optional[int] = None,
    refresh_epochs:         int =              1,
    shuffled:               bool =             False,
    device:                 str =              "auto",
    output_path:            str =              "results/temperature_experiment",
) -> Dict[str, Any]:
    """# Run Isolated Temperature-Parameterized Curriculum Experiment."""
    logger = get_logger("temperature-curriculum-experiment")

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

    sampler: Optional[TemperatureCurriculum] = None
    if shuffled:
        train_loader = DataLoader(
            dataset =      dataset.train_data,
            batch_size =   batch_size,
            shuffle =      True,
            num_workers =  dataset._max_workers_,
            pin_memory =   True
        )
        print("Running plain-shuffled reference (no curriculum sampler).")
    else:
        sampler = TemperatureCurriculum(
            batch_size =           batch_size,
            seed =                 seed,
            network =              network,
            dataset =              dataset.train_data,
            device =               resolved_device,
            initial_tau =          initial_tau,
            tau_step =             tau_step,
            tau_ceiling =          tau_ceiling,
            grad_var_threshold =   grad_var_threshold,
            grad_var_smooth_window = grad_var_smooth_window,
            grad_var_force_ceiling_epoch = grad_var_force_ceiling_epoch,
            refresh_epochs =       refresh_epochs
        )
        print(f"Temperature sampler: initial_tau={initial_tau}, tau_step={tau_step}x, "
                    f"tau_ceiling={tau_ceiling}, threshold={grad_var_threshold}, "
                    f"refresh every {refresh_epochs} epoch(s).")
        train_loader = DataLoader(
            dataset =       dataset.train_data,
            batch_sampler = sampler,
            num_workers =   dataset._max_workers_,
            pin_memory =    True
        )

    test_loader = dataset.test_loader

    probe_loader: Optional[DataLoader] = None
    if not shuffled:
        probe_loader = DataLoader(
            dataset =      dataset.train_data,
            batch_size =   batch_size,
            shuffle =      True,
            num_workers =  dataset._max_workers_,
            pin_memory =   True
        )

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

        for samples, targets in tqdm(
            train_loader,
            desc = f"Epoch {epoch}/{epochs} [tau={sampler._tau_:.3f}]" if sampler else f"Epoch {epoch}/{epochs} [shuffled]",
            leave = False
        ):
            samples, targets = samples.to(resolved_device), targets.to(resolved_device)

            predictions:     Tensor = network(samples)
            per_sample_loss: Tensor = cross_entropy(input = predictions, target = targets, reduction = "none")
            loss:             Tensor = per_sample_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        measured_grad_var = sum(batch_grad_norm_vars) / len(batch_grad_norm_vars)
        probe_grad_var: Optional[float] = None
        if probe_loader is not None:
            probe_vars: List[float] = []
            with no_grad():
                probe_iter = iter(probe_loader)
                for _ in range(grad_var_probe_batches):
                    p_samples, p_targets = next(probe_iter)
                    p_samples, p_targets = p_samples.to(resolved_device), p_targets.to(resolved_device)
                    p_predictions = network(p_samples)
                    p_probs = p_predictions.softmax(dim = 1)
                    p_one_hot = torch.zeros_like(p_probs).scatter_(1, p_targets.unsqueeze(1), 1.0)
                    p_sample_grad_norms = (p_probs - p_one_hot).norm(dim = 1)
                    probe_vars.append(p_sample_grad_norms.var(unbiased = False).item())
            probe_grad_var = sum(probe_vars) / len(probe_vars)

        if sampler is not None:
            control_record = sampler.adapt(
                val_accuracy, measured_grad_var = measured_grad_var, probe_grad_var = probe_grad_var
            )
            control_log.append({"epoch": epoch, **control_record})
            grad_var_note = ""
            if "smoothed_grad_var_ratio" in control_record:
                grad_var_note = f"  grad_var_ratio(smoothed)={control_record['smoothed_grad_var_ratio']:.3f}"
            print(
                f"Epoch {epoch:3}: tau={control_record['tau']:.3f} -> {control_record['next_tau']:.3f}"
                f"{grad_var_note}"
            )
        else:
            control_record = None

        record: Dict[str, Any] = {
            "epoch":                    epoch,
            "tau":                      control_record["tau"] if control_record else None,
            "train_loss":               round(train_total_loss / (train_total / batch_size), 4),
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
            "network_id":            network_id,
            "dataset_id":            dataset_id,
            "batch_size":            batch_size,
            "epochs":                epochs,
            "seed":                  seed,
            "initial_tau":           initial_tau,
            "tau_step":              tau_step,
            "tau_ceiling":           tau_ceiling,
            "grad_var_threshold":    grad_var_threshold,
            "grad_var_smooth_window": grad_var_smooth_window,
            "grad_var_probe_batches": grad_var_probe_batches,
            "grad_var_force_ceiling_epoch": grad_var_force_ceiling_epoch,
            "refresh_epochs":        refresh_epochs,
            "shuffled":              shuffled,
        },
        "epochs":      epoch_records,
        "control_log": control_log,
    }

    _save(result = result, output_path = output_path, seed = seed, shuffled = shuffled,
          initial_tau = initial_tau, grad_var_threshold = grad_var_threshold,
          grad_var_force_ceiling_epoch = grad_var_force_ceiling_epoch)

    return result


def _save(
    result: Dict[str, Any], output_path: str, seed: int, shuffled: bool = False,
    initial_tau: float = 2.0, grad_var_threshold: float = 0.85,
    grad_var_force_ceiling_epoch: Optional[int] = None
) -> None:
    """# Write Results to Disk. Distinctive filename tag avoids the same
    seed-overwrite footgun documented in scratch_stride_diagnostic.py."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    if shuffled:
        tag = f"seed-{seed}_shuffled"
    else:
        ceiling_str = f"-ceil{grad_var_force_ceiling_epoch}" if grad_var_force_ceiling_epoch else ""
        tag = f"seed-{seed}_temp-init{initial_tau}-thresh{grad_var_threshold}{ceiling_str}"

    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent = 2)

    print(f"Saved: {out_dir / f'{tag}.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Temperature-parameterized curriculum experiment.")
    parser.add_argument("--network-id",   type = str,   default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-100")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 100)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument(
        "--initial-tau", type = float, default = 2.0,
        help = "Starting temperature. Lower = stronger bias toward easy examples. z-scored scores "
               "mean tau in the 1-5 range gives meaningful bias without being near-degenerate."
    )
    parser.add_argument(
        "--tau-step", type = float, default = 1.5,
        help = "Multiplicative step applied to tau each trigger (tau *= tau_step), capped at "
               "tau_ceiling. Multiplicative, not additive, since tau's effect on the sampling "
               "distribution is exponential."
    )
    parser.add_argument(
        "--tau-ceiling", type = float, default = 50.0,
        help = "Temperature treated as 'effectively uniform' (shuffling-equivalent). tau never "
               "exceeds this."
    )
    parser.add_argument(
        "--grad-var-threshold", type = float, default = 0.85,
        help = "Smoothed curriculum/shuffled gradient-norm-variance ratio below this fires a tau "
               "increase. Not yet calibrated for this mechanism specifically — verify via "
               "simulation before trusting, same discipline used for the stride family."
    )
    parser.add_argument("--grad-var-smooth-window", type = int, default = 3)
    parser.add_argument("--grad-var-probe-batches", type = int, default = 15)
    parser.add_argument(
        "--grad-var-force-ceiling-epoch", type = int, default = None,
        help = "Safety net: from this epoch onward, force tau toward tau_ceiling every epoch "
               "regardless of the ratio, guaranteeing effectively-uniform sampling is reached "
               "before training ends. Same rationale as the stride family's force-floor guarantee."
    )
    parser.add_argument(
        "--refresh-epochs", type = int, default = 1,
        help = "How often (in epochs) to recompute live scores. Default 1 = every epoch."
    )
    parser.add_argument("--shuffled", action = "store_true")
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/temperature_experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        network_id =            args.network_id,
        dataset_id =             args.dataset_id,
        batch_size =             args.batch_size,
        epochs =                 args.epochs,
        seed =                   args.seed,
        initial_tau =            args.initial_tau,
        tau_step =               args.tau_step,
        tau_ceiling =            args.tau_ceiling,
        grad_var_threshold =     args.grad_var_threshold,
        grad_var_smooth_window = args.grad_var_smooth_window,
        grad_var_probe_batches = args.grad_var_probe_batches,
        grad_var_force_ceiling_epoch = args.grad_var_force_ceiling_epoch,
        refresh_epochs =         args.refresh_epochs,
        shuffled =               args.shuffled,
        device =                 args.device,
        output_path =            args.output_path,
    )