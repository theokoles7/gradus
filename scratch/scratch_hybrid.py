"""# scratch_hybrid.py

Combines the two validated mechanisms from this session rather than
tuning either further:

  * scratch_stride_diagnostic.py's --local-pool: freshly reshuffles the
    dataset into pools every refresh and ranks difficulty live within
    each pool, giving every sample exactly one batch assignment per
    epoch (full coverage) and, empirically, immunity to the early-epoch
    gradient-diversity cliff that plagued the static global sort. Its
    remaining weakness: assignment from a sorted pool into its k batches
    is a hard, deterministic rule (supergroup[b::k]).

  * scratch_temperature.py's continuous, tau-parameterized sampler:
    p_i proportional to exp(s_i/tau), no discrete jumps for tau to
    suffer transition costs from. Its weakness: batches are drawn i.i.d.
    from the whole dataset each time, so there is no guarantee every
    sample gets used; some may never be drawn in a given epoch while
    others repeat, and (confirmed this session) a persistent, slow
    "bleed" cost accumulates while tau sits at any fixed, moderate bias.

Hybrid design
-------------
Each refresh: shuffle the full dataset, chunk into pools of size
batch_size*k (as in local-pool), compute live per-sample scores. Within
each pool, assign samples to its k batches via SEQUENTIAL WEIGHTED
SAMPLING WITHOUT REPLACEMENT instead of the deterministic b::k
interleave:

  1. Compute k target scores, the pool's own score quantiles at
     (b+0.5)/k for b = 0..k-1 (batch 0's target is the easiest end,
     batch k-1's the hardest, matching this session's ascending
     convention).
  2. For b = 0..k-1, weight every still-unassigned pool member by
     exp(-|score_i - target_b| / tau), draw batch_size of them without
     replacement using those weights, remove them from the pool, move
     to the next batch.

This guarantees every pool member is assigned to exactly one batch (full
coverage, like local-pool), while the assignment itself is soft and
tau-controlled (like the temperature sampler), not a hard rule. As
tau -> 0, weight concentrates entirely on the single closest-matching
remaining member each draw, converging to a strict sorted assignment
(maximally homogeneous). As tau -> infinity, weights become uniform
regardless of score, so each batch draws effectively at random from
whatever remains — since the pool itself was already freshly shuffled,
this is equivalent in distribution to slicing the random pool directly
into batch_size chunks, i.e. shuffling-equivalent. Verified in isolation
before implementation: full coverage holds at every tau tested, and the
batch-mean-difficulty trend is correctly (near-perfectly) ordered at low
tau, with only the necessarily-leftover final batch as an exception.

tau adaptation reuses the same validated one-way, multiplicative,
force-ceiling-guaranteed controller from scratch_temperature.py, driven
by the same live measured gradient-variance-ratio signal.

Does not modify gradus.curricula.Curriculum or any other framework file.
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

    Same analytic per-sample output-layer gradient norm validated
    throughout this session (softmax(logits) - onehot(target), exact, no
    backward pass needed). eval() mode, restores whatever mode the
    network was in beforehand.
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


class HybridPoolSampler(BatchSampler):
    """# Full-Coverage, Soft-Assignment Pool Sampler.

    See module docstring for the full design. batch_size*k defines the
    pool size, same as --local-pool; k is fixed per epoch (not adapted
    directly), only tau adapts. Higher k = smaller pools = more batches
    per pool-refresh-cycle; k here plays a much smaller role than in the
    stride family, since tau (not k) is now the primary structure-control
    variable. Fixed at a moderate default (k=8) unless overridden — small
    enough that quantile targets are meaningfully spread, large enough
    that per-batch weighted draws aren't trivially degenerate.
    """

    def __init__(
        self,
        batch_size:           int,
        k:                    int,
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
        self._k_ =              k
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

        self._cached_scores_: Optional[np.ndarray] = None
        self._batches_ = self._build(refresh_scores = True)

    def _build(self, refresh_scores: bool) -> List[List[int]]:
        if refresh_scores or self._cached_scores_ is None:
            self._cached_scores_ = compute_live_scores(
                self._network_, self._score_loader_, self._device_, self._n_
            )
        scores = self._cached_scores_
        perm = self._rng_.permutation(self._n_)
        pool_sz = self._batch_size_ * self._k_

        batches: List[List[int]] = []
        for start in range(0, self._n_, pool_sz):
            pool = perm[start:start + pool_sz]

            if len(pool) < pool_sz:
                # Leftover partial pool: not enough members for a full
                # k-batch soft assignment. Fall back to plain contiguous
                # chunking, same tail-handling used throughout this
                # session's stride/local-pool scripts.
                for i in range(0, len(pool), self._batch_size_):
                    batches.append(pool[i:i + self._batch_size_].tolist())
                continue

            pool_scores = scores[pool]
            targets = np.quantile(pool_scores, [(b + 0.5) / self._k_ for b in range(self._k_)])

            remaining_idx = pool.copy()
            remaining_scores = pool_scores.copy()
            for b in range(self._k_):
                d = np.abs(remaining_scores - targets[b])
                logits = -d / self._tau_
                logits = logits - logits.max()
                w = np.exp(logits)
                p = w / w.sum()
                chosen = self._rng_.choice(len(remaining_idx), size = self._batch_size_, replace = False, p = p)
                batches.append(remaining_idx[chosen].tolist())
                mask = np.ones(len(remaining_idx), dtype = bool)
                mask[chosen] = False
                remaining_idx = remaining_idx[mask]
                remaining_scores = remaining_scores[mask]

        return batches

    def adapt(
        self, val_accuracy: float,
        measured_grad_var: Optional[float] = None,
        probe_grad_var: Optional[float] = None,
    ) -> Dict[str, Any]:
        """# Update tau from a Live Measured Signal, One-Way (Only Increases).

        Identical control philosophy to scratch_temperature.py's
        TemperatureCurriculum.adapt() — see that file for full rationale.
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
        # DataLoader calls __iter__ once per epoch. Scores are only
        # recomputed on the refresh_epochs cadence (an extra forward pass
        # over the full dataset); the pool shuffle and soft assignment are
        # always redrawn fresh every epoch regardless, same as
        # --local-pool and the pure temperature sampler — this is a
        # distribution, not a fixed partition.
        due_for_refresh = self._epoch_counter_ % self._refresh_epochs_ == 0
        self._batches_ = self._build(refresh_scores = due_for_refresh)
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
    k:                      int =              8,
    initial_tau:            float =            2.0,
    tau_step:               float =            1.5,
    tau_ceiling:            float =            50.0,
    grad_var_threshold:     float =            0.85,
    grad_var_smooth_window: int =              3,
    grad_var_probe_batches: int =              30,
    grad_var_force_ceiling_epoch: Optional[int] = None,
    refresh_epochs:         int =              1,
    shuffled:               bool =             False,
    device:                 str =              "auto",
    output_path:            str =              "results/hybrid_experiment",
) -> Dict[str, Any]:
    """# Run Isolated Hybrid Pool+Temperature Curriculum Experiment."""
    logger = get_logger("hybrid-curriculum-experiment")

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

    sampler: Optional[HybridPoolSampler] = None
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
        sampler = HybridPoolSampler(
            batch_size =           batch_size,
            k =                    k,
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
        print(f"Hybrid pool sampler: k={k} (pool size {batch_size*k}), initial_tau={initial_tau}, "
                    f"tau_step={tau_step}x, tau_ceiling={tau_ceiling}, threshold={grad_var_threshold}, "
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
            "k":                     k,
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
          k = k, initial_tau = initial_tau, grad_var_threshold = grad_var_threshold,
          grad_var_force_ceiling_epoch = grad_var_force_ceiling_epoch)

    return result


def _save(
    result: Dict[str, Any], output_path: str, seed: int, shuffled: bool = False,
    k: int = 8, initial_tau: float = 2.0, grad_var_threshold: float = 0.85,
    grad_var_force_ceiling_epoch: Optional[int] = None
) -> None:
    """# Write Results to Disk. Distinctive filename tag avoids the
    seed-overwrite footgun documented throughout this session."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    if shuffled:
        tag = f"seed-{seed}_shuffled"
    else:
        ceiling_str = f"-ceil{grad_var_force_ceiling_epoch}" if grad_var_force_ceiling_epoch else ""
        tag = f"seed-{seed}_hybrid-k{k}-init{initial_tau}-thresh{grad_var_threshold}{ceiling_str}"

    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent = 2)

    print(f"Saved: {out_dir / f'{tag}.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Hybrid pool+temperature curriculum experiment.")
    parser.add_argument("--network-id",   type = str,   default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-100")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 100)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument(
        "--k", type = int, default = 8,
        help = "Pool size multiplier (pool = batch_size*k). Unlike the stride family, k here is "
               "fixed per run, not adapted — tau is the primary structure-control variable. Default "
               "8 gives meaningfully separated quantile targets without huge pools."
    )
    parser.add_argument("--initial-tau",  type = float, default = 2.0)
    parser.add_argument("--tau-step",     type = float, default = 1.5)
    parser.add_argument("--tau-ceiling",  type = float, default = 50.0)
    parser.add_argument("--grad-var-threshold",      type = float, default = 0.85)
    parser.add_argument("--grad-var-smooth-window",  type = int,   default = 3)
    parser.add_argument("--grad-var-probe-batches",  type = int,   default = 30)
    parser.add_argument("--grad-var-force-ceiling-epoch", type = int, default = None)
    parser.add_argument("--refresh-epochs", type = int, default = 1)
    parser.add_argument("--shuffled", action = "store_true")
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/hybrid_experiment")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        network_id =            args.network_id,
        dataset_id =             args.dataset_id,
        batch_size =             args.batch_size,
        epochs =                 args.epochs,
        seed =                   args.seed,
        k =                      args.k,
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