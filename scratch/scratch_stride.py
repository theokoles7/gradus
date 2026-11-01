"""# stride_curriculum_experiment.py

Isolated experiment script for the stride-constructed, proportionally-controlled
homogeneity curriculum.

Does not modify gradus.curricula.Curriculum or any other framework file. Loads
network and dataset objects through the existing Gradus registries, bypasses the
framework's Curriculum entirely, and supplies a standalone batch construction
rule plus a proportional controller.

Algorithm
---------
Given the full dataset sorted by difficulty (holistic order), partition the
sorted order into supergroups of size (batch_size * k), then within each
supergroup assign samples to k batches by interleaving: batch b within a
supergroup takes supergroup[b::k].

k = 1:
    each supergroup equals one batch, taken as a consecutive sorted block.
    Exactly reproduces holistic scope (maximum within-batch difficulty
    homogeneity).
k = batches_per_epoch:
    each supergroup is the entire dataset; each batch draws one sample from
    every k-th position across the full difficulty range. Deterministic
    full-range coverage per batch, the same conceptual endpoint as batch-wise
    scope (minimum homogeneity), reached by even spacing rather than random
    shuffling.

Homogeneity, h(k), is the mean within-batch variance of the (raw) difficulty
score, normalized against the dataset's total score variance, so h=1 means
every batch is maximally homogeneous by difficulty (k=1, holistic) and h=0
means batches draw evenly across the full difficulty range (k=batches_per_epoch).
This is a deterministic property of the sorted score array and the stride
construction, not something that needs to be measured from a training run, so
it is computed once for every valid k before training starts (see
`_build_homogeneity_table`). An earlier version of this script tried to infer
homogeneity from measured batch-level class entropy at runtime; that signal
turned out to be close to insensitive to k for this dataset/metric pair,
since class label and difficulty score are close to independent here, so it
could not actually drive the controller. Direct computation replaces it.

Adaptivity is a monotonic ratchet, not a bandit and not a bidirectional
controller. Each epoch, a moving average of recent validation accuracy
(`smooth_window` epochs) is compared against the moving average from `window`
epochs before that, giving an improvement rate. Rather than comparing that
rate to a fixed absolute constant, it's compared to a fraction
(`deceleration_threshold`) of its own historical peak, tracked over the run
so far. Once the rate falls below that fraction of its peak, h_target is
lowered by `step` (push toward more diversity), floored at `h_floor`. While
the rate stays above that fraction, h_target simply holds where it is; it is
never raised back toward more structure. The k whose precomputed h(k) is
nearest to h_target is used to construct the next epoch's batches. Class
entropy is still logged per epoch for diagnostic purposes, but does not drive
k.

The relative trigger replaces an earlier fixed absolute threshold
(plateau_threshold=0.02 on raw improvement). That threshold was tuned by eye
against CIFAR-10 behavior, where meaningful improvement mostly stops by
roughly epoch 16-20. CIFAR-100 keeps improving substantially for most of a
100-epoch run (still gaining through epoch 80 in observed runs), so a
constant tuned for CIFAR-10's decay shape was never well-calibrated for
CIFAR-100's. Comparing against each run's own peak improvement rate instead
adapts automatically to how fast that particular dataset/model/metric
combination actually learns, without a hand-picked constant that only suits
one dataset's dynamics.

h_floor exists because h_target=0 corresponds to k=batches_per_epoch, which
is a deterministic full-range spread, conceptually the same endpoint as
shuffling; the original reasoning was that a policy converging toward the
thing it's being compared against cannot structurally beat it. That reasoning
did not hold up empirically: across four separate CIFAR-100 tests (varying
the reversal mechanism from hysteresis=3 through hysteresis=1 to the ratchet),
every run with h_floor=0.15 underperformed the otherwise-identical run with
h_floor=0.0, consistently. Reaching full diversity by the end of training
helped, not hurt, contrary to the original hypothesis. h_floor now defaults to
0.0 to match the evidence; it's retained as a parameter in case a nonzero
floor is worth re-testing under this new trigger, but should not be assumed
beneficial without that retest.

The ratchet itself replaces an earlier bidirectional design (symmetric and
then asymmetric hysteresis, allowing h_target to move back toward more
structure when validation accuracy resumed improving). On CIFAR-100, where
genuine improvement continues for most of the run rather than plateauing
early the way CIFAR-10 does, that reversal behavior repeatedly re-committed
the policy to near-full homogeneity, producing results close to the static
curriculum's rather than anything meaningfully adaptive (a
floor=0.15/hysteresis=3 run scored 74.75% vs. the static curriculum's
74.49%, essentially no improvement). A controlled same-seed test on CIFAR-10
also found hysteresis reaching the diversity floor faster and producing a
worse result than no hysteresis at all (0.9460 vs. 0.9502). Both findings
point the same direction: allowing h_target to move back toward structure was
actively harmful, not merely unhelpful. The ratchet removes that capability
by construction rather than continuing to tune around it.

Known gaps (framework internals not available when this script was written):
    * RANK_REGISTRY.sort_indices caching behavior is not visible from here,
      same caveat as reshuffle_experiment.py.
    * Per-sample class labels are read from dataset.train_data.targets if
      present (true for torchvision CIFAR10/CIFAR100), else built by iterating
      the dataset once at startup. Used only for the diagnostic entropy log,
      not for control.
    * The homogeneity table ignores each k's remainder supergroup (the tail
      that doesn't fill a full batch_size*k block) for speed; this is an
      approximation, not exact, worth checking if k trends toward the high
      end of the range where remainders are a larger fraction of the total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing   import Any, Dict, List, Optional

import numpy as np
from torch                  import device as t_device, no_grad, Tensor
from torch.nn.functional    import cross_entropy
from torch.optim            import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data       import BatchSampler, DataLoader, SequentialSampler
from tqdm                   import tqdm

from gradus.artifacts        import DatasetMetrics
from gradus.datasets         import Dataset
from gradus.networks         import Network
from gradus.registration     import DATASET_REGISTRY, NETWORK_REGISTRY, RANK_REGISTRY
from gradus.utilities        import determine_device, get_logger, set_seed


def get_labels(dataset: Dataset) -> List[int]:
    """# Extract Per-Sample Class Labels.

    Uses dataset.train_data.targets if present (torchvision CIFAR10/CIFAR100),
    otherwise falls back to iterating the dataset once.
    """
    targets = getattr(dataset.train_data, "targets", None)
    if targets is not None:
        return list(targets)
    return [int(dataset.train_data[i][1]) for i in range(len(dataset.train_data))]


class StrideHomogeneityCurriculum(BatchSampler):
    """# Stride-Constructed Homogeneity Curriculum with Plateau-Driven Target Selection."""

    def __init__(self,
        dataset_id:         str,
        scores:               DatasetMetrics,
        metric:               str,
        rank:                 str,
        batch_size:           int,
        num_classes:          int,
        labels:               List[int],
        seed:                 int,
        window:               int = 5,
        smooth_window:        int = 3,
        step:                 float = 0.05,
        deceleration_threshold: float = 0.05,
        h_floor:              float = 0.0,
        initial_h_target:     float = 1.0,
    ):
        super().__init__(
            sampler =    SequentialSampler(scores.scores),
            batch_size = batch_size,
            drop_last =  False
        )

        self._dataset_id_:      str =              dataset_id
        self._scores_:          DatasetMetrics =    scores
        self._metric_:          List[str] =         [metric] if isinstance(metric, str) else metric
        self._rank_:            str =                rank
        self._batch_size_:      int =                batch_size
        self._num_classes_:     int =                num_classes
        self._labels_:          List[int] =          labels
        self._seed_:            int =                seed
        self._window_:          int =                window
        self._smooth_window_:   int =                smooth_window
        self._step_:            float =              step
        self._decel_thresh_:    float =              deceleration_threshold
        self._h_floor_:          float =              h_floor
        self._peak_improvement_: Optional[float] =    None

        # Holistic difficulty order, computed once. Only the batch construction
        # (via k) changes epoch to epoch, never the underlying ranking.
        self._sorted_indices_:  List[int] =          RANK_REGISTRY.sort_indices(
                                                          rank_id =    rank,
                                                          dataset_id = dataset_id,
                                                          metric =     self._metric_,
                                                          scores =     scores.scores,
                                                          seed =       seed
                                                      )
        self._n_batches_:       int =                math.ceil(len(self._sorted_indices_) / batch_size)

        # Homogeneity is a deterministic function of k, given the fixed sorted
        # order and batch size, so it's computed once here rather than
        # inferred from any training-time measurement.
        self._k_table_, self._h_table_ = self._build_homogeneity_table()

        self._h_target_:        float =              initial_h_target
        self._k_:               int =                self._nearest_k(initial_h_target)
        self._val_acc_history_: List[float] =        []
        self._batches_:          List[List[int]] =    self._build(self._k_)

    # HELPERS ======================================================================================

    def _build_homogeneity_table(self):
        """# Precompute Homogeneity h(k) for Every Valid Stride k.

        h(k) = 1 - mean_within_batch_variance(k) / total_score_variance.
        The remainder supergroup (any tail shorter than batch_size * k) is
        excluded from the mean for speed; an approximation, not exact.

        ## Returns:
            * (np.ndarray, np.ndarray): sorted k values and their h(k), for
                                       nearest-value lookup.
        """
        scores_df = self._scores_.scores.loc[self._sorted_indices_]
        raw = scores_df[self._metric_[0]].to_numpy(dtype = float)  # sorted order, ascending difficulty
        total_var = raw.var()
        n = len(raw)

        ks: List[int] = []
        hs: List[float] = []
        for k in range(1, self._n_batches_ + 1):
            sg = self._batch_size_ * k
            usable = (n // sg) * sg
            if usable == 0:
                continue
            block = raw[:usable].reshape(-1, self._batch_size_, k)
            batch_var = block.var(axis = 1)  # (num_supergroups, k)
            mean_var = batch_var.mean()
            h = 1.0 - (mean_var / total_var) if total_var > 0 else 0.0
            ks.append(k)
            hs.append(float(h))

        return np.array(ks), np.array(hs)

    def _nearest_k(self, h_target: float) -> int:
        """# Find the k Whose Precomputed h(k) Is Closest to a Target."""
        idx = int(np.argmin(np.abs(self._h_table_ - h_target)))
        return int(self._k_table_[idx])

    def _build(self, k: int) -> List[List[int]]:
        """# Construct Batches at a Given Stride k."""
        indices:      List[int] =       self._sorted_indices_
        n:            int =             len(indices)
        supergroup_sz: int =            self._batch_size_ * k

        batches: List[List[int]] = []
        for start in range(0, n, supergroup_sz):
            supergroup = indices[start:start + supergroup_sz]

            # Last supergroup may be short; fall back to plain chunking for it
            # rather than a degenerate interleave.
            if len(supergroup) < supergroup_sz:
                for i in range(0, len(supergroup), self._batch_size_):
                    batches.append(supergroup[i:i + self._batch_size_])
                continue

            for b in range(k):
                batches.append(supergroup[b::k])

        return batches

    def _batch_class_entropy(self, batch: List[int]) -> float:
        """# Shannon Entropy (nats) of the Class Distribution Within a Batch.

        Diagnostic only; does not drive k. Kept for comparison against h(k)
        and to keep visibility into whether class composition is behaving as
        expected across the run.
        """
        counts: Dict[int, int] = {}
        for idx in batch:
            c = self._labels_[idx]
            counts[c] = counts.get(c, 0) + 1

        total = len(batch)
        ent = 0.0
        for c, n in counts.items():
            p = n / total
            ent -= p * math.log(p)
        return ent

    # METHODS ======================================================================================

    def adapt(self, val_accuracy: float) -> Dict[str, Any]:
        """# Update h_target from Validation-Accuracy Plateau Detection, Adjust k.

        Called once per epoch, after that epoch's validation accuracy is
        known, so k for the *next* epoch reflects this epoch's result. Before
        enough history exists for both moving averages (window + smooth_window
        epochs), k is left at its current value.

        ## Args:
            * val_accuracy (float): This epoch's validation accuracy.

        ## Returns:
            * Dict[str, Any]: logging record.
        """
        self._val_acc_history_.append(val_accuracy)

        mean_entropy = sum(self._batch_class_entropy(b) for b in self._batches_) / len(self._batches_)

        record: Dict[str, Any] = {
            "k":               self._k_,
            "h_target":        self._h_target_,
            "h_achieved":      float(self._h_table_[np.argmin(np.abs(self._k_table_ - self._k_))]),
            "mean_entropy":    mean_entropy,
            "val_accuracy":    val_accuracy,
        }

        needed = self._window_ + self._smooth_window_
        if len(self._val_acc_history_) <= needed:
            record["next_k"] = self._k_
            return record

        # Compare a moving average of recent validation accuracy against the
        # moving average from `window` epochs earlier, rather than two raw
        # single-epoch points, which was sensitive enough to noise to flip
        # direction almost every epoch.
        current_avg = sum(self._val_acc_history_[-self._smooth_window_:]) / self._smooth_window_
        past_window = self._val_acc_history_[-(needed):-self._window_]
        past_avg = sum(past_window) / len(past_window)
        improvement = current_avg - past_avg

        record["improvement"] = improvement
        record["current_avg"] = current_avg
        record["past_avg"] = past_avg

        # Track the historical peak of the improvement rate, then trigger a
        # ratchet step when the current rate falls below a fraction of that
        # peak, rather than comparing to a fixed absolute value. A fixed
        # threshold (e.g. 0.02) implicitly assumes every dataset decelerates
        # at the same absolute rate; CIFAR-100 keeps improving substantially
        # for most of a 100-epoch run, well past the point where CIFAR-10
        # plateaus, so a threshold tuned by eye on CIFAR-10 behavior was
        # never well-calibrated for CIFAR-100 to begin with. Comparing against
        # each run's own peak rate adapts automatically to how fast that
        # particular dataset/model/metric combination actually learns,
        # without needing a hand-picked constant.
        if self._peak_improvement_ is None or improvement > self._peak_improvement_:
            self._peak_improvement_ = improvement

        record["peak_improvement"] = self._peak_improvement_

        if self._peak_improvement_ <= 0:
            # Degenerate case: never seen positive improvement. Treat any
            # non-positive reading as a plateau rather than dividing by a
            # non-positive peak.
            is_plateau = improvement <= 0
        else:
            is_plateau = improvement < (self._decel_thresh_ * self._peak_improvement_)

        record["is_plateau"] = is_plateau

        # Ratchet: h_target only ever moves toward more diversity, never back
        # toward more structure. A plateau/deceleration epoch advances it by
        # one step; a genuinely-improving epoch just holds it where it is.
        # This replaces the earlier hysteresis-based design, which delayed
        # reversals but still allowed them. On CIFAR-100, allowing reversal at
        # all meant the policy repeatedly re-committed to full homogeneity
        # every time real improvement resumed (which, on CIFAR-100, is most
        # of the run), producing a result close to the static curriculum's
        # rather than anything meaningfully adaptive. The ratchet forbids that
        # by construction: the transition toward diversity happens once,
        # whenever the data says to, and never undoes itself.
        if not is_plateau:
            record["next_k"] = self._k_
            return record

        self._h_target_ = max(self._h_floor_, self._h_target_ - self._step_)
        new_k = self._nearest_k(self._h_target_)
        record["next_k"] = new_k

        if new_k != self._k_:
            self._k_ = new_k
            self._batches_ = self._build(self._k_)

        return record

    # DUNDERS ======================================================================================

    def __iter__(self):
        yield from self._batches_

    def __len__(self) -> int:
        return len(self._batches_)


def run(
    network_id:          str,
    dataset_id:           str,
    metric:                str,
    rank:                  str =              "ascending",
    batch_size:            int =              128,
    epochs:                int =              200,
    seed:                  int =              1,
    window:                int =              5,
    smooth_window:         int =              3,
    step:                  float =            0.05,
    deceleration_threshold: float =            0.05,
    h_floor:                float =            0.0,
    initial_h_target:      float =            1.0,
    device:                str =              "auto",
    output_path:           str =              "results/stride_curriculum_experiment",
    cache_path:            str =              ".cache",
) -> Dict[str, Any]:
    """# Run Isolated Stride-Homogeneity Curriculum Experiment."""
    logger = get_logger("stride-curriculum-experiment")

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

    scores = DatasetMetrics(
        dataset_id =  dataset_id,
        num_samples = len(dataset.train_data),
        seed =        seed,
        scores_path = f"{cache_path}/scores"
    )

    labels = get_labels(dataset)

    sampler = StrideHomogeneityCurriculum(
        dataset_id =         dataset_id,
        scores =             scores,
        metric =             metric,
        rank =               rank,
        batch_size =         batch_size,
        num_classes =        dataset.num_classes,
        labels =             labels,
        seed =               seed,
        window =             window,
        smooth_window =      smooth_window,
        step =               step,
        deceleration_threshold = deceleration_threshold,
        h_floor =            h_floor,
        initial_h_target =   initial_h_target
    )
    print(f"Homogeneity table built: {len(sampler._k_table_)} valid k values, "
                f"h range [{sampler._h_table_.min():.4f}, {sampler._h_table_.max():.4f}]")

    train_loader = DataLoader(
        dataset =       dataset.train_data,
        batch_sampler = sampler,
        num_workers =   dataset._max_workers_,
        pin_memory =    True
    )
    test_loader = dataset.test_loader

    optimizer = SGD(
        params =        network.parameters(),
        lr =            0.1,
        weight_decay =  5e-4,
        momentum =      0.9
    )
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    epoch_records:  List[Dict[str, Any]] = []
    control_log:     List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):

        network.train()
        train_total_loss, train_correct, train_total = 0.0, 0, 0

        for samples, targets in tqdm(
            train_loader, desc = f"Epoch {epoch}/{epochs} [train, k={sampler._k_}]", leave = False
        ):
            samples, targets = samples.to(resolved_device), targets.to(resolved_device)

            predictions: Tensor = network(samples)
            loss:        Tensor = cross_entropy(input = predictions, target = targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            _, predicted = predictions.max(dim = 1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

        network.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0

        with no_grad():
            for samples, targets in tqdm(
                test_loader, desc = f"Epoch {epoch}/{epochs} [val]", leave = False
            ):
                samples, targets = samples.to(resolved_device), targets.to(resolved_device)
                predictions: Tensor = network(samples)
                loss:        Tensor = cross_entropy(input = predictions, target = targets)

                val_total_loss += loss.item()
                _, predicted = predictions.max(dim = 1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)

        val_accuracy = round(val_correct / val_total, 4)

        control_record = sampler.adapt(val_accuracy)
        control_log.append({"epoch": epoch, **control_record})
        print(
            f"Epoch {epoch:3}: k={control_record['k']} -> {control_record['next_k']}  "
            f"h_target={control_record['h_target']:.3f} h_achieved={control_record['h_achieved']:.3f}  "
            f"entropy={control_record['mean_entropy']:.4f}"
        )

        record: Dict[str, Any] = {
            "epoch":          epoch,
            "k":              control_record["k"],
            "h_target":       control_record["h_target"],
            "h_achieved":     control_record["h_achieved"],
            "mean_entropy":   control_record["mean_entropy"],
            "train_loss":     round(train_total_loss / len(train_loader), 4),
            "train_accuracy": round(train_correct / train_total, 4),
            "val_loss":       round(val_total_loss / len(test_loader), 4),
            "val_accuracy":   val_accuracy,
        }
        epoch_records.append(record)

        print(
            f"Epoch {epoch:3}/{epochs}: "
            f"Train Accuracy = {record['train_accuracy']:.4f}; "
            f"Train Loss = {record['train_loss']:.4f}; "
            f"Validation Accuracy = {record['val_accuracy']:.4f}; "
            f"Validation Loss = {record['val_loss']:.4f}"
        )

        scheduler.step()

    result: Dict[str, Any] = {
        "config": {
            "network_id":          network_id,
            "dataset_id":          dataset_id,
            "metric":              metric,
            "rank":                rank,
            "batch_size":          batch_size,
            "epochs":              epochs,
            "seed":                seed,
            "window":              window,
            "smooth_window":       smooth_window,
            "step":                step,
            "deceleration_threshold": deceleration_threshold,
            "h_floor":             h_floor,
            "initial_h_target":    initial_h_target,
        },
        "epochs":       epoch_records,
        "control_log":  control_log,
    }

    _save(result = result, output_path = output_path, seed = seed, t = deceleration_threshold)

    return result


def _save(result: Dict[str, Any], output_path: str, seed: int, t: float) -> None:
    """# Write Results to Disk (plain JSON/CSV, not TrainingRecord)."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    tag = f"seed-{seed}"
    with open(out_dir / f"{tag}_thresh-{t}.json", "w") as f:
        json.dump(result, f, indent = 2)

    # with open(out_dir / f"{tag}.csv", "w", newline = "") as f:
    #     writer = csv.DictWriter(f, fieldnames = list(result["epochs"][0].keys()))
    #     writer.writeheader()
    #     writer.writerows(result["epochs"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Stride-constructed homogeneity curriculum experiment.")
    parser.add_argument("--network-id",  type = str, default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-10")
    parser.add_argument("--metric",       type = str,   default = "wavelet-energy")
    parser.add_argument("--rank",         type = str,   default = "ascending")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 100)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument(
        "--window", type = int, default = 5,
        help = "Epochs back to compare the smoothed validation accuracy against for plateau detection."
    )
    parser.add_argument(
        "--smooth-window", type = int, default = 3,
        help = "Epochs to average over when computing the current and past validation-accuracy points."
    )
    parser.add_argument(
        "--step", type = float, default = 0.05,
        help = "Amount to move h_target by when improving vs. plateauing."
    )
    parser.add_argument(
        "--deceleration-threshold", type = float, default = 0.05,
        help = "Trigger a ratchet step once the smoothed improvement rate falls below this fraction "
               "of its own historical peak. Simulated against a real CIFAR-100 run before this default "
               "was chosen: 0.3 (an earlier guess) reached the diversity floor by epoch 36, since the "
               "peak is set during the initial fast-learning burst and is much higher than later "
               "sustained-but-real improvement. 0.05 reaches the floor around epoch 80, matching where "
               "genuine improvement was independently observed to taper off. Re-check this if you "
               "change dataset, metric, or architecture, since the right value depends on how much "
               "higher the early burst is than the later sustained rate, which isn't necessarily the "
               "same across settings."
    )
    parser.add_argument(
        "--h-floor", type = float, default = 0.0,
        help = "Minimum h_target the policy can reach (ratchet floor). Prevents full convergence to "
               "shuffling-equivalent behavior (h=0), so the policy retains some difficulty structure "
               "for the entire run."
    )
    parser.add_argument(
        "--initial-h-target", type = float, default = 1.0,
        help = "Starting target homogeneity. 1.0 = start at holistic (max homogeneity)."
    )
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/stride_curriculum_experiment")
    parser.add_argument("--cache-path",   type = str,   default = ".cache")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run(
        network_id =          args.network_id,
        dataset_id =           args.dataset_id,
        metric =               args.metric,
        rank =                 args.rank,
        batch_size =           args.batch_size,
        epochs =               args.epochs,
        seed =                 args.seed,
        window =               args.window,
        smooth_window =        args.smooth_window,
        step =                 args.step,
        deceleration_threshold = args.deceleration_threshold,
        h_floor =              args.h_floor,
        initial_h_target =     args.initial_h_target,
        device =               args.device,
        output_path =          args.output_path,
        cache_path =           args.cache_path,
    )