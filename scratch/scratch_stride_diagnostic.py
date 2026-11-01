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
from typing   import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
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
        fixed_h_target:       Optional[float] = None,
        phase_schedule:       Optional[List[Tuple[int, float]]] = None,
        phase_ramp_epochs:    int = 0,
        grad_var_triggered:   bool = False,
        grad_var_threshold:   float = 0.7,
        grad_var_smooth_window: int = 3,
        grad_var_force_floor_epoch: Optional[int] = None,
        sort_seed:            Optional[int] = None,
        local_pool:           bool = False,
        local_pool_refresh_epochs: int = 1,
        local_pool_network:   Optional[Any] = None,
        local_pool_dataset:   Optional[Any] = None,
        local_pool_device:    Optional[Any] = None,
        local_pool_rank_size: Optional[int] = None,
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
        # When set, adapt() never changes h_target/k: no ratchet, no trigger
        # logic, no moving-average bookkeeping. h_target is fixed at this
        # value for the entire run. This is a separate mode from the ratchet,
        # not the ratchet with a threshold tuned to never fire: it exists to
        # test whether avoiding h_target=1.0 (shown to cost early-epoch
        # gradient diversity, see scratch_stride_diagnostic.py header) at a
        # constant, non-adaptive homogeneity level changes the outcome, with
        # no confound from any adaptive logic at all.
        self._fixed_h_target_:   Optional[float] =    fixed_h_target
        # Piecewise fixed schedule: list of (epoch, h_target) breakpoints,
        # sorted ascending by epoch, e.g. [(1, 0.0), (11, 0.5), (71, 0.0)]
        # means h_target=0.0 for epochs 1-10, 0.5 for epochs 11-70, back to
        # 0.0 for epochs 71+. Like fixed_h_target, bypasses the ratchet's
        # trigger/moving-average logic entirely; h_target is a pure lookup
        # against epoch number, not driven by any measured signal. Exists to
        # test whether avoiding h_target=1.0 in both the early window
        # (epochs 1-5, confirmed to cost gradient diversity) and the late
        # window (epoch ~70+, where the single-fixed-value h=0.5 run lost
        # ~3pp relative to both shuffling and the ratchet) while keeping
        # some structure only in the middle, recovers the deficit either
        # mechanism showed on its own. Mutually exclusive with
        # fixed_h_target; takes priority if both are somehow set.
        self._phase_schedule_:   Optional[List[Tuple[int, float]]] = (
            sorted(phase_schedule, key = lambda p: p[0]) if phase_schedule else None
        )
        # 0 = instant switch at each breakpoint (original phase-schedule
        # behavior). > 0 = linearly ramp k/h_target over that many epochs
        # instead, see _h_target_for_epoch.
        self._phase_ramp_epochs_: int = phase_ramp_epochs
        # Gradient-variance-triggered mode: same one-way monotonic shape as
        # the original ratchet (only ever decreases h_target, never
        # reverses; asymmetric hysteresis/reversal was confirmed harmful in
        # the original val-accuracy-driven ratchet, see history of dead
        # ends), but the trigger is a live, measured signal instead of
        # lagging validation accuracy: mean_batch_grad_norm_var (real
        # analytic per-sample output-layer gradient norm variance,
        # confirmed via diagnostic runs to track both the early-epoch-1-5
        # and the late-epoch-70+ homogeneity harm) compared each epoch
        # against a probe measurement of what a plain shuffled batch would
        # produce at the current model state. run() supplies both
        # measurements to adapt(); this class only does the trigger/step
        # logic and a short moving average, it never touches the network.
        self._grad_var_triggered_:    bool =  grad_var_triggered
        self._grad_var_threshold_:    float = grad_var_threshold
        self._grad_var_smooth_window_: int =  grad_var_smooth_window
        self._grad_var_ratio_history_: List[float] = []
        # Safety net, not a substitute for the measured trigger: the ratio
        # signal was confirmed to become unreliable near convergence, since
        # both the measured and probe gradient-norm variances shrink toward
        # zero and their ratio's relative noise blows up (observed swinging
        # 0.16-4.7 epoch-to-epoch once mean_batch_grad_norm dropped below
        # ~0.02 in an earlier run), so the trigger can stall well short of
        # h_floor even though the fixed-h_target=0.5 diagnostic independently
        # confirmed reaching full diversity by end of training matters. If
        # set, from this epoch onward h_target is forced down by one step
        # every epoch regardless of the ratio, guaranteeing h_floor is
        # reached before training ends.
        self._grad_var_force_floor_epoch_: Optional[int] = grad_var_force_floor_epoch

        # Local-pool mode: instead of one static offline global sort held
        # fixed for the whole run, each refresh shuffles the dataset fresh,
        # chunks it into pools of size batch_size*k, and ranks difficulty
        # only *within* each pool using the current model's own live
        # per-sample gradient norm (same analytic softmax-onehot signal used
        # throughout the diagnostics). Motivated directly by the confirmed
        # finding that the static offline score increasingly diverges from
        # the model's real, evolving difficulty ordering as training
        # progresses. The k<->h(k) table below still comes from the static
        # sort and is only used to translate a target homogeneity into a
        # pool-size multiplier; only *which samples end up in a pool, and
        # their live-scored order within it* changes with this mode, not the
        # k-selection control logic (still driven by whichever of
        # fixed/phase/grad_var/ratchet mode is active).
        self._local_pool_:                bool =              local_pool
        self._local_pool_refresh_epochs_: int =                local_pool_refresh_epochs
        self._local_pool_network_:        Optional[Any] =     local_pool_network
        self._local_pool_device_:         Optional[Any] =     local_pool_device
        self._local_pool_epoch_counter_:  int =                0
        self._local_pool_rng_:            np.random.RandomState = np.random.RandomState(seed)
        self._local_pool_score_loader_:   Optional[DataLoader] = None
        # Decouples "how large a sample to rank difficulty within" from
        # "how many samples end up biased together into k batches". Without
        # this, both are the same number (batch_size*k), so aggressive
        # homogeneity settings (small k) also shrink the ranking pool down
        # to a small, noisy sample right when the ranking is leaned on
        # hardest. None = old behavior (rank_size == batch_size*k). Set to
        # a larger fixed value (or the full dataset size) to rank over a
        # bigger, more stable, still-live-refreshed sample, then still carve
        # batch_size*k-sized chunks out of that larger sorted ranking for
        # the actual b::k interleave construction.
        self._local_pool_rank_size_:      Optional[int] = local_pool_rank_size
        if local_pool:
            if local_pool_network is None or local_pool_dataset is None or local_pool_device is None:
                raise ValueError(
                    "local_pool=True requires local_pool_network, local_pool_dataset, and "
                    "local_pool_device (needed to compute live per-sample scores each refresh)."
                )
            self._local_pool_score_loader_ = DataLoader(
                dataset =     local_pool_dataset,
                batch_size =  256,
                shuffle =     False,
                num_workers = 0,
                pin_memory =  True
            )

        # Holistic difficulty order, computed once. Only the batch construction
        # (via k) changes epoch to epoch, never the underlying ranking.
        #
        # sort_seed (defaults to `seed` when unset, preserving old behavior
        # exactly) decouples the difficulty sort from the training seed.
        # RANK_REGISTRY.sort_indices takes seed as a parameter and was
        # confirmed to sometimes shift the resulting k<->h(k) mapping
        # noticeably across seeds (e.g. k=310 vs k=199 for the identical
        # h_target=0.5 in one local-pool comparison) — every training seed
        # was quietly starting from a different structural configuration,
        # not just different weight init/data order, inflating run-to-run
        # variance for reasons unrelated to whether the mechanism works.
        # Passing a fixed sort_seed makes every training seed start from
        # the identical difficulty ordering.
        self._sorted_indices_:  List[int] =          RANK_REGISTRY.sort_indices(
                                                          rank_id =    rank,
                                                          dataset_id = dataset_id,
                                                          metric =     self._metric_,
                                                          scores =     scores.scores,
                                                          seed =       sort_seed if sort_seed is not None else seed
                                                      )
        self._n_batches_:       int =                math.ceil(len(self._sorted_indices_) / batch_size)

        # Homogeneity is a deterministic function of k, given the fixed sorted
        # order and batch size, so it's computed once here rather than
        # inferred from any training-time measurement.
        self._k_table_, self._h_table_ = self._build_homogeneity_table()

        self._h_target_:        float =              self._initial_h_target(fixed_h_target, initial_h_target)
        self._k_:               int =                self._nearest_k(self._h_target_)
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

    def _h_target_for_epoch(self, epoch: int) -> float:
        """# Look Up the Phase-Schedule h_target for a Given (1-Indexed) Epoch.

        Returns the value attached to the latest breakpoint at or before
        `epoch`, unless `_phase_ramp_epochs_` is set, in which case the
        value is linearly interpolated between the previous breakpoint's
        h_target and the new one over that many epochs starting at the
        breakpoint, rather than switching in a single epoch. Motivated by
        the phase-schedule diagnostic result: an instant k=310->390 switch
        at epoch 71 cost a full point of val accuracy in that one epoch and
        was never recovered, while the ratchet (which only ever moves k in
        small discrete steps, never a single large jump) showed no
        comparable single-epoch drop anywhere in its run. Only meaningful
        when `_phase_schedule_` is set.
        """
        idx = 0
        for i, (e, _) in enumerate(self._phase_schedule_):
            if e <= epoch:
                idx = i
            else:
                break

        breakpoint_epoch, target_h = self._phase_schedule_[idx]

        if idx == 0 or not self._phase_ramp_epochs_:
            return target_h

        prev_epoch, prev_h = self._phase_schedule_[idx - 1]
        progress = min(1.0, (epoch - breakpoint_epoch) / self._phase_ramp_epochs_)
        return prev_h + (target_h - prev_h) * progress

    def _initial_h_target(self, fixed_h_target: Optional[float], initial_h_target: float) -> float:
        if self._phase_schedule_ is not None:
            return self._h_target_for_epoch(1)
        return fixed_h_target if fixed_h_target is not None else initial_h_target

    def _nearest_k(self, h_target: float) -> int:
        """# Find the k Whose Precomputed h(k) Is Closest to a Target."""
        idx = int(np.argmin(np.abs(self._h_table_ - h_target)))
        return int(self._k_table_[idx])

    def _compute_live_scores(self) -> np.ndarray:
        """# Forward-Pass-Only Live Difficulty Scores for Every Sample.

        Same analytic per-sample output-layer gradient norm used throughout
        the diagnostics (softmax(logits) - onehot(target), exact, no
        backward pass needed). Runs the current network in eval() mode
        (restoring whatever mode it was in beforehand) over the full
        dataset via a shuffle=False loader, so returned scores align
        index-for-index with the dataset's natural ordering.
        """
        network = self._local_pool_network_
        was_training = network.training
        network.eval()

        scores = np.empty(len(self._sorted_indices_), dtype = np.float32)
        ptr = 0
        with no_grad():
            for samples, targets in self._local_pool_score_loader_:
                samples = samples.to(self._local_pool_device_)
                targets = targets.to(self._local_pool_device_)
                predictions = network(samples)
                probs = predictions.softmax(dim = 1)
                one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
                sample_grad_norms = (probs - one_hot).norm(dim = 1)
                n = sample_grad_norms.shape[0]
                scores[ptr:ptr + n] = sample_grad_norms.cpu().numpy()
                ptr += n

        if was_training:
            network.train()

        return scores

    def _build_local_pool(self, k: int) -> List[List[int]]:
        """# Construct Batches from a Freshly-Shuffled, Locally-Scored Pool Partition.

        Unlike `_build`, which slices a single static global sort held fixed
        for the whole run, this reshuffles the dataset from scratch and
        ranks difficulty using this epoch's live per-sample gradient norm.
        Pool membership and ranking both change on every call, driven by
        real, current model state rather than a fixed offline score.

        Two nested groupings, controlled separately:
          * rank_group_sz (local_pool_rank_size, or batch_size*k if unset):
            how many freshly-shuffled samples get ranked together each time.
          * supergroup_sz (batch_size*k, always): within a sorted rank
            group, how many samples get carved into one b::k interleaved
            set of k batches. When rank_group_sz > supergroup_sz, one
            ranked group yields several such sets, each spanning a narrower
            difficulty slice of the larger, more stably-ranked group.
        """
        n = len(self._sorted_indices_)
        scores = self._compute_live_scores()
        perm = self._local_pool_rng_.permutation(n)
        supergroup_sz = self._batch_size_ * k
        rank_group_sz = self._local_pool_rank_size_ or supergroup_sz
        # A rank group must be built from whole supergroups so every batch
        # produced is still exactly batch_size, and must cover at least one
        # supergroup.
        rank_group_sz = max(supergroup_sz, (rank_group_sz // supergroup_sz) * supergroup_sz)

        batches: List[List[int]] = []
        for rg_start in range(0, n, rank_group_sz):
            rank_group = perm[rg_start:rg_start + rank_group_sz]

            if len(rank_group) < supergroup_sz:
                for i in range(0, len(rank_group), self._batch_size_):
                    batches.append(rank_group[i:i + self._batch_size_].tolist())
                continue

            order = np.argsort(scores[rank_group])
            sorted_group = rank_group[order]

            for start in range(0, len(sorted_group), supergroup_sz):
                pool = sorted_group[start:start + supergroup_sz]
                if len(pool) < supergroup_sz:
                    for i in range(0, len(pool), self._batch_size_):
                        batches.append(pool[i:i + self._batch_size_].tolist())
                    continue
                for b in range(k):
                    batches.append(pool[b::k].tolist())

        return batches

    def _build(self, k: int) -> List[List[int]]:
        """# Construct Batches at a Given Stride k."""
        if self._local_pool_:
            return self._build_local_pool(k)

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

    def adapt(
        self, val_accuracy: float,
        measured_grad_var: Optional[float] = None,
        probe_grad_var: Optional[float] = None,
    ) -> Dict[str, Any]:
        """# Update h_target, Adjust k.

        Called once per epoch, after that epoch's validation accuracy (and,
        in grad_var_triggered mode, this epoch's measured/probe gradient
        variance) is known, so k for the *next* epoch reflects this epoch's
        result. Before enough history exists for the relevant moving
        average, k is left at its current value.

        ## Args:
            * val_accuracy (float): This epoch's validation accuracy.
            * measured_grad_var (Optional[float]): This epoch's mean batch
              gradient-norm variance under the curriculum's own batches.
              Only used in grad_var_triggered mode.
            * probe_grad_var (Optional[float]): This epoch's mean batch
              gradient-norm variance measured on a plain-shuffled probe
              batch at the same model state. Only used in grad_var_triggered
              mode.

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

        # Phase-schedule mode: h_target for the next epoch is a pure lookup
        # against epoch number, no trigger, no moving averages, no measured
        # signal at all. Takes priority over fixed_h_target if both are set.
        if self._phase_schedule_ is not None:
            next_epoch = len(self._val_acc_history_) + 1
            new_h_target = self._h_target_for_epoch(next_epoch)
            new_k = self._nearest_k(new_h_target)
            record["next_k"] = new_k
            if new_k != self._k_ or new_h_target != self._h_target_:
                self._h_target_ = new_h_target
                self._k_ = new_k
                self._batches_ = self._build(self._k_)
            return record

        # Fixed mode: no trigger, no moving averages, no peak tracking. k
        # never changes from what was set at construction, for every epoch
        # of the run.
        if self._fixed_h_target_ is not None:
            record["next_k"] = self._k_
            return record

        # Gradient-variance-triggered mode: real-time measured signal, no
        # dependence on validation-accuracy history at all. Ratio < 1 means
        # this epoch's curriculum batches were less internally diverse
        # (gradient-wise) than a plain shuffled batch would be at the same
        # model state; ratio ~1 means the curriculum isn't costing anything
        # right now, whatever h_target currently is. Smoothed over a short
        # window since a single epoch's ratio is noisy (small probe sample),
        # but short enough (default 3) to react much faster than the
        # original ratchet's ~26-epoch lag.
        if self._grad_var_triggered_:
            if measured_grad_var is not None and probe_grad_var:
                ratio = measured_grad_var / probe_grad_var
                self._grad_var_ratio_history_.append(ratio)
                record["grad_var_ratio"] = ratio

            next_epoch = len(self._val_acc_history_) + 1
            forcing_floor = (
                self._grad_var_force_floor_epoch_ is not None
                and next_epoch >= self._grad_var_force_floor_epoch_
            )
            record["forcing_floor"] = forcing_floor

            if forcing_floor and self._h_target_ > self._h_floor_:
                self._h_target_ = max(self._h_floor_, self._h_target_ - self._step_)
                new_k = self._nearest_k(self._h_target_)
                if new_k != self._k_:
                    self._k_ = new_k
                    self._batches_ = self._build(self._k_)
            elif len(self._grad_var_ratio_history_) >= self._grad_var_smooth_window_:
                recent = self._grad_var_ratio_history_[-self._grad_var_smooth_window_:]
                smoothed_ratio = sum(recent) / len(recent)
                record["smoothed_grad_var_ratio"] = smoothed_ratio

                if smoothed_ratio < self._grad_var_threshold_ and self._h_target_ > self._h_floor_:
                    self._h_target_ = max(self._h_floor_, self._h_target_ - self._step_)
                    new_k = self._nearest_k(self._h_target_)
                    if new_k != self._k_:
                        self._k_ = new_k
                        self._batches_ = self._build(self._k_)

            record["next_k"] = self._k_
            return record

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
        # DataLoader calls __iter__ once per epoch (each fresh `for x in
        # loader:`), which gives a natural once-per-epoch hook for
        # local-pool mode's periodic refresh without any change needed in
        # run()'s training loop.
        if self._local_pool_ and self._local_pool_epoch_counter_ % self._local_pool_refresh_epochs_ == 0:
            self._batches_ = self._build(self._k_)
        if self._local_pool_:
            self._local_pool_epoch_counter_ += 1
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
    fixed_h_target:         Optional[float] =  None,
    phase_schedule:         Optional[List[Tuple[int, float]]] = None,
    phase_ramp_epochs:      int =              0,
    grad_var_triggered:     bool =             False,
    grad_var_threshold:     float =            0.7,
    grad_var_smooth_window: int =              3,
    grad_var_probe_batches: int =              30,
    grad_var_force_floor_epoch: Optional[int] = None,
    sort_seed:              Optional[int] =    None,
    local_pool:             bool =             False,
    local_pool_refresh_epochs: int =          1,
    local_pool_rank_size:   Optional[int] =    None,
    shuffled:              bool =             False,
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

    # `shuffled=True` bypasses the curriculum sampler entirely and uses plain
    # shuffled batching instead, so this script can produce a reference run
    # with identical model/optimizer/schedule/seed handling and identical
    # instrumentation, for a clean same-code comparison against the ratchet
    # curriculum. sampler stays None in this branch; every downstream use of
    # sampler is guarded accordingly.
    sampler: Optional[StrideHomogeneityCurriculum] = None
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
            initial_h_target =   initial_h_target,
            fixed_h_target =     fixed_h_target,
            phase_schedule =     phase_schedule,
            phase_ramp_epochs =  phase_ramp_epochs,
            grad_var_triggered =    grad_var_triggered,
            grad_var_threshold =    grad_var_threshold,
            grad_var_smooth_window = grad_var_smooth_window,
            grad_var_force_floor_epoch = grad_var_force_floor_epoch,
            sort_seed =          sort_seed,
            local_pool =         local_pool,
            local_pool_refresh_epochs = local_pool_refresh_epochs,
            local_pool_network = network,
            local_pool_dataset = dataset.train_data,
            local_pool_device =  resolved_device,
            local_pool_rank_size = local_pool_rank_size
        )
        if sort_seed is not None:
            print(f"Difficulty sort seed decoupled from training seed: sort_seed={sort_seed}, "
                        f"training seed={seed} (every training seed shares the identical k<->h(k) mapping).")
        if local_pool:
            rank_note = f"rank pool size = {local_pool_rank_size} (decoupled from batch grouping)" if local_pool_rank_size else "rank pool size = batch_size*k (coupled, default)"
            print(f"Local-pool mode: live per-sample gradient-norm scoring, fresh shuffle+pool "
                        f"every {local_pool_refresh_epochs} epoch(s), {rank_note} "
                        f"(k={sampler._k_} at construction).")
        if grad_var_triggered:
            floor_note = f", forcing floor from epoch {grad_var_force_floor_epoch}" if grad_var_force_floor_epoch else ""
            print(f"Gradient-variance-triggered controller: initial h_target={initial_h_target} "
                        f"(k={sampler._k_}), threshold={grad_var_threshold}, "
                        f"smooth_window={grad_var_smooth_window}, probe_batches={grad_var_probe_batches}"
                        f"{floor_note}. "
                        f"h range [{sampler._h_table_.min():.4f}, {sampler._h_table_.max():.4f}]")
        elif phase_schedule is not None:
            ramp_note = f", ramped over {phase_ramp_epochs} epochs per transition" if phase_ramp_epochs else ", instant transitions"
            print(f"Phase schedule={phase_schedule}{ramp_note} (no ratchet, k={sampler._k_} for epoch 1). "
                        f"h range [{sampler._h_table_.min():.4f}, {sampler._h_table_.max():.4f}]")
        elif fixed_h_target is not None:
            print(f"Fixed h_target={fixed_h_target} (no ratchet, k={sampler._k_} for the entire run). "
                        f"h range [{sampler._h_table_.min():.4f}, {sampler._h_table_.max():.4f}]")
        else:
            print(f"Homogeneity table built: {len(sampler._k_table_)} valid k values, "
                        f"h range [{sampler._h_table_.min():.4f}, {sampler._h_table_.max():.4f}]")

        train_loader = DataLoader(
            dataset =       dataset.train_data,
            batch_sampler = sampler,
            num_workers =   dataset._max_workers_,
            pin_memory =    True
        )
    test_loader = dataset.test_loader

    # Probe loader for grad_var_triggered mode: gives a fresh random batch
    # each time iter() is called (shuffle=True), used only to measure what
    # a plain-shuffled batch's gradient-norm variance looks like at the
    # current model state. Forward pass only, eval mode, no backward, no
    # optimizer step, so it doesn't touch training at all and doesn't
    # disturb batchnorm running stats.
    probe_loader: Optional[DataLoader] = None
    if grad_var_triggered:
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

    epoch_records:  List[Dict[str, Any]] = []
    control_log:     List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):

        network.train()
        train_total_loss, train_correct, train_total = 0.0, 0, 0

        # Per-batch diagnostics for within-batch training-dynamics diversity.
        # mean_batch_grad_norm / _var use the analytic per-sample gradient of
        # cross-entropy at the output layer: d(loss_i)/d(logits_i) =
        # softmax(logits_i) - onehot(y_i), exactly, for every sample i. This
        # is real per-sample gradient information, not a proxy for it, and
        # costs nothing beyond tensor ops on predictions already computed;
        # no extra backward pass. batch_loss_vars is a second, independent
        # view of within-batch diversity computed the same cheap way.
        batch_loss_vars:        List[float] = []
        batch_grad_norm_means:  List[float] = []
        batch_grad_norm_vars:   List[float] = []

        for samples, targets in tqdm(
            train_loader,
            desc = f"Epoch {epoch}/{epochs} [train, k={sampler._k_ if sampler else 'shuffled'}]",
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

        measured_grad_var = sum(batch_grad_norm_vars) / len(batch_grad_norm_vars)
        probe_grad_var: Optional[float] = None
        if probe_loader is not None:
            # network is already in eval() mode from the validation loop
            # above; forward pass only, same analytic per-sample gradient
            # norm as the training-loop instrumentation, no backward, no
            # optimizer step, no effect on batchnorm running stats.
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
                f"Epoch {epoch:3}: k={control_record['k']} -> {control_record['next_k']}  "
                f"h_target={control_record['h_target']:.3f} h_achieved={control_record['h_achieved']:.3f}  "
                f"entropy={control_record['mean_entropy']:.4f}{grad_var_note}"
            )
        else:
            control_record = None

        record: Dict[str, Any] = {
            "epoch":                    epoch,
            "k":                        control_record["k"] if control_record else None,
            "h_target":                 control_record["h_target"] if control_record else None,
            "h_achieved":               control_record["h_achieved"] if control_record else None,
            "mean_entropy":             control_record["mean_entropy"] if control_record else None,
            "train_loss":               round(train_total_loss / len(train_loader), 4),
            "train_accuracy":           round(train_correct / train_total, 4),
            "val_loss":                 round(val_total_loss / len(test_loader), 4),
            "val_accuracy":             val_accuracy,
            "mean_batch_loss_var":      round(sum(batch_loss_vars) / len(batch_loss_vars), 6),
            "mean_batch_grad_norm":     round(sum(batch_grad_norm_means) / len(batch_grad_norm_means), 6),
            "mean_batch_grad_norm_var": round(sum(batch_grad_norm_vars) / len(batch_grad_norm_vars), 6),
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
            "fixed_h_target":      fixed_h_target,
            "phase_schedule":      phase_schedule,
            "phase_ramp_epochs":   phase_ramp_epochs,
            "grad_var_triggered":       grad_var_triggered,
            "grad_var_threshold":       grad_var_threshold,
            "grad_var_smooth_window":   grad_var_smooth_window,
            "grad_var_probe_batches":   grad_var_probe_batches,
            "grad_var_force_floor_epoch": grad_var_force_floor_epoch,
            "sort_seed":           sort_seed,
            "local_pool":          local_pool,
            "local_pool_refresh_epochs": local_pool_refresh_epochs,
            "local_pool_rank_size":      local_pool_rank_size,
            "shuffled":            shuffled,
        },
        "epochs":       epoch_records,
        "control_log":  control_log,
    }

    _save(
        result = result, output_path = output_path, seed = seed, t = deceleration_threshold,
        shuffled = shuffled, fixed_h_target = fixed_h_target,
        phase_schedule = phase_schedule, phase_ramp_epochs = phase_ramp_epochs,
        grad_var_triggered = grad_var_triggered, initial_h_target = initial_h_target,
        grad_var_threshold = grad_var_threshold, grad_var_force_floor_epoch = grad_var_force_floor_epoch,
        local_pool = local_pool, local_pool_refresh_epochs = local_pool_refresh_epochs,
        local_pool_rank_size = local_pool_rank_size, sort_seed = sort_seed
    )

    return result


def _save(
    result: Dict[str, Any], output_path: str, seed: int, t: float,
    shuffled: bool = False, fixed_h_target: Optional[float] = None,
    phase_schedule: Optional[List[Tuple[int, float]]] = None, phase_ramp_epochs: int = 0,
    grad_var_triggered: bool = False, initial_h_target: float = 1.0, grad_var_threshold: float = 0.7,
    grad_var_force_floor_epoch: Optional[int] = None,
    local_pool: bool = False, local_pool_refresh_epochs: int = 1,
    local_pool_rank_size: Optional[int] = None, sort_seed: Optional[int] = None
) -> None:
    """# Write Results to Disk (plain JSON/CSV, not TrainingRecord).

    tag includes a shuffled/curriculum/fixed/phase/grad-var/local-pool marker
    on top of seed, on top of the existing seed-only naming footgun (same
    seed run twice with the same output_path silently overwrites): any two
    of these run types at the same seed would otherwise collide on the exact
    same filename, silently destroying one of the two runs this diagnostic
    pass depends on.
    """
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    rank_str = f"-rank{local_pool_rank_size}" if local_pool_rank_size else ""
    local_pool_str = f"-localpool{local_pool_refresh_epochs}{rank_str}" if local_pool else ""
    sort_seed_str = f"-sortseed{sort_seed}" if sort_seed is not None else ""

    if shuffled:
        tag = f"seed-{seed}_shuffled"
    elif grad_var_triggered:
        floor_str = f"-floor{grad_var_force_floor_epoch}" if grad_var_force_floor_epoch else ""
        tag = f"seed-{seed}_gradvar-init{initial_h_target}-thresh{grad_var_threshold}{floor_str}{local_pool_str}{sort_seed_str}"
    elif phase_schedule is not None:
        phase_str = "-".join(f"{e}@{v}" for e, v in phase_schedule)
        ramp_str = f"_ramp{phase_ramp_epochs}" if phase_ramp_epochs else ""
        tag = f"seed-{seed}_phase-{phase_str}{ramp_str}"
    elif fixed_h_target is not None:
        tag = f"seed-{seed}_fixed-h{fixed_h_target}"
    else:
        tag = f"seed-{seed}_curriculum_thresh-{t}"

    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent = 2)

    # with open(out_dir / f"{tag}.csv", "w", newline = "") as f:
    #     writer = csv.DictWriter(f, fieldnames = list(result["epochs"][0].keys()))
    #     writer.writeheader()
    #     writer.writerows(result["epochs"])


def _parse_phase_schedule(s: str) -> List[Tuple[int, float]]:
    """# Parse a Phase-Schedule CLI String.

    Format: "epoch:h_target,epoch:h_target,...", e.g.
    "1:0.0,11:0.5,71:0.0" means h_target=0.0 for epochs 1-10, 0.5 for
    epochs 11-70, back to 0.0 for epochs 71 onward. Breakpoints need not be
    given in sorted order here; the sampler sorts them.
    """
    phases: List[Tuple[int, float]] = []
    for chunk in s.split(","):
        epoch_str, h_str = chunk.split(":")
        phases.append((int(epoch_str), float(h_str)))
    return phases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Stride-constructed homogeneity curriculum experiment.")
    parser.add_argument("--network-id",  type = str, default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-100")
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
        help = "Starting target homogeneity. 1.0 = start at holistic (max homogeneity). Ignored if "
               "--fixed-h-target is set."
    )
    parser.add_argument(
        "--fixed-h-target", type = float, default = None,
        help = "If set, disables the ratchet entirely: h_target (and therefore k) is held at this "
               "value for every epoch of the run, no trigger, no moving averages, no step. Mutually "
               "exclusive in effect with --initial-h-target/--deceleration-threshold/--step/--h-floor, "
               "which are all ignored once this is set (their values are still recorded in config for "
               "the record, but do nothing). Overridden by --phase-schedule if both are set."
    )
    parser.add_argument(
        "--phase-schedule", type = _parse_phase_schedule, default = None,
        help = "Piecewise fixed h_target schedule, no ratchet, no measured signal: "
               "'epoch:h_target,epoch:h_target,...', e.g. '1:0.0,11:0.5,71:0.0' holds h_target=0.0 "
               "for epochs 1-10, 0.5 for epochs 11-70, back to 0.0 for epochs 71+. Overrides "
               "--fixed-h-target and --initial-h-target if set."
    )
    parser.add_argument(
        "--phase-ramp-epochs", type = int, default = 0,
        help = "If > 0 and --phase-schedule is set, linearly ramp h_target/k over this many epochs "
               "at each breakpoint instead of switching in a single epoch. Default 0 = instant "
               "switch, matching the original phase-schedule behavior. Motivated by a confirmed "
               "cost from an instant k=310->390 switch (one epoch, -1pp val accuracy, never fully "
               "recovered) in an earlier phase-schedule run."
    )
    parser.add_argument(
        "--grad-var-triggered", action = "store_true",
        help = "Replace the val-accuracy-deceleration trigger with a live, measured one: each "
               "epoch, compare this epoch's mean_batch_grad_norm_var against a plain-shuffled "
               "probe measured at the same model state, and ratchet h_target down (one-way, same "
               "as the original ratchet, never reverses) when the smoothed ratio falls below "
               "--grad-var-threshold. Use --initial-h-target to start below 1.0 (avoids the "
               "confirmed epoch 1-5 harm structurally, no trigger needed for that part)."
    )
    parser.add_argument(
        "--grad-var-threshold", type = float, default = 0.7,
        help = "Ratchet-step trigger for --grad-var-triggered: smoothed curriculum/shuffled "
               "gradient-norm-variance ratio below this fires a step. Not yet calibrated against "
               "real logs; verify via simulation before trusting, same as deceleration_threshold "
               "was for the original ratchet."
    )
    parser.add_argument(
        "--grad-var-smooth-window", type = int, default = 3,
        help = "Epochs to average the measured/probe grad-var ratio over before triggering, for "
               "--grad-var-triggered. Short by design (default 3) so this reacts far faster than "
               "the original ratchet's ~26-epoch lag, while still smoothing single-epoch probe noise."
    )
    parser.add_argument(
        "--grad-var-probe-batches", type = int, default = 30,
        help = "Number of random shuffled batches averaged per epoch for the probe reference in "
               "--grad-var-triggered mode. Forward pass only, no backward, no optimizer step. "
               "Raised from an initial default of 3 (then 15) after real runs showed the ratio "
               "becoming unusably noisy once gradients shrank near convergence; the curriculum "
               "side already averages ~370+ batches per epoch, the probe was the noisy side. "
               "Raised again 15->30 as a second noise-reduction pass targeting seed-to-seed "
               "variance in exactly when the controller fires."
    )
    parser.add_argument(
        "--grad-var-force-floor-epoch", type = int, default = None,
        help = "Safety net for --grad-var-triggered: from this epoch onward, force h_target down "
               "by one step every epoch regardless of the ratio, guaranteeing h_floor is reached "
               "before training ends. Motivated by a confirmed real run where the ratio-only "
               "trigger stalled at h_target=0.10 (never reaching h_floor=0.0) because the ratio "
               "signal itself becomes unreliable near convergence, even though reaching h_floor by "
               "end of training was independently confirmed to matter (h_floor=0.15 underperformed "
               "0.0 in four earlier controlled tests). Not a replacement for the measured trigger, "
               "just a guarantee for the regime where that trigger can't be trusted."
    )
    parser.add_argument(
        "--sort-seed", type = int, default = None,
        help = "Decouples the difficulty sort from the training seed (default: unset, uses the "
               "training seed exactly as before). RANK_REGISTRY.sort_indices takes seed as a "
               "parameter and was confirmed to sometimes shift the resulting k<->h(k) mapping "
               "noticeably across seeds, meaning every training seed was quietly starting from a "
               "different structural configuration, not just different weight init/data order. "
               "Setting this to a fixed value (e.g. 1) makes every training seed share the "
               "identical difficulty ordering, removing that as a source of run-to-run variance."
    )
    parser.add_argument(
        "--local-pool", action = "store_true",
        help = "Replace the static, offline, whole-dataset global sort with a freshly-shuffled, "
               "locally-scored pool partition every refresh: shuffle the dataset, chunk into pools "
               "of size batch_size*k, rank difficulty only within each pool using this epoch's live "
               "per-sample gradient norm (same analytic signal used throughout), instead of a score "
               "computed once before training and held fixed. Composable with --grad-var-triggered, "
               "--fixed-h-target, --phase-schedule, or the original ratchet: this only changes how "
               "batches are built from a given k, not how k itself is chosen."
    )
    parser.add_argument(
        "--local-pool-refresh-epochs", type = int, default = 1,
        help = "How often (in epochs) --local-pool reshuffles and rescores pools. Default 1 = "
               "every epoch. Each refresh costs one extra forward-only pass over the full dataset "
               "(no backward, no optimizer step)."
    )
    parser.add_argument(
        "--local-pool-rank-size", type = int, default = None,
        help = "Decouples ranking-sample size from batch-grouping size in --local-pool mode. "
               "Default (unset) ranks difficulty within a pool of exactly batch_size*k freshly "
               "shuffled samples, same size as the batches it produces — at aggressive (small) k "
               "this can mean ranking from a small, noisy sample right when the ranking matters "
               "most. Setting this to a larger value (e.g. several thousand, or the full dataset "
               "size) ranks over a bigger, more stable, still-live-refreshed sample each time, "
               "then carves batch_size*k-sized chunks out of that larger sorted ranking for the "
               "actual batches. Rounded down to a multiple of batch_size*k internally."
    )
    parser.add_argument(
        "--shuffled", action = "store_true",
        help = "Bypass the curriculum sampler; plain shuffled batching, for a same-code diagnostic "
               "reference run. All other config (model, optimizer, schedule, seed, instrumentation) "
               "stays identical, so the only difference vs. a curriculum run is batch composition."
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
        fixed_h_target =       args.fixed_h_target,
        phase_schedule =       args.phase_schedule,
        phase_ramp_epochs =    args.phase_ramp_epochs,
        grad_var_triggered =       args.grad_var_triggered,
        grad_var_threshold =       args.grad_var_threshold,
        grad_var_smooth_window =   args.grad_var_smooth_window,
        grad_var_probe_batches =   args.grad_var_probe_batches,
        grad_var_force_floor_epoch = args.grad_var_force_floor_epoch,
        sort_seed =             args.sort_seed,
        local_pool =           args.local_pool,
        local_pool_refresh_epochs = args.local_pool_refresh_epochs,
        local_pool_rank_size = args.local_pool_rank_size,
        shuffled =             args.shuffled,
        device =               args.device,
        output_path =          args.output_path,
        cache_path =           args.cache_path,
    )