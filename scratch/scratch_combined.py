"""# combined_experiment.py

Combined experiment: the validated saturation-time homogeneity ratchet (batch
composition, driven by validation-accuracy deceleration) and the gradient-
noise controller (optimization-time noise, driven by gradient-norm
deceleration) run simultaneously and independently in the same training loop.

Neither mechanism is aware of the other. Each watches its own signal and
ratchets its own controlled quantity down on its own schedule. This is
deliberate: the hypothesis being tested is that these two mechanisms address
different points in the training pipeline (batch composition vs. per-step
optimization noise) and may compound rather than duplicate each other's
effect. If they turn out to trigger their respective transitions at very
similar epochs, driven by the same underlying "the model is still learning
fast" signal, that would suggest they're capturing overlapping ground rather
than genuinely separate mechanisms; the per-epoch logs from both controllers
are recorded specifically so that can be checked after the fact.

Both classes below are copied verbatim from the individually-validated
standalone scripts (scratch_stride.py's StrideHomogeneityCurriculum,
grad_noise_experiment.py's GradientNoiseController and
inject_gradient_noise), not reimplemented, to avoid introducing drift from
what was actually tested. Confirmed-best settings from each standalone
script are used as defaults here: metric=saturation-time, h_floor=0.0,
deceleration_threshold=0.05 for the homogeneity ratchet; noise_floor=0.0,
initial_noise_scale=0.10 for the gradient-noise controller.
deceleration_threshold for the gradient-noise controller also defaults to
0.05, matching the homogeneity value, but this has NOT been independently
verified for gradient-norm dynamics the way 0.05 was checked for validation
accuracy; treat it as an untested carry-over, not a calibrated choice.

Does not modify the Gradus framework.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing   import Any, Dict, List, Optional

import numpy as np
from torch                     import device as t_device, no_grad, randn_like, Tensor
from torch.nn.functional       import cross_entropy
from torch.optim               import SGD
from torch.optim.lr_scheduler  import CosineAnnealingLR
from torch.utils.data          import BatchSampler, DataLoader, SequentialSampler
from tqdm                      import tqdm

from gradus.artifacts      import DatasetMetrics
from gradus.datasets       import Dataset
from gradus.networks       import Network
from gradus.registration   import DATASET_REGISTRY, NETWORK_REGISTRY, RANK_REGISTRY
from gradus.utilities      import determine_device, get_logger, set_seed


# ==================================================================================================
# Copied verbatim from scratch_stride.py
# ==================================================================================================

def get_labels(dataset: Dataset) -> List[int]:
    """# Extract Per-Sample Class Labels."""
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

        self._sorted_indices_:  List[int] =          RANK_REGISTRY.sort_indices(
                                                          rank_id =    rank,
                                                          dataset_id = dataset_id,
                                                          metric =     self._metric_,
                                                          scores =     scores.scores,
                                                          seed =       seed
                                                      )
        self._n_batches_:       int =                math.ceil(len(self._sorted_indices_) / batch_size)
        self._k_table_, self._h_table_ = self._build_homogeneity_table()

        self._h_target_:        float =              initial_h_target
        self._k_:               int =                self._nearest_k(initial_h_target)
        self._val_acc_history_: List[float] =        []
        self._batches_:          List[List[int]] =    self._build(self._k_)

    def _build_homogeneity_table(self):
        scores_df = self._scores_.scores.loc[self._sorted_indices_]
        raw = scores_df[self._metric_[0]].to_numpy(dtype = float)
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
            batch_var = block.var(axis = 1)
            mean_var = batch_var.mean()
            h = 1.0 - (mean_var / total_var) if total_var > 0 else 0.0
            ks.append(k)
            hs.append(float(h))

        return np.array(ks), np.array(hs)

    def _nearest_k(self, h_target: float) -> int:
        idx = int(np.argmin(np.abs(self._h_table_ - h_target)))
        return int(self._k_table_[idx])

    def _build(self, k: int) -> List[List[int]]:
        indices:      List[int] =       self._sorted_indices_
        n:            int =             len(indices)
        supergroup_sz: int =            self._batch_size_ * k

        batches: List[List[int]] = []
        for start in range(0, n, supergroup_sz):
            supergroup = indices[start:start + supergroup_sz]
            if len(supergroup) < supergroup_sz:
                for i in range(0, len(supergroup), self._batch_size_):
                    batches.append(supergroup[i:i + self._batch_size_])
                continue
            for b in range(k):
                batches.append(supergroup[b::k])

        return batches

    def _batch_class_entropy(self, batch: List[int]) -> float:
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

    def adapt(self, val_accuracy: float) -> Dict[str, Any]:
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

        current_avg = sum(self._val_acc_history_[-self._smooth_window_:]) / self._smooth_window_
        past_window = self._val_acc_history_[-(needed):-self._window_]
        past_avg = sum(past_window) / len(past_window)
        improvement = current_avg - past_avg

        record["improvement"] = improvement
        record["current_avg"] = current_avg
        record["past_avg"] = past_avg

        if self._peak_improvement_ is None or improvement > self._peak_improvement_:
            self._peak_improvement_ = improvement

        record["peak_improvement"] = self._peak_improvement_

        if self._peak_improvement_ <= 0:
            is_plateau = improvement <= 0
        else:
            is_plateau = improvement < (self._decel_thresh_ * self._peak_improvement_)

        record["is_plateau"] = is_plateau

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

    def __iter__(self):
        yield from self._batches_

    def __len__(self) -> int:
        return len(self._batches_)


# ==================================================================================================
# Copied verbatim from grad_noise_experiment.py
# ==================================================================================================

class GradientNoiseController:
    """# Deceleration-Triggered, Monotonically-Ratcheted Gradient Noise Controller."""

    def __init__(self,
        window:                  int = 5,
        smooth_window:           int = 3,
        deceleration_threshold:  float = 0.05,
        step:                    float = 0.02,
        noise_floor:             float = 0.0,
        initial_noise_scale:     float = 0.10,
    ):
        self._window_:          int =              window
        self._smooth_window_:   int =              smooth_window
        self._decel_thresh_:    float =            deceleration_threshold
        self._step_:            float =            step
        self._noise_floor_:     float =            noise_floor
        self._noise_scale_:     float =            initial_noise_scale

        self._grad_norm_history_: List[float] =    []
        self._peak_improvement_:  Optional[float] = None

    @property
    def noise_scale(self) -> float:
        return self._noise_scale_

    def adapt(self, mean_grad_norm: float) -> Dict[str, Any]:
        self._grad_norm_history_.append(mean_grad_norm)

        record: Dict[str, Any] = {
            "noise_scale":    self._noise_scale_,
            "mean_grad_norm": mean_grad_norm,
        }

        needed = self._window_ + self._smooth_window_
        if len(self._grad_norm_history_) <= needed:
            record["next_noise_scale"] = self._noise_scale_
            return record

        current_avg = sum(self._grad_norm_history_[-self._smooth_window_:]) / self._smooth_window_
        past_window = self._grad_norm_history_[-(needed):-self._window_]
        past_avg = sum(past_window) / len(past_window)
        improvement = past_avg - current_avg

        record["improvement"] = improvement
        record["current_avg"] = current_avg
        record["past_avg"] = past_avg

        if self._peak_improvement_ is None or improvement > self._peak_improvement_:
            self._peak_improvement_ = improvement

        record["peak_improvement"] = self._peak_improvement_

        if self._peak_improvement_ <= 0:
            is_plateau = improvement <= 0
        else:
            is_plateau = improvement < (self._decel_thresh_ * self._peak_improvement_)

        record["is_plateau"] = is_plateau

        if not is_plateau:
            record["next_noise_scale"] = self._noise_scale_
            return record

        self._noise_scale_ = max(self._noise_floor_, self._noise_scale_ - self._step_)
        record["next_noise_scale"] = self._noise_scale_
        return record


def inject_gradient_noise(network: Network, noise_scale: float) -> None:
    """# Add Per-Parameter Gaussian Noise to Gradients, Scaled by Each Gradient's Own Std."""
    if noise_scale <= 0:
        return

    for param in network.parameters():
        if param.grad is None:
            continue
        grad_std = param.grad.std()
        if grad_std == 0 or grad_std.isnan():
            continue
        param.grad.add_(randn_like(param.grad) * (noise_scale * grad_std))


# ==================================================================================================
# Combined training loop
# ==================================================================================================

def run(
    network_id:                  str,
    dataset_id:                   str,
    metric:                        str =              "saturation-time",
    rank:                          str =              "ascending",
    batch_size:                    int =              128,
    epochs:                        int =              200,
    seed:                          int =              1,
    # Homogeneity ratchet settings
    curriculum_window:             int =              5,
    curriculum_smooth_window:      int =              3,
    curriculum_step:               float =            0.05,
    curriculum_deceleration_threshold: float =        0.05,
    curriculum_h_floor:            float =            0.0,
    # Gradient-noise controller settings
    noise_window:                  int =              5,
    noise_smooth_window:           int =              3,
    noise_step:                    float =            0.02,
    noise_deceleration_threshold:  float =            0.05,
    noise_floor:                   float =            0.0,
    initial_noise_scale:           float =            0.10,
    device:                        str =              "auto",
    output_path:                   str =              "results/combined_experiment",
    cache_path:                    str =              ".cache",
) -> Dict[str, Any]:
    """# Run Combined Homogeneity-Ratchet + Gradient-Noise Experiment."""
    logger = get_logger("combined-experiment")

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

    curriculum = StrideHomogeneityCurriculum(
        dataset_id =         dataset_id,
        scores =             scores,
        metric =             metric,
        rank =               rank,
        batch_size =         batch_size,
        num_classes =        dataset.num_classes,
        labels =             labels,
        seed =               seed,
        window =             curriculum_window,
        smooth_window =      curriculum_smooth_window,
        step =               curriculum_step,
        deceleration_threshold = curriculum_deceleration_threshold,
        h_floor =            curriculum_h_floor,
    )

    noise_controller = GradientNoiseController(
        window =                  noise_window,
        smooth_window =           noise_smooth_window,
        deceleration_threshold =  noise_deceleration_threshold,
        step =                    noise_step,
        noise_floor =              noise_floor,
        initial_noise_scale =      initial_noise_scale
    )

    train_loader = DataLoader(
        dataset =       dataset.train_data,
        batch_sampler = curriculum,
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

    epoch_records:       List[Dict[str, Any]] = []
    curriculum_log:       List[Dict[str, Any]] = []
    noise_log:             List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):

        network.train()
        train_total_loss, train_correct, train_total = 0.0, 0, 0
        grad_norms_this_epoch: List[float] = []

        for samples, targets in tqdm(
            train_loader,
            desc = f"Epoch {epoch}/{epochs} [k={curriculum._k_}, noise={noise_controller.noise_scale:.3f}]",
            leave = False
        ):
            samples, targets = samples.to(resolved_device), targets.to(resolved_device)

            predictions: Tensor = network(samples)
            loss:        Tensor = cross_entropy(input = predictions, target = targets)

            optimizer.zero_grad()
            loss.backward()

            inject_gradient_noise(network = network, noise_scale = noise_controller.noise_scale)

            grad_norms = [
                p.grad.data.norm(2).item()
                for p in network.parameters()
                if p.grad is not None
            ]
            if grad_norms:
                grad_norms_this_epoch.append(sum(grad_norms) / len(grad_norms))

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
        mean_grad_norm = sum(grad_norms_this_epoch) / len(grad_norms_this_epoch)

        curriculum_record = curriculum.adapt(val_accuracy)
        curriculum_log.append({"epoch": epoch, **curriculum_record})

        noise_record = noise_controller.adapt(mean_grad_norm)
        noise_log.append({"epoch": epoch, **noise_record})

        print(
            f"Epoch {epoch:3}: k={curriculum_record['k']}->{curriculum_record['next_k']}  "
            f"h_target={curriculum_record['h_target']:.3f}  "
            f"noise={noise_record['noise_scale']:.3f}->{noise_record['next_noise_scale']:.3f}  "
            f"grad_norm={mean_grad_norm:.4f}"
        )

        record: Dict[str, Any] = {
            "epoch":          epoch,
            "k":              curriculum_record["k"],
            "h_target":       curriculum_record["h_target"],
            "noise_scale":    noise_record["noise_scale"],
            "mean_grad_norm": mean_grad_norm,
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
            "network_id":  network_id,
            "dataset_id":  dataset_id,
            "metric":      metric,
            "rank":        rank,
            "batch_size":  batch_size,
            "epochs":      epochs,
            "seed":        seed,
            "curriculum_deceleration_threshold": curriculum_deceleration_threshold,
            "curriculum_h_floor":                curriculum_h_floor,
            "noise_deceleration_threshold":       noise_deceleration_threshold,
            "noise_floor":                        noise_floor,
            "initial_noise_scale":                initial_noise_scale,
        },
        "epochs":          epoch_records,
        "curriculum_log":  curriculum_log,
        "noise_log":       noise_log,
    }

    _save(result = result, output_path = output_path, seed = seed)

    return result


def _save(result: Dict[str, Any], output_path: str, seed: int) -> None:
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    tag = f"seed-{seed}"
    with open(out_dir / f"{tag}.json", "w") as f:
        json.dump(result, f, indent = 2)

    # with open(out_dir / f"{tag}.csv", "w", newline = "") as f:
    #     writer = csv.DictWriter(f, fieldnames = list(result["epochs"][0].keys()))
    #     writer.writeheader()
    #     writer.writerows(result["epochs"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Combined homogeneity-ratchet + gradient-noise experiment.")
    parser.add_argument("--network-id",  type = str, default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-100")
    parser.add_argument("--metric",       type = str,   default = "saturation-time")
    parser.add_argument("--rank",         type = str,   default = "ascending")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 200)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument("--curriculum-deceleration-threshold", type = float, default = 0.05)
    parser.add_argument("--curriculum-h-floor",                type = float, default = 0.0)
    parser.add_argument("--noise-deceleration-threshold",      type = float, default = 0.05)
    parser.add_argument("--noise-floor",                       type = float, default = 0.0)
    parser.add_argument("--initial-noise-scale",                type = float, default = 0.10)
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/combined_experiment")
    parser.add_argument("--cache-path",   type = str,   default = ".cache")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run(
        network_id =              args.network_id,
        dataset_id =               args.dataset_id,
        metric =                   args.metric,
        rank =                     args.rank,
        batch_size =               args.batch_size,
        epochs =                   args.epochs,
        seed =                     args.seed,
        curriculum_deceleration_threshold = args.curriculum_deceleration_threshold,
        curriculum_h_floor =                args.curriculum_h_floor,
        noise_deceleration_threshold =      args.noise_deceleration_threshold,
        noise_floor =                        args.noise_floor,
        initial_noise_scale =                args.initial_noise_scale,
        device =                   args.device,
        output_path =              args.output_path,
        cache_path =               args.cache_path,
    )