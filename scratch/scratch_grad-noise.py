"""# grad_noise_experiment.py

Isolated experiment: injected gradient noise, annealed by a validated
deceleration trigger, independent of the batch-composition/curriculum
mechanism entirely.

Does not modify the Gradus framework. Uses standard random shuffling (no
curriculum sampler at all), so this tests a genuinely different lever in
isolation before it's combined with the validated homogeneity-ratchet result.

Mechanism
---------
The trigger is validation-accuracy deceleration, the same design validated
for the homogeneity ratchet: a moving average of recent accuracy compared
against the moving average from `window` epochs before that, with a relative
deceleration check (current improvement vs. a fraction, `deceleration_threshold`,
of its own historical peak) rather than a fixed absolute threshold.

An earlier version of this script used mean gradient L2 norm as the trigger,
on the assumption that it would shrink over training the way validation
accuracy improves, fast early, decelerating later. That assumption was wrong
and never checked against real data before being built. A real run showed
gradient norm actually *increasing* for roughly the first 60% of training
(0.09 at epoch 1 rising to a peak near 0.27 by epoch ~61) before falling
sharply after that, the opposite shape from what the trigger assumed. Because
"improvement" was defined as the amount norm had *decreased*, and norm was
mostly *increasing*, improvement was negative for nearly the entire early-to-
mid run, which triggers the plateau condition regardless of what
deceleration_threshold is set to (a sweep from 0.02 to 0.5 produced
byte-identical results because the threshold value never mattered once
improvement was persistently negative). Noise collapsed to floor by epoch 12
in every case. This means the earlier standalone gradient-noise result
(77.47% mean, nearly tied with the 77.53% shuffled baseline) almost certainly
reflected noise being active for only the first ~12 epochs, not the sustained
annealing schedule that was assumed to be running, and should not be trusted
as evidence the annealing design itself works well until re-run with a
trigger that's actually behaving as intended.

The controlled quantity is unchanged: injected gradient noise, added after
backward() and before optimizer.step(), with per-parameter std = noise_scale
* grad.std(), so the injected noise scales naturally with each layer's own
gradient magnitude. noise_scale starts at `initial_noise_scale` and ratchets
down toward `noise_floor` (never back up), reusing the monotonic-ratchet
design validated for the homogeneity mechanism (h_floor=0.15 was shown to
consistently underperform h_floor=0.0 across four separate CIFAR-100 tests,
and hysteresis was shown to underperform a pure one-way ratchet).

Mean gradient norm and mean activation std are still computed and logged
every epoch as diagnostics, useful for understanding what's actually
happening during training, but neither drives the trigger in this version.

Known gaps:
    * The gradient-noise injection assumes network.parameters() and .grad are
      the standard PyTorch interface; no framework-specific assumptions here.
    * `return_activations=True` on the forward pass is assumed to match the
      ResNet family's existing signature (seen in the framework's adaptive-
      scheduling code path); untested against non-ResNet architectures.
    * No curriculum, no batch-composition mechanism. Standard shuffled
      DataLoader throughout, matching the "shuffled baseline" comparison
      point rather than the saturation-time homogeneity-ratchet result. This
      is deliberate, to isolate this lever's own effect before combining it
      with anything curriculum-related.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing   import Any, Dict, List, Optional

from torch                     import device as t_device, no_grad, Tensor
from torch.nn.functional       import cross_entropy
from torch.optim               import SGD
from torch.optim.lr_scheduler  import CosineAnnealingLR
from torch.utils.data          import DataLoader
from tqdm                      import tqdm

from gradus.datasets      import Dataset
from gradus.networks      import Network
from gradus.registration  import DATASET_REGISTRY, NETWORK_REGISTRY
from gradus.utilities     import determine_device, get_logger, set_seed


class GradientNoiseController:
    """# Deceleration-Triggered, Monotonically-Ratcheted Gradient Noise Controller.

    Triggered by validation-accuracy deceleration (see module docstring for
    why gradient-norm deceleration was tried first and abandoned).
    """

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

        self._val_acc_history_:   List[float] =    []
        self._peak_improvement_:  Optional[float] = None

    @property
    def noise_scale(self) -> float:
        """# Current Noise Scale."""
        return self._noise_scale_

    def adapt(self, val_accuracy: float) -> Dict[str, Any]:
        """# Update Noise Scale from Validation-Accuracy Deceleration Detection.

        Called once per epoch, after that epoch's validation accuracy is
        known, so noise_scale for the *next* epoch reflects this epoch's
        result. Same construction as StrideHomogeneityCurriculum.adapt.

        ## Args:
            * val_accuracy (float): This epoch's validation accuracy.

        ## Returns:
            * Dict[str, Any]: logging record.
        """
        self._val_acc_history_.append(val_accuracy)

        record: Dict[str, Any] = {
            "noise_scale":  self._noise_scale_,
            "val_accuracy": val_accuracy,
        }

        needed = self._window_ + self._smooth_window_
        if len(self._val_acc_history_) <= needed:
            record["next_noise_scale"] = self._noise_scale_
            return record

        current_avg = sum(self._val_acc_history_[-self._smooth_window_:]) / self._smooth_window_
        past_window = self._val_acc_history_[-(needed):-self._window_]
        past_avg = sum(past_window) / len(past_window)
        improvement = current_avg - past_avg  # positive = accuracy still improving

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


def inject_gradient_noise(network: Network, noise_scale: float) -> Optional[float]:
    """# Add Per-Parameter Gaussian Noise to Gradients, Scaled by Each Gradient's Own Std.

    Call after loss.backward() and before optimizer.step(). No-op if
    noise_scale <= 0.

    ## Returns:
        * Optional[float]: mean ratio of injected-noise norm to pre-injection
                          gradient norm, averaged across all parameters that
                          received noise. None if noise_scale <= 0 or no
                          parameter had a usable gradient. This exists to
                          verify the injection is actually perturbing
                          gradients by a nontrivial amount, rather than
                          assuming noise_scale alone tells you that; a ratio
                          near zero would mean the injected noise is
                          negligible in practice regardless of what
                          noise_scale is set to.
    """
    if noise_scale <= 0:
        return None

    from torch import randn_like

    ratios: List[float] = []

    for param in network.parameters():
        if param.grad is None:
            continue
        grad_std = param.grad.std()
        if grad_std == 0 or grad_std.isnan():
            continue

        original_norm = param.grad.data.norm(2).item()
        noise = randn_like(param.grad) * (noise_scale * grad_std)
        param.grad.add_(noise)

        if original_norm > 0:
            ratios.append(noise.norm(2).item() / original_norm)

    return (sum(ratios) / len(ratios)) if ratios else None


def run(
    network_id:              str,
    dataset_id:               str,
    batch_size:                int =              128,
    epochs:                    int =              100,
    seed:                      int =              1,
    window:                    int =              5,
    smooth_window:             int =              3,
    deceleration_threshold:    float =            0.05,
    step:                      float =            0.02,
    noise_floor:               float =            0.0,
    initial_noise_scale:       float =            0.10,
    device:                    str =              "auto",
    output_path:               str =              "results/grad_noise_experiment",
    cache_path:                str =              ".cache",
) -> Dict[str, Any]:
    """# Run Isolated Gradient-Noise Experiment."""
    logger = get_logger("grad-noise-experiment")

    set_seed(seed = seed)
    resolved_device: t_device = determine_device(device = device)
    print(f"Device: {resolved_device}")

    dataset: Dataset = DATASET_REGISTRY.load_dataset(
        dataset_id = dataset_id,
        seed =        seed,
        epochs =      epochs,
        batch_size =  batch_size,
        shuffle =     True,
        metric =      None
    )

    network: Network = NETWORK_REGISTRY.load_network(
        network_id =  network_id,
        input_shape = dataset.input_shape,
        num_classes = dataset.num_classes
    ).to(resolved_device)

    train_loader: DataLoader = dataset.train_loader
    test_loader:  DataLoader = dataset.test_loader

    optimizer = SGD(
        params =        network.parameters(),
        lr =            0.1,
        weight_decay =  5e-4,
        momentum =      0.9
    )
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    controller = GradientNoiseController(
        window =                  window,
        smooth_window =           smooth_window,
        deceleration_threshold =  deceleration_threshold,
        step =                    step,
        noise_floor =              noise_floor,
        initial_noise_scale =      initial_noise_scale
    )

    epoch_records:   List[Dict[str, Any]] = []
    control_log:      List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):

        network.train()
        train_total_loss, train_correct, train_total = 0.0, 0, 0
        grad_norms_this_epoch:      List[float] = []
        activation_stds_this_epoch: List[float] = []
        noise_ratios_this_epoch:    List[float] = []

        for samples, targets in tqdm(
            train_loader, desc = f"Epoch {epoch}/{epochs} [train, noise={controller.noise_scale:.3f}]",
            leave = False
        ):
            samples, targets = samples.to(resolved_device), targets.to(resolved_device)

            predictions, activations = network(samples, return_activations = True)
            loss:        Tensor = cross_entropy(input = predictions, target = targets)

            optimizer.zero_grad()
            loss.backward()

            noise_ratio = inject_gradient_noise(network = network, noise_scale = controller.noise_scale)
            if noise_ratio is not None:
                noise_ratios_this_epoch.append(noise_ratio)

            # Record gradient norm and activation std for this batch, after
            # noise injection, since that's what actually drives the update.
            grad_norms = [
                p.grad.data.norm(2).item()
                for p in network.parameters()
                if p.grad is not None
            ]
            if grad_norms:
                grad_norms_this_epoch.append(sum(grad_norms) / len(grad_norms))

            activation_stds_this_epoch.append(
                sum(act.std().item() for act in activations) / len(activations)
            )

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

        mean_grad_norm = sum(grad_norms_this_epoch) / len(grad_norms_this_epoch)
        mean_activation_std = sum(activation_stds_this_epoch) / len(activation_stds_this_epoch)
        mean_noise_ratio = (
            sum(noise_ratios_this_epoch) / len(noise_ratios_this_epoch)
            if noise_ratios_this_epoch else 0.0
        )
        val_accuracy = round(val_correct / val_total, 4)

        control_record = controller.adapt(val_accuracy)
        control_log.append({
            "epoch": epoch,
            "mean_grad_norm": mean_grad_norm,
            "mean_activation_std": mean_activation_std,
            "mean_noise_ratio": mean_noise_ratio,
            **control_record
        })
        print(
            f"Epoch {epoch:3}: noise_scale={control_record['noise_scale']:.3f} -> "
            f"{control_record['next_noise_scale']:.3f}  "
            f"mean_grad_norm={mean_grad_norm:.4f}  mean_activation_std={mean_activation_std:.4f}  "
            f"mean_noise_ratio={mean_noise_ratio:.4f}"
        )

        record: Dict[str, Any] = {
            "epoch":              epoch,
            "noise_scale":        control_record["noise_scale"],
            "mean_grad_norm":     mean_grad_norm,
            "mean_activation_std": mean_activation_std,
            "mean_noise_ratio":   mean_noise_ratio,
            "train_loss":         round(train_total_loss / len(train_loader), 4),
            "train_accuracy":     round(train_correct / train_total, 4),
            "val_loss":           round(val_total_loss / len(test_loader), 4),
            "val_accuracy":       val_accuracy,
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
            "network_id":              network_id,
            "dataset_id":              dataset_id,
            "batch_size":              batch_size,
            "epochs":                  epochs,
            "seed":                    seed,
            "window":                  window,
            "smooth_window":           smooth_window,
            "deceleration_threshold":  deceleration_threshold,
            "step":                    step,
            "noise_floor":             noise_floor,
            "initial_noise_scale":     initial_noise_scale,
        },
        "epochs":       epoch_records,
        "control_log":  control_log,
    }

    _save(result = result, output_path = f"{output_path}", seed = seed, ins = initial_noise_scale, dt = deceleration_threshold)

    return result


def _save(result: Dict[str, Any], output_path: str, seed: int, ins: float, dt: float) -> None:
    """# Write Results to Disk (plain JSON/CSV, not TrainingRecord)."""
    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    tag = f"seed-{seed}"
    with open(out_dir / f"{tag}_ins-{ins}_dt-{dt}.json", "w") as f:
        json.dump(result, f, indent = 2)

    # with open(out_dir / f"{tag}.csv", "w", newline = "") as f:
    #     writer = csv.DictWriter(f, fieldnames = list(result["epochs"][0].keys()))
    #     writer.writeheader()
    #     writer.writerows(result["epochs"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Gradient/activation-driven noise-annealing experiment.")
    parser.add_argument("--network-id",  type = str, default = "resnet-18")
    parser.add_argument("--dataset-id",   type = str,   default = "cifar-10")
    parser.add_argument("--batch-size",   type = int,   default = 128)
    parser.add_argument("--epochs",       type = int,   default = 100)
    parser.add_argument("--seed",         type = int,   default = 1)
    parser.add_argument("--window",       type = int,   default = 5)
    parser.add_argument("--smooth-window", type = int,  default = 3)
    parser.add_argument(
        "--deceleration-threshold", type = float, default = 0.05,
        help = "Trigger a ratchet step once the validation-accuracy improvement rate falls below this "
               "fraction of its own historical peak. Same signal and same calibrated value (0.05) as "
               "the homogeneity mechanism, not gradient norm, which was tried first and abandoned "
               "after a real run showed it rises for most of training rather than shrinking, see the "
               "module docstring."
    )
    parser.add_argument(
        "--step", type = float, default = 0.02,
        help = "Amount to decrease noise_scale by per ratchet step."
    )
    parser.add_argument(
        "--noise-floor", type = float, default = 0.0,
        help = "Minimum noise_scale the controller can reach."
    )
    parser.add_argument(
        "--initial-noise-scale", type = float, default = 0.10,
        help = "Starting noise_scale (fraction of each gradient's own std added as noise)."
    )
    parser.add_argument("--device",       type = str,   default = "auto")
    parser.add_argument("--output-path",  type = str,   default = "results/grad_noise_experiment")
    parser.add_argument("--cache-path",   type = str,   default = ".cache")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    run(
        network_id =              args.network_id,
        dataset_id =               args.dataset_id,
        batch_size =               args.batch_size,
        epochs =                   args.epochs,
        seed =                     args.seed,
        window =                   args.window,
        smooth_window =            args.smooth_window,
        deceleration_threshold =   args.deceleration_threshold,
        step =                     args.step,
        noise_floor =              args.noise_floor,
        initial_noise_scale =      args.initial_noise_scale,
        device =                   args.device,
        output_path =              args.output_path,
        cache_path =               args.cache_path,
    )