"""# scratch_sptec.py

First intervention experiment for SP-TEC:
Stochasticity-Preserving Temporal Evidence Curriculum.

Modes
-----
  baseline : ordinary shuffled SGD with mean cross-entropy.
  loss     : SP-TEC-L, using temporal loss state and loss trajectory only.
  full     : full SP-TEC, adding temporal class-conditioned gradient geometry.
  all      : run baseline, loss, and full sequentially with the same seed/noise seed.

Design invariants
-----------------
* The DataLoader always uses shuffle=True.
* Every sample is presented exactly once per epoch (modulo the ordinary final partial batch).
* The synthetic corruption mask and clean/original labels are evaluation-only.
* Epoch t weights use evidence frozen at the end of epoch t-1. There is no current-batch
  loss -> current-batch weight feedback.
* Class-gradient prototypes are estimated from the UNWEIGHTED per-example gradient directions
  produced by the ordinary shuffled stream. They are frozen within an epoch and updated once at
  epoch end, so agreement in epoch t is measured against the prototype available before epoch t.
* Epoch 1 is exactly ordinary shuffled SGD in every mode and initializes the temporal state.

Algorithm 0.1
-------------
For each sample i, maintain

    L_i  : EMA loss
    DL_i : EMA change in L_i
    A_i  : EMA gradient agreement with the observed-class prototype
    DA_i : EMA change in A_i

Within each observed class, robustly standardize these states using median/MAD. The Algorithm 0.1
suspicion evidence is

    E_L = [z(L)]_+ + [z(-DL)]_+
    E_G = R_y * ([z(-A)]_+ + [z(DA)]_+)
    E   = E_L                      (mode=loss)
          E_L + E_G                (mode=full)

where R_y is the norm/reliability of the observed-class gradient prototype.

Candidate influence is

    q_i propto exp(-E_i / tau),   mean_B(q) = 1.

Actual influence interpolates from the shuffled-SGD fixed point:

    w_i(lambda) = 1 + lambda (q_i - 1).

lambda is chosen analytically so Kish effective sample size satisfies

    ESS(w) >= rho * B.

With S_q = sum_i (q_i - 1)^2,

    lambda* = min(1, sqrt(B (1/rho - 1) / S_q))

when S_q > 0. If q is uniform, w=1 exactly.

The implementation logs both requested and actual concentration, lambda, ESS/B, clean/noisy/hard-clean
weights (evaluation-only), noisy-label memorization, and a classifier-layer update-deviation diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch                    import Tensor, device as t_device, no_grad
from torch.nn                 import Linear
from torch.nn.functional      import cross_entropy
from torch.optim              import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data         import DataLoader
from tqdm                     import tqdm

from gradus.datasets      import Dataset
from gradus.networks      import Network
from gradus.registration  import DATASET_REGISTRY, NETWORK_REGISTRY
from gradus.utilities     import determine_device, get_logger, set_seed


# ==================================================================================================
# Dataset wrapper: fixed synthetic label corruption with evaluation-only oracle metadata.
# ==================================================================================================

class IndexedNoisyLabelDataset(torch.utils.data.Dataset):
    """# Fixed Synthetic Label Noise with Stable Sample Indices.

    The corruption construction intentionally mirrors the earlier diagnostic/reweight experiments.
    The training loop receives only (image, OBSERVED label, index). Original labels and the noise
    mask are never used by the SP-TEC state estimator or controller.
    """

    def __init__(self, base_dataset: Any, num_classes: int, noise_rate: float, noise_seed: int):
        self._base_ = base_dataset
        n = len(base_dataset)
        rng = np.random.RandomState(noise_seed)
        noisy_mask = rng.rand(n) < noise_rate

        true_labels = np.array([base_dataset[i][1] for i in range(n)], dtype = np.int64)
        observed_labels = true_labels.copy()

        for i in np.where(noisy_mask)[0]:
            true_label = observed_labels[i]
            wrong = rng.randint(0, num_classes)
            while wrong == true_label:
                wrong = rng.randint(0, num_classes)
            observed_labels[i] = wrong

        self._true_labels_ =     true_labels
        self._labels_ =          observed_labels
        self._noisy_mask_ =      noisy_mask.astype(bool)
        self.actual_noise_rate = float(self._noisy_mask_.mean())

    def __len__(self) -> int:
        return len(self._base_)

    def __getitem__(self, idx: int):
        image, _ = self._base_[idx]
        return image, int(self._labels_[idx]), idx


# ==================================================================================================
# Final-classifier feature capture.
# ==================================================================================================

class FinalLinearFeatureCapture:
    """# Capture the Input to the Network's Last torch.nn.Linear Module."""

    def __init__(self, network: Network):
        linear_modules: List[Tuple[str, Linear]] = [
            (name, module)
            for name, module in network.named_modules()
            if isinstance(module, Linear)
        ]
        if not linear_modules:
            raise RuntimeError(
                "No torch.nn.Linear module found in the network; cannot capture classifier features."
            )

        self.module_name, self.module = linear_modules[-1]
        self.features: Optional[Tensor] = None
        self._handle_ = self.module.register_forward_hook(self._hook)

    def _hook(self, module: Linear, inputs: Tuple[Tensor, ...], output: Tensor) -> None:
        if not inputs:
            raise RuntimeError("Final Linear hook received no input tensor.")
        features = inputs[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim = 1)
        if features.ndim != 2:
            raise RuntimeError(
                f"Expected final Linear input [batch, features], got {tuple(features.shape)}."
            )
        self.features = features.detach()

    def pop(self) -> Tensor:
        if self.features is None:
            raise RuntimeError("Classifier feature hook did not fire.")
        features = self.features
        self.features = None
        return features

    def close(self) -> None:
        self._handle_.remove()


# ==================================================================================================
# Temporal evidence state.
# ==================================================================================================

class TemporalEvidenceTracker:
    """# Maintain Per-Example Temporal State and Slow Class Gradient Prototypes.

    Prototypes are frozen during each epoch. `observe` accumulates an unweighted epoch-level class
    mean; `end_epoch` updates the slow prototype once. This removes minibatch-order dependence from
    the reference frame and makes epoch t agreement strictly use information available before t.
    """

    def __init__(
        self,
        num_samples:     int,
        num_classes:     int,
        prototype_beta:  float = 0.99,
        agreement_beta:  float = 0.90,
        loss_beta:       float = 0.90,
        derivative_beta: float = 0.90,
        eps:             float = 1e-8,
    ):
        for name, value in (
            ("prototype_beta", prototype_beta),
            ("agreement_beta", agreement_beta),
            ("loss_beta", loss_beta),
            ("derivative_beta", derivative_beta),
        ):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0, 1).")

        self._num_samples_ =      num_samples
        self._num_classes_ =      num_classes
        self._prototype_beta_ =   prototype_beta
        self._agreement_beta_ =   agreement_beta
        self._loss_beta_ =        loss_beta
        self._derivative_beta_ =  derivative_beta
        self._eps_ =              eps

        self._prototypes_: Optional[Tensor] = None
        self._prototype_initialized_: Optional[Tensor] = None
        self._epoch_gradient_sums_: Optional[Tensor] = None
        self._epoch_gradient_counts_: Optional[Tensor] = None

        # Latest observations.
        self.latest_loss =                np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_grad_norm =           np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_agreement =           np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_observed_confidence = np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_true_confidence =     np.full(num_samples, np.nan, dtype = np.float32)

        # Persistent state.
        self.loss_ema =              np.full(num_samples, np.nan, dtype = np.float32)
        self.loss_derivative_ema =   np.full(num_samples, np.nan, dtype = np.float32)
        self.agreement_ema =         np.full(num_samples, np.nan, dtype = np.float32)
        self.agreement_derivative_ema = np.full(num_samples, np.nan, dtype = np.float32)

    def _ensure_geometry(self, feature_dim_augmented: int, device: t_device) -> None:
        if self._prototypes_ is not None:
            if self._prototypes_.shape[2] != feature_dim_augmented:
                raise RuntimeError(
                    "Classifier feature dimensionality changed during training: "
                    f"expected {self._prototypes_.shape[2]}, got {feature_dim_augmented}."
                )
            return

        shape = (self._num_classes_, self._num_classes_, feature_dim_augmented)
        self._prototypes_ = torch.zeros(shape, dtype = torch.float32, device = device)
        self._prototype_initialized_ = torch.zeros(
            self._num_classes_, dtype = torch.bool, device = device
        )
        self._epoch_gradient_sums_ = torch.zeros(shape, dtype = torch.float32, device = device)
        self._epoch_gradient_counts_ = torch.zeros(
            self._num_classes_, dtype = torch.long, device = device
        )

    def begin_epoch(self) -> None:
        if self._epoch_gradient_sums_ is not None:
            self._epoch_gradient_sums_.zero_()
        if self._epoch_gradient_counts_ is not None:
            self._epoch_gradient_counts_.zero_()

    @staticmethod
    def _ema_with_derivative(
        state: np.ndarray,
        derivative_state: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        state_beta: float,
        derivative_beta: float,
    ) -> None:
        previous = state[indices]
        valid_value = np.isfinite(values)
        first = valid_value & ~np.isfinite(previous)
        continuing = valid_value & np.isfinite(previous)

        updated = previous.copy()
        updated[first] = values[first]
        updated[continuing] = (
            state_beta * previous[continuing]
            + (1.0 - state_beta) * values[continuing]
        )

        # A trajectory is defined only after two valid state observations.
        delta = np.full(len(indices), np.nan, dtype = np.float32)
        delta[continuing] = updated[continuing] - previous[continuing]

        previous_d = derivative_state[indices]
        first_d = np.isfinite(delta) & ~np.isfinite(previous_d)
        continuing_d = np.isfinite(delta) & np.isfinite(previous_d)
        updated_d = previous_d.copy()
        updated_d[first_d] = delta[first_d]
        updated_d[continuing_d] = (
            derivative_beta * previous_d[continuing_d]
            + (1.0 - derivative_beta) * delta[continuing_d]
        )

        state[indices] = updated
        derivative_state[indices] = updated_d

    def observe(
        self,
        indices:          Tensor,
        logits:           Tensor,
        features:         Tensor,
        observed_targets: Tensor,
        true_targets:     Tensor,
        per_sample_loss:  Tensor,
    ) -> Dict[str, float]:
        """# Observe Current Examples Without Affecting the Optimization Graph."""
        with no_grad():
            probs = logits.detach().softmax(dim = 1)
            observed_targets = observed_targets.detach()
            true_targets = true_targets.detach()
            features = features.detach()

            ones = torch.ones((features.shape[0], 1), device = features.device, dtype = features.dtype)
            h_aug = torch.cat([features, ones], dim = 1)

            one_hot = torch.zeros_like(probs).scatter_(1, observed_targets.unsqueeze(1), 1.0)
            delta = probs - one_hot

            grad_norm = delta.norm(dim = 1)
            delta_hat = delta / grad_norm.clamp_min(self._eps_).unsqueeze(1)
            h_norm = h_aug.norm(dim = 1)
            h_hat = h_aug / h_norm.clamp_min(self._eps_).unsqueeze(1)

            self._ensure_geometry(h_hat.shape[1], features.device)
            assert self._prototypes_ is not None
            assert self._prototype_initialized_ is not None
            assert self._epoch_gradient_sums_ is not None
            assert self._epoch_gradient_counts_ is not None

            # Prototypes are frozen within the epoch, so this is agreement against the previous
            # epoch's reference frame.
            agreement = torch.full(
                (features.shape[0],), float("nan"), dtype = torch.float32, device = features.device
            )
            for c in observed_targets.unique().tolist():
                c = int(c)
                mask = observed_targets == c
                if bool(self._prototype_initialized_[c].item()):
                    prototype = self._prototypes_[c]
                    prototype_norm = prototype.norm()
                    if prototype_norm > self._eps_:
                        numer = ((delta_hat[mask] @ prototype) * h_hat[mask]).sum(dim = 1)
                        agreement[mask] = numer / prototype_norm

                # Accumulate an UNWEIGHTED observed-class epoch mean direction.
                self._epoch_gradient_sums_[c].add_(delta_hat[mask].T @ h_hat[mask])
                self._epoch_gradient_counts_[c] += int(mask.sum().item())

            observed_confidence = probs.gather(1, observed_targets.unsqueeze(1)).squeeze(1)
            true_confidence = probs.gather(1, true_targets.unsqueeze(1)).squeeze(1)

            index_np = indices.detach().cpu().numpy().astype(np.int64)
            loss_np = per_sample_loss.detach().cpu().numpy().astype(np.float32)
            grad_norm_np = grad_norm.detach().cpu().numpy().astype(np.float32)
            agreement_np = agreement.detach().cpu().numpy().astype(np.float32)
            observed_conf_np = observed_confidence.detach().cpu().numpy().astype(np.float32)
            true_conf_np = true_confidence.detach().cpu().numpy().astype(np.float32)

            self.latest_loss[index_np] =                loss_np
            self.latest_grad_norm[index_np] =           grad_norm_np
            self.latest_agreement[index_np] =           agreement_np
            self.latest_observed_confidence[index_np] = observed_conf_np
            self.latest_true_confidence[index_np] =     true_conf_np

            self._ema_with_derivative(
                self.loss_ema,
                self.loss_derivative_ema,
                index_np,
                loss_np,
                self._loss_beta_,
                self._derivative_beta_,
            )
            self._ema_with_derivative(
                self.agreement_ema,
                self.agreement_derivative_ema,
                index_np,
                agreement_np,
                self._agreement_beta_,
                self._derivative_beta_,
            )

            valid = torch.isfinite(agreement)
            return {
                "agreement_valid_fraction": float(valid.float().mean().item()),
                "mean_agreement": float(agreement[valid].mean().item()) if valid.any() else float("nan"),
            }

    def end_epoch(self) -> None:
        """# Update Slow Prototypes Once from the Completed Epoch's Unweighted Class Means."""
        if self._prototypes_ is None:
            return
        assert self._prototype_initialized_ is not None
        assert self._epoch_gradient_sums_ is not None
        assert self._epoch_gradient_counts_ is not None

        with no_grad():
            for c in range(self._num_classes_):
                count = int(self._epoch_gradient_counts_[c].item())
                if count <= 0:
                    continue
                epoch_mean = self._epoch_gradient_sums_[c] / count
                if not bool(self._prototype_initialized_[c].item()):
                    self._prototypes_[c].copy_(epoch_mean)
                    self._prototype_initialized_[c] = True
                else:
                    self._prototypes_[c].mul_(self._prototype_beta_).add_(
                        epoch_mean, alpha = 1.0 - self._prototype_beta_
                    )

    def prototype_reliability(self) -> np.ndarray:
        if self._prototypes_ is None or self._prototype_initialized_ is None:
            return np.zeros(self._num_classes_, dtype = np.float32)
        with no_grad():
            reliability = self._prototypes_.norm(dim = (1, 2)).clamp(max = 1.0)
            reliability = torch.where(
                self._prototype_initialized_, reliability, torch.zeros_like(reliability)
            )
            return reliability.detach().cpu().numpy().astype(np.float32)


# ==================================================================================================
# Robust evidence model.
# ==================================================================================================

def _robust_class_z(
    values: np.ndarray,
    observed_labels: np.ndarray,
    num_classes: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """# Median/MAD Standardization Within Observed Class; Missing/Undefined Values -> Zero Evidence."""
    z = np.zeros(len(values), dtype = np.float32)
    for c in range(num_classes):
        mask = (observed_labels == c) & np.isfinite(values)
        if mask.sum() < 2:
            continue
        subset = values[mask].astype(np.float64)
        median = float(np.median(subset))
        mad = float(np.median(np.abs(subset - median)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= eps:
            continue
        z[mask] = ((subset - median) / scale).astype(np.float32)
    return z


def build_evidence(
    tracker: TemporalEvidenceTracker,
    observed_labels: np.ndarray,
    num_classes: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    """# Build Frozen Next-Epoch Evidence From the Completed Current-Epoch State."""
    if mode not in {"baseline", "loss", "full"}:
        raise ValueError(f"Unsupported evidence mode: {mode}")

    z_loss = _robust_class_z(tracker.loss_ema, observed_labels, num_classes)
    z_neg_dloss = _robust_class_z(-tracker.loss_derivative_ema, observed_labels, num_classes)
    loss_evidence = np.maximum(z_loss, 0.0) + np.maximum(z_neg_dloss, 0.0)

    z_neg_agreement = _robust_class_z(-tracker.agreement_ema, observed_labels, num_classes)
    z_dagreement = _robust_class_z(
        tracker.agreement_derivative_ema, observed_labels, num_classes
    )
    reliability = tracker.prototype_reliability()
    per_sample_reliability = reliability[observed_labels]
    gradient_evidence = per_sample_reliability * (
        np.maximum(z_neg_agreement, 0.0) + np.maximum(z_dagreement, 0.0)
    )

    if mode == "baseline":
        total = np.zeros(len(observed_labels), dtype = np.float32)
    elif mode == "loss":
        total = loss_evidence.astype(np.float32)
    else:
        total = (loss_evidence + gradient_evidence).astype(np.float32)

    return {
        "loss": loss_evidence.astype(np.float32),
        "gradient": gradient_evidence.astype(np.float32),
        "total": total,
    }


# ==================================================================================================
# Analytically constrained curriculum controller.
# ==================================================================================================

class SPTECController:
    """# Convert Frozen Per-Example Evidence into ESS-Constrained Batch Weights."""

    def __init__(self, tau: float = 1.0, rho: float = 0.80, eps: float = 1e-12):
        if tau <= 0.0:
            raise ValueError("tau must be > 0.")
        if not (0.0 < rho <= 1.0):
            raise ValueError("rho must be in (0, 1].")
        self.tau = tau
        self.rho = rho
        self.eps = eps

    def weights(self, evidence: Tensor, intervene: bool) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        b = int(evidence.numel())
        ones = torch.ones_like(evidence, dtype = torch.float32)

        if (not intervene) or b == 0:
            return ones, ones, {
                "lambda": 0.0,
                "requested_ess_fraction": 1.0,
                "actual_ess_fraction": 1.0,
                "constraint_active": 0.0,
            }

        finite_evidence = torch.where(torch.isfinite(evidence), evidence, torch.zeros_like(evidence))
        logits = -finite_evidence / self.tau
        logits = logits - logits.max()
        raw = torch.exp(logits)
        q = raw * (b / raw.sum().clamp_min(self.eps))

        centered = q - 1.0
        s_q = float(centered.square().sum().item())
        if s_q <= self.eps:
            lam = 0.0
            w = ones
        else:
            upper = b * (1.0 / self.rho - 1.0)
            lam = min(1.0, float(np.sqrt(max(0.0, upper / s_q))))
            w = 1.0 + lam * centered

        requested_ess = (float(q.sum().item()) ** 2) / float(q.square().sum().item())
        actual_ess = (float(w.sum().item()) ** 2) / float(w.square().sum().item())
        actual_fraction = actual_ess / b

        # Floating-point tolerance only. Violating this means the controller derivation/implementation diverged.
        if actual_fraction + 1e-6 < self.rho:
            raise RuntimeError(
                f"ESS constraint violated: ESS/B={actual_fraction:.8f} < rho={self.rho:.8f}."
            )

        return w, q, {
            "lambda": lam,
            "requested_ess_fraction": requested_ess / b,
            "actual_ess_fraction": actual_fraction,
            "constraint_active": float(lam < 1.0 - 1e-8),
        }


# ==================================================================================================
# Metrics / summaries.
# ==================================================================================================

def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind = "mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype = np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    valid = np.isfinite(scores)
    labels = labels[valid].astype(np.int8)
    scores = scores[valid].astype(np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return round(float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)), 6)


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray(list(values), dtype = np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def _group_means(values: np.ndarray, noisy_mask: np.ndarray, hard_clean_mask: np.ndarray) -> Dict[str, Optional[float]]:
    groups = {
        "clean": ~noisy_mask,
        "noisy": noisy_mask,
        "hard_clean": hard_clean_mask,
    }
    out: Dict[str, Optional[float]] = {}
    for name, mask in groups.items():
        subset = values[mask & np.isfinite(values)]
        out[name] = round(float(subset.mean()), 6) if len(subset) else None
    return out


def _matched_hard_clean_mask(losses: np.ndarray, noisy_mask: np.ndarray) -> np.ndarray:
    clean_indices = np.where(~noisy_mask & np.isfinite(losses))[0]
    noisy_indices = np.where(noisy_mask & np.isfinite(losses))[0]
    n = min(len(clean_indices), len(noisy_indices))
    hard_mask = np.zeros_like(noisy_mask, dtype = bool)
    if n == 0:
        return hard_mask
    clean_order = clean_indices[np.argsort(losses[clean_indices], kind = "mergesort")]
    hard_mask[clean_order[-n:]] = True
    return hard_mask


def _classifier_update_geometry(
    logits: Tensor,
    features: Tensor,
    targets: Tensor,
    weights: Tensor,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    """# Relative Deviation and Cosine for the Final-Classifier Gradient Only.

    This is a diagnostic of the theorem's update-deviation concept, not the full-network gradient.
    It uses exact batch-mean classifier gradients without materializing per-example CxD tensors.
    """
    with no_grad():
        probs = logits.detach().softmax(dim = 1)
        one_hot = torch.zeros_like(probs).scatter_(1, targets.unsqueeze(1), 1.0)
        delta = probs - one_hot
        ones = torch.ones((features.shape[0], 1), device = features.device, dtype = features.dtype)
        h_aug = torch.cat([features.detach(), ones], dim = 1)
        b = targets.shape[0]

        g_base = (delta.T @ h_aug) / b
        g_weighted = ((delta * weights.unsqueeze(1)).T @ h_aug) / b
        difference = g_weighted - g_base

        base_norm = float(g_base.norm().item())
        weighted_norm = float(g_weighted.norm().item())
        diff_norm = float(difference.norm().item())
        relative = diff_norm / max(base_norm, eps)
        cosine = float(
            (g_base * g_weighted).sum().item()
            / max(base_norm * weighted_norm, eps)
        )
        return relative, cosine


# ==================================================================================================
# Experiment.
# ==================================================================================================

def _parse_epoch_list(value: str) -> List[int]:
    epochs = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if any(epoch <= 0 for epoch in epochs):
        raise argparse.ArgumentTypeError("checkpoint epochs must all be positive integers")
    return epochs


def run(
    network_id:        str,
    dataset_id:        str,
    mode:              str = "full",
    batch_size:        int = 128,
    epochs:            int = 100,
    seed:              int = 1,
    label_noise_rate:  float = 0.30,
    label_noise_seed:  int = 1,
    prototype_beta:    float = 0.99,
    agreement_beta:    float = 0.90,
    loss_beta:         float = 0.90,
    derivative_beta:   float = 0.90,
    tau:               float = 1.0,
    rho:               float = 0.80,
    checkpoint_epochs: Optional[Sequence[int]] = None,
    device:            str = "auto",
    output_path:       str = "results/sptec",
) -> Dict[str, Any]:
    """# Run One SP-TEC Condition."""
    if mode not in {"baseline", "loss", "full"}:
        raise ValueError("mode must be one of: baseline, loss, full")

    logger = get_logger(f"sptec-{mode}")
    set_seed(seed = seed)
    resolved_device: t_device = determine_device(device = device)
    print(f"Device: {resolved_device}")
    print(f"SP-TEC mode: {mode}")

    dataset: Dataset = DATASET_REGISTRY.load_dataset(
        dataset_id = dataset_id,
        seed =        seed,
        epochs =      epochs,
        batch_size =  batch_size,
        metric =      None,
    )
    network: Network = NETWORK_REGISTRY.load_network(
        network_id =  network_id,
        input_shape = dataset.input_shape,
        num_classes = dataset.num_classes,
    ).to(resolved_device)

    train_data = IndexedNoisyLabelDataset(
        base_dataset = dataset.train_data,
        num_classes =  dataset.num_classes,
        noise_rate =   label_noise_rate,
        noise_seed =   label_noise_seed,
    )
    print(
        f"Label noise: requested {label_noise_rate:.2%}, actual {train_data.actual_noise_rate:.2%}; "
        f"noise_seed={label_noise_seed}, training_seed={seed}."
    )

    train_loader = DataLoader(
        dataset =     train_data,
        batch_size =  batch_size,
        shuffle =     True,
        num_workers = dataset._max_workers_,
        pin_memory =  True,
    )
    test_loader = dataset.test_loader

    optimizer = SGD(
        params =       network.parameters(),
        lr =           0.1,
        weight_decay = 5e-4,
        momentum =     0.9,
    )
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    tracker = TemporalEvidenceTracker(
        num_samples =      len(train_data),
        num_classes =      dataset.num_classes,
        prototype_beta =   prototype_beta,
        agreement_beta =   agreement_beta,
        loss_beta =        loss_beta,
        derivative_beta =  derivative_beta,
    )
    controller = SPTECController(tau = tau, rho = rho)
    feature_capture = FinalLinearFeatureCapture(network)

    print(
        f"Classifier='{feature_capture.module_name}', beta_P={prototype_beta}, beta_A={agreement_beta}, "
        f"beta_L={loss_beta}, beta_D={derivative_beta}, tau={tau}, rho={rho}."
    )
    print("Epoch 1 is forced to ordinary shuffled SGD; evidence has a strict one-epoch lag.")

    checkpoint_set = set(checkpoint_epochs or [1, 5, 10, 20, 40, 60, 80, epochs])
    checkpoint_set = {epoch for epoch in checkpoint_set if 1 <= epoch <= epochs}

    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    n_samples = len(train_data)
    current_evidence = np.zeros(n_samples, dtype = np.float32)
    current_loss_evidence = np.zeros(n_samples, dtype = np.float32)
    current_gradient_evidence = np.zeros(n_samples, dtype = np.float32)

    applied_weight = np.ones(n_samples, dtype = np.float32)
    requested_q = np.ones(n_samples, dtype = np.float32)
    evidence_used = np.zeros(n_samples, dtype = np.float32)

    epoch_records: List[Dict[str, Any]] = []
    best_val_accuracy = -float("inf")
    best_val_epoch = 0

    try:
        for epoch in range(1, epochs + 1):
            network.train()
            tracker.begin_epoch()

            train_total_weighted_loss = 0.0
            train_total_unweighted_loss = 0.0
            train_correct = train_true_correct = train_total = 0
            noisy_observed_correct = noisy_true_correct = noisy_total = 0

            lambdas: List[float] = []
            requested_ess: List[float] = []
            actual_ess: List[float] = []
            constraint_flags: List[float] = []
            deviation_ratios: List[float] = []
            update_cosines: List[float] = []
            agreement_valid_fractions: List[float] = []

            intervene = (mode != "baseline") and (epoch > 1)

            for samples, targets, indices in tqdm(
                train_loader,
                desc = f"Epoch {epoch}/{epochs} [SP-TEC:{mode}]",
                leave = False,
            ):
                samples = samples.to(resolved_device)
                targets = targets.to(resolved_device)
                indices = indices.to(resolved_device)
                index_np = indices.detach().cpu().numpy().astype(np.int64)
                true_targets = torch.as_tensor(
                    train_data._true_labels_[index_np], dtype = torch.long, device = resolved_device
                )

                predictions: Tensor = network(samples)
                features = feature_capture.pop()
                per_sample_loss: Tensor = cross_entropy(
                    input = predictions, target = targets, reduction = "none"
                )

                evidence_tensor = torch.as_tensor(
                    current_evidence[index_np], dtype = torch.float32, device = resolved_device
                )
                weights, q, control = controller.weights(evidence_tensor, intervene = intervene)
                weighted_loss: Tensor = (weights * per_sample_loss).mean()
                unweighted_loss: Tensor = per_sample_loss.mean()

                diagnostic = tracker.observe(
                    indices =          indices,
                    logits =           predictions,
                    features =         features,
                    observed_targets = targets,
                    true_targets =     true_targets,
                    per_sample_loss =  per_sample_loss,
                )
                agreement_valid_fractions.append(diagnostic["agreement_valid_fraction"])

                deviation, update_cosine = _classifier_update_geometry(
                    logits = predictions,
                    features = features,
                    targets = targets,
                    weights = weights.detach(),
                )
                deviation_ratios.append(deviation)
                update_cosines.append(update_cosine)

                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

                applied_weight[index_np] = weights.detach().cpu().numpy().astype(np.float32)
                requested_q[index_np] = q.detach().cpu().numpy().astype(np.float32)
                evidence_used[index_np] = current_evidence[index_np]

                lambdas.append(control["lambda"])
                requested_ess.append(control["requested_ess_fraction"])
                actual_ess.append(control["actual_ess_fraction"])
                constraint_flags.append(control["constraint_active"])

                train_total_weighted_loss += weighted_loss.item()
                train_total_unweighted_loss += unweighted_loss.item()
                _, predicted = predictions.max(dim = 1)
                train_correct += predicted.eq(targets).sum().item()
                train_true_correct += predicted.eq(true_targets).sum().item()
                train_total += targets.size(0)

                noisy_batch = targets.ne(true_targets)
                if noisy_batch.any():
                    noisy_total += noisy_batch.sum().item()
                    noisy_observed_correct += predicted[noisy_batch].eq(targets[noisy_batch]).sum().item()
                    noisy_true_correct += predicted[noisy_batch].eq(true_targets[noisy_batch]).sum().item()

            tracker.end_epoch()
            scheduler.step()

            # Build evidence only after the full epoch is complete. It is frozen and used next epoch.
            next_evidence = build_evidence(
                tracker = tracker,
                observed_labels = train_data._labels_,
                num_classes = dataset.num_classes,
                mode = mode,
            )

            network.eval()
            val_total_loss = 0.0
            val_correct = val_total = 0
            with no_grad():
                for samples, targets in test_loader:
                    samples, targets = samples.to(resolved_device), targets.to(resolved_device)
                    predictions = network(samples)
                    feature_capture.pop()  # validation must not enter training state
                    loss = cross_entropy(input = predictions, target = targets)
                    val_total_loss += loss.item()
                    _, predicted = predictions.max(dim = 1)
                    val_correct += predicted.eq(targets).sum().item()
                    val_total += targets.size(0)

            val_accuracy = float(val_correct / val_total)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_val_epoch = epoch

            hard_clean_mask = _matched_hard_clean_mask(tracker.latest_loss, train_data._noisy_mask_)
            noisy_labels = train_data._noisy_mask_.astype(np.int8)

            record: Dict[str, Any] = {
                "epoch":                       epoch,
                "mode":                        mode,
                "intervention_active":         intervene,
                "train_weighted_loss":         round(train_total_weighted_loss / len(train_loader), 6),
                "train_unweighted_loss":       round(train_total_unweighted_loss / len(train_loader), 6),
                "train_accuracy_observed":     round(train_correct / train_total, 6),
                "train_accuracy_true_oracle":  round(train_true_correct / train_total, 6),
                "noisy_accuracy_observed":     round(noisy_observed_correct / noisy_total, 6) if noisy_total else None,
                "noisy_accuracy_true_oracle":  round(noisy_true_correct / noisy_total, 6) if noisy_total else None,
                "val_loss":                    round(val_total_loss / len(test_loader), 6),
                "val_accuracy":                round(val_accuracy, 6),
                "best_val_accuracy_so_far":    round(best_val_accuracy, 6),
                "best_val_epoch_so_far":       best_val_epoch,
                "control": {
                    "lambda":                    _stats(lambdas),
                    "requested_ess_fraction":    _stats(requested_ess),
                    "actual_ess_fraction":       _stats(actual_ess),
                    "constraint_active_fraction": round(float(np.mean(constraint_flags)), 6),
                    "classifier_update_deviation_ratio": _stats(deviation_ratios),
                    "classifier_update_cosine":          _stats(update_cosines),
                },
                "weights": {
                    "applied":      _group_means(applied_weight, train_data._noisy_mask_, hard_clean_mask),
                    "requested_q":  _group_means(requested_q, train_data._noisy_mask_, hard_clean_mask),
                    "evidence_used": _group_means(evidence_used, train_data._noisy_mask_, hard_clean_mask),
                },
                "state_diagnostics": {
                    "agreement_valid_fraction": round(float(np.mean(agreement_valid_fractions)), 6),
                    "prototype_reliability": _stats(tracker.prototype_reliability()),
                    "loss_ema_noisy_auc": _roc_auc(noisy_labels, tracker.loss_ema),
                    "negative_agreement_ema_noisy_auc": _roc_auc(
                        noisy_labels, -tracker.agreement_ema
                    ),
                    "next_total_evidence_noisy_auc": _roc_auc(noisy_labels, next_evidence["total"]),
                    "next_loss_evidence_noisy_auc": _roc_auc(noisy_labels, next_evidence["loss"]),
                    "next_gradient_evidence_noisy_auc": _roc_auc(
                        noisy_labels, next_evidence["gradient"]
                    ),
                },
            }
            epoch_records.append(record)

            print(
                f"Epoch {epoch:3}: val={val_accuracy:.4f} best={best_val_accuracy:.4f}@{best_val_epoch} "
                f"train(obs)={record['train_accuracy_observed']:.4f} "
                f"noisy-mem={record['noisy_accuracy_observed'] or 0.0:.4f} "
                f"lambda={record['control']['lambda']['mean']} "
                f"ESS/B={record['control']['actual_ess_fraction']['mean']} "
                f"w(clean/noisy)={record['weights']['applied']['clean']}/{record['weights']['applied']['noisy']}"
            )

            if epoch in checkpoint_set:
                snapshot_path = out_dir / (
                    f"seed-{seed}_sptec-{mode}_noise{label_noise_rate}_noise-seed{label_noise_seed}_epoch-{epoch}.npz"
                )
                np.savez_compressed(
                    snapshot_path,
                    index =                     np.arange(n_samples, dtype = np.int64),
                    true_label =                train_data._true_labels_,
                    observed_label =            train_data._labels_,
                    noisy_mask =                train_data._noisy_mask_,
                    loss =                      tracker.latest_loss,
                    loss_ema =                  tracker.loss_ema,
                    loss_derivative_ema =       tracker.loss_derivative_ema,
                    grad_norm =                 tracker.latest_grad_norm,
                    agreement =                 tracker.latest_agreement,
                    agreement_ema =             tracker.agreement_ema,
                    agreement_derivative_ema =  tracker.agreement_derivative_ema,
                    observed_confidence =       tracker.latest_observed_confidence,
                    true_confidence_oracle =    tracker.latest_true_confidence,
                    prototype_reliability =     tracker.prototype_reliability(),
                    evidence_used =             evidence_used,
                    applied_weight =            applied_weight,
                    requested_q =               requested_q,
                    next_loss_evidence =        next_evidence["loss"],
                    next_gradient_evidence =    next_evidence["gradient"],
                    next_total_evidence =       next_evidence["total"],
                )
                print(f"Saved SP-TEC snapshot: {snapshot_path}")

            current_loss_evidence = next_evidence["loss"]
            current_gradient_evidence = next_evidence["gradient"]
            current_evidence = next_evidence["total"]

    finally:
        feature_capture.close()

    result: Dict[str, Any] = {
        "config": {
            "algorithm":             "SP-TEC Algorithm 0.1",
            "mode":                  mode,
            "network_id":            network_id,
            "dataset_id":            dataset_id,
            "batch_size":            batch_size,
            "epochs":                epochs,
            "seed":                  seed,
            "label_noise_rate":      label_noise_rate,
            "actual_noise_rate":     train_data.actual_noise_rate,
            "label_noise_seed":      label_noise_seed,
            "prototype_beta":        prototype_beta,
            "agreement_beta":        agreement_beta,
            "loss_beta":             loss_beta,
            "derivative_beta":       derivative_beta,
            "tau":                   tau,
            "rho":                   rho,
            "evidence_lag_epochs":   1,
            "epoch1_intervention":   False,
            "sampling":              "ordinary shuffle=True, full epoch coverage",
            "classifier_module":     feature_capture.module_name,
            "checkpoint_epochs":     sorted(checkpoint_set),
            "oracle_metadata_used_for_training": False,
        },
        "summary": {
            "best_val_accuracy": round(best_val_accuracy, 6),
            "best_val_epoch": best_val_epoch,
            "final_val_accuracy": epoch_records[-1]["val_accuracy"] if epoch_records else None,
            "final_train_accuracy_observed": epoch_records[-1]["train_accuracy_observed"] if epoch_records else None,
            "final_noisy_accuracy_observed": epoch_records[-1]["noisy_accuracy_observed"] if epoch_records else None,
        },
        "epochs": epoch_records,
    }

    result_path = out_dir / (
        f"seed-{seed}_sptec-{mode}-noise{label_noise_rate}-noise-seed{label_noise_seed}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent = 2)
    print(f"Saved: {result_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "SP-TEC Algorithm 0.1 intervention experiment.")
    parser.add_argument("--network-id",        type = str,   default = "resnet-18")
    parser.add_argument("--dataset-id",        type = str,   default = "cifar-100")
    parser.add_argument("--mode",              choices = ["baseline", "loss", "full", "all"], default = "all")
    parser.add_argument("--batch-size",        type = int,   default = 128)
    parser.add_argument("--epochs",            type = int,   default = 100)
    parser.add_argument("--seed",              type = int,   default = 1)
    parser.add_argument("--label-noise-rate",  type = float, default = 0.30)
    parser.add_argument("--label-noise-seed",  type = int,   default = 1)
    parser.add_argument("--prototype-beta",    type = float, default = 0.99)
    parser.add_argument("--agreement-beta",    type = float, default = 0.90)
    parser.add_argument("--loss-beta",         type = float, default = 0.90)
    parser.add_argument("--derivative-beta",   type = float, default = 0.90)
    parser.add_argument("--tau",               type = float, default = 1.0)
    parser.add_argument("--rho",               type = float, default = 0.80)
    parser.add_argument(
        "--checkpoint-epochs",
        type = _parse_epoch_list,
        default = [1, 5, 10, 20, 40, 60, 80, 100],
        help = "Comma-separated per-example snapshot epochs.",
    )
    parser.add_argument("--device",            type = str,   default = "auto")
    parser.add_argument("--output-path",       type = str,   default = "results/sptec")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = ["baseline", "loss", "full"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print("\n" + "=" * 100)
        print(f"Running condition: {mode}")
        print("=" * 100)
        run(
            network_id =        args.network_id,
            dataset_id =        args.dataset_id,
            mode =              mode,
            batch_size =        args.batch_size,
            epochs =            args.epochs,
            seed =              args.seed,
            label_noise_rate =  args.label_noise_rate,
            label_noise_seed =  args.label_noise_seed,
            prototype_beta =    args.prototype_beta,
            agreement_beta =    args.agreement_beta,
            loss_beta =         args.loss_beta,
            derivative_beta =   args.derivative_beta,
            tau =               args.tau,
            rho =               args.rho,
            checkpoint_epochs = args.checkpoint_epochs,
            device =            args.device,
            output_path =       args.output_path,
        )


if __name__ == "__main__":
    main()
