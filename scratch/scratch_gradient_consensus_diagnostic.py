"""# scratch_gradient_consensus_diagnostic.py

Non-interventional diagnostic for the proposed consensus-gated,
stochasticity-preserving curriculum direction.

Purpose
-------
The prior experiment families manipulate batch composition, sampling
probability, or per-sample loss weight. This script does NONE of those.
Training is exactly ordinary shuffled SGD with mean cross-entropy. Its only
purpose is to test the key identifiability assumption needed before building
an intervention:

    Does temporal classifier-gradient direction distinguish persistently
    useful/hard-clean examples from corrupted-label examples better than
    scalar difficulty signals such as loss or gradient norm?

The experiment uses the same synthetic label-noise construction as
scratch_reweight.py (fixed noise_seed independent of training seed), but
retains the corruption mask, original labels, and sample indices for
evaluation-only diagnostics. The corruption metadata NEVER enters the loss,
optimizer, sampling rule, or class-consensus estimator.

Gradient geometry
-----------------
For a final linear classifier with weights W and bias b, per-example
cross-entropy gives

    delta_i = softmax(z_i) - onehot(y_i)
    dL_i/dW = delta_i h_i^T
    dL_i/db = delta_i

Augmenting h_i with a constant 1 folds the bias term into the same outer
product. After independently normalizing delta_i and augmented h_i, the
resulting outer product is a unit-Frobenius-norm classifier-gradient
direction.

For each OBSERVED class c, maintain an unweighted EMA prototype M_c of these
normalized gradient directions. An example's online agreement is the cosine
between its current classifier-gradient direction and the PRE-UPDATE
prototype of its observed class. The prototype is updated only AFTER that
agreement is measured, preventing the current example from trivially
agreeing with a reference that already contains itself.

No [B, C, D] per-example gradient tensor is materialized. For normalized
outer-product gradients g_i = delta_i h_i^T,

    <g_i, M_c>_F = delta_i^T M_c h_i,

so agreement is computed directly from delta_i, h_i, and the C x D class
prototype.

Temporal state
--------------
Each sample keeps an EMA of directional agreement and an EMA of loss. The
per-sample EMA updates once whenever that sample is encountered; under the
plain shuffled, one-pass-per-epoch loader this is once per epoch.

Evaluation
----------
Every epoch, the known synthetic corruption mask is used ONLY after training
to evaluate how well the following online signals identify noisy labels:

  * instantaneous loss
  * temporal EMA loss
  * scalar output-gradient norm ||delta_i||_2 (the signal used by prior
    local-pool/temperature/hybrid diagnostics)
  * negative instantaneous directional agreement
  * negative temporal directional agreement
  * negative confidence in the observed label

The script reports AUROC and tie-safe average precision (stepwise PR-AUC)
for (a) all clean vs. noisy examples and (b) a deliberately difficult
matched comparison: every noisy example versus the N highest-loss clean
examples, where N is matched to the smaller group size.

Selected epochs also save compressed per-example .npz snapshots so the
clean/noisy/hard-clean distributions can be plotted later without rerunning
training.

This script does not modify the Gradus framework and does not implement the
proposed curriculum controller. A positive result is a go/no-go condition
for implementing that controller; a negative result should kill the idea
before a larger intervention sweep is launched.
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
# Dataset wrapper: identical corruption logic to scratch_reweight.py, but retains evaluation metadata.
# ==================================================================================================

class IndexedNoisyLabelDataset(torch.utils.data.Dataset):
    """# Training Dataset with Fixed Synthetic Label Noise and Stable Sample Indices.

    The corruption process intentionally mirrors scratch_reweight.py so a
    given noise_rate/noise_seed pair produces the same noisy-label pattern.
    true labels and noisy_mask are retained strictly for post-hoc diagnostic
    evaluation; __getitem__ exposes only image, OBSERVED target, and index to
    the training loop.
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
    """# Capture the Input to the Network's Last torch.nn.Linear Module.

    The final Linear module's input is the penultimate representation used by
    the classifier. This avoids depending on a model-specific attribute name
    (fc, classifier, head, etc.) and does not require modifying Gradus.
    """

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
                f"Expected final Linear input to be rank-2 [batch, features], got shape {tuple(features.shape)}."
            )
        self.features = features.detach()

    def pop(self) -> Tensor:
        if self.features is None:
            raise RuntimeError(
                "Classifier feature hook did not fire. The selected last Linear module may not participate "
                "in the network's forward path."
            )
        features = self.features
        self.features = None
        return features

    def close(self) -> None:
        self._handle_.remove()


# ==================================================================================================
# Online gradient-consensus tracker. Diagnostic only: never changes the training loss or optimizer.
# ==================================================================================================

class GradientConsensusTracker:
    """# Track Class-Conditioned Gradient-Direction Prototypes and Per-Sample Temporal State."""

    def __init__(
        self,
        num_samples:      int,
        num_classes:      int,
        prototype_beta:   float = 0.99,
        agreement_beta:   float = 0.90,
        loss_beta:        float = 0.90,
        eps:              float = 1e-8,
    ):
        if not (0.0 <= prototype_beta < 1.0):
            raise ValueError("prototype_beta must be in [0, 1).")
        if not (0.0 <= agreement_beta < 1.0):
            raise ValueError("agreement_beta must be in [0, 1).")
        if not (0.0 <= loss_beta < 1.0):
            raise ValueError("loss_beta must be in [0, 1).")

        self._num_samples_ =      num_samples
        self._num_classes_ =      num_classes
        self._prototype_beta_ =   prototype_beta
        self._agreement_beta_ =   agreement_beta
        self._loss_beta_ =        loss_beta
        self._eps_ =              eps

        self._prototypes_: Optional[Tensor] = None
        self._prototype_initialized_: Optional[Tensor] = None
        self._device_: Optional[t_device] = None

        # Latest value from this epoch (all samples are seen once per epoch under shuffle=True).
        self.latest_loss =                np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_grad_norm =           np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_agreement =           np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_observed_confidence = np.full(num_samples, np.nan, dtype = np.float32)
        self.latest_true_confidence =     np.full(num_samples, np.nan, dtype = np.float32)

        # Persistent per-example temporal state.
        self.loss_ema =                   np.full(num_samples, np.nan, dtype = np.float32)
        self.agreement_ema =              np.full(num_samples, np.nan, dtype = np.float32)

    def _ensure_prototypes(self, feature_dim_augmented: int, device: t_device) -> None:
        if self._prototypes_ is not None:
            if self._prototypes_.shape[2] != feature_dim_augmented:
                raise RuntimeError(
                    "Classifier feature dimensionality changed during training: "
                    f"expected {self._prototypes_.shape[2]}, got {feature_dim_augmented}."
                )
            return

        self._device_ = device
        self._prototypes_ = torch.zeros(
            (self._num_classes_, self._num_classes_, feature_dim_augmented),
            dtype = torch.float32,
            device = device,
        )
        self._prototype_initialized_ = torch.zeros(
            self._num_classes_, dtype = torch.bool, device = device
        )

    @staticmethod
    def _update_numpy_ema(state: np.ndarray, indices: np.ndarray, values: np.ndarray, beta: float) -> None:
        previous = state[indices]
        valid_value = np.isfinite(values)
        first = valid_value & ~np.isfinite(previous)
        continuing = valid_value & np.isfinite(previous)

        updated = previous.copy()
        updated[first] = values[first]
        updated[continuing] = beta * previous[continuing] + (1.0 - beta) * values[continuing]
        state[indices] = updated

    def observe(
        self,
        indices:          Tensor,
        logits:           Tensor,
        features:         Tensor,
        observed_targets: Tensor,
        true_targets:     Tensor,
        per_sample_loss:  Tensor,
    ) -> Dict[str, float]:
        """# Measure Agreement Against PRE-UPDATE Prototypes, Then Update the Prototypes.

        All calculations are detached from autograd. Nothing returned by this
        method participates in backward() or optimizer.step().
        """
        with no_grad():
            probs = logits.detach().softmax(dim = 1)
            observed_targets = observed_targets.detach()
            true_targets = true_targets.detach()
            features = features.detach()

            # Include the final Linear bias gradient by augmenting h with 1.
            ones = torch.ones((features.shape[0], 1), device = features.device, dtype = features.dtype)
            h_aug = torch.cat([features, ones], dim = 1)

            one_hot = torch.zeros_like(probs).scatter_(1, observed_targets.unsqueeze(1), 1.0)
            delta = probs - one_hot

            grad_norm = delta.norm(dim = 1)
            delta_hat = delta / grad_norm.clamp_min(self._eps_).unsqueeze(1)
            h_norm = h_aug.norm(dim = 1)
            h_hat = h_aug / h_norm.clamp_min(self._eps_).unsqueeze(1)

            self._ensure_prototypes(h_hat.shape[1], features.device)
            assert self._prototypes_ is not None
            assert self._prototype_initialized_ is not None

            # Agreement with the OBSERVED class's prototype before this batch
            # contributes to that prototype. This is the candidate online signal.
            agreement = torch.full(
                (features.shape[0],), float("nan"), dtype = torch.float32, device = features.device
            )

            for c in observed_targets.unique().tolist():
                c = int(c)
                mask = observed_targets == c
                if not bool(self._prototype_initialized_[c].item()):
                    continue

                prototype = self._prototypes_[c]
                prototype_norm = prototype.norm()
                if prototype_norm <= self._eps_:
                    continue

                # delta_hat @ M_c -> [n_c, D]; row-wise dot with h_hat -> [n_c].
                numer = ((delta_hat[mask] @ prototype) * h_hat[mask]).sum(dim = 1)
                agreement[mask] = numer / prototype_norm

            observed_confidence = probs.gather(1, observed_targets.unsqueeze(1)).squeeze(1)
            true_confidence = probs.gather(1, true_targets.unsqueeze(1)).squeeze(1)

            # Copy per-example diagnostics to CPU state.
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

            self._update_numpy_ema(self.loss_ema, index_np, loss_np, self._loss_beta_)
            self._update_numpy_ema(
                self.agreement_ema, index_np, agreement_np, self._agreement_beta_
            )

            # Only now update the observed-class prototypes with the unweighted
            # normalized directions from this ordinary shuffled batch.
            for c in observed_targets.unique().tolist():
                c = int(c)
                mask = observed_targets == c
                batch_prototype = (delta_hat[mask].T @ h_hat[mask]) / mask.sum()

                if not bool(self._prototype_initialized_[c].item()):
                    self._prototypes_[c].copy_(batch_prototype)
                    self._prototype_initialized_[c] = True
                else:
                    self._prototypes_[c].mul_(self._prototype_beta_).add_(
                        batch_prototype, alpha = 1.0 - self._prototype_beta_
                    )

            valid = torch.isfinite(agreement)
            return {
                "agreement_valid_fraction": float(valid.float().mean().item()),
                "mean_agreement": float(agreement[valid].mean().item()) if valid.any() else float("nan"),
            }

    def prototype_reliability(self) -> np.ndarray:
        """# Frobenius Norm of Each Class Prototype; 0 for Classes Not Yet Initialized."""
        if self._prototypes_ is None or self._prototype_initialized_ is None:
            return np.zeros(self._num_classes_, dtype = np.float32)

        with no_grad():
            reliability = self._prototypes_.norm(dim = (1, 2))
            reliability = torch.where(
                self._prototype_initialized_, reliability, torch.zeros_like(reliability)
            )
            return reliability.detach().cpu().numpy().astype(np.float32)


# ==================================================================================================
# Metrics: no sklearn dependency required.
# ==================================================================================================

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """# 1-Based Average Ranks with Exact Tie Handling."""
    order = np.argsort(values, kind = "mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype = np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        # Positions i..j-1 correspond to 1-based ranks i+1..j.
        average_rank = ((i + 1) + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j

    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """# Mann-Whitney AUROC, Positive Class = 1."""
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = _average_ranks(scores)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """# Tie-Safe Stepwise Area Under the Precision-Recall Curve.

    Scores are grouped at equal thresholds before precision/recall is updated,
    so the result is invariant to arbitrary ordering inside a tie group.
    """
    labels = labels.astype(np.int8)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind = "mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp, fp = 0, 0
    previous_recall = 0.0
    ap = 0.0
    i = 0

    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1

        group = sorted_labels[i:j]
        tp += int(group.sum())
        fp += int(len(group) - group.sum())
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
        i = j

    return float(ap)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Optional[float]]:
    """# Evaluate a Suspicion Score Where Higher Means 'More Likely Noisy'."""
    valid = np.isfinite(scores)
    labels = labels[valid].astype(np.int8)
    scores = scores[valid].astype(np.float64)

    if len(labels) == 0 or labels.sum() == 0 or labels.sum() == len(labels):
        return {
            "auroc": None,
            "auprc": None,
            "coverage": float(valid.mean()),
            "n": int(len(labels)),
        }

    return {
        "auroc": round(_roc_auc(labels, scores), 6),
        "auprc": round(_average_precision(labels, scores), 6),
        "coverage": round(float(valid.mean()), 6),
        "n": int(len(labels)),
    }


def _group_summary(values: np.ndarray, noisy_mask: np.ndarray) -> Dict[str, Optional[float]]:
    """# Raw Signal Means/SDs for Interpretation; Does Not Flip Signal Direction."""
    out: Dict[str, Optional[float]] = {}
    for name, mask in (("clean", ~noisy_mask), ("noisy", noisy_mask)):
        subset = values[mask & np.isfinite(values)]
        if len(subset) == 0:
            out[f"{name}_mean"] = None
            out[f"{name}_std"] = None
        else:
            out[f"{name}_mean"] = round(float(subset.mean()), 6)
            out[f"{name}_std"] = round(float(subset.std()), 6)
    return out


def _matched_hard_clean_mask(losses: np.ndarray, noisy_mask: np.ndarray) -> np.ndarray:
    """# Select the Highest-Loss Clean Examples, Count-Matched to Noisy Examples When Possible."""
    clean_indices = np.where(~noisy_mask & np.isfinite(losses))[0]
    noisy_indices = np.where(noisy_mask & np.isfinite(losses))[0]
    n = min(len(clean_indices), len(noisy_indices))

    hard_mask = np.zeros_like(noisy_mask, dtype = bool)
    if n == 0:
        return hard_mask

    clean_order = clean_indices[np.argsort(losses[clean_indices], kind = "mergesort")]
    hard_mask[clean_order[-n:]] = True
    return hard_mask


def _diagnostic_metrics(tracker: GradientConsensusTracker, noisy_mask: np.ndarray) -> Dict[str, Any]:
    """# All-Clean-vs-Noisy and Matched-Hard-Clean-vs-Noisy Diagnostics."""
    suspicion_scores: Dict[str, np.ndarray] = {
        "loss":                         tracker.latest_loss,
        "loss_ema":                     tracker.loss_ema,
        "grad_norm":                    tracker.latest_grad_norm,
        "negative_agreement":           -tracker.latest_agreement,
        "negative_agreement_ema":       -tracker.agreement_ema,
        "negative_observed_confidence": -tracker.latest_observed_confidence,
    }

    labels_all = noisy_mask.astype(np.int8)
    all_metrics = {
        name: _binary_metrics(labels_all, score)
        for name, score in suspicion_scores.items()
    }

    hard_clean_mask = _matched_hard_clean_mask(tracker.latest_loss, noisy_mask)
    hard_compare_mask = noisy_mask | hard_clean_mask
    hard_labels = noisy_mask[hard_compare_mask].astype(np.int8)
    hard_metrics = {
        name: _binary_metrics(hard_labels, score[hard_compare_mask])
        for name, score in suspicion_scores.items()
    }

    raw_group_summaries = {
        "loss":                 _group_summary(tracker.latest_loss, noisy_mask),
        "loss_ema":             _group_summary(tracker.loss_ema, noisy_mask),
        "grad_norm":            _group_summary(tracker.latest_grad_norm, noisy_mask),
        "agreement":            _group_summary(tracker.latest_agreement, noisy_mask),
        "agreement_ema":        _group_summary(tracker.agreement_ema, noisy_mask),
        "observed_confidence":  _group_summary(tracker.latest_observed_confidence, noisy_mask),
        "true_confidence_oracle": _group_summary(tracker.latest_true_confidence, noisy_mask),
    }

    return {
        "all_clean_vs_noisy": all_metrics,
        "matched_hard_clean_vs_noisy": {
            "hard_clean_count": int(hard_clean_mask.sum()),
            "noisy_count": int(noisy_mask.sum()),
            "metrics": hard_metrics,
        },
        "raw_group_summaries": raw_group_summaries,
    }


# ==================================================================================================
# Experiment.
# ==================================================================================================

def _parse_epoch_list(value: str) -> List[int]:
    epochs = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if any(epoch <= 0 for epoch in epochs):
        raise argparse.ArgumentTypeError("checkpoint epochs must all be positive integers")
    return epochs


def run(
    network_id:             str,
    dataset_id:             str,
    batch_size:             int =              128,
    epochs:                 int =              100,
    seed:                   int =              1,
    label_noise_rate:       float =            0.30,
    label_noise_seed:       int =              1,
    prototype_beta:         float =            0.99,
    agreement_beta:         float =            0.90,
    loss_beta:              float =            0.90,
    checkpoint_epochs:      Optional[Sequence[int]] = None,
    device:                 str =              "auto",
    output_path:            str =              "results/gradient_consensus_diagnostic",
) -> Dict[str, Any]:
    """# Run the Non-Interventional Gradient-Consensus Diagnostic."""
    logger = get_logger("gradient-consensus-diagnostic")

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

    train_data = IndexedNoisyLabelDataset(
        base_dataset = dataset.train_data,
        num_classes =  dataset.num_classes,
        noise_rate =   label_noise_rate,
        noise_seed =   label_noise_seed,
    )
    print(
        f"Label noise injected: requested {label_noise_rate:.2%}, actual {train_data.actual_noise_rate:.2%} "
        f"(noise_seed={label_noise_seed}, fixed independent of training seed={seed}). Test set unchanged."
    )

    # IMPORTANT: ordinary shuffled SGD. This diagnostic NEVER changes sample
    # order, coverage, selection probability, or loss contribution.
    train_loader = DataLoader(
        dataset =      train_data,
        batch_size =   batch_size,
        shuffle =      True,
        num_workers =  dataset._max_workers_,
        pin_memory =   True,
    )
    test_loader = dataset.test_loader

    optimizer = SGD(
        params =        network.parameters(),
        lr =            0.1,
        weight_decay =  5e-4,
        momentum =      0.9,
    )
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = epochs)

    tracker = GradientConsensusTracker(
        num_samples =     len(train_data),
        num_classes =     dataset.num_classes,
        prototype_beta =  prototype_beta,
        agreement_beta =  agreement_beta,
        loss_beta =       loss_beta,
    )
    feature_capture = FinalLinearFeatureCapture(network)
    print(
        f"Classifier feature hook: last Linear module '{feature_capture.module_name}'. "
        f"prototype_beta={prototype_beta}, agreement_beta={agreement_beta}, loss_beta={loss_beta}."
    )

    checkpoint_set = set(checkpoint_epochs or [1, 5, 10, 20, 40, 60, 80, epochs])
    checkpoint_set = {epoch for epoch in checkpoint_set if 1 <= epoch <= epochs}

    out_dir = Path(output_path)
    out_dir.mkdir(parents = True, exist_ok = True)

    epoch_records: List[Dict[str, Any]] = []

    try:
        for epoch in range(1, epochs + 1):
            network.train()
            train_total_loss, train_correct, train_true_correct, train_total = 0.0, 0, 0, 0
            noisy_observed_correct, noisy_true_correct, noisy_total = 0, 0, 0
            agreement_valid_fractions: List[float] = []

            for samples, targets, indices in tqdm(
                train_loader,
                desc = f"Epoch {epoch}/{epochs} [shuffled, diagnostic-only]",
                leave = False,
            ):
                samples = samples.to(resolved_device)
                targets = targets.to(resolved_device)
                indices = indices.to(resolved_device)
                true_targets = torch.as_tensor(
                    train_data._true_labels_[indices.detach().cpu().numpy()],
                    dtype = torch.long,
                    device = resolved_device,
                )

                predictions: Tensor = network(samples)
                features = feature_capture.pop()
                per_sample_loss: Tensor = cross_entropy(
                    input = predictions, target = targets, reduction = "none"
                )
                loss: Tensor = per_sample_loss.mean()

                # Observe BEFORE optimizer.step(), but entirely detached from
                # autograd and from the training objective.
                diagnostic = tracker.observe(
                    indices =          indices,
                    logits =           predictions,
                    features =         features,
                    observed_targets = targets,
                    true_targets =     true_targets,
                    per_sample_loss =  per_sample_loss,
                )
                agreement_valid_fractions.append(diagnostic["agreement_valid_fraction"])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()
                _, predicted = predictions.max(dim = 1)
                train_correct += predicted.eq(targets).sum().item()
                train_true_correct += predicted.eq(true_targets).sum().item()
                train_total += targets.size(0)

                noisy_batch = targets.ne(true_targets)
                if noisy_batch.any():
                    noisy_total += noisy_batch.sum().item()
                    noisy_observed_correct += (
                        predicted[noisy_batch].eq(targets[noisy_batch]).sum().item()
                    )
                    noisy_true_correct += (
                        predicted[noisy_batch].eq(true_targets[noisy_batch]).sum().item()
                    )

            scheduler.step()

            network.eval()
            val_total_loss, val_correct, val_total = 0.0, 0, 0
            with no_grad():
                for samples, targets in test_loader:
                    samples, targets = samples.to(resolved_device), targets.to(resolved_device)
                    predictions = network(samples)
                    # Validation forward also triggers the hook; discard its
                    # captured features because validation is not part of the
                    # training-sample consensus state.
                    feature_capture.pop()
                    loss = cross_entropy(input = predictions, target = targets)
                    val_total_loss += loss.item()
                    _, predicted = predictions.max(dim = 1)
                    val_correct += predicted.eq(targets).sum().item()
                    val_total += targets.size(0)

            reliability = tracker.prototype_reliability()
            metrics = _diagnostic_metrics(tracker, train_data._noisy_mask_)

            val_accuracy = round(val_correct / val_total, 4)
            record: Dict[str, Any] = {
                "epoch":                         epoch,
                "train_loss":                    round(train_total_loss / len(train_loader), 4),
                "train_accuracy_observed":       round(train_correct / train_total, 4),
                "train_accuracy_true_oracle":    round(train_true_correct / train_total, 4),
                "noisy_accuracy_observed":       round(noisy_observed_correct / noisy_total, 4) if noisy_total else None,
                "noisy_accuracy_true_oracle":    round(noisy_true_correct / noisy_total, 4) if noisy_total else None,
                "val_loss":                      round(val_total_loss / len(test_loader), 4),
                "val_accuracy":                  val_accuracy,
                "agreement_valid_fraction":      round(float(np.mean(agreement_valid_fractions)), 6),
                "prototype_reliability_mean":    round(float(reliability.mean()), 6),
                "prototype_reliability_median":  round(float(np.median(reliability)), 6),
                "diagnostics":                   metrics,
            }
            epoch_records.append(record)

            all_auc = metrics["all_clean_vs_noisy"]
            hard_auc = metrics["matched_hard_clean_vs_noisy"]["metrics"]
            print(
                f"Epoch {epoch:3}: val={val_accuracy:.4f}  "
                f"train(obs)={record['train_accuracy_observed']:.4f}  "
                f"noisy-memorized={record['noisy_accuracy_observed'] if record['noisy_accuracy_observed'] is not None else 0.0:.4f}  "
                f"AUROC loss={all_auc['loss']['auroc']}  "
                f"agreeEMA={all_auc['negative_agreement_ema']['auroc']}  "
                f"hard-v-noisy agreeEMA={hard_auc['negative_agreement_ema']['auroc']}"
            )

            if epoch in checkpoint_set:
                snapshot_path = out_dir / (
                    f"seed-{seed}_noise{label_noise_rate}_noise-seed{label_noise_seed}_epoch-{epoch}.npz"
                )
                np.savez_compressed(
                    snapshot_path,
                    index =                  np.arange(len(train_data), dtype = np.int64),
                    true_label =             train_data._true_labels_,
                    observed_label =         train_data._labels_,
                    noisy_mask =             train_data._noisy_mask_,
                    loss =                   tracker.latest_loss,
                    loss_ema =               tracker.loss_ema,
                    grad_norm =              tracker.latest_grad_norm,
                    agreement =              tracker.latest_agreement,
                    agreement_ema =          tracker.agreement_ema,
                    observed_confidence =    tracker.latest_observed_confidence,
                    true_confidence_oracle = tracker.latest_true_confidence,
                    prototype_reliability =  reliability,
                )
                print(f"Saved diagnostic snapshot: {snapshot_path}")

    finally:
        feature_capture.close()

    result: Dict[str, Any] = {
        "config": {
            "network_id":          network_id,
            "dataset_id":          dataset_id,
            "batch_size":          batch_size,
            "epochs":              epochs,
            "seed":                seed,
            "label_noise_rate":    label_noise_rate,
            "actual_noise_rate":   train_data.actual_noise_rate,
            "label_noise_seed":    label_noise_seed,
            "prototype_beta":      prototype_beta,
            "agreement_beta":      agreement_beta,
            "loss_beta":           loss_beta,
            "checkpoint_epochs":   sorted(checkpoint_set),
            "classifier_module":   feature_capture.module_name,
            "training_intervention": False,
        },
        "epochs": epoch_records,
    }

    result_path = out_dir / (
        f"seed-{seed}_gradient-consensus-diagnostic-noise{label_noise_rate}-noise-seed{label_noise_seed}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent = 2)

    print(f"Saved: {result_path}")
    return result


# ==================================================================================================
# CLI.
# ==================================================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Non-interventional temporal gradient-consensus diagnostic under label noise."
    )
    parser.add_argument("--network-id",       type = str,   default = "resnet-18")
    parser.add_argument("--dataset-id",       type = str,   default = "cifar-100")
    parser.add_argument("--batch-size",       type = int,   default = 128)
    parser.add_argument("--epochs",           type = int,   default = 100)
    parser.add_argument("--seed",             type = int,   default = 1)
    parser.add_argument(
        "--label-noise-rate", type = float, default = 0.30,
        help = "Fraction of TRAINING labels randomly flipped to a wrong class. Test set remains clean."
    )
    parser.add_argument(
        "--label-noise-seed", type = int, default = 1,
        help = "Fixed independently of --seed so every training seed sees the identical corruption pattern."
    )
    parser.add_argument(
        "--prototype-beta", type = float, default = 0.99,
        help = "EMA coefficient for each observed-class gradient-direction prototype."
    )
    parser.add_argument(
        "--agreement-beta", type = float, default = 0.90,
        help = "Per-example EMA coefficient for temporal directional agreement."
    )
    parser.add_argument(
        "--loss-beta", type = float, default = 0.90,
        help = "Per-example EMA coefficient for the temporal loss baseline."
    )
    parser.add_argument(
        "--checkpoint-epochs", type = _parse_epoch_list, default = _parse_epoch_list("1,5,10,20,40,60,80,100"),
        help = "Comma-separated epochs at which to save per-example .npz snapshots."
    )
    parser.add_argument("--device",      type = str, default = "auto")
    parser.add_argument(
        "--output-path", type = str, default = "results/gradient_consensus_diagnostic"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        network_id =          args.network_id,
        dataset_id =          args.dataset_id,
        batch_size =          args.batch_size,
        epochs =              args.epochs,
        seed =                args.seed,
        label_noise_rate =    args.label_noise_rate,
        label_noise_seed =    args.label_noise_seed,
        prototype_beta =      args.prototype_beta,
        agreement_beta =      args.agreement_beta,
        loss_beta =           args.loss_beta,
        checkpoint_epochs =   args.checkpoint_epochs,
        device =              args.device,
        output_path =         args.output_path,
    )
