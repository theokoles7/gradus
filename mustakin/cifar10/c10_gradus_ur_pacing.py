import math
import time
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import label as scipy_label
from scipy.stats import pearsonr
from sklearn.metrics import classification_report, confusion_matrix
from thop import profile  # For calculating FLOPs
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms

# ============================================================
# Complexity Metric Functions (inlined — unchanged from source)
# ============================================================


def color_variance(sample):
    """Mean of per-channel pixel variances. Higher = more complex."""
    if sample.dim() == 2:
        return sample.var().item()
    return sum(sample[c].var().item() for c in range(sample.shape[0])) / sample.shape[0]


def compression_ratio(sample, quality=95):
    """JPEG compression ratio (original_size / compressed_size). Higher = simpler."""
    image = sample.detach().cpu().numpy()
    if image.ndim == 3:
        image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    if image.max() <= 1.0:
        image = image * 255
    image = image.astype(np.uint8)
    original_size = image.nbytes
    _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return original_size / len(encoded)


def edge_density(sample, low=100, high=200):
    """Fraction of edge pixels via Canny detection. Higher = more complex."""
    image = sample.detach().cpu().numpy()
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image.squeeze(0)
        else:
            image = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
    if image.max() <= 1.0:
        image = image * 255
    image = image.astype(np.uint8)
    edges = cv2.Canny(image, low, high)
    return int(np.count_nonzero(edges)) / edges.size


def spatial_frequency(sample):
    """RMS of row and column pixel differences combined. Higher = more complex."""
    image = sample.float()
    if image.dim() == 3:
        image = image.mean(dim=0)
    row_diff = image[1:, :] - image[:-1, :]
    col_diff = image[:, 1:] - image[:, :-1]
    rf = (row_diff**2).mean().item() ** 0.5
    cf = (col_diff**2).mean().item() ** 0.5
    return math.sqrt(rf**2 + cf**2)


def wavelet_energy(sample, wavelet="db2", level=None):
    """Total wavelet energy via 2-D DWT decomposition. Higher = more complex."""
    import pywt

    image = sample
    if image.dim() == 3:
        image = image.mean(dim=0)
    image = image.detach().cpu().numpy()
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    energy = sum(
        float(np.sum(d**2)) for detail_coeffs in coeffs[1:] for d in detail_coeffs
    )
    return 0.0 if energy < 1e-6 else energy


def wavelet_entropy(sample, wavelet="db2", level=None):
    """Normalised Shannon entropy of wavelet energy distribution. Higher = more complex."""
    import pywt

    image = sample
    if image.dim() == 3:
        image = image.mean(dim=0)
    image = image.detach().cpu().numpy()
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    level_energies = [
        sum(float(np.sum(d**2)) for d in detail_coeffs) for detail_coeffs in coeffs[1:]
    ]
    total_energy = sum(level_energies)
    if total_energy < 1e-6:
        return 0.0
    dist = [e / total_energy for e in level_energies]
    entropy = -sum(p * np.log2(p) for p in dist if p > 0)
    max_entropy = np.log2(len(level_energies)) if len(level_energies) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ============================================================
# GPU Configuration
# ============================================================


def configure_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Num GPUs Available: {torch.cuda.device_count()}")
        torch.cuda.empty_cache()
    return device


# ============================================================
# Unsupervised Correlation-Based Ranking (unchanged)
# ============================================================


def correlation_based_ranking(batch_attributes):
    """
    Unsupervised correlation-based ranking algorithm.

    Each entry in batch_attributes is a batch with an attribute vector.
    Pairwise Pearson correlations drive weight updates that determine
    the final ranking order.

    Args:
        batch_attributes: dict {batch_index: attribute_vector (np.array)}

    Returns:
        ranked_indices: batch indices sorted by weight (descending)
        W: weight array
    """
    if len(batch_attributes) < 2:
        return list(batch_attributes.keys()), np.zeros(len(batch_attributes))

    batch_indices = list(batch_attributes.keys())
    attributes_matrix = np.array([batch_attributes[idx] for idx in batch_indices])
    m = len(batch_indices)
    W = np.zeros(m)

    for i in range(m):
        for j in range(i + 1, m):
            corr_coeff, _ = pearsonr(attributes_matrix[i], attributes_matrix[j])
            if np.isnan(corr_coeff):
                corr_coeff = 0.0
            P_ij = corr_coeff

            if P_ij >= 0:
                if W[i] >= 0 and W[j] >= 0:
                    W[i] += P_ij
                    W[j] += P_ij
                elif W[i] < 0 and W[j] < 0:
                    W[i] -= P_ij
                    W[j] -= P_ij
                elif W[i] >= 0 and W[j] < 0:
                    W[i] -= P_ij
                    W[j] += P_ij
                else:
                    W[i] += P_ij
                    W[j] -= P_ij
            else:
                if W[i] >= 0 and W[j] >= 0:
                    if W[i] < W[j]:
                        W[i] += P_ij
                        W[j] -= P_ij
                    else:
                        W[i] -= P_ij
                        W[j] += P_ij
                elif W[i] < 0 and W[j] < 0:
                    if W[i] < W[j]:
                        W[i] += P_ij
                        W[j] -= P_ij
                    else:
                        W[i] -= P_ij
                        W[j] += P_ij
                elif W[i] >= 0 and W[j] < 0:
                    W[i] -= P_ij
                    W[j] += P_ij
                else:
                    W[i] += P_ij
                    W[j] -= P_ij

    ranked_indices = [batch_indices[i] for i in np.argsort(W)[::-1]]
    return ranked_indices, W


# ============================================================
# Image Complexity Features (unchanged)
# ============================================================


def compute_edge_object_count(img_np):
    """
    Estimate number of objects via Canny edge detection + connected components.
    img_np: H x W x 3 uint8 numpy array
    Returns: int (number of detected object regions)
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    num_labels, _ = cv2.connectedComponents(edges_dilated)
    return max(num_labels - 1, 0)


def compute_all_complexity_features(dataset, device, max_samples=None):
    """
    Compute complexity features for all images in the dataset.

    Returns:
        features_df: DataFrame with all metric columns + complexity_score
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"\nComputing complexity features for {n} images...")

    raw_transform = transforms.Compose([transforms.ToTensor()])
    raw_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=raw_transform
    )

    color_var_scores, comp_ratio_scores, edge_den_scores = [], [], []
    spatial_freq_scores, wav_energy_scores, wav_entropy_scores, edge_obj_scores = (
        [],
        [],
        [],
        [],
    )

    batch_report = max(n // 10, 1)
    for i in range(n):
        if (i + 1) % batch_report == 0 or i == 0:
            print(f"  Processing image {i + 1}/{n} ({100 * (i + 1) / n:.1f}%)")

        img_tensor, _ = raw_dataset[i]
        color_var_scores.append(color_variance(sample=img_tensor))
        comp_ratio_scores.append(compression_ratio(sample=img_tensor))
        edge_den_scores.append(edge_density(sample=img_tensor))
        spatial_freq_scores.append(spatial_frequency(sample=img_tensor))
        wav_energy_scores.append(wavelet_energy(sample=img_tensor))
        wav_entropy_scores.append(wavelet_entropy(sample=img_tensor))
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        edge_obj_scores.append(compute_edge_object_count(img_np))

    features_df = pd.DataFrame(
        {
            "color_variance": color_var_scores,
            "compression_ratio": comp_ratio_scores,
            "edge_density": edge_den_scores,
            "spatial_frequency": spatial_freq_scores,
            "wavelet_energy": wav_energy_scores,
            "wavelet_entropy": wav_entropy_scores,
            "edge_object_count": edge_obj_scores,
        }
    )

    print("\nComplexity feature statistics:")
    print(features_df.describe())

    features_normalized = features_df.copy()
    for col in features_normalized.columns:
        col_min = features_normalized[col].min()
        col_max = features_normalized[col].max()
        if col_max - col_min > 1e-10:
            features_normalized[col] = (features_normalized[col] - col_min) / (
                col_max - col_min
            )
        else:
            features_normalized[col] = 0.0
    features_normalized["compression_ratio"] = (
        1.0 - features_normalized["compression_ratio"]
    )
    features_normalized["complexity_score"] = features_normalized.mean(axis=1)
    features_df["complexity_score"] = features_normalized["complexity_score"]

    print(
        f"\nComplexity score range: [{features_df['complexity_score'].min():.4f}, "
        f"{features_df['complexity_score'].max():.4f}]"
    )
    print(f"Mean complexity: {features_df['complexity_score'].mean():.4f}")
    return features_df


# ============================================================
# Individual Image Ranking — Unsupervised (unchanged)
# ============================================================


def rank_images_unsupervised(features_df, train_indices):
    """
    Rank individual training images using the unsupervised correlation-based
    ranking algorithm.

    Returns:
        top_to_bottom: list of dataset indices (highest-weight image first)
        bottom_to_top: list of dataset indices (lowest-weight image first)
        weights: per-image ranking weights (aligned with train_indices order)
    """
    metric_cols = [
        "color_variance",
        "compression_ratio",
        "edge_density",
        "spatial_frequency",
        "wavelet_energy",
        "wavelet_entropy",
        "edge_object_count",
    ]

    train_features = features_df.iloc[train_indices][metric_cols].copy()
    for col in metric_cols:
        col_min = train_features[col].min()
        col_max = train_features[col].max()
        if col_max - col_min > 1e-10:
            train_features[col] = (train_features[col] - col_min) / (col_max - col_min)
        else:
            train_features[col] = 0.0
    train_features["compression_ratio"] = 1.0 - train_features["compression_ratio"]

    attributes = train_features[metric_cols].values  # (n, 7)
    n = len(attributes)

    X_centered = attributes - attributes.mean(axis=1, keepdims=True)
    X_norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    X_norms[X_norms < 1e-10] = 1.0
    X_normed = X_centered / X_norms

    print(f"\nRanking {n} individual images using correlation-based algorithm...")
    print(f"Total pairwise comparisons: {n * (n - 1) // 2:,}")

    W = [0.0] * n
    ranking_start = time.time()

    for i in range(n):
        if i % 5000 == 0:
            elapsed = time.time() - ranking_start
            if i > 0:
                rate = i / elapsed
                remaining = (n - i) / rate
                print(
                    f"  Ranking: image {i}/{n} ({100 * i / n:.1f}%) "
                    f"- elapsed {elapsed:.0f}s, ~{remaining:.0f}s remaining"
                )
            else:
                print(f"  Ranking: image {i}/{n} (0.0%)")

        corrs = X_normed[i] @ X_normed[i + 1 :].T
        corrs_list = np.nan_to_num(corrs, nan=0.0).tolist()

        wi = W[i]
        for idx, P_ij in enumerate(corrs_list):
            j = i + 1 + idx
            wj = W[j]
            if P_ij >= 0:
                if wi >= 0 and wj >= 0:
                    wi += P_ij
                    W[j] = wj + P_ij
                elif wi < 0 and wj < 0:
                    wi -= P_ij
                    W[j] = wj - P_ij
                elif wi >= 0 and wj < 0:
                    wi -= P_ij
                    W[j] = wj + P_ij
                else:
                    wi += P_ij
                    W[j] = wj - P_ij
            else:
                if wi >= 0 and wj >= 0:
                    if wi < wj:
                        wi += P_ij
                        W[j] = wj - P_ij
                    else:
                        wi -= P_ij
                        W[j] = wj + P_ij
                elif wi < 0 and wj < 0:
                    if wi < wj:
                        wi += P_ij
                        W[j] = wj - P_ij
                    else:
                        wi -= P_ij
                        W[j] = wj + P_ij
                elif wi >= 0 and wj < 0:
                    wi -= P_ij
                    W[j] = wj + P_ij
                else:
                    wi += P_ij
                    W[j] = wj - P_ij
        W[i] = wi

    ranking_time = time.time() - ranking_start
    W = np.array(W)
    print(f"\nRanking complete in {ranking_time:.1f}s")
    print(f"Weight range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"Weight mean: {W.mean():.4f}, std: {W.std():.4f}")

    train_idx_list = (
        train_indices.tolist()
        if hasattr(train_indices, "tolist")
        else list(train_indices)
    )
    sorted_order = np.argsort(W)[::-1]
    top_to_bottom = [train_idx_list[k] for k in sorted_order]
    bottom_to_top = list(reversed(top_to_bottom))

    return top_to_bottom, bottom_to_top, W


# ============================================================
# Step 2 — Group Ranked Images into Ordered Batches
# ============================================================


def create_ordered_batches(sorted_indices, batch_size):
    """
    Partition sorted image indices into sequential curriculum batches.

    Args:
        sorted_indices: list of dataset indices in curriculum order
                        (e.g. easiest-first from unsupervised ranking)
        batch_size: number of images per batch

    Returns:
        batches: list of lists — batches[i] holds image indices for the
                 i-th curriculum batch (batch 0 = easiest, batch K = hardest).
    """
    batches = []
    for i in range(0, len(sorted_indices), batch_size):
        batches.append(sorted_indices[i : i + batch_size])
    return batches


# ============================================================
# Step 4 — Build DataLoader from Active Batches
# ============================================================


def build_curriculum_loader(
    full_train_dataset,
    ordered_batches,
    active_batch_indices,
    batch_size,
    shuffle_within_batch=True,
):
    """
    Build a DataLoader that iterates over only the active curriculum batches
    in their ranked order.

    Each curriculum batch (128 images) maps 1:1 to one DataLoader mini-batch,
    so batch_index_order[mini_batch_idx] always gives the correct curriculum
    batch index for activation / gradient tracking.

    Args:
        full_train_dataset: augmented CIFAR-10 training dataset
        ordered_batches: list of lists from create_ordered_batches()
        active_batch_indices: list of ints — which batches are currently active
        batch_size: DataLoader mini-batch size
        shuffle_within_batch: shuffle image order inside each curriculum batch
                              (prevents position memorisation without breaking order)

    Returns:
        train_loader: DataLoader in curriculum order over active batches
        batch_index_order: list[int] — curriculum batch index for each
                           DataLoader iteration (len == len(train_loader))
    """
    flat_indices = []
    batch_index_order = []

    for batch_idx in active_batch_indices:
        sample_indices = list(ordered_batches[batch_idx])
        if shuffle_within_batch:
            np.random.shuffle(sample_indices)
        flat_indices.extend(sample_indices)
        batch_index_order.append(batch_idx)

    class OrderedIndexSampler(Sampler):
        """Simple sampler that yields indices in a fixed, predetermined order."""

        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    train_loader = DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        sampler=OrderedIndexSampler(flat_indices),
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, batch_index_order


# ============================================================
# Step 5 — Plateau Detection (Self-Calibrating)
# ============================================================


def detect_plateau(values, window):
    """
    Determine whether a scalar time series has plateaued.

    Uses self-calibrating comparison: the coefficient of variation (CV)
    over the recent `window` steps is compared to the CV over the full
    history.  If the recent CV is less than 25% of the historical CV,
    the series is considered to have plateaued.

    This avoids hardcoded delta thresholds — the decision is always
    relative to the signal's own historical fluctuation range.

    Args:
        values: list of floats — the full time series
        window: int — number of recent steps to treat as "recent"

    Returns:
        is_plateau: bool
        stability_score: float in [0, 1]
                         0 = actively changing, 1 = completely flat
    """
    if len(values) < window + 2:
        return False, 0.0

    recent = values[-window:]
    full = values[:]

    cv_recent = np.std(recent) / (abs(np.mean(recent)) + 1e-8)
    cv_full = np.std(full) / (abs(np.mean(full)) + 1e-8)

    if cv_full < 1e-8:
        return True, 1.0

    ratio = cv_recent / cv_full
    stability_score = max(0.0, 1.0 - ratio)
    is_plateau = ratio < 0.25

    return is_plateau, stability_score


# ============================================================
# Step 6 — Four Readiness Signals
# ============================================================


def compute_loss_plateau_signal(loss_history, window):
    """
    Signal 1: Loss Plateau  →  score in [0, 1].

    High score means the training loss has stopped decreasing.
    Compares mean loss over the most recent `window` epochs to the
    preceding `window` epochs using a relative change — no absolute threshold.

    Sigmoid steepness = 10: controls transition sharpness, not scale.
    """
    if len(loss_history) < 2 * window:
        return 0.0

    prior = loss_history[-(2 * window) : -window]
    recent = loss_history[-window:]

    mean_prior = np.mean(prior)
    mean_recent = np.mean(recent)

    if mean_prior < 1e-8:
        return 1.0  # loss already near zero

    # positive relative_change → loss got worse (or flat) → high signal
    relative_change = (mean_recent - mean_prior) / mean_prior
    signal = 1.0 / (1.0 + np.exp(-10.0 * relative_change))
    return float(signal)


def compute_val_acc_trend_signal(val_acc_history, window):
    """
    Signal 2: Validation Accuracy Trend  →  score in [0, 1].

    High score means val accuracy has plateaued or is declining.
    Fits a linear trend over the recent `window` epochs and normalises
    the slope by the within-window std — scale-independent.

    Sigmoid steepness = 5: controls transition sharpness, not scale.
    """
    if len(val_acc_history) < window:
        return 0.0

    recent = val_acc_history[-window:]
    x = np.arange(len(recent), dtype=float)
    slope = np.polyfit(x, recent, 1)[0]

    std_val = np.std(recent)
    if std_val < 1e-8:
        return 1.0  # perfectly flat → plateau

    normalized_slope = slope / std_val
    # positive slope (still improving) → low signal
    # near-zero / negative slope → high signal
    signal = 1.0 / (1.0 + np.exp(5.0 * normalized_slope))
    return float(signal)


def compute_activation_stability_signal(
    batch_std_history, active_batch_indices, window
):
    """
    Signal 3: Activation Stability  →  score in [0, 1].

    High score means the forward-pass activation stds for most active
    curriculum batches have plateaued (model represents them consistently).

    For each active batch, runs detect_plateau() on its activation-std
    time series and averages the stability_scores.
    """
    if not active_batch_indices:
        return 0.0

    total_score = 0.0
    count = 0
    for batch_idx in active_batch_indices:
        if batch_idx not in batch_std_history:
            continue
        _, stability_score = detect_plateau(batch_std_history[batch_idx], window)
        total_score += stability_score
        count += 1

    return total_score / count if count > 0 else 0.0


def compute_gradient_norm_signal(batch_grad_norm_history, active_batch_indices, window):
    """
    Signal 4: Gradient Norm Stability  →  score in [0, 1].

    High score means the pre-clip gradient L2 norms for active batches
    have plateaued (model is no longer learning much from them).

    Identical structure to compute_activation_stability_signal(); detect_plateau()
    is scale-invariant so the same logic applies to gradient norms.

    Why distinct from Signal 3:
      Activations measure the forward pass (consistent representations).
      Gradient norms measure the backward pass (parameter update pressure).
      Both can diverge: stable activations with non-trivial gradients, or
      vanishing gradients while hard-batch activations are still shifting.
    """
    if not active_batch_indices:
        return 0.0

    total_score = 0.0
    count = 0
    for batch_idx in active_batch_indices:
        if batch_idx not in batch_grad_norm_history:
            continue
        _, stability_score = detect_plateau(batch_grad_norm_history[batch_idx], window)
        total_score += stability_score
        count += 1

    return total_score / count if count > 0 else 0.0


# ============================================================
# Step 7 — AdaptiveCurriculumPacer
# ============================================================


class AdaptiveCurriculumPacer:
    """
    Controls the rate at which new curriculum batches are introduced.

    After each epoch, the pacer ingests training loss, validation accuracy,
    per-batch activation stds and per-batch gradient norms, then computes a
    composite readiness score from four signals to decide how many new
    batches to add.

    Design guarantees:
      • Data is introduced monotonically (active set never shrinks).
      • At least one new batch is added per epoch until all data is active
        (dynamic floor — no hardcoded full_data_epoch needed).
      • High readiness can add up to 3× the minimum floor in a single epoch.

    Hardcoded constants justified by the plan:
      window=5      : lookback window for all signals
      ratio < 0.25  : plateau threshold in detect_plateau()
      sigmoid k=10  : sharpness for loss signal (scale-independent)
      sigmoid k=5   : sharpness for val-acc signal (scale-independent)
      3× multiplier : max pace = 3× min pace (structural choice)
    """

    def __init__(self, total_batches, total_epochs, start_fraction=0.3, window=5):
        """
        Args:
            total_batches:  total number of curriculum batches
            total_epochs:   total training epochs
            start_fraction: fraction of batches to activate at epoch 1
            window:         lookback window used by all four signals
        """
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.window = window

        initial_count = max(1, int(total_batches * start_fraction))
        self.active_batch_indices = list(range(initial_count))
        self.next_batch_to_add = initial_count

        remaining = total_batches - initial_count
        self.min_batches_per_epoch = max(1, remaining // total_epochs)

        # Metric histories
        self.loss_history = []
        self.val_acc_history = []
        self.batch_std_history = {}  # {batch_idx: [mean_std_ep1, ...]}
        self.batch_grad_norm_history = {}  # {batch_idx: [mean_grad_norm_ep1, ...]}

        self.pacing_log = []  # list of dicts, one per epoch

    # ----------------------------------------------------------
    def update_metrics(self, train_loss, val_acc, std_df, grad_norm_df):
        """
        Ingest one epoch's training metrics.

        Args:
            train_loss:   float — average training loss this epoch
            val_acc:      float — validation accuracy this epoch
            std_df:       pd.DataFrame indexed by curriculum batch_idx,
                          column 'mean_std' (mean activation std across 4 layers)
            grad_norm_df: pd.DataFrame indexed by curriculum batch_idx,
                          column 'mean_grad_norm' (mean gradient L2 norm)
        """
        self.loss_history.append(train_loss)
        self.val_acc_history.append(val_acc)

        if not std_df.empty and "mean_std" in std_df.columns:
            for batch_idx in std_df.index:
                if batch_idx not in self.batch_std_history:
                    self.batch_std_history[batch_idx] = []
                self.batch_std_history[batch_idx].append(
                    float(std_df.loc[batch_idx, "mean_std"])
                )

        if not grad_norm_df.empty and "mean_grad_norm" in grad_norm_df.columns:
            for batch_idx in grad_norm_df.index:
                if batch_idx not in self.batch_grad_norm_history:
                    self.batch_grad_norm_history[batch_idx] = []
                self.batch_grad_norm_history[batch_idx].append(
                    float(grad_norm_df.loc[batch_idx, "mean_grad_norm"])
                )

    # ----------------------------------------------------------
    def compute_readiness(self):
        """
        Compute the composite readiness score from all four signals.

        Returns:
            readiness:      float in [0, 1] — equal-weight average
            signal_details: dict with individual signal values
        """
        r_loss = compute_loss_plateau_signal(self.loss_history, self.window)
        r_val = compute_val_acc_trend_signal(self.val_acc_history, self.window)
        r_act = compute_activation_stability_signal(
            self.batch_std_history, self.active_batch_indices, self.window
        )
        r_grad = compute_gradient_norm_signal(
            self.batch_grad_norm_history, self.active_batch_indices, self.window
        )

        readiness = (r_loss + r_val + r_act + r_grad) / 4.0
        return readiness, {
            "loss_plateau": r_loss,
            "val_acc_trend": r_val,
            "activation_stability": r_act,
            "gradient_norm_stability": r_grad,
        }

    # ----------------------------------------------------------
    def step(self, epoch):
        """
        Decide how many batches to add this epoch and update the active set.
        Must be called AFTER update_metrics() for the same epoch.

        Args:
            epoch: current epoch number (1-indexed)

        Returns:
            active_batch_indices: updated list of active curriculum batch indices
            pacing_info:          dict with decision details (appended to pacing_log)
        """
        if self.next_batch_to_add >= self.total_batches:
            pacing_info = {
                "epoch": epoch,
                "readiness": 1.0,
                "signals": {},
                "batches_added": 0,
                "active_batches": len(self.active_batch_indices),
                "fraction": 1.0,
            }
            self.pacing_log.append(pacing_info)
            return self.active_batch_indices, pacing_info

        readiness, signals = self.compute_readiness()

        remaining_batches = self.total_batches - self.next_batch_to_add
        remaining_epochs = self.total_epochs - epoch

        # Dynamic floor: guarantees all data is used by the last epoch
        if remaining_epochs > 0:
            min_to_add = max(1, math.ceil(remaining_batches / remaining_epochs))
        else:
            min_to_add = remaining_batches

        # Max: readiness=1 allows up to 3× the minimum floor
        max_to_add = min(remaining_batches, 3 * min_to_add)
        batches_to_add = int(min_to_add + readiness * (max_to_add - min_to_add))
        batches_to_add = max(1, min(batches_to_add, remaining_batches))

        new_indices = list(
            range(self.next_batch_to_add, self.next_batch_to_add + batches_to_add)
        )
        self.active_batch_indices.extend(new_indices)
        self.next_batch_to_add += batches_to_add

        fraction = len(self.active_batch_indices) / self.total_batches

        pacing_info = {
            "epoch": epoch,
            "readiness": readiness,
            "signals": signals,
            "batches_added": batches_to_add,
            "active_batches": len(self.active_batch_indices),
            "fraction": fraction,
        }
        self.pacing_log.append(pacing_info)
        return self.active_batch_indices, pacing_info


# ============================================================
# CurriculumSampler — kept intact for fixed-pacing comparison
# ============================================================


class CurriculumSampler(Sampler):
    """
    Fixed linear-pacing sampler.
    Ramps data fraction from start_fraction to 1.0 linearly by
    full_data_epoch, then uses all data for the remaining epochs.
    Kept alongside AdaptiveCurriculumPacer for direct comparison.
    """

    def __init__(
        self, sorted_indices, total_epochs, start_fraction=0.3, full_data_epoch=None
    ):
        self.sorted_indices = np.array(sorted_indices)
        self.total_epochs = total_epochs
        self.start_fraction = start_fraction
        self.full_data_epoch = (
            full_data_epoch if full_data_epoch is not None else int(0.6 * total_epochs)
        )
        self.current_epoch = 0
        self.current_indices = None
        self._update_indices()

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self._update_indices()

    def _update_indices(self):
        n = len(self.sorted_indices)
        if self.current_epoch >= self.full_data_epoch:
            fraction = 1.0
        else:
            fraction = self.start_fraction + (1.0 - self.start_fraction) * (
                self.current_epoch / self.full_data_epoch
            )
        fraction = min(fraction, 1.0)
        num_samples = max(int(n * fraction), 1)
        self.current_indices = self.sorted_indices[:num_samples]

    def __iter__(self):
        return iter(self.current_indices.tolist())

    def __len__(self):
        return len(self.current_indices)


# ============================================================
# Data Loading (unchanged)
# ============================================================


def load_images(batch_size=128, val_size=5000):
    """
    Loads CIFAR-10 with train / val / test split.
    Returns the full training dataset, val/test loaders, class names, and train indices.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    transform_val_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    full_train_dataset_for_val = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_val_test
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val_test
    )

    class_names = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    num_total_train = len(full_train_dataset)
    indices = np.arange(num_total_train)
    np.random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    print(f"\nData Split:")
    print(f"  Training samples  : {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples      : {len(test_dataset)}")

    val_dataset = Subset(full_train_dataset_for_val, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return full_train_dataset, val_loader, test_loader, class_names, train_indices


# ============================================================
# ResNet-18  (Step 1: forward() supports return_activations)
# ============================================================


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        # Named reference used by train_epoch() for per-layer gradient norms
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_activations=False):
        """
        Args:
            x: input tensor  [B, C, H, W]
            return_activations: when True returns (logits, [act1, act2, act3, act4])
                                where each act is detached (no gradient overhead).
        """
        activations = []
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        if return_activations:
            activations.append(out.detach())

        out = self.layer2(out)
        if return_activations:
            activations.append(out.detach())

        out = self.layer3(out)
        if return_activations:
            activations.append(out.detach())

        out = self.layer4(out)
        if return_activations:
            activations.append(out.detach())

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)

        if return_activations:
            return out, activations
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def calculate_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


def calculate_flops(model, input_size=(3, 32, 32)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\nFLOPs: {flops:,} ({flops / 1e9:.2f} GFLOPs)")
    return flops


# ============================================================
# Step 3 — train_epoch(): activation stds + gradient norms
# ============================================================


def train_epoch(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    batch_index_order=None,
    scheduler=None,
):
    """
    Train for one epoch, optionally collecting per-batch activation stds
    and pre-clip gradient L2 norms for the adaptive pacer.

    Activation stds are collected after the forward pass (detached, zero overhead).
    Gradient norms are collected after loss.backward() and BEFORE
    clip_grad_norm_(), preserving the true per-batch learning signal.

    Args:
        model:             ResNet model with return_activations support
        device:            torch.device
        train_loader:      DataLoader for this epoch
        optimizer:         SGD optimizer
        epoch:             current epoch number (for logging)
        batch_index_order: list[int] — curriculum batch index for each
                           DataLoader mini-batch.  None = no tracking.
        scheduler:         LR scheduler (stepped once at epoch end)

    Returns:
        epoch_loss, epoch_acc, epoch_time, total_images, std_df, grad_norm_df

        std_df:       pd.DataFrame indexed by curriculum batch_idx
                      columns: layer1, layer2, layer3, layer4, mean_std
        grad_norm_df: pd.DataFrame indexed by curriculum batch_idx
                      columns: layer1, layer2, layer3, layer4, mean_grad_norm
        Both are empty DataFrames when batch_index_order is None.
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    total_images = 0
    start_time = time.time()
    num_batches = len(train_loader)

    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    std_devs = {name: [] for name in layer_names}
    grad_norms = {name: [] for name in layer_names}
    batch_indices_list = []

    layer_modules = {
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
    }

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # ---- Forward pass (activation extraction always on) ----
        if np.random.random() > 0.5:
            # Mixup augmentation (50 % chance)
            lam = np.random.beta(0.2, 0.2)
            rand_idx = torch.randperm(data.size(0)).to(device)
            mixed = lam * data + (1 - lam) * data[rand_idx]
            ta, tb = target, target[rand_idx]
            optimizer.zero_grad()
            output, activations = model(mixed, return_activations=True)
            loss = lam * F.cross_entropy(output, ta) + (1 - lam) * F.cross_entropy(
                output, tb
            )
        else:
            optimizer.zero_grad()
            output, activations = model(data, return_activations=True)
            loss = F.cross_entropy(output, target)

        # ---- Collect activation stds (forward-pass signal) ----
        for layer_idx, act in enumerate(activations):
            std_devs[layer_names[layer_idx]].append(torch.std(act).item())

        # ---- Backward pass ----
        loss.backward()

        # ---- Collect gradient norms BEFORE clipping ----
        # Raw (pre-clip) norms reflect true per-batch learning pressure.
        # Clipping would collapse all batches to the same ceiling (1.0).
        for name in layer_names:
            sq_sum = 0.0
            for p in layer_modules[name].parameters():
                if p.grad is not None:
                    sq_sum += p.grad.data.norm(2).item() ** 2
            grad_norms[name].append(sq_sum**0.5)

        # ---- Clip + step ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Record curriculum batch index (adaptive mode only) ----
        if batch_index_order is not None:
            batch_indices_list.append(batch_index_order[batch_idx])

        # ---- Accumulate epoch metrics ----
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        total_images += data.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Epoch {epoch} [{total_images}/{num_batches * data.size(0)} "
                f"({100.0 * (batch_idx + 1) / num_batches:.0f}%)]  "
                f"Loss: {loss.item():.4f}  "
                f"Acc: {100.0 * correct / total:.2f}%"
            )

    if scheduler is not None:
        scheduler.step()

    epoch_time = time.time() - start_time
    epoch_loss = train_loss / num_batches
    epoch_acc = 100.0 * correct / total
    images_per_second = total_images / epoch_time if epoch_time > 0 else 0

    print(
        f"  Epoch {epoch} done in {epoch_time:.1f}s | "
        f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}% | "
        f"{images_per_second:.0f} img/s | Samples used: {total_images}"
    )

    # ---- Build per-batch DataFrames ----
    if batch_index_order is not None and len(batch_indices_list) > 0:
        std_df = pd.DataFrame(std_devs)
        std_df["batch_idx"] = batch_indices_list
        std_df = std_df.set_index("batch_idx")
        std_df["mean_std"] = std_df[layer_names].mean(axis=1)

        grad_norm_df = pd.DataFrame(grad_norms)
        grad_norm_df["batch_idx"] = batch_indices_list
        grad_norm_df = grad_norm_df.set_index("batch_idx")
        grad_norm_df["mean_grad_norm"] = grad_norm_df[layer_names].mean(axis=1)
    else:
        std_df = pd.DataFrame()
        grad_norm_df = pd.DataFrame()

    return epoch_loss, epoch_acc, epoch_time, total_images, std_df, grad_norm_df


def validate(model, device, data_loader, dataset_name="Validation"):
    """Evaluate the model — uses the default forward() (no activation collection)."""
    model.eval()
    val_loss = 0.0
    correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # return_activations=False (default)
            val_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    val_loss /= len(data_loader)
    val_acc = 100.0 * correct / len(data_loader.dataset)
    print(f"  {dataset_name} Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    return val_loss, val_acc, all_targets, all_preds


# ============================================================
# Plotting
# ============================================================


def plot_complexity_distribution(
    features_df, save_path="complexity_distribution_ur_pacing.pdf"
):
    """Plot histograms of all 7 complexity features and the combined score."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    cols = [
        "color_variance",
        "compression_ratio",
        "edge_density",
        "spatial_frequency",
        "wavelet_energy",
        "wavelet_entropy",
        "edge_object_count",
        "complexity_score",
    ]
    titles = [
        "Color Variance",
        "Compression Ratio",
        "Edge Density",
        "Spatial Frequency",
        "Wavelet Energy",
        "Wavelet Entropy",
        "Edge Object Count",
        "Combined Complexity Score",
    ]
    colors = [
        "#2196F3",
        "#4CAF50",
        "#FF9800",
        "#E91E63",
        "#9C27B0",
        "#00BCD4",
        "#795548",
        "#F44336",
    ]

    for ax, col, title, color in zip(axes.flatten(), cols, titles, colors):
        ax.hist(
            features_df[col],
            bins=50,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        m = features_df[col].mean()
        ax.axvline(
            m, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {m:.3f}"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Image Complexity Feature Distributions (CIFAR-10 Training Set)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved complexity distribution plot to {save_path}")


def plot_curriculum_schedule(
    total_epochs,
    start_fraction,
    full_data_epoch,
    total_samples,
    samples_per_epoch,
    recorded_fractions=None,
    save_path="curriculum_schedule_ur_pacing.pdf",
):
    """
    Plot the fixed-pacing curriculum schedule.

    Args:
        recorded_fractions: optional list of actual data fractions per epoch.
                            When provided, overlays the measured curve on the
                            theoretical linear ramp for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, total_epochs + 1))

    theoretical = []
    for e in epochs:
        if e >= full_data_epoch:
            f = 1.0
        else:
            f = start_fraction + (1.0 - start_fraction) * (e / full_data_epoch)
        theoretical.append(min(f, 1.0))

    axes[0].plot(
        epochs,
        [f * 100 for f in theoretical],
        color="#2196F3",
        linewidth=2,
        label="Theoretical (linear)",
    )
    if recorded_fractions is not None:
        axes[0].plot(
            epochs[: len(recorded_fractions)],
            [f * 100 for f in recorded_fractions],
            color="#FF5722",
            linewidth=1.5,
            linestyle="--",
            label="Actual",
        )
    axes[0].axhline(100, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(
        full_data_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Full data at epoch {full_data_epoch}",
    )
    axes[0].set_title("Fixed Pacing: Data Fraction Over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Data Used (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs[: len(samples_per_epoch)],
        samples_per_epoch,
        color="#4CAF50",
        linewidth=2,
    )
    axes[1].axhline(
        total_samples,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Total: {total_samples}",
    )
    axes[1].set_title("Fixed Pacing: Samples Used Per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Number of Samples")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved curriculum schedule to {save_path}")


def plot_training_metrics(
    epoch_times,
    train_losses,
    train_accs,
    val_losses,
    val_accs,
    save_path="training_metrics_ur_pacing.pdf",
):
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, marker="o", markersize=2)
    plt.title("Training Time per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    cumulative_time = np.cumsum(epoch_times)
    plt.plot(range(1, len(cumulative_time) + 1), cumulative_time / 60)
    plt.title("Cumulative Training Time")
    plt.xlabel("Epoch")
    plt.ylabel("Time (minutes)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved training metrics to {save_path}")


def plot_confusion_matrix(
    all_targets,
    all_preds,
    class_names,
    title="Confusion Matrix",
    save_path="confusion_matrix_ur_pacing.pdf",
):
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_comparison(results_list, labels, save_path="comparison_ur_pacing.pdf"):
    """
    Overlay training curves for all experiments on three shared axes.

    Args:
        results_list: list of result dicts from run_single_experiment()
        labels:       list of display-name strings (one per result)
        save_path:    output PDF path
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for res, label in zip(results_list, labels):
        n = len(res["train_accs"])
        epochs = range(1, n + 1)
        axes[0].plot(epochs, res["train_losses"], label=label, linewidth=1.5)
        axes[1].plot(epochs, res["val_accs"], label=label, linewidth=1.5)
        axes[2].plot(epochs, res["train_accs"], label=label, linewidth=1.5)

    for ax, title, ylabel in zip(
        axes,
        ["Training Loss", "Validation Accuracy", "Training Accuracy"],
        ["Loss", "Accuracy (%)", "Accuracy (%)"],
    ):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Unsupervised Ranking: Fixed vs Adaptive Pacing — All Experiments",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_path}")


# ============================================================
# Step 9 — plot_adaptive_schedule()
# ============================================================


def plot_adaptive_schedule(
    pacing_log,
    total_batches,
    total_samples,
    save_path="adaptive_schedule_ur_pacing.pdf",
):
    """
    2×2 visualisation of the adaptive pacing schedule.

    [0,0] Actual data fraction introduced over epochs
    [0,1] Composite readiness score over epochs
    [1,0] Breakdown of all four individual signals
    [1,1] Number of batches added each epoch
    """
    if not pacing_log:
        print("Warning: empty pacing_log — skipping plot_adaptive_schedule.")
        return

    epochs = [p["epoch"] for p in pacing_log]
    fractions = [p["fraction"] for p in pacing_log]
    readiness = [p["readiness"] for p in pacing_log]
    loss_sigs = [p["signals"].get("loss_plateau", 0) for p in pacing_log]
    val_sigs = [p["signals"].get("val_acc_trend", 0) for p in pacing_log]
    act_sigs = [p["signals"].get("activation_stability", 0) for p in pacing_log]
    grad_sigs = [p["signals"].get("gradient_norm_stability", 0) for p in pacing_log]
    added = [p["batches_added"] for p in pacing_log]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # [0,0] Data fraction
    axes[0, 0].plot(epochs, [f * 100 for f in fractions], color="#2196F3", linewidth=2)
    axes[0, 0].axhline(100, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Adaptive Data Introduction", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Training Data Used (%)")
    axes[0, 0].set_ylim(0, 108)
    axes[0, 0].grid(True, alpha=0.3)

    # [0,1] Composite readiness
    axes[0, 1].plot(epochs, readiness, color="#E91E63", linewidth=2)
    axes[0, 1].set_title("Composite Readiness Score", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Readiness [0–1]")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    # [1,0] Signal breakdown
    axes[1, 0].plot(epochs, loss_sigs, label="Loss Plateau", linewidth=1.5)
    axes[1, 0].plot(epochs, val_sigs, label="Val Acc Trend", linewidth=1.5)
    axes[1, 0].plot(epochs, act_sigs, label="Activation Stability", linewidth=1.5)
    axes[1, 0].plot(epochs, grad_sigs, label="Gradient Norm Stability", linewidth=1.5)
    axes[1, 0].set_title("Readiness Signal Breakdown", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Signal [0–1]")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # [1,1] Batches added per epoch
    axes[1, 1].bar(epochs, added, color="#4CAF50", alpha=0.7)
    axes[1, 1].set_title("Batches Added Per Epoch", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Batches Added")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Adaptive Pacing Schedule (Unsupervised Ranking)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved adaptive schedule plot to {save_path}")


# ============================================================
# Step 8 — run_single_experiment(): fixed OR adaptive pacing
# ============================================================


def run_single_experiment(
    direction_label,
    curriculum_order,
    full_train_dataset,
    val_loader,
    test_loader,
    class_names,
    total_train_samples,
    device,
    total_epochs,
    batch_size,
    learning_rate,
    weight_decay,
    momentum,
    start_fraction,
    full_data_epoch,
    adaptive=False,
    window=5,
    ordered_batches=None,
):
    """
    Run one complete training experiment.

    Args:
        direction_label: string tag, e.g. 't2b_fixed' or 'b2t_adaptive'
        curriculum_order: flat list of dataset indices in curriculum order
                          (consumed by CurriculumSampler when adaptive=False)
        full_train_dataset: augmented CIFAR-10 Dataset
        val_loader / test_loader: DataLoaders (val/test, no augmentation)
        class_names: list of 10 class-name strings
        total_train_samples: int — size of the training split
        device: torch.device
        total_epochs, batch_size, learning_rate, weight_decay, momentum: hyperparams
        start_fraction: initial data fraction (shared by both modes)
        full_data_epoch: epoch at which fixed pacing reaches 100% (unused in adaptive mode)
        adaptive: if True → AdaptiveCurriculumPacer + build_curriculum_loader
                  if False → CurriculumSampler (fixed linear schedule)
        window: lookback window for the four adaptive signals (adaptive mode only)
        ordered_batches: list of lists from create_ordered_batches()
                         REQUIRED when adaptive=True

    Returns:
        dict with keys: model, summary_df, train_losses, train_accs, val_losses,
                        val_accs, epoch_times, samples_per_epoch, best_val_acc,
                        final_test_acc, final_val_acc, total_training_time, pacing_log
    """
    print(f"\n{'=' * 60}")
    print(
        f"Experiment: {direction_label}  [{'Adaptive' if adaptive else 'Fixed'} Pacing]"
    )
    print(f"{'=' * 60}")

    # ---- Initialise pacing controller ----
    if adaptive:
        if ordered_batches is None:
            raise ValueError(
                "ordered_batches must be provided when adaptive=True. "
                "Call create_ordered_batches(curriculum_order, batch_size) first."
            )
        pacer = AdaptiveCurriculumPacer(
            total_batches=len(ordered_batches),
            total_epochs=total_epochs,
            start_fraction=start_fraction,
            window=window,
        )
        print(
            f"  AdaptiveCurriculumPacer: {len(ordered_batches)} batches, "
            f"start={start_fraction * 100:.0f}%, window={window}"
        )
    else:
        curriculum_sampler = CurriculumSampler(
            sorted_indices=curriculum_order,
            total_epochs=total_epochs,
            start_fraction=start_fraction,
            full_data_epoch=full_data_epoch,
        )
        print(
            f"  CurriculumSampler: fixed linear pacing, "
            f"full data at epoch {full_data_epoch}"
        )

    # ---- Model ----
    model = ResNet18(num_classes=10).to(device)
    total_params, _ = calculate_model_parameters(model)

    try:
        calculate_flops(model)
    except Exception:
        print("  Skipping FLOPs calculation.")

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    # ---- Training loop ----
    print(f"\n{'=' * 60}")
    print(f"Training: {direction_label}")
    print(f"{'=' * 60}")

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epoch_times = []
    samples_per_epoch = []
    recorded_fractions = []

    best_val_acc = 0.0
    best_model_state = None
    total_start = time.time()

    for epoch in range(1, total_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"[{direction_label}] EPOCH {epoch}/{total_epochs}")
        print(f"{'=' * 50}")

        # ---- Build loader for this epoch ----
        if adaptive:
            active_indices = pacer.active_batch_indices
            train_loader_ep, batch_index_order = build_curriculum_loader(
                full_train_dataset,
                ordered_batches,
                active_indices,
                batch_size,
                shuffle_within_batch=True,
            )
            num_samples = sum(len(ordered_batches[i]) for i in active_indices)
            pct = num_samples / total_train_samples * 100
            print(
                f"  Adaptive: {len(active_indices)}/{len(ordered_batches)} batches "
                f"— {num_samples}/{total_train_samples} samples ({pct:.1f}%)"
            )
        else:
            curriculum_sampler.set_epoch(epoch)
            num_samples = len(curriculum_sampler)
            pct = num_samples / total_train_samples * 100
            batch_index_order = None
            train_loader_ep = DataLoader(
                full_train_dataset,
                batch_size=batch_size,
                sampler=curriculum_sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
            )
            print(f"  Fixed: {num_samples}/{total_train_samples} samples ({pct:.1f}%)")

        samples_per_epoch.append(num_samples)
        recorded_fractions.append(num_samples / total_train_samples)

        # ---- Train ----
        (train_loss, train_acc, epoch_time, total_images, std_df, grad_norm_df) = (
            train_epoch(
                model,
                device,
                train_loader_ep,
                optimizer,
                epoch,
                batch_index_order=batch_index_order,
                scheduler=scheduler,
            )
        )

        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ---- Validate ----
        val_loss, val_acc, _, _ = validate(
            model, device, val_loader, dataset_name="Validation"
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ---- Update adaptive pacer ----
        if adaptive:
            pacer.update_metrics(train_loss, val_acc, std_df, grad_norm_df)
            active_indices, pacing_info = pacer.step(epoch)

            sigs = pacing_info.get("signals", {})
            if sigs:
                print(
                    f"  Readiness: {pacing_info['readiness']:.3f} "
                    f"(loss={sigs['loss_plateau']:.3f}, "
                    f"val={sigs['val_acc_trend']:.3f}, "
                    f"act={sigs['activation_stability']:.3f}, "
                    f"grad={sigs['gradient_norm_stability']:.3f})"
                )
            print(
                f"  +{pacing_info['batches_added']} batches → "
                f"{pacing_info['active_batches']}/{len(ordered_batches)} "
                f"({pacing_info['fraction'] * 100:.1f}%)"
            )

        # ---- Save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                model.state_dict(), f"best_resnet18_{direction_label}_ur_pacing.pth"
            )
            print(f"  ** New best validation accuracy: {best_val_acc:.2f}% **")

    # ---- Final evaluation ----
    print(f"\n{'=' * 60}")
    print(f"Final Evaluation: {direction_label}")
    print(f"{'=' * 60}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  Loaded best model (val acc: {best_val_acc:.2f}%)")

    total_training_time = time.time() - total_start

    print("\nFinal Validation Set Evaluation:")
    final_val_loss, final_val_acc, _, _ = validate(
        model, device, val_loader, dataset_name="Validation"
    )

    print("\nFinal Test Set Evaluation:")
    final_test_loss, final_test_acc, test_targets, test_preds = validate(
        model, device, test_loader, dataset_name="Test"
    )

    # ---- Plots ----
    plot_training_metrics(
        epoch_times,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        save_path=f"training_metrics_{direction_label}_ur_pacing.pdf",
    )

    plot_confusion_matrix(
        test_targets,
        test_preds,
        class_names,
        title=f"Test Confusion Matrix ({direction_label})",
        save_path=f"confusion_matrix_{direction_label}_ur_pacing.pdf",
    )

    if adaptive:
        plot_adaptive_schedule(
            pacer.pacing_log,
            len(ordered_batches),
            total_train_samples,
            save_path=f"adaptive_schedule_{direction_label}_ur_pacing.pdf",
        )
        pacing_log = pacer.pacing_log
    else:
        plot_curriculum_schedule(
            total_epochs,
            start_fraction,
            full_data_epoch,
            total_train_samples,
            samples_per_epoch,
            recorded_fractions=recorded_fractions,
            save_path=f"curriculum_schedule_{direction_label}_ur_pacing.pdf",
        )
        pacing_log = []

    # ---- Summary ----
    total_processed = sum(samples_per_epoch)
    max_possible = total_epochs * total_train_samples
    data_utilization_idx = total_processed / max_possible
    data_savings_idx = 1.0 - data_utilization_idx
    avg_epoch_time = float(np.mean(epoch_times))

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {direction_label}")
    print(f"{'=' * 60}")
    print(f"  Pacing type         : {'adaptive' if adaptive else 'fixed'}")
    print(
        f"  Total training time : {total_training_time:.1f}s "
        f"({total_training_time / 60:.1f} min)"
    )
    print(f"  Average epoch time  : {avg_epoch_time:.1f}s")
    print(f"  Best validation acc : {best_val_acc:.2f}%")
    print(f"  Final validation acc: {final_val_acc:.2f}%")
    print(f"  Final test accuracy : {final_test_acc:.2f}%")
    print(f"  DUI                 : {data_utilization_idx:.4f}")
    print(f"  DSI                 : {data_savings_idx:.4f}")
    print(f"  Total samples proc. : {total_processed:,}")
    print(f"  Total parameters    : {total_params:,}")

    summary = {
        "direction": [direction_label],
        "pacing_type": ["adaptive" if adaptive else "fixed"],
        "total_training_time": [total_training_time],
        "average_epoch_time": [avg_epoch_time],
        "epochs_completed": [total_epochs],
        "best_validation_accuracy": [best_val_acc],
        "final_validation_accuracy": [final_val_acc],
        "final_test_accuracy": [final_test_acc],
        "total_parameters": [total_params],
        "data_utilization_index": [data_utilization_idx],
        "data_savings_index": [data_savings_idx],
        "total_samples_processed": [total_processed],
        "curriculum_start_fraction": [start_fraction],
        "curriculum_full_data_epoch": [None if adaptive else full_data_epoch],
        "adaptive_window": [window if adaptive else None],
        "learning_rate": [learning_rate],
        "weight_decay": [weight_decay],
        "batch_size": [batch_size],
        "train_samples": [total_train_samples],
        "val_samples": [5000],
        "test_samples": [10000],
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"summary_{direction_label}_ur_pacing.csv", index=False)
    print(f"  Summary saved to summary_{direction_label}_ur_pacing.csv")

    epoch_df = pd.DataFrame(
        {
            "epoch": range(1, total_epochs + 1),
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
            "epoch_time": epoch_times,
            "samples_used": samples_per_epoch,
            "data_fraction": recorded_fractions,
        }
    )
    epoch_df.to_csv(f"epoch_history_{direction_label}_ur_pacing.csv", index=False)
    print(f"  Epoch history saved to epoch_history_{direction_label}_ur_pacing.csv")

    return {
        "model": model,
        "summary_df": summary_df,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "epoch_times": epoch_times,
        "samples_per_epoch": samples_per_epoch,
        "best_val_acc": best_val_acc,
        "final_val_acc": final_val_acc,
        "final_test_acc": final_test_acc,
        "total_training_time": total_training_time,
        "pacing_log": pacing_log,
    }


# ============================================================
# Step 10 — Main: 4 experiments (fixed + adaptive × t2b + b2t)
# ============================================================


def run_cifar10_curriculum_learning():
    device = configure_gpu()

    # ---- Hyperparameters (identical to c10_gradus_unsupervised_ranking.py) ----
    total_epochs = 200
    batch_size = 128
    learning_rate = 0.05
    weight_decay = 5e-4
    momentum = 0.9

    # Curriculum parameters
    start_fraction = 0.3  # both fixed and adaptive start at 30 %
    full_data_epoch = 120  # used only by fixed-pacing sampler
    window = 5  # lookback window for adaptive signals

    # ---- Load data ----
    full_train_dataset, val_loader, test_loader, class_names, train_indices = (
        load_images(batch_size, val_size=5000)
    )

    total_train_samples = len(train_indices)
    print(f"\nTotal training samples: {total_train_samples}")

    # ----------------------------------------------------------
    # STEP 1 — Image complexity features
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Computing Image Complexity Features")
    print("=" * 60)

    features_df = compute_all_complexity_features(full_train_dataset, device)
    features_df.to_csv("image_complexity_features_ur_pacing.csv", index=True)
    print("Saved complexity features to image_complexity_features_ur_pacing.csv")
    plot_complexity_distribution(
        features_df, save_path="complexity_distribution_ur_pacing.pdf"
    )

    # ----------------------------------------------------------
    # STEP 2 — Unsupervised correlation-based ranking
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Unsupervised Correlation-Based Ranking (Per-Image)")
    print("=" * 60)

    top_to_bottom, bottom_to_top, weights = rank_images_unsupervised(
        features_df, train_indices
    )

    print(f"\nt2b: first 5 = {top_to_bottom[:5]}, last 5 = {top_to_bottom[-5:]}")
    print(f"b2t: first 5 = {bottom_to_top[:5]}, last 5 = {bottom_to_top[-5:]}")

    ranking_df = pd.DataFrame(
        {
            "dataset_index": train_indices,
            "weight": weights,
        }
    )
    ranking_df.to_csv("unsupervised_ranking_weights_ur_pacing.csv", index=False)
    print("Saved ranking weights to unsupervised_ranking_weights_ur_pacing.csv")

    # ----------------------------------------------------------
    # STEP 3 — Create ordered batches for adaptive experiments
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Creating Ordered Batches for Adaptive Pacing")
    print("=" * 60)

    ordered_batches_t2b = create_ordered_batches(top_to_bottom, batch_size)
    ordered_batches_b2t = create_ordered_batches(bottom_to_top, batch_size)

    print(f"  t2b: {len(ordered_batches_t2b)} batches × ~{batch_size} images")
    print(f"  b2t: {len(ordered_batches_b2t)} batches × ~{batch_size} images")

    # ----------------------------------------------------------
    # EXPERIMENT 1 — Top-to-Bottom, Fixed Pacing
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TOP-TO-BOTTOM — FIXED LINEAR PACING")
    print("=" * 70)

    results_t2b_fixed = run_single_experiment(
        "t2b_fixed",
        top_to_bottom,
        full_train_dataset,
        val_loader,
        test_loader,
        class_names,
        total_train_samples,
        device,
        total_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        momentum,
        start_fraction,
        full_data_epoch,
        adaptive=False,
    )

    # ----------------------------------------------------------
    # EXPERIMENT 2 — Top-to-Bottom, Adaptive Pacing
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TOP-TO-BOTTOM — ADAPTIVE PACING")
    print("=" * 70)

    results_t2b_adaptive = run_single_experiment(
        "t2b_adaptive",
        top_to_bottom,
        full_train_dataset,
        val_loader,
        test_loader,
        class_names,
        total_train_samples,
        device,
        total_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        momentum,
        start_fraction,
        full_data_epoch,
        adaptive=True,
        window=window,
        ordered_batches=ordered_batches_t2b,
    )

    # ----------------------------------------------------------
    # EXPERIMENT 3 — Bottom-to-Top, Fixed Pacing
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: BOTTOM-TO-TOP — FIXED LINEAR PACING")
    print("=" * 70)

    results_b2t_fixed = run_single_experiment(
        "b2t_fixed",
        bottom_to_top,
        full_train_dataset,
        val_loader,
        test_loader,
        class_names,
        total_train_samples,
        device,
        total_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        momentum,
        start_fraction,
        full_data_epoch,
        adaptive=False,
    )

    # ----------------------------------------------------------
    # EXPERIMENT 4 — Bottom-to-Top, Adaptive Pacing
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: BOTTOM-TO-TOP — ADAPTIVE PACING")
    print("=" * 70)

    results_b2t_adaptive = run_single_experiment(
        "b2t_adaptive",
        bottom_to_top,
        full_train_dataset,
        val_loader,
        test_loader,
        class_names,
        total_train_samples,
        device,
        total_epochs,
        batch_size,
        learning_rate,
        weight_decay,
        momentum,
        start_fraction,
        full_data_epoch,
        adaptive=True,
        window=window,
        ordered_batches=ordered_batches_b2t,
    )

    # ----------------------------------------------------------
    # Comparison
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON: All 4 Experiments")
    print("=" * 70)

    all_results = [
        results_t2b_fixed,
        results_t2b_adaptive,
        results_b2t_fixed,
        results_b2t_adaptive,
    ]
    exp_labels = [
        "t2b Fixed",
        "t2b Adaptive",
        "b2t Fixed",
        "b2t Adaptive",
    ]

    plot_comparison(all_results, exp_labels, save_path="comparison_all_ur_pacing.pdf")

    # Print aligned summary table
    col_w = 16
    print(
        f"\n{'Experiment':<{col_w}} {'Best Val%':>10} {'Test%':>8} "
        f"{'DUI':>8} {'DSI':>8} {'Time(min)':>10}"
    )
    print("-" * (col_w + 48))
    for res, label in zip(all_results, exp_labels):
        dui = float(res["summary_df"]["data_utilization_index"].iloc[0])
        dsi = float(res["summary_df"]["data_savings_index"].iloc[0])
        t = res["total_training_time"] / 60.0
        print(
            f"{label:<{col_w}} "
            f"{res['best_val_acc']:>9.2f}% "
            f"{res['final_test_acc']:>7.2f}% "
            f"{dui:>8.4f} {dsi:>8.4f} {t:>9.1f}m"
        )

    # Save combined CSV
    comparison_df = pd.concat([r["summary_df"] for r in all_results], ignore_index=True)
    comparison_df.to_csv("comparison_all_ur_pacing.csv", index=False)
    print("\nCombined comparison saved to comparison_all_ur_pacing.csv")

    return (
        results_t2b_fixed["model"],
        results_t2b_adaptive["model"],
        results_b2t_fixed["model"],
        results_b2t_adaptive["model"],
        comparison_df,
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_cifar10_curriculum_learning()
