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
# Complexity Metric Functions (inlined)
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
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
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
    rf = (row_diff ** 2).mean().item() ** 0.5
    cf = (col_diff ** 2).mean().item() ** 0.5
    return math.sqrt(rf ** 2 + cf ** 2)


def wavelet_energy(sample, wavelet='db2', level=None):
    """Total wavelet energy via 2D DWT decomposition. Higher = more complex."""
    import pywt
    image = sample
    if image.dim() == 3:
        image = image.mean(dim=0)
    image = image.detach().cpu().numpy()
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    energy = sum(
        float(np.sum(d ** 2))
        for detail_coeffs in coeffs[1:]
        for d in detail_coeffs
    )
    return 0.0 if energy < 1e-6 else energy


def wavelet_entropy(sample, wavelet='db2', level=None):
    """Normalized Shannon entropy of wavelet energy distribution. Higher = more complex."""
    import pywt
    image = sample
    if image.dim() == 3:
        image = image.mean(dim=0)
    image = image.detach().cpu().numpy()
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)
    level_energies = [
        sum(float(np.sum(d ** 2)) for d in detail_coeffs)
        for detail_coeffs in coeffs[1:]
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
# Unsupervised Correlation-Based Ranking
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
                else:  # W[i] < 0 and W[j] >= 0
                    W[i] += P_ij
                    W[j] -= P_ij
            else:  # P_ij < 0
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
                else:  # W[i] < 0 and W[j] >= 0
                    W[i] += P_ij
                    W[j] -= P_ij

    ranked_indices = [batch_indices[i] for i in np.argsort(W)[::-1]]
    return ranked_indices, W


# ============================================================
# Image Complexity Features
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
    Compute complexity features for all images in the dataset using gradus metrics
    plus edge object count.

    Returns:
        features_df: DataFrame with all metric columns + complexity_score
    """
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"\nComputing complexity features for {n} images...")

    raw_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    raw_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=raw_transform
    )

    color_var_scores = []
    comp_ratio_scores = []
    edge_den_scores = []
    spatial_freq_scores = []
    wav_energy_scores = []
    wav_entropy_scores = []
    edge_obj_scores = []

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

    # Normalize each feature to [0, 1] and compute combined complexity_score
    # (used for plotting and within-batch ordering)
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
    # Invert compression_ratio: high ratio = simple -> low normalized = simple
    features_normalized["compression_ratio"] = (
        1.0 - features_normalized["compression_ratio"]
    )
    features_normalized["complexity_score"] = features_normalized.mean(axis=1)
    features_df["complexity_score"] = features_normalized["complexity_score"]

    print(
        f"\nComplexity score range: [{features_df['complexity_score'].min():.4f}, {features_df['complexity_score'].max():.4f}]"
    )
    print(f"Mean complexity: {features_df['complexity_score'].mean():.4f}")

    return features_df


# ============================================================
# Individual Image Ranking (Unsupervised)
# ============================================================


def rank_images_unsupervised(features_df, train_indices):
    """
    Rank individual training images using the unsupervised correlation-based
    ranking algorithm.

    Each image's attribute vector is its 7 normalized complexity metrics.
    Pairwise Pearson correlations between all image attribute vectors drive
    the weight-based ranking.

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

    # Normalize metrics for training images
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

    # Pre-compute centered & L2-normalized attribute vectors for fast Pearson correlation.
    # For vectors a, b: pearson(a, b) = dot(a_normed, b_normed)
    # where a_normed = (a - mean(a)) / ||a - mean(a)||
    X_centered = attributes - attributes.mean(axis=1, keepdims=True)
    X_norms = np.linalg.norm(X_centered, axis=1, keepdims=True)
    X_norms[X_norms < 1e-10] = 1.0  # avoid division by zero for constant vectors
    X_normed = X_centered / X_norms  # (n, 7)

    print(f"\nRanking {n} individual images using correlation-based algorithm...")
    print(f"Total pairwise comparisons: {n * (n - 1) // 2:,}")

    W = [0.0] * n  # Python list for fast element access in tight loop
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

        # Pearson correlations of image i with all images j > i (vectorized)
        corrs = X_normed[i] @ X_normed[i + 1 :].T  # (n-i-1,)
        corrs_list = np.nan_to_num(corrs, nan=0.0).tolist()

        wi = W[i]  # cache locally — accessed every inner iteration
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
                else:  # wi < 0, wj >= 0
                    wi += P_ij
                    W[j] = wj - P_ij
            else:  # P_ij < 0
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
                else:  # wi < 0, wj >= 0
                    wi += P_ij
                    W[j] = wj - P_ij
        W[i] = wi  # write back accumulated W[i]

    ranking_time = time.time() - ranking_start
    W = np.array(W)

    print(f"\nRanking complete in {ranking_time:.1f}s")
    print(f"Weight range: [{W.min():.4f}, {W.max():.4f}]")
    print(f"Weight mean: {W.mean():.4f}, std: {W.std():.4f}")

    # Map ranked positions back to dataset indices
    train_idx_list = (
        train_indices.tolist()
        if hasattr(train_indices, "tolist")
        else list(train_indices)
    )
    sorted_order = np.argsort(W)[::-1]  # descending by weight

    top_to_bottom = [train_idx_list[k] for k in sorted_order]
    bottom_to_top = list(reversed(top_to_bottom))

    return top_to_bottom, bottom_to_top, W


# ============================================================
# Curriculum Sampler (No Random Shuffling)
# ============================================================


class CurriculumSampler(Sampler):
    """
    Sampler that introduces training data progressively over the course of training.

    At epoch 0: only the first `start_fraction` of data is used.
    Linearly increases to 100% of data by `full_data_epoch`.
    After that, all data is used.

    The curriculum ordering is ALWAYS maintained -- no random shuffling.
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
# Data Loading
# ============================================================


def load_images(batch_size=128, val_size=5000):
    """
    Loads CIFAR-100 with train/val/test split.
    Returns the full training dataset, val/test loaders, class names, and train indices.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    transform_val_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    full_train_dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    full_train_dataset_for_val = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_val_test
    )
    test_dataset = datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_val_test
    )

    class_names = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    num_total_train = len(full_train_dataset)
    indices = np.arange(num_total_train)
    np.random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    print(f"\nData Split:")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    print(f"  Test samples: {len(test_dataset)}")

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
# ResNet Model
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
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out


def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def calculate_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params


def calculate_flops(model, input_size=(3, 32, 32)):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"\nModel Computational Requirements:")
    print(f"FLOPs: {flops:,} ({flops / 1e9:.2f} GFLOPs)")
    return flops


# ============================================================
# Training & Evaluation
# ============================================================


def train_epoch(model, device, train_loader, optimizer, epoch, scheduler=None):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    total_images = 0
    num_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Mixup augmentation (50% chance)
        if np.random.random() > 0.5:
            lam = np.random.beta(0.2, 0.2)
            rand_idx = torch.randperm(data.size(0)).to(device)
            mixed_data = lam * data + (1 - lam) * data[rand_idx]
            target_a, target_b = target, target[rand_idx]
            optimizer.zero_grad()
            output = model(mixed_data)
            loss = lam * F.cross_entropy(output, target_a) + (
                1 - lam
            ) * F.cross_entropy(output, target_b)
        else:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        total_images += data.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Epoch {epoch} [{total_images}/{num_batches * data.size(0)} "
                f"({100.0 * (batch_idx + 1) / num_batches:.0f}%)]  "
                f"Loss: {loss.item():.4f}  Acc: {100.0 * correct / total:.2f}%"
            )

    if scheduler is not None:
        scheduler.step()

    epoch_time = time.time() - start_time
    epoch_loss = train_loss / num_batches
    epoch_acc = 100.0 * correct / total
    images_per_second = total_images / epoch_time if epoch_time > 0 else 0

    print(
        f"  Epoch {epoch} done in {epoch_time:.1f}s | Loss: {epoch_loss:.4f} | "
        f"Acc: {epoch_acc:.2f}% | {images_per_second:.0f} img/s | "
        f"Samples used: {total_images}"
    )

    return epoch_loss, epoch_acc, epoch_time, total_images


def validate(model, device, data_loader, dataset_name="Validation"):
    model.eval()
    val_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
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
    features_df, save_path="complexity_distribution_c100_gradus.pdf"
):
    """Plot distribution of complexity features and combined score."""
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
        ax.axvline(
            features_df[col].mean(),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean: {features_df[col].mean():.3f}",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Image Complexity Feature Distributions (CIFAR-100 Training Set)",
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
    save_path="curriculum_schedule_c100_gradus.pdf",
):
    """Plot the curriculum learning schedule."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = list(range(1, total_epochs + 1))
    fractions = []
    for e in epochs:
        if e >= full_data_epoch:
            f = 1.0
        else:
            f = start_fraction + (1.0 - start_fraction) * (e / full_data_epoch)
        fractions.append(min(f, 1.0))

    axes[0].plot(epochs, [f * 100 for f in fractions], color="#2196F3", linewidth=2)
    axes[0].axhline(100, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(
        full_data_epoch,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Full data at epoch {full_data_epoch}",
    )
    axes[0].set_title("Curriculum: Data Fraction Over Epochs", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Data Used (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, samples_per_epoch, color="#4CAF50", linewidth=2)
    axes[1].axhline(
        total_samples,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Total: {total_samples}",
    )
    axes[1].set_title("Curriculum: Samples Used Per Epoch", fontweight="bold")
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
    save_path="training_metrics_curriculum_c100_gradus.pdf",
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
    save_path="confusion_matrix_c100_gradus.pdf",
):
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_comparison(results_t2b, results_b2t, save_path="comparison_c100_gradus.pdf"):
    """Plot comparison between top-to-bottom and bottom-to-top curricula."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    epochs = range(1, len(results_t2b["train_accs"]) + 1)

    # Training Loss
    axes[0].plot(
        epochs, results_t2b["train_losses"], label="Top-to-Bottom", linewidth=1.5
    )
    axes[0].plot(
        epochs, results_b2t["train_losses"], label="Bottom-to-Top", linewidth=1.5
    )
    axes[0].set_title("Training Loss Comparison", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation Accuracy
    axes[1].plot(epochs, results_t2b["val_accs"], label="Top-to-Bottom", linewidth=1.5)
    axes[1].plot(epochs, results_b2t["val_accs"], label="Bottom-to-Top", linewidth=1.5)
    axes[1].set_title("Validation Accuracy Comparison", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Training Accuracy
    axes[2].plot(
        epochs, results_t2b["train_accs"], label="Top-to-Bottom", linewidth=1.5
    )
    axes[2].plot(
        epochs, results_b2t["train_accs"], label="Bottom-to-Top", linewidth=1.5
    )
    axes[2].set_title("Training Accuracy Comparison", fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        "CIFAR-100 Unsupervised Ranking: Top-to-Bottom vs Bottom-to-Top Curriculum",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {save_path}")


# ============================================================
# Single Experiment Runner
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
):
    """
    Run one complete training experiment with the given curriculum order.

    Args:
        direction_label: "top_to_bottom" or "bottom_to_top"
        curriculum_order: list of image indices in presentation order
        (remaining args: data, model config, hyperparameters)

    Returns:
        dict with model, metrics, and summary DataFrame
    """
    print(f"\n{'=' * 60}")
    print(f"Setting Up Curriculum: {direction_label}")
    print(f"{'=' * 60}")

    curriculum_sampler = CurriculumSampler(
        sorted_indices=curriculum_order,
        total_epochs=total_epochs,
        start_fraction=start_fraction,
        full_data_epoch=full_data_epoch,
    )

    # Create fresh model
    model = ResNet18(num_classes=100).to(device)
    total_params, trainable_params = calculate_model_parameters(model)

    try:
        flops = calculate_flops(model)
    except Exception:
        print("Skipping FLOPs calculation.")
        flops = None

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)

    # Training loop
    print(f"\n{'=' * 60}")
    print(f"Training: {direction_label}")
    print(f"{'=' * 60}")

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epoch_times = []
    samples_per_epoch = []

    best_val_acc = 0
    best_model_state = None
    total_training_start = time.time()

    for epoch in range(1, total_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"[{direction_label}] EPOCH {epoch}/{total_epochs}")
        print(f"{'=' * 50}")

        curriculum_sampler.set_epoch(epoch)
        num_samples_this_epoch = len(curriculum_sampler)
        fraction_used = num_samples_this_epoch / total_train_samples * 100
        print(
            f"Curriculum: using {num_samples_this_epoch}/{total_train_samples} samples ({fraction_used:.1f}%)"
        )
        samples_per_epoch.append(num_samples_this_epoch)

        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            sampler=curriculum_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

        train_loss, train_acc, epoch_time, total_images = train_epoch(
            model, device, train_loader, optimizer, epoch, scheduler=scheduler
        )

        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_loss, val_acc, _, _ = validate(
            model, device, val_loader, dataset_name="Validation"
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            torch.save(
                model.state_dict(), f"best_resnet18_{direction_label}_c100_gradus.pth"
            )
            print(f"  ** New best validation accuracy: {best_val_acc:.2f}% **")

    # -------------------------------------------------------
    # Final Evaluation
    # -------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Final Evaluation: {direction_label}")
    print(f"{'=' * 60}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model (val acc: {best_val_acc:.2f}%)")

    total_training_time = time.time() - total_training_start

    print(f"\nFinal Validation Set Evaluation:")
    final_val_loss, final_val_acc, _, _ = validate(
        model, device, val_loader, dataset_name="Validation"
    )

    print(f"\nFinal Test Set Evaluation:")
    final_test_loss, final_test_acc, test_targets, test_preds = validate(
        model, device, test_loader, dataset_name="Test"
    )

    # -------------------------------------------------------
    # Plots
    # -------------------------------------------------------
    plot_training_metrics(
        epoch_times,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        save_path=f"training_metrics_{direction_label}_c100_gradus.pdf",
    )
    plot_confusion_matrix(
        test_targets,
        test_preds,
        class_names,
        title=f"Test Confusion Matrix ({direction_label})",
        save_path=f"confusion_matrix_{direction_label}_c100_gradus.pdf",
    )
    plot_curriculum_schedule(
        total_epochs,
        start_fraction,
        full_data_epoch,
        total_train_samples,
        samples_per_epoch,
        save_path=f"curriculum_schedule_{direction_label}_c100_gradus.pdf",
    )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------
    total_samples_processed = sum(samples_per_epoch)
    max_possible_samples = total_epochs * total_train_samples
    data_utilization_index = total_samples_processed / max_possible_samples
    data_savings_index = 1 - data_utilization_index
    avg_epoch_time = np.mean(epoch_times)

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {direction_label}")
    print(f"{'=' * 60}")
    print(
        f"Total training time: {total_training_time:.1f}s ({total_training_time / 60:.1f} min)"
    )
    print(f"Average epoch time: {avg_epoch_time:.1f}s")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"Data Utilization Index (DUI): {data_utilization_index:.4f}")
    print(f"Data Savings Index (DSI): {data_savings_index:.4f}")
    print(f"Total samples processed: {total_samples_processed:,}")
    print(f"Max possible samples: {max_possible_samples:,}")
    print(f"Total parameters: {total_params:,}")

    summary = {
        "direction": [direction_label],
        "total_training_time": [total_training_time],
        "average_epoch_time": [avg_epoch_time],
        "epochs_completed": [total_epochs],
        "best_validation_accuracy": [best_val_acc],
        "final_validation_accuracy": [final_val_acc],
        "final_test_accuracy": [final_test_acc],
        "total_parameters": [total_params],
        "data_utilization_index": [data_utilization_index],
        "data_savings_index": [data_savings_index],
        "total_samples_processed": [total_samples_processed],
        "curriculum_start_fraction": [start_fraction],
        "curriculum_full_data_epoch": [full_data_epoch],
        "learning_rate": [learning_rate],
        "weight_decay": [weight_decay],
        "batch_size": [batch_size],
        "train_samples": [total_train_samples],
        "val_samples": [5000],
        "test_samples": [10000],
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        f"curriculum_summary_{direction_label}_c100_gradus.csv", index=False
    )
    print(f"Summary saved to curriculum_summary_{direction_label}_c100_gradus.csv")

    epoch_df = pd.DataFrame(
        {
            "epoch": range(1, total_epochs + 1),
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": val_losses,
            "val_acc": val_accs,
            "epoch_time": epoch_times,
            "samples_used": samples_per_epoch,
        }
    )
    epoch_df.to_csv(f"epoch_history_{direction_label}_c100_gradus.csv", index=False)
    print(f"Epoch history saved to epoch_history_{direction_label}_c100_gradus.csv")

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
        "final_test_acc": final_test_acc,
        "total_training_time": total_training_time,
    }


# ============================================================
# Main: Curriculum Learning with Unsupervised Ranking
# ============================================================


def run_cifar100_curriculum_learning():
    device = configure_gpu()

    # Hyperparameters
    total_epochs = 200
    batch_size = 128
    learning_rate = 0.1
    weight_decay = 5e-4
    momentum = 0.9

    # Curriculum learning parameters
    start_fraction = 0.3  # Start with 30% of data
    full_data_epoch = 120  # Use all data by epoch 120

    # Load data
    full_train_dataset, val_loader, test_loader, class_names, train_indices = (
        load_images(batch_size, val_size=5000)
    )

    total_train_samples = len(train_indices)
    print(f"\nTotal training samples: {total_train_samples}")

    # -------------------------------------------------------
    # Step 1: Compute image complexity for ALL training images
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Computing Image Complexity Features (Gradus Metrics)")
    print("=" * 60)

    features_df = compute_all_complexity_features(full_train_dataset, device)

    features_df.to_csv("image_complexity_features_c100_gradus.csv", index=True)
    print("Saved complexity features to image_complexity_features_c100_gradus.csv")

    plot_complexity_distribution(
        features_df, save_path="complexity_distribution_c100_gradus.pdf"
    )

    # -------------------------------------------------------
    # Step 2: Unsupervised correlation-based ranking (individual images)
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Unsupervised Correlation-Based Ranking (Per-Image)")
    print("=" * 60)

    top_to_bottom, bottom_to_top, weights = rank_images_unsupervised(
        features_df, train_indices
    )

    print(f"\nTop-to-bottom curriculum: first 5 indices = {top_to_bottom[:5]}")
    print(f"Top-to-bottom curriculum: last  5 indices = {top_to_bottom[-5:]}")
    print(f"Bottom-to-top curriculum: first 5 indices = {bottom_to_top[:5]}")
    print(f"Bottom-to-top curriculum: last  5 indices = {bottom_to_top[-5:]}")

    # Save per-image ranking weights (aligned with train_indices order)
    ranking_df = pd.DataFrame(
        {
            "dataset_index": train_indices,
            "weight": weights,
        }
    )
    ranking_df.to_csv("unsupervised_ranking_weights_c100_gradus.csv", index=False)
    print(
        "Saved per-image ranking weights to unsupervised_ranking_weights_c100_gradus.csv"
    )

    # -------------------------------------------------------
    # Step 3: Experiment 1 — Top-to-Bottom Curriculum
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TOP-TO-BOTTOM CURRICULUM (highest-ranked images first)")
    print("=" * 70)

    results_t2b = run_single_experiment(
        "top_to_bottom",
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
    )

    # -------------------------------------------------------
    # Step 4: Experiment 2 — Bottom-to-Top Curriculum
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BOTTOM-TO-TOP CURRICULUM (lowest-ranked images first)")
    print("=" * 70)

    results_b2t = run_single_experiment(
        "bottom_to_top",
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
    )

    # -------------------------------------------------------
    # Step 5: Comparison
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON: Top-to-Bottom vs Bottom-to-Top")
    print("=" * 70)

    plot_comparison(results_t2b, results_b2t, save_path="comparison_c100_gradus.pdf")

    print(
        f"\n{'Direction':<20} {'Best Val Acc':>12} {'Test Acc':>10} {'Train Time':>12}"
    )
    print(f"{'-' * 56}")
    print(
        f"{'Top-to-Bottom':<20} {results_t2b['best_val_acc']:>11.2f}% {results_t2b['final_test_acc']:>9.2f}% {results_t2b['total_training_time']:>10.1f}s"
    )
    print(
        f"{'Bottom-to-Top':<20} {results_b2t['best_val_acc']:>11.2f}% {results_b2t['final_test_acc']:>9.2f}% {results_b2t['total_training_time']:>10.1f}s"
    )

    # Save combined comparison
    comparison_df = pd.concat(
        [results_t2b["summary_df"], results_b2t["summary_df"]], ignore_index=True
    )
    comparison_df.to_csv("curriculum_comparison_c100_gradus.csv", index=False)
    print("\nComparison saved to curriculum_comparison_c100_gradus.csv")

    return results_t2b["model"], results_b2t["model"], comparison_df


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_t2b, model_b2t, comparison_df = run_cifar100_curriculum_learning()
