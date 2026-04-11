import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from thop import profile
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
# Data Loading
# ============================================================


def load_images(batch_size=128, val_size=5000):
    """
    Loads CIFAR-10 with train/val/test split.

    Identical split logic to the curriculum scripts for a fair comparison:
      - 45,000 training samples
      -  5,000 validation samples (held-out from the 50 k training set)
      - 10,000 test samples (official CIFAR-10 test set)

    Returns:
        full_train_dataset : augmented training dataset (RandomCrop + HFlip)
        val_loader         : validation DataLoader (no augmentation)
        test_loader        : test DataLoader (no augmentation)
        class_names        : list of 10 class name strings
        train_indices      : numpy array of the 45 k training indices
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

    # Shuffle all 50 k indices, then split — mirrors curriculum scripts exactly
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
# ResNet-18 Model  (identical to curriculum scripts)
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
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
    print(f"\nModel Computational Requirements:")
    print(f"  FLOPs: {flops:,} ({flops / 1e9:.2f} GFLOPs)")
    return flops


# ============================================================
# Training & Evaluation
# ============================================================


def train_epoch(model, device, train_loader, optimizer, epoch, scheduler=None):
    """Train for one epoch with Mixup augmentation (same as curriculum scripts)."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    total_images = 0
    start_time = time.time()
    num_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Mixup augmentation — 50 % chance per batch
        if np.random.random() > 0.5:
            lam = np.random.beta(0.2, 0.2)
            rand_idx = torch.randperm(data.size(0)).to(device)
            mixed = lam * data + (1 - lam) * data[rand_idx]
            ta, tb = target, target[rand_idx]
            optimizer.zero_grad()
            output = model(mixed)
            loss = lam * F.cross_entropy(output, ta) + (1 - lam) * F.cross_entropy(
                output, tb
            )
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
    img_per_s = total_images / epoch_time if epoch_time > 0 else 0

    print(
        f"  Epoch {epoch} done in {epoch_time:.1f}s | Loss: {epoch_loss:.4f} | "
        f"Acc: {epoch_acc:.2f}% | {img_per_s:.0f} img/s | "
        f"Samples used: {total_images}"
    )

    return epoch_loss, epoch_acc, epoch_time, total_images


def validate(model, device, data_loader, dataset_name="Validation"):
    """Evaluate model on a given DataLoader."""
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


def plot_training_metrics(
    epoch_times,
    train_losses,
    train_accs,
    val_losses,
    val_accs,
    save_path="training_metrics_baseline.pdf",
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
    save_path="confusion_matrix_baseline.pdf",
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


# ============================================================
# Main Baseline Training Loop
# ============================================================


def run_cifar10_baseline():
    device = configure_gpu()

    # ----------------------------------------------------------
    # Hyperparameters — identical to curriculum scripts
    # ----------------------------------------------------------
    total_epochs = 200
    batch_size = 128
    learning_rate = 0.05
    weight_decay = 5e-4
    momentum = 0.9

    # ----------------------------------------------------------
    # Step 1: Load data with the same split as curriculum scripts
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data (same split as curriculum scripts)")
    print("=" * 60)

    full_train_dataset, val_loader, test_loader, class_names, train_indices = (
        load_images(batch_size, val_size=5000)
    )

    total_train_samples = len(train_indices)
    print(f"\nTotal training samples: {total_train_samples}")

    # Build a fixed training subset using the same 45 k indices.
    # shuffle=True gives standard random re-ordering every epoch —
    # no curriculum ordering whatsoever.
    train_subset = Subset(full_train_dataset, train_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"Training DataLoader: {len(train_loader)} batches/epoch "
        f"(shuffle=True, batch_size={batch_size})"
    )

    # ----------------------------------------------------------
    # Step 2: Build model, optimiser, scheduler
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Building ResNet-18 Model")
    print("=" * 60)

    model = ResNet18(num_classes=10).to(device)
    print(model)

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

    # ----------------------------------------------------------
    # Step 3: Standard training loop
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Standard Training (random shuffle, no curriculum)")
    print("=" * 60)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epoch_times = []
    samples_per_epoch = []

    best_val_acc = 0.0
    best_model_state = None
    total_training_start = time.time()

    for epoch in range(1, total_epochs + 1):
        print(f"\n{'=' * 50}")
        print(f"EPOCH {epoch}/{total_epochs}")
        print(f"{'=' * 50}")
        print(f"Baseline: using all {total_train_samples} samples (randomly shuffled)")

        train_loss, train_acc, epoch_time, total_images = train_epoch(
            model, device, train_loader, optimizer, epoch, scheduler=scheduler
        )

        epoch_times.append(epoch_time)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        samples_per_epoch.append(total_images)

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
            torch.save(model.state_dict(), "best_resnet18_baseline.pth")
            print(f"  ** New best validation accuracy: {best_val_acc:.2f}% **")

    # ----------------------------------------------------------
    # Step 4: Final evaluation on best checkpoint
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Final Evaluation")
    print("=" * 60)

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

    # ----------------------------------------------------------
    # Step 5: Plots and summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Generating Plots and Summary")
    print("=" * 60)

    plot_training_metrics(
        epoch_times,
        train_losses,
        train_accs,
        val_losses,
        val_accs,
        save_path="training_metrics_baseline.pdf",
    )
    plot_confusion_matrix(
        test_targets,
        test_preds,
        class_names,
        title="Test Set Confusion Matrix (Baseline - Standard Training)",
        save_path="confusion_matrix_baseline.pdf",
    )

    # Data utilisation metrics (always 1.0 / 0.0 for baseline —
    # included so the summary CSV is directly comparable with curriculum CSVs)
    total_samples_processed = sum(samples_per_epoch)
    max_possible_samples = total_epochs * total_train_samples
    data_utilization_index = total_samples_processed / max_possible_samples
    data_savings_index = 1.0 - data_utilization_index

    avg_epoch_time = np.mean(epoch_times)

    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"Total training time    : {total_training_time:.1f}s "
        f"({total_training_time / 60:.1f} min)"
    )
    print(f"Average epoch time     : {avg_epoch_time:.1f}s")
    print(f"Best validation acc    : {best_val_acc:.2f}%")
    print(f"Final validation acc   : {final_val_acc:.2f}%")
    print(f"Final test accuracy    : {final_test_acc:.2f}%")
    print(f"Data Utilization Index : {data_utilization_index:.4f}")
    print(f"Data Savings Index     : {data_savings_index:.4f}")
    print(f"Total samples processed: {total_samples_processed:,}")
    print(f"Max possible samples   : {max_possible_samples:,}")
    print(f"Total parameters       : {total_params:,}")

    # Save summary CSV
    summary = {
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
        "learning_rate": [learning_rate],
        "weight_decay": [weight_decay],
        "batch_size": [batch_size],
        "train_samples": [total_train_samples],
        "val_samples": [5000],
        "test_samples": [10000],
        "curriculum_type": ["baseline_random_shuffle"],
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("baseline_learning_summary.csv", index=False)
    print("\nSummary saved to baseline_learning_summary.csv")

    # Save per-epoch history
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
    epoch_df.to_csv("baseline_epoch_history.csv", index=False)
    print("Epoch history saved to baseline_epoch_history.csv")

    return model, summary_df


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    # Fix seeds for reproducibility — same values as curriculum scripts
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, summary_df = run_cifar10_baseline()
