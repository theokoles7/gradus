# Adaptive Pacing Plan for CIFAR-10 Curriculum Learning

## Problem Statement

The current `CurriculumSampler` in `cifar10/c10_gradus_unsupervised_ranking.py` uses a **fixed linear schedule**: starts at 30% of data and linearly ramps to 100% by epoch 120. This ignores how well the model is actually learning. If the model masters the current data quickly, it wastes epochs. If it struggles, it gets flooded with harder data too soon.

**Goal**: Replace the fixed pacing with an **adaptive pacing function** that monitors four training signals — **loss plateau**, **validation accuracy trend**, **activation stability**, and **gradient norm stability** — to decide *when* and *how much* new data to introduce. Minimize hardcoded thresholds; derive decisions from the data itself.

---

## Current Architecture (What We're Replacing)

### Fixed pacing in `CurriculumSampler._update_indices()` (lines 429–439)
```python
fraction = start_fraction + (1.0 - start_fraction) * (epoch / full_data_epoch)
```

### Training loop (lines 894–916)
- Calls `curriculum_sampler.set_epoch(epoch)` — purely epoch-driven.
- `train_epoch()` does NOT return activations — the current ResNet model lacks `return_activations` support.
- No feedback from training metrics to the sampler.

### What bpas.py does differently (reference implementation)
- ResNet model supports `forward(x, return_activations=True)` → returns intermediate activations from layer1–layer4.
- Per batch: computes `torch.std(activation)` for each layer → `mean_std` across layers.
- Compares `mean_std` between consecutive epochs using a delta threshold to decide if a batch is "stable".
- **Our change**: instead of a handcrafted delta, detect stability via **plateau detection** on the per-batch activation time series.

---

## Proposed Architecture

### Overview

```
┌─────────────────────────────────────────────────┐
│             Unsupervised Ranking                 │
│  sorted_indices = [img_0, img_1, ..., img_N]    │
│  (ranked easiest → hardest by correlation algo)  │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│           Group into Ordered Batches             │
│  batch_0 = sorted_indices[0:128]     (easiest)   │
│  batch_1 = sorted_indices[128:256]               │
│  ...                                             │
│  batch_K = sorted_indices[-128:]     (hardest)   │
│  Total: ~352 batches for 45,000 images           │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│         Adaptive Pacing Controller               │
│                                                  │
│  Epoch 1: activate batch_0 ... batch_105 (30%)   │
│  Epoch 2: train → collect signals → readiness    │
│           → add N more batches from queue        │
│  ...                                             │
│  Epoch T: all 352 batches active                 │
└─────────────────────────────────────────────────┘
```

**Unit of data introduction = batches (not individual images).** This aligns activation tracking (per batch) with the pacing decision (add N batches).

---

## Step-by-Step Implementation

### Step 1: Modify ResNet to Support Activation Extraction

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: Modify the existing `ResNet` class to optionally return intermediate activations, matching the bpas.py pattern.

**Changes to `ResNet` class (lines 531–563)**:

1. Add `self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']` in `__init__`.
2. Change `forward()` signature to `forward(self, x, return_activations=False)`.
3. When `return_activations=True`, collect `out.detach()` after each of `self.layer1`–`self.layer4`.
4. Return `(output, activations_list)` when requested, else just `output`.

```python
def forward(self, x, return_activations=False):
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
```

---

### Step 2: Group Ranked Images into Ordered Batches

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: New helper function to partition the ranked image indices into sequential batches.

```python
def create_ordered_batches(sorted_indices, batch_size):
    """
    Partition sorted image indices into sequential batches.

    Args:
        sorted_indices: list of dataset indices in curriculum order
                        (e.g., easiest-first from unsupervised ranking)
        batch_size: number of images per batch

    Returns:
        batches: list of lists, where batches[i] contains the image
                 indices for the i-th batch in curriculum order.
                 batch 0 = easiest, batch K = hardest.
    """
    batches = []
    for i in range(0, len(sorted_indices), batch_size):
        batches.append(sorted_indices[i:i + batch_size])
    return batches
```

**Called in main** right after `rank_images_unsupervised()` returns `top_to_bottom` / `bottom_to_top`:
```python
ordered_batches = create_ordered_batches(top_to_bottom, batch_size)
# ordered_batches[0] = easiest 128 images
# ordered_batches[-1] = hardest 128 images
```

---

### Step 3: Modify `train_epoch()` to Collect Per-Batch Activation Stds AND Gradient Norms

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: Update `train_epoch()` to:
1. Call `model(data, return_activations=True)`.
2. Compute `torch.std(activation).item()` for each of the 4 layers.
3. After `loss.backward()`, compute the L2 gradient norm for each layer's parameters.
4. Track which logical batch (in curriculum order) each mini-batch belongs to.
5. Return DataFrames for both per-batch activation stds and per-batch gradient norms.

**Current signature** (line 592):
```python
def train_epoch(model, device, train_loader, optimizer, epoch, scheduler=None)
```

**New signature**:
```python
def train_epoch(model, device, train_loader, optimizer, epoch,
                batch_index_order=None, scheduler=None)
```

- `batch_index_order`: list of logical batch indices corresponding to each mini-batch in the loader (so we know which curriculum batch each iteration belongs to). `None` for backward-compat.

**Key changes inside the function**:
```python
# At the top:
layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
std_devs = {name: [] for name in layer_names}
grad_norms = {name: [] for name in layer_names}
batch_indices_list = []

# Map layer names to actual layer modules (for gradient access)
layer_modules = {
    'layer1': model.layer1,
    'layer2': model.layer2,
    'layer3': model.layer3,
    'layer4': model.layer4,
}

# In the training loop, replace model(data) / model(mixed_data) with:
output, activations = model(data, return_activations=True)
# or
output, activations = model(mixed_data, return_activations=True)

# After getting activations (forward pass stats):
for layer_idx, act in enumerate(activations):
    std_devs[layer_names[layer_idx]].append(torch.std(act).item())

# After loss.backward(), BEFORE optimizer.step() and BEFORE grad clipping:
# Compute per-layer gradient L2 norms from the weights
loss.backward()

for name in layer_names:
    layer = layer_modules[name]
    sq_sum = 0.0
    for p in layer.parameters():
        if p.grad is not None:
            sq_sum += p.grad.data.norm(2).item() ** 2
    grad_norms[name].append(sq_sum ** 0.5)  # L2 norm for this layer

# THEN clip and step:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

if batch_index_order is not None:
    batch_indices_list.append(batch_index_order[batch_idx])

# At the end, build and return BOTH DataFrames:

# Activation std DataFrame
std_df = pd.DataFrame(std_devs)
std_df['batch_idx'] = batch_indices_list
std_df = std_df.set_index('batch_idx')
std_df['mean_std'] = std_df.mean(axis=1)

# Gradient norm DataFrame
grad_norm_df = pd.DataFrame(grad_norms)
grad_norm_df['batch_idx'] = batch_indices_list
grad_norm_df = grad_norm_df.set_index('batch_idx')
grad_norm_df['mean_grad_norm'] = grad_norm_df.mean(axis=1)
```

**Why compute gradient norms BEFORE clipping**: Clipping caps the total norm at `max_norm=1.0`, which would collapse all batches to the same ceiling value and destroy the per-batch signal. Raw (pre-clip) gradient norms reflect the true learning pressure each batch exerts on the model.

**Why per-layer**: Different layers learn at different rates (early layers converge faster). Per-layer norms let us detect when ALL layers have converged for a batch, not just the average. The `mean_grad_norm` column aggregates this, but individual layer columns are preserved for analysis.

**Updated return**: `(epoch_loss, epoch_acc, epoch_time, total_images, std_df, grad_norm_df)`

---

### Step 4: Build the Data Loader from Active Batches

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: New function to create a DataLoader from a subset of ordered batches.

```python
def build_curriculum_loader(full_train_dataset, ordered_batches,
                            active_batch_indices, batch_size,
                            shuffle_within_batch=True):
    """
    Build a DataLoader that iterates over only the active batches
    in curriculum order.

    Args:
        full_train_dataset: the full CIFAR-10 training dataset
        ordered_batches: list of lists (output of create_ordered_batches)
        active_batch_indices: list of ints — which batches are active
                              (e.g., [0, 1, 2, ..., 105] for 30%)
        batch_size: mini-batch size for the DataLoader
        shuffle_within_batch: if True, shuffle image order within each
                              batch (prevents memorization of order)

    Returns:
        train_loader: DataLoader iterating active batches in order
        batch_index_order: list mapping each loader iteration to a
                           logical batch index (for activation tracking)
    """
    flat_indices = []
    batch_index_order = []

    for batch_idx in active_batch_indices:
        sample_indices = list(ordered_batches[batch_idx])
        if shuffle_within_batch:
            np.random.shuffle(sample_indices)
        flat_indices.extend(sample_indices)
        batch_index_order.append(batch_idx)

    class OrderedIndexSampler:
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
        drop_last=False
    )

    return train_loader, batch_index_order
```

---

### Step 5: Implement Plateau Detection (Self-Calibrating)

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: A function that determines whether a time series has plateaued, using only the data itself — no hardcoded delta threshold.

**Design principle**: Compare *recent variation* to *historical variation*. If recent variation is a small fraction of what the signal showed during its "active change" phase, it's a plateau.

```python
def detect_plateau(values, window):
    """
    Determine whether a time series has plateaued.

    Uses self-calibrating comparison: the coefficient of variation (CV)
    over the recent `window` epochs is compared to the CV over the full
    history. If recent CV is much smaller than historical CV, the series
    has plateaued.

    Args:
        values: list of floats — the full time series for this batch
        window: int — how many recent epochs to consider

    Returns:
        is_plateau: bool — True if the series has plateaued
        stability_score: float in [0, 1] — 0 = actively changing,
                         1 = completely flat
    """
    if len(values) < window + 2:
        # Not enough history — assume NOT plateaued
        return False, 0.0

    recent = values[-window:]
    full = values[:]

    mean_recent = np.mean(recent)
    std_recent = np.std(recent)
    mean_full = np.mean(full)
    std_full = np.std(full)

    # Coefficient of variation: relative spread
    cv_recent = std_recent / (abs(mean_recent) + 1e-8)
    cv_full = std_full / (abs(mean_full) + 1e-8)

    if cv_full < 1e-8:
        # The entire history is flat — consider it plateaued
        return True, 1.0

    # Ratio: how much of the historical variation is still present
    # in the recent window. Low ratio = plateau.
    ratio = cv_recent / cv_full

    # stability_score: 1 when ratio → 0 (flat), 0 when ratio → 1+ (active)
    stability_score = max(0.0, 1.0 - ratio)

    # Plateau if the recent variation is less than 25% of historical
    # (0.25 is a mild assumption — it means the signal has settled to
    #  less than a quarter of its historical fluctuation range)
    is_plateau = ratio < 0.25

    return is_plateau, stability_score
```

**Why this avoids handcrafted deltas**: The decision threshold is *relative* — it compares the signal to itself. The same code works whether activation stds are in the range [0.01, 0.02] or [5.0, 10.0]. The `0.25` ratio is a structural choice ("recent variation < 25% of historical"), not a scale-dependent magic number.

---

### Step 6: Implement the Four Readiness Signals

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: Four functions that each return a signal in [0, 1] where 1 = "ready for more data".

#### Signal 1: Loss Plateau

```python
def compute_loss_plateau_signal(loss_history, window):
    """
    Returns a score in [0, 1]. High = loss has stopped improving.

    Approach:
        Compare mean loss in the recent `window` epochs vs the
        previous `window` epochs. If the recent mean is not
        meaningfully lower, the loss has plateaued.

    Uses relative comparison (recent vs prior), no absolute threshold.
    """
    if len(loss_history) < 2 * window:
        return 0.0  # not enough history, assume not plateaued

    prior = loss_history[-(2 * window):-window]
    recent = loss_history[-window:]

    mean_prior = np.mean(prior)
    mean_recent = np.mean(recent)

    if mean_prior < 1e-8:
        return 1.0  # loss already near zero

    # Relative improvement: positive = loss got worse (or stagnated)
    relative_change = (mean_recent - mean_prior) / mean_prior

    # Map to [0, 1]:
    #   relative_change << 0 → still improving → signal ≈ 0
    #   relative_change ≈ 0  → plateau → signal ≈ 0.5–1.0
    #   relative_change > 0  → loss worsening → signal = 1.0
    signal = 1.0 / (1.0 + np.exp(-10 * relative_change))
    # The sigmoid steepness (10) just controls how sharply the
    # transition happens around 0. It is not scale-dependent.

    return float(signal)
```

#### Signal 2: Validation Accuracy Trend

```python
def compute_val_acc_trend_signal(val_acc_history, window):
    """
    Returns a score in [0, 1]. High = val accuracy has plateaued or is declining.

    Approach:
        Fit a linear regression over the last `window` val accuracies.
        Normalize the slope by the standard deviation of the values.
        A near-zero or negative slope indicates readiness.
    """
    if len(val_acc_history) < window:
        return 0.0

    recent = val_acc_history[-window:]
    x = np.arange(len(recent), dtype=float)
    slope = np.polyfit(x, recent, 1)[0]  # slope of linear fit

    # Normalize slope by the std of values (makes it scale-independent)
    std_val = np.std(recent)
    if std_val < 1e-8:
        # Perfectly flat → plateau
        return 1.0

    normalized_slope = slope / std_val

    # Map: positive slope (improving) → low signal
    #       zero slope (plateau) → high signal
    #       negative slope (declining) → high signal
    signal = 1.0 / (1.0 + np.exp(5 * normalized_slope))

    return float(signal)
```

#### Signal 3: Activation Stability

```python
def compute_activation_stability_signal(batch_std_history, active_batch_indices, window):
    """
    Returns a score in [0, 1]. High = most active batches have stabilized.

    Args:
        batch_std_history: dict {batch_idx: [mean_std_epoch1, mean_std_epoch2, ...]}
        active_batch_indices: list of currently active batch indices
        window: plateau detection window size

    Approach:
        For each active batch, run detect_plateau() on its activation
        std time series. The signal is the fraction of active batches
        that have plateaued. Weighted by each batch's stability_score
        for a smoother signal.
    """
    if not active_batch_indices:
        return 0.0

    total_score = 0.0
    count = 0

    for batch_idx in active_batch_indices:
        if batch_idx not in batch_std_history:
            continue
        values = batch_std_history[batch_idx]
        _, stability_score = detect_plateau(values, window)
        total_score += stability_score
        count += 1

    if count == 0:
        return 0.0

    return total_score / count
```

#### Signal 4: Gradient Norm Stability

**Why this is distinct from activation stability**:
- **Activation stability** (Signal 3) measures the forward pass: "are the model's internal representations stable for this batch?" When activations plateau, the model consistently maps the same input to the same intermediate features.
- **Gradient norm stability** (Signal 4) measures the backward pass: "is this batch still causing significant parameter updates?" When gradient norms plateau at low values, the batch has been fully absorbed into the model's weights — it's no longer teaching anything new.

These can diverge: a batch might have stable activations (consistent features) but still produce non-trivial gradients (the classifier head is still adjusting decision boundaries). Conversely, gradients might vanish for easy batches while activations for harder batches are still shifting. Using both gives a more complete picture of convergence.

```python
def compute_gradient_norm_signal(batch_grad_norm_history, active_batch_indices, window):
    """
    Returns a score in [0, 1]. High = gradient norms for active batches
    have plateaued (the model is no longer learning much from them).

    Args:
        batch_grad_norm_history: dict {batch_idx: [mean_grad_norm_ep1, ...]}
            Per-batch mean gradient norm (mean across layer1–layer4)
            tracked over epochs.
        active_batch_indices: list of currently active batch indices
        window: plateau detection window size

    Approach:
        Identical structure to compute_activation_stability_signal().
        For each active batch, run detect_plateau() on its gradient
        norm time series. Return the weighted-average stability_score
        across all active batches.

    Why the same approach works:
        detect_plateau() is scale-invariant (uses CV ratios). Gradient
        norms and activation stds live on different scales, but the
        plateau detection logic doesn't care — it only compares recent
        variation to historical variation within each batch's own series.
    """
    if not active_batch_indices:
        return 0.0

    total_score = 0.0
    count = 0

    for batch_idx in active_batch_indices:
        if batch_idx not in batch_grad_norm_history:
            continue
        values = batch_grad_norm_history[batch_idx]
        _, stability_score = detect_plateau(values, window)
        total_score += stability_score
        count += 1

    if count == 0:
        return 0.0

    return total_score / count
```

---

### Step 7: Implement `AdaptiveCurriculumPacer`

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: The central controller that combines the four readiness signals and decides how many batches to add.

```python
class AdaptiveCurriculumPacer:
    """
    Controls the pace of data introduction in curriculum learning.

    After each epoch, the pacer receives training metrics and activation
    data, computes a readiness score from four signals, and decides
    how many new batches to add.

    The pacer guarantees:
    - Data is introduced gradually (at least 1 batch per epoch)
    - All data is used by `total_epochs` (hard deadline)
    - The rate of introduction adapts to the model's learning dynamics

    Readiness is computed from four signals:
    1. Loss plateau (training loss stopped improving)
    2. Validation accuracy trend (val acc plateaued or declining)
    3. Activation stability (per-batch activation stds plateaued)
    4. Gradient norm stability (per-batch gradient norms plateaued)
    """

    def __init__(self, total_batches, total_epochs, start_fraction=0.3,
                 window=5):
        """
        Args:
            total_batches: total number of curriculum batches
            total_epochs: total training epochs
            start_fraction: fraction of batches to start with
            window: lookback window for plateau detection and trend
                    analysis (used for all four signals)
        """
        self.total_batches = total_batches
        self.total_epochs = total_epochs
        self.window = window

        # Initialize active batches
        initial_count = max(1, int(total_batches * start_fraction))
        self.active_batch_indices = list(range(initial_count))
        self.next_batch_to_add = initial_count

        # Compute minimum batches to add per epoch to guarantee
        # all data is included by the last epoch.
        # This replaces a hardcoded `max_epoch_for_full_data`.
        remaining_batches = total_batches - initial_count
        self.min_batches_per_epoch = max(1, remaining_batches // total_epochs)

        # History for signal computation
        self.loss_history = []
        self.val_acc_history = []
        self.batch_std_history = {}       # {batch_idx: [mean_std_ep1, ...]}
        self.batch_grad_norm_history = {} # {batch_idx: [mean_grad_norm_ep1, ...]}

        # Logging
        self.pacing_log = []  # stores per-epoch pacing decisions

    def update_metrics(self, train_loss, val_acc, std_df, grad_norm_df):
        """
        Called after each epoch with the epoch's training metrics,
        per-batch activation std DataFrame, and per-batch gradient
        norm DataFrame.

        Args:
            train_loss: float — average training loss this epoch
            val_acc: float — validation accuracy this epoch
            std_df: pd.DataFrame — indexed by batch_idx, with column
                    'mean_std' (mean activation std across 4 layers)
            grad_norm_df: pd.DataFrame — indexed by batch_idx, with
                    column 'mean_grad_norm' (mean gradient L2 norm
                    across 4 layers)
        """
        self.loss_history.append(train_loss)
        self.val_acc_history.append(val_acc)

        # Update per-batch activation std history
        for batch_idx in std_df.index:
            if batch_idx not in self.batch_std_history:
                self.batch_std_history[batch_idx] = []
            self.batch_std_history[batch_idx].append(
                std_df.loc[batch_idx, 'mean_std']
            )

        # Update per-batch gradient norm history
        for batch_idx in grad_norm_df.index:
            if batch_idx not in self.batch_grad_norm_history:
                self.batch_grad_norm_history[batch_idx] = []
            self.batch_grad_norm_history[batch_idx].append(
                grad_norm_df.loc[batch_idx, 'mean_grad_norm']
            )

    def compute_readiness(self):
        """
        Compute the composite readiness score from four signals.

        Returns:
            readiness: float in [0, 1]
            signal_details: dict with individual signal values
        """
        r_loss = compute_loss_plateau_signal(
            self.loss_history, self.window
        )
        r_val = compute_val_acc_trend_signal(
            self.val_acc_history, self.window
        )
        r_act = compute_activation_stability_signal(
            self.batch_std_history, self.active_batch_indices, self.window
        )
        r_grad = compute_gradient_norm_signal(
            self.batch_grad_norm_history, self.active_batch_indices,
            self.window
        )

        # Equal-weight average of the four signals
        readiness = (r_loss + r_val + r_act + r_grad) / 4.0

        return readiness, {
            'loss_plateau': r_loss,
            'val_acc_trend': r_val,
            'activation_stability': r_act,
            'gradient_norm_stability': r_grad
        }

    def step(self, epoch):
        """
        Decide how many batches to add and update active set.
        Called AFTER update_metrics() each epoch.

        Args:
            epoch: current epoch number (1-indexed)

        Returns:
            active_batch_indices: updated list of active batch indices
            pacing_info: dict with decision details for logging
        """
        if self.next_batch_to_add >= self.total_batches:
            # All batches already active
            pacing_info = {
                'epoch': epoch,
                'readiness': 1.0,
                'signals': {},
                'batches_added': 0,
                'active_batches': len(self.active_batch_indices),
                'fraction': 1.0
            }
            self.pacing_log.append(pacing_info)
            return self.active_batch_indices, pacing_info

        readiness, signals = self.compute_readiness()

        # --- Determine how many batches to add ---
        remaining_batches = self.total_batches - self.next_batch_to_add
        remaining_epochs = self.total_epochs - epoch

        # Minimum: enough to guarantee all data by the end
        min_to_add = self.min_batches_per_epoch
        if remaining_epochs > 0:
            # Recalculate based on what's actually left
            min_to_add = max(1, math.ceil(
                remaining_batches / remaining_epochs
            ))

        # Maximum: scale with readiness.
        # At readiness=0: add only min_to_add
        # At readiness=1: add up to 3x min_to_add
        # (The 3x multiplier is structural, not scale-dependent.
        #  It just means high readiness can triple the pace.)
        max_to_add = min(remaining_batches, 3 * min_to_add)

        batches_to_add = int(
            min_to_add + readiness * (max_to_add - min_to_add)
        )
        batches_to_add = max(1, min(batches_to_add, remaining_batches))

        # Add the next N batches from the ranked queue
        new_indices = list(range(
            self.next_batch_to_add,
            self.next_batch_to_add + batches_to_add
        ))
        self.active_batch_indices.extend(new_indices)
        self.next_batch_to_add += batches_to_add

        fraction = len(self.active_batch_indices) / self.total_batches

        pacing_info = {
            'epoch': epoch,
            'readiness': readiness,
            'signals': signals,
            'batches_added': batches_to_add,
            'active_batches': len(self.active_batch_indices),
            'fraction': fraction
        }
        self.pacing_log.append(pacing_info)

        return self.active_batch_indices, pacing_info
```

**How `min_batches_per_epoch` guarantees all data is used**:
After each epoch, `min_to_add` is recalculated as `ceil(remaining_batches / remaining_epochs)`. This is a dynamic floor that increases as the deadline approaches. No hardcoded `max_epoch_for_full_data` needed — the math guarantees convergence.

---

### Step 8: Modify `run_single_experiment()` for Adaptive Mode

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: Update the training loop to use the adaptive pacer and batch-based data loading.

**New parameter**: `adaptive=False`. When `True`, use `AdaptiveCurriculumPacer` + `build_curriculum_loader`; when `False`, use the existing `CurriculumSampler` for baseline comparison.

**Changes to function signature** (line 840):
```python
def run_single_experiment(direction_label, curriculum_order, full_train_dataset,
                          val_loader, test_loader, class_names, total_train_samples,
                          device, total_epochs, batch_size, learning_rate, weight_decay,
                          momentum, start_fraction, full_data_epoch,
                          adaptive=False, window=5):
```

**New parameter**: `ordered_batches` — the pre-grouped batch list, passed in when `adaptive=True`.

**Training loop changes** (inside the `for epoch` loop, lines 894–924):

```python
# --- BEFORE epoch training ---
if adaptive:
    # Build loader from active batches
    active_indices = pacer.active_batch_indices
    train_loader, batch_index_order = build_curriculum_loader(
        full_train_dataset, ordered_batches, active_indices,
        batch_size, shuffle_within_batch=True
    )
    num_samples_this_epoch = sum(
        len(ordered_batches[i]) for i in active_indices
    )
else:
    # Existing fixed-pacing path (unchanged)
    curriculum_sampler.set_epoch(epoch)
    num_samples_this_epoch = len(curriculum_sampler)
    train_loader = DataLoader(...)

# --- Train ---
train_loss, train_acc, epoch_time, total_images, std_df, grad_norm_df = train_epoch(
    model, device, train_loader, optimizer, epoch,
    batch_index_order=batch_index_order if adaptive else None,
    scheduler=scheduler
)

# --- Validate ---
val_loss, val_acc, _, _ = validate(model, device, val_loader)

# --- AFTER epoch: update pacer and decide next step ---
if adaptive:
    pacer.update_metrics(train_loss, val_acc, std_df, grad_norm_df)
    active_indices, pacing_info = pacer.step(epoch)

    print(f"  Readiness: {pacing_info['readiness']:.3f} "
          f"(loss={pacing_info['signals']['loss_plateau']:.3f}, "
          f"val={pacing_info['signals']['val_acc_trend']:.3f}, "
          f"act={pacing_info['signals']['activation_stability']:.3f}, "
          f"grad={pacing_info['signals']['gradient_norm_stability']:.3f})")
    print(f"  Added {pacing_info['batches_added']} batches → "
          f"{pacing_info['active_batches']}/{len(ordered_batches)} "
          f"({pacing_info['fraction']*100:.1f}%)")
```

---

### Step 9: Update Plotting

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: Add a new plotting function for the adaptive pacing schedule and readiness signals.

```python
def plot_adaptive_schedule(pacing_log, total_batches, total_samples,
                           save_path='adaptive_schedule.pdf'):
    """
    Plot the adaptive curriculum schedule with readiness breakdown.

    Creates a 2x2 grid:
    1. Data fraction over epochs (the actual adaptive curve)
    2. Readiness score over epochs (composite)
    3. Individual signal breakdown (all 4 signals)
    4. Batches added per epoch (shows pacing decisions)
    """
    epochs = [p['epoch'] for p in pacing_log]
    fractions = [p['fraction'] for p in pacing_log]
    readiness = [p['readiness'] for p in pacing_log]
    loss_signals = [p['signals'].get('loss_plateau', 0) for p in pacing_log]
    val_signals = [p['signals'].get('val_acc_trend', 0) for p in pacing_log]
    act_signals = [p['signals'].get('activation_stability', 0) for p in pacing_log]
    grad_signals = [p['signals'].get('gradient_norm_stability', 0) for p in pacing_log]
    batches_added = [p['batches_added'] for p in pacing_log]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Subplot 1: Data fraction
    axes[0, 0].plot(epochs, [f * 100 for f in fractions], linewidth=2)
    axes[0, 0].axhline(100, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Adaptive Data Introduction', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Data Used (%)')
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Readiness score
    axes[0, 1].plot(epochs, readiness, color='#E91E63', linewidth=2)
    axes[0, 1].set_title('Composite Readiness Score', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Readiness [0–1]')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Signal breakdown (all 4)
    axes[1, 0].plot(epochs, loss_signals, label='Loss Plateau', linewidth=1.5)
    axes[1, 0].plot(epochs, val_signals, label='Val Acc Trend', linewidth=1.5)
    axes[1, 0].plot(epochs, act_signals, label='Activation Stability', linewidth=1.5)
    axes[1, 0].plot(epochs, grad_signals, label='Gradient Norm Stability', linewidth=1.5)
    axes[1, 0].set_title('Readiness Signal Breakdown', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Signal [0–1]')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Batches added per epoch
    axes[1, 1].bar(epochs, batches_added, color='#4CAF50', alpha=0.7)
    axes[1, 1].set_title('Batches Added Per Epoch', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Batches Added')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

Also update `plot_curriculum_schedule()` to accept an optional `recorded_fractions` parameter, so both fixed and adaptive schedules can be plotted on the same chart for comparison.

---

### Step 10: Wire It All Together in `run_cifar10_curriculum_learning()`

**File**: `cifar10/c10_gradus_unsupervised_ranking.py`
**What**: After ranking, run both fixed and adaptive experiments for comparison.

```python
# After Step 2 (ranking):
ordered_batches_t2b = create_ordered_batches(top_to_bottom, batch_size)
ordered_batches_b2t = create_ordered_batches(bottom_to_top, batch_size)

# Experiment 1: Top-to-Bottom, Fixed Pacing (baseline)
results_t2b_fixed = run_single_experiment(
    "t2b_fixed", ..., adaptive=False
)

# Experiment 2: Top-to-Bottom, Adaptive Pacing
results_t2b_adaptive = run_single_experiment(
    "t2b_adaptive", ..., adaptive=True, window=5,
    ordered_batches=ordered_batches_t2b
)

# Experiment 3: Bottom-to-Top, Fixed Pacing (baseline)
results_b2t_fixed = run_single_experiment(
    "b2t_fixed", ..., adaptive=False
)

# Experiment 4: Bottom-to-Top, Adaptive Pacing
results_b2t_adaptive = run_single_experiment(
    "b2t_adaptive", ..., adaptive=True, window=5,
    ordered_batches=ordered_batches_b2t
)
```

---

## Summary of All Hardcoded Constants

| Constant | Value | Where | Justification |
|----------|-------|-------|---------------|
| `start_fraction` | 0.3 | `AdaptiveCurriculumPacer.__init__` | Same as existing code — controls initial data pool |
| `window` | 5 | All 4 signal functions | Lookback window — structural choice, not scale-dependent |
| 0.25 | ratio threshold | `detect_plateau()` | "Recent variation < 25% of historical" — relative, not absolute. Shared by both activation stability and gradient norm signals. |
| 10 | sigmoid steepness | `compute_loss_plateau_signal()` | Controls transition sharpness in sigmoid; does not depend on loss scale |
| 5 | sigmoid steepness | `compute_val_acc_trend_signal()` | Same; controls transition sharpness |
| 3x | max multiplier | `AdaptiveCurriculumPacer.step()` | Max pace = 3x minimum pace; structural choice |
| 1e-8 | epsilon | Various | Division-by-zero guard |

**What was eliminated** (vs. bpas.py / old plan):
- ~~`delta_start = 0.000001`~~ — replaced by self-calibrating plateau detection
- ~~`delta_end = 0.00005`~~ — eliminated
- ~~`delta_loss = 0.01`~~ — replaced by relative loss comparison
- ~~`min_step = 0.01, max_step = 0.10`~~ — replaced by dynamically computed `min_to_add` / `max_to_add`
- ~~`max_epoch_for_full_data = 150`~~ — replaced by `ceil(remaining / remaining_epochs)` guarantee

---

## File Changes Summary

| File | Change | ~Lines |
|------|--------|--------|
| `c10_gradus_unsupervised_ranking.py` | Modify `ResNet.forward()` to support `return_activations` | 15 |
| `c10_gradus_unsupervised_ranking.py` | Add `create_ordered_batches()` | 10 |
| `c10_gradus_unsupervised_ranking.py` | Add `build_curriculum_loader()` | 30 |
| `c10_gradus_unsupervised_ranking.py` | Add `detect_plateau()` | 25 |
| `c10_gradus_unsupervised_ranking.py` | Add `compute_loss_plateau_signal()` | 20 |
| `c10_gradus_unsupervised_ranking.py` | Add `compute_val_acc_trend_signal()` | 20 |
| `c10_gradus_unsupervised_ranking.py` | Add `compute_activation_stability_signal()` | 20 |
| `c10_gradus_unsupervised_ranking.py` | Add `compute_gradient_norm_signal()` | 20 |
| `c10_gradus_unsupervised_ranking.py` | Add `AdaptiveCurriculumPacer` class | 110 |
| `c10_gradus_unsupervised_ranking.py` | Modify `train_epoch()` — activation + gradient norm collection | 35 |
| `c10_gradus_unsupervised_ranking.py` | Modify `run_single_experiment()` — adaptive mode | 35 |
| `c10_gradus_unsupervised_ranking.py` | Add `plot_adaptive_schedule()` | 40 |
| `c10_gradus_unsupervised_ranking.py` | Update `run_cifar10_curriculum_learning()` | 20 |
| **Total** | | **~400** |

All changes in a single file. The existing `CurriculumSampler` and fixed-pacing path are kept intact for baseline comparison.

---

## Implementation Order

1. **Modify `ResNet.forward()`** to support `return_activations=True` (Step 1)
2. **Add `create_ordered_batches()`** and **`build_curriculum_loader()`** (Steps 2 & 4)
3. **Modify `train_epoch()`** to collect per-batch activation stds AND per-batch gradient norms, accept `batch_index_order` (Step 3)
4. **Add `detect_plateau()`** and **four signal functions**: `compute_loss_plateau_signal()`, `compute_val_acc_trend_signal()`, `compute_activation_stability_signal()`, `compute_gradient_norm_signal()` (Steps 5 & 6)
5. **Add `AdaptiveCurriculumPacer`** class with 4-signal readiness (Step 7)
6. **Modify `run_single_experiment()`** to support `adaptive=True` mode, passing both `std_df` and `grad_norm_df` to pacer (Step 8)
7. **Add `plot_adaptive_schedule()`** with 2x2 grid showing all 4 signals, update existing plotting (Step 9)
8. **Wire up experiments in `run_cifar10_curriculum_learning()`** (Step 10)
9. **Test** with a short run (e.g., 20 epochs) to verify all 4 signals produce values and data fraction grows monotonically

---

## Potential Risks and Mitigations

1. **Activation + gradient norm collection overhead**: Calling `return_activations=True` and computing `torch.std()` per layer adds computation. Since we call `.detach()` before std, no gradient graph overhead. Gradient norms are computed from `p.grad.data` which already exists after `loss.backward()` — this is a read-only operation (`.norm(2)`) with no additional backward pass. Expected total overhead: ~5–10% per epoch.

2. **Noisy signals from mixup**: The 50% mixup augmentation (line 606) makes per-batch activation stds noisier. Mitigation: the `window`-based plateau detection smooths over this — individual noisy epochs don't trigger plateau.

3. **Cold start**: In the first `window` epochs, all four signals return 0.0 (not enough history). During this warmup, the pacer adds `min_batches_per_epoch` each epoch (the guaranteed minimum). This is intentional — the pacer is conservative until it has data to make informed decisions.

4. **Interaction with cosine annealing LR**: The LR schedule spans all 200 epochs regardless of data pacing. This is acceptable and matches how the fixed schedule works. The LR provides a decreasing "learning aggressiveness" that naturally complements the increasing data difficulty.
