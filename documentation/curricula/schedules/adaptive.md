[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [schedules](./README.md) / adaptive

# Adaptive Schedule

## Definition

The adaptive schedule expands the set of active batches based on a composite readiness score computed each epoch from four training signals: loss plateau, validation accuracy trend, activation stability, and gradient norm stability. When readiness is high — indicating the model has absorbed the current data — more batches are introduced. When readiness is low, the schedule advances conservatively.

Like the [linear schedule](./linear.md), the adaptive schedule always exposes batches in natural curriculum order and expands the active set monotonically.

## Readiness Score

Each epoch, four signals are computed and averaged into a scalar readiness score $\rho \in [0, 1]$:

$$\rho = \frac{r_{\text{loss}} + r_{\text{val}} + r_{\text{act}} + r_{\text{grad}}}{4}$$

| Signal            | Description                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------|
| $r_{\text{loss}}$ | Loss plateau signal — high when recent loss has stopped decreasing relative to a prior window     |
| $r_{\text{val}}$  | Validation accuracy trend signal — high when validation accuracy has stopped increasing           |
| $r_{\text{act}}$  | Activation stability signal — high when per-batch activation standard deviations have plateaued   |
| $r_{\text{grad}}$ | Gradient norm stability signal — high when per-batch gradient norms have plateaued                |

Each signal is computed over a lookback `window` of epochs. Signals for which insufficient history is available return `0.0`.

## Pacing Formula

Let $B_r$ be the remaining batches not yet activated and $E_r$ the remaining epochs. A dynamic floor ensures all data is active by the final epoch:

$$\text{min\_to\_add} = \max\!\left(1,\ \left\lceil \frac{B_r}{E_r} \right\rceil \right)$$

A readiness-scaled ceiling allows up to $3\times$ the floor in a single epoch:

$$\text{max\_to\_add} = \min(B_r,\ 3 \cdot \text{min\_to\_add})$$

$$\text{batches\_to\_add} = \text{min\_to\_add} + \rho \cdot (\text{max\_to\_add} - \text{min\_to\_add})$$

## Parameters

| Parameter         | Type      | Default   | Description                                                   |
|-------------------|-----------|-----------|---------------------------------------------------------------|
| `total_samples`   | `int`     | —         | Total number of training samples                              |
| `total_epochs`    | `int`     | —         | Total number of training epochs                               |
| `batch_size`      | `int`     | —         | Number of samples per batch                                   |
| `start_fraction`  | `float`   | `0.3`     | Fraction of batches exposed at epoch 1; must be in `(0, 1)`   |
| `window`          | `int`     | `5`       | Lookback window (in epochs) for all four signals              |

## Behavior

| Property                              | Value                                                     |
|---------------------------------------|-----------------------------------------------------------|
| Responds to training signals          | Yes — loss, val accuracy, activation std, gradient norm   |
| Active set monotonically increasing   | Yes                                                       |
| All batches seen by final epoch       | Guaranteed                                                |
| Batch order within active set         | Natural curriculum order                                  |

## Notes

- All four signals are optional. If a signal is not provided to `schedule.step()`, it returns `0.0` and contributes a conservative bias to the readiness score.
- The dynamic floor guarantees that even with consistently low readiness, all batches will be active by the final epoch.
- Activation and gradient norm signals are tracked per-batch, not per-epoch globally, providing finer-grained information about which parts of the curriculum the model has stabilized on.