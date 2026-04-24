[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [schedules](./README.md) / linear

# Linear Schedule

## Definition

The linear schedule is a fixed, epoch-driven pacing strategy. It linearly ramps the number of active batches from `start_fraction × total_batches` at epoch 1 to `total_batches` by a configurable `full_data_epoch`, then holds at the full dataset for all remaining epochs. Batches are always exposed in natural curriculum order — the order determined by the rank — starting from batch 0.

The linear schedule ignores all training signals. Its pacing is determined entirely by the epoch counter, making it fully reproducible and free of feedback loops.

## Pacing Formula

Let $B$ be the total number of batches, $f_0$ the start fraction, and $E_f$ the full-data epoch. The number of active batches at epoch $e$ is:

$$
\text{active}(e) =
\begin{cases}
B & \text{if } e \geq E_f \\
\max\!\left(1,\ \left\lfloor \left( f_0 + (1 - f_0) \cdot \dfrac{e}{E_f} \right) \cdot B \right\rfloor \right) & \text{otherwise}
\end{cases}
$$

## Parameters

| Parameter         | Type      | Default               | Description                                                   |
|-------------------|-----------|-----------------------|---------------------------------------------------------------|
| `total_samples`   | `int`     | —                     | Total number of training samples                              |
| `total_epochs`    | `int`     | —                     | Total number of training epochs                               |
| `batch_size`      | `int`     | —                     | Number of samples per batch                                   |
| `start_fraction`  | `float`   | `0.3`                 | Fraction of batches exposed at epoch 1; must be in `(0, 1)`   |
| `full_data_epoch` | `int`     | `0.6 × total_epochs`  | Epoch at which 100% of batches are first exposed              |

## Behavior

| Property                              | Value                         |
|---------------------------------------|-------------------------------|
| Responds to training signals          | No                            |
| Active set monotonically increasing   | Yes                           |
| All batches seen every epoch          | No (until `full_data_epoch`)  |
| Batch order within active set         | Natural curriculum order      |

## Notes

- The linear schedule is the simplest baseline pacing strategy. Its behavior is fully determined by `start_fraction` and `full_data_epoch`, making it straightforward to reason about and reproduce.
- Setting `full_data_epoch = 1` exposes the full dataset immediately (all epochs at full data), effectively disabling pacing while preserving the ranked batch order.
- The active batch count is guaranteed to be at least 1 at every epoch.