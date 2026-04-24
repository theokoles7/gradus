[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [schedules](./README.md) / gradient

# Gradient Schedule

## Definition

The gradient schedule reorders all curriculum batches each epoch so that the batches with the highest mean gradient norm — where the model is currently most confounded — are presented first. All batches are seen every epoch; only their presentation order changes.

The motivation is information-theoretic: maximum learning occurs when the model is maximally confounded. By presenting the most confounding batches first each epoch, the model receives its strongest gradient signal when its learning capacity is freshest, and consolidates on easier batches afterward. This is a dynamic complement to static difficulty ordering — where a rank fixes sample order before training, the gradient schedule continuously re-ranks batches based on observed learning dynamics.

## Ordering

Each epoch, the schedule computes the mean gradient norm per batch across all recorded epochs and applies rolling mean smoothing across adjacent batches to reduce noise:

$$\bar{g}_b = \frac{1}{|\mathcal{H}_b|} \sum_{e \in \mathcal{H}_b} g_b^{(e)}$$

where $g_b^{(e)}$ is the mean gradient norm of batch $b$ at epoch $e$ and $\mathcal{H}_b$ is the set of epochs for which batch $b$ has been recorded. Batches are then sorted by descending smoothed gradient norm. Batches with no recorded history are assigned a norm of $0.0$ and fall naturally to the end of the order.

## Parameters

| Parameter             | Type      | Default   | Description                                                                           |
|-----------------------|-----------|-----------|---------------------------------------------------------------------------------------|
| `total_samples`       | `int`     | —         | Total number of training samples                                                      |
| `total_epochs`        | `int`     | —         | Total number of training epochs                                                       |
| `batch_size`          | `int`     | —         | Number of samples per batch                                                           |
| `start_fraction`      | `float`   | `0.3`     | Unused; retained for CLI consistency with other schedules                             |
| `smooth_window`       | `int`     | `5`       | Number of adjacent batches to average gradient norms over                             |
| `cold_start_epochs`   | `int`     | `1`       | Epochs to use natural curriculum order before activating gradient-based reordering    |

## Behavior

| Property                              | Value                                 |
|---------------------------------------|---------------------------------------|
| Responds to training signals          | Yes — per-batch gradient norms        |
| Active set monotonically increasing   | N/A (all batches active every epoch)  |
| All batches seen every epoch          | Yes                                   |
| Batch order within active set         | Descending mean gradient norm         |

## Notes

- During `cold_start_epochs`, batches are presented in natural curriculum order. This provides the gradient history needed for the first reordering step.
- Because all batches are seen every epoch, the data sufficiency index (DSI) is always `0.0` — the gradient schedule makes no tradeoff between data exposure and learning signal.
- The gradient schedule is compatible with any rank. The rank determines the natural curriculum order used during cold start and as the fallback when gradient history is unavailable.