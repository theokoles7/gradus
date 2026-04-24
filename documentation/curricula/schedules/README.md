[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / schedules

# Schedules

A **schedule** controls which subset of ranked batches the curriculum exposes each epoch and in what order. Each epoch, the training loop calls `schedule.step(epoch, **signals)`, which returns a list of batch indices. The curriculum applies that ordering via `curriculum.set_order()`.

All schedules inherit from the abstract [`Schedule`](https://github.com/theokoles7/gradus/blob/main/gradus/curricula/schedules/protocol.py) protocol and are registered via [`@register_schedule`](https://github.com/theokoles7/gradus/blob/e43b1b3045210d6901a11677ff1a948fa81b24fd/gradus/registration/decorators.py#L224).

## Pacing Schedules

Pacing schedules control how much of the ranked dataset is exposed over time, expanding the active set monotonically as training progresses.

| Schedule                  | Description                                                                                       |
|---------------------------|---------------------------------------------------------------------------------------------------|
| [Linear](./linear.md)     | Linearly ramps active batches from a start fraction to the full dataset by a fixed epoch          |
| [Adaptive](./adaptive.md) | Expands the active set based on a composite readiness score derived from four training signals    |

## Reordering Schedules

Reordering schedules expose all batches every epoch but change their presentation order based on dynamic training signals.

| Schedule                  | Description                                                                                   |
|---------------------------|-----------------------------------------------------------------------------------------------|
| [Gradient](./gradient.md) | Reorders batches each epoch so that the most gradient-confounding batches are presented first |