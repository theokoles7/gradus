[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / ranks

# Ranks

A **rank** is a function that maps a `DataFrame` of per-sample difficulty scores to a permutation
of sample indices - the order in which a curriculum presents samples to the model during
training.

All ranks inherit from the abstract [`Rank`](https://github.com/theokoles7/gradus/blob/main/gradus/curricula/ranks/protocol.py) protocol and are registered via [`@register_rank`](https://github.com/theokoles7/gradus/blob/7b0a13c96fd287be5be4728b35704292fc23e8dd/gradus/registration/decorators.py#L185).

## Single-Metric

Single-metric ranks accept exactly one metric column and sort by it directly.

| Rank                                          | Description                                                   |
|-----------------------------------------------|---------------------------------------------------------------|
| [Ascending](./ascending.md)                   | Sort from lowest to highest score (easiest first)             |
| [Descending](./descending.md)                 | Sort from highest to lowest score (hardest first)             |
| [Distance from Mean](./distance-from-mean.md) | Sort by absolute deviation from the dataset mean, ascending   |

## Composite

Composite ranks combine multiple metric columns into a single ordering signal.

| Rank                                              | Description                                                                       |
|---------------------------------------------------|-----------------------------------------------------------------------------------|
| [Lexicographic](./lexicographic.md)               | Priority-ordered tie-breaking across an ordered list of metrics                   |
| [Normalized Mean](./normalized-mean.md)           | Equal-weighted mean of min-max normalized metric columns                          |
| [Pairwise Correlation](./pairwise-correlation.md) | Accumulates votes from all pairwise Pearson correlations across the metric space  |
| [Weighted](./weighted.md)                         | Correlation-weighted composite score anchored to one primary metric               |