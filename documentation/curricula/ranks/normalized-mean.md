[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / normalized-mean

# Normalized Mean

## Definition

The normalized mean rank reduces multiple difficulty metrics to a single composite score by taking the equal-weighted mean of their min-max normalized values, then sorting samples ascending by that composite. Metrics tagged `"inverted"` in the registry are flipped prior to averaging so that all columns align on a unified scale where higher values indicate greater complexity.

This is the appropriate composite rank when no single metric dominates and all metrics should contribute equally to the difficulty assessment.

## Formula

Let $\{m_1, \ldots, m_K\}$ be the selected metrics and $s^{(k)}_i$ the score of sample $i$ on metric $k$. Each column is min-max normalized:

$$\tilde{s}^{(k)}_i = \frac{s^{(k)}_i - \min_j s^{(k)}_j}{\max_j s^{(k)}_j - \min_j s^{(k)}_j}$$

For any metric $k$ tagged `"inverted"`, the normalized value is flipped:

$$\tilde{s}^{(k)}_i \leftarrow 1 - \tilde{s}^{(k)}_i$$

The composite score is the equal-weighted mean across all $K$ metrics:

$$c_i = \frac{1}{K} \sum_{k=1}^{K} \tilde{s}^{(k)}_i$$

Samples are then sorted ascending by $c_i$.

## Parameters

| Parameter     | Type                  | Default           | Description                                       |
|---------------|-----------------------|-------------------|---------------------------------------------------|
| `metric`      | `str \| List[str]`    | —                 | Metric columns to include in the composite        |
| `dataset_id`  | `str`                 | —                 | Dataset identifier, used as part of the cache key |
| `scores`      | `DataFrame`           | —                 | Per-sample metric scores                          |
| `seed`        | `int`                 | `1`               | Random seed, used as part of the cache key        |
| `cache_dir`   | `str \| Path`         | `.cache/ranks`    | Directory for cached rank indices                 |

## Behavior

| Property                  | Value                                 |
|---------------------------|---------------------------------------|
| Scope compatibility       | `holistic`, `batch-wise`              |
| Multi-metric support      | Yes                                   |
| Inverted metric handling  | Yes — automatic via registry tags     |
| Cache key fields          | `rank`, `dataset`, `metric`, `seed`   |

## Notes

- With a single non-inverted metric, normalized mean ordering is equivalent to [`Ascending`](./ascending.md) on that metric (min-max normalization preserves rank order).
- Constant metric columns (all samples have the same value) produce a zero range and are set to `0.0` after normalization, effectively removing their contribution to the composite.
- All metrics contribute equally regardless of their original scale. A metric with values in `[0, 1000]` and one with values in `[0, 1]` are weighted identically after normalization.