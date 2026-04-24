[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / weighted

# Weighted

## Definition

The weighted rank sorts samples by a composite score that combines a designated anchor metric with all other available metrics, weighted by their Pearson correlation to the anchor. Metrics that correlate strongly with the anchor contribute more to the composite; metrics that are uncorrelated contribute less.

This is the appropriate composite rank when one metric is the primary difficulty signal but correlated signals should reinforce it proportionally.

## Formula

Let $m_a$ be the anchor metric and $\{m_1, \ldots, m_K\}$ all numeric metric columns excluding the index. All columns are z-score normalized:

$$z^{(k)}_i = \frac{s^{(k)}_i - \mu_k}{\sigma_k}$$

The Pearson correlation of each non-anchor metric $m_k$ against the anchor is computed on the normalized columns:

$$r_k = \text{corr}(z^{(a)},\ z^{(k)})$$

Weights are the absolute correlations, normalized to sum to 1:

$$w_k = \frac{|r_k|}{\sum_{j \neq a} |r_j|}$$

The composite score for sample $i$ is:

$$c_i = z^{(a)}_i + \sum_{k \neq a} w_k \cdot z^{(k)}_i$$

Samples are sorted ascending by $c_i$.

## Parameters

| Parameter     | Type          | Default           | Description                                       |
|---------------|---------------|-------------------|---------------------------------------------------|
| `metric`      | `str`         | —                 | The anchor metric                                 |
| `dataset_id`  | `str`         | —                 | Dataset identifier, used as part of the cache key |
| `scores`      | `DataFrame`   | —                 | Per-sample metric scores                          |
| `seed`        | `int`         | `1`               | Random seed, used as part of the cache key        |
| `cache_dir`   | `str \| Path` | `.cache/ranks`    | Directory for cached rank indices                 |

Weighted rank accepts exactly one anchor metric. Passing a list of more than one metric raises a `ValueError`.

## Behavior

| Property                  | Value                                                     |
|---------------------------|-----------------------------------------------------------|
| Scope compatibility       | `holistic`, `batch-wise`                                  |
| Multi-metric support      | No (single anchor; all other columns used automatically)  |
| Inverted metric handling  | No                                                        |
| Cache key fields          | `rank`, `dataset`, `metric`, `seed`                       |

## Notes

- All numeric non-index columns in the `scores` DataFrame are included as non-anchor metrics, regardless of which metrics were specified during scoring. Ensure the `scores` artifact contains only intended metric columns.
- When non-anchor metrics are uncorrelated with the anchor (all $r_k \approx 0$), the composite reduces toward the anchor z-score alone and ordering converges to [`Ascending`](./ascending.md) on the anchor.
- When non-anchor metrics are highly correlated with the anchor, the composite amplifies the anchor's signal and ordering diverges from plain ascending. This divergence is a sign the correlated metrics are adding information consistent with the anchor's difficulty assessment.