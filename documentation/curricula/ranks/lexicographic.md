[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / lexicographic

# Lexicographic Rank

## Definition

The lexicographic rank sorts samples using an ordered list of metrics as a strict priority sequence. Samples are sorted by the first metric; ties are broken by the second metric; remaining ties are broken by the third; and so on — exactly as a dictionary sort works over a sequence of keys.

This is the appropriate composite rank when metrics have a natural priority ordering and tie-breaking behavior should be explicit and deterministic rather than averaged.

## Formula

Let $\mathbf{m} = (m_1, m_2, \ldots, m_K)$ be the ordered list of metrics. The lexicographic rank produces the permutation $\sigma$ such that for all consecutive pairs:

$$
(s^{(1)}_{\sigma(i)},\ s^{(2)}_{\sigma(i)},\ \ldots,\ s^{(K)}_{\sigma(i)})
\ \leq_{\text{lex}}\
(s^{(1)}_{\sigma(i+1)},\ s^{(2)}_{\sigma(i+1)},\ \ldots,\ s^{(K)}_{\sigma(i+1)})
$$

where $\leq_{\text{lex}}$ denotes the standard lexicographic (dictionary) order on tuples.

## Parameters

| Parameter     | Type                  | Default           | Description                                               |
|---------------|-----------------------|-------------------|-----------------------------------------------------------|
| `metric`      | `str \| List[str]`    | —                 | Ordered list of metric columns, highest priority first    |
| `dataset_id`  | `str`                 | —                 | Dataset identifier, used as part of the cache key         |
| `scores`      | `DataFrame`           | —                 | Per-sample metric scores                                  |
| `seed`        | `int`                 | `1`               | Random seed, used as part of the cache key                |
| `cache_dir`   | `str \| Path`         | `.cache/ranks`    | Directory for cached rank indices                         |

## Behavior

| Property                  | Value                                 |
|---------------------------|---------------------------------------|
| Scope compatibility       | `holistic`, `batch-wise`              |
| Multi-metric support      | Yes                                   |
| Inverted metric handling  | No — all metrics sorted ascending     |
| Cache key fields          | `rank`, `dataset`, `metric`, `seed`   |

## Notes

- With a single metric, lexicographic ordering is mathematically equivalent to [`Ascending`](./ascending.md) on that metric.
- Metric order matters: `["saturation-time", "color-variance"]` and `["color-variance", "saturation-time"]` will generally produce different orderings.
- All metrics are sorted ascending. Lexicographic does not apply inversion for metrics tagged `"inverted"`. If a consistent directionality across metrics is required, prefer [`NormalizedMean`](./normalized-mean.md), which handles inversion automatically.
- The cache key encodes the full ordered metric list, so changing metric order produces a new cache entry and triggers recomputation.