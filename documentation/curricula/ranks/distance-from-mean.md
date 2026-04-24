[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / distance-from-mean

# Distance from Mean Rank

## Definition

The distance-from-mean rank sorts samples by their absolute deviation from the dataset mean difficulty score, ascending. Samples closest to the mean are presented first; samples furthest from the mean - whether unusually easy or unusually hard - are presented last.

This ordering is useful when the most representative samples are the best starting point for training. Rather than anchoring on the globally easiest or hardest end of the difficulty distribution, the curriculum begins at the center of the distribution and expands outward.

## Formula

Let $s_i$ denote the scalar difficulty score of sample $i$ and $\bar{s}$ the dataset mean:

$$\bar{s} = \frac{1}{N} \sum_{i=1}^{N} s_i$$

The distance-from-mean rank produces the permutation $\sigma$ such that:

$$|s_{\sigma(1)} - \bar{s}| \leq |s_{\sigma(2)} - \bar{s}| \leq \cdots \leq |s_{\sigma(N)} - \bar{s}|$$

## Parameters

| Parameter     | Type          | Default           | Description                                       |
|---------------|---------------|-------------------|---------------------------------------------------|
| `metric`      | `str`         | -                 | The single metric column to sort by               |
| `dataset_id`  | `str`         | -                 | Dataset identifier, used as part of the cache key |
| `scores`      | `DataFrame`   | -                 | Per-sample metric scores                          |
| `seed`        | `int`         | `1`               | Random seed, used as part of the cache key        |
| `cache_dir`   | `str \| Path` | `.cache/ranks`    | Directory for cached rank indices                 |

Distance-from-mean rank accepts exactly one metric. Passing a list of more than one metric raises a `ValueError`.

## Behavior

| Property                  | Value                                 |
|---------------------------|---------------------------------------|
| Scope compatibility       | `holistic`, `batch-wise`              |
| Multi-metric support      | No                                    |
| Inverted metric handling  | N/A (single metric)                   |
| Cache key fields          | `rank`, `dataset`, `metric`, `seed`   |

## Notes

- Ties at equal absolute deviation (samples equidistant from the mean on opposite sides) are resolved by the underlying `sort_values` stable sort, so their relative order is determined by their original position in the `DataFrame`.
- Unlike [`Ascending`](./ascending.md) and [`Descending`](./descending.md), this rank is symmetric around the mean: it does not distinguish between samples that are easier than average and samples that are harder than average by the same margin.