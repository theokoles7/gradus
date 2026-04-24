[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / ascending

# Ascending Rank

## Definition

The ascending rank sorts sample indices such that metric values are non-decreasing from first to last - easiest samples first, hardest last. This is the canonical curriculum learning ordering, directly implementing the intuition from Bengio et al. (2009) that models benefit from beginning training on samples the difficulty measure identifies as simplest.

## Formula

Let $s_i$ denote the scalar difficulty score of sample $i$ under the chosen metric. The ascending rank produces the permutation $\sigma$ such that:

$$s_{\sigma(1)} \leq s_{\sigma(2)} \leq \cdots \leq s_{\sigma(N)}$$

## Parameters

| Parameter     | Type          | Default           | Description                                       |
|---------------|---------------|-------------------|---------------------------------------------------|
| `metric`      | `str`         | -                 | The single metric column to sort by               |
| `dataset_id`  | `str`         | -                 | Dataset identifier, used as part of the cache key |
| `scores`      | `DataFrame`   | -                 | Per-sample metric scores                          |
| `seed`        | `int`         | `1`               | Random seed, used as part of the cache key        |
| `cache_dir`   | `str \| Path` | `.cache/ranks`    | Directory for cached rank indices                 |

Ascending rank accepts exactly one metric. Passing a list of more than one metric raises a `ValueError`.

## Behavior

| Property                  | Value                                 |
|---------------------------|---------------------------------------|
| Scope compatibility       | `holistic`, `batch-wise`              |
| Multi-metric support      | No                                    |
| Inverted metric handling  | N/A (single metric)                   |
| Cache key fields          | `rank`, `dataset`, `metric`, `seed`   |

## Notes

- Ascending with a single non-inverted metric is mathematically equivalent to [`NormalizedMean`](./normalized-mean.md) with the same single metric, and to [`Lexicographic`](./lexicographic.md) with a single-element metric list.
- For metrics tagged `"inverted"` (e.g. `CompressionRatio`, where lower = more complex), ascending order places the *most* complex samples first. Use [`Descending`](./descending.md) or a composite rank with inversion handling when working with inverted metrics.

## References

- [Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.](https://dl.acm.org/doi/pdf/10.1145/1553374.1553380)