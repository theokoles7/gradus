[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / descending

# Descending Rank

## Definition

The descending rank sorts sample indices such that metric values are non-increasing from first to last - hardest samples first, easiest last. This is the anti-curriculum ordering, useful as a controlled baseline against which standard curriculum learning can be compared.

## Formula

Let $s_i$ denote the scalar difficulty score of sample $i$ under the chosen metric. The
descending rank produces the permutation $\sigma$ such that:

$$s_{\sigma(1)} \geq s_{\sigma(2)} \geq \cdots \geq s_{\sigma(N)}$$

This is exactly the reverse of the [ascending](./ascending.md) permutation:

$$\sigma_{\text{desc}} = \text{rev}(\sigma_{\text{asc}})$$

## Parameters

| Parameter     | Type          | Default           | Description                                       |
|---------------|---------------|-------------------|---------------------------------------------------|
| `metric`      | `str`         | -                 | The single metric column to sort by               |
| `dataset_id`  | `str`         | -                 | Dataset identifier, used as part of the cache key |
| `scores`      | `DataFrame`   | -                 | Per-sample metric scores                          |
| `seed`        | `int`         | `1`               | Random seed, used as part of the cache key        |
| `cache_dir`   | `str \| Path` | `.cache/ranks`    | Directory for cached rank indices                 |

Descending rank accepts exactly one metric. Passing a list of more than one metric raises a `ValueError`.

## Behavior

| Property                  | Value                                 |
|---------------------------|---------------------------------------|
| Scope compatibility       | `holistic`, `batch-wise`              |
| Multi-metric support      | No                                    |
| Inverted metric handling  | N/A (single metric)                   |
| Cache key fields          | `rank`, `dataset`, `metric`, `seed`   |

## Notes

- Descending is primarily used as an anti-curriculum baseline. An experiment comparing ascending, descending, and random orderings isolates the directional effect of difficulty ordering from the effect of using a difficulty measure at all.
- For metrics tagged `"inverted"` (e.g. `CompressionRatio`), descending order places the *easiest* samples first, which may be the intended curriculum direction.

## References

- [Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.](https://dl.acm.org/doi/pdf/10.1145/1553374.1553380)
- [Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.](https://proceedings.mlr.press/v97/hacohen19a/hacohen19a.pdf)