[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [scopes](./README.md) / holistic

# Holistic Scope

## Definition

In the holistic scope, all samples in the dataset are ranked globally before being chunked into batches. The result is a strict end-to-end ordering: the first batch contains the globally easiest samples, the last batch contains the globally hardest, and every batch in between is a contiguous window of the global difficulty ranking.

This is the most direct implementation of the curriculum learning hypothesis — the model sees data ordered from simplest to most complex, without any within-batch randomization or class mixing.

## Procedure

1. The configured rank is applied to all $N$ sample indices simultaneously, using the full `DatasetMetrics` artifact.
2. The resulting globally sorted index list is chunked into batches of size `batch_size` in order.
3. The result is a list of batches where batch $k$ contains samples ranked $[(k-1) \cdot B, k \cdot B)$ globally.

## Comparison with Batch-Wise

| Property                                      | Holistic                      | Batch-Wise                |
|---                                            |---                            |---                        |
| Global difficulty ordering                    | Yes                           | No                        |
| Intra-batch difficulty ordering               | Yes                           | Yes                       |
| Class mixing within batches                   | Depends on dataset ordering   | Yes (via shuffle)         |
| First batch contains globally easiest samples | Yes                           | No                        |
| Rank computed on                              | Full dataset                  | Each chunk independently  |

## When to Use

Holistic scope is the right choice when the experiment requires the strongest possible curriculum signal — the model should see the globally easiest samples first and the globally hardest last, with no deviation. It is also the appropriate scope for ablation studies where scope itself is an experimental variable.

When class imbalance within early batches is a concern (because easy samples may cluster by class in the global ranking), consider [batch-wise](./batch-wise.md) scope instead.

## Notes

- Holistic ranking is computed once before training and cached. The cache key covers the full dataset, so changing the dataset, metric, rank, or seed triggers a full recomputation. Because no shuffle is applied, the class distribution within each batch is determined entirely by the correlation between difficulty scores and class labels. On datasets where one class is systematically easier than another (e.g. MNIST digit 1 vs. digit 8), early batches may be classmbalanced.
- The rank cache for holistic scope covers all $N$ samples in a single `.npy` file. For batch-wise scope, the cache is per-chunk. Holistic caching is therefore more compact and faster to load.