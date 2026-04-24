[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [scopes](./README.md) / batch-wise

# Batch-Wise Scope

## Definition

In the batch-wise scope, dataset indices are shuffled before being chunked into batches, and then each chunk is ranked independently. This produces batches that are internally sorted by difficulty but not globally ordered relative to each other — the "easiest" samples are distributed across all batches rather than concentrated in the first ones.

The shuffle step is seeded for reproducibility and serves a specific purpose: without it, batches would inherit class grouping from the dataset's natural ordering, meaning each batch would contain predominantly one class. The shuffle ensures each batch gets a representative cross-section of the dataset before intra-batch ranking is applied.

## Procedure

1. All sample indices $\{0, 1, \ldots, N-1\}$ are shuffled using the configured seed.
2. The shuffled indices are chunked into batches of size `batch_size`.
3. Each chunk is ranked independently using the configured rank and metric(s).
4. The result is a list of batches, each internally sorted by difficulty, in no particular global order.

## Comparison with Holistic

| Property                                      | Batch-Wise                | Holistic                      |
|---                                            |---                        |---                            |
| Global difficulty ordering                    | No                        | Yes                           |
| Intra-batch difficulty ordering               | Yes                       | Yes                           |
| Class mixing within batches                   | Yes (via shuffle)         | Depends on dataset ordering   |
| First batch contains globally easiest samples | No                        | Yes                           |
| Rank computed on                              | Each chunk independently  | Full dataset                  |

## When to Use

Batch-wise scope is appropriate when global difficulty ordering is less important than ensuring each batch contains a spread of classes, or when the dataset is large enough that holistic ranking is computationally expensive. It also introduces more stochasticity into the curriculum, which may act as a regularizer.

For experiments where the curriculum hypothesis depends specifically on the model seeing globally easy samples before globally hard ones, use [holistic](./holistic.md) scope instead.

## Notes

- The shuffle seed is the same seed passed to the `Curriculum` constructor, ensuring that batch composition is fully reproducible across runs with the same configuration.
- Because each batch is ranked independently, the rank cache key includes the batch's specific subset of sample indices. Different seeds produce different shuffles, different chunks, and therefore different cache entries.
- Intra-batch ranking uses the same rank class and metric(s) as holistic ranking — the only difference is the scope of the sorted index set.