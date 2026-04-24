[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [ranks](./README.md) / pairwise-correlation

# Pairwise Correlation Rank

## Definition

The pairwise correlation rank assigns each sample a scalar weight by accumulating votes from all pairwise Pearson correlations across the metric space, then sorts samples ascending by that weight. Samples with low weight are consistently easy across all metrics relative to their peers; samples with high weight are consistently hard.

Unlike [`NormalizedMean`](./normalized-mean.md) and [`Weighted`](./weighted.md), which reduce metrics to a single scalar via averaging or anchored weighting, the pairwise correlation rank derives its ordering from the relational structure of the metric space — it is sensitive to which samples agree and disagree across metrics, not just their marginal values.

## Formula

Let $\mathbf{x}_i \in \mathbb{R}^K$ be the row vector of normalized difficulty scores for sample $i$ across $K$ metrics, after min-max normalization and inversion of `"inverted"` metrics. Each row is mean-centered and L2-normalized to enable Pearson correlation via dot product:

$$\hat{\mathbf{x}}_i = \frac{\mathbf{x}_i - \bar{x}_i}{\|\mathbf{x}_i - \bar{x}_i\|_2}$$

The Pearson correlation between samples $i$ and $j$ is then:

$$P_{ij} = \hat{\mathbf{x}}_i \cdot \hat{\mathbf{x}}_j$$

Each sample accumulates a weight $W_i$ initialized to $0$. For each pair $(i, j)$ with $i < j$, $P_{ij}$ is used to cast a signed vote that updates both $W_i$ and $W_j$ according to whether the pair's current weights are in agreement or opposition. Samples are sorted ascending by their final weight $W_i$.

## Parameters

| Parameter     | Type                  | Default           | Description                                       |
|---------------|-----------------------|-------------------|---------------------------------------------------|
| `metric`      | `str \| List[str]`    | —                 | Metric columns to include                         |
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
| Time complexity           | $O(N^2 \cdot K)$                      |

## Notes

- When all metrics are perfectly correlated (all agree on difficulty ordering), pairwise correlation ordering converges to ascending order on any single metric.
- Runtime scales quadratically with dataset size. For large datasets, consider [`NormalizedMean`](./normalized-mean.md) or [`Weighted`](./weighted.md) as more efficient alternatives.
- Constant attribute vectors (samples with zero L2 norm after centering) have their norm clamped to `1.0` to avoid division by zero, effectively contributing zero correlation to all pairs.