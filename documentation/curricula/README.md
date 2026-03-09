[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../README.md) / curricula

# Curricula

## Definition

A curriculum is a two-component specification consisting of:

### 1. **Difficulty Measure**:

A vector-valued function that produces a per-sample difficulty vector for each sample in a dataset, where each dimension captures a distinct aspect of sample difficulty. Difficulty may be assessed intrinsically (e.g. via image complexity metrics) or relative to a model's learning dynamics (e.g. time-to-convergence). In the degenerate case, the difficulty measure reduces to a scalar-valued function over a single dimension.

### 2. **Ordering Scheme**:

A policy that maps a sequence of difficulty vectors to a sorted sequence of sample indices, determining the order in which samples are presented to the model during training.