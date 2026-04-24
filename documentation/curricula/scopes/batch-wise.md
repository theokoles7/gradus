[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [scopes](./README.md) / batch-wise

# Batch-Wise Scope

For the batch-wise scope, dataset indices are shuffled first (to break class grouping), then chunked, then each chunk is ranked independently - producing batches that are internally sorted by difficulty but not globally ordered relative to each other. This means that the "easiest" samples will be distributed across batches, and each batch will contain a mix of easier and harder samples. The network will see a more varied distribution of sample difficulties within each batch.