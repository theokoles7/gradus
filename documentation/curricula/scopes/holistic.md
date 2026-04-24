[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [scopes](./README.md) / holistic

# Holistic Scope

The holistic scope will globally rank and sort all samples within a dataset ***before*** chunking the samples into batches. This means that regardless of batch size, all samples will be presented to the network in sequential order, according to their rank in the dataset (i.e., the first batch will be the globally "easiest" samples, and the last batch will be the globally "hardest" samples).