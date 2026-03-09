[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / metrics

# Metrics

A **difficulty metric** is a function $f: \mathcal{X} \rightarrow \mathbb{R}$ (or $\mathbb{R}^k$ in the multi-dimensional case) that assigns a scalar or vector difficulty score to each sample in a dataset.

## Complexity-Based

Complexity metrics assess difficulty from the image itself, independent of any model.

| Metric                                        | Description                                                   |
|-----------------------------------------------|---------------------------------------------------------------|
| [Color Variance](./color-variance.md)         | Spread of pixel intensity values across all channels          |
| [Compression Ratio](./compression-ratio.md)   | Ratio of raw to compressed image size; proxy for redundancy   |
| [Edge Density](./edge-density.md)             | Fraction of pixels classified as edges                        |
| [Spatial Frequency](./spatial-frequency.md)   | RMS magnitude of pixel-level intensity gradients              |
| [Wavelet Energy](./wavelet-energy.md)         | Total energy in wavelet detail subbands                       |
| [Wavelet Entropy](./wavelet-entropy.md)       | Shannon entropy of the wavelet subband energy distribution    |

## Model-Informed

Model-informed metrics assess difficulty relative to a model's learning dynamics.

| Metric                                                | Description                                                           |
|-------------------------------------------------------|-----------------------------------------------------------------------|
| [Time-to-Convergence (TTC)](./convergence-time.md) | Iterations until per-sample loss stabilizes                           |
| [Time-to-Saturation (TTS)](./saturation-time.md)   | Iterations until the L2 norm of the delta of model weights stabilizes |