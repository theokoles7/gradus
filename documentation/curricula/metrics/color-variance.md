[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / color-variance

# Color Variance

## Definition

The color variance metric measures the spread of pixel intensity values across an image. Images with low color variance are visually uniform — dominated by a single color or smooth gradient — and tend to be easier for a model to learn from. Images with high color variance exhibit a wide range of intensities and color distributions, reflecting greater visual diversity and structural complexity. In multi-channel images, variance is computed globally across all channels and pixels.

## Formula

Let $\{P_i\}_{i=1}^{N}$ denote the set of all pixel values in the image (flattened across all channels), where $N$ is the total number of pixel-channel values. The mean pixel value is:

$$\mu = \frac{1}{N} \sum_{i=1}^{N} P_i$$

The color variance is:

$$\text{Var} = \frac{1}{N} \sum_{i=1}^{N} (P_i - \mu)^2$$

## Interpretation & Range

| Value | Interpretation |
|---|---|
| $\text{Var} \approx 0$ | Uniform image — single color or flat region; minimal visual complexity |
| Low–Moderate | Some color spread — simple objects against plain backgrounds |
| High $\text{Var}$ | Rich color distribution — diverse hues, high contrast, complex scenes |

For 8-bit images with pixel values in $[0, 255]$, the theoretical maximum variance is $\approx 16256$ (half pixels at 0, half at 255). In practice, natural images occupy a much narrower range.

## References

- Gonzalez, R. C., & Woods, R. E. (2008). *Digital Image Processing* (3rd ed.). Pearson.
- Peng, T., Jermyn, I. H., Prinet, V., & Zerubia, J. (2009). Incorporating generic and specific prior knowledge in a multi-scale phase field model for road extraction from VHR images. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2(2), 139–150.

---

## Formula Summary

| Symbol | Description | Formula |
|---|---|---|
| $\mu$ | Mean pixel value | $\frac{1}{N} \sum_{i=1}^{N} P_i$ |
| $\text{Var}$ | Color variance | $\frac{1}{N} \sum_{i=1}^{N} (P_i - \mu)^2$ |