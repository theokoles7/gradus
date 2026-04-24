[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / color-variance

# Color Variance

## Definition

The color variance metric measures the spread of pixel intensity values across an image. Images with low color variance are visually uniform - dominated by a single color or smooth gradient - and tend to be easier for a model to learn from. Images with high color variance exhibit a wide range of intensities and color distributions, reflecting greater visual diversity and structural complexity. In multi-channel images, variance is computed independently per channel and then averaged across channels.

## Formula

Let $\{P_i\}_{i=1}^{N}$ denote the set of all pixel values in the image (flattened across all channels), where $N$ is the total number of pixel-channel values. The mean pixel value is:

$$\mu = \frac{1}{N} \sum_{i=1}^{N} P_i$$

The color variance is:

$$\text{Var} = \frac{1}{N} \sum_{i=1}^{N} (P_i - \mu)^2$$

For multi-channel images, variance is computed per channel and the mean channel variance is returned as the scalar difficulty score.

## Interpretation & Range

| Value                     | Interpretation                                                            |
|---                        |---                                                                        |
| $\text{Var} \approx 0$    | Uniform image - single color or flat region; minimal visual complexity    |
| Low–Moderate              | Some color spread - simple objects against plain backgrounds              |
| High $\text{Var}$         | Rich color distribution - diverse hues, high contrast, complex scenes     |

For 8-bit images with pixel values in $[0, 255]$, the theoretical maximum variance is $\approx 16256$ (half pixels at 0, half at 255). In practice, natural images occupy a much narrower range.

## Notes

- Color variance is a global statistic — it captures overall intensity spread but is insensitive to spatial structure. Two images with identical variance can have very different spatial complexity (e.g. a uniform noise image vs. a checkerboard). For spatial structure, see [Spatial Frequency](./spatial-frequency.md) or [Edge Density](./edge-density.md).
- Grayscale (single-channel) inputs are handled correctly: the channel loop reduces to a single variance computation.
- Color variance is not tagged `"inverted"`. Higher values indicate greater complexity, consistent with the convention used by all other non-inverted metrics.
- Among the complexity-based metrics, color variance tends to correlate most strongly with [Spatial Frequency](./spatial-frequency.md) on natural image datasets, since both respond to pixel-level intensity variation.

## References

- [Gonzalez, R. C., & Woods, R. E. (2008). *Digital Image Processing* (3rd ed.). Pearson.](https://link.springer.com/chapter/10.1007/978-1-84628-968-2_12)
- [Peng, T., Jermyn, I. H., Prinet, V., & Zerubia, J. (2009). Incorporating generic and specific prior knowledge in a multi-scale phase field model for road extraction from VHR images. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 2(2), 139–150.](https://ieeexplore.ieee.org/document/4609445)

---

## Formula Summary

| Symbol        | Description       | Formula                                       |
|---            |---                |---                                            |
| $\mu$         | Mean pixel value  | $\frac{1}{N} \sum_{i=1}^{N} P_i$              |
| $\text{Var}$  | Color variance    | $\frac{1}{N} \sum_{i=1}^{N} (P_i - \mu)^2$    |