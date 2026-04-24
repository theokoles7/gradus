[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / compression-ratio

# Compression Ratio

## Definition

The compression ratio metric estimates the informational complexity of an image by measuring how compressible it is. Images with repetitive or uniform structure compress well, yielding a high ratio, while images with rich detail, high entropy, or complex textures resist compression, yielding a ratio closer to 1. Since compression algorithms exploit statistical redundancy, the compression ratio serves as a proxy for the amount of non-redundant information present in an image.

Compression is performed using JPEG encoding at a configurable quality level. The ratio is computed as original size divided by compressed size.

## Formula

Let $S_{\text{original}}$ denote the size of the raw (uncompressed) image in bytes and $S_{\text{compressed}}$ denote the size after lossless compression. The compression ratio is:

$$CR = \frac{S_{\text{original}}}{S_{\text{compressed}}}$$

A value of $CR \approx 1$ indicates no compressibility (maximum complexity); higher values indicate increasing redundancy and thus lower complexity.

## Interpretation & Range

| Value             | Interpretation                                                        |
|-------------------|-----------------------------------------------------------------------|
| $CR \approx 1$    | Highly complex - little to no redundancy; resists compression         |
| $1 < CR < 3$      | Moderate complexity - some spatial structure or pattern               |
| $CR \geq 3$       | Low complexity - high redundancy; uniform regions or simple patterns  |

The range is $[1, \infty)$ in theory, though natural images typically fall in the range $[1, 5]$.

## Notes

- Compression ratio is the **only** complexity metric in Gradus tagged `"inverted"`. A high ratio means the image is *easy* (low complexity), while a low ratio means the image is *hard* (high complexity) — the opposite of all other metrics. Composite ranks that use compression ratio alongside other metrics apply inversion automatically via the registry tag. When using compression ratio with [`Ascending`](../ranks/ascending.md) or [`Descending`](../ranks/descending.md) directly, be aware that ascending order places the most complex (lowest-ratio) samples *last*, not first.
- JPEG is a lossy codec, so `S_compressed` is affected by the `quality` parameter. Higher quality yields less compression and ratios closer to 1 for all images; lower quality compresses more aggressively and exaggerates differences between simple and complex images. The default quality of 95 preserves most detail while remaining sensitive to structural redundancy. The metric operates on the raw pixel array rather than the original file, so it is independent of whatever format the dataset stores images in.

## Parameters

| Parameter | Type  | Default   | Description                                                               |
|---        |---    |---        |---                                                                        |
| `quality` | `int` | `95`      | JPEG compression quality (1–100). Higher values yield less compression.   |

## References

- [Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission*, 1(1), 1–7.](http://alexander.shen.free.fr/library/Kolmogorov65_Three-Approaches-to-Information.pdf)

---

## Formula Summary

| Symbol    | Description           | Formula                                       |
|---        |---                    |---                                            |
| $CR$      | Compression ratio     | $S_{\text{original}} / S_{\text{compressed}}$ |