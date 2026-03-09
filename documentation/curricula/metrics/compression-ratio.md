[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / compression-ratio

# Compression Ratio

## Definition

The compression ratio metric estimates the informational complexity of an image by measuring how compressible it is. Images with repetitive or uniform structure compress well, yielding a low ratio, while images with rich detail, high entropy, or complex textures resist compression, yielding a high ratio. Since compression algorithms exploit statistical redundancy, the compression ratio serves as a proxy for the amount of non-redundant information present in an image.

## Formula

Let $S_{\text{original}}$ denote the size of the raw (uncompressed) image in bytes and $S_{\text{compressed}}$ denote the size after lossless compression. The compression ratio is:

$$CR = \frac{S_{\text{original}}}{S_{\text{compressed}}}$$

A value of $CR = 1$ indicates no compressibility (maximum complexity); values $CR > 1$ indicate increasing redundancy and thus lower complexity.

## Interpretation & Range

| Value             | Interpretation                                                        |
|-------------------|-----------------------------------------------------------------------|
| $CR \approx 1$    | Highly complex — little to no redundancy; resists compression         |
| $1 < CR < 3$      | Moderate complexity — some spatial structure or pattern               |
| $CR \geq 3$       | Low complexity — high redundancy; uniform regions or simple patterns  |

The range is $[1, \infty)$ in theory, though natural images typically fall in the range $[1, 5]$.

## References

- Kolmogorov, A. N. (1965). Three approaches to the quantitative definition of information. *Problems of Information Transmission*, 1(1), 1–7.
- Sheinwald, J., Lempel, A., & Ziv, J. (1992). On the compression of data with unknown statistics. *IEEE Transactions on Information Theory*, 38(3), 877–884.
## Formula Summary

| Symbol    | Description       | Formula                                       |
|-----------|-------------------|-----------------------------------------------|
| $CR$      | Compression ratio | $S_{\text{original}} / S_{\text{compressed}}$ |