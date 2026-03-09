[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / wavelet-energy

# Wavelet Energy

## Definition

The wavelet energy metric measures the total energy distributed across the detail subbands of a wavelet-decomposed image. Applying a discrete wavelet transform (DWT) to an image decomposes it into approximation and detail components at multiple scales. The energy concentrated in the detail subbands — horizontal, vertical, and diagonal — reflects the presence of edges, textures, and fine structure. Images with high wavelet energy are structurally complex and carry more high-frequency information, making them more difficult for a model to learn from early in training.

## Formula

Let $\psi$ denote the chosen wavelet basis (e.g. Haar, Daubechies). Applying the 2D DWT to image $I$ at decomposition level $l$ produces detail coefficient matrices $cH_l$, $cV_l$, and $cD_l$ corresponding to horizontal, vertical, and diagonal details respectively.

The energy of a coefficient matrix $C$ is:

$$E(C) = \sum_{i,j} C_{ij}^2$$

The total wavelet energy across all detail subbands at all levels $L$ is:

$$WE = \sum_{l=1}^{L} \left[ E(cH_l) + E(cV_l) + E(cD_l) \right]$$

To compare across images of different sizes, the energy is normalized by the total number of coefficients $N$:

$$WE_{\text{norm}} = \frac{WE}{N}$$

## Interpretation & Range

| Value         | Interpretation                                                            |
|---------------|---------------------------------------------------------------------------|
| Low $WE$      | Smooth image — energy concentrated in approximation band; little detail   |
| Moderate $WE$ | Structured image — meaningful edges and textures present                  |
| High $WE$     | Complex image — rich multi-scale detail; energy spread across subbands    |

Range is $[0, \infty)$; normalized values are bounded by the square of the maximum pixel intensity.

## References

- Mallat, S. G. (1989). A theory for multiresolution signal decomposition: The wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674–693.
- Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.

---

## Formula Summary

| Symbol                | Description                       | Formula                                           |
|-----------------------|-----------------------------------|---------------------------------------------------|
| $E(C)$                | Energy of coefficient matrix $C$  | $\sum_{i,j} C_{ij}^2$                             |
| $cH_l, cV_l, cD_l$    | Detail subbands at level $l$      | DWT decomposition of $I$                          |
| $WE$                  | Total wavelet energy              | $\sum_{l=1}^{L} [E(cH_l) + E(cV_l) + E(cD_l)]$    |
| $WE_{\text{norm}}$    | Normalized wavelet energy         | $WE / N$                                          |