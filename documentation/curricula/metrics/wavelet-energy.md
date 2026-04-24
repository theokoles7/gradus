[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / wavelet-energy

# Wavelet Energy

## Definition

The wavelet energy metric measures the total energy distributed across the detail subbands of a wavelet-decomposed image. Applying a discrete wavelet transform (DWT) to an image decomposes it into approximation and detail components at multiple scales. The energy concentrated in the detail subbands - horizontal, vertical, and diagonal - reflects the presence of edges, textures, and fine structure. Images with high wavelet energy are structurally complex and carry more high-frequency information, making them more difficult for a model to learn from early in training.

## Formula

Let $\psi$ denote the chosen wavelet basis (e.g. Haar, Daubechies). Applying the 2D DWT to image $I$ at decomposition level $l$ produces detail coefficient matrices $cH_l$, $cV_l$, and $cD_l$ corresponding to horizontal, vertical, and diagonal details respectively.

The energy of a coefficient matrix $C$ is:

$$E(C) = \sum_{i,j} C_{ij}^2$$

The total wavelet energy across all detail subbands at all levels $L$ is:

$$WE = \sum_{l=1}^{L} \left[ E(cH_l) + E(cV_l) + E(cD_l) \right]$$

Values below $10^{-6}$ are treated as zero to avoid floating-point noise on uniform images.

## Interpretation & Range

| Value         | Interpretation                                                            |
|---------------|---------------------------------------------------------------------------|
| $WE = 0$      | Uniform image - no detail at any decomposition level                      |
| Low $WE$      | Smooth image - energy concentrated in approximation band; little detail   |
| Moderate $WE$ | Structured image - meaningful edges and textures present                  |
| High $WE$     | Complex image - rich multi-scale detail; energy spread across subbands    |

Range is $[0, \infty)$. Values are not normalized by image size in the current implementation,
so wavelet energy is not directly comparable across images of different resolutions.

## Parameters

| Parameter | Type    | Default | Description |
|---|---|---|---|
| `wavelet` | `str`   | `"db2"` | Wavelet family to use for decomposition (e.g. `"haar"`, `"db2"`, `"sym4"`) |
| `level`   | `int`   | `None`  | Decomposition level. Defaults to the maximum possible for the image size.  |

## Notes

- Wavelet energy and [Wavelet Entropy](./wavelet-entropy.md) are complementary. Energy measures *how much* detail is present across all subbands; entropy measures *how evenly* that energy is distributed. A high-energy image can have low entropy (if all energy is concentrated in a single subband) or high entropy (if energy is spread uniformly). Both metrics share the same DWT computation internally and are efficient to compute together.
- Multi-channel inputs are averaged to a single channel before decomposition. The metric captures luminance-domain structure rather than per-channel structure.
- The default wavelet `"db2"` (Daubechies-2) provides a good balance between spatial and frequency localization for natural images. `"haar"` is faster and simpler but less smooth. Higher-order Daubechies wavelets (`"db4"`, `"db8"`) capture smoother features but require larger images to support the additional decomposition levels.
- Because `level=None` uses the maximum decomposition depth, total energy values are not comparable across images of different sizes — deeper decompositions accumulate more levels and thus more total energy. Fix `level` to a constant when comparing across heterogeneous datasets.

## References

- [Mallat, S. G. (1989). A theory for multiresolution signal decomposition: The wavelet representation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(7), 674–693.](https://repository.upenn.edu/server/api/core/bitstreams/85f1440b-7ecc-4263-aac7-672c2f7b82b8/content)
- [Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.](https://epubs.siam.org/doi/pdf/10.1137/1.9781611970104.fm)

---

## Formula Summary

| Symbol                | Description                      | Formula                                           |
|-----------------------|----------------------------------|---------------------------------------------------|
| $E(C)$                | Energy of coefficient matrix $C$ | $\sum_{i,j} C_{ij}^2$                             |
| $cH_l, cV_l, cD_l$   | Detail subbands at level $l$     | DWT decomposition of $I$                          |
| $WE$                  | Total wavelet energy             | $\sum_{l=1}^{L} [E(cH_l) + E(cV_l) + E(cD_l)]$   |