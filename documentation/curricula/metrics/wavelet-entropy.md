[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / wavelet-entropy

# Wavelet Entropy

## Definition

The wavelet entropy metric measures the disorder or unpredictability in how energy is distributed across the subbands of a wavelet-decomposed image. Where wavelet energy measures the total magnitude of detail coefficients, wavelet entropy measures how evenly that energy is spread. An image whose energy is concentrated in a single subband (e.g. one dominant edge orientation) has low entropy. An image whose energy is spread uniformly across many subbands - indicating complex, multi-scale, multi-directional structure - has high entropy. Wavelet entropy therefore captures a qualitatively different aspect of difficulty than wavelet energy alone.

## Formula

Let $WE_k$ denote the energy in the $k$-th wavelet subband (across all detail subbands and decomposition levels), and let $WE_{\text{total}} = \sum_k WE_k$ be the total energy. The relative energy of subband $k$ is:

$$p_k = \frac{WE_k}{WE_{\text{total}}}$$

Wavelet entropy is the Shannon entropy of this energy distribution:

$$H_W = -\sum_{k} p_k \log_2(p_k)$$

where the sum is taken over all subbands with $p_k > 0$.

The maximum possible entropy is $H_{\max} = \log_2(K)$ where $K$ is the total number of subbands. The normalized wavelet entropy returned as the metric value is:

$$H_{W,\text{norm}} = \frac{H_W}{H_{\max}}$$

## Interpretation & Range

| Value                  | Interpretation                                                                |
|------------------------|-------------------------------------------------------------------------------|
| $H_W \approx 0$        | Energy concentrated in one subband - simple, directionally uniform structure  |
| Moderate $H_W$         | Partial spread - some multi-scale complexity                                  |
| $H_W \approx H_{\max}$ | Energy uniformly distributed - rich, complex, multi-scale image               |

Normalized range is $[0, 1]$, where $1$ indicates maximum complexity in the wavelet domain.

## Parameters

| Parameter | Type  | Default   | Description                                                               |
|---        |---    |---        |---                                                                        |
| `wavelet` | `str` | `"db2"`   | Wavelet family to use for decomposition                                   |
| `level`   | `int` | `None`    | Decomposition level. Defaults to the maximum possible for the image size. |

## Notes

- Wavelet entropy returns `0.0` for images with zero total wavelet energy (e.g. fully uniform images), rather than producing a division-by-zero error or undefined entropy.
- When `level=None`, the number of subbands $K$ grows with the image size, and so does $H_{\max}$. Normalizing by $H_{\max}$ makes wavelet entropy comparable across images of different resolutions, unlike [Wavelet Energy](./wavelet-energy.md).
- Wavelet entropy is insensitive to the *amount* of detail — two images can have identical entropy but very different total energy. For this reason, wavelet energy and wavelet entropy are most informative when used together, either as separate metrics in a composite rank or as complementary signals in analysis.
- The same DWT is computed for both metrics. When computing both in the same pipeline, the coefficient arrays are shared via `cached_property`, so the decomposition is not duplicated.

## References

- [Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.](https://pure.mpg.de/rest/items/item_2383164/component/file_2383163/content)
- [Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A., Schürmann, M., & Başar, E. (2001). Wavelet entropy: a new tool for analysis of short duration brain electrical signals. *Journal of Neuroscience Methods*, 105(1), 65–75.](https://www.sciencedirect.com/science/article/pii/S0165027000003563?casa_token=wXjBl3eRM-YAAAAA:V92o1bgKsUy6jFDNNm3_1eqAT3OZKNnimcYgVpuz5SpwTHdPvScbyMZcRckLJ8aaW35R2TsQyFI)

---

## Formula Summary

| Symbol              | Description                    | Formula                       |
|---------------------|--------------------------------|-------------------------------|
| $WE_k$              | Energy in subband $k$          | $\sum_{i,j} C_{k,ij}^2$       |
| $p_k$               | Relative energy of subband $k$ | $WE_k / WE_{\text{total}}$    |
| $H_W$               | Wavelet entropy                | $-\sum_k p_k \log_2(p_k)$     |
| $H_{\max}$          | Maximum possible entropy       | $\log_2(K)$                   |
| $H_{W,\text{norm}}$ | Normalized wavelet entropy     | $H_W / H_{\max}$              |