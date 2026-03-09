[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / wavelet-entropy

# Wavelet Entropy

## Definition

The wavelet entropy metric measures the disorder or unpredictability in how energy is distributed across the subbands of a wavelet-decomposed image. Where wavelet energy measures the total magnitude of detail coefficients, wavelet entropy measures how evenly that energy is spread. An image whose energy is concentrated in a single subband (e.g. one dominant edge orientation) has low entropy. An image whose energy is spread uniformly across many subbands — indicating complex, multi-scale, multi-directional structure — has high entropy. Wavelet entropy therefore captures a qualitatively different aspect of difficulty than wavelet energy alone.

## Formula

Let $WE_k$ denote the energy in the $k$-th wavelet subband (across all detail subbands and decomposition levels), and let $WE_{\text{total}} = \sum_k WE_k$ be the total energy. The relative energy of subband $k$ is:

$$p_k = \frac{WE_k}{WE_{\text{total}}}$$

Wavelet entropy is the Shannon entropy of this energy distribution:

$$H_W = -\sum_{k} p_k \log_2(p_k)$$

where the sum is taken over all subbands with $p_k > 0$.

The maximum possible entropy $H_{\max} = \log_2(K)$ where $K$ is the total number of subbands. Normalized wavelet entropy is:

$$H_{W,\text{norm}} = \frac{H_W}{H_{\max}}$$

## Interpretation & Range

| Value                     | Interpretation                                                                |
|---------------------------|-------------------------------------------------------------------------------|
| $H_W \approx 0$           | Energy concentrated in one subband — simple, directionally uniform structure  |
| Moderate $H_W$            | Partial spread — some multi-scale complexity                                  |
| $H_W \approx H_{\max}$    | Energy uniformly distributed — rich, complex, multi-scale image               |

Normalized range is $[0, 1]$, where $1$ indicates maximum complexity in the wavelet domain.

## References

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423.
- Rosso, O. A., Blanco, S., Yordanova, J., Kolev, V., Figliola, A., Schürmann, M., & Başar, E. (2001). Wavelet entropy: a new tool for analysis of short duration brain electrical signals. *Journal of Neuroscience Methods*, 105(1), 65–75.

---

## Formula Summary

| Symbol                            | Description                       | Formula                       |
|-----------------------------------|-----------------------------------|-------------------------------|
| $WE_k$                            | Energy in subband $k$             | $\sum_{i,j} C_{k,ij}^2$       |
| $p_k$                             | Relative energy of subband $k$    | $WE_k / WE_{\text{total}}$    |
| $H_W$                             | Wavelet entropy                   | $-\sum_k p_k \log_2(p_k)$     |
| $H_{\max}$                        | Maximum possible entropy          | $\log_2(K)$                   |
| $H_{W,\text{norm}}$               | Normalized wavelet entropy        | $H_W / H_{\max}$              |