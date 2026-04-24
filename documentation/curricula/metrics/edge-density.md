[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / edge-density

# Edge Density

## Definition

The edge density metric measures the proportion of pixels in an image that are classified as edges. Edges correspond to sharp transitions in pixel intensity - boundaries between objects, texture gradients, or fine structural detail. Images with many edges tend to be visually complex and informationally rich, making them harder for a model to learn from early in training. Edge density is computed by applying the Canny edge detector and counting the resulting edge pixels relative to the total image area.

## Formula

Let $E(x, y) \in \{0, 1\}$ denote the binary edge map produced by the Canny detector, where $E(x, y) = 1$ if pixel $(x, y)$ is an edge pixel. For an image of $H \times W$ pixels:

$$\text{Edge Density} = \frac{1}{H \cdot W} \sum_{x=1}^{H} \sum_{y=1}^{W} E(x, y)$$

The gradient magnitude used internally by the Canny detector is:

$$G = \sqrt{G_x^2 + G_y^2}$$

where $G_x$ and $G_y$ are the horizontal and vertical Sobel gradient responses respectively.

## Interpretation & Range

| Value       | Interpretation                                                        |
|-------------|-----------------------------------------------------------------------|
| $\approx 0$ | Smooth, uniform image - very few edges; low structural complexity     |
| $0.1$–$0.3$ | Moderate structure - distinct objects or clear boundaries             |
| $> 0.3$     | High complexity - dense textures, fine detail, or busy scenes         |

Range is $[0, 1]$, representing the fraction of pixels identified as edges.

## Parameters

| Parameter | Type  | Default   | Description                                       |
|---        |---    |---        |---                                                |
| `low`     | `int` | `100`     | Canny low threshold for hysteresis edge linking   |
| `high`    | `int` | `200`     | Canny high threshold for strong edge detection    |

## Notes

- Edge density and [Object Count](./object-count.md) share the same Canny-based edge detection pipeline. Edge density measures *how much* edge is present; object count uses connected components on the same edge map to estimate *how many* distinct objects are present. Both metrics respond to the same `low` and `high` threshold parameters.
- Multi-channel inputs are converted to grayscale before edge detection, so edge density captures luminance boundaries rather than color boundaries specifically.
- The Canny thresholds interact with image scale. The defaults (100/200) are calibrated for 8-bit images with pixel values in $[0, 255]$. If inputs are normalized to $[0, 1]$, the thresholds should be scaled accordingly, or the metric will detect very few or no edges. On small images such as CIFAR-10 ($32 \times 32$), edge density values tend to cluster in a narrow range because there are fewer pixels to distinguish between sparse and dense edge distributions. On larger images the metric is more discriminative.

## References

- [Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.](https://d1wqtxts1xzle7.cloudfront.net/64038952/1-libre.pdf?1595942484=&response-content-disposition=inline%3B+filename%3D10_Important_AI_Research_Papers.pdf&Expires=1777066966&Signature=cYivbiRGrJuud~MAzaxQBvuNYo627XJ7GR-rHD6DMe23H4Fo5d~jPwxDzwJ2OyK3QmGhkewzGInOQn4T4NQ~w~wI5RdVL2dsVIGTWpfcY2IhmsAjz4rV1sMNgoMfDQgmyhyp3tBpV3Ck8gehVEJ1zszlbOctDMI38AHTy-D7NMfQzrvBLKJGUjtMl2RmKiJXmn3lK~iEGBEKrmkOO3J6Jg~UkuRsFM3QvoduxD1F~oQDX0iH5mf3pxAPVggOA~Tax7NgZnAvYtvONfnD4S8L5kjfa4P7ru2Mn~iXivxTv0Uzr-PP~a40~UdjGetPg9gWc4U7OOZoYLDLUhcuqNUswA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- Sobel, I., & Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. *Stanford Artificial Intelligence Project Technical Report*.

---

## Formula Summary

| Symbol       | Description                                | Formula                           |
|--------------|--------------------------------------------|-----------------------------------|
| $G_x, G_y$   | Horizontal and vertical gradient responses | Sobel convolution                 |
| $G$          | Gradient magnitude                         | $\sqrt{G_x^2 + G_y^2}$            |
| $E(x,y)$     | Binary edge map                            | $1$ if $G > \tau$, else $0$       |
| Edge Density | Fraction of edge pixels                    | $\frac{1}{HW} \sum_{x,y} E(x,y)$  |