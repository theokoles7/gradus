[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / object-count

# Object Count

## Definition

The object count metric estimates the number of distinct objects in an image using Canny edge
detection followed by connected-component analysis on the resulting edge map. Images containing
many objects are generally more visually complex and structurally dense, presenting a broader
range of features that a model must resolve to correctly classify or represent the image.

Unlike pixel-level metrics such as [Edge Density](./edge-density.md) or
[Spatial Frequency](./spatial-frequency.md), object count operates at a higher level of
abstraction — it attempts to count semantically meaningful regions rather than measure
low-level signal properties. Because the estimate is derived from edge structure rather than
semantic segmentation, it is best interpreted as an approximation of object count rather than a
precise measurement.

## Procedure

1. The image tensor is converted to a grayscale `uint8` NDArray.
2. Canny edge detection is applied to produce a binary edge map.
3. The edge map is dilated with a $3 \times 3$ kernel to close small gaps between edge segments, connecting nearby contours that belong to the same object.
4. Connected-component labeling is applied to the dilated edge map.
5. The background label is subtracted, yielding the estimated object count.

## Formula

Let $E$ denote the binary edge map produced by the Canny detector and $D = \text{dilate}(E)$ the dilated map. The connected-component count is:

$$\text{ObjectCount} = \max(\text{labels}(D) - 1,\ 0)$$

where $\text{labels}(D)$ is the number of connected components in $D$ (including the background), and subtracting 1 removes the background component.

## Parameters

| Parameter | Type  | Default   | Description                                       |
|---        |---    |---        |---                                                |
| `low`     | `int` | `100`     | Canny low threshold for hysteresis edge linking   |
| `high`    | `int` | `200`     | Canny high threshold for strong edge detection    |

## Interpretation & Range

| Value | Interpretation                                    |
|---    |---                                                |
| $0$   | No detectable edges; uniform or featureless image |
| Low   | Few distinct regions; simple composition          |
| High  | Many distinct regions; dense or cluttered scene   |

Range is $[0, \infty)$ in theory, though practical values are bounded by image resolution and the Canny threshold configuration.

## Notes

- Object count returns an `int` rather than a `float`. Composite ranks that normalize metric columns handle this correctly, but downstream code that assumes `float` output from all metrics should cast accordingly.
- The Canny thresholds (`low`, `high`) strongly affect the estimate. Lower thresholds detect more edges and tend to inflate object counts; higher thresholds suppress weak edges and produce more conservative estimates.
- Dilation is applied with a fixed $3 \times 3$ kernel and one iteration. On very small images (e.g. CIFAR-10 at $32 \times 32$), this can over-merge nearby objects; on larger images it helps bridge legitimate contour gaps.
- Because connected components are counted on the *edge map* rather than on filled regions, objects with interior texture will be counted as a single component only if their edges form a closed contour after dilation.

## References

- [Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.](https://d1wqtxts1xzle7.cloudfront.net/64038952/1-libre.pdf?1595942484=&response-content-disposition=inline%3B+filename%3D10_Important_AI_Research_Papers.pdf&Expires=1777065972&Signature=bHLBf97FUj09Qs~Xi2gFVQNpyL5CzDfc1RBA610s6ka~WotjNHGLP~oIX9exvAT6ah2fEdBYSRAjt7g-JykJcvR2qwLMrI76D6sSIJ5TB1Ae7~MkXA3Veh983G5zbbACAOsS~vK3DvzHH2banKY1kuSMN3B-Og8cmVQTLVyhvp0WjRiO07pGy~E-FUanf6K2fTAdLjn-8k96MAjd6cNW5eyHpwYno2CTqyvEQ07uc-Os6o49iBtlT9OWLuIXTICqEmTR-eV8-bouqE99ZAnugQm-g5mSlsWrNIchxkH6jaoKJUAKyp7K4KWPywYYmhXD10MCZ-EZQdAfg4qsQM7SXw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- [He, L., Ren, X., Gao, Q., Zhao, X., Yao, B., & Chao, Y. (2017). The connected-components labeling problem: A review of state-of-the-art algorithms. *Pattern Recognition*, 70, 25–43.](https://www.sciencedirect.com/science/article/pii/S0031320317301693)

---

## Formula Summary

| Symbol        | Description               | Formula                                           |
|---            |---                        |---                                                |
| $E$           | Binary edge map           | Canny$(\text{image},\ \text{low},\ \text{high})$  |
| $D$           | Dilated edge map          | $\text{dilate}(E,\ 3 \times 3,\ \text{iter}=1)$   |
| ObjectCount   | Estimated object count    | $\max(\text{labels}(D) - 1,\ 0)$                  |