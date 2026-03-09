[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / edge-density

# Edge Density

## Definition

The edge density metric measures the proportion of pixels in an image that are classified as edges. Edges correspond to sharp transitions in pixel intensity — boundaries between objects, texture gradients, or fine structural detail. Images with many edges tend to be visually complex and informationally rich, making them harder for a model to learn from early in training. Edge density is computed by applying a gradient-based or threshold-based edge detector and counting the resulting edge pixels relative to the total image area.

## Formula

Let $E(x, y) \in \{0, 1\}$ denote the binary edge map produced by an edge detector (e.g. Canny), where $E(x, y) = 1$ if pixel $(x, y)$ is an edge pixel. For an image of $H \times W$ pixels:

$$\text{Edge Density} = \frac{1}{H \cdot W} \sum_{x=1}^{H} \sum_{y=1}^{W} E(x, y)$$

The gradient magnitude used by edge detectors such as Sobel is:

$$G = \sqrt{G_x^2 + G_y^2}$$

where $G_x$ and $G_y$ are the horizontal and vertical gradient responses respectively.

## Interpretation & Range

| Value             | Interpretation                                                        |
|-------------------|-----------------------------------------------------------------------|
| $\approx 0$       | Smooth, uniform image — very few edges; low structural complexity     |
| $0.1$–$0.3$       | Moderate structure — distinct objects or clear boundaries             |
| $> 0.3$           | High complexity — dense textures, fine detail, or busy scenes         |

Range is $[0, 1]$, representing the fraction of pixels identified as edges.

## References

- Canny, J. (1986). A computational approach to edge detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 8(6), 679–698.
- Sobel, I., & Feldman, G. (1968). A 3x3 isotropic gradient operator for image processing. *Stanford Artificial Intelligence Project Technical Report*.

---

## Formula Summary

| Symbol        | Description                                   | Formula                           |
|---------------|-----------------------------------------------|-----------------------------------|
| $G_x, G_y$    | Horizontal and vertical gradient responses    | Sobel convolution                 |
| $G$           | Gradient magnitude                            | $\sqrt{G_x^2 + G_y^2}$            |
| $E(x,y)$      | Binary edge map                               | $1$ if $G > \tau$, else $0$       |
| Edge Density  | Fraction of edge pixels                       | $\frac{1}{HW} \sum_{x,y} E(x,y)$  |