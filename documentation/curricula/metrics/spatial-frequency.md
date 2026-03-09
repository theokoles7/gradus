[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / spatial-frequency

# Spatial Frequency

## Definition

The spatial frequency metric quantifies the rate of intensity variation across an image by measuring the average magnitude of pixel-level gradients. Images dominated by low spatial frequencies contain slow, smooth transitions — solid regions or gentle gradients. Images with high spatial frequencies exhibit rapid changes — sharp edges, fine textures, and intricate patterns. As a difficulty measure, spatial frequency captures the density of fine-grained information that a model must resolve to correctly classify or represent the image.

## Formula

Let $I(x, y)$ denote the intensity at pixel $(x, y)$ in an image of size $H \times W$. The row frequency $RF$ and column frequency $CF$ are computed as the root-mean-square of horizontal and vertical intensity differences respectively:

$$RF = \sqrt{\frac{1}{H \cdot W} \sum_{x=1}^{H} \sum_{y=1}^{W-1} \left[ I(x, y+1) - I(x, y) \right]^2}$$

$$CF = \sqrt{\frac{1}{H \cdot W} \sum_{x=1}^{H-1} \sum_{y=1}^{W} \left[ I(x+1, y) - I(x, y) \right]^2}$$

The overall spatial frequency is the combined magnitude:

$$SF = \sqrt{RF^2 + CF^2}$$

## Interpretation & Range

| Value         | Interpretation                                                |
|---------------|---------------------------------------------------------------|
| Low $SF$      | Smooth image — uniform regions, slow intensity transitions    |
| Moderate $SF$ | Structured image — distinct features, clear edges             |
| High $SF$     | Complex image — dense texture, fine detail, rapid variation   |

Range is $[0, \infty)$, though practical values for natural images are bounded by the maximum pixel intensity difference per step.

## References

- Mannos, J., & Sakrison, D. (1974). The effects of a visual fidelity criterion on the encoding of images. *IEEE Transactions on Information Theory*, 20(4), 525–536.
- Roberts, L. G. (1963). Machine perception of three-dimensional solids. *MIT Lincoln Laboratory Technical Report*.

---

## Formula Summary

| Symbol    | Description                   | Formula                                                   |
|-----------|-------------------------------|-----------------------------------------------------------|
| $RF$      | Row (horizontal) frequency    | $\sqrt{\frac{1}{HW} \sum_{x,y} [I(x,y+1) - I(x,y)]^2}$    |
| $CF$      | Column (vertical) frequency   | $\sqrt{\frac{1}{HW} \sum_{x,y} [I(x+1,y) - I(x,y)]^2}$    |
| $SF$      | Spatial frequency             | $\sqrt{RF^2 + CF^2}$                                      |