[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / time-to-convergence

# Time-to-Convergence (TTC)

## Definition

Time-to-convergence is a model-informed difficulty metric that measures how many training iterations a model requires before its loss on a given sample stabilizes. Unlike intrinsic metrics that assess image properties independently of any model, TTC captures difficulty as experienced by a specific network — a sample is considered hard if the model's per-sample loss takes many iterations to stop decreasing meaningfully. Samples that converge quickly are well-handled early in training; samples that converge slowly represent persistent challenges that may benefit from delayed or repeated presentation.

TTC is estimated by training a lightweight proxy model on individual samples (or small batches) and measuring the iteration at which the per-sample loss falls within a tolerance $\epsilon$ of a stable minimum and remains there.

## Formula

Let $\mathcal{L}^{(t)}(x)$ denote the loss of a model on sample $x$ at training iteration $t$. Define the convergence criterion with tolerance $\epsilon > 0$ and patience window $W$:

$$\text{TTC}(x) = \min \left\{ t : \left| \mathcal{L}^{(t')}(x) - \mathcal{L}^{(t)}(x) \right| < \epsilon \quad \forall\, t' \in [t, t + W] \right\}$$

In practice, TTC is estimated as the first iteration at which the exponentially weighted moving average (EWMA) of the loss gradient drops below $\epsilon$:

$$\hat{\mathcal{L}}^{(t)} = \alpha \cdot \mathcal{L}^{(t)} + (1 - \alpha) \cdot \hat{\mathcal{L}}^{(t-1)}$$

$$\text{TTC}(x) \approx \min \left\{ t : \left| \hat{\mathcal{L}}^{(t)} - \hat{\mathcal{L}}^{(t-1)} \right| < \epsilon \right\}$$

## Interpretation & Range

| Value         | Interpretation                                                                        |
|---------------|---------------------------------------------------------------------------------------|
| Low TTC       | Easy sample — model loss stabilizes quickly; well within the model's current capacity |
| Moderate TTC  | Typical sample — requires meaningful training signal before converging                |
| High TTC      | Hard sample — loss remains unstable for many iterations; persistent difficulty        |

Range is $[1, T_{\max}]$ where $T_{\max}$ is the maximum number of iterations allowed. Samples that never meet the convergence criterion within $T_{\max}$ are assigned $\text{TTC} = T_{\max}$.

## References

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.
- Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

---

## Formula Summary

| Symbol                    | Description                       | Formula                                                                               |
|---------------------------|-----------------------------------|---------------------------------------------------------------------------------------|
| $\mathcal{L}^{(t)}(x)$    | Per-sample loss at iteration $t$  | Model forward pass + loss function                                                    |
| $\hat{\mathcal{L}}^{(t)}$ | EWMA-smoothed loss                | $\alpha \cdot \mathcal{L}^{(t)} + (1-\alpha) \cdot \hat{\mathcal{L}}^{(t-1)}$         |
| $\text{TTC}(x)$           | Time-to-convergence               | $\min \{ t : \|\hat{\mathcal{L}}^{(t)} - \hat{\mathcal{L}}^{(t-1)}\| < \epsilon \}$   |