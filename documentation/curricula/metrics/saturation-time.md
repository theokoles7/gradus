[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / time-to-saturation

# Time-to-Saturation (TTS)

## Definition

Time-to-saturation is a model-informed difficulty metric that measures how many training iterations a self-supervised autoencoder requires before its reconstruction error on a given sample reaches a minimum and stops improving. Where time-to-convergence tracks loss stabilization, TTS specifically tracks the point of *saturation* — when further training yields no meaningful reduction in reconstruction error. Samples that saturate quickly are easy to reconstruct and thus low in structural complexity; samples that saturate slowly contain detail that the model struggles to faithfully reproduce, indicating high difficulty.

TTS is estimated using a lightweight autoencoder trained in a self-supervised fashion on individual samples. The metric is task-agnostic and label-free, making it applicable to any image dataset without requiring class annotations.

## Formula

Let $\mathcal{R}^{(t)}(x)$ denote the reconstruction error (e.g. mean squared error) of an autoencoder on sample $x$ at training iteration $t$. Saturation is defined as the first iteration at which the reconstruction error ceases to decrease by more than a threshold $\delta$:

$$\text{TTS}(x) = \min \left\{ t : \mathcal{R}^{(t)}(x) - \min_{t' \leq t} \mathcal{R}^{(t')}(x) < \delta \quad \text{and} \quad \mathcal{R}^{(t)}(x) \leq \mathcal{R}^{(t-1)}(x) + \delta \right\}$$

In practice, TTS is estimated as the iteration at which the minimum reconstruction error is first achieved and sustained:

$$\text{TTS}(x) \approx \underset{t}{\arg\min} \left\{ \mathcal{R}^{(t)}(x) \right\}$$

with early stopping applied when $\mathcal{R}^{(t)}(x) - \mathcal{R}^{(t-1)}(x) < \delta$ for $W$ consecutive iterations.

## Interpretation & Range

| Value         | Interpretation                                                                            |
|---------------|-------------------------------------------------------------------------------------------|
| Low TTS       | Easy sample — reconstruction error drops and saturates quickly; low structural complexity |
| Moderate TTS  | Typical sample — autoencoder requires moderate training to achieve good reconstruction    |
| High TTS      | Hard sample — reconstruction remains poor for many iterations; high structural complexity |

Range is $[1, T_{\max}]$. Samples that never saturate within $T_{\max}$ iterations are assigned $\text{TTS} = T_{\max}$.

## References

- Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504–507.

---

## Formula Summary

| Symbol                    | Description                           | Formula                                                   |
|---------------------------|---------------------------------------|-----------------------------------------------------------|
| $\mathcal{R}^{(t)}(x)$    | Reconstruction error at iteration $t$ | $\frac{1}{N}\sum_{i}(\hat{x}_i - x_i)^2$ (MSE)            |
| $\text{TTS}(x)$           | Time-to-saturation                    | $\arg\min_t \{ \mathcal{R}^{(t)}(x) \}$ with patience $W$ |