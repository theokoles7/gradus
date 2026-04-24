[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / time-to-convergence

# Time-to-Convergence (TTC)

## Definition

Time-to-convergence is a model-informed difficulty metric that measures how many training iterations a model requires before its loss on a given sample stabilizes. Unlike intrinsic metrics that assess image properties independently of any model, TTC captures difficulty as experienced by a specific network - a sample is considered hard if the model's per-sample loss takes many iterations to stop decreasing meaningfully. Samples that converge quickly are well-handled early in training; samples that converge slowly represent persistent challenges that may benefit from delayed or repeated presentation.

TTC is estimated by training a lightweight proxy autoencoder on a single sample and measuring the iteration at which the per-sample loss falls within a tolerance $\epsilon$ of a stable minimum and remains there for a patience window $W$.

## Formula

Let $\mathcal{L}^{(t)}(x)$ denote the loss of a model on sample $x$ at training iteration $t$. Define the convergence criterion with tolerance $\epsilon > 0$ and patience window $W$:

$$\text{TTC}(x) = \min \left\{ t : \left| \mathcal{L}^{(t')}(x) - \mathcal{L}^{(t)}(x) \right| < \epsilon \quad \forall\, t' \in [t, t + W] \right\}$$

In practice, TTC is estimated as the first iteration at which the absolute loss delta falls below $\epsilon$ and remains there for $W$ consecutive iterations:

$$\text{TTC}(x) \approx \min \left\{ t : \left| \mathcal{L}^{(t)}(x) - \mathcal{L}^{(t-1)}(x) \right| < \epsilon \text{ for } W \text{ consecutive iterations} \right\}$$

## Interpretation & Range

| Value        | Interpretation                                                                        |
|--------------|---------------------------------------------------------------------------------------|
| Low TTC      | Easy sample - model loss stabilizes quickly; well within the model's current capacity |
| Moderate TTC | Typical sample - requires meaningful training signal before converging                |
| High TTC     | Hard sample - loss remains unstable for many iterations; persistent difficulty        |

Range is $[1, T_{\max}]$ where $T_{\max}$ is the maximum number of iterations allowed. Samples that never meet the convergence criterion within $T_{\max}$ are assigned $\text{TTC} = T_{\max}$.

## Parameters

| Parameter        | Type    | Default | Description |
|---|---|---|---|
| `max_iterations` | `int`   | `1000`  | Maximum iterations before abandoning measurement |
| `threshold`      | `float` | `1e-3`  | Loss delta below which an iteration is considered stable |
| `window`         | `int`   | `5`     | Consecutive stable iterations required to declare convergence |
| `learning_rate`  | `float` | `0.05`  | SGD learning rate for the proxy autoencoder |

## Notes

- TTC and [Time-to-Saturation (TTS)](./saturation-time.md) both use a lightweight proxy autoencoder, but they track different signals. TTC tracks loss stabilization — when the per-sample reconstruction error stops changing. TTS tracks weight saturation — when the per-layer weight update norms fall below threshold. In practice, TTS was found to correlate more reliably with visual complexity metrics and was selected as the primary model-informed difficulty anchor for Gradus experiments.
- A fresh autoencoder is instantiated for each sample. This means TTC captures how quickly a model learns a specific sample *from scratch*, not how difficult the sample is relative to a model that has already seen other data. This is the intended behavior for a pre-training difficulty assessment.
- Samples that never converge within `max_iterations` are assigned `TTC = max_iterations`. These samples are treated as maximally difficult, but it is worth verifying that the `threshold` and `learning_rate` are well-calibrated for the dataset before interpreting maxed-out TTC values.
- TTC computation is expensive: each sample requires up to `max_iterations` forward and backward passes on the proxy autoencoder. Score caching is strongly recommended for any dataset larger than a few thousand samples.

## References

- [Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.](https://dl.acm.org/doi/pdf/10.1145/1553374.1553380)
- [Hacohen, G., & Weinshall, D. (2019). On the power of curriculum learning in training deep networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.](https://proceedings.mlr.press/v97/hacohen19a/hacohen19a.pdf)

---

## Formula Summary

| Symbol                    | Description                       | Formula                                                                                           |
|---                        |---                                |---                                                                                                |
| $\mathcal{L}^{(t)}(x)$    | Per-sample loss at iteration $t$  | Model forward pass + MSE loss                                                                     |
| $\text{TTC}(x)$           | Time-to-convergence               | $\min \{ t : \|\mathcal{L}^{(t)}(x) - \mathcal{L}^{(t-1)}(x)\| < \epsilon,\ W \text{ times} \}$   |