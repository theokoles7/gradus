[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](../../README.md) / [curricula](../README.md) / [metrics](./README.md) / time-to-saturation

# Time-to-Saturation (TTS)

## Definition

Time-to-saturation is a model-informed difficulty metric that measures how many training iterations a self-supervised autoencoder requires before the L2 norm of per-layer weight updates falls below a stability threshold across all layers. Where [time-to-convergence](./convergence-time.md) tracks loss stabilization, TTS specifically tracks *weight saturation* — the point at which the model's internal representations stop meaningfully changing in response to the sample.

Samples that saturate quickly are easy for the autoencoder to encode, indicating low structural complexity. Samples that saturate slowly contain detail the model struggles to capture, indicating high difficulty. TTS is task-agnostic and label-free, making it applicable to any image dataset without requiring class annotations.

## Formula

Let $\Delta_l^{(t)}$ denote the Frobenius norm of the weight update for layer $l$ at iteration $t$:

$$\Delta_l^{(t)} = \left\| W_l^{(t)} - W_l^{(t-1)} \right\|_F$$

Layer $l$ is considered saturated at the first iteration $t^*_l$ such that $\Delta_l^{(t)} < \delta$ for $W$ consecutive iterations:

$$t^*_l = \min \left\{ t : \Delta_l^{(t')} < \delta \quad \forall\, t' \in [t, t + W] \right\}$$

The overall TTS for sample $x$ is reported as the iteration at which the *last* layer saturates:

$$\text{TTS}(x) = \max_l\ t^*_l$$

## Interpretation & Range

| Value        | Interpretation                                                                            |
|--------------|-------------------------------------------------------------------------------------------|
| Low TTS      | Easy sample - weight updates stabilize quickly; low structural complexity                 |
| Moderate TTS | Typical sample - autoencoder requires moderate training before weights settle             |
| High TTS     | Hard sample - weight updates remain large for many iterations; high structural complexity |

Range is $[1, T_{\max}]$. Samples for which any layer never saturates within $T_{\max}$ iterations are assigned $\text{TTS} = T_{\max}$.

## Parameters

| Parameter         | Type      | Default   | Description                                                           |
|---                |---        |---        |---                                                                    |
| `max_iterations`  | `int`     | `1000`    | Maximum iterations before abandoning measurement                      |
| `threshold`       | `float`   | `1e-3`    | Frobenius norm delta below which a layer is considered stable         |
| `window`          | `int`     | `5`       | Consecutive stable iterations required to declare a layer saturated   |
| `learning_rate`   | `float`   | `0.05`    | SGD learning rate for the proxy autoencoder                           |

## Notes

- TTS was selected over [TTC](./convergence-time.md) as the primary model-informed difficulty anchor for Gradus experiments based on stronger empirical correlation with visual complexity metrics (color variance, edge density, spatial frequency, wavelet energy, wavelet entropy) across MNIST, CIFAR-10, and CIFAR-100.
- Saturation is tracked per-layer. Layers that saturate early are removed from subsequent weight snapshot computations, reducing overhead as training progresses. Only layers with learnable weights (`weight` attribute, `requires_grad=True`) are tracked.
- A fresh autoencoder is instantiated for each sample. TTS is therefore a pre-training, per-sample measurement rather than a dynamic training signal.
- Like TTC, TTS computation is expensive. Score caching is essential for datasets of any meaningful size.
- The per-layer saturation iteration is available via `layer_saturation_iters` after computation, which can be useful for diagnosing which layers are driving high TTS values for particular samples.

## References

- [Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning (ICML)*, 41–48.](https://dl.acm.org/doi/pdf/10.1145/1553374.1553380)
- [Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. *Science*, 313(5786), 504–507.](https://www.science.org/doi/pdf/10.1126/science.1127647?casa_token=R4vQf-MuivMAAAAA:u4a7WTGmO6FUuLtQc12gBeJKkK55YU2IkVMFIlh4VbfrVdd_yzQBNfiX2qdebYdvVex68om9PfjN2_U)

---

## Formula Summary

| Symbol            | Description                               | Formula                                                       |
|-------------------|------------------------------------------ |---------------------------------------------------------------|
| $\Delta_l^{(t)}$  | Frobenius norm of layer $l$ weight delta  | $\| W_l^{(t)} - W_l^{(t-1)} \|_F$                             |
| $t^*_l$           | Saturation iteration for layer $l$        | $\min \{ t : \Delta_l^{(t')} < \delta,\ W \text{ times} \}$   |
| $\text{TTS}(x)$   | Time-to-saturation                        | $\max_l\ t^*_l$                                               |