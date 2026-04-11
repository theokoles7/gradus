# Complexity Metrics — Easy/Hard Ordering

| Metric | Source | Score Interpretation | Easy → Hard Direction |
|---|---|---|---|
| `color_variance` | gradus | Mean channel-wise pixel variance | Low → High (ascending) |
| `compression_ratio` | gradus | Original size / JPEG compressed size; higher ratio = image compresses well = simpler | High → Low (descending; inverted in combined score) |
| `edge_density` | gradus | Fraction of edge pixels via Canny detection | Low → High (ascending) |
| `spatial_frequency` | gradus | RMS of row/column pixel differences | Low → High (ascending) |
| `wavelet_energy` | gradus | Total wavelet decomposition energy | Low → High (ascending) |
| `wavelet_entropy` | gradus | Normalized Shannon entropy of wavelet energy distribution | Low → High (ascending) |
| `edge_object_count` | original | Canny edge detection → dilation → connected components; more regions = more complex | Low → High (ascending) |

## Combined Score

Each metric is min-max normalized to [0, 1]. `compression_ratio` is inverted (`1 − normalized`) so that all metrics align on a unified scale where **higher = more complex**. The combined complexity score is the equal-weighted mean of all seven normalized values.

## Curriculum Schedule

- Starts with the **30% easiest** samples (lowest combined complexity score).
- Linearly expands to **100%** of training data by epoch 120.
- Maintains **100%** of data from epoch 120 to 200.
- **No random shuffling** at any point — samples are always presented in strict easy-to-hard curriculum order.
