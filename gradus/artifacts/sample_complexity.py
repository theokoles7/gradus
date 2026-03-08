"""# gradus.artifacts.sample_complexity

Data structure implementation for storing individual sample complexity score(s).
"""

from dataclasses    import dataclass
from typing         import Union

@dataclass()
class SampleComplexity():
    """# Dataset Sample Complexity Scores"""

    # Sample properties.
    index:              int
    label:              Union[str, int]

    # Sample metrics.
    color_variance:     float
    compression_ratio:  float
    edge_density:       float
    spatial_freq:       float
    spatial_freq_col:   float
    spatial_freq_row:   float
    wavelet_energy:     float
    wavelet_entropy:    float