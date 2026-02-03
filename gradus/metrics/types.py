"""# gradus.metrics.types

Image complexity & quantification artifact/object types.
"""

__all__ =   [
                "ArrayLike",
                "ChannelMode",
                "WaveletEnergyResult",
                "WaveletEntropyResult",
            ]

from dataclasses    import dataclass
from typing         import Literal, Union, Tuple

from numpy.typing   import NDArray


ArrayLike =     Union[NDArray]
ChannelMode =   Literal["grayscale", "luminance", "per_channel"]


@dataclass(frozen = True)
class WaveletEnergyResult():
    """# Wavelet Energy Calculation Data"""
    total_energy:                   float
    approximate_energy:             float
    detail_energy_total:            float
    detail_energy_by_level:         Tuple[float, ...]
    approximate_coefficient_shape:  Tuple[int, int]
    levels:                         int
    wavelet:                        str
    mode:                           str


@dataclass(frozen = True)
class WaveletEntropyResult():
    """# Wavelet Entropy Calculation Data"""
    entropy:            float
    normalized_entropy: float
    probabilities:      Tuple[float, ...]
    energies:           Tuple[float, ...]
    levels:             int
    wavelet:            str
    mode:               str