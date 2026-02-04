"""# gradus.metrics

Image complexity quantification & metrics utilities.
"""

__all__ =   [
                # Types
                "WaveletEnergyResult",
                "WaveletEntropyResult",

                # Calculations
                "color_variance",
                "compression_ratio",
                "edge_density",
                "spatial_frequency",
                "wavelet_energy",
                "wavelet_entropy",
            ]

from gradus.metrics.standard    import *
from gradus.metrics.wavelet     import *

from gradus.metrics.types       import WaveletEnergyResult, WaveletEntropyResult