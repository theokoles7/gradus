"""# gradus.curricula

Curriculum construction/components module.
"""

__all__ =   [
                # Protocol
                "Curriculum",

                # METRICS ==========================================================================

                # Protocol
                "Metric",

                # Complexity
                "ColorVariance",
                "CompressionRatio",
                "EdgeDensity",
                "SpatialFrequency",
                "WaveletEnergy",
                "WaveletEntropy",

                "color_variance",
                "compression_ratio",
                "edge_density",
                "spatial_frequency",
                "wavelet_energy",
                "wavelet_entropy",

                # Model-informed
                "TimeToConvergence",
                "TimeToSaturation",
                
                "time_to_convergence",
                "time_to_saturation",

                # RANKS ============================================================================

                "Ascending",
                "Descending", 
                "DistanceFromMean",
                "Lexicographic",
                "Weighted",
            ]

from gradus.curricula.protocol  import Curriculum
from gradus.curricula.metrics   import *
from gradus.curricula.ranks     import *