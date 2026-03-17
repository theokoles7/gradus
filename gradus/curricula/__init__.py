"""# gradus.curricula

Curriculum components & utilities.
"""

__all__ =   [
                # Metrics --------------------------------------------------------------------------
                
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

                # Scheduling -----------------------------------------------------------------------

                # Ranking
                "ascending",
                "descending",
                "distance_from_mean",

                # Samplers
                "CurriculumBatchSampler",
                "CurriculumDatasetSampler",
            ]

from gradus.curricula.metrics       import *
from gradus.curricula.scheduling    import *