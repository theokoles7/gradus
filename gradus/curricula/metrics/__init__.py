"""# gradus.curricula.metrics

Image complexity quantification & metrics.
"""

__all__ =   [
                # Protocol
                "Metric",

                # Complexity
                "ColorVariance",
                "CompressionRatio",
                "EdgeDensity",
                "SpatialFrequency",
                "WaveletEnergy",
                "WaveletEntropy",

                # Model-informed
                "TimeToConvergence",
                "TimeToSaturation",
            ]

from gradus.curricula.metrics.complexity        import *
from gradus.curricula.metrics.model_informed    import *
from gradus.curricula.metrics.protocol          import Metric