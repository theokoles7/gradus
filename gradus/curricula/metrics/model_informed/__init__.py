"""# gradus.curricula.metrics.model_informed

Image sample metrics that are informed by model learning activity.
"""

__all__ =   [
                # Classes
                "TimeToConvergence",
                "TimeToSaturation",
            ]

from gradus.curricula.metrics.model_informed.convergence_time   import *
from gradus.curricula.metrics.model_informed.saturation_time    import *