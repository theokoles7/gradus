"""# gradus.metrics.model_informed

Image sample metrics that are informed by model learning activity.
"""

__all__ =   [
                # Classes
                "TimeToConvergence",
                "TimeToSaturation",
                
                # Quick-access functions
                "time_to_convergence",
                "time_to_saturation",
            ]

from gradus.metrics.model_informed.convergence_time import *
from gradus.metrics.model_informed.saturation_time  import *