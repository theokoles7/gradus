"""# gradus.scheduling.ranks

Curriculum ranking sheme implementations.
"""

__all__ =   [
                "ascending",
                "descending",
                "distance_from_mean",
            ]

from gradus.scheduling.ranks.ascending          import ascending
from gradus.scheduling.ranks.descending         import descending
from gradus.scheduling.ranks.distance_from_mean import distance_from_mean