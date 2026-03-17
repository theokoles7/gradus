"""# gradus.curricula.scheduling.ranks

Curriculum ranking sheme implementations.
"""

__all__ =   [
                "ascending",
                "descending",
                "distance_from_mean",
            ]

from gradus.curricula.scheduling.ranks.ascending            import ascending
from gradus.curricula.scheduling.ranks.descending           import descending
from gradus.curricula.scheduling.ranks.distance_from_mean   import distance_from_mean