"""# gradus.curricula.ranks

Curriculum ranking sheme implementations.
"""

__all__ =   [
                "ascending",
                "descending",
                "distance_from_mean",
            ]

from gradus.curricula.ranks.ascending           import ascending
from gradus.curricula.ranks.descending          import descending
from gradus.curricula.ranks.distance_from_mean  import distance_from_mean