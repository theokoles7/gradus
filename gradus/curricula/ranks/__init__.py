"""# gradus.curricula.ranks

Curriculum ranking sheme implementations.
"""

__all__ =   [
                # Protocol
                "Rank",

                # Concrete
                "Ascending",
                "Descending", 
                "DistanceFromMean",
                "Lexicographic",
                "NormalizedMean",
                "PairwiseCorrelation",
                "Weighted",
            ]

from gradus.curricula.ranks.ascending               import Ascending
from gradus.curricula.ranks.descending              import Descending
from gradus.curricula.ranks.distance_from_mean      import DistanceFromMean
from gradus.curricula.ranks.lexicographic           import Lexicographic
from gradus.curricula.ranks.normalized_mean         import NormalizedMean
from gradus.curricula.ranks.pairwise_correlation    import PairwiseCorrelation
from gradus.curricula.ranks.protocol                import Rank
from gradus.curricula.ranks.weighted                import Weighted