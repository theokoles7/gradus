"""# gradus.curricula.ranks.lexicographic

Lexicographic composite metrics ranking implementation.
"""

__all__ = ["Lexicographic"]

from pathlib                            import Path
from typing                             import List, override, Union

from pandas                             import DataFrame

from gradus.curricula.ranks.protocol    import Rank
from gradus.registration                import register_rank

@register_rank(
    id =    "lexicographic",
    tags =  ["composite"]
)
class Lexicographic(Rank):
    """# Lexicographic Curriculum Ranking"""

    def __init__(self,
        scores:     DataFrame,
        metric:     Union[str, List[str]],
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Lexicographic Curriculum Ranking.

        ## Args:
            * scores    (DataFrame):        Metric scores sheet.
            * metric    (str | List[str]):  Metrics by which sample indices should be ranked.
            * cache_dir (str | Path):       Path at which ranked indices will be cached for future 
                                            use. Defaults to ".cache/ranks".
        """
        # Define properties.
        self._metric_:  List[str] = [metric] if isinstance(metric, str) else metric

        # Initialize protocol.
        super(Lexicographic, self).__init__(
            rank_id =   "lexicographic",
            scores =    scores,
            cache_dir = cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices in Lexicographic Order According to Specified Metrics.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        return self._scores_.sort_values(by = self._metric_)["index"].tolist()