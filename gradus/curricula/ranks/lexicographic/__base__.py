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
        metric:     Union[str, List[str]],
        dataset_id: str,
        scores:     DataFrame,
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Lexicographic Curriculum Ranking.

        ## Args:
            * metric        (str | List[str]):  Metrics by which ranking will be determined.
            * dataset_id    (str):              Identifier of dataset whose samples are being 
                                                ranked.
            * scores        (DataFrame):        Dataset metric scores.
            * seed          (int):              Random number generation seed. Defaults to 1.
            * cache_dir     (str | Path):       Directory under which keyed indices will be cached. 
                                                Defaults to "./.cache/ranks/".
        """
        # Define properties.
        self._metric_:  List[str] = [metric] if isinstance(metric, str) else metric

        # Initialize protocol.
        super(Lexicographic, self).__init__(
            rank_id =       "lexicographic",
            dataset_id =    dataset_id,
            scores =        scores,
            seed =          seed,
            cache_dir =     cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices in Lexicographic Order According to Specified Metrics.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        return self._scores_.sort_values(by = self._metric_)["index"].tolist()