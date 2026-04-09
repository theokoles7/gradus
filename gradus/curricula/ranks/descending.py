"""# gradus.curricula.ranks.descending

Descending ranking implementation.
"""

__all__ = ["Descending"]

from pathlib                            import Path
from typing                             import List, override, Union

from pandas                             import DataFrame

from gradus.curricula.ranks.protocol    import Rank
from gradus.registration                import register_rank

@register_rank(
    id =    "descending",
    tags =  ["monotonic"]
)
class Descending(Rank):
    """# Descending Curriculum Rank"""

    def __init__(self,
        metric:     Union[str, List[str]],
        dataset_id: str,
        scores:     DataFrame,
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Descending Curriculum Ranking.

        ## Args:
            * metric        (str | List[str]):  Metric(s) by which ranking will be determined.
            * dataset_id    (str):              Identifier of dataset whose samples are being 
                                                ranked.
            * scores        (DataFrame):        Dataset metric scores.
            * seed          (int):              Random number generation seed. Defaults to 1.
            * cache_dir     (str | Path):       Directory under which keyed indices will be cached. 
                                                Defaults to "./.cache/ranks/".
        """
        # Define properties.
        self._metric_:  List[str] = [metric] if isinstance(metric, str) else metric

        # If more than one metric is specified...
        if len(self._metric_) > 1:  raise ValueError(
                                        f"Descending ranking only supports a single metric;"
                                        f"got {len(self._metric_)}"
                                    )

        # Initialize protocol.
        super(Descending, self).__init__(
            rank_id =       "descending",
            dataset_id =    dataset_id,
            scores =        scores,
            seed =          seed,
            cache_dir =     cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices in Descending Order According to Specified Metric.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        return self._scores_.sort_values(by = self._metric_, ascending = False)["index"].tolist()