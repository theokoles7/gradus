"""# gradus.curricula.ranks.distance_from_mean

Distance from mean ranking implementation.
"""

__all__ = ["DistanceFromMean"]

from pathlib                            import Path
from typing                             import List, override, Union

from pandas                             import DataFrame

from gradus.curricula.ranks.protocol    import Rank
from gradus.registration                import register_rank

@register_rank(
    id =    "distance-from-mean",
    tags =  ["deviation", "distance-based"]
)
class DistanceFromMean(Rank):
    """# Distance from Mean Curriculum Rank"""

    def __init__(self,
        metric:     Union[str, List[str]],
        dataset_id: str,
        scores:     DataFrame,
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Distance from Mean Curriculum Ranking.

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
        if len(self._metric_) > 1: raise ValueError(
                                        f"Distance from mean ranking only supports a single metric;"
                                        f"got {len(self._metric_)}"
                                    )

        # Initialize protocol.
        super(DistanceFromMean, self).__init__(
            rank_id =       "distance-from-mean",
            dataset_id =    dataset_id,
            scores =        scores,
            seed =          seed,
            cache_dir =     cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices by Absolute Deviation from Dataset Mean Score.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        # Compute mean score value.
        mean: float = self._scores_[self._metric_[0]].mean()
    
        # Provide indices in order of absolute distance from the calculated mean.
        return  self._scores_.assign(distance = (self._scores_[self._metric_[0]] - mean).abs()) \
                .sort_values("distance")["index"].tolist()