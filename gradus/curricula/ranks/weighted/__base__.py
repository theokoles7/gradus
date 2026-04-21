"""# gradus.curricula.ranks.weighted

Weighted composite metrics ranking implementation.
"""

__all__ = ["Weighted"]

from pathlib                            import Path
from typing                             import List, override, Union

from pandas                             import DataFrame

from gradus.curricula.ranks.protocol    import Rank
from gradus.registration                import register_rank

@register_rank(
    id =    "weighted",
    tags =  ["composite"]
)
class Weighted(Rank):
    """# Weighted Curriculum Ranking"""

    def __init__(self,
        metric:     Union[str, List[str]],
        dataset_id: str,
        scores:     DataFrame,
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Weighted Curriculum Ranking.

        ## Args:
            * scores    (DataFrame):        Metric scores sheet.
            * metric    (str | List[str]):  Metric by which sample indices should be anchored. This 
                                            serves as the anchor metric, upon which Peasron 
                                            correlation will be determined for weighting.
            * cache_dir (str | Path):       Path at which ranked indices will be cached for future 
                                            use. Defaults to ".cache/ranks".
        """
        # Define metric.
        self._metric_:  List[str] = [metric] if isinstance(metric, str) else metric

        # If more than one metric is specified...
        if len(self._metric_) > 1:  raise ValueError(
                                        f"Weighted ranking only supports a single metric anchor;"
                                        f"got {len(self._metric_)}"
                                    )

        # Initialize protocol.
        super(Weighted, self).__init__(
            rank_id =   "weighted",
            dataset_id =    dataset_id,
            scores =        scores,
            seed =          seed,
            cache_dir =     cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices in Weighted Order According to Specified Metric.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        from pandas import Series

        # Identify all numeric columns excluding index.
        numerics:       List[str] = [
                                        col for col in self._scores_.columns
                                        if col != "index"
                                        and self._scores_[col].dtype in ("float64", "int64")
                                    ]
        
        # Z-score normalize all numeric columns.
        normalized:     DataFrame = self._scores_[numerics].apply(
                                        lambda col: (col - col.mean()) / col.std()
                                    )
        
        # Compute Pearson correlation of all metrics against anchor.
        correlations:   Series =    normalized.corrwith(
                                        normalized[self._metric_[0]]
                                    ).drop(self._metric_)
        
        # Take absolute value & normalize to sum to 1.
        weights:        Series =    correlations.abs()
        weights =                   weights / weights.sum()

        # Compute weighted composite score (anchor gets weight 1.0, others are relative).
        composite:      Series =    normalized[self._metric_[0]] + \
                                    normalized[weights.index].mul(weights).sum(axis = 1)
        
        # Assign composite score and sort.
        return                      self._scores_.assign(composite = composite) \
                                    .sort_values("composite")["index"].tolist()