"""# gradus.curricula.ranks.normalized_mean

Normalized mean composite ranking implementation.
"""

__all__ = ["NormalizedMean"]

from pathlib                            import Path
from typing                             import List, override, Union

from pandas                             import DataFrame

from gradus.curricula.ranks.protocol    import Rank
from gradus.registration                import METRIC_REGISTRY, register_rank

@register_rank(
    id =    "normalized-mean",
    tags =  ["composite"]
)
class NormalizedMean(Rank):
    """# Normalized Mean Curriculum Ranking

    Ranks samples by the equal-weighted mean of min-max normalized metric
    columns. Metrics tagged as inverted in the registry are flipped
    (1 - normalized) so that all columns align on a unified scale where
    higher = more complex. Samples are sorted ascending - lowest mean
    complexity score first.
    """

    def __init__(self,
        dataset_id: str,
        scores:     DataFrame,
        metric:     Union[str, List[str]],
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Normalized Mean Curriculum Ranking.

        ## Args:
            * dataset_id    (str):                      Identifier of dataset whose samples are 
                                                        being ranked.
            * scores        (DataFrame):                Dataset metric scores.
            * metric        (str | List[str] | None):   Metric columns to include.
            * seed          (int):                      Random number generation seed. Defaults to 1.
            * cache_dir     (str | Path):               Directory under which keyed indices will be 
                                                        cached. Defaults to "./.cache/ranks/".
        """
        # Define properties.
        self._metric_:  List[str] = [metric] if isinstance(metric, str) else metric

        # Initialize protocol.
        super(NormalizedMean, self).__init__(
            rank_id =       "normalized-mean",
            dataset_id =    dataset_id,
            scores =        scores,
            seed =          seed,
            cache_dir =     cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices by Normalized Mean Complexity Score.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        from pandas import DataFrame, Series

        # Take note of inverted metrics.
        INVERTED:   List[str] = METRIC_REGISTRY.list_entries(filter_by = ["inverted"])

        # Extract and copy relevant columns.
        normalized:     DataFrame = self._scores_[self._metric_].copy().astype(float)

        # Min-max normalize each column.
        for col in self._metric_:
            col_min:    float =     normalized[col].min()
            col_max:    float =     normalized[col].max()
            col_range:  float =     col_max - col_min

            # Avoid division by zero for constant columns.
            if col_range < 1e-10:   normalized[col] = 0.0
            else:                   normalized[col] = (normalized[col] - col_min) / col_range

        # For any inverted metrics...
        for col in INVERTED:

            # Invert the values, such that higher = more complex.
            if col in self._metric_:         normalized[col] = 1.0 - normalized[col]

        # Compute equal-weighted mean across all columns as the composite score.
        composite:      Series =    normalized.mean(axis = 1)

        # Sort ascending - lowest composite score = simplest samples first.
        return  self._scores_.assign(composite = composite) \
                .sort_values("composite")["index"].tolist()