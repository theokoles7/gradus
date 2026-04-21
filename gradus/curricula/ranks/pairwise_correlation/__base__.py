"""# gradus.curricula.ranks.pairwise_correlation

Pairwise correlation composite ranking implementation.
"""

__all__ = ["PairwiseCorrelation"]

from pathlib                            import Path
from typing                             import List, override, Union

from pandas                             import DataFrame

from gradus.curricula.ranks.protocol    import Rank
from gradus.registration                import METRIC_REGISTRY, register_rank

@register_rank(
    id =    "pairwise-correlation",
    tags =  ["composite"]
)
class PairwiseCorrelation(Rank):
    """# Pairwise Correlation Curriculum Ranking"""

    def __init__(self,
        dataset_id: str,
        scores:     DataFrame,
        metric:     Union[str, List[str]],
        cache_dir:  Union[str, Path] =      ".cache/ranks",
        seed:       int =                   1,
    ):
        """# Instantiate Pairwise Correlation Curriculum Ranking.

        ## Args:
            * dataset_id    (str):              Identifier of dataset whose samples are being 
                                                ranked.
            * scores        (DataFrame):        Dataset metric scores.
            * metric        (str | List[str])   Subset of metric columns to base correlation on.
            * cache_dir     (str | Path):       Directory under which keyed indices will be cached. 
                                                Defaults to "./.cache/ranks/".
            * seed          (int):              Random number generation seed. Defaults to 1.
        """
        # Define properties.
        self._metric_:  List[str] = [metric] if isinstance(metric, str) else metric

        # Initialize protocol.
        super(PairwiseCorrelation, self).__init__(
            rank_id =       "pairwise-correlation",
            dataset_id =    dataset_id,
            scores =        scores,
            seed =          seed,
            cache_dir =     cache_dir
        )

    # HELPERS ======================================================================================

    @override
    def _rank_(self) -> List[int]:
        """# Sort Indices by Pairwise Correlation According to Specified Matric(s).

        ## Returns:
            * List[int]:    Ranked indices.
        """
        from numpy          import argsort, nan_to_num
        from numpy.linalg   import norm
        from numpy.typing   import NDArray

        # Take note of inverted metrics.
        INVERTED:   List[str] = METRIC_REGISTRY.list_entries(filter_by = ["inverted"])

        # Extract attribute matrix.
        attributes: NDArray =    self._scores_[self._metric_].values.astype(float)

        # Min-max normalize each column.
        col_mins:   NDArray =    attributes.min(axis = 0)
        col_maxs:   NDArray =    attributes.max(axis = 0)
        col_ranges: NDArray =    col_maxs - col_mins

        # Avoid division by zero for constant columns.
        col_ranges[col_ranges < 1e-10] = 1.0

        # Normalize.
        normalized: NDArray =    (attributes - col_mins) / col_ranges

        # Invert necessary metrics so higher = more complex.
        for col in INVERTED:
            if col in self._metric_:
                normalized[:, self._metric_.index(col)] = 1.0 - normalized[:, self._metric_.index(col)]

        # Center each row and L2-normalize — enables Pearson via dot product.
        centered:   NDArray =    normalized - normalized.mean(axis = 1, keepdims = True)
        norms:      NDArray =    norm(centered, axis = 1, keepdims = True)

        # Avoid division by zero for constant attribute vectors.
        norms[norms < 1e-10] =      1.0
        X:          NDArray =       centered / norms

        # Initialize weight vector.
        n:          int =           len(X)
        W:          List[float] =   [0.0] * n

        # Pairwise correlation voting.
        for i in range(n):

            # Vectorized Pearson correlations of sample i against all j > i.
            corrs:      NDArray =       X[i] @ X[i + 1:].T
            corrs_list: List[float] =   nan_to_num(corrs, nan = 0.0).tolist()

            # Cache W[i] locally for inner loop performance.
            wi:         float =         W[i]

            for idx, P_ij in enumerate(corrs_list):
                j:  int =   i + 1 + idx
                wj: float = W[j]

                if P_ij >= 0:
                    if      wi >= 0 and wj >= 0:    wi += P_ij; W[j] += P_ij
                    elif    wi < 0  and wj < 0:     wi -= P_ij; W[j] -= P_ij
                    elif    wi >= 0 and wj < 0:     wi -= P_ij; W[j] += P_ij
                    else:                           wi += P_ij; W[j] -= P_ij
                else:
                    if wi >= 0 and wj >= 0:
                        if  wi < wj:                wi += P_ij; W[j] -= P_ij
                        else:                       wi -= P_ij; W[j] += P_ij
                    elif wi < 0 and wj < 0:
                        if wi < wj:                 wi += P_ij; W[j] -= P_ij
                        else:                       wi -= P_ij; W[j] += P_ij
                    elif wi >= 0 and wj < 0:        wi -= P_ij; W[j] += P_ij
                    else:                           wi += P_ij; W[j] -= P_ij

            # Write cached wi back.
            W[i] = wi

        # Sort ascending by weight — low weight = easy/simple.
        ranked_order:   NDArray =    argsort(W)

        # Map back to original sample indices.
        return self._scores_["index"].iloc[ranked_order].tolist()