"""# gradus.curricula.ranks.pairwise_correlation.test

Test suite for pairwise correlation composite curriculum rank.
"""

from pathlib                                        import Path
from typing                                         import List, Set

from numpy                                          import nan_to_num
from numpy.linalg                                   import norm
from numpy.typing                                   import NDArray
from pandas                                         import DataFrame, Series
from pytest                                         import TempPathFactory

from gradus.curricula.ranks.pairwise_correlation    import PairwiseCorrelation


# HELPERS ==========================================================================================

def pairwise_correlation_indices(
    scores:     DataFrame,
    tmp_path,
    metric:     List[str] = ["saturation-time", "color-variance", "edge-density", "spatial-frequency"],
) -> List[int]:
    """Instantiate PairwiseCorrelation Rank with a Fresh Cache Directory and Return Indices."""
    return PairwiseCorrelation(
        dataset_id =    "test-dataset",
        scores =        scores,
        metric =        metric,
        seed =          1,
        cache_dir =     str(tmp_path / "ranks")
    ).indices


def compute_pairwise_weights(
    scores:     DataFrame,
    metrics:    List[str],
    inverted:   List[str] = [],
) -> List[float]:
    """Compute Pairwise Correlation Weights Independently of the PairwiseCorrelation Implementation.

    Provides a ground-truth implementation for test assertions.

    ## Args:
        * scores    (DataFrame):    Metric scores DataFrame.
        * metrics   (List[str]):    Metric columns to include.
        * inverted  (List[str]):    Metric columns to invert before computing correlations.

    ## Returns:
        * List[float]:  Weight W[i] for each sample i, in original DataFrame order.
    """
    # Extract attribute matrix.
    attributes: NDArray =    scores[metrics].values.astype(float)

    # Min-max normalize each column.
    col_mins:   NDArray =    attributes.min(axis=0)
    col_maxs:   NDArray =    attributes.max(axis=0)
    col_ranges: NDArray =    col_maxs - col_mins
    col_ranges[col_ranges < 1e-10] = 1.0
    normalized: NDArray =    (attributes - col_mins) / col_ranges

    # Invert necessary metrics.
    for col in inverted:
        if col in metrics:
            normalized[:, metrics.index(col)] = 1.0 - normalized[:, metrics.index(col)]

    # Center each row and L2-normalize for Pearson via dot product.
    centered:   NDArray =    normalized - normalized.mean(axis=1, keepdims=True)
    norms:      NDArray =    norm(centered, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    X:          NDArray =    centered / norms

    # Pairwise correlation voting.
    n:          int =           len(X)
    W:          List[float] =   [0.0] * n

    for i in range(n):
        corrs =         X[i] @ X[i + 1:].T
        corrs_list =    nan_to_num(corrs, nan=0.0).tolist()
        wi =            W[i]

        for idx, P_ij in enumerate(corrs_list):
            j =     i + 1 + idx
            wj =    W[j]

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

        W[i] = wi

    return W


# TESTS ============================================================================================

class TestPairwiseCorrelationReturnType:
    """# Return Type Tests"""

    def test_returns_list(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """PairwiseCorrelation Rank Should Return a List."""
        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Ensure indices is a list.
        assert isinstance(indices, list), \
            "PairwiseCorrelation rank should return a list of indices."

    def test_returns_ints(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """All Returned Values Should be Integers."""
        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Ensure indices are integers.
        assert all(isinstance(i, int) for i in indices), \
            "All indices should be integers."


class TestPairwiseCorrelationCompleteness:
    """# Completeness Tests

    The ranked indices must be a permutation of the original index set - no samples added, none
    dropped, none duplicated.
    """

    def test_correct_length(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Have Same Length as Input DataFrame."""
        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Assert that indices is same length as dataframe.
        assert  len(indices) == len(synthetic_scores), \
                f"Expected {len(synthetic_scores)} indices, got {len(indices)}."

    def test_no_duplicates(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Contain no Duplicates."""
        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Assert that indices are unique.
        assert len(indices) == len(set(indices)), \
            "Ranked indices contain duplicates."

    def test_covers_all_samples(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Cover Every Sample in the Dataset."""
        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Note all expected indices.
        expected:   Set[int] =  set(synthetic_scores["index"].tolist())

        # Assert that indices account for all samples.
        assert set(indices) == expected, \
            "Ranked indices do not cover all samples."


class TestPairwiseCorrelationOrder:
    """# Ordering Tests

    The core mathematical guarantee: samples are ordered by ascending pairwise correlation
    weight W[i], where W[i] accumulates votes from all pairwise Pearson correlations. Lower
    weight = simpler sample (more consistently easy across all metrics).
    """

    def test_weights_nondecreasing_at_ranked_indices(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Pairwise Correlation Weights at Ranked Indices Should be Non-Decreasing."""
        # Define metrics.
        metrics:    List[str] = ["saturation-time", "color-variance", "edge-density",
                                 "spatial-frequency"]

        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Compute ground-truth weights.
        weights:    List[float] = compute_pairwise_weights(synthetic_scores, metrics)

        # Extract weights in ranked order.
        values:     List[float] = [weights[i] for i in indices]

        # For each consecutive pair...
        for pos in range(len(values) - 1):

            # Assert that weight is non-decreasing.
            assert  values[pos] <= values[pos + 1] + 1e-9,         \
                    f"Weight ordering violated at position {pos}: " \
                    f"{values[pos]:.6f} > {values[pos + 1]:.6f} "   \
                    f"(indices {indices[pos]} → {indices[pos + 1]})"

    def test_first_index_minimizes_weight(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """First Ranked Index Should Correspond to the Minimum Pairwise Weight."""
        # Define metrics.
        metrics:    List[str] = ["saturation-time", "color-variance", "edge-density",
                                 "spatial-frequency"]

        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Compute ground-truth weights.
        weights:    List[float] = compute_pairwise_weights(synthetic_scores, metrics)

        # Assert that first index has minimum weight.
        assert  weights[indices[0]] == min(weights), \
                f"First ranked sample does not have minimum pairwise weight. " \
                f"Got {weights[indices[0]]:.6f}, minimum is {min(weights):.6f}."

    def test_last_index_maximizes_weight(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Last Ranked Index Should Correspond to the Maximum Pairwise Weight."""
        # Define metrics.
        metrics:    List[str] = ["saturation-time", "color-variance", "edge-density",
                                 "spatial-frequency"]

        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Compute ground-truth weights.
        weights:    List[float] = compute_pairwise_weights(synthetic_scores, metrics)

        # Assert that last index has maximum weight.
        assert  weights[indices[-1]] == max(weights), \
                f"Last ranked sample does not have maximum pairwise weight. " \
                f"Got {weights[indices[-1]]:.6f}, maximum is {max(weights):.6f}."

    def test_perfectly_ordered_dataset_produces_clear_ordering(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """A Dataset Where All Metrics Agree on Difficulty Should Produce a Clear Ordering.

        When all metrics are perfectly correlated (all increase together), every sample's
        difficulty is unambiguous. The resulting ordering should match a plain ascending sort
        on any single metric.
        """
        n = 10

        # Construct perfectly correlated metrics — all agree on difficulty.
        scores: DataFrame = DataFrame({
            "index":    list(range(n)),
            "class":    ["a"] * n,
            "metric-a": [float(i) for i in range(n)],
            "metric-b": [float(i) * 2.0 for i in range(n)],
            "metric-c": [float(i) * 0.5 for i in range(n)],
        })

        # Load indices.
        indices:    List[int] = pairwise_correlation_indices(
                                    scores, tmp_path,
                                    metric = ["metric-a", "metric-b", "metric-c"]
                                )

        # With perfectly correlated metrics, ordering should match ascending on any metric.
        score_map:  Series =    scores.set_index("index")["metric-a"]
        values:     List[float] = [score_map[i] for i in indices]

        for pos in range(len(values) - 1):
            assert  values[pos] <= values[pos + 1] + 1e-9, \
                    f"Perfectly correlated metrics should produce ascending order " \
                    f"at position {pos}: {values[pos]:.4f} > {values[pos + 1]:.4f}"

    def test_ordering_consistent_across_metric_subsets(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Weight-Based Ordering Should Hold for Any Subset of Metrics."""
        # For each pair of metrics as a subset...
        for metrics in [
            ["saturation-time", "color-variance"],
            ["saturation-time", "edge-density"],
            ["color-variance", "spatial-frequency"],
        ]:
            # Load indices.
            indices:    List[int] = pairwise_correlation_indices(
                                        synthetic_scores,
                                        tmp_path / "-".join(metrics),
                                        metric = metrics
                                    )

            # Compute ground-truth weights for this subset.
            weights:    List[float] = compute_pairwise_weights(synthetic_scores, metrics)

            # Extract weights in ranked order.
            values:     List[float] = [weights[i] for i in indices]

            # Assert non-decreasing.
            for pos in range(len(values) - 1):
                assert  values[pos] <= values[pos + 1] + 1e-9, \
                        f"Weight ordering violated for metrics {metrics} at position {pos}."


class TestPairwiseCorrelationCaching:
    """# Cache Tests

    Rank indices should be cached to disk and reloaded correctly, producing identical results on
    subsequent calls.
    """

    def test_cache_file_created(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """A .npy Cache File Should be Created After Ranking."""
        # Form cache path.
        cache_dir:  Path =  tmp_path / "ranks"

        # Construct pairwise correlation indices.
        pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Determine path to rank file.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that ranks were cached.
        assert  len(npy_files) == 1, \
                f"Expected 1 cache file, found {len(npy_files)}."

    def test_cached_result_matches_original(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Loading from Cache Should Produce Identical Indices."""
        # Conduct two runs on the same seed.
        first_run:  List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)
        second_run: List[int] = pairwise_correlation_indices(synthetic_scores, tmp_path)

        # Assert that the two runs are equal.
        assert  first_run == second_run, \
                "Cached rank result differs from original computation."

    def test_different_metric_subsets_produce_different_cache_keys(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Different Metric Subsets Should Produce Separate Cache Files."""
        # Form cache path.
        cache_dir:  Path =  tmp_path / "ranks"

        # Construct two runs with different metric subsets.
        pairwise_correlation_indices(
            synthetic_scores, tmp_path,
            metric = ["saturation-time", "color-variance"]
        )
        pairwise_correlation_indices(
            synthetic_scores, tmp_path,
            metric = ["saturation-time", "edge-density"]
        )

        # Determine path to rank files.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that there are two separate cache files.
        assert  len(npy_files) == 2, \
                "Different metric subsets should produce separate cache files."