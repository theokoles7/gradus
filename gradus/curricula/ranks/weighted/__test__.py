"""# gradus.curricula.ranks.weighted.test

Test suite for weighted composite curriculum rank.
"""

from pathlib                            import Path
from typing                             import Generator, List, Set

from numpy                              import linspace
from numpy.random                       import default_rng
from numpy.typing                       import NDArray
from pandas                             import DataFrame, Series
from pytest                             import approx, raises, TempPathFactory

from gradus.curricula.ranks.weighted    import Weighted


# HELPERS ==========================================================================================

def weighted_indices(
    scores:     DataFrame,
    metric:     str,
    tmp_path,
) -> List[int]:
    """Instantiate Weighted Rank with a Fresh Cache Directory and Return Indices."""
    return Weighted(
        metric =        metric,
        dataset_id =    "test-dataset",
        scores =        scores,
        seed =          1,
        cache_dir =     str(tmp_path / "ranks")
    ).indices


def compute_composite(
    scores:     DataFrame,
    anchor:     str,
) -> Series:
    """Compute the Weighted Composite Score Independently of the Weighted Rank Implementation.

    Replicates the mathematical definition so tests can assert against a ground-truth value
    rather than just checking internal consistency.

    ## Args:
        * scores    (DataFrame):    Metric scores DataFrame.
        * anchor    (str):          Anchor metric identifier.

    ## Returns:
        * Series:   Composite score per sample, indexed by DataFrame position.
    """
    # Identify numeric non-index columns.
    numerics:       List[str] = [
                                    col for col in scores.columns
                                    if col not in ("index", "class")
                                    and scores[col].dtype in ("float64", "int64")
                                ]

    # Z-score normalize all numeric columns.
    normalized:     DataFrame = scores[numerics].apply(lambda col: (col - col.mean()) / col.std())

    # Compute Pearson correlation of all metrics against anchor, drop anchor itself.
    correlations:   Series =    normalized.corrwith(normalized[anchor]).drop([anchor])

    # Take absolute value and normalize weights to sum to 1.
    weights:        Series =    correlations.abs()
    weights =                   weights / weights.sum()

    # Composite = anchor z-score + weighted sum of all other z-scores.
    return normalized[anchor] + normalized[weights.index].mul(weights).sum(axis=1)


# TESTS ============================================================================================

class TestWeightedReturnType:
    """# Return Type Tests"""

    def test_returns_list(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Weighted Rank Should Return a List."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Ensure indices is a list.
        assert isinstance(indices, list), "Weighted rank should return a list of indices."

    def test_returns_ints(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """All Returned Values Should be Integers."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Ensure indices are integers.
        assert all(isinstance(i, int) for i in indices), "All indices should be integers."


class TestWeightedCompleteness:
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
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Assert that indices is same length as dataframe.
        assert  len(indices) == len(synthetic_scores), \
                f"Expected {len(synthetic_scores)} indices, got {len(indices)}."

    def test_no_duplicates(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Contain no Duplicates."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Assert that indices are unique.
        assert len(indices) == len(set(indices)), "Ranked indices contain duplicates."

    def test_covers_all_samples(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Cover Every Sample in the Dataset."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Note all expected indices.
        expected:   Set[int] =  set(synthetic_scores["index"].tolist())

        # Assert that indices account for all samples.
        assert set(indices) == expected, "Ranked indices do not cover all samples."


class TestWeightedOrder:
    """# Ordering Tests

    The core mathematical guarantee: samples are ordered by a weighted composite score, ascending.
    The composite is the anchor z-score plus a correlation-weighted sum of all other metric
    z-scores. The first sample minimizes the composite; the last maximizes it.
    """

    def test_composite_scores_nondecreasing(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Composite Scores at Ranked Indices Should be Non-Decreasing."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Compute ground-truth composite scores.
        composite:  Series =    compute_composite(synthetic_scores, "saturation-time")

        # Extract composite values in ranked order.
        values:     List[float] = [composite.iloc[i] for i in indices]

        # For each consecutive pair...
        for pos in range(len(values) - 1):

            # Assert that composite score is non-decreasing.
            assert  values[pos] <= values[pos + 1] + 1e-9,              \
                    f"Composite ordering violated at position {pos}: "  \
                    f"{values[pos]:.6f} > {values[pos + 1]:.6f} "       \
                    f"(indices {indices[pos]} -> {indices[pos + 1]})"

    def test_first_index_minimizes_composite(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """First Ranked Index Should Correspond to the Minimum Composite Score."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Compute ground-truth composite scores.
        composite:  Series =    compute_composite(synthetic_scores, "saturation-time")

        # Assert that first index has minimum composite score.
        assert  composite.iloc[indices[0]] == approx(composite.min(), abs=1e-9), \
                "First ranked sample does not have minimum composite score."

    def test_last_index_maximizes_composite(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Last Ranked Index Should Correspond to the Maximum Composite Score."""
        # Load indices.
        indices:    List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Compute ground-truth composite scores.
        composite:  Series =    compute_composite(synthetic_scores, "saturation-time")

        # Assert that last index has maximum composite score.
        assert  composite.iloc[indices[-1]] == approx(composite.max(), abs=1e-9), \
                "Last ranked sample does not have maximum composite score."

    def test_anchor_dominates_when_others_uncorrelated(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """When Non-Anchor Metrics are Uncorrelated with Anchor, Ordering Should Follow Anchor.

        If all other metrics have zero Pearson correlation with the anchor, their weights become
        zero and the composite reduces to the anchor z-score alone. The resulting composite order
        should still be non-decreasing.
        """
        rng:            Generator =     default_rng(seed = 0)
        n:              int =           30

        # Construct anchor with clear ordering signal.
        anchor_values:  NDArray =       linspace(0.0, 1.0, n)

        # Construct other metrics as pure noise, orthogonal to anchor.
        scores:         DataFrame =     DataFrame({
                                            "index":            list(range(n)),
                                            "class":            ["a"] * n,
                                            "saturation-time":  anchor_values,
                                            "color-variance":   rng.uniform(0.0, 1.0, n),
                                            "edge-density":     rng.uniform(0.0, 0.5, n),
                                        })

        # Load weighted indices.
        indices:        List[int] =     weighted_indices(scores, "saturation-time", tmp_path)

        # Compute composite and verify non-decreasing order.
        composite:      Series =        compute_composite(scores, "saturation-time")
        values:         List[float] =   [composite.iloc[i] for i in indices]

        for pos in range(len(values) - 1):
            assert  values[pos] <= values[pos + 1] + 1e-9, \
                    f"Composite ordering violated at position {pos} with uncorrelated metrics."

    def test_composite_differs_from_anchor_only_ordering(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Weighted Composite Ordering Should Differ from Plain Anchor-Only Ordering.

        When non-anchor metrics are correlated with the anchor, the weighted composite
        incorporates their signal. The resulting order should not be identical to a plain
        ascending sort on the anchor alone, confirming that correlation weighting has an effect.
        """
        from gradus.curricula.ranks.ascending import Ascending

        # Load weighted indices.
        weighted:   List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Load plain ascending indices on same anchor.
        ascending:  List[int] = Ascending(
                                    metric =        "saturation-time",
                                    dataset_id =    "test-dataset",
                                    scores =        synthetic_scores,
                                    seed =          1,
                                    cache_dir =     str(tmp_path / "ranks" / "asc")
                                ).indices

        # The two orderings should differ because the composite incorporates correlated metrics.
        assert  weighted != ascending,                                  \
                "Weighted ordering is identical to plain ascending - "  \
                "correlation weighting appears to have no effect."

    def test_ordering_consistent_across_anchor_metrics(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Composite Ordering Should Hold Regardless of Which Metric is the Anchor."""
        # For each metric as anchor...
        for anchor in ["saturation-time", "color-variance", "edge-density"]:

            # Load indices.
            indices:    List[int] = weighted_indices(
                                        synthetic_scores, anchor, tmp_path / anchor
                                    )

            # Compute ground-truth composite for this anchor.
            composite:  Series =        compute_composite(synthetic_scores, anchor)
            values:     List[float] =   [composite.iloc[i] for i in indices]

            # For each consecutive pair...
            for pos in range(len(values) - 1):

                # Assert that composite score is non-decreasing.
                assert  values[pos] <= values[pos + 1] + 1e-9, \
                        f"Composite ordering violated for anchor '{anchor}' at position {pos}."


class TestWeightedCaching:
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

        # Construct weighted indices.
        weighted_indices(synthetic_scores, "saturation-time", tmp_path)

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
        first_run:  List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)
        second_run: List[int] = weighted_indices(synthetic_scores, "saturation-time", tmp_path)

        # Assert that the two runs are equal.
        assert  first_run == second_run, \
                "Cached rank result differs from original computation."

    def test_different_anchors_produce_different_cache_keys(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Different Anchor Metrics Should Produce Separate Cache Files."""
        # Form cache path.
        cache_dir:  Path =  tmp_path / "ranks"

        # Construct two runs with different anchors.
        weighted_indices(synthetic_scores, "saturation-time", tmp_path)
        weighted_indices(synthetic_scores, "color-variance",  tmp_path)

        # Determine path to rank file.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that there are two separate cache files.
        assert  len(npy_files) == 2, \
                "Different anchor metrics should produce separate cache files."


class TestWeightedValidation:
    """# Validation Tests"""

    def test_rejects_multiple_metrics(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Weighted Rank Should Reject More Than One Anchor Metric."""
        with raises(ValueError):
            Weighted(
                metric =        ["saturation-time", "color-variance"],
                dataset_id =    "test-dataset",
                scores =        synthetic_scores,
                seed =          1,
                cache_dir =     str(tmp_path / "ranks")
            )