"""# gradus.curricula.ranks.normalized_mean.test

Test suite for normalized mean composite curriculum rank.
"""

from pathlib                                            import Path
from typing                                             import List, Set

from pandas                                             import DataFrame, Series
from pytest                                             import approx, TempPathFactory

from gradus.curricula.ranks.normalized_mean.__base__    import NormalizedMean


# HELPERS ==========================================================================================

def normalized_mean_indices(
    scores:     DataFrame,
    tmp_path,
    metric:     List[str] = ["saturation-time", "color-variance", "edge-density", "spatial-frequency"],
) -> List[int]:
    """Instantiate NormalizedMean Rank with a Fresh Cache Directory and Return Indices."""
    return NormalizedMean(
        dataset_id =    "test-dataset",
        scores =        scores,
        metric =        metric,
        seed =          1,
        cache_dir =     str(tmp_path / "ranks")
    ).indices


def compute_normalized_mean(
    scores:     DataFrame,
    metrics:    List[str],
    inverted:   List[str] = [],
) -> Series:
    """Compute the Normalized Mean Composite Score Independently of the NormalizedMean Implementation.

    Provides a ground-truth implementation for test assertions.

    ## Args:
        * scores    (DataFrame):    Metric scores DataFrame.
        * metrics   (List[str]):    Metric columns to include.
        * inverted  (List[str]):    Metric columns to invert (1 - normalized).

    ## Returns:
        * Series:   Composite score per sample, indexed by DataFrame position.
    """
    # Copy and cast relevant columns.
    normalized: DataFrame = scores[metrics].copy().astype(float)

    # Min-max normalize each column.
    for col in metrics:
        col_min:    float = normalized[col].min()
        col_max:    float = normalized[col].max()
        col_range:  float = col_max - col_min

        if col_range < 1e-10:   normalized[col] = 0.0
        else:                   normalized[col] = (normalized[col] - col_min) / col_range

    # Invert columns where lower raw value means more complex.
    for col in inverted:
        if col in metrics: normalized[col] = 1.0 - normalized[col]

    # Equal-weighted mean across all columns.
    return normalized.mean(axis=1)


# TESTS ============================================================================================

class TestNormalizedMeanReturnType:
    """# Return Type Tests"""

    def test_returns_list(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """NormalizedMean Rank Should Return a List."""
        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Ensure indices is a list.
        assert isinstance(indices, list), \
            "NormalizedMean rank should return a list of indices."

    def test_returns_ints(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """All Returned Values Should be Integers."""
        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Ensure indices are integers.
        assert all(isinstance(i, int) for i in indices), \
            "All indices should be integers."


class TestNormalizedMeanCompleteness:
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
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Assert that indices is same length as dataframe.
        assert  len(indices) == len(synthetic_scores), \
                f"Expected {len(synthetic_scores)} indices, got {len(indices)}."

    def test_no_duplicates(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Contain no Duplicates."""
        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Assert that indices are unique.
        assert len(indices) == len(set(indices)), \
            "Ranked indices contain duplicates."

    def test_covers_all_samples(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Cover Every Sample in the Dataset."""
        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Note all expected indices.
        expected:   Set[int] =  set(synthetic_scores["index"].tolist())

        # Assert that indices account for all samples.
        assert set(indices) == expected, "Ranked indices do not cover all samples."


class TestNormalizedMeanOrder:
    """# Ordering Tests

    The core mathematical guarantee: samples are ordered by the equal-weighted mean of their
    min-max normalized metric scores, ascending. Lower composite = simpler sample.
    """

    def test_composite_scores_nondecreasing(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Normalized Mean Composite Scores at Ranked Indices Should be Non-Decreasing."""
        # Identify numeric non-meta columns.
        metrics:    List[str] = [
                                    c for c in synthetic_scores.columns
                                    if c not in ("index", "class")
                                ]

        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Compute ground-truth composite scores (no inverted metrics in synthetic_scores).
        composite:  Series =    compute_normalized_mean(synthetic_scores, metrics)

        # Extract composite values in ranked order.
        values:     List[float] = [composite.iloc[i] for i in indices]

        # For each consecutive pair...
        for pos in range(len(values) - 1):

            # Assert that composite score is non-decreasing.
            assert  values[pos] <= values[pos + 1] + 1e-9,             \
                    f"Composite ordering violated at position {pos}: "  \
                    f"{values[pos]:.6f} > {values[pos + 1]:.6f} "       \
                    f"(indices {indices[pos]} → {indices[pos + 1]})"

    def test_first_index_minimizes_composite(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """First Ranked Index Should Correspond to the Minimum Composite Score."""
        # Identify numeric non-meta columns.
        metrics:    List[str] = [
                                    c for c in synthetic_scores.columns
                                    if c not in ("index", "class")
                                ]

        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Compute ground-truth composite scores.
        composite:  Series =    compute_normalized_mean(synthetic_scores, metrics)

        # Assert that first index has minimum composite score.
        assert  composite.iloc[indices[0]] == approx(composite.min(), abs=1e-9), \
                "First ranked sample does not have minimum composite score."

    def test_last_index_maximizes_composite(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Last Ranked Index Should Correspond to the Maximum Composite Score."""
        # Identify numeric non-meta columns.
        metrics:    List[str] = [
                                    c for c in synthetic_scores.columns
                                    if c not in ("index", "class")
                                ]

        # Load indices.
        indices:    List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

        # Compute ground-truth composite scores.
        composite:  Series =    compute_normalized_mean(synthetic_scores, metrics)

        # Assert that last index has maximum composite score.
        assert  composite.iloc[indices[-1]] == approx(composite.max(), abs=1e-9), \
                "Last ranked sample does not have maximum composite score."

    def test_single_noninverted_metric_matches_ascending(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Single Non-Inverted Metric Should Produce Identical Order to Ascending.

        NormalizedMean with one non-inverted metric is mathematically equivalent to a plain
        ascending sort - min-max normalization preserves rank order, equal weighting is trivial
        with one metric, and no inversion is applied.
        """
        from gradus.curricula.ranks.ascending import Ascending

        # Load normalized mean indices on single metric.
        nm:         List[int] = normalized_mean_indices(
                                    synthetic_scores,
                                    tmp_path / "nm",
                                    metric = "saturation-time"
                                )

        # Load ascending indices on same metric.
        ascending:  List[int] = Ascending(
                                    metric =        "saturation-time",
                                    dataset_id =    "test-dataset",
                                    scores =        synthetic_scores,
                                    seed =          1,
                                    cache_dir =     str(tmp_path / "asc")
                                ).indices

        # Assert identical ordering.
        assert  nm == ascending, \
                "Single-metric NormalizedMean ordering differs from ascending - " \
                "these must be equivalent."

    def test_inverted_metric_flips_contribution(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Inverting a Metric Should Flip Its Normalized Contribution."""
        n:          int =       6

        # Load scores.
        scores:     DataFrame = DataFrame({
                                    "index":            list(range(n)),
                                    "class":            ["a"] * n,
                                    "normal-metric":    [float(i) for i in range(n)],
                                    "inverted-metric":  [float(n - 1 - i) for i in range(n)],
                                })

        # Without inversion, the two metrics are mirrors - composite is flat (0.5 for all).
        flat:       Series =    compute_normalized_mean(
                                    scores,
                                    ["normal-metric", "inverted-metric"],
                                    inverted = []
                                )
        
        # Ensure mirrored metrics produce flat composite.
        assert flat.nunique() == 1, \
            "Without inversion, mirrored metrics should produce a flat composite."

        # With inversion applied to the inverted metric, they reinforce - composite is strictly
        # increasing, matching the order of the normal metric.
        ordered:    Series =    compute_normalized_mean(
                                    scores,
                                    ["normal-metric", "inverted-metric"],
                                    inverted = ["inverted-metric"]
                                )
        
        for pos in range(len(ordered) - 1):
            assert  ordered.iloc[pos] <= ordered.iloc[pos + 1] + 1e-9, \
                    f"With inversion, composite should be non-decreasing at position {pos}."

    def test_equal_weighting_scale_invariant(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Equal Weighting Should be Scale-Invariant After Min-Max Normalization.

        A metric with values in [0, 1000] and a metric with values in [0, 1] should contribute
        equally to the composite after normalization. If normalization is not applied correctly,
        the large-scale metric would dominate.
        """
        n = 10

        # Construct two metrics with very different scales but identical rank orders.
        scores: DataFrame = DataFrame({
            "index":            list(range(n)),
            "class":            ["a"] * n,
            "large-scale":      [float(i * 1000) for i in range(n)],
            "small-scale":      [float(i) / n for i in range(n)],
        })

        # Load indices.
        indices:    List[int] = normalized_mean_indices(
                                    scores, tmp_path, metric=["large-scale", "small-scale"]
                                )

        # Compute ground-truth composite.
        composite:  Series =    compute_normalized_mean(
                                    scores, ["large-scale", "small-scale"]
                                )

        # Extract composite values in ranked order.
        values:     List[float] = [composite.iloc[i] for i in indices]

        # Assert composite is non-decreasing.
        for pos in range(len(values) - 1):
            assert  values[pos] <= values[pos + 1] + 1e-9, \
                    f"Scale-invariance violated at position {pos}: " \
                    f"{values[pos]:.6f} > {values[pos + 1]:.6f}"

    def test_constant_metric_column_does_not_break_ranking(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """A Constant Metric Column Should Not Cause Division by Zero or Corrupt the Ordering.

        If all samples have the same value for a metric, min-max normalization produces 0/0.
        The implementation should handle this gracefully (typically by setting the column to 0.0)
        and the remaining metrics should still produce a valid ordering.
        """
        n = 10

        # Construct scores where one metric is constant.
        scores: DataFrame = DataFrame({
            "index":        list(range(n)),
            "class":        ["a"] * n,
            "varying":      [float(i) for i in range(n)],
            "constant":     [5.0] * n,
        })

        # Load indices - should not raise.
        indices:    List[int] = normalized_mean_indices(
                                    scores, tmp_path, metric=["varying", "constant"]
                                )

        # Verify completeness.
        assert  set(indices) == set(range(n)), \
                "Ranked indices do not cover all samples."

        # Verify ordering still follows the varying metric.
        score_map:  Series =    scores.set_index("index")["varying"]
        values:     List[float] = [score_map[i] for i in indices]

        for pos in range(len(values) - 1):
            assert  values[pos] <= values[pos + 1] + 1e-9, \
                    f"Ordering violated at position {pos} with constant column present."


class TestNormalizedMeanCaching:
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

        # Construct normalized mean indices.
        normalized_mean_indices(synthetic_scores, tmp_path)

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
        first_run:  List[int] = normalized_mean_indices(synthetic_scores, tmp_path)
        second_run: List[int] = normalized_mean_indices(synthetic_scores, tmp_path)

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
        normalized_mean_indices(
            synthetic_scores, tmp_path,
            metric = ["saturation-time", "color-variance"]
        )
        normalized_mean_indices(
            synthetic_scores, tmp_path,
            metric = ["saturation-time", "edge-density"]
        )

        # Determine path to rank files.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that there are two separate cache files.
        assert  len(npy_files) == 2, \
                "Different metric subsets should produce separate cache files."