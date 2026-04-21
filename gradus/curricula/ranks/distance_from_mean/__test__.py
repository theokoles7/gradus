"""# gradus.curricula.ranks.distance_from_mean.test

Test suite for distance-from-mean curriculum rank.
"""

from pathlib                                    import Path
from typing                                     import List, Set

from pandas                                     import DataFrame, Series
from pytest                                     import approx, raises, TempPathFactory

from gradus.curricula.ranks.distance_from_mean  import DistanceFromMean


# HELPERS ==========================================================================================

def distance_from_mean_indices(
    scores:     DataFrame,
    metric:     str,
    tmp_path,
) -> List[int]:
    """Instantiate DistanceFromMean Rank with a Fresh Cache Directory and Return Indices."""
    return DistanceFromMean(
        metric =        metric,
        dataset_id =    "test-dataset",
        scores =        scores,
        seed =          1,
        cache_dir =     str(tmp_path / "ranks")
    ).indices


# TESTS ============================================================================================

class TestDistanceFromMeanReturnType:
    """# Return Type Tests"""

    def test_returns_list(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """DistanceFromMean Rank Should Return a List."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Ensure indices is a list.
        assert isinstance(indices, list), \
            "DistanceFromMean rank should return a list of indices."

    def test_returns_ints(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """All Returned Values Should be Integers."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Ensure indices are integers.
        assert all(isinstance(i, int) for i in indices), \
            "All indices should be integers."


class TestDistanceFromMeanCompleteness:
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
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Assert that indices is same length as dataframe.
        assert  len(indices) == len(synthetic_scores), \
                f"Expected {len(synthetic_scores)} indices, got {len(indices)}."

    def test_no_duplicates(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Contain no Duplicates."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Assert that indices are unique.
        assert len(indices) == len(set(indices)), \
            "Ranked indices contain duplicates."

    def test_covers_all_samples(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Cover Every Sample in the Dataset."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Note all expected indices.
        expected:   Set[int] =  set(synthetic_scores["index"].tolist())

        # Assert that indices account for all samples.
        assert set(indices) == expected, \
            "Ranked indices do not cover all samples."


class TestDistanceFromMeanOrder:
    """# Ordering Tests

    The core mathematical guarantee: samples are ordered by absolute deviation from the dataset
    mean, ascending. The first sample is closest to the mean; the last is furthest.
    """

    def test_absolute_deviations_nondecreasing(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Absolute Deviations from Mean Should be Non-Decreasing Across Ranked Indices."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Compute dataset mean.
        mean:       float =     synthetic_scores["saturation-time"].mean()

        # Look up metric value for each ranked index in order.
        score_map:  DataFrame = synthetic_scores.set_index("index")["saturation-time"]

        # Compute absolute deviation from mean for each ranked index.
        deviations: List[float] =   [abs(score_map[i] - mean) for i in indices]

        # For each consecutive pair...
        for pos in range(len(deviations) - 1):

            # Assert that absolute deviation is non-decreasing.
            assert  deviations[pos] <= deviations[pos + 1],             \
                    f"Deviation ordering violated at position {pos}: "  \
                    f"|{score_map[indices[pos]]:.4f} - mean| = "        \
                    f"{deviations[pos]:.4f} > "                         \
                    f"{deviations[pos + 1]:.4f} = "                     \
                    f"|{score_map[indices[pos + 1]]:.4f} - mean|"

    def test_first_index_minimizes_deviation(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """First Ranked Index Should be Closest to the Dataset Mean."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Compute dataset mean.
        mean:           float =     synthetic_scores["saturation-time"].mean()

        # Extract saturation-time scores.
        score_map:      DataFrame = synthetic_scores.set_index("index")["saturation-time"]

        # Compute deviation of first ranked sample.
        first_deviation:    float = abs(score_map[indices[0]] - mean)

        # Compute minimum possible deviation across all samples.
        min_deviation:      float = (synthetic_scores["saturation-time"] - mean).abs().min()

        # Assert that first ranked sample has minimum deviation.
        assert  first_deviation == approx(min_deviation),                   \
                f"First ranked sample has deviation {first_deviation:.4f}, "\
                f"but minimum deviation is {min_deviation:.4f}."

    def test_last_index_maximizes_deviation(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Last Ranked Index Should be Furthest from the Dataset Mean."""
        # Load indices.
        indices:    List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Compute dataset mean.
        mean:           float =     synthetic_scores["saturation-time"].mean()

        # Extract saturation-time scores.
        score_map:      DataFrame = synthetic_scores.set_index("index")["saturation-time"]

        # Compute deviation of last ranked sample.
        last_deviation:     float = abs(score_map[indices[-1]] - mean)

        # Compute maximum possible deviation across all samples.
        max_deviation:      float = (synthetic_scores["saturation-time"] - mean).abs().max()

        # Assert that last ranked sample has maximum deviation.
        assert  last_deviation == approx(max_deviation),                    \
                f"Last ranked sample has deviation {last_deviation:.4f}, "  \
                f"but maximum deviation is {max_deviation:.4f}."

    def test_ordering_consistent_across_metrics(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ordering Should Respect Whichever Metric is Specified as Anchor."""
        # For each metric...
        for metric in ["saturation-time", "color-variance", "edge-density"]:

            # Load indices.
            indices:    List[int] = distance_from_mean_indices(
                                        synthetic_scores, metric, tmp_path / metric
                                    )

            # Compute dataset mean for this metric.
            mean:       float =     synthetic_scores[metric].mean()

            # Extract scores for metric.
            score_map:  DataFrame = synthetic_scores.set_index("index")[metric]

            # Compute absolute deviations.
            deviations: List[float] = [abs(score_map[i] - mean) for i in indices]

            # For each consecutive pair...
            for pos in range(len(deviations) - 1):

                # Assert that absolute deviation is non-decreasing.
                assert  deviations[pos] <= deviations[pos + 1], \
                        f"Deviation ordering violated for metric '{metric}' at position {pos}."

    def test_symmetric_samples_equidistant_from_mean(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Samples Equidistant from the Mean May Appear in Either Order but Must be Adjacent."""
        import pandas as pd

        # Construct a scores DataFrame with two samples equidistant from the mean (0.0 and 2.0,
        # mean = 1.0, both have deviation 1.0), plus one sample at the mean itself.
        scores: DataFrame = pd.DataFrame({
            "index":            [0, 1, 2, 3, 4],
            "class":            ["a", "b", "c", "d", "e"],
            "saturation-time":  [1.0, 0.0, 2.0, -1.0, 3.0],
        })

        # Load indices.
        indices:    List[int] = distance_from_mean_indices(scores, "saturation-time", tmp_path)

        # Compute mean and deviations.
        mean:       float =         scores["saturation-time"].mean()
        score_map:  DataFrame =     scores.set_index("index")["saturation-time"]
        deviations: List[float] =   [abs(score_map[i] - mean) for i in indices]

        # Assert that deviations are non-decreasing regardless of tie resolution.
        for pos in range(len(deviations) - 1):
            assert  deviations[pos] <= deviations[pos + 1], \
                    f"Deviation ordering violated at position {pos} with tied samples."


class TestDistanceFromMeanCaching:
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

        # Construct distance-from-mean indices.
        distance_from_mean_indices(synthetic_scores, "saturation-time", tmp_path)

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
        first_run:  List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )
        second_run: List[int] = distance_from_mean_indices(
                                    synthetic_scores, "saturation-time", tmp_path
                                )

        # Assert that the two runs are equal.
        assert  first_run == second_run, \
                "Cached rank result differs from original computation."

    def test_different_metrics_produce_different_cache_keys(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Different Metrics Should Produce Separate Cache Files."""
        # Form cache path.
        cache_dir:  Path =  tmp_path / "ranks"

        # Construct two runs of different metrics.
        distance_from_mean_indices(synthetic_scores, "saturation-time", tmp_path)
        distance_from_mean_indices(synthetic_scores, "color-variance",  tmp_path)

        # Determine path to rank file.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that there are two separate cache files.
        assert  len(npy_files) == 2, \
                "Different metrics should produce separate cache files."


class TestDistanceFromMeanValidation:
    """# Validation Tests"""

    def test_rejects_multiple_metrics(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """DistanceFromMean Rank Should Reject More Than One Metric."""
        with raises(ValueError):
            DistanceFromMean(
                metric =        ["saturation-time", "color-variance"],
                dataset_id =    "test-dataset",
                scores =        synthetic_scores,
                seed =          1,
                cache_dir =     str(tmp_path / "ranks")
            )