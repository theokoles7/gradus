"""# gradus.curricula.ranks.ascending.test

Test suite for ascending curriculum rank.
"""

from pathlib                            import Path
from typing                             import List, Set

from pandas                             import DataFrame, Series
from pytest                             import approx, raises, TempPathFactory

from gradus.curricula.ranks.ascending   import Ascending


# HELPERS ==========================================================================================

def ascending_indices(
    scores:     DataFrame,
    metric:     str,
    tmp_path,
) -> List[int]:
    """Instantiate Ascending Rank with a Fresh Cache Directory and Return Indices."""
    return Ascending(
        metric =        metric,
        dataset_id =    "test-dataset",
        scores =        scores,
        seed =          1,
        cache_dir =     str(tmp_path / "ranks")
    ).indices


# TESTS ============================================================================================

class TestAscendingReturnType:
    """# Return Type Tests"""

    def test_returns_list(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ):
        """Ascending Rank Should Return a List."""
        # Load indices.
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Esnure indices is a list.
        assert isinstance(indices, list), "Ascending rank should return a list of indices."

    def test_returns_ints(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ):
        """All Returned Values Should be Integers."""
        # Load indices.
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Esnure indices are integers.
        assert all(isinstance(i, int) for i in indices), "All indices should be integers."


class TestAscendingCompleteness:
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
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Assert that indices is same length as dataframe.
        assert  len(indices) == len(synthetic_scores), \
                f"Expected {len(synthetic_scores)} indices, got {len(indices)}."

    def test_no_duplicates(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Contain no Duplicates."""
        # Load indices.
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Assert that indices are unique.
        assert len(indices) == len(set(indices)), "Ranked indices contain duplicates."

    def test_covers_all_samples(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Cover Every Sample in the Dataset."""
        # Load indices.
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Note all expected indices.
        expected:   Set[int] =  set(synthetic_scores["index"].tolist())

        # Assert that indices account for all samples.
        assert set(indices) == expected, "Ranked indices do not cover all samples."


class TestAscendingOrder:
    """# Ordering Tests

    The core mathematical guarantee: Samples should be ordered such that metric values are 
    non-decreasing from first to last index.
    """

    def test_metric_values_nondecreasing(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Metric Values at Ranked Indices Should be Non-Decreasing."""
        # Load indices.
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Look up metric value for each ranked index in order.
        values: List[float] =   [
                                    synthetic_scores.set_index("index")["saturation-time"][i] 
                                    for i in indices
                                ]

        # For each sample...
        for pos in range(len(values) - 1):

            # Assert that the metric is not greater than the preceding sample.
            assert  values[pos] <= values[pos + 1],                 \
                    f"Ordering violated at position {pos}: "        \
                    f"{values[pos]:.4f} > {values[pos + 1]:.4f} "   \
                    f"(indices {indices[pos]} → {indices[pos + 1]})"

    def test_first_index_has_minimum_value(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """First ranked index should correspond to the minimum metric value."""
        # Load indices.
        indices:        List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Extract saturation-time scores.
        score_map:      DataFrame = synthetic_scores.set_index("index")["saturation-time"]

        # Find the first value.
        first_value:    float =     score_map[indices[0]]

        # Find the minimum value.
        min_value:      float =     synthetic_scores["saturation-time"].min()

        # Assert that the first value is the lowest.
        assert  first_value == approx(min_value),                       \
                f"First ranked sample has value {first_value:.4f}, "    \
                f"but minimum is {min_value:.4f}."

    def test_last_index_has_maximum_value(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Last ranked index should correspond to the maximum metric value."""
        # Load indices.
        indices:    List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Extract saturation-time scores.
        score_map:  DataFrame = synthetic_scores.set_index("index")["saturation-time"]

        # Find the last value.
        last_value: float =     score_map[indices[-1]]

        # Find the maximum value.
        max_value:  float =     synthetic_scores["saturation-time"].max()

        # Assert that the last value is the highest.
        assert  last_value == approx(max_value), \
                f"Last ranked sample has value {last_value:.4f}, " \
                f"but maximum is {max_value:.4f}."

    def test_ordering_consistent_across_metrics(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ordering should respect whichever metric is specified as anchor."""
        # For each metric...
        for metric in ["saturation-time", "color-variance", "edge-density"]:
            
            # Load indices.
            indices:    List[int] = ascending_indices(synthetic_scores, metric, tmp_path)

            # Extract scores for metric.
            score_map:  DataFrame = synthetic_scores.set_index("index")[metric]

            # Extract values from metric.
            values:     Series =    [score_map[i] for i in indices]

            # For each sample...
            for pos in range(len(values) - 1):

                # Assert that the metric is not less than the preceding sample.
                assert  values[pos] <= values[pos + 1], \
                        f"Ordering violated for metric '{metric}' at position {pos}."


class TestAscendingCaching:
    """# Cache Tests

    Rank indices should be cached to disk and reloaded correctly, producing identical results on 
    subsequent calls.
    """

    def test_cache_file_created(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """A .npy cache file should be created after ranking."""
        # Form cache path.
        cache_dir:  Path =  tmp_path / "ranks"

        # Construct ascending indices according to saturation time.
        ascending_indices(synthetic_scores, "saturation-time", tmp_path)

        # Determine path to rank file.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that ranks were cached.
        assert  len(npy_files) == 1, \
                f"Expected 1 cache file, found {len(npy_files)}."

    def test_cached_result_matches_original(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Loading from cache should produce identical indices."""
        # Conduct two runs on the same seed.
        first_run:  List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)
        second_run: List[int] = ascending_indices(synthetic_scores, "saturation-time", tmp_path)

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
        ascending_indices(synthetic_scores, "saturation-time", tmp_path)
        ascending_indices(synthetic_scores, "color-variance",  tmp_path)

        # Determine path to rank file.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that there are two separate cache files.
        assert  len(npy_files) == 2, \
                "Different metrics should produce separate cache files."


class TestAscendingValidation:
    """# Validation Tests"""

    def test_rejects_multiple_metrics(self, synthetic_scores, tmp_path):
        """Ascending rank should reject more than one metric."""
        with raises(ValueError):
            Ascending(
                metric =        ["saturation-time", "color-variance"],
                dataset_id =    "test-dataset",
                scores =        synthetic_scores,
                seed =          1,
                cache_dir =     str(tmp_path / "ranks")
            )