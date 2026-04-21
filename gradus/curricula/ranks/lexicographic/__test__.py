"""# gradus.curricula.ranks.lexicographic.test

Test suite for lexicographic composite curriculum rank.
"""

from pathlib                                import Path
from typing                                 import List, Set

from pandas                                 import DataFrame, Series
from pytest                                 import approx, TempPathFactory

from gradus.curricula.ranks.lexicographic   import Lexicographic


# HELPERS ==========================================================================================

def lexicographic_indices(
    scores:     DataFrame,
    metric:     "str | List[str]",
    tmp_path,
) -> List[int]:
    """Instantiate Lexicographic Rank with a Fresh Cache Directory and Return Indices."""
    return Lexicographic(
        metric =        metric,
        dataset_id =    "test-dataset",
        scores =        scores,
        seed =          1,
        cache_dir =     str(tmp_path / "ranks")
    ).indices


# TESTS ============================================================================================

class TestLexicographicReturnType:
    """# Return Type Tests"""

    def test_returns_list(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Lexicographic Rank Should Return a List."""
        # Load indices.
        indices:    List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
                                )

        # Ensure indices is a list.
        assert isinstance(indices, list), \
            "Lexicographic rank should return a list of indices."

    def test_returns_ints(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """All Returned Values Should be Integers."""
        # Load indices.
        indices:    List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
                                )

        # Ensure indices are integers.
        assert all(isinstance(i, int) for i in indices), "All indices should be integers."


class TestLexicographicCompleteness:
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
        indices:    List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
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
        indices:    List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
                                )

        # Assert that indices are unique.
        assert len(indices) == len(set(indices)), "Ranked indices contain duplicates."

    def test_covers_all_samples(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Ranked Indices Should Cover Every Sample in the Dataset."""
        # Load indices.
        indices:    List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
                                )

        # Note all expected indices.
        expected:   Set[int] =  set(synthetic_scores["index"].tolist())

        # Assert that indices account for all samples.
        assert set(indices) == expected, "Ranked indices do not cover all samples."


class TestLexicographicOrder:
    """# Ordering Tests

    The core mathematical guarantee: samples are sorted by the first metric, then by the second
    metric where the first is tied, then by the third where the first two are tied, and so on —
    exactly as a dictionary sort works.
    """

    def test_primary_metric_nondecreasing(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Primary Metric Values at Ranked Indices Should be Non-Decreasing."""
        # Construct scores with deliberate ties on the primary metric to exercise tie-breaking.
        scores: DataFrame = DataFrame({
            "index":            list(range(10)),
            "class":            ["a"] * 10,
            "saturation-time":  [1.0, 3.0, 1.0, 5.0, 2.0, 3.0, 5.0, 2.0, 4.0, 1.0],
            "color-variance":   [0.9, 0.2, 0.1, 0.8, 0.5, 0.7, 0.3, 0.4, 0.6, 0.2],
        })

        # Load indices.
        indices:    List[int] =     lexicographic_indices(
                                        scores,
                                        ["saturation-time", "color-variance"],
                                        tmp_path
                                    )

        # Look up primary metric value for each ranked index.
        score_map:  DataFrame =     scores.set_index("index")["saturation-time"]
        values:     List[float] =   [score_map[i] for i in indices]

        # Assert primary metric is non-decreasing.
        for pos in range(len(values) - 1):
            assert  values[pos] <= values[pos + 1],                 \
                    f"Primary metric ordering violated at position {pos}: "     \
                    f"{values[pos]:.4f} > {values[pos + 1]:.4f} "               \
                    f"(indices {indices[pos]} → {indices[pos + 1]})"

    def test_secondary_metric_breaks_ties_in_primary(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Where Primary Metric Values are Equal, Secondary Metric Should be Non-Decreasing.

        This is the defining property of lexicographic ordering — tie-breaking by successive
        metrics. If it fails, the rank is just sorting by the primary metric and ignoring the rest.
        """
        # Construct scores where primary metric has deliberate ties.
        scores: DataFrame = DataFrame({
            "index":            list(range(6)),
            "class":            ["a"] * 6,
            "saturation-time":  [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "color-variance":   [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
        })

        # Load indices.
        indices:        List[int] = lexicographic_indices(
                                        scores,
                                        ["saturation-time", "color-variance"],
                                        tmp_path
                                    )

        # Build lookup maps.
        primary_map:    Series =    scores.set_index("index")["saturation-time"]
        secondary_map:  Series =    scores.set_index("index")["color-variance"]

        # For each consecutive pair where primary values are equal...
        for pos in range(len(indices) - 1):

            primary_a:      float = primary_map[indices[pos]]
            primary_b:      float = primary_map[indices[pos + 1]]
            secondary_a:    float = secondary_map[indices[pos]]
            secondary_b:    float = secondary_map[indices[pos + 1]]

            if primary_a == primary_b:

                # Secondary metric must be non-decreasing within the tie group.
                assert  secondary_a <= secondary_b,                         \
                        f"Tie-breaking violated at position {pos}: "        \
                        f"primary={primary_a}, "                            \
                        f"secondary {secondary_a:.4f} > {secondary_b:.4f} " \
                        f"(indices {indices[pos]} → {indices[pos + 1]})"

    def test_single_metric_matches_ascending(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """With a Single Metric, Lexicographic Should Produce Identical Order to Ascending.

        Lexicographic with one metric is mathematically equivalent to a plain ascending sort.
        This cross-rank consistency check ensures no divergence from that guarantee.
        """
        from gradus.curricula.ranks.ascending import Ascending

        # Load lexicographic indices with single metric.
        lexico:     List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time"],
                                    tmp_path / "lexico"
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
        assert  lexico == ascending, \
                "Single-metric lexicographic ordering differs from ascending — " \
                "these must be equivalent."

    def test_metric_order_affects_ranking(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Swapping Metric Priority Should Produce a Different Ordering.

        If [A, B] and [B, A] produce identical orderings, the secondary metric is having no
        effect — which would mean tie-breaking is broken.
        """
        # Construct scores with ties on both metrics to ensure priority matters.
        scores: DataFrame = DataFrame({
            "index":            list(range(8)),
            "class":            ["a"] * 8,
            "saturation-time":  [1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "color-variance":   [0.1, 0.9, 0.2, 0.8, 0.5, 0.6, 0.3, 0.7],
        })

        # Load indices with [saturation-time, color-variance] priority.
        order_ab:   List[int] = lexicographic_indices(
                                    scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path / "ab"
                                )

        # Load indices with [color-variance, saturation-time] priority.
        order_ba:   List[int] = lexicographic_indices(
                                    scores,
                                    ["color-variance", "saturation-time"],
                                    tmp_path / "ba"
                                )

        # The two orderings must differ — metric priority must matter.
        assert  order_ab != order_ba, \
                "Swapping metric priority produced identical ordering — " \
                "secondary metric appears to have no effect."

    def test_three_metric_ordering(self,
        tmp_path:   TempPathFactory
    ) -> None:
        """Three-Metric Lexicographic Sort Should Respect All Three Priority Levels."""
        # Construct scores where all three levels of tie-breaking are needed.
        scores: DataFrame = DataFrame({
            "index":            list(range(8)),
            "class":            ["a"] * 8,
            "saturation-time":  [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "color-variance":   [0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
            "edge-density":     [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
        })

        # Load indices.
        indices:    List[int] = lexicographic_indices(
                                    scores,
                                    ["saturation-time", "color-variance", "edge-density"],
                                    tmp_path
                                )

        # Build lookup maps.
        p1: Series = scores.set_index("index")["saturation-time"]
        p2: Series = scores.set_index("index")["color-variance"]
        p3: Series = scores.set_index("index")["edge-density"]

        # Verify full three-level lexicographic ordering.
        for pos in range(len(indices) - 1):
            a, b = indices[pos], indices[pos + 1]

            if p1[a] == p1[b] and p2[a] == p2[b]:
                # Both primary and secondary are tied — tertiary must break the tie.
                assert  p3[a] <= p3[b], \
                        f"Tertiary tie-breaking violated at position {pos}: " \
                        f"edge-density {p3[a]:.4f} > {p3[b]:.4f} " \
                        f"(indices {a} → {b})"
            elif p1[a] == p1[b]:
                # Primary is tied — secondary must break the tie.
                assert  p2[a] <= p2[b], \
                        f"Secondary tie-breaking violated at position {pos}: " \
                        f"color-variance {p2[a]:.4f} > {p2[b]:.4f} " \
                        f"(indices {a} → {b})"
            else:
                # Primary is not tied — it must be non-decreasing.
                assert  p1[a] <= p1[b], \
                        f"Primary ordering violated at position {pos}: " \
                        f"saturation-time {p1[a]:.4f} > {p1[b]:.4f} " \
                        f"(indices {a} → {b})"


class TestLexicographicCaching:
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

        # Construct lexicographic indices.
        lexicographic_indices(
            synthetic_scores,
            ["saturation-time", "color-variance"],
            tmp_path
        )

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
        first_run:  List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
                                )
        second_run: List[int] = lexicographic_indices(
                                    synthetic_scores,
                                    ["saturation-time", "color-variance"],
                                    tmp_path
                                )

        # Assert that the two runs are equal.
        assert  first_run == second_run, \
                "Cached rank result differs from original computation."

    def test_different_metric_lists_produce_different_cache_keys(self,
        synthetic_scores:   DataFrame,
        tmp_path:           TempPathFactory
    ) -> None:
        """Different Metric Lists Should Produce Separate Cache Files."""
        # Form cache path.
        cache_dir:  Path =  tmp_path / "ranks"

        # Construct two runs with different metric lists.
        lexicographic_indices(
            synthetic_scores,
            ["saturation-time", "color-variance"],
            tmp_path
        )
        lexicographic_indices(
            synthetic_scores,
            ["saturation-time", "edge-density"],
            tmp_path
        )

        # Determine path to rank files.
        npy_files = list(cache_dir.glob("*.npy"))

        # Assert that there are two separate cache files.
        assert  len(npy_files) == 2, \
                "Different metric lists should produce separate cache files."