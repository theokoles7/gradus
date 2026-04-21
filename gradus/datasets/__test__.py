"""# gradus.datasets.protocol.test

Integration test suite for Dataset train loader curriculum pipeline.

These tests verify the end-to-end guarantee that the curriculum ranking is correctly applied from 
rank file through to the actual batches yielded by the train loader. This is the test that catches 
ordering regressions.
"""

from math               import ceil
from pathlib            import Path
from typing             import List

from pandas             import DataFrame
from pytest             import TempPathFactory

from gradus.artifacts   import DatasetMetrics
from gradus.curricula   import Curriculum


# HELPERS ==========================================================================================

def rank_indices_for(
    scores:     DataFrame,
    rank_id:    str,
    metric:     "str | List[str]",
    cache_dir:  Path,
) -> List[int]:
    """Compute Ranked Indices Directly via the Rank Registry."""
    from gradus.registration import RANK_REGISTRY

    return RANK_REGISTRY.sort_indices(
        rank_id =       rank_id,
        dataset_id =    "test-dataset",
        metric =        metric,
        scores =        scores,
        seed =          1,
        cache_dir =     str(cache_dir)
    )


def build_dataset_metrics(
    synthetic_scores_path:  Path,
    n:                      int,
) -> DatasetMetrics:
    """Build DatasetMetrics from the Synthetic Scores Parquet Fixture."""
    return DatasetMetrics(
        dataset_id =    "test-dataset",
        num_samples =   n,
        seed =          1,
        scores_path =   str(synthetic_scores_path)
    )


# TESTS ============================================================================================

class TestCurriculumBatchComposition:
    """# Curriculum Batch Composition Tests

    Verifies that the Curriculum sampler yields batches whose indices are exactly the ranked 
    indices chunked into batch_size groups — no reordering, no missing samples, no extra samples.
    """

    def test_first_batch_matches_first_n_ranked_indices(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """First Batch Indices Should Exactly Match First batch_size Ranked Indices.

        This is the core regression test. If the curriculum ordering is applied correctly, the 
        first batch must contain exactly the first batch_size samples from the rank file — no 
        more, no less, in the same order.
        """
        batch_size:         int =               4

        # Compute the expected ranked order from the same scores fixture.
        expected_indices:   List[int] =         rank_indices_for(
                                                    scores =        synthetic_scores,
                                                    rank_id =       "ascending",
                                                    metric =        "saturation-time",
                                                    cache_dir =     tmp_path / "ranks"
                                                )

        # Build DatasetMetrics — loads the same data written by the fixture.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct curriculum.
        curriculum:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "holistic",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Get the first batch from the curriculum.
        first_batch:        List[int] =         next(iter(curriculum))

        # Assert exact match.
        assert  first_batch == expected_indices[:batch_size],       \
                f"First batch indices do not match first {batch_size} ranked indices.\n" \
                f"Expected: {expected_indices[:batch_size]}\n"      \
                f"Got:      {first_batch}"

    def test_all_batches_cover_full_ranked_order(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """Concatenating All Batches Should Reproduce the Full Ranked Index Sequence.

        The curriculum is a partition of the ranked indices into batches. Concatenating all batches 
        in order must reproduce the original ranked sequence exactly.
        """
        batch_size:         int =               4

        # Compute the expected ranked order.
        expected_indices:   List[int] =         rank_indices_for(
                                                    scores =        synthetic_scores,
                                                    rank_id =       "ascending",
                                                    metric =        "saturation-time",
                                                    cache_dir =     tmp_path / "ranks"
                                                )

        # Build DatasetMetrics.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct curriculum.
        curriculum:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "holistic",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Concatenate all batches.
        all_indices:        List[int] =         [i for batch in curriculum for i in batch]

        # Assert that concatenation reproduces the full ranked sequence.
        assert  all_indices == expected_indices,                                            \
                "Concatenated batches do not reproduce the full ranked index sequence.\n"   \
                f"Expected: {expected_indices}\n"                                           \
                f"Got:      {all_indices}"

    def test_batch_count_matches_expected(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """Number of Batches Should Equal ceil(n_samples / batch_size).

        With drop_last=False, the final batch may be smaller than batch_size but must still be 
        present.
        """
        batch_size:         int =               6

        # Build DatasetMetrics.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct curriculum.
        curriculum:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "holistic",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Compute expected batch count.
        expected_batches:   int =               ceil(len(synthetic_scores) / batch_size)

        # Assert batch count.
        assert  len(curriculum) == expected_batches, \
                f"Expected {expected_batches} batches, got {len(curriculum)}."

    def test_no_samples_lost_or_duplicated_across_batches(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """Every Sample Should Appear Exactly Once Across All Batches.

        The curriculum is a complete partition of the training set — no sample is skipped and no 
        sample appears in more than one batch.
        """
        batch_size:         int =               4

        # Build DatasetMetrics.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct curriculum.
        curriculum:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "holistic",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Collect all yielded indices.
        all_indices:        List[int] =         [i for batch in curriculum for i in batch]

        # Note expected sample count.
        n:                  int =               len(synthetic_scores)

        # Assert completeness.
        assert  len(all_indices) == n, \
                f"Expected {n} total indices, got {len(all_indices)}."

        assert  len(set(all_indices)) == n, "Duplicate indices found across batches."

        assert  set(all_indices) == set(synthetic_scores["index"].tolist()), \
                "Not all sample indices are covered by the curriculum batches."

    def test_weighted_rank_first_batch_matches_ranked_indices(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """Weighted Rank First Batch Should Match First batch_size Weighted Ranked Indices.

        Repeats the core regression test with the weighted rank — the rank that was at the center 
        of the original regression investigation.
        """
        batch_size:         int =               4

        # Compute the expected weighted ranked order.
        expected_indices:   List[int] =         rank_indices_for(
                                                    scores =        synthetic_scores,
                                                    rank_id =       "weighted",
                                                    metric =        "saturation-time",
                                                    cache_dir =     tmp_path / "ranks"
                                                )

        # Build DatasetMetrics.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct curriculum with weighted rank.
        curriculum:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "weighted",
                                                    scope =         "holistic",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Get the first batch.
        first_batch:        List[int] =         next(iter(curriculum))

        # Assert exact match.
        assert  first_batch == expected_indices[:batch_size],           \
                f"First batch does not match first {batch_size} weighted ranked indices.\n" \
                f"Expected: {expected_indices[:batch_size]}\n"          \
                f"Got:      {first_batch}"


class TestCurriculumScopeBatchWise:
    """# Batch-Wise Scope Tests

    Verifies that batch-wise ranking applies the rank independently within each batch rather than 
    globally across the full dataset.
    """

    def test_batch_wise_each_batch_is_internally_sorted(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """Each Batch in Batch-Wise Scope Should be Internally Sorted by Metric.

        In batch-wise scope, each batch is ranked independently. Within each batch, metric values 
        should be non-decreasing (for ascending rank).
        """
        batch_size:         int =               4

        # Build DatasetMetrics.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct curriculum with batch-wise scope.
        curriculum:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "batch-wise",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Build score lookup from the same fixture data.
        score_map = synthetic_scores.set_index("index")["saturation-time"]

        # For each batch...
        for batch_num, batch in enumerate(curriculum):

            # Extract metric values within batch.
            values: List[float] = [score_map[i] for i in batch]

            # Assert internal sort order is non-decreasing.
            for pos in range(len(values) - 1):
                assert  values[pos] <= values[pos + 1] + 1e-9,             \
                        f"Batch {batch_num} internal ordering violated at position {pos}: " \
                        f"{values[pos]:.4f} > {values[pos + 1]:.4f}"

    def test_holistic_and_batch_wise_produce_different_orderings(self,
        synthetic_scores:       DataFrame,
        synthetic_scores_path:  Path,
        tmp_path:               TempPathFactory
    ) -> None:
        """Holistic and Batch-Wise Scopes Should Produce Different Sample Orderings.

        Holistic ranks globally then chunks. Batch-wise chunks first then ranks within each chunk. 
        These are mathematically distinct operations and should produce different orderings in 
        general.
        """
        batch_size:         int =               4

        # Build DatasetMetrics.
        dataset_metrics:    DatasetMetrics =    build_dataset_metrics(
                                                    synthetic_scores_path,
                                                    len(synthetic_scores)
                                                )

        # Construct holistic curriculum.
        holistic:           Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "holistic",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Construct batch-wise curriculum.
        batch_wise:         Curriculum =        Curriculum(
                                                    dataset_id =    "test-dataset",
                                                    scores =        dataset_metrics,
                                                    metric =        "saturation-time",
                                                    rank =          "ascending",
                                                    scope =         "batch-wise",
                                                    batch_size =    batch_size,
                                                    seed =          1
                                                )

        # Flatten both curricula.
        holistic_indices:   List[int] =         [i for batch in holistic   for i in batch]
        batch_wise_indices: List[int] =         [i for batch in batch_wise for i in batch]

        # Assert that the two orderings differ.
        assert  holistic_indices != batch_wise_indices, \
                "Holistic and batch-wise scopes produced identical orderings — " \
                "they are mathematically distinct and should differ."