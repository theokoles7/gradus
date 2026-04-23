"""# gradus.curricula.protocol

Curriculum protocol implementation.
"""

__all__ = ["Curriculum"]

from logging                import Logger
from typing                 import Any, Dict, Generator, List, override, Union

from torch.utils.data       import BatchSampler, SequentialSampler

from gradus.artifacts       import DatasetMetrics
from gradus.registration    import RANK_REGISTRY
from gradus.utilities       import get_logger

class Curriculum(BatchSampler):
    """# Curriculum Sampler"""

    def __init__(self,
        dataset_id: str,
        scores:     DatasetMetrics,
        metric:     Union[str, List[str]],
        rank:       str,
        scope:      str,
        batch_size: int =                   128,
        seed:       int =                   1
    ):
        """# Instantiate Curriculum.

        ## Args:
            * dataset_id    (str):              Identifier of dataset to whom curriculum will be 
                                                applied.
            * scores        (DatasetMetrics):   Dataset metrics artifact.
            * metric        (str | List[str]):  Metric(s) by which samples will be ranked.
            * rank          (str):              Order by which samples will be sorted, based on 
                                                rank.
            * scope         (str):              Scope of sorting.
            * batch_size    (int):              Number of samples in each batch. Defaults to 128.
            * seed          (int):              Random number generation seed. Defaults to 1.
        """
        # Initialize logger.
        self.__logger__:    Logger =    get_logger("curriculum")

        # Initialize batch sampler.
        super(Curriculum, self).__init__(
            sampler =       SequentialSampler(scores.scores),
            batch_size =    batch_size,
            drop_last =     False
        )

        # Define properties.
        self._dataset_id_:  str =               dataset_id
        self._scores_:      DatasetMetrics =    scores
        self._metric_:      List[str] =         [metric] if isinstance(metric, str) else metric
        self._rank_:        str =               rank
        self._scope_:       str =               scope
        self._seed_:        int =               seed
        self._batch_size_:  int =               batch_size
        self._batches_:     List[List[int]]

        # Match scope.
        match scope:

            # Batch-wise
            case "batch-wise":  self._batches_ = self._batch_wise_rank_()

            # Holistic
            case "holistic":    self._batches_ = self._holistic_rank_()

            # Invalid
            case _:             raise ValueError(
                                    f"Invalid scope specified: {scope}. "
                                    f"Valid scopes are: batch-wise, holistic"
                                )

        # Initialize active batches to full curriculum in natural order.
        self._active_:      List[List[int]] =   self._batches_

    # PROPERTIES ===================================================================================

    @property
    def batch_indices(self) -> List[int]:
        """# Original Batch Indices in Current Active Order"""
        return getattr(self, "_current_order_", list(range(len(self._batches_))))

    @property
    def dict(self) -> Dict[str, Any]:
        """# Curriculum Dictionary Representation"""
        return  {
                    "rank":     self._rank_,
                    "metric":   self._metric_,
                    "scope":    self._scope_,
                }

    # METHODS ======================================================================================

    def set_order(self,
        order:  List[int]
    ) -> None:
        """# Set Active Batch Ordering.

        Reorders and/or subsets the curriculum batches according to the provided
        list of batch indices. This is the single method through which all schedule
        types control what the training loop sees each epoch:

            - Linear/Adaptive: pass [0, 1, ..., N] to expose a prefix of the curriculum.
            - Gradient: pass all indices sorted by descending gradient norm.

        ## Args:
            * order (List[int]):    Ordered list of batch indices to expose. Each index
                                    must be in [0, len(self._batches_)).
        """
        # Clamp indices to valid range and deduplicate while preserving order.
        total:          int =           len(self._batches_)
        valid_order:    List[int] =     [i for i in order if 0 <= i < total]
        self._current_order_ = valid_order

        # Always expose at least one batch.
        if not valid_order: valid_order = [0]

        # Apply ordering.
        self._active_ = [self._batches_[i] for i in valid_order]

        # Debug action.
        self.__logger__.debug(
            f"Active order: {len(self._active_)}/{total} batches, "
            f"first 5 indices: {valid_order[:5]}"
        )

    # HELPERS ======================================================================================

    def _batch_wise_rank_(self) -> List[List[int]]:
        """# Construct Batch-Wise Ranks.

        Shuffles indices before chunking to break class grouping, then sorts
        samples within each chunk by the specified metric.

        ## Returns:
            * List[List[int]]:  Batch-wise ranked indices.
        """
        from numpy.random   import default_rng

        # Log action.
        self.__logger__.info(
            f"Constructing batch-wise ranks (metrics = {self._metric_}, rank = {self._rank_})"
        )

        # Shuffle indices before chunking to break class grouping.
        indices:    List[int] = list(range(len(self._scores_.scores)))
        default_rng(seed = self._seed_).shuffle(indices)

        # Construct batch-wise ranks on shuffled chunks.
        return  [
                    RANK_REGISTRY.sort_indices(
                        rank_id =       self._rank_,
                        dataset_id =    self._dataset_id_,
                        metric =        self._metric_,
                        scores =        self._scores_.scores.iloc[indices[i:i + self._batch_size_]],
                        seed =          self._seed_
                    )
                    for i in range(0, len(indices), self._batch_size_)
                ]

    def _holistic_rank_(self) -> List[List[int]]:
        """# Construct Holistic Ranks.

        ## Returns:
            * List[List[int]]:  Holistic ranked indices.
        """
        # Log action.
        self.__logger__.info(
            f"Constructing holistic ranks (metrics = {self._metric_}, rank = {self._rank_})"
        )

        # Rank indices globally.
        indices:    List[int] = RANK_REGISTRY.sort_indices(
                                    rank_id =       self._rank_,
                                    dataset_id =    self._dataset_id_,
                                    metric =        self._metric_,
                                    scores =        self._scores_.scores,
                                    seed =          self._seed_
                                )

        # Break out into batches.
        return  [
                    indices[i:i + self._batch_size_]
                    for i in range(0, len(indices), self._batch_size_)
                ]

    # DUNDERS ======================================================================================

    @override
    def __iter__(self) -> Generator:
        """# Generate Batched Indices on Iteration.

        ## Yields:
            * Generator:    Batch-wise indices.
        """
        yield from self._active_

    @override
    def __len__(self) -> int:
        """# Active Batch Quantity

        ## Returns:
            * int:  Number of active batches.
        """
        return len(self._active_)