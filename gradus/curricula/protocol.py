"""# gradus.curricula.protocol

Curriculum protocol implementation.
"""

__all__ = ["Curriculum"]

from logging                import Logger
from typing                 import Any, Dict, Generator, List, override, Union

from torch.utils.data       import BatchSampler

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
            sampler =       range(len(scores.scores)),
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
                                    f"Invalid scope specified: {scope}"
                                    "Valid scopes are: batch-wise, holistic"
                                )
            
    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Curriculum Dictionary Representation"""
        return  {
            "rank":     self._rank_,
            "metric":   self._metric_,
            "scope":    self._scope_        
        }

    # HELPERS ======================================================================================

    def _batch_wise_rank_(self) -> List[List[int]]:
        """# Construct Batch-Wise Ranks.

        ## Returns:
            * List[List[int]]:  Batch-wise indice ranks.
        """
        # Log action.
        self.__logger__.info(
            f"Constructing batch-wise ranks (metrics = {self._metric_}, rank = {self._rank_})"
        )

        # Extract indices.
        indices:    List[int] = list(range(len(self._scores_.scores)))

        # Construct batch-wise ranks.
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
            * List[List[int]]:  Holistic indice ranks.
        """
        # Log action.
        self.__logger__.info(
            f"Constructing holistic ranks (metrics = {self._metric_}, rank = {self._rank_})"
        )

        # Rank indices.
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
        yield from self._batches_

    @override
    def __len__(self) -> int:
        """# Batch Quantity

        ## Returns:
            * int:  Number of batches in sampler.
        """
        return len(self._batches_)