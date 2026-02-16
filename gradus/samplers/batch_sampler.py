"""# gradus.samplers.batch_sampler

Batch-wise dataset sampler implementation.
"""

__all__ = ["CurriculumBatchSampler"]

from logging            import Logger
from typing             import Callable, Generator, List

from torch.utils.data   import BatchSampler, Dataset

from gradus.utilities   import get_logger

class CurriculumBatchSampler(BatchSampler):
    """# Batch-wise Dataset Sampler"""

    def __init__(self,
        dataset:    Dataset,
        metric_fn:  Callable,
        batch_size: int
    ):
        """# Instantiate Curriculum Batch Sampler.

        ## Args:
            * dataset       (Dataset):  Dataset from which to sample.
            * metric_fn     (Callable): Metric function by which sample sorting will be governed.
            * batch_size    (int):      Number of samples to include in each batch.
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger("batch-sampler")

        # Define batches.
        self._batches_:     List[List[int]] =   [
                                                    sorted(
                                                        iterable =  range(
                                                                        image_index,
                                                                        min(
                                                                            image_index + batch_size,
                                                                            len(dataset)
                                                                        )
                                                                    ),
                                                        key =       lambda image: metric_fn(dataset[image][0])
                                                    )
                                                    for image_index in  range(
                                                                            0,
                                                                            len(dataset),
                                                                            batch_size
                                                                        )
                                                ]

    # DUNDERS ======================================================================================

    def __iter__(self) -> Generator:
        """# Generate Batch Indices on Iteration.

        ## Yields:
            * Generator:    Batch-wise indices.
        """
        yield from self._batches_

    def __len__(self) -> int:
        """# Batch Quantity

        ## Returns:
            * int:  Number of batches in sampler.
        """
        return len(self._batches_)