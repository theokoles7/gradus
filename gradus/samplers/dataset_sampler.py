"""# gradus.samplers.dataset_sampler

Holistic/Stochastic dataset sampler implementation.
"""

__all__ = ["CurriculumDatasetSampler"]

from logging            import Logger
from typing             import Callable, Iterator

from numpy              import argsort
from numpy.typing       import NDArray
from torch.utils.data   import Dataset, Sampler

from gradus.utilities   import get_logger

class CurriculumDatasetSampler(Sampler):
    """# Holistic Dataset Sampler"""

    def __init__(self,
        dataset:    Dataset,
        metric_fn:  Callable
    ):
        """# Instantiate Holistic Batch Sampler.

        ## Args:
            * dataset   (Dataset):  Dataset from which to sample.
            * metric_fn (Callable): Metric function by which sample sorting will be governed.
        """
        # Initialize logger.
        self.__logger__:    Logger =    get_logger("dataset-sampler")

        # Sort dataset and record sorted indices.
        self._indices_:     NDArray =   argsort([
                                            metric_fn(dataset[image][0])
                                            for image in range(len(dataset))
                                        ])

    # DUNDERS ======================================================================================

    def __iter__(self) -> Iterator:
        """# Provide Sorted Sample Indices.

        ## Yields:
            * Iterator:    Sorted indices.
        """
        return iter(self._indices_)

    def __len__(self) -> int:
        """# Sample Quantity

        ## Returns:
            * int:  Length of dataset.
        """
        return len(self._indices_)