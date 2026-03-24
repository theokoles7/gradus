"""# gradus.samplers.dataset_sampler

Holistic dataset sampler implementation.
"""

__all__ = ["CurriculumDatasetSampler"]

from logging            import Logger
from typing             import Iterator, List

from torch.utils.data   import Sampler

from gradus.utilities   import get_logger

class CurriculumDatasetSampler(Sampler[int]):
    """# Holistic Dataset Sampler"""

    def __init__(self,
        indices:    List[int]
    ):
        """# Instantiate Holistic Dataset Sampler.

        ## Args:
            * indices   (List[int]):    Pre-sorted dataset indices defining curriculum order.
        """
        # Initialize logger.
        self.__logger__:    Logger =        get_logger("dataset-sampler")

        # Store pre-sorted indices.
        self._indices_:     List[int] =     indices

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