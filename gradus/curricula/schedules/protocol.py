"""# gradus.curricula.schedules.protocol

Curriculum schedule protocol implementation.
"""

__all__ = ["Schedule"]

from abc                import ABC, abstractmethod
from logging            import Logger
from math               import ceil
from typing             import Any, Dict, List, Tuple

from gradus.utilities   import get_logger

class Schedule(ABC):
    """# Abstract Curriculum Schedule"""

    def __init__(self,
        schedule_id:    str,
        total_samples:  int,
        total_epochs:   int,
        batch_size:     int
    ):
        """# Instantiate Curriculum Schedule.

        ## Args:
            * schedule_id   (str):  Schedule identifier.
            * total_samples (int):  Total number of training samples.
            * total_epochs  (int):  Total number of training epochs.
            * batch_size    (int):  Number of samples within each batch.
        """
        # Initialize logger.
        self.__logger__:        Logger =    get_logger(f"{schedule_id}-schedule")

        # Define properties.
        self._id_:              str =       schedule_id
        self._total_epochs_:    int =       int(total_epochs)
        self._total_samples_:   int =       int(total_samples)
        self._batch_size_:      int =       int(batch_size)
        self._total_batches_:   int =       ceil(self._total_samples_ / self._batch_size_)

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Schedule Dictionary Representation"""
        return  {
                    "id":               self._id_,
                    "total_epochs":     self._total_epochs_,
                    "total_samples":    self._total_samples_,
                    "batch_size":       self._batch_size_,
                    "total_batches":    self._total_batches_,
                }

    @property
    def id(self) -> str:
        """# Schedule Identifier"""
        return self._id_
    
    @property
    def total_epochs(self) -> int:
        """# Total Training Epochs"""
        return self._total_epochs_
    
    @property
    def total_samples(self) -> int:
        """# Total Training Samples"""
        return self._total_samples_
    
    # METHODS ======================================================================================
 
    def step(self,
        epoch:      int,
        **metrics:  Any
    ) -> List[int]:
        """# Update Schedule Based on Current Epoch.
 
        ## Args:
            * epoch     (int):  Current training epoch.
            * metrics   (Any):  Training metrics from the current epoch. Each schedule 
                                implementation consumes only the metrics it needs.
 
        ## Returns:
            * List[int]:    Ordered list of batch indices to expose this epoch.
        """
        # Compute ordering.
        order:  List[int] = self._order_(epoch, **metrics)
 
        # Assert tha at least one batch is being used.
        assert len(order) > 0,                                      \
            f"Schedule returned empty batch order at epoch {epoch}"
        
        # Assert that all batches are valid.
        assert all(0 <= i < self._total_batches_ for i in order),   \
            f"Schedule returned out-of-range batch indices at epoch {epoch}"
 
        # Debug ordering.
        self.__logger__.debug(
            f"Epoch {epoch}/{self._total_epochs_}: "
            f"{len(order)}/{self._total_batches_} batches active, "
            f"first 5: {order[:5]}"
        )
 
        # Provide batch ordering.
        return order
    
    # HELPERS ======================================================================================
 
    @abstractmethod
    def _order_(self,
        epoch:      int,
        **metrics:  Any
    ) -> List[int]:
        """# Compute Batch Ordering for Current Epoch.
 
        ## Args:
            * epoch     (int):  Current epoch (1-indexed).
            * metrics   (Any):  Training metrics passed through from step().
 
        ## Returns:
            * List[int]:    Ordered list of batch indices to expose this epoch.
        """
        ...
        
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Schedule Object Representation"""
        return f"""<{self.id.upper()}Schedule()>"""