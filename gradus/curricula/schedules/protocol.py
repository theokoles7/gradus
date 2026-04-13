"""# gradus.curricula.schedules.protocol

Curriculum schedule protocol implementation.
"""

__all__ = ["Schedule"]

from abc                import ABC, abstractmethod
from logging            import Logger
from typing             import Any, Dict

from gradus.utilities   import get_logger

class Schedule(ABC):
    """# Abstract Curriculum Schedule"""

    def __init__(self,
        schedule_id:    str,
        total_samples:  int,
        total_epochs:   int
    ):
        """# Instantiate Curriculum Schedule.

        ## Args:
            * schedule_id   (str):  Schedule identifier.
            * total_samples (int):  Total number of training samples.
            * total_epochs  (int):  Total number of training epochs.
        """
        # Initialize logger.
        self.__logger__:        Logger =    get_logger(f"{schedule_id}-schedule")

        # Define properties.
        self._id_:              str =       schedule_id
        self._total_epochs_:    int =       int(total_epochs)
        self._total_samples_:   int =       int(total_samples)

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def dict(self) -> Dict[str, Any]:
        """# Schedule Dictionary Representation"""
        return  {
                    "id":               self._id_,
                    "total_epochs":     self._total_epochs_,
                    "total_samples":    self._total_samples_
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
    ) -> float:
        """# Update Schedule Based on Current Epoch.

        ## Args:
            * epoch     (int):  Current training epoch.
            * metrics   (Any):  Training metrics from the current epoch. Each schedule 
                                implementation consumes only the metrics it needs.

        ## Returns:
            * float:    Fraction of training data to expose.
        """
        # Clamp to valid range.
        fraction:   float = max(0.0, min(1.0, self._fraction_(epoch, **metrics)))

        # Debug fraction.
        self.__logger__.debug(f"Epoch {epoch}/{self._total_epochs_}: fraction = {fraction:.4f}")

        # Provide updated fraction.
        return fraction
    
    # HELPERS ======================================================================================

    @abstractmethod
    def _fraction_(self,
        epoch:      int,
        **metrics:  Any
    ) -> float:
        """# Compute Data Fraction for Current Epoch.

        ## Args:
            * epoch     (int):  Current epoch (1-indexed).
            * metrics   (Any):  Training metrics passed through from step().

        ## Returns:
            * float:    Desired data fraction in [0, 1].
        """
        ...
        
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Schedule Object Representation"""
        return f"""<{self.id.upper()}Schedule()>"""