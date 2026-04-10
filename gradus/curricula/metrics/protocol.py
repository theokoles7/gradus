"""# gradus.curricula.metrics.protocol

Abstract metric protocol.
"""

__all__ = ["Metric"]

from abc                import ABC, abstractmethod
from logging            import Logger
from typing             import Union

from gradus.utilities   import get_logger

class Metric(ABC):
    """# Abstract Curriculum Metric"""

    def __init__(self,
        metric_id:  str
    ):
        """# Instantiate Metric.

        ## Args:
            * metric_id (str):  Metric identifier.
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger(f"{metric_id}-metric")

        # Define properties.
        self._id_:          str =               metric_id
        self._value_:       Union[int, float] = None

    # PROPERTIES ===================================================================================

    @property
    def id(self) -> str:
        """# Metric Identifier"""
        return self._id_
    
    @property
    @abstractmethod
    def value(self) -> Union[int, float]:
        """# Overall Metric Value"""
        ...