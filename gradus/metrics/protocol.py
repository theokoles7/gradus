"""# gradus.metrics.protocol

Abstract metric protocol.
"""

__all__ = ["Metric"]

from abc                import ABC
from logging            import Logger
from typing             import Union

from gradus.utilities   import get_logger

class Metric(ABC):
    """# Abstract Metric"""

    def __init__(self,
        metric_id:  str
    ):
        """# Instantiate Metric.

        ## Args:
            * metric_id (str):  Metric identifier.
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger(metric_id)

        # Define properties.
        self._id_:          str =               metric_id
        self._value_:       Union[int, float]

    # PROPERTIES ===================================================================================

    @property
    def id(self) -> str:
        """# Metric Identifier"""
        return self._id_
    
    @property
    def value(self) -> Union[int, float]:
        """# Overall Metric Value"""
        # If metric has not been calculated yet...
        if self._value_ is None:

            # Report error.
            raise AttributeError(f"Metric must be calculated before accessing value")
        
        # Provide metric value.
        return self._value_