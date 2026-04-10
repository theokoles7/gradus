"""# gradus.registration.entries.metric_entry

Defines structure & utility of metric registration entry.
"""

__all__ = ["MetricEntry"]

from typing                     import Callable, List, Type, TYPE_CHECKING

from gradus.registration.core   import Entry

if TYPE_CHECKING:
    from gradus.configuration   import MetricConfig
    from gradus.curricula       import Metric

class MetricEntry(Entry):
    """# Metric Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type["Metric"],
        config: "MetricConfig",
        tags:   List[str] =     []
    ):
        """# Instantiate Metric Registration Entry.

        ## Args:
            * id        (str):          Metric identifier.
            * cls       (Type[Metric]): Metric class.
            * config    (MetricConfig): Metric configuration handler.
            * tags      (List[str]):    Taxonomical keywords applicable to metric.
        """
        # Initialize entry.
        super(MetricEntry, self).__init__(id = id, config = config, tags = tags)

        # Define properties.
        self._cls_: Type["Metric"] =    cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type["Metric"]:
        """# Metric Class"""
        return self._cls_