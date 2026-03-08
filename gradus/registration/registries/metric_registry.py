"""# gradus.registration.registries.metric_registry

Metric registry system implementation.
"""

__all__ = ["MetricRegistry"]

from typing                         import Any, Dict, List, override, TYPE_CHECKING

from torch                          import Tensor

from gradus.registration.core       import Registry
from gradus.registration.entries    import MetricEntry

class MetricRegistry(Registry):
    """# Metric Registry System"""

    def __init__(self):
        """# Instantiate Metric Registry System"""
        super(MetricRegistry, self).__init__(id = "metrics")

    # PROPERTIES ===================================================================================

    @property
    def entries(self) -> Dict[str, MetricEntry]:
        """# Registered Metric Entries"""
        return self._entries_.copy()
    
    # METHODS ======================================================================================

    def compute(self,
        metric_id:  str,
        sample:     Tensor
    ) -> Any:
        """# Compute Registered Metric.

        ## Args:
            * metric_id (str):      Identifier of metric being computed.
            * sample    (Tensor):   Image sample tensor whose metric is being computed.

        ## Returns:
            * Any:  Result of metric computation.
        """
        return self.get_entry(entry_id = metric_id).fn(sample)
    
    def compute_all(self,
        sample:     Tensor,
        filter_by:  List[str] = []
    ) -> Dict[str, Any]:
        """# Compute All Registered Metrics.

        ## Args:
            * sample    (Tensor):       Image sample tensor whose metric(s) are being computed.
            * filter_by (List[str]):    Taxonomical keywords to filter which metrics are computed.

        ## Returns:
            * Dict[str, Any]:   Mapping of metric IDs to their computed values.
        """
        return  {
                    metric_id: self.get_entry(entry_id = metric_id).fn(sample)
                    for metric_id in self.list_entries(filter_by = filter_by)
                }
    
    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> MetricEntry:
        """# Create Metric Entry.

        ## Returns:
            * MetricEntry:  New metric entry instance.
        """
        return MetricEntry(**kwargs)

    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        metric_id:  str
    ) -> MetricEntry:
        """# Query Registered Metrics.

        ## Args:
            * metric_id (str):  Identifier of metric being queried.

        ## Returns:
            * MetricEntry:  Metric entry, if registered.
        """
        return self.get_entry(entry_id = metric_id)