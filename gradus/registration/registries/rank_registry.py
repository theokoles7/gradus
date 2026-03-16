"""# gradus.registration.registries.rank_registry

Rank registry system implementation.
"""

__all__ = ["RankRegistry"]

from typing                         import Dict, List, override, Union

from pandas                         import DataFrame

from gradus.registration.core       import Registry
from gradus.registration.entries    import RankEntry

class RankRegistry(Registry):
    """# Rank Registry System"""

    def __init__(self):
        """# Instantiate Rank Registry System."""
        super(RankRegistry, self).__init__(id = "ranks")

    # PROPERTIES ===================================================================================

    @property
    def entries(self) -> Dict[str, RankEntry]:
        """# Registered Rank Entries"""
        return self._entries_.copy()
    
    # METHODS ======================================================================================

    def sort_indices(self,
        rank_id:    str,
        metric:     Union[str, List[str]],
        scores:     DataFrame
    ) -> List[int]:
        """# Rank Indices Based on Metric Scores.

        ## Args:
            * rank_id   (str):              Identifier of ranking scheme being used to sort indices.
            * metric    (str | List[str]):  Metric(s) by which samples will be ranked.
            * scores    (DataFrame):        Metric scores data sheet.

        ## Returns:
            * List[int]:    Indices sorted by order of metric + ranking scheme.
        """
        return self.get_entry(entry_id = rank_id).fn(metric, scores)
        
    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> RankEntry:
        """# Create Rank Entry.

        ## Returns:
            * RankEntry:    New rank entry instance.
        """
        return RankEntry(**kwargs)
    
    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        rank_id:    str
    ) -> RankEntry:
        """# Query Registered Ranks.

        ## Args:
            * rank_id   (str):  Identifier of rank being queried.

        ## Returns:
            * RankEntry:    Rank entry, if registered.
        """
        return self.get_entry(entry_id = rank_id)