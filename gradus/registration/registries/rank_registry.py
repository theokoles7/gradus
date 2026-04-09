"""# gradus.registration.registries.rank_registry

Rank registry system implementation.
"""

__all__ = ["RankRegistry"]

from pathlib                        import Path
from typing                         import Dict, List, override, Union

from pandas                         import DataFrame

from gradus.registration.core       import Registry
from gradus.registration.entries    import RankEntry

class RankRegistry(Registry):
    """# Rank Registry System"""

    def __init__(self):
        """# Instantiate Rank Registry System."""
        super(RankRegistry, self).__init__(id = "curricula.ranks")

    # PROPERTIES ===================================================================================

    @property
    def entries(self) -> Dict[str, RankEntry]:
        """# Registered Rank Entries"""
        return self._entries_.copy()
    
    # METHODS ======================================================================================

    def sort_indices(self,
        rank_id:    str,
        dataset_id: str,
        metric:     Union[str, List[str]],
        scores:     DataFrame,
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ) -> List[int]:
        """# Rank Indices Based on Metric Scores.

        ## Args:
            * rank_id       (str):              Identifier of ranking scheme being used to sort indices.
            * dataset_id    (str):              Identifier of dataset whose samples are being 
                                                ranked.
            * metric        (str | List[str]):  Metric(s) by which samples will be ranked.
            * scores        (DataFrame):        Metric scores data sheet.
            * seed          (int):              Random number generation seed.
            * cache_dir     (str | Path):       Directory under which keyed indices will be cached. 
                                                Defaults to "./.cache/ranks/".

        ## Returns:
            * List[int]:    Indices sorted by order of metric + ranking scheme.
        """
        return self.get_entry(entry_id = rank_id).cls(
            dataset_id =    dataset_id,
            scores =        scores,
            metric =        metric,
            seed =          seed,
            cache_dir =     cache_dir
        ).indices
        
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