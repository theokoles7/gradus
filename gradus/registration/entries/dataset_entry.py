"""# gradus.registration.entries.dataset_entry

Defines structure & utility of dataset registration entry.
"""

__all__ = ["DatasetEntry"]

from typing                     import List, Type

from gradus.configuration       import DatasetConfig
from gradus.datasets            import GradusDataset
from gradus.registration.core   import Entry

class DatasetEntry(Entry):
    """# Dataset Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type[GradusDataset],
        config: Type[DatasetConfig],
        tags:   List[str] =     []
    ):
        """# Instantiate Dataset Registration Entry.

        ## Args:
            * id        (str):                  Dataset identifier.
            * cls       (Type[GradusDataset]):  Dataset class.
            * config    (Type[DatasetConfig]):  Dataset configuration.
            * tags      (List[str]):            Taxonomical keywords applicable to dataset.
        """
        # Initialize entry.
        super(DatasetEntry, self).__init__(id = id, config = config, tags = tags)

        # Define properties.
        self._cls_: Type[GradusDataset] =   cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type[GradusDataset]:
        """# Dataset Class"""
        return self._cls_