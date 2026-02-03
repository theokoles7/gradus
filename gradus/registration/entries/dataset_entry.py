"""# gradus.registration.entries.dataset_entry

Defines structure & utility of dataset registration entry.
"""

__all__ = ["DatasetEntry"]

from typing                     import List, Type, TYPE_CHECKING

from gradus.registration.core   import Entry

# Defer until runtime.
if TYPE_CHECKING:
    from gradus.configuration   import DatasetConfig
    from gradus.datasets        import Dataset

class DatasetEntry(Entry):
    """# Dataset Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type["Dataset"],
        config: Type["DatasetConfig"],
        tags:   List[str] =             []
    ):
        """# Instantiate Dataset Registration Entry.

        ## Args:
            * id        (str):                  Dataset identifier.
            * cls       (Type[Dataset]):        Dataset class.
            * config    (Type[DatasetConfig]):  Dataset configuration.
            * tags      (List[str]):            Taxonomical keywords applicable to dataset.
        """
        # Initialize entry.
        super(DatasetEntry, self).__init__(id = id, config = config, tags = tags)

        # Define properties.
        self._cls_: Type["Dataset"] = cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type["Dataset"]:
        """# Dataset Class"""
        return self._cls_