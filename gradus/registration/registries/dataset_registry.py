"""# gradus.registration.registries.dataset_registry

Dataset registry system implementation.
"""

__all__ = ["DatasetRegistry"]

from typing                         import Dict, override

from gradus.registration.core       import Registry
from gradus.datasets                import GradusDataset
from gradus.registration.entries    import DatasetEntry

class DatasetRegistry(Registry):
    """# Dataset Registry System"""

    def __init__(self):
        """# Instantiate Dataset Registry System."""
        # Initialize registry.
        super(DatasetRegistry, self).__init__(id = "datasets")

    # PROPERTIES ===================================================================================

    @override
    @property
    def entries(self) -> Dict[str, DatasetEntry]:
        """# Registered Dataset Entries"""
        return self._entries_.copy()
    
    # METHODS ======================================================================================

    def load_dataset(self,
        dataset_id: str,
        *args,
        **kwargs
    ) -> GradusDataset:
        """# Load & Instantiate Dataset.

        ## Args:
            * dataset_id    (str):  Name of dataset being loaded.

        ## Returns:
            * GradusDataset:    Dataset instantiated with provided arguments.
        """
        # Query for registered dataset.
        entry:  DatasetEntry =  self.get_entry(entry_id = dataset_id)

        # Debug loading.
        self.__logger__.debug(f"Loading {dataset_id}: {kwargs}")

        # Load dataset.
        return entry.cls(*args, **kwargs)
    
    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> DatasetEntry:
        """# Create Dataset Entry.

        ## Returns:
            * DatasetEntry: New dataset entry instance.
        """
        return DatasetEntry(**kwargs)

    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        dataset_id: str
    ) -> DatasetEntry:
        """# Query Registered Commands.

        ## Args:
            * dataset_id    (str):  Identifier of dataset being queried.

        ## Raises:
            * EntryNotFoundError:   If dataset queried is not registered.

        ## Returns:
            * DatasetEntry: Dataset entry, if registered.
        """
        return self.get_entry(entry_id = dataset_id)