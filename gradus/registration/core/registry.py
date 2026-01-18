"""# gradus.registration.core.registry

Abstract registry system protocol.
"""

__all__ = ["Registry"]

from abc                                    import ABC, abstractmethod
from argparse                               import _SubParsersAction
from logging                                import Logger
from typing                                 import Dict, List

from gradus.registration.core.entry         import Entry
from gradus.registration.core.exceptions    import DuplicateEntryError, EntryNotFoundError
from gradus.registration.core.types         import EntryType
from gradus.utilities                       import get_logger

class Registry(ABC):
    """# Abstract Registry System"""

    def __init__(self,
        id: str
    ):
        """# Instantiate Registry System.

        ## Args:
            * id    (str):  Registry ID.
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger(f"{id}-registry")

        # Define properties.
        self._id_:          str =               id
        self._entries_:     Dict[str, Entry] =  {}
        self._loaded_:      bool =              False

    # PROPERTIES ===================================================================================

    @property
    def entries(self) -> Dict[str, Entry]:
        """# Registered Entries"""
        return self._entries_.copy()
    
    @property
    def id(self) -> str:
        """# Registry Identifier"""
        return self._id_
    
    @property
    def is_loaded(self) -> bool:
        """# Registry is Loaded?"""
        return self._loaded_
    
    # METHODS ======================================================================================

    def get_entry(self,
        entry_id:   str
    ) -> Entry:
        """# Query for Registered Entry.

        ## Args:
            * entry_id  (str):  Identifier of entry being queried.

        ## Raises:
            * EntryNotFoundError:   If entry queried is not registered.

        ## Returns:
            * Entry:    Queried entry, if registered.
        """
        # Ensure that registry is loaded.
        self._ensure_loaded_()

        # If queried entry is not registered...
        if entry_id not in self._entries_:

            # Report error.
            raise EntryNotFoundError(entry_id = entry_id, registry_id = self._id_)
        
        # Debug query.
        self.__logger__.debug(f"Entry queried: {entry_id}")

        # Query entry.
        return self._entries_[entry_id]
    
    def list_entries(self,
        filter_by:  List[str] = []
    ) -> List[str]:
        """# List Registered Entries.

        ## Args:
            * filter_by (List[str]):    Taxonomical keywords by which entries will be filtered.

        ## Returns:
            * List[str]:    List of registered entry IDs.
        """
        # Ensure that registry is loaded.
        self._ensure_loaded_()

        # Debug action.
        self.__logger__.debug(f"List entries filtered by {filter_by}")

        # If no filter is provided, return all entries.
        if len(filter_by) == 0: return list(self._entries_.keys())

        # Otherwise, return filtered entries.
        return  [
                    id for id, entry
                    in self._entries_.items()
                    if  all(
                            tag in entry.tags
                            for tag
                            in filter_by
                        )
                ]
    
    def register(self,
        entry_id:   str,
        **kwargs
    ) -> None:
        """# Register Entry.

        ## Args:
            * entry_id  (str):  Identifier of entry.

        ## Raises:
            * DuplicateEntryError:  If entry is already registered.
        """
        # If entry is already registered...
        if entry_id in self._entries_:

            # Report duplication.
            raise DuplicateEntryError(entry_id = entry_id, registry_id = self._id_)
        
        # Debug registration.
        self.__logger__.debug(f"Registering {entry_id} with arguments: {kwargs}")

        # Create & register entry.
        self._entries_[entry_id] = self._create_entry_(id = entry_id, **kwargs)

    def register_configurations(self,
        subparser:  _SubParsersAction
    ) -> None:
        """# Register Configurations of All Current Entries.

        ## Args:
            * subparser (_SubParsersAction):    Sub-parser group of parent command under which entry 
                                                configurations will be registered.
        """
        # Ensure that registry is loaded.
        self._ensure_loaded_()

        # For each registered entry...
        for entry in self._entries_.values():

            # Debug action.
            self.__logger__.debug(f"Registered configuration for {entry.id}")

            # Register configuration.
            entry.register_configuration(subparser = subparser)

    # HELPERS ======================================================================================

    @abstractmethod
    def _create_entry_(self, **kwargs) -> EntryType:
        """# Create Registration Entry.
        
        Factory method to create the appropriate entry type for this registry.

        ## Returns:
            * EntryType:    New entry instance.
        """
        pass

    def _ensure_loaded_(self) -> None:
        """# Ensure Registry is Loaded."""
        if not self._loaded_: self._load_all_()

    def _import_all_modules_(self) -> None:
        """# Import All Applicable Modules."""
        from importlib  import import_module
        from pkgutil    import walk_packages
        from types      import ModuleType

        try:# Import the main package to get its path.
            package:    ModuleType =    import_module(f"gradus.{self._id_}")

        # If import error occurs...
        except ImportError as e:

            # Warn of complications.
            self.__logger__.warning(f"Error importing package gradus.{self._id_}: {e}"); return
        
        # Debug action.
        self.__logger__.debug(f"Walking package: {package}")

        try:# For each module within package...
            for _, module, _ in walk_packages(
                path =      package.__path__,
                prefix =    f"gradus.{self._id_}.",
                onerror =   lambda x: None
            ):
                try:# Attempt import of module.
                    import_module(name = module)

                    # Debug action.
                    self.__logger__.debug(f"Walk of {module} complete")

                # If import error occurs...
                except ImportError as e:

                    # Warn of complications.
                    self.__logger__.warning(f"Error importing {module}: {e}")

        # If package cannot be imported...
        except ImportError as e:

            # Warn of error.
            self.__logger__.warning(f"Error importing {package}: {e}")

    def _load_all_(self) -> None:
        """# Load All Applicable Modules."""
        # Import modules.
        self._import_all_modules_()

        # Debug action.
        self.__logger__.debug(f"Registry loaded")

        # Update states.
        self._loaded_:  bool =  True

    # DUNDERS ======================================================================================

    def __contains__(self,
        entry_id:   str
    ) -> bool:
        """# Entry is Registered?

        ## Args:
            * entry_id  (str):  ID of entry being queried.

        ## Returns:
            * bool: True if entry is registered.
        """
        return entry_id in self._entries_
    
    def __getitem__(self,
        entry_id:   str
    ) -> Entry:
        """# Query for Registered Entry.

        ## Args:
            * entry_id  (str):  Identifier of entry being queried.

        ## Raises:
            * EntryNotFoundError:   If entry queried is not registered.

        ## Returns:
            * Entry:    Queried entry, if registered.
        """
        return self.get_entry(entry_id = entry_id)
    
    def __len__(self) -> int:
        """# Quantity of Registered Entries"""
        return len(self._entries_)
    
    def __repr__(self) -> str:
        """# Registry System Object Representation"""
        return f"""<{self._id_.capitalize()}Registry({len(self._entries_)} entries)>"""