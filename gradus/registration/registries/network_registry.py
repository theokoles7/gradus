"""# gradus.registration.registries.network_registry

Neural network registry system implementation.
"""

__all__ = ["NetworkRegistry"]

from typing                         import Dict, override

from torch.nn                       import Module

from gradus.registration.core       import Registry
from gradus.registration.entries    import NetworkEntry

class NetworkRegistry(Registry):
    """# Neural Network Registration System"""

    def __init__(self):
        """# Instantiate Neural Network Registry System"""
        # Initialize registry.
        super(NetworkRegistry, self).__init__(id = "networks")

    # PROPERTIES ===================================================================================

    @override
    @property
    def entries(self) -> Dict[str, NetworkEntry]:
        """# Registered Neural Network Entries"""
        return self._entries_.copy()
    
    # METHODS ======================================================================================

    def load_network(self,
        network_id: str,
        *args,
        **kwargs
    ) -> Module:
        """# Load & Instantiate Neural Network.

        ## Args:
            * network_id    (str):  Name of neural network being loaded.

        ## Returns:
            * Module:   Neural network instantiated with provided arguments.
        """
        # Query for registered neural network.
        entry:  NetworkEntry =  self.get_entry(entry_id = network_id)

        # Debug loading.
        self.__logger__.debug(f"Loading {network_id}: {kwargs}")

        # Load neural network.
        return entry.cls(*args, **kwargs)
    
    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> NetworkEntry:
        """# Create Neural Network Entry.

        ## Returns:
            * NetworkEntry: New neural network entry instance.
        """
        return NetworkEntry(**kwargs)
    
    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        network_id: str
    ) -> NetworkEntry:
        """# Query Registered Neural Networks.

        ## Args:
            * network_id    (str):  Identifier of neural network being queried.

        ## Raises:
            * EntryNotFoundError:   If dataset queried is not registered.

        ## Returns:
            * NetworkEntry: Neural network entry, if registered.
        """
        return self.get_entry(entry_id = network_id)