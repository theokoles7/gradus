"""# gradus.registration.entries.network_entry

Defines structure & utility of neural network registration entry.
"""

__all__ = ["NetworkEntry"]

from typing                     import List, Type, TYPE_CHECKING

from torch.nn                   import Module

from gradus.registration.core   import Entry

# Defer until runtime.
if TYPE_CHECKING:
    from gradus.configuration   import NetworkConfig

class NetworkEntry(Entry):
    """# Neural Network Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type[Module],
        config: Type["NetworkConfig"],
        tags:   List[str] =             []
    ):
        """# Instantiate Neural Network Registration Entry.

        ## Args:
            * id        (str):                  Neural network identifier.
            * cls       (Type[Module]):         Neural network class.
            * config    (Type[NetworkConfig]):  Neural network configuration handler class.
            * tags      (List[str]):            Taxonomical keywords applicable to neural network.
        """
        # Initialize entry.
        super(NetworkEntry, self).__init__(id = id, config = config, tags = tags)

        # Define properties.
        self._cls_: Type[Module] =   cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type[Module]:
        """# Neural Network Class"""
        return self._cls_