"""# gradus.networks.protocol

Abstract network protocol.
"""

from abc                import ABC
from logging            import Logger
from pathlib            import Path
from typing             import Any, Dict, Union

from torch.nn           import Module

from gradus.utilities   import get_logger

class Network(Module, ABC):
    """# Gradus Network Wrapper & Protocol."""

    def __init__(self,
        network_id:     str
    ):
        """# Instantiate Neural Network.

        ## Args:
            * network_id    (str):          Network identifier.
            * weights_path  (str |Path):    Path to/from which network weights will be saved/loaded. 
                                            Defaults to ".cache/weights/".
        """
        # Initialize module.
        super(Network, self).__init__()

        # Initialize logger.
        self.__logger__:    Logger =    get_logger(f"{network_id}-network")

        # Define properties.
        self._id_:          str =       network_id

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def id(self) -> str:
        """# Network Identifier"""
        return self._id_
    
    # METHODS ======================================================================================

    def load_weights(self,
        path:   Union[str, Path]
    ) -> None:
        """# Load Network Weights.

        ## Args:
            * path  (Union[str, Path]): Path from which weights will be loaded.
        """
        from torch  import load

        # Load weights.
        self.load_state_dict(load(Path(path), weights_only = True))

        # Debug action.
        self.__logger__.debug(f"Weights loaded from {path}")
    
    def save_weights(self,
        path:   Union[str, Path]
    ) -> Path:
        """# Save Network Weights.

        ## Args:
            * path  (Union[str, Path]): Path at which network weights will be saved.

        ## Returns:
            * Path: Path at which weights were saved.
        """
        from torch  import save

        # Resolve path.
        weights_file:   Path =  Path(path)

        # Ensure path exists.
        weights_file.parent.mkdir(parents = True, exist_ok = True)

        # Save weights.
        save(self.state_dict(), weights_file)

        # Debug action.
        self.__logger__.debug(f"Weights saved to {weights_file}")

        # Provide path at which weights were saved.
        return weights_file

    def to_dict(self) -> Dict[str, Any]:
        """# Network Dictionary Representation

        ## Returns:
            * Dict[str, Any]:   Network configuration.
        """
        return  {"id":  self._id_}
    
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Network Object Representation"""
        return f"""<{self._id_.upper()}Network>"""