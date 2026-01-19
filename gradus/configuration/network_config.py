"""# gradus.configuration.network_config

Neural network configuration & parsing hander implementation.
"""

__all__ = ["NetworkConfig"]

from argparse                       import _ArgumentGroup, ArgumentParser
from typing                         import override

from gradus.configuration.protocol  import Config

class NetworkConfig(Config):
    """# Neural Network Configuration & Argument Handler"""

    def __init__(self,
        name:   str,
        help:   str
    ):
        """# Instantiate Neural Network Configuration.
        
        ## Args:
            * name  (str):  Neural network identifier.
            * help  (str):  Description of neural network.
        """
        # Initialize configuration.
        super(NetworkConfig, self).__init__(
            parser_id =         name,
            parser_help =       help,
            subparser_title =   f"{name}-command",
            subparser_help =    f"{name.upper()} neural network operations."
        )

    # HELPERS ======================================================================================