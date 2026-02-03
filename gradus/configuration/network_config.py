"""# gradus.configuration.network_config

Neural network configuration & parsing hander implementation.
"""

__all__ = ["NetworkConfig"]

from argparse                       import _ArgumentGroup, ArgumentParser, _SubParsersAction
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
            subparser_title =   f"dataset-id",
            subparser_help =    f"Dataset with which neural network will operate."
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Neural Network Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        from gradus.registration            import DATASET_REGISTRY
        
        # ARTIFACTS --------------------------------------------------------------------------------
        artifacts:  _ArgumentGroup =    parser.add_argument_group(
                                            title =         "Artifacts",
                                            description =   """Artifact management."""
                                        )
        
        artifacts.add_argument(
            "--checkpoint-path",
            dest =      "checkpoint_path",
            type =      str,
            default =   "checkpoints",
            help =      """Path to directory to/from which model checkpoint files can be 
                        saved/loaded. Defaults to "./checkpoints/"."""
        )

        # Create sub-parser.
        subparser:  _SubParsersAction = self._create_subparser_(parser = parser)

        # Register datasets as sub-command.
        DATASET_REGISTRY.register_configurations(subparser = subparser)