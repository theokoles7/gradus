"""# gradus.configuration.dataset_config

Dataset configuration & parsing handler implementation.
"""

__all__ = ["DatasetConfig"]

from argparse                       import _ArgumentGroup, ArgumentParser
from typing                         import override

from gradus.configuration.protocol  import Config
from gradus.utilities               import get_system_core_count

class DatasetConfig(Config):
    """# Dataset Configuration & Argument Handler"""

    def __init__(self,
        name:   str,
        help:   str,
    ):
        """# Instantiate Dataset Configuration.

        ## Args:
            * name  (str):  Dataset identifier.
            * help  (str):  Description of dataset.
        """
        # Initialize configuration.
        super(DatasetConfig, self).__init__(
            parser_id =         name,
            parser_help =       help,
            subparser_title =   f"{name}-command",
            subparser_help =    f"{name.upper()} dataset operations."
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Dataset Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attriuted.
        """
        # GENERAL ----------------------------------------------------------------------------------
        general:    _ArgumentGroup =    parser.add_argument_group(
                                            title =         "General",
                                            description =   """General dataset configuration 
                                                            parameters."""
                                        )
        
        general.add_argument(
            "--root",
            dest =      "root",
            type =      str,
            default =   "data",
            help =      """Path to directory from/to which datasets should be loaded/downloaded. 
                        Defaults to "./data/"."""
        )

        general.add_argument(
            "--batch-size",
            dest =      "batch_size",
            type =      int,
            default =   64,
            help =      """Number of samples per batch to load. Defaults to 64."""
        )

        general.add_argument(
            "--max-workers",
            dest =      "max_workers",
            type =      int,
            default =   get_system_core_count(),
            help =      """Maximum number of worker threads to use for data loading. Defaults to the 
                        number of CPU cores available on the system."""
        )