"""# gradus.args

Argument definitions & parsing for Gradus application.
"""

__all__ = ["parse_gradus_arguments"]

from argparse               import _ArgumentGroup, ArgumentParser, Namespace, _SubParsersAction
from typing                 import Optional, Sequence

from gradus.registration    import COMMAND_REGISTRY

def parse_gradus_arguments(
    args:       Optional[Sequence[str]] =   None,
    namespace:  Optional[Namespace] =       None
) -> Namespace:
    """# Parse Gradus Arguments.

    ## Args:
        * args      (Sequence[str] | None): Sequence of string arguments.
        * namespace (Namespace | None):     Mapping of arguments to their values.

    ## Returns:
        * Namespace:    Mapping of arugments & their values.
    """
    # Initilize parser.
    parser:     ArgumentParser =    ArgumentParser(
                                        prog =          "gradus",
                                        description =   """Experiments in quantifying image 
                                                        complexity and determining the efficacy of 
                                                        curriculum learning in image classification 
                                                        tasks."""
                                    )
    
    # Initialize sub-parser.
    subparser:  _SubParsersAction = parser.add_subparsers(
                                        title =         "gradus-command",
                                        dest =          "gradus_command",
                                        help =          """Gradus command being executed.""",
                                        description =   """Gradus command being executed."""
                                    )
    
    # +============================================================================================+
    # | BEGIN ARGUMENTS                                                                            |
    # +============================================================================================+

    # LOGGING ======================================================================================
    logging:     _ArgumentGroup =    parser.add_argument_group(
                                        title =         "Logging",
                                        description =   """Logging utility configuration."""
                                    )

    logging.add_argument(
        "--logging-level",
        dest =      "logging_level",
        type =      str,
        choices =   ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"],
        default =   "INFO",
        help =      """Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). 
                    Defaults to "INFO"."""
    )

    logging.add_argument(
        "--logging-path",
        dest =      "logging_path",
        type =      str,
        default =   "logs",
        help =      """Path at which logs will be written. Defaults to "./logs/"."""
    )

    logging.add_argument(
        "--debug", "-v",
        dest =      "logging_level",
        action =    "store_const",
        const =     "DEBUG",
        help =      """Set logging level to DEBUG."""
    )
    
    # +============================================================================================+
    # | END ARGUMENTS                                                                              |
    # +============================================================================================+

    # Register commands.
    COMMAND_REGISTRY.register_configurations(subparser = subparser)

    # Parse arguments.
    return parser.parse_args(args, namespace)