"""# gradus.main

Primary application process.
"""

__all__ = ["gradus_entry_point"]

from argparse   import Namespace
from logging    import Logger
from typing     import Any

def gradus_entry_point(*args, **kwargs) -> Any:
    """# Execute Gradus Command.

    ## Returns:
        * Any:  Data returned from sub-process(es).
    """
    # Package imports.
    from gradus.__args__       import parse_gradus_arguments
    from gradus.registration   import COMMAND_REGISTRY
    from gradus.utilities      import configure_logger

    # Parse arguments.
    arguments:  Namespace = parse_gradus_arguments(*args, **kwargs)

    # Initialize logger.
    logger:     Logger =    configure_logger(
                                logging_level = arguments.logging_level,
                                logging_path =  arguments.logging_path
                            )
    
    # Debug arguments.
    logger.debug(f"Gradus arguments: {vars(arguments)}")

    try:# Dispatch to command.
        COMMAND_REGISTRY.dispatch(command_id = arguments.gradus_command, **vars(arguments))

    # Catch wildcard errors.
    except Exception as e:  logger.critical(f"Unexpected error: {e}", exc_info = True)

    # Exit gracefully.
    finally:                logger.debug("Exiting...")


if __name__ == "__main__": gradus_entry_point()