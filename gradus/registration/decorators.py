"""# gradus.registration.decorators

Function annotation decorators for registration of components.
"""

__all__ =   [
                "register_command",
                "register_dataset",
            ]

from typing                 import Callable, List, Type

from gradus.configuration   import CommandConfig, DatasetConfig
from gradus.datasets        import GradusDataset


def register_command(
    id:     str,
    config: Type[CommandConfig]
) -> Callable:
    """# Register Command.

    ## Args:
        * id        (str):                  Command identifier/parser ID.
        * config    (Type[CommandConfig]):  Command's configuration handler class.

    ## Returns:
        * Callable: Registration decorator.
    """
    # Define decorator.
    def decorator(
        entry_point:    Callable
    ) -> Callable:
        """# Command Registration Decorator.

        ## Args:
            * entry_point   (Callable): Command's main process entry point.
        """
        # Load registry.
        from gradus.registration    import COMMAND_REGISTRY

        # Register command.
        COMMAND_REGISTRY.register(
            entry_id =      id,
            config =        config,
            entry_point =   entry_point
        )

        # Expose entry point.
        return entry_point
    
    # Expose decorator.
    return decorator


def register_dataset(
    id:     str,
    config: Type[DatasetConfig],
    tags:   List[str] =         []
) -> Callable:
    """# Register Dataset.

    ## Args:
        * id        (str):                  Dataset identifier/parser ID.
        * config    (Type[DatasetConfig]):  Dataset's configuration handler class.
        * tags      (List[str]):            Taxonomical dataset keywords.

    ## Returns:
        * Callable: Registration decorator.
    """
    # Define decorator.
    def decorator(
        cls:    Type[GradusDataset]
    ) -> Type[GradusDataset]:
        """# Dataset Registration Decorator.

        ## Args:
            * cls   (Type[GradusDataset]):  Dataset class being registered.
        """
        # Load registry.
        from gradus.registration    import DATASET_REGISTRY

        # Register dataset.
        DATASET_REGISTRY.register(
            entry_id =  id,
            cls =       cls,
            config =    config,
            tags =      tags
        )

        # Expose dataset class.
        return cls
    
    # Expose decorator.
    return decorator