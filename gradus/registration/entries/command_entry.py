"""# gradus.registration.entries.command_entry

Defines structure & utility of command registration entry.
"""

__all__ = ["CommandEntry"]

from typing                     import Callable

from gradus.configuration       import CommandConfig
from gradus.registration.core   import Entry

class CommandEntry(Entry):
    """# Command Registration Entry"""

    def __init__(self,
        id:             str,
        config:         CommandConfig,
        entry_point:    Callable
    ):
        """# Instantiate Command Registration Entry.

        ## Args:
            * id            (str):              Name of command/ID of command parser.
            * config        (CommandConfig):    Command's argument configuration.
            * entry_point   (Callable):         Command's main process entry point.
        """
        # Initialize entry.
        super(CommandEntry, self).__init__(id = id, config = config)

        # Define properties.
        self._entry_point_: Callable =  entry_point

    # PROPERTIES ===================================================================================

    @property
    def entry_point(self) -> Callable:
        """# Main Process Entry Point"""
        return self._entry_point_