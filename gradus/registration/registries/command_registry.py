"""# gradus.registration.registries.command_registry

Command registry system implementation.
"""

__all__ = ["CommandRegistry"]

from typing                         import Any, Dict, override

from gradus.registration.core       import EntryPointNotConfiguredError, Registry
from gradus.registration.entries    import CommandEntry

class CommandRegistry(Registry):
    """# Command Registry System"""

    def __init__(self):
        """# Instantiate Command Registry System."""
        # Initialize registry.
        super(CommandRegistry, self).__init__(id = "commands")

    # PROPERTIES ===================================================================================

    @override
    @property
    def entries(self) -> Dict[str, CommandEntry]:
        """# Registered Command Entries"""
        return self._entries_.copy()

    # METHODS ======================================================================================

    def dispatch(self,
        command_id: str,
        *args,
        **kwargs
    ) -> Any:
        """# Dispatch Arguments to Command Entry Point.

        ## Args:
            * command_id    (str):  Identifier of command to whom arguments are being dispatched.

        ## Raises:
            * EntryPointNotConfiguredError: If command was not registered with an entry point.

        ## Returns:
            * Any:  Data returned from command process.
        """
        # Query for registered command.
        entry:  CommandEntry =  self.get_entry(entry_id = command_id)

        # If this command was not registered with an entry point, report error.
        if entry.entry_point is None: raise EntryPointNotConfiguredError(entry_id = command_id)

        # Debug dispatch.
        self.__logger__.debug(f"Dispatching {command_id}: {kwargs}")

        # Dispatch command.
        return entry.entry_point(*args, **kwargs)

    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> CommandEntry:
        """# Create Command Entry.
        
        ## Args:
            * CommandEntry: New command entry instance.
        """
        return CommandEntry(**kwargs)

    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        command_id: str
    ) -> CommandEntry:
        """# Query Registered Commands.

        ## Args:
            * command_id    (str):  Identifier of command being queried.

        ## Raises:
            * EntryNotFoundError:   If command queried is not registered.

        ## Returns:
            * CommandEntry: Command entry, if registered.
        """
        return self.get_entry(entry_id = command_id)