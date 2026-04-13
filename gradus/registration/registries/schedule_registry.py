"""# gradus.registration.registries.schedule_registry

Schedule registry system implementation.
"""

__all__ = ["ScheduleRegistry"]

from typing                         import Dict, override, Type

from gradus.registration.core       import Registry
from gradus.registration.entries    import ScheduleEntry


class ScheduleRegistry(Registry):
    """# Schedule Registry System"""

    def __init__(self):
        """# Instantiate Schedule Registry System."""
        super(ScheduleRegistry, self).__init__(id = "curricula.schedules")

    # PROPERTIES ===================================================================================

    @property
    def entries(self) -> Dict[str, ScheduleEntry]:
        """# Registered Schedule Entries"""
        return self._entries_.copy()

    # METHODS ======================================================================================

    def load_schedule(self,
        schedule_id:    str,
        *args,
        **kwargs
    ) -> Type:
        """# Load Schedule Instance.

        ## Args:
            * schedule_id   (str):  Identifier of schedule to load.

        ## Returns:
            * Schedule: Instantiated schedule.
        """
        # Query for registered schedule.
        entry:  ScheduleEntry = self.get_entry(entry_id = schedule_id)

        # Debug loading.
        self.__logger__.debug(f"Loading {schedule_id}: {kwargs}")

        # Load schedule.
        return entry.cls(*args, **kwargs)

    # HELPERS ======================================================================================

    @override
    def _create_entry_(self, **kwargs) -> ScheduleEntry:
        """# Create Schedule Entry.
        
        ## Returns:
            * ScheduleEntry:    New schedule entry instance.
        """
        return ScheduleEntry(**kwargs)

    # DUNDERS ======================================================================================

    @override
    def __getitem__(self,
        schedule_id: str
    ) -> ScheduleEntry:
        """# Query Registered Schedules.
        
        ## Args:
            * schedule_id   (str):  Identifier of schedule being queried.

        ## Raises:
            * EntryNotFoundError:   If schedule queried is not registered.

        ## Returns:
            * ScheduleEntry:    Schedule entry, if registered.
        """
        return self.get_entry(entry_id = schedule_id)