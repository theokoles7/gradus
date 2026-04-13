"""# gradus.registration.entries.schedule_entry

Define structure & utility of schedule registration entry.
"""

__all__ = ["ScheduleEntry"]

from typing                         import List, Type, TYPE_CHECKING

from gradus.registration.core       import Entry

# Defer until runtime.
if TYPE_CHECKING:

    from gradus.curricula.schedules import Schedule

class ScheduleEntry(Entry):
    """# Schedule Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type["Schedule"],
        tags:   List[str]
    ):
        """# Instantiate Schedule Registration Entry.

        ## Args:
            * id    (str):              Schedule identifier.
            * cls   (Type[Schedule]):   Curriculum schedule class.
            * tags  (List[str]):        Taxonomical keywords applicable to schedule.
        """
        # Initialize entry.
        super(ScheduleEntry, self).__init__(id = id, tags = tags)

        # Define properties.
        self._cls_: Type["Schedule"] =  cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type["Schedule"]:
        """# Curriculum Schedule Class"""
        return self._cls_