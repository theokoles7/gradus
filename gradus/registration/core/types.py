"""# gradus.registration.core.types

Registration component typing classes.
"""

__all__ =   [
                "EntryType",
            ]

from typing                         import TypeVar

from gradus.registration.core.entry import Entry


EntryType = TypeVar(name = "EntryType", bound = Entry)