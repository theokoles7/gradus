"""# gradus.registration.entries.rank_entry

Define structure & utility of rank registration entry.
"""

__all__ = ["RankEntry"]

from typing                     import Callable, List, Type, TYPE_CHECKING

from gradus.registration.core   import Entry

# Defer until runtime.
if TYPE_CHECKING:

    from gradus.curricula.ranks import Rank

class RankEntry(Entry):
    """# Rank Registration Entry"""

    def __init__(self,
        id:     str,
        cls:    Type[Rank],
        tags:   List[str] = []
    ):
        """# Instantiate Rank Registration Entry.

        ## Args:
            * id    (str):          Rank identifier.
            * cls   (Type[Module]): Curriculum ranking class.
            * tags  (List[str]):    Taxonomical keywords applicable to metric.
        """
        # Initialize entry.
        super(RankEntry, self).__init__(id = id, tags = tags)

        # Define properties.
        self._cls_: Type[Rank] =   cls

    # PROPERTIES ===================================================================================

    @property
    def cls(self) -> Type[Rank]:
        """# Curriculum Ranking Class"""
        return self._cls_