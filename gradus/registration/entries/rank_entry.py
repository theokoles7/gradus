"""# gradus.registration.entries.rank_entry

Define structure & utility of rank registration entry.
"""

__all__ = ["RankEntry"]

from typing                     import Callable, List

from gradus.registration.core   import Entry

class RankEntry(Entry):
    """# Rank Registration Entry"""

    def __init__(self,
        id:     str,
        fn:     Callable,
        tags:   List[str] = []
    ):
        """# Instantiate Rank Registration Entry.

        ## Args:
            * id    (str):          Rank identifier.
            * fn    (Callable):     Rank ordering scheme function.
            * tags  (List[str]):    Taxonomical keywords applicable to metric.
        """
        # Initialize entry.
        super(RankEntry, self).__init__(id = id, tags = tags)

        # Define properties.
        self._fn_:  Callable =  fn

    # PROPERTIES ===================================================================================

    @property
    def fn(self) -> Callable:
        """# Rank Ordering Scheme Function"""
        return self._fn_