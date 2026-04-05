"""# gradus.curricula.ranks.protocol

Curriculum ranking protocol implementation.
"""

__all__ = ["Rank"]

from abc                import ABC, abstractmethod
from logging            import Logger
from pathlib            import Path
from typing             import List, Union

from pandas             import DataFrame

from gradus.utilities   import get_logger

class Rank(ABC):
    """# Abstract Curriculum Ranking"""

    def __init__(self,
        rank_id:    str,
        scores:     DataFrame,
        cache_dir:  Union[str, Path] =  ".cache/ranks"
    ):
        """# Instantiate Curriculum Ranking.

        ## Args:
            * rank_id   (str):          Rank identifier.
            * scores    (DataFrame):    Dataset metric scores.
            * cache_dir (str | Path):   Directory under which keyed indices will be cached. Defaults 
                                        to "./.cache/ranks/".
        """
        # Initialize logger.
        self.__logger__:    Logger =    get_logger(f"{rank_id}-ranker")

        # Define properties.
        self._id_:          str =       rank_id
        self._scores_:      DataFrame = scores
        self._cache_dir_:   Path =      Path(cache_dir)
        self._cache_path_:  Path =      self._cache_dir_ / f"{self._id_}.npy"
        self._indices_:     List[int] = self._load_()                   \
                                        if self._cache_path_.exists()   \
                                        else self._compute_and_save_()

    # PROPERTIES ===================================================================================

    @property
    def indices(self) -> List[int]:
        """# Ranked Sample Indices"""
        return self._indices_

    # HELPERS ======================================================================================

    def _compute_and_save_(self) -> List[int]:
        """# Compute & Cache Indices.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        from numpy  import array, int64, save

        # Log action.
        self.__logger__.info(f"Computing {self._id_} ranks; Caching to {self._cache_dir_}")

        # Compute rank.
        indices:    List[int] = self._rank_()
        
        # Ensure cache directory exists.
        self._cache_dir_.mkdir(parents = True, exist_ok = True)

        # Cache indices for future use.
        save(self._cache_path_, arr = array(indices, dtype = int64))

        # Provide computed. indices.
        return indices
    
    def _load_(self) -> List[int]:
        """# Load Cached Ranked Indices.

        ## Returns:
            * List[int]:    Ranked indices.
        """
        from numpy  import load

        # Log action.
        self.__logger__.info(f"Loading cached ranks from {self._cache_dir_}")

        # Load ranks.
        return load(self._cache_path_).tolist()

    @abstractmethod
    def _rank_(self) -> List[int]:
        """# Compute Ranked Indices.

        ## Returns:
            * List[int]:    Ranked sample indices.
        """
        ...