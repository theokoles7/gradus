"""# gradus.curricula.ranks.protocol

Curriculum ranking protocol implementation.
"""

__all__ = ["Rank"]

from abc        import ABC, abstractmethod
from pathlib    import Path
from typing     import List, Union

from pandas     import DataFrame

class Rank(ABC):
    """# Abstract Curriculum Ranking"""

    _metric_:   List[str]

    def __init__(self,
        rank_id:    str,
        dataset_id: str,
        scores:     DataFrame,
        seed:       int =                   1,
        cache_dir:  Union[str, Path] =      ".cache/ranks"
    ):
        """# Instantiate Curriculum Ranking.

        ## Args:
            * rank_id       (str):              Rank identifier.
            * dataset_id    (str):              Identifier of dataset whose samples are being 
                                                ranked.
            * scores        (DataFrame):        Dataset metric scores.
            * seed          (int):              Random number generation seed. Defaults to 1.
            * cache_dir     (str | Path):       Directory under which keyed indices will be cached. 
                                                Defaults to "./.cache/ranks/".
        """
        from hashlib            import md5
        from json               import dumps
        from logging            import Logger

        from gradus.utilities   import get_logger

        # Initialize logger.
        self.__logger__:    Logger =    get_logger(f"{rank_id}-ranker")

        # Define properties.
        self._id_:          str =       rank_id
        self._dataset_id_:  str =       dataset_id
        self._scores_:      DataFrame = scores
        self._seed_:        int =       seed

        # Resolve cache paths.
        self._cache_dir_:   Path =      Path(cache_dir)
        self._cache_key_:   str =       md5(dumps({
                                            "rank":     self._id_,
                                            "dataset":  self._dataset_id_,
                                            "metric":   self._metric_,
                                            "seed":     self._seed_
                                        }).encode()).hexdigest()
        self._cache_path_:  Path =      self._cache_dir_ / f"{self._cache_key_}.npy"

        # Rank indices.
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