"""# gradus.artifacts.dataset_metrics

Data structure implementation for storing dataset metric scores & statistics.
"""

__all__ = ["DatasetMetrics"]

from logging    import Logger
from pathlib    import Path
from typing     import Dict, List, Optional, Union

from pandas     import concat, DataFrame, read_csv, Series

class DatasetMetrics():
    """# Dataset Metrics
    
    Dataset metric scores & statistics.
    """

    # Define meta (non-metric) columns.
    _META_COLUMNS_: List[str] = ["index", "class"]

    def __init__(self,
        dataset_id:     str,
        num_samples:    int,
        seed:           int =               1,
        scores_path:    Union[str, Path] =  ".cache/scores",
    ):
        """# Instantiate Dataset Metrics.
        
        ## Args:
            * dataset_id    (str):          Identifier of dataset to whom scores are attributed.
            * num_samples   (int):          Number of samples within dataset.
            * seed          (int):          Random seed used when scores were computed. Used to 
                                            distinguish score files across runs for 
                                            non-deterministic metrics. Defaults to 1.
            * scores_path   (str | Path):   Path at which dataset scores will be written/managed. 
                                            Defaults to "./.cache/scores".
        """
        from gradus.utilities   import get_logger

        # Initialize logger.
        self.__logger__:    Logger =    get_logger(f"{dataset_id}-metrics")

        # Define properties.
        self._dataset_id_:  str =                   dataset_id
        self._num_samples_: int =                   num_samples
        self._seed_:        int =                   seed
        self._root_:        Path =                  Path(scores_path) / dataset_id
        self._scores_:      Optional[DataFrame] =   None

        # Load scores.
        self._load_()

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def dataset_id(self) -> str:
        """# Dataset Identifier"""
        return self._dataset_id_
    
    @property
    def metrics(self) -> List[str]:
        """# Metric Identifiers in Loaded Scores"""
        return [m for m in self._scores_.columns if m not in self._META_COLUMNS_]
    
    @property
    def scores(self) -> DataFrame:
        """# Loaded Scores DataFrame"""
        return self._scores_
    
    @property
    def scores_path(self) -> Path:
        """# Absolute Path to Score Files"""
        return self._root_ / f"metric-scores_seed-{self._seed_}.csv"
    
    @property
    def seed(self) -> int:
        """# Seed Used When Scores Were Computed"""
        return self._seed_
    
    # METHODS ======================================================================================

    def get(self,
        index:  int,
        metric: str
    ) -> Union[int, float]:
        """# Get Metric for Sample.

        ## Args:
            * index     (int):  Index of sample within dataset.
            * metric    (str):  Sample's metric being queried.

        ## Returns:
            * Union[int, float]:    Sample's metric score.
        """
        # If metric is not defined, report error.
        if metric not in self.metrics: raise KeyError(f"Metric not defined: {metric}")

        # LOcate row by sample index value.
        row:    Series =    self._scores_.loc[self._scores_["index"] == index, metric]

        # If row is empty, report error.
        if row.empty: raise IndexError(f"No score found index: {index}")

        # Provide queried score.
        return row.iloc[0]
    
    def get_unscored(self,
        metric: str
    ) -> List[int]:
        """# Get Indices of Samples Unscored for Metric.

        ## Args:
            * metric    (str):  Metric scores being inspected.

        ## Returns:
            * List[int]:    Sample indices that still need computing for this metric.
        """
        # If metric column doesn't exist, all rows are unscored.
        if metric not in self._scores_.columns: return set(self._scores_["index"].tolist())

        # Otherwise, return indices where the value is NaN or empty.
        return set(self._scores_.loc[self._scores_[metric].isna(), "index"].tolist())
    
    def record_row(self,
        index:  int,
        label:  str,
        scores: Dict[str, float]
    ) -> None:
        """# Record Row of Scores for Sample.

        ## Args:
            * index     (int):              Index of samples within dataset.
            * label     (str):              Ground truth label/class for sample.
            * scores    (Dict[str, float]): Map of sample's metric scores.
        """
        # Construct mask.
        mask:   Series =    self._scores_["index"] == index

        # Assign class.
        self._scores_.loc[mask, "class"] = label

        # Record metric scores.
        for m, s in scores.items(): self._scores_.loc[mask, m] = s
    
    def record_score(self,
        index:  int,
        metric: str,
        score:  Union[int, float]
    ) -> None:
        """# Record Metric for Sample.

        ## Args:
            * index     (int):          Index of sample within dataset.
            * metric    (str):          Identifier of sample's metric being record.
            * score     (int | float):  Value of sample's metric being recorded.
        """
        # Record metric.
        self._scores_.loc[self._scores_["index"] == index, metric] = score

    def save(self) -> Path:
        """# Save Metrics to File.

        ## Returns:
            * Path: Path at which metrics were saved.
        """
        # Save metrics to file.
        self._scores_.to_csv(self.scores_path, index = False)

        # Indicate path at which metrics were saved.
        return self.scores_path.absolute()

    # HELPERS ======================================================================================

    def _load_(self) -> None:
        """# Load Metrics."""
        # If path already exists, load scores from file.
        if self.scores_path.exists(): self._scores_ = read_csv(self.scores_path); return

        # Otherwise, ensure that the leading directories at least exist.
        self._root_.mkdir(parents = True, exist_ok = True)

        # Import metrics registry.
        from gradus.registration    import METRIC_REGISTRY

        # Create scores data frame.
        self._scores_ = DataFrame(
                            {
                                "index": range(self._num_samples_),
                                "class": None,
                                **{metric: float("nan") for metric in METRIC_REGISTRY.list_entries()}
                            }
                        )