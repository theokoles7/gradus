"""# gradus.datasets.protocol

Abstract dataset protocol.
"""

__all__ = ["Dataset"]

from abc                import ABC
from functools          import cached_property
from logging            import Logger
from typing             import List, Optional, Tuple, Union

from torch.utils.data   import DataLoader, Dataset as t_Dataset

from gradus.utilities   import get_logger, get_system_core_count

class Dataset(ABC):
    """# Gradus Dataset Wrapper & Protocol"""

    def __init__(self,
        id:                 str,
        train_data:         t_Dataset,
        test_data:          t_Dataset,
        batch_size:         int =                               128,
        shuffle:            bool =                              False,
        max_workers:        int =                               get_system_core_count(),
        metric:             Optional[Union[str, List[str]]] =   None,
        rank:               str =                               "ascending",
        scope:              str =                               "holistic",
        normalize_classes:  bool =                              False,
        seed:               int =                               1,
    ):
        """# Instantiate Dataset.

        ## Args:
            * id            (str):                      Dataset identifier/name.
            * train_data    (Dataset):                  Dataset's training split.
            * test_data     (Dataset):                  Dataset's test/validation split.
            * batch_size    (int):                      Number of sample to include in training 
                                                        batches.
            * shuffle       (bool):                     Shuffle training samples.
            * max_workers   (int):                      Maximum number of workers allocated to data 
                                                        preprocessing. Defaults to max system core 
                                                        count.
            * metric        (str | List[str] | None):   Metric by which samples will be ranked 
                                                        (constitutes curriculum).
            * rank          (str):                      Order by which dataset samples will be 
                                                        sorted, based on rank. Defaults to 
                                                        "ascending".
            * scope         (str):                      Scope of sorting (i.e., "holistic", 
                                                        "batch-wise"). Defaults to "holistic".
            * seed          (int):                      Random number generation seed.
        """
        # Initialize logger.
        self.__logger__:        Logger =        get_logger(f"{id}-dataset")

        # Define properties.
        self._id_:                  str =           id
        self._train_data_:          t_Dataset =     train_data
        self._test_data_:           t_Dataset =     test_data
        self._batch_size_:          int =           int(batch_size)
        self._max_workers_:         int =           max_workers
        self._shuffle_:             bool =          shuffle
        self._normalize_classes_:   bool =          normalize_classes
        self._seed_:                int =           seed

        # Define curriculum.
        self._metric_:              Optional[str] = metric
        self._rank_:                str =           rank
        self._scope_:               str =           scope

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @cached_property
    def channels(self) -> int:
        """# Input Channels"""
        return self.input_shape[0]
    
    @cached_property
    def classes(self) -> List[str]:
        """# Classification Classes"""
        return self._train_data_.classes
    
    @cached_property
    def height(self) -> int:
        """# Input Height"""
        return self.input_shape[1]

    @cached_property
    def input_shape(self) -> Tuple[int, int, int]:
        """# Expected Input Shape (C, H, W)"""
        return tuple(self._train_data_[0][0].shape)
    
    @cached_property
    def num_classes(self) -> int:
        """# Number of Classes"""
        return len(self.classes)
    
    @cached_property
    def size(self) -> int:
        """# Sample Quantity"""
        return len(self._train_data_) + len(self._test_data_)
    
    @property
    def test_data(self) -> t_Dataset:
        """# Test Split Data"""
        return self._test_data_
    
    @cached_property
    def test_loader(self) -> DataLoader:
        """# Test Split Loader"""
        return  DataLoader(
                    dataset =       self._test_data_,
                    batch_size =    self._batch_size_,
                    num_workers =   self._max_workers_,
                    pin_memory =    True,
                    shuffle =       False,
                    drop_last =     False
                )
    
    @property
    def train_data(self) -> t_Dataset:
        """# Train Split Data"""
        return self._train_data_
    
    @cached_property
    def train_loader(self) -> DataLoader:
        """# Train Split Loader"""
        # If a metric is not specified...
        if self._metric_ is None:

            # Provide a generic dataloader, randomly shuffled.
            return  DataLoader(
                        dataset =       self._train_data_,
                        batch_size =    self._batch_size_,
                        num_workers =   self._max_workers_,
                        shuffle =       self._shuffle_,
                        pin_memory =    True,
                        drop_last =     False
                    )
        
        # Otherwise, load curriculum imports.
        from gradus.artifacts   import DatasetMetrics
        from gradus.curricula   import Curriculum

        # Initialize dataset metric scores.
        scores: DatasetMetrics =    DatasetMetrics(
                                        dataset_id =    self._id_,
                                        num_samples =   len(self._train_data_),
                                        seed =          self._seed_,
                                        scores_path =   ".cache/scores"
                                    )
        
        # Initialize data loader with curriculum.
        return  DataLoader(
            dataset =       self._train_data_,
            batch_sampler = Curriculum(
                                dataset_id =    self._id_,
                                scores =        scores,
                                metric =        self._metric_,
                                rank =          self._rank_,
                                scope =         self._scope_,
                                batch_size =    self._batch_size_,
                                seed =          self._seed_ 
                            ),
            num_workers =   self._max_workers_,
            pin_memory =    True
        )
    
    @cached_property
    def width(self) -> int:
        """# Input Width"""
        return self.input_shape[2]
    
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Dataset Object Representation"""
        return  (
                    f"""<{self._id_.upper()}Dataset({self.num_classes} classes, """
                    f"""{self.size} samples)>"""
                )