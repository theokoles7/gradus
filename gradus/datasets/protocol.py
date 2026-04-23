"""# gradus.datasets.protocol

Abstract dataset protocol.
"""

__all__ = ["Dataset"]

from abc                import ABC
from functools          import cached_property
from logging            import Logger
from typing             import Any, Dict, List, Optional, Tuple, Union

from torch.utils.data   import DataLoader, Dataset as t_Dataset

from gradus.artifacts   import DatasetMetrics
from gradus.curricula   import Curriculum, Schedule
from gradus.utilities   import get_logger, get_system_core_count

class Dataset(ABC):
    """# Gradus Dataset Wrapper & Protocol"""

    def __init__(self,
        id:                 str,
        train_data:         t_Dataset,
        test_data:          t_Dataset,
        epochs:             int,
        batch_size:         int =                               128,
        shuffle:            bool =                              False,
        max_workers:        int =                               get_system_core_count(),
        metric:             Optional[Union[str, List[str]]] =   None,
        rank:               str =                               "ascending",
        scope:              str =                               "holistic",
        schedule:           Optional[str] =                     None,
        start_fraction:     float =                             0.3,
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
        self.__logger__:            Logger =        get_logger(f"{id}-dataset")

        # Define properties.
        self._id_:                  str =           id
        self._train_data_:          t_Dataset =     train_data
        self._test_data_:           t_Dataset =     test_data
        self._batch_size_:          int =           int(batch_size)
        self._max_workers_:         int =           max_workers
        self._shuffle_:             bool =          shuffle
        self._normalize_classes_:   bool =          normalize_classes
        self._seed_:                int =           seed
        self._metric_:              Optional[str] = metric
        self._rank_:                str =           rank
        self._scope_:               str =           scope
        self._schedule_id_:         Optional[str] = schedule
        self._start_fraction_:      float =         start_fraction
        self._total_epochs_:        int =           epochs

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
    def curriculum(self) -> Optional[Curriculum]:
        """# Dataset Curriculum"""
        # If no curriculum is being used, return None.
        if self._metric_ is None: return None

        # Otherwise, cache curriculum as attribute.
        self._curriculum_:  Curriculum =    Curriculum(
                                                dataset_id =    self._id_,
                                                scores =        DatasetMetrics(
                                                                    dataset_id =    self._id_,
                                                                    num_samples =   len(self._train_data_),
                                                                    seed =          self._seed_,
                                                                    scores_path =   ".cache/scores"
                                                                ),
                                                metric =        self._metric_,
                                                rank =          self._rank_,
                                                scope =         self._scope_,
                                                batch_size =    self._batch_size_,
                                                seed =          self._seed_
                                            )
        
        # Provide curriculum.
        return self._curriculum_
    
    @property
    def dict(self) -> Dict[str, Any]:
        """# Dataset Dictionary Representation"""
        return  {
                    "id":                   self._id_,
                    "shuffled":             self._shuffle_,
                    "normalize_classes":    self._normalize_classes_,
                    "curriculum":           None if self.curriculum is None \
                                            else self.curriculum.dict,
                    "schedule_id":          self._schedule_id_,
                    "start_fraction":       self._start_fraction_
                }
    
    @cached_property
    def height(self) -> int:
        """# Input Height"""
        return self.input_shape[1]
    
    @property
    def id(self) -> str:
        """# Dataset Identifier"""
        return self._id_

    @cached_property
    def input_shape(self) -> Tuple[int, int, int]:
        """# Expected Input Shape (C, H, W)"""
        return tuple(self._train_data_[0][0].shape)
    
    @cached_property
    def num_classes(self) -> int:
        """# Number of Classes"""
        return len(self.classes)
    
    @cached_property
    def schedule(self) -> Optional[Schedule]:
        """# Curriculum Pacing Schedule"""
        # If no curriculum/schedule is defined, no curriculum.
        if self._schedule_id_ is None or self._metric_ is None: return None

        # Otherwise, import schedule registry.
        from gradus.registration    import SCHEDULE_REGISTRY

        # Load schedule.
        return  SCHEDULE_REGISTRY.load_schedule(
                    schedule_id =       self._schedule_id_,
                    total_samples =     len(self._train_data_),
                    total_epochs =      self._total_epochs_,
                    start_fraction =    self._start_fraction_,
                    batch_size =        self._batch_size_
                )
    
    @property
    def shuffled(self) -> bool:
        """# Training Data is Shuffled?"""
        return self._shuffle_
    
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
        
        # Otherwise, initialize data loader with curriculum.
        return  DataLoader(
                    dataset =       self._train_data_,
                    batch_sampler = self.curriculum,
                    num_workers =   self._max_workers_,
                    pin_memory =    True
                )
    
    @cached_property
    def width(self) -> int:
        """# Input Width"""
        return self.input_shape[2]
    
    # METHODS ======================================================================================
 
    def step(self,
        epoch:      int,
        **metrics:  Any
    ) -> None:
        # If no curriculum or no schedule, no-op.
        if self.curriculum is None or self._schedule_id_ is None: return

        # Step schedule — returns ordered list of batch indices.
        order: List[int] = self.schedule.step(epoch = epoch, **metrics)

        # Apply ordering to curriculum.
        self.curriculum.set_order(order)
    
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Dataset Object Representation"""
        return  (
                    f"""<{self._id_.upper()}Dataset({self.num_classes} classes, """
                    f"""{self.size} samples, batch-size = {self._batch_size_})>"""
                )