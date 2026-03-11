"""# gradus.datasets.protocol

Abstract dataset protocol.
"""

__all__ = ["Dataset"]

from abc                import ABC
from logging            import Logger
from typing             import List, Tuple

from torch.utils.data   import DataLoader, Dataset as t_Dataset

from gradus.utilities   import get_logger, get_system_core_count

class Dataset(ABC):
    """# Gradus Dataset Wrapper & Protocol"""

    # Declare properties.
    _channels_:     int
    _classes_:      List[str]
    _height_:       int
    _num_classes_:  int
    _size_:         int
    _test_data_:    t_Dataset
    _test_loader_:  DataLoader
    _train_data_:   t_Dataset
    _train_loader_: DataLoader
    _width_:        int

    def __init__(self,
        id: str
    ):
        """# Instantiate Dataset.

        ## Args:
            * id    (str):  Dataset identifier/name.
        """
        # Initialize logger.
        self.__logger__:    Logger =    get_logger(f"{id}-dataset")

        # Define properties.
        self._id_:  str =   id

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def channels(self) -> int:
        """# Input Channels"""
        return self._channels_
    
    @property
    def classes(self) -> List[str]:
        """# Classification Classes"""
        return self._classes_
    
    @property
    def height(self) -> int:
        """# Input Height"""
        return self._height_

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """# Expected Input Shape (C, H, W)"""
        return self._channels_, self._height_, self._width_
    
    @property
    def num_classes(self) -> int:
        """# Number of Classes"""
        return self._num_classes_
    
    @property
    def size(self) -> int:
        """# Sample Quantity"""
        return self._size_
    
    @property
    def test_data(self) -> t_Dataset:
        """# Test Split Data"""
        return self._test_data_
    
    @property
    def test_loader(self) -> DataLoader:
        """# Test Split Loader"""
        return self._test_loader_
    
    @property
    def train_data(self) -> t_Dataset:
        """# Train Split Data"""
        return self._train_data_
    
    @property
    def train_loader(self) -> DataLoader:
        """# Train Split Loader"""
        return self._train_loader_
    
    @property
    def width(self) -> int:
        """# Input Width"""
        return self._width_
    
    # HELPERS ======================================================================================

    def _resolve_dataloader_(self,
        data:               t_Dataset,
        batch_size:         int =       64,
        max_workers:        int =       get_system_core_count(),
        metric:             str =       None,
        order:              str =       "ascending",
        scope:              str =       "holistic",
        normalize_classes:  bool =      False
    ) -> DataLoader:
        """# Instantiate DataLoader.

        ## Args:
            * data              (Dataset):  Dataset being loaded.
            * batch_size        (int):      Number of samples to load into batches. Defaults to 64.
            * max_workers       (int):      Maximum number of workers allocated to data 
                                            preprocessing. Defaults to max system core count.
            * metric            (str):      Metric by which dataset samples will be ranked.
            * order             (str):      Order by which dataset samples will be sorted, based on 
                                            rank. Defaults to "ascending".
            * scope             (str):      Scope of sorting (i.e., "holistic", "batch-wise"). 
                                            Defaults to "holistic".
            * normalize_classes (bool):     Distribute classes across batches as equally as 
                                            possible.

        ## Returns:
            * DataLoader:   Instantiated dataloader.
        """
        from gradus.registration    import METRIC_REGISTRY

        # If a metric is not specified...
        if metric is None:

            # Provide a generic dataloader, randomly shuffled.
            return  DataLoader(
                        dataset =       data,
                        batch_size =    batch_size,
                        num_workers =   max_workers,
                        pin_memory =    True,
                        shuffle =       True,
                        drop_last =     False
                    )
        
        # TODO: Otherwise...
    
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Dataset Object Representation"""
        return f"""<{self._id_.upper()}Dataset({self._num_classes_} classes, {self._size_} samples)>"""