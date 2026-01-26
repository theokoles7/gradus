"""# gradus.datasets.protocol

Abstract dataset protocol.
"""

__all__ = ["Dataset"]

from abc                import ABC
from logging            import Logger
from typing             import List, Tuple

from torch.utils.data   import DataLoader, Dataset as t_Dataset

from gradus.utilities   import get_logger

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
    
    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Dataset Object Representation"""
        return f"""<{self._id_.upper()}Dataset({self._num_classes_} classes, {self._size_} samples)>"""