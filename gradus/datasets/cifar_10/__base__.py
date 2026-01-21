"""# gradus.datasets.cifar_10.base

CIFAR-10 dataset implementation.
"""

__all__ = ["CIFAR_10"]

from typing                             import List

from torch.utils.data                   import DataLoader
from torchvision.datasets               import CIFAR10
from torchvision.transforms             import Compose, Normalize, Resize, ToTensor

from gradus.datasets.cifar_10.__args__  import CIFAR10Config
from gradus.datasets.protocol           import Dataset
from gradus.registration                import register_dataset
from gradus.utilities                   import get_system_core_count

@register_dataset(
    id =        "cifar-10",
    config =    CIFAR10Config,
    tags =      ["rgb"]
)
class CIFAR_10(Dataset):
    """# CIFAR-10 Dataset
    
    60,000 32x32 RGB images in 10 classes (50k train, 10k test).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self,
        root:           str =   "data",
        batch_size:     int =   64,
        max_workers:    int =   get_system_core_count(),
        **kwargs
    ):
        """# Intantiate CIFAR-10 Dataset.

        ## Args:
            * root          (str):  Path to directory from/to which datasets should be 
                                    loaded/downloaded. Defaults to "./data/".
            * batch_size    (int):  Number of samples to load into batches. Defaults to 64.
            * max_workers   (int):  Maximum number of workers allocated to data preprocessing. 
                                    Defaults to max system core count.
        """
        # Initialize dataset.
        super(CIFAR_10, self).__init__(id = "cifar-10")
        
        # Define transform.
        self._transform_:       Compose =       Compose([
                                                    # Resize images to 32x32.
                                                    Resize(size = 32),

                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.5, 0.5, 0.5,),
                                                        std =   (0.5, 0.5, 0.5,)
                                                    )
                                                ])
        
        # Load training data.
        self._train_data_:      CIFAR10 =       CIFAR10(
                                                    root =      root,
                                                    train =     True,
                                                    transform = self._transform_,
                                                    download =  True
                                                )
        
        # Load test data.
        self._test_data_:       CIFAR10 =       CIFAR10(
                                                    root =      root,
                                                    train =     False,
                                                    transform = self._transform_,
                                                    download =  True
                                                )
        
        # Initialize train loader.
        self._train_loader_:    DataLoader =    DataLoader(
                                                    dataset =       self._train_data_,
                                                    batch_size =    batch_size,
                                                    num_workers =   max_workers,
                                                    pin_memory =    True,
                                                    shuffle =       True,
                                                    drop_last =     True
                                                )
        
        # Initialize test loader.
        self._test_loader_:     DataLoader =    DataLoader(
                                                    dataset =       self._test_data_,
                                                    batch_size =    batch_size,
                                                    num_workers =   max_workers,
                                                    pin_memory =    True,
                                                    shuffle =       True,
                                                    drop_last =     False
                                                )
        
        # Define properties.
        self._channels_:        int =           3
        self._classes_:         List[str] =     self._train_data_.classes
        self._height_:          int =           32
        self._num_classes_:     int =           len(self._classes_)
        self._size_:            int =           len(self._train_data_) + len(self._test_data_)
        self._width_:           int =           32

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")