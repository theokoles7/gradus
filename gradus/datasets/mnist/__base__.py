"""# gradus.datasets.mnist.base

MNIST dataset implementation.
"""

__all__ = ["MNIST"]

from typing                         import List

from torch.utils.data               import DataLoader
from torchvision.datasets           import MNIST as t_MNIST
from torchvision.transforms         import Compose, Normalize, ToTensor

from gradus.datasets.mnist.__args__ import MNISTConfig
from gradus.datasets.protocol       import Dataset
from gradus.registration            import register_dataset
from gradus.utilities               import get_system_core_count

@register_dataset(
    id =        "mnist",
    config =    MNISTConfig,
    tags =      ["grayscale"]
)
class MNIST(Dataset):
    """# MNIST Dataset
    
    70,000 28x28 grayscale images in 10 classes (60k train, 10k test).

    Reference: http://yann.lecun.com/exdb/mnist/
    """

    def __init__(self,
        root:           str =   "data",
        batch_size:     int =   64,
        max_workers:    int =   get_system_core_count(),
        **kwargs
    ):
        """# Intantiate MNIST Dataset."""
        # Define transform.
        self._transform_:       Compose =       Compose([
                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.5,),
                                                        std =   (0.5,)
                                                    )
                                                ])
        
        # Load training data.
        self._train_data_:      t_MNIST =       t_MNIST(
                                                    root =      root,
                                                    train =     True,
                                                    download =  True,
                                                    transform = self._transform_
                                                )
        
        # Load test data.
        self._test_data_:       t_MNIST =       t_MNIST(
                                                    root =      root,
                                                    train =     False,
                                                    download =  True,
                                                    transform = self._transform_
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
        self._channels_:        int =           1
        self._classes_:         List[str] =     self._train_data_.classes
        self._height_:          int =           28
        self._num_classes_:     int =           len(self._classes_)
        self._size_:            int =           len(self._train_data_) + len(self._test_data_)
        self._width_:           int =           28
        
        # Initialize dataset.
        super(MNIST, self).__init__(id = "mnist")