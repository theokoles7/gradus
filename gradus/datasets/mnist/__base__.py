"""# gradus.datasets.mnist.base

MNIST dataset implementation.
"""

__all__ = ["MNIST"]

from typing                         import List, Optional, Union

from torchvision.datasets           import MNIST as t_MNIST
from torchvision.transforms         import Compose, Normalize, RandomCrop, ToTensor

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
        root:               str =                               ".cache/data",
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
        **kwargs
    ):
        """# Instantiate MNIST Dataset.

        ## Args:
            * root              (str):                      Path to directory from/to which datasets 
                                                            should be loaded/downloaded. Defaults to 
                                                            "./.cache/data/".
            * batch_size        (int):                      Number of samples to load into batches. 
                                                            Defaults to 64.
            * shuffle           (bool):                     Shuffle training set.
            * max_workers       (int):                      Maximum number of workers allocated to 
                                                            data preprocessing. Defaults to max 
                                                            system core count.
            * metric            (str | List[str] | None):   Metric by which samples will be ranked 
                                                            (constitutes curriculum).
            * rank              (str):                      Order by which dataset samples will be 
                                                            sorted, based on rank. Defaults to 
                                                            "ascending".
            * scope             (str):                      Scope of sorting (i.e., "holistic", 
                                                            "batch-wise"). Defaults to "holistic".
            * normalize_classes (bool):                     Distribute classes across batches as 
                                                            equally as possible.
            * seed              (int):                      Random number generation seed.
        """
        # Initialize dataset.
        super(MNIST, self).__init__(
            id = "mnist",
            train_data =        t_MNIST(
                                    root =      root,
                                    train =     True,
                                    transform = Compose([
                                                    # Randomly crop with padding.
                                                    RandomCrop(size = 32, padding = 4),

                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.5,),
                                                        std =   (0.5,)
                                                    )
                                                ]),
                                    download =  True
                                ),
            test_data =         t_MNIST(
                                    root =      root,
                                    train =     False,
                                    transform = Compose([
                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.5,),
                                                        std =   (0.5,)
                                                    )
                                                ]),
                                    download =  True
                                ),
            batch_size =        batch_size,
            shuffle =           shuffle,
            max_workers =       max_workers,
            metric =            metric,
            rank =              rank,
            scope =             scope,
            schedule =          schedule,
            start_fraction =    start_fraction,
            normalize_classes = normalize_classes,
            seed =              seed
        )