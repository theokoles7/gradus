"""# gradus.datasets.cifar_100.base

CIFAR-100 dataset implementation.
"""

__all__ = ["CIFAR_100"]

from typing                             import List, Optional, Union

from torchvision.datasets               import CIFAR100
from torchvision.transforms             import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from gradus.datasets.cifar_100.__args__ import CIFAR100Config
from gradus.datasets.protocol           import Dataset
from gradus.registration                import register_dataset
from gradus.utilities                   import get_system_core_count

@register_dataset(
    id =        "cifar-100",
    config =    CIFAR100Config,
    tags =      ["rgb"]
)
class CIFAR_100(Dataset):
    """# CIFAR-100 Dataset
    
    60,000 32x32 RGB images in 100 classes (50k train, 10k test).

    Reference: https://www.cs.toronto.edu/~kriz/cifar.html
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
        """# Instantiate CIFAR-100 Dataset.

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
        super(CIFAR_100, self).__init__(
            id = "cifar-100",
            train_data =        CIFAR100(
                                    root =      root,
                                    train =     True,
                                    transform = Compose([
                                                    # Randomly crop with padding.
                                                    RandomCrop(size = 32, padding = 4),

                                                    # Randomly flip horizontally.
                                                    RandomHorizontalFlip(),

                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.5071, 0.4865, 0.4409),
                                                        std =   (0.2673, 0.2564, 0.2761)
                                                    )
                                                ]),
                                    download =  True
                                ),
            test_data =         CIFAR100(
                                    root =      root,
                                    train =     False,
                                    transform = Compose([
                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.5071, 0.4865, 0.4409),
                                                        std =   (0.2673, 0.2564, 0.2761)
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