"""# gradus.datasets.imagenet.__base__

ImageNet (ILSVRC 2012) dataset implementation.
"""

__all__ = ["ImageNet"]

from functools                          import cached_property
from typing                             import List, Optional, override, Union

from torchvision.datasets               import ImageNet as tv_ImageNet
from torchvision.transforms             import Compose, Normalize, RandomCrop, RandomHorizontalFlip, \
                                               Resize, ToTensor

from gradus.datasets.imagenet.__args__  import ImageNetConfig
from gradus.datasets.protocol           import Dataset
from gradus.registration                import register_dataset
from gradus.utilities                   import get_system_core_count

@register_dataset(
    id =        "imagenet",
    config =    ImageNetConfig,
    tags =      ["rgb", "large-scale"]
)
class ImageNet(Dataset):
    """# ImageNet Dataset (ILSVRC 2012)

    1.28M training images and 50k validation images across 1,000 classes.

    Requires manual download from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.
    Place the following files in the root directory before first use:

        * ILSVRC2012_devkit_t12.tar.gz
        * ILSVRC2012_img_train.tar
        * ILSVRC2012_img_val.tar

    Torchvision will extract these automatically on first instantiation.

    Reference: https://www.image-net.org/
    """

    def __init__(self,
        root:               str =                               ".cache/data",
        batch_size:         int =                               128,
        shuffle:            bool =                              False,
        max_workers:        int =                               get_system_core_count(),
        metric:             Optional[Union[str, List[str]]] =   None,
        rank:               str =                               "ascending",
        scope:              str =                               "holistic",
        normalize_classes:  bool =                              False,
        seed:               int =                               1,
        **kwargs
    ):
        """# Instantiate ImageNet Dataset.

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
        super(ImageNet, self).__init__(
            id = "imagenet",
            train_data =        tv_ImageNet(
                                    root =      root,
                                    split =     "train",
                                    transform = Compose([
                                                    # Resize to 256x256.
                                                    Resize(size = 256),

                                                    # Randomly crop with padding.
                                                    RandomCrop(size = 224, padding = 4),

                                                    # Randomly flip horizontally.
                                                    RandomHorizontalFlip(),

                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.485, 0.456, 0.406),
                                                        std =   (0.229, 0.224, 0.225)
                                                    )
                                                ])
                                ),
            test_data =         tv_ImageNet(
                                    root =      root,
                                    split =     "val",
                                    transform = Compose([
                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.485, 0.456, 0.406),
                                                        std =   (0.229, 0.224, 0.225)
                                                    )
                                                ])
                                ),
            batch_size =        batch_size,
            shuffle =           shuffle,
            max_workers =       max_workers,
            metric =            metric,
            rank =              rank,
            scope =             scope,
            normalize_classes = normalize_classes,
            seed =              seed
        )

    # PROPERTIES ===================================================================================

    @override
    @cached_property
    def classes(self) -> List[str]:
        """# Classification Classes"""
        return [c[0] for c in self._train_data_.classes]