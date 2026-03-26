"""# gradus.datasets.imagenet.__base__

ImageNet (ILSVRC 2012) dataset implementation.
"""

__all__ = ["ImageNet"]

from typing                             import List

from torch.utils.data                   import DataLoader
from torchvision.datasets               import ImageNet as tv_ImageNet
from torchvision.transforms             import CenterCrop, Compose, Normalize, Resize, ToTensor

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
        root:           str =   ".cache/data",
        batch_size:     int =   64,
        shuffle:        bool =  False,
        max_workers:    int =   get_system_core_count(),
        **kwargs
    ):
        """# Instantiate ImageNet Dataset.

        ## Args:
            * root          (str):  Path to directory containing the downloaded archive files.
                                    Defaults to ".cache/data".
            * batch_size    (int):  Samples per batch. Defaults to 64.
            * shuffle       (bool): Shuffle training split. Defaults to False.
            * max_workers   (int):  DataLoader worker threads. Defaults to system core count.
        """
        # Define standard ImageNet transform.
        self._transform_:       Compose =       Compose([
                                                    # Resize images to 256x256.
                                                    Resize(size = 256),

                                                    # Crop to 224x224.
                                                    CenterCrop(size = 224),

                                                    # Convert images to tensors.
                                                    ToTensor(),

                                                    # Normalize pixel values.
                                                    Normalize(
                                                        mean =  (0.485, 0.456, 0.406),
                                                        std =   (0.229, 0.224, 0.225)
                                                    )
                                                ])

        # Load training data.
        self._train_data_:      tv_ImageNet =   tv_ImageNet(
                                                    root =      root,
                                                    split =     "train",
                                                    transform = self._transform_
                                                )

        # Load test data.
        self._test_data_:       tv_ImageNet =   tv_ImageNet(
                                                    root =      root,
                                                    split =     "val",
                                                    transform = self._transform_
                                                )

        # Initialize train loader.
        self._train_loader_:    DataLoader =    DataLoader(
                                                    data =          self._train_data_,
                                                    batch_size =    batch_size,
                                                    shuffle =       shuffle,
                                                    max_workers =   max_workers,
                                                    drop_last =     True
                                                )

        # Initialize test loader.
        self._test_loader_:     DataLoader =    DataLoader(
                                                    data =          self._test_data_,
                                                    batch_size =    batch_size,
                                                    shuffle =       False,
                                                    max_workers =   max_workers,
                                                    drop_last =     False
                                                )

        # Define properties.
        self._channels_:        int =           3
        self._height_:          int =           224
        self._width_:           int =           224
        self._classes_:         List[str] =     [c[0] for c in self._train_data_.classes]
        self._num_classes_:     int =           len(self._classes_)
        self._size_:            int =           len(self._train_data_) + len(self._test_data_)

        # Initialize protocol.
        super(ImageNet, self).__init__(id = "imagenet")