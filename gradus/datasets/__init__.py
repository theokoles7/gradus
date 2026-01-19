"""# gradus.datasets

Image classification dataset wrappers/implementations.
"""

__all__ =   [
                # Concrete Datasets
                "CIFAR_10",
                "CIFAR_100",
                "MNIST",

                # Protocol
                "GradusDataset",
            ]

from gradus.datasets.cifar_10.__base__  import CIFAR_10
from gradus.datasets.cifar_100.__base__ import CIFAR_100
from gradus.datasets.mnist.__base__     import MNIST
from gradus.datasets.protocol           import GradusDataset
