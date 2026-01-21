"""# gradus.datasets

Image classification dataset wrappers/implementations.
"""

__all__ =   [
                # Concrete Datasets
                "CIFAR_10",
                "CIFAR_100",
                "MNIST",

                # Protocol
                "Dataset",
            ]

from gradus.datasets.cifar_10   import CIFAR_10
from gradus.datasets.cifar_100  import CIFAR_100
from gradus.datasets.mnist      import MNIST
from gradus.datasets.protocol   import Dataset