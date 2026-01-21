"""# gradus.datasets.cifar_100.args

Argument definitions & parsing for CIFAR-100 dataset.
"""

from gradus.configuration   import DatasetConfig

class CIFAR100Config(DatasetConfig):
    """# CIFAR-10 Dataset Configuration"""

    def __init__(self):
        """# Instantiate CIFAR-100 Dataset Configuration."""
        # Initialize configuration.
        super(CIFAR100Config, self).__init__(
            name =  "cifar-100",
            help =  """60,000 32x32 RGB images in 100 classes (50k train, 10k test)."""
        )