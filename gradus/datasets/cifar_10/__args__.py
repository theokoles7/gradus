"""# gradus.datasets.cifar_10.args

Argument definitions & parsing for CIFAR-10 dataset.
"""

from gradus.configuration   import DatasetConfig

class CIFAR10Config(DatasetConfig):
    """# CIFAR-10 Dataset Configuration"""

    def __init__(self):
        """# Instantiate CIFAR-10 Dataset Configuration."""
        # Initialize configuration.
        super(CIFAR10Config, self).__init__(
            name =  "cifar-10",
            help =  """60,000 32x32 RGB images in 10 classes (50k train, 10k test)."""
        )