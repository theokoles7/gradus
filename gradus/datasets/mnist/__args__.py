"""# gradus.datasets.mnist.args

Argument definitions & parsing for MNIST dataset.
"""

from gradus.configuration   import DatasetConfig

class MNISTConfig(DatasetConfig):
    """# MNIST Dataset Configuration"""

    def __init__(self):
        """# Instantiate MNIST Dataset Configuration."""
        # Initialize configuration.
        super(MNISTConfig, self).__init__(
            name =  "mnist",
            help =  """70,000 28x28 grayscale images in 10 classes (60k train, 10k test)."""
        )