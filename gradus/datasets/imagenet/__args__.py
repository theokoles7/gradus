"""# gradus.datasets.imagenet.args

Argument definitions & parsing for ImageNet dataset.
"""

from gradus.configuration   import DatasetConfig

class ImageNetConfig(DatasetConfig):
    """# ImageNet Dataset Configuration"""

    def __init__(self):
        """# Instantiate ImageNet Dataset Configuration."""
        # Initialize configuration.
        super(ImageNetConfig, self).__init__(
            name =  "imagenet",
            help =  """ILSVRC 2012: 1.28M 224x224 RGB images in 1,000 classes (1.28M train, 50k test)."""
        )