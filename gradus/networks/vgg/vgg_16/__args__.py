"""# gradus.networks.vgg.vgg_16

Argument definitions & parsing for 16-layer VGG network.
"""

__all__ = ["VGG16Config"]

from gradus.configuration   import NetworkConfig

class VGG16Config(NetworkConfig):
    """# VGG-16 Network Configuration"""

    def __init__(self):
        """# Instantiate VGG-16 Network Configuration."""
        super(VGG16Config, self).__init__(
            name =  "vgg-16",
            help =  """VGG neural network with 16 learnable layers."""
        )
