"""# gradus.networks.vgg.vgg_13

Argument definitions & parsing for 13-layer VGG network.
"""

__all__ = ["VGG13Config"]

from gradus.configuration   import NetworkConfig

class VGG13Config(NetworkConfig):
    """# VGG-13 Network Configuration"""

    def __init__(self):
        """# Instantiate VGG-13 Network Configuration."""
        super(VGG13Config, self).__init__(
            name =  "vgg-13",
            help =  """VGG neural network with 13 learnable layers."""
        )
