"""# gradus.networks.vgg.vgg_11

Argument definitions & parsing for 11-layer VGG network.
"""

__all__ = ["VGG11Config"]

from gradus.configuration   import NetworkConfig

class VGG11Config(NetworkConfig):
    """# VGG-11 Network Configuration"""

    def __init__(self):
        """# Instantiate VGG-11 Network Configuration."""
        super(VGG11Config, self).__init__(
            name =  "vgg-11",
            help =  """VGG neural network with 11 learnable layers."""
        )
