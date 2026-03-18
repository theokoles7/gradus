"""# gradus.networks.vgg.vgg_19

Argument definitions & parsing for 19-layer VGG network.
"""

__all__ = ["VGG19Config"]

from gradus.configuration   import NetworkConfig

class VGG19Config(NetworkConfig):
    """# VGG-19 Network Configuration"""

    def __init__(self):
        """# Instantiate VGG-19 Network Configuration."""
        super(VGG19Config, self).__init__(
            name =  "vgg-19",
            help =  """VGG neural network with 19 learnable layers."""
        )
