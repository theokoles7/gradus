"""# gradus.networks.resnet.resnet_50

Argument definitions & parsing for 50-layer residual network.
"""

__all__ = ["ResNet50Config"]

from gradus.configuration   import NetworkConfig

class ResNet50Config(NetworkConfig):
    """# ResNet-50 Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-50 Network Configuration."""
        super(ResNet50Config, self).__init__(
            name =  "resnet-50",
            help =  """Residual neural network with 50 learnable layers."""
        )