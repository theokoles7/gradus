"""# gradus.networks.resnet.resnet_18

Argument definitions & parsing for 18-layer residual network.
"""

__all__ = ["ResNet18Config"]

from gradus.configuration   import NetworkConfig

class ResNet18Config(NetworkConfig):
    """# ResNet-18 Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-18 Network Configuration."""
        super(ResNet18Config, self).__init__(
            name =  "resnet-18",
            help =  """Residual neural network with 18 learnable layers."""
        )