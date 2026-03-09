"""# gradus.networks.resnet.resnet_101

Argument definitions & parsing for 101-layer residual network.
"""

__all__ = ["ResNet101Config"]

from gradus.configuration   import NetworkConfig

class ResNet101Config(NetworkConfig):
    """# ResNet-101 Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-101 Network Configuration."""
        super(ResNet101Config, self).__init__(
            name =  "resnet-101",
            help =  """Residual neural network with 101 learnable layers."""
        )