"""# gradus.networks.resnet.resnet_152

Argument definitions & parsing for 152-layer residual network.
"""

__all__ = ["ResNet152Config"]

from gradus.configuration   import NetworkConfig

class ResNet152Config(NetworkConfig):
    """# ResNet-152 Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-152 Network Configuration."""
        super(ResNet152Config, self).__init__(
            name =  "resnet-152",
            help =  """Residual neural network with 152 learnable layers."""
        )