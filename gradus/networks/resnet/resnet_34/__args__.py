"""# gradus.networks.resnet.resnet_34.args

Argument definitions & parsing for 34-layer residual network.
"""

__all__ = ["ResNet34Config"]

from gradus.configuration   import NetworkConfig

class ResNet34Config(NetworkConfig):
    """# ResNet-34 Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-34 Network Configuration."""
        super(ResNet34Config, self).__init__(
            name =  "resnet-34",
            help =  """Residual neural network with 34 learnable layers."""
        )