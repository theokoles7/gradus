"""# gradus.networks.resnet_34.args

Argument definitions & parsing for ResNet-34 neural network.
"""

__all__ = ["ResNet34Config"]

from gradus.configuration   import NetworkConfig

class ResNet34Config(NetworkConfig):
    """# ResNet-34 Neural Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-34 Neural Network Configuration."""
        # Initialize configuration.
        super(ResNet34Config, self).__init__(
            name =  "resnet-34",
            help =  """Residual Network with 34 layers."""
        )