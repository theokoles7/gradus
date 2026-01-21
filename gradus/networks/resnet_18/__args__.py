"""# gradus.networks.resnet_18.args

Argument definitions & parsing for ResNet-18 neural network.
"""

__all__ = ["ResNet18Config"]

from gradus.configuration   import NetworkConfig

class ResNet18Config(NetworkConfig):
    """# ResNet-18 Neural Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-18 Neural Network Configuration."""
        # Initialize configuration.
        super(ResNet18Config, self).__init__(
            name =  "resnet-18",
            help =  """Residual Network with 18 layers."""
        )