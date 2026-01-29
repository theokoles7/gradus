"""# gradus.networks.resnet_50.args

Argument definitions & parsing for ResNet-50 neural network.
"""

__all__ = ["ResNet50Config"]

from gradus.configuration   import NetworkConfig

class ResNet50Config(NetworkConfig):
    """# ResNet-50 Neural Network Configuration"""

    def __init__(self):
        """# Instantiate ResNet-50 Neural Network Configuration."""
        # Initialize configuration.
        super(ResNet50Config, self).__init__(
            name =  "resnet-50",
            help =  """Residual Network with 50 layers."""
        )