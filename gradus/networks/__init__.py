"""# gradus.networks

Neural network architecture implementations.
"""

__all__ =   [
                # Generic
                "Autoencoder",
                "CNN",

                # ResNet
                "ResNet18",
                "ResNet34",
                "ResNet50",
                "ResNet101",
                "ResNet152",

                # Blocks
                "ResNetBlock",
                "ResNetBottleneck",
            ]

from gradus.networks.autoencoder    import Autoencoder
from gradus.networks.cnn            import CNN
from gradus.networks.resnet         import *