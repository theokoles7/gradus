"""# gradus.networks

Neural network architecture implementations.
"""

__all__ =   [
                # Protocol
                "Network",

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

                # VGG
                "VGG11",
                "VGG16",
                "VGG19",
            ]

from gradus.networks.autoencoder    import Autoencoder
from gradus.networks.cnn            import CNN
from gradus.networks.protocol       import Network
from gradus.networks.resnet         import *
from gradus.networks.vgg            import *