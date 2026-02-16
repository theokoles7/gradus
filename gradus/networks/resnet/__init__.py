"""# gradus.networks.resnet

ResNet (Residual Neural Network) architectures.
"""

__all__ =   [
                # Networks
                "ResNet18",
                "ResNet34",
                "ResNet50",
                "ResNet101",
                "ResNet152",

                # Blocks
                "ResNetBlock",
                "ResNetBottleneck",
            ]

from gradus.networks.resnet.blocks  import *

from gradus.networks.resnet.resnet_18   import ResNet18
from gradus.networks.resnet.resnet_34   import ResNet34
from gradus.networks.resnet.resnet_50   import ResNet50
from gradus.networks.resnet.resnet_101  import ResNet101
from gradus.networks.resnet.resnet_152  import ResNet152