"""# gradus.networks.resnet.blocks

Residual neural network block architecture implementations.
"""

__all__ =   [
                # Blocks
                "ResNetBlock",
                "ResNetBottleneck",

                # Typing
                "BlockType",
            ]

from gradus.networks.resnet.blocks.block        import ResNetBlock
from gradus.networks.resnet.blocks.bottleneck   import ResNetBottleneck
from gradus.networks.resnet.blocks.typing       import BlockType