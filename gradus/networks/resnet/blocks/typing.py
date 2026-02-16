"""# gradus.networks.resnet.blocks.typing

Residual neural network type variables/constraints.
"""

__all__ = ["BlockType"]

from typing                                     import Type, Union

from gradus.networks.resnet.blocks.block        import ResNetBlock
from gradus.networks.resnet.blocks.bottleneck   import ResNetBottleneck

BlockType = Type[Union[ResNetBlock, ResNetBottleneck]]