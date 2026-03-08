"""# gradus.networks.resnet.resnet_34

ResNet-34 neural network implementation.
"""

__all__ = ["ResNet34"]

from typing                             import Tuple

from gradus.configuration               import NetworkConfig
from gradus.networks.resnet.__base__    import ResNet
from gradus.networks.resnet.blocks      import ResNetBlock
from gradus.registration                import register_network

@register_network(
    id =        "resnet-34",
    config =    NetworkConfig(
                    name = "resnet-34",
                    help = "Residual neural network with 34 learnable layers."
                ),
    tags =      ["residual", "cnn", "34-layers"]
)
class ResNet34(ResNet):
    """# 34-Layer Residual Neural Network"""

    def __init__(self,
        input_shape:        Tuple[int, ...],
        num_classes:        int,
        zero_init_residual: bool =              False
    ):
        """# Instantiate Residual Neural Network.

        ## Args:
            * input_shape           (Tuple[int]):   Shape of input samples.
            * num_classes           (int):          Number of classes (output logits).
            * zero_init_residual    (bool):         Initialize residual blocks with zero weights. 
                                                    Defaults to False.
        """
        # Initialize network.
        super(ResNet34, self).__init__(
            block =                 ResNetBlock,
            layers =                [3, 4, 6, 3],
            input_shape =           input_shape,
            num_classes =           num_classes,
            zero_init_residual =    zero_init_residual
        )