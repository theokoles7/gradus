"""# gradus.networks.resnet.resnet_152

ResNet-152 neural network implementation.
"""

__all__ = ["ResNet152"]

from typing                             import Tuple

from gradus.configuration               import NetworkConfig
from gradus.networks.resnet.__base__    import ResNet
from gradus.networks.resnet.blocks      import ResNetBlock
from gradus.registration                import register_network

@register_network(
    id =        "resnet-152",
    config =    NetworkConfig(
                    name = "resnet-152",
                    help = "Residual neural network with 152 learnable layers."
                ),
    tags =      ["residual", "cnn", "152-layers"]
)
class ResNet152(ResNet):
    """# 152-Layer Residual Neural Network"""

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
        super(ResNet152, self).__init__(
            block =                 ResNetBlock,
            layers =                [3, 8, 36, 3],
            input_shape =           input_shape,
            num_classes =           num_classes,
            zero_init_residual =    zero_init_residual
        )