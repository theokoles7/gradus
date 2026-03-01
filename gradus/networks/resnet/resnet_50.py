"""# gradus.networks.resnet.resnet_50

ResNet-50 neural network implementation.
"""

__all__ = ["ResNet50"]

from typing                             import Tuple

from gradus.configuration               import NetworkConfig
from gradus.networks.resnet.__base__    import ResNet
from gradus.networks.resnet.blocks      import ResNetBlock
from gradus.registration                import register_network

@register_network(
    id =        "resnet-50",
    config =    NetworkConfig(
                    name = "resnet-50",
                    help = "Residual neural network with 50 learnable layers."
                ),
    tags =      ["residual", "cnn", "50-layers"]
)
class ResNet50(ResNet):
    """# 50-Layer Residual Neural Network"""

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
        super(ResNet50, self).__init__(
            block =                 ResNetBlock,
            layers =                [3, 4, 6, 3],
            input_shape =           input_shape,
            num_classes =           num_classes,
            zero_init_residual =    zero_init_residual
        )