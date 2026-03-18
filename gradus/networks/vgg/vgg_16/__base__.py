"""# gradus.networks.vgg.vgg_16

VGG-16 neural network implementation.
"""

__all__ = ["VGG16"]

from typing                                     import Tuple

from gradus.networks.vgg.__base__              import VGG
from gradus.networks.vgg.vgg_16.__args__       import VGG16Config
from gradus.registration                        import register_network

@register_network(
    id =        "vgg-16",
    config =    VGG16Config,
    tags =      ["vgg", "cnn", "16-layers"]
)
class VGG16(VGG):
    """# 16-Layer VGG Neural Network"""

    def __init__(self,
        input_shape:    Tuple[int, ...],
        num_classes:    int,
        batch_norm:     bool =              True,
        **kwargs
    ):
        """# Instantiate VGG-16 Neural Network.

        ## Args:
            * input_shape   (Tuple[int]):   Shape of input samples.
            * num_classes   (int):          Number of classes (output logits).
            * batch_norm    (bool):         Use batch normalization. Defaults to True.
        """
        # Initialize network.
        super(VGG16, self).__init__(
            config =        "D",
            input_shape =   input_shape,
            num_classes =   num_classes,
            batch_norm =    batch_norm
        )
