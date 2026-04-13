"""# gradus.networks.vgg.vgg_11

VGG-11 neural network implementation.
"""

__all__ = ["VGG11"]

from typing                                 import List, Tuple, Union

from gradus.networks.vgg.__base__           import VGG
from gradus.networks.vgg.vgg_11.__args__    import VGG11Config
from gradus.registration                    import register_network

@register_network(
    id =        "vgg-11",
    config =    VGG11Config,
    tags =      ["vgg", "cnn", "11-layers"]
)
class VGG11(VGG):
    """# 11-Layer VGG Neural Network"""

    def __init__(self,
        input_shape:    Tuple[int, ...],
        num_classes:    int,
        batch_norm:     bool =              True,
        **kwargs
    ):
        """# Instantiate VGG-11 Neural Network.

        ## Args:
            * input_shape   (Tuple[int]):   Shape of input samples.
            * num_classes   (int):          Number of classes (output logits).
            * batch_norm    (bool):         Use batch normalization. Defaults to True.
        """
        # Define layer config.
        layer_config:   List[Union[int, str]] = [
                                                    64,         "M",
                                                    128,        "M",
                                                    256, 256,   "M",
                                                    512, 512,   "M",
                                                    512, 512,   "M"
                                                ]

        # Initialize network.
        super(VGG11, self).__init__(
            id =            "vgg-11",
            layer_config =  layer_config,
            input_shape =   input_shape,
            num_classes =   num_classes,
            batch_norm =    batch_norm
        )
