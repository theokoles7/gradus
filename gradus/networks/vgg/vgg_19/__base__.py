"""# gradus.networks.vgg.vgg_19

VGG-19 neural network implementation.
"""

__all__ = ["VGG19"]

from typing                                 import List, Tuple, Union

from gradus.networks.vgg.__base__           import VGG
from gradus.networks.vgg.vgg_19.__args__    import VGG19Config
from gradus.registration                    import register_network

@register_network(
    id =        "vgg-19",
    config =    VGG19Config,
    tags =      ["vgg", "cnn", "19-layers"]
)
class VGG19(VGG):
    """# 19-Layer VGG Neural Network"""

    def __init__(self,
        input_shape:    Tuple[int, ...],
        num_classes:    int,
        batch_norm:     bool =              True,
        **kwargs
    ):
        """# Instantiate VGG-19 Neural Network.

        ## Args:
            * input_shape   (Tuple[int]):   Shape of input samples.
            * num_classes   (int):          Number of classes (output logits).
            * batch_norm    (bool):         Use batch normalization. Defaults to True.
        """
        # Define layer config.
        layer_config:   List[Union[int, str]] = [
                                                    64,   64,           "M",
                                                    128, 128,           "M",
                                                    256, 256, 256, 256, "M",
                                                    512, 512, 512, 512, "M",
                                                    512, 512, 512, 512, "M"
                                                ]

        # Initialize network.
        super(VGG19, self).__init__(
            layer_config =  layer_config,
            input_shape =   input_shape,
            num_classes =   num_classes,
            batch_norm =    batch_norm
        )
