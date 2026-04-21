"""# gradus.networks.cvt.cvt_13.base

CvT-13 neural network implementation.
"""

__all__ = ["CvT13"]

from typing     import Tuple

from gradus.networks.cvt.__base__               import CvT
from gradus.networks.cvt.cvt_13.__args__        import CvT13Config
from gradus.registration                        import register_network


@register_network(
    id =        "cvt-13",
    config =    CvT13Config,
    tags =      ["transformer", "convolutional", "vision-transformer", "13-blocks"]
)
class CvT13(CvT):
    """# 13-Block Convolutional Vision Transformer.

    Follows the CvT-13 configuration from "CvT: Introducing Convolutions to
    Vision Transformers" (Wu et al., 2021): three hierarchical stages with
    (1, 2, 10) transformer blocks and embedding dimensions (64, 192, 384).
    """

    def __init__(self,
        input_shape:    Tuple[int, ...],
        num_classes:    int,
        drop_rate:      float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        **kwargs
    ):
        """# Instantiate CvT-13 Neural Network.

        ## Args:
            * input_shape       (Tuple[int]):   Shape of input samples (C, H, W).
            * num_classes       (int):          Number of output logits.
            * drop_rate         (float):        Dropout applied in MLPs and output projections.
            * attn_drop_rate    (float):        Dropout applied to attention weights.
            * drop_path_rate    (float):        Maximum stochastic depth rate.
        """
        # For small inputs (e.g. CIFAR-10) shrink the stem stride so we do not
        # collapse spatial resolution to zero before the attention stages run.
        small_stem:     bool =  input_shape[1] <= 64

        patch_sizes =       [7,         3,      3]
        patch_strides =     [2 if small_stem else 4,    2,      2]
        patch_paddings =    [2,         1,      1]

        super(CvT13, self).__init__(
            id =                "cvt-13",
            input_shape =       input_shape,
            num_classes =       num_classes,
            embed_dims =        [64,        192,    384],
            depths =            [1,         2,      10],
            num_heads =         [1,         3,      6],
            patch_sizes =       patch_sizes,
            patch_strides =     patch_strides,
            patch_paddings =    patch_paddings,
            mlp_ratios =        [4.0,       4.0,    4.0],
            qkv_bias =          True,
            drop_rate =         drop_rate,
            attn_drop_rate =    attn_drop_rate,
            drop_path_rate =    drop_path_rate
        )
