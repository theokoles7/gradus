"""# gradus.networks.cvt.blocks.conv_embed

Convolutional Token Embedding module used by CvT stages.
"""

__all__ = ["ConvEmbed"]

from typing     import Tuple

from torch      import Tensor
from torch.nn   import Conv2d, LayerNorm, Module


class ConvEmbed(Module):
    """# Convolutional Token Embedding.

    Projects a (B, C_in, H, W) feature map into a new feature map
    (B, embed_dim, H', W') using a strided convolution, then applies
    layer normalization across the embedding dimension.
    """

    def __init__(self,
        in_channels:    int,
        embed_dim:      int,
        kernel_size:    int,
        stride:         int,
        padding:        int
    ):
        """# Instantiate Convolutional Token Embedding.

        ## Args:
            * in_channels   (int):  Number of input channels.
            * embed_dim     (int):  Target embedding dimension.
            * kernel_size   (int):  Convolution kernel size.
            * stride        (int):  Convolution stride.
            * padding       (int):  Convolution padding.
        """
        super(ConvEmbed, self).__init__()

        self._proj_:    Conv2d =    Conv2d(
                                        in_channels =   in_channels,
                                        out_channels =  embed_dim,
                                        kernel_size =   kernel_size,
                                        stride =        stride,
                                        padding =       padding
                                    )
        self._norm_:    LayerNorm = LayerNorm(normalized_shape = embed_dim)

    def forward(self,
        X:  Tensor
    ) -> Tuple[Tensor, int, int]:
        """# Forward Pass.

        ## Args:
            * X (Tensor):   Input feature map of shape (B, C_in, H, W).

        ## Returns:
            * Tuple[Tensor, int, int]:  Output feature map of shape
              (B, embed_dim, H', W') along with the new spatial size.
        """
        X =             self._proj_(X)
        _, _, H, W =    X.shape

        # LayerNorm operates over the channel/embedding dimension.
        X = X.flatten(2).transpose(1, 2)
        X = self._norm_(X)
        X = X.transpose(1, 2).reshape(-1, X.shape[-1], H, W)

        return X, H, W
