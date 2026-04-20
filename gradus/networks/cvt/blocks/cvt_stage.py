"""# gradus.networks.cvt.blocks.cvt_stage

One CvT stage: convolutional token embedding followed by a sequence of
convolutional Transformer blocks.
"""

__all__ = ["CvTStage"]

from typing     import List, Tuple

from torch      import Tensor
from torch.nn   import Dropout, Module, ModuleList

from gradus.networks.cvt.blocks.conv_embed  import ConvEmbed
from gradus.networks.cvt.blocks.cvt_block   import CvTBlock


class CvTStage(Module):
    """# CvT Stage.

    Applies a convolutional token embedding to the incoming feature map and
    then processes the resulting tokens with `depth` CvT blocks.
    """

    def __init__(self,
        in_channels:    int,
        embed_dim:      int,
        depth:          int,
        num_heads:      int,
        patch_size:     int,
        patch_stride:   int,
        patch_padding:  int,
        mlp_ratio:      float =         4.0,
        qkv_bias:       bool =          True,
        drop_rate:      float =         0.0,
        attn_drop_rate: float =         0.0,
        drop_path_rates:List[float] =   None,
        kernel_size:    int =           3,
        stride_kv:      int =           2,
        stride_q:       int =           1,
        padding:        int =           1
    ):
        """# Instantiate CvT Stage.

        ## Args:
            * in_channels       (int):          Number of input channels.
            * embed_dim         (int):          Target embedding dimension.
            * depth             (int):          Number of CvT blocks in the stage.
            * num_heads         (int):          Number of attention heads per block.
            * patch_size        (int):          Convolutional embedding kernel size.
            * patch_stride      (int):          Convolutional embedding stride.
            * patch_padding     (int):          Convolutional embedding padding.
            * mlp_ratio         (float):        Ratio of MLP hidden dim to embed dim.
            * qkv_bias          (bool):         Whether Q/K/V projections use a bias.
            * drop_rate         (float):        Dropout applied after the embedding and
                                                within MLP / output projections.
            * attn_drop_rate    (float):        Dropout applied to attention weights.
            * drop_path_rates   (List[float]):  Per-block stochastic depth rates; must be
                                                of length `depth` when provided.
            * kernel_size       (int):          Convolutional projection kernel size.
            * stride_kv         (int):          Stride for K/V depth-wise projection.
            * stride_q          (int):          Stride for Q depth-wise projection.
            * padding           (int):          Convolutional projection padding.
        """
        super(CvTStage, self).__init__()

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        if len(drop_path_rates) != depth:
            raise ValueError(
                f"drop_path_rates must have length {depth}, got {len(drop_path_rates)}"
            )

        self._patch_embed_: ConvEmbed = ConvEmbed(
                                            in_channels =   in_channels,
                                            embed_dim =     embed_dim,
                                            kernel_size =   patch_size,
                                            stride =        patch_stride,
                                            padding =       patch_padding
                                        )
        self._pos_drop_:    Dropout =   Dropout(p = drop_rate)

        self._blocks_:      ModuleList = ModuleList(
                                            [
                                                CvTBlock(
                                                    dim =           embed_dim,
                                                    num_heads =     num_heads,
                                                    mlp_ratio =     mlp_ratio,
                                                    qkv_bias =      qkv_bias,
                                                    drop =          drop_rate,
                                                    attn_drop =     attn_drop_rate,
                                                    drop_path =     drop_path_rates[i],
                                                    kernel_size =   kernel_size,
                                                    stride_kv =     stride_kv,
                                                    stride_q =      stride_q,
                                                    padding =       padding
                                                )
                                                for i in range(depth)
                                            ]
                                        )

        self._embed_dim_:   int =       embed_dim

    @property
    def embed_dim(self) -> int:
        """# Stage Output Embedding Dimension."""
        return self._embed_dim_

    def forward(self,
        X:  Tensor
    ) -> Tuple[Tensor, int, int]:
        """# Forward Pass Through CvT Stage.

        ## Args:
            * X (Tensor):   Feature map of shape (B, C_in, H, W).

        ## Returns:
            * Tuple[Tensor, int, int]:  Feature map of shape (B, embed_dim, H', W')
              together with the output spatial dimensions (H', W').
        """
        X, H, W =   self._patch_embed_(X)
        B, C, _, _ = X.shape

        # Flatten to tokens for attention blocks.
        X = X.flatten(2).transpose(1, 2)
        X = self._pos_drop_(X)

        for block in self._blocks_:
            X = block(X, H, W)

        # Restore spatial layout for the next stage.
        X = X.transpose(1, 2).reshape(B, C, H, W)
        return X, H, W
