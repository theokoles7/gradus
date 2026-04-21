"""# gradus.networks.cvt.blocks.cvt_block

Single CvT Transformer block: pre-norm convolutional attention + MLP.
"""

__all__ = ["CvTBlock"]

from torch      import Tensor
from torch.nn   import Dropout, GELU, Identity, LayerNorm, Linear, Module, Sequential

from gradus.networks.cvt.blocks.conv_attention  import ConvAttention


class _DropPath(Module):
    """# Stochastic Depth (Drop Path).

    Drops the residual branch with probability `drop_prob` during training.
    At inference it behaves as an identity transformation.
    """

    def __init__(self,
        drop_prob:  float = 0.0
    ):
        super(_DropPath, self).__init__()
        self._drop_prob_:   float = drop_prob

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        if self._drop_prob_ == 0.0 or not self.training:
            return X

        keep_prob:  float =     1.0 - self._drop_prob_
        shape =                 (X.shape[0],) + (1,) * (X.dim() - 1)
        random =                X.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random =            random.div(keep_prob)
        return X * random


class _MLP(Module):
    """# Feed-Forward MLP used inside a CvT block."""

    def __init__(self,
        dim:            int,
        hidden_dim:     int,
        drop:           float = 0.0
    ):
        super(_MLP, self).__init__()
        self._net_: Sequential =    Sequential(
                                        Linear(dim, hidden_dim),
                                        GELU(),
                                        Dropout(p = drop),
                                        Linear(hidden_dim, dim),
                                        Dropout(p = drop)
                                    )

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        return self._net_(X)


class CvTBlock(Module):
    """# CvT Transformer Block.

    Applies convolutional multi-head self-attention followed by an MLP,
    both wrapped by pre-norm residual connections and optional stochastic
    depth.
    """

    def __init__(self,
        dim:            int,
        num_heads:      int,
        mlp_ratio:      float = 4.0,
        qkv_bias:       bool =  True,
        drop:           float = 0.0,
        attn_drop:      float = 0.0,
        drop_path:      float = 0.0,
        kernel_size:    int =   3,
        stride_kv:      int =   2,
        stride_q:       int =   1,
        padding:        int =   1
    ):
        """# Instantiate CvT Block.

        ## Args:
            * dim           (int):      Embedding dimension.
            * num_heads     (int):      Number of attention heads.
            * mlp_ratio     (float):    Ratio of hidden MLP dim to embedding dim.
            * qkv_bias      (bool):     Whether Q/K/V projections use a bias.
            * drop          (float):    Dropout applied in the MLP and output projection.
            * attn_drop     (float):    Dropout applied to attention weights.
            * drop_path     (float):    Stochastic depth rate.
            * kernel_size   (int):      Convolutional projection kernel size.
            * stride_kv     (int):      Stride for K/V depth-wise convolution.
            * stride_q      (int):      Stride for Q depth-wise convolution.
            * padding       (int):      Convolutional projection padding.
        """
        super(CvTBlock, self).__init__()

        self._norm1_:       LayerNorm =     LayerNorm(normalized_shape = dim)
        self._attn_:        ConvAttention = ConvAttention(
                                                dim =           dim,
                                                num_heads =     num_heads,
                                                kernel_size =   kernel_size,
                                                stride_kv =     stride_kv,
                                                stride_q =      stride_q,
                                                padding =       padding,
                                                attn_drop =     attn_drop,
                                                proj_drop =     drop,
                                                qkv_bias =      qkv_bias
                                            )
        self._drop_path_:   Module =        _DropPath(drop_prob = drop_path) if drop_path > 0.0 else Identity()
        self._norm2_:       LayerNorm =     LayerNorm(normalized_shape = dim)
        self._mlp_:         _MLP =          _MLP(
                                                dim =           dim,
                                                hidden_dim =    int(dim * mlp_ratio),
                                                drop =          drop
                                            )

    def forward(self,
        X:  Tensor,
        H:  int,
        W:  int
    ) -> Tensor:
        """# Forward Pass.

        ## Args:
            * X (Tensor):   Tokens of shape (B, H * W, C).
            * H (int):      Spatial height of the token grid.
            * W (int):      Spatial width of the token grid.

        ## Returns:
            * Tensor:   Output tokens of shape (B, H * W, C).
        """
        X = X + self._drop_path_(self._attn_(self._norm1_(X), H, W))
        X = X + self._drop_path_(self._mlp_(self._norm2_(X)))
        return X
