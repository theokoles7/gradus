"""# gradus.networks.cvt.blocks.conv_attention

Convolutional Projection Multi-Head Self-Attention block used by CvT stages.
"""

__all__ = ["ConvAttention"]

from torch          import einsum, Tensor
from torch.nn       import BatchNorm2d, Conv2d, Dropout, Linear, Module, Sequential
from torch.nn.functional    import softmax


class ConvAttention(Module):
    """# Convolutional Projection Multi-Head Self-Attention.

    Replaces the linear Q/K/V projections of a Transformer with
    depth-wise separable convolutions. Operates on token tensors of
    shape (B, H * W, C) together with the spatial dimensions (H, W).
    """

    def __init__(self,
        dim:            int,
        num_heads:      int,
        kernel_size:    int =   3,
        stride_kv:      int =   2,
        stride_q:       int =   1,
        padding:        int =   1,
        attn_drop:      float = 0.0,
        proj_drop:      float = 0.0,
        qkv_bias:       bool =  True
    ):
        """# Instantiate Convolutional Attention Block.

        ## Args:
            * dim           (int):      Embedding dimension.
            * num_heads     (int):      Number of attention heads.
            * kernel_size   (int):      Depth-wise convolution kernel size.
            * stride_kv     (int):      Stride used for K/V depth-wise conv.
            * stride_q      (int):      Stride used for Q depth-wise conv.
            * padding       (int):      Depth-wise convolution padding.
            * attn_drop     (float):    Dropout applied to attention weights.
            * proj_drop     (float):    Dropout applied to the output projection.
            * qkv_bias      (bool):     Whether the Q/K/V linear projections use a bias.
        """
        super(ConvAttention, self).__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self._dim_:         int =   dim
        self._num_heads_:   int =   num_heads
        self._head_dim_:    int =   dim // num_heads
        self._scale_:       float = self._head_dim_ ** -0.5

        # Depth-wise separable projections (DW conv + BN) for Q, K, V.
        self._conv_proj_q_: Sequential =    self._build_conv_proj_(
                                                dim =           dim,
                                                kernel_size =   kernel_size,
                                                stride =        stride_q,
                                                padding =       padding
                                            )
        self._conv_proj_k_: Sequential =    self._build_conv_proj_(
                                                dim =           dim,
                                                kernel_size =   kernel_size,
                                                stride =        stride_kv,
                                                padding =       padding
                                            )
        self._conv_proj_v_: Sequential =    self._build_conv_proj_(
                                                dim =           dim,
                                                kernel_size =   kernel_size,
                                                stride =        stride_kv,
                                                padding =       padding
                                            )

        # Linear projections after the convolutional projection.
        self._proj_q_:      Linear =    Linear(dim, dim, bias = qkv_bias)
        self._proj_k_:      Linear =    Linear(dim, dim, bias = qkv_bias)
        self._proj_v_:      Linear =    Linear(dim, dim, bias = qkv_bias)

        self._attn_drop_:   Dropout =   Dropout(p = attn_drop)
        self._proj_:        Linear =    Linear(dim, dim)
        self._proj_drop_:   Dropout =   Dropout(p = proj_drop)

    @staticmethod
    def _build_conv_proj_(
        dim:            int,
        kernel_size:    int,
        stride:         int,
        padding:        int
    ) -> Sequential:
        """# Build Depth-wise Convolutional Projection.

        ## Args:
            * dim           (int):  Feature dimension.
            * kernel_size   (int):  Convolution kernel size.
            * stride        (int):  Convolution stride.
            * padding       (int):  Convolution padding.

        ## Returns:
            * Sequential:   Depth-wise convolution followed by batch normalization.
        """
        return Sequential(
            Conv2d(
                in_channels =   dim,
                out_channels =  dim,
                kernel_size =   kernel_size,
                stride =        stride,
                padding =       padding,
                groups =        dim,
                bias =          False
            ),
            BatchNorm2d(num_features = dim)
        )

    def _project_(self,
        X:      Tensor,
        H:      int,
        W:      int,
        conv:   Sequential
    ) -> Tensor:
        """# Apply Convolutional Projection & Flatten to Tokens.

        ## Args:
            * X     (Tensor):           Input tokens of shape (B, HW, C).
            * H     (int):              Spatial height of the input map.
            * W     (int):              Spatial width of the input map.
            * conv  (Sequential):       Depth-wise projection module.

        ## Returns:
            * Tensor:   Projected tokens of shape (B, H'W', C).
        """
        B, _, C = X.shape

        # Reshape tokens to (B, C, H, W) for the depth-wise conv.
        feature_map:    Tensor =    X.transpose(1, 2).reshape(B, C, H, W)
        feature_map =               conv(feature_map)

        # Flatten spatial dimensions back to tokens.
        return feature_map.flatten(2).transpose(1, 2)

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
            * Tensor:   Attention output of shape (B, H * W, C).
        """
        q:  Tensor =    self._project_(X = X, H = H, W = W, conv = self._conv_proj_q_)
        k:  Tensor =    self._project_(X = X, H = H, W = W, conv = self._conv_proj_k_)
        v:  Tensor =    self._project_(X = X, H = H, W = W, conv = self._conv_proj_v_)

        q = self._proj_q_(q)
        k = self._proj_k_(k)
        v = self._proj_v_(v)

        # Split into heads: (B, N, C) -> (B, heads, N, head_dim).
        B, _, _ = q.shape
        q = q.reshape(B, -1, self._num_heads_, self._head_dim_).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self._num_heads_, self._head_dim_).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self._num_heads_, self._head_dim_).permute(0, 2, 1, 3)

        # Scaled dot-product attention.
        attn:   Tensor =    einsum("bhqd,bhkd->bhqk", q, k) * self._scale_
        attn =              softmax(attn, dim = -1)
        attn =              self._attn_drop_(attn)

        out:    Tensor =    einsum("bhqk,bhkd->bhqd", attn, v)
        out =               out.permute(0, 2, 1, 3).reshape(B, -1, self._dim_)

        out = self._proj_(out)
        out = self._proj_drop_(out)

        return out
