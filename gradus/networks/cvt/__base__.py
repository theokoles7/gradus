"""# gradus.networks.cvt.__base__

Shared Convolutional Vision Transformer (CvT) protocol.
"""

__all__ = ["CvT"]

from typing     import List, Tuple

from torch      import Tensor
from torch.nn   import Conv2d, LayerNorm, Linear, Module, ModuleList
from torch.nn.init  import constant_, trunc_normal_

from gradus.networks.cvt.blocks     import CvTStage
from gradus.networks.protocol       import Network


class CvT(Network):
    """# Convolutional Vision Transformer.

    Three-stage hierarchical Transformer that replaces patch-embeddings
    with convolutional token embeddings and the linear Q/K/V projections
    with depth-wise separable convolutional projections.
    """

    def __init__(self,
        id:                 str,
        input_shape:        Tuple[int, ...],
        num_classes:        int,
        embed_dims:         List[int],
        depths:             List[int],
        num_heads:          List[int],
        patch_sizes:        List[int],
        patch_strides:      List[int],
        patch_paddings:     List[int],
        mlp_ratios:         List[float],
        qkv_bias:           bool =          True,
        drop_rate:          float =         0.0,
        attn_drop_rate:     float =         0.0,
        drop_path_rate:     float =         0.1
    ):
        """# Instantiate Convolutional Vision Transformer.

        ## Args:
            * id                (str):          Network identifier.
            * input_shape       (Tuple[int]):   Shape of input samples (C, H, W).
            * num_classes       (int):          Number of output logits.
            * embed_dims        (List[int]):    Per-stage embedding dimensions.
            * depths            (List[int]):    Per-stage number of blocks.
            * num_heads         (List[int]):    Per-stage attention head counts.
            * patch_sizes       (List[int]):    Per-stage conv-embedding kernel sizes.
            * patch_strides     (List[int]):    Per-stage conv-embedding strides.
            * patch_paddings    (List[int]):    Per-stage conv-embedding paddings.
            * mlp_ratios        (List[float]):  Per-stage MLP expansion ratios.
            * qkv_bias          (bool):         Whether Q/K/V projections use a bias.
            * drop_rate         (float):        Dropout applied in MLPs and output projection.
            * attn_drop_rate    (float):        Dropout applied to attention weights.
            * drop_path_rate    (float):        Maximum stochastic depth rate (linear schedule).
        """
        super(CvT, self).__init__(network_id = id)

        num_stages:     int =   len(depths)
        for name, values in (
            ("embed_dims",      embed_dims),
            ("num_heads",       num_heads),
            ("patch_sizes",     patch_sizes),
            ("patch_strides",   patch_strides),
            ("patch_paddings",  patch_paddings),
            ("mlp_ratios",      mlp_ratios),
        ):
            if len(values) != num_stages:
                raise ValueError(
                    f"{name} must have length {num_stages}, got {len(values)}"
                )

        # Linear stochastic depth schedule across every block in every stage.
        total_blocks:   int =           sum(depths)
        if total_blocks > 1:
            dpr:        List[float] =   [
                                            drop_path_rate * i / (total_blocks - 1)
                                            for i in range(total_blocks)
                                        ]
        else:
            dpr =                       [0.0] * total_blocks

        stages:         List[CvTStage] =    []
        in_channels:    int =               input_shape[0]
        cursor:         int =               0

        for stage_idx in range(num_stages):
            depth:      int =   depths[stage_idx]
            stage:      CvTStage =  CvTStage(
                                        in_channels =       in_channels,
                                        embed_dim =         embed_dims[stage_idx],
                                        depth =             depth,
                                        num_heads =         num_heads[stage_idx],
                                        patch_size =        patch_sizes[stage_idx],
                                        patch_stride =      patch_strides[stage_idx],
                                        patch_padding =     patch_paddings[stage_idx],
                                        mlp_ratio =         mlp_ratios[stage_idx],
                                        qkv_bias =          qkv_bias,
                                        drop_rate =         drop_rate,
                                        attn_drop_rate =    attn_drop_rate,
                                        drop_path_rates =   dpr[cursor : cursor + depth]
                                    )
            stages.append(stage)

            in_channels =   embed_dims[stage_idx]
            cursor +=       depth

        self._stages_:  ModuleList =    ModuleList(stages)
        self._norm_:    LayerNorm =     LayerNorm(normalized_shape = embed_dims[-1])
        self._head_:    Linear =        Linear(
                                            in_features =   embed_dims[-1],
                                            out_features =  num_classes
                                        )

        self._initialize_weights_()

    def forward(self,
        X:                  Tensor,
        return_activations: bool =  False
    ) -> Tensor:
        """# Forward Pass Through Network.

        ## Args:
            * X                     (Tensor):   Input tensor of shape (B, C, H, W).
            * return_activations    (bool):     Return per-stage activations alongside logits.

        ## Returns:
            * Tensor:   Logits of shape (B, num_classes). When `return_activations`
                        is True a tuple (logits, activations) is returned instead.
        """
        activations:    List[Tensor] = []

        for stage in self._stages_:
            X, _, _ = stage(X)

            if return_activations:
                activations.append(X.detach())

        # Global average pool over the final spatial dimensions.
        B, C, _, _ =    X.shape
        tokens:         Tensor =    X.flatten(2).transpose(1, 2)
        tokens =                    self._norm_(tokens)
        pooled:         Tensor =    tokens.mean(dim = 1)

        logits:         Tensor =    self._head_(pooled)

        if return_activations:
            return logits, activations

        return logits

    def _initialize_weights_(self) -> None:
        """# Initialize Weights."""
        for m in self.modules():

            if isinstance(m, Linear):
                trunc_normal_(tensor = m.weight, std = 0.02)
                if m.bias is not None:
                    constant_(tensor = m.bias, val = 0)

            elif isinstance(m, LayerNorm):
                constant_(tensor = m.bias,   val = 0)
                constant_(tensor = m.weight, val = 1)

            elif isinstance(m, Conv2d):
                trunc_normal_(tensor = m.weight, std = 0.02)
                if m.bias is not None:
                    constant_(tensor = m.bias, val = 0)
