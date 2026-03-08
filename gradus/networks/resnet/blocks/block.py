"""# gradus.networks.resnet.block

ResNet (Residual Neural Network) basic block implementation.
"""

__all__ = ["ResNetBlock"]

from itertools          import count
from logging            import Logger
from typing             import Optional

from torch              import Tensor
from torch.nn           import BatchNorm2d, Conv2d, Module, ReLU
from torch.nn.init      import constant_

from gradus.utilities   import get_logger

class ResNetBlock(Module):
    """# Residual Neural Network Block"""

    # Initialize block ID counter.
    block_id:       count = count()

    # Define expansion property.
    _expansion_:    int =   1

    def __init__(self,
        in_planes:  int,
        planes:     int,
        stride:     int =               1,
        downsample: Optional[Module] =  None
    ):
        """# Instantiate ResNet Block.

        ## Args:
            * in_planes     (int):              Number of input feature channels.
            * planes        (int):              Number of output feature channels produced by block 
                                                before expansion.
            * stride        (int):              Stride applied in first convolution of block. Used 
                                                for spatial downsampling. Defaults to 1.
            * downsample    (Module | None):    Module applied to identity (skip) connection when 
                                                input and output dimensions differ. Defaults to 
                                                None.
        """
        # Initialize module.
        super(ResNetBlock, self).__init__()

        # If invalid arguments are provided...
        if (stride != 1 or in_planes != planes) and downsample is None:

            # Report invalid configuration.
            raise   ValueError(
                        "Downsample layer must be provided when "
                        "stride != 1 or when in_planes != planes"
                    )

        # Initialize logger.
        self.__logger__:    Logger =            get_logger(f"resnet-block-{next(self.block_id)}")

        # Define layers.
        self._conv1_:       Conv2d =            Conv2d(
                                                    in_channels =   in_planes,
                                                    out_channels =  planes,
                                                    stride =        stride,
                                                    kernel_size =   3,
                                                    padding =       1,
                                                    groups =        1,
                                                    bias =          False,
                                                    dilation =      1
                                                )
        self._conv2_:       Conv2d =            Conv2d(
                                                    in_channels =   planes,
                                                    out_channels =  planes,
                                                    stride =        1,
                                                    kernel_size =   3,
                                                    padding =       1,
                                                    groups =        1,
                                                    bias =          False,
                                                    dilation =      1
                                                )
        self._bn1_:         BatchNorm2d =       BatchNorm2d(num_features = planes)
        self._bn2_:         BatchNorm2d =       BatchNorm2d(num_features = planes)
        self._relu_:        ReLU =              ReLU(inplace = True)
        self._downsample_:  Optional[Module] =  downsample
        
        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def expansion(self) -> int:
        """# Expansion Factor of Block"""
        return self._expansion_

    # METHODS ======================================================================================

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Forward Pass Through Network.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Pass through first layer.
        X_1:    Tensor =    self._relu_(self._bn1_(self._conv1_(X)))

        # Pass through second layer.
        X_2:    Tensor =    self._bn2_(self._conv2_(X_1))

        # Perform residual skip connection (with downsampling if configured).
        X_2 +=              X if self._downsample_ is None else self._downsample_(X)

        # Final non-linearity.
        return self._relu_(X_2)
    
    def zero_init(self) -> None:
        """# Initialize Batch Normalization with Zero Weights."""
        # If batch normalization weights are not None...
        if self._bn2_.weight is not None:

            # Initialize them to zero.
            constant_(tensor = self._bn2_.weight, val = 0)