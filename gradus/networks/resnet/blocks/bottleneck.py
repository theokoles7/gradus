"""# gradus.networks.resnet.bottleneck

ResNet (Residual Neural Network) bottleneck block implementation.
"""

__all__ = ["ResNetBottleneck"]

from itertools          import count
from logging            import Logger
from typing             import Optional

from torch              import Tensor
from torch.nn           import BatchNorm2d, Conv2d, Module, ReLU
from torch.nn.init      import constant_

from gradus.utilities   import get_logger

class ResNetBottleneck(Module):
    """# Residual Neural Network Bottleneck Block"""

    # Initialize block ID counter.
    block_id:       count = count()

    # Define expansion property.
    _expansion_:    int =   4

    def __init__(self,
        in_planes:  int,
        planes:     int,
        stride:     int =               1,
        downsample: Optional[Module] =  None
    ):
        """# Instantiate ResNet Bottleneck Block.

        ## Args:
            * in_planes     (int):              Number of input feature channels.
            * planes        (int):              Base number of feature channels for the bottleneck.
                                                The block outputs `planes * expansion` channels.
            * stride        (int):              Stride applied for spatial downsampling. In the
                                                Bottleneck design, this stride is applied on the 3x3 
                                                convolution. Defaults to 1.
            * downsample    (Module | None):    Module applied to identity (skip) connection when
                                                input and output dimensions differ. Defaults to 
                                                None.
        """
        # Initialize module.
        super(ResNetBottleneck, self).__init__()

        # If invalid arguments are provided...
        if (stride != 1 or in_planes != planes * self._expansion_) and downsample is None:

            # Report invalid configuration.
            raise   ValueError(
                        "Downsample layer must be provided when "
                        "stride != 1 or when in_planes != planes"
                    )
        
        # Initialize logger.
        self.__logger__:    Logger =            get_logger(f"resnet-bottleneck-{next(self.block_id)}")

        # Define layers.
        self._conv1_:       Conv2d =            Conv2d(
                                                    in_channels =   in_planes,
                                                    out_channels =  planes,
                                                    kernel_size =   1,
                                                    stride =        1,
                                                    bias =          False
                                                )
        self._conv2_:       Conv2d =            Conv2d(
                                                    in_channels =   planes,
                                                    out_channels =  planes,
                                                    kernel_size =   3,
                                                    stride =        stride,
                                                    padding =       1,
                                                    groups =        1,
                                                    dilation =      1,
                                                    bias =          False
                                                )
        self._conv3_:       Conv2d =            Conv2d(
                                                    in_channels =   planes,
                                                    out_channels =  planes * self._expansion_,
                                                    kernel_size =   1,
                                                    stride =        1,
                                                    bias =          False
                                                )
        self._bn1_:         BatchNorm2d =       BatchNorm2d(num_features = planes)
        self._bn2_:         BatchNorm2d =       BatchNorm2d(num_features = planes)
        self._bn3_:         BatchNorm2d =       BatchNorm2d(num_features = planes * self._expansion_)
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
        X_2:    Tensor =    self._relu_(self._bn2_(self._conv2_(X_1)))

        # Pass through third layer.
        X_3:    Tensor =    self._bn3_(self._conv3_(X_2))

        # Perform residual skip connection (with downsampling if configured).
        X_3 +=              X if self._downsample_ is None else self._downsample_(X)

        # Final non-linearity.
        return self._relu_(X_3)
    
    def zero_init(self) -> None:
        """# Initialize Batch Normalization with Zero Weights."""
        # If batch normalization weights are not None...
        if self._bn3_.weight is not None:

            # Initialize them to zero.
            constant_(tensor = self._bn3_.weight, val = 0)