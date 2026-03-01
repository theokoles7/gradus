"""# gradus.networks.resnet.base

ResNet (Residual Neural Network) protocol.
"""

__all__ = ["ResNet"]

from logging                        import Logger
from typing                         import List, Optional, Tuple

from torch                          import flatten, Tensor
from torch.nn                       import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, \
                                           MaxPool2d, Module, ReLU, Sequential
from torch.nn.init                  import constant_, kaiming_normal_

from gradus.networks.resnet.blocks  import BlockType
from gradus.utilities               import get_logger

class ResNet(Module):
    """# Residual Neural Network"""

    def __init__(self,
        block:              BlockType,
        layers:             List[int],
        input_shape:        Tuple[int, ...],
        num_classes:        int,
        zero_init_residual: bool =              False
    ):
        """# Instantiate Residual Neural Network.

        ## Args:
            * block                 (BlockType):    Type of block to use within network's residual 
                                                    layers.
            * layers                (List[int]):    Number of residual layers.
            * input_shape           (Tuple[int]):   Shape of input samples.
            * num_classes           (int):          Number of classes (output logits).
            * zero_init_residual    (bool):         Initialize residual blocks with zero weights. 
                                                    Defaults to False.
        """
        # Initialize module.
        super(ResNet, self).__init__()

        # Initialize logger.
        self.__logger__:        Logger =            get_logger("resnet")

        # Define properties.
        self._in_planes_:       int =               64
        self._width_per_group_: int =               64
        self._dilation_:        int =               1

        # Define layers.
        self._conv1_:           Conv2d =            Conv2d(
                                                        in_channels =   input_shape[0],
                                                        out_channels =  self._in_planes_,
                                                        kernel_size =   7,
                                                        stride =        2,
                                                        padding =       3,
                                                        bias =          False
                                                    )
        self._bn1_:             BatchNorm2d =       BatchNorm2d(
                                                        num_features =  self._in_planes_
                                                    )
        self._max_pool_:        MaxPool2d =         MaxPool2d(
                                                        kernel_size =   3,
                                                        stride =        2,
                                                        padding =       1
                                                    )
        self._avg_pool_:        AdaptiveAvgPool2d = AdaptiveAvgPool2d(
                                                        output_size =   (1, 1)
                                                    )
        self._relu_:            ReLU =              ReLU(in_place = True)
        self._fc_:              Linear =            Linear(
                                                        in_features =   512 * block.expansion,
                                                        out_features =  num_classes
                                                    )
        self._layer1_:          Sequential =        self._make_layer_(
                                                        block =         block,
                                                        planes =        64,
                                                        blocks =        layers[0]
                                                    )
        self._layer2_:          Sequential =        self._make_layer_(
                                                        block =         block,
                                                        planes =        128,
                                                        blocks =        layers[1],
                                                        stride =        2
                                                    )
        self._layer3_:          Sequential =        self._make_layer_(
                                                        block =         block,
                                                        planes =        256,
                                                        blocks =        layers[2],
                                                        stride =        2
                                                    )
        self._layer4_:          Sequential =        self._make_layer_(
                                                        block =         block,
                                                        planes =        512,
                                                        blocks =        layers[3],
                                                        stride =        2
                                                    )
        
        # Initialize weights.
        self._initialize_weights_()

        # Initialize blocks' batch normalization layers to zero weights.
        if zero_init_residual: self._zero_init_residual_()
        
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
        # Convolutional layer.
        X_1:    Tensor =    self._max_pool_(self._relu_(self._bn1_(self._conv1_(X))))

        # First block layer.
        X_2:    Tensor =    self._layer1_(X_1)
        X_3:    Tensor =    self._layer2_(X_2)
        X_4:    Tensor =    self._layer3_(X_3)
        X_5:    Tensor =    self._layer4_(X_4)

        # Classification layer.
        return self._fc_(flatten(input = self._avg_pool_(X_5), start_dim = 1))

    # HELPERS ======================================================================================

    def _initialize_weights_(self):
        """# Initialize Weights."""
        # For each module in network...
        for m in self.modules():

            # If convolving layer...
            if isinstance(m, Conv2d):

                # Initialize with Kaiming normal distribution.
                kaiming_normal_(tensor = m.weight, mode = "fan_out", nonlinearity = "relu")

            # If batch normalization layer...
            elif isinstance(m, BatchNorm2d):

                # Initialize weights to 1 and biases to 0. 
                constant_(tensor = m.weight, val = 1)
                constant_(tensor = m.bias,   val = 0)

    def _make_layer_(self,
        block:  BlockType,
        planes: int,
        blocks: int,
        stride: int =   1
    ) -> Sequential:
        """# Create ResNet Block Layer.

        ## Args:
            * block     (BlockType):    ResNet block type.
            * planes    (int):          Number of input planes.
            * blocks    (int):          Number of blocks to create for layer.
            * stride    (int):          Convolving stride. Defaults to 1.

        ## Returns:
            Sequential: ResNet block layer.
        """
        # Declare downsampling layer.
        downsample:         Optional[Module] =  None

        # If spatial downsampling is needed or if input and output dimensions differ...
        if stride != 1 or self._in_planes_ != planes * block.expansion:

            # Define downsampling layer to match dimensions.
            downsample: Sequential =    Sequential(
                                            Conv2d(
                                                in_channels =   self.inplanes,
                                                out_channels =  planes * block.expansion,
                                                stride =        stride,
                                                kernel_size =   1,
                                                bias =          False
                                            ),
                                            BatchNorm2d(
                                                num_features =  planes * block.expansion
                                            ),
                                        )

        # Initialize list of layers.
        layers:             List[Module] =      []

        # Update in planes.
        self._in_planes_:   int =               planes * block.expansion

        # Create other block layers.
        layers.extend(
            [
                block(in_planes = self._in_planes_, planes = planes)
                for _ in range(1, blocks)
            ]
        )

        # Provide new block layer.
        return Sequential(*layers)
    
    def _zero_init_residual_(self) -> None:
        """# Initialize Residual Branch with Zeros."""
        # For each module in network...
        for m in self.modules(): 

            # If block layer
            if isinstance(m, BlockType): m.zero_init()