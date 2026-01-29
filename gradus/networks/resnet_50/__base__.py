"""# gradus.networks.resnet_50.base

ResNet-50 neural network implementation.
"""

from logging                            import Logger
from typing                             import Tuple

from torch                              import Tensor
from torch.nn                           import Conv2d, Module
from torchvision.models                 import resnet50, ResNet

from gradus.networks.resnet_50.__args__ import ResNet50Config
from gradus.registration                import register_network
from gradus.utilities                   import get_logger

@register_network(
    id =        "resnet-50",
    config =    ResNet50Config,
    tags =      ["residual", "cnn", "50-layers"]
)
class ResNet50(Module):
    """# Residual Neural Network with 50 Layers
    
    Reference: https://arxiv.org/pdf/1512.03385
    """

    def __init__(self,
        # Dataset
        input_shape:        Tuple[int, int, int],
        num_classes:        int,
        **kwargs
    ):
        """# Instantiate ResNet50 Neural Network.

        ## Args:
            * input_shape   (Tuple[int, int, int]): Expected input shape (C, H, W).
            * num_classes   (int):                  Number of classes contained in dataset.
        """
        # Initialize network.
        super(ResNet50, self).__init__()

        # Initialize logger.
        self.__logger__:    Logger =                get_logger("resnet-50")

        # Define properties.
        self._input_shape_: Tuple[int, int, int] =  input_shape
        self._num_classes_: int =                   num_classes

        # Initialize ResNet-50 model.
        self.model:         ResNet =                resnet50(num_classes = num_classes)
        
        # Replace first convolving layer to match input shape.
        self.model.conv1 =                          Conv2d(
                                                        in_channels =   self._input_shape_[0],
                                                        out_channels =  64,
                                                        kernel_size =   7,
                                                        stride =        2,
                                                        padding =       3,
                                                        bias =          False
                                                    )

        # Debug initialization.
        self.__logger__.debug(f"Initialized {self}")

    # PROPERTIES ===================================================================================

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """# Expected Input Shape (C, H, W)"""
        return self._input_shape_
    
    @property
    def num_classes(self) -> int:
        """# Number of Outputs from Final Layer"""
        return self._num_classes_
    
    @property
    def num_weights(self) -> int:
        """# Total Number of Trainable Parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        return self.model(X)