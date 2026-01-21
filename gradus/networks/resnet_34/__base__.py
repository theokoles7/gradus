"""# gradus.networks.resnet_34.base

ResNet-34 neural network implementation.
"""

from logging                            import Logger
from typing                             import Tuple

from torch                              import Tensor
from torch.nn                           import Module
from torchvision.models                 import resnet34, ResNet

from gradus.networks.resnet_34.__args__ import ResNet34Config
from gradus.registration                import register_network
from gradus.utilities                   import get_logger

@register_network(
    id =        "resnet-34",
    config =    ResNet34Config,
    tags =      ["residual", "cnn", "34-layers"]
)
class ResNet34(Module):
    """# Residual Neural Network with 34 Layers
    
    Reference: https://arxiv.org/pdf/1512.03385
    """

    def __init__(self,
        # Dataset
        input_shape:        Tuple[int, int, int],
        num_classes:        int
    ):
        """# Instantiate ResNet34 Neural Network.

        ## Args:
            * input_shape   (Tuple[int, int, int]): Expected input shape (C, H, W).
            * num_classes   (int):                  Number of classes contained in dataset.
        """
        # Initialize network.
        super(ResNet34, self).__init__()

        # Initialize logger.
        self.__logger__:    Logger =                get_logger("resnet-34")

        # Initialize ResNet-34 model.
        self.model:         ResNet =                resnet34(
                                                        num_classes =   num_classes
                                                    )

        # Define properties.
        self._input_shape_: Tuple[int, int, int] =  input_shape
        self._num_classes_: int =                   num_classes

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