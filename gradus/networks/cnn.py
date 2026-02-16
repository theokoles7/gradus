"""# gradus.networks.cnn

Rudimentary implementation of a convolutional neural network for light-duty tasks.
"""

__all__ = ["CNN"]

from typing     import List

from torch      import Tensor
from torch.nn   import BatchNorm2d, Conv2d, Dropout, Linear, MaxPool2d, Module, ReLU

class CNN(Module):
    """# Convolutional Neural Network"""

    def __init__(self,
        input_channels: int,
        num_classes:    int,
        input_size:     int
    ):
        """# Instantiate Convolutional Neural Network.

        ## Args:
            * input_channels    (int):  Number of channels contained in input images.
            * num_classes       (int):  Number of classes represented in dataset.
            * input_size        (int):  Dimension of (square) input image.
        """
        # Initialize module.
        super(CNN, self).__init__()

        # Calculate flattened size after convolving layers.
        final_spatial_size: int =           input_size // (2**3)
        flattened_size:     int =           128 * (final_spatial_size ** 2)

        # Define layers.
        self._conv_1_:      Conv2d =        Conv2d(
                                                in_channels =   input_channels,
                                                out_channels =  32,
                                                kernel_size =   3,
                                                padding =       1
                                            )
        self._conv_2_:      Conv2d =        Conv2d(
                                                in_channels =   32,
                                                out_channels =  64,
                                                kernel_size =   3,
                                                padding =       1
                                            )
        self._conv_3_:      Conv2d =        Conv2d(
                                                in_channels =   64,
                                                out_channels =  128,
                                                kernel_size =   3,
                                                padding =       1
                                            )
        self._fc_1_:        Linear =        Linear(
                                                in_features =   flattened_size,
                                                out_features =  256
                                            )
        self._fc_2_:        Linear =        Linear(
                                                in_features =   256,
                                                out_features =  num_classes
                                            )
        self._bn_1_:        BatchNorm2d =   BatchNorm2d(num_features = 32)
        self._bn_2_:        BatchNorm2d =   BatchNorm2d(num_features = 64)
        self._bn_3_:        BatchNorm2d =   BatchNorm2d(num_features = 128)
        self._pool_:        MaxPool2d =     MaxPool2d(kernel_size = 2, stride = 2)
        self._drop_:        Dropout =       Dropout(p = 0.5)
        self._relu_:        ReLU =          ReLU()

    # METHODS ======================================================================================

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Forward Pass Through Network.

        ## Args:
            * X (Tensor):   Tensor of dataset sample(s).

        ## Returns:
            * Tensor:   Tensor of sample classification(s).
        """
        # Pass through convolving layer 1.
        X_1:    Tensor =    self._pool_(self._relu_(self._bn_1_(self._conv_1_(X))))

        # Pass through convolving layer 2.
        X_2:    Tensor =    self._pool_(self._relu_(self._bn_2_(self._conv_2_(X_1))))

        # Pass through convolving layer 3.
        X_3:    Tensor =    self._pool_(self._relu_(self._bn_3_(self._conv_3_(X_2))))

        # Flatten.
        X_4:    Tensor =    X_3.view(X_3.size(0), -1)

        # Classify sample(s).
        return self._fc_2_(self._drop_(self._relu_(self._fc_1_(X_4))))