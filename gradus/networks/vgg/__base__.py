"""# gradus.networks.vgg.base

VGG (Visual Geometry Group) network protocol.
"""

__all__ = ["VGG"]

from typing                     import List, Tuple, Union

from torch                      import flatten, Tensor
from torch.nn                   import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout, Linear, \
                                       MaxPool2d, Module, ReLU, Sequential
from torch.nn.init              import constant_, kaiming_normal_, normal_

from gradus.networks.protocol   import Network

class VGG(Network):
    """# VGG Neural Network"""

    def __init__(self,
        id:             str,
        layer_config:   List[Union[int, str]],
        input_shape:    Tuple[int, ...],
        num_classes:    int,
        batch_norm:     bool =              True
    ):
        """# Instantiate VGG Neural Network.

        ## Args:
            * id            (str):              Network identifier.
            * config        (List[int | str]):  VGG layer configuration.
            * input_shape   (Tuple[int]):       Shape of input samples.
            * num_classes   (int):              Number of classes (output logits).
            * batch_norm    (bool):             Use batch normalization. Defaults to True.
        """
        # Initialize module.
        super(VGG, self).__init__(network_id = id)

        # If input size is 64 or less, we'll use smaller convolution on first layer.
        self._small_stem_:  bool =              input_shape[1] <= 64

        # If using small stem, state such.
        if self._small_stem_: self.__logger__.info(f"Using small stem configuration")

        # Build feature extraction layers.
        self._features_:    Sequential =        self._make_layers_(
                                                    config =        layer_config,
                                                    in_channels =   input_shape[0],
                                                    batch_norm =    batch_norm,
                                                    small_stem =    self._small_stem_
                                                )

        # Adaptive pooling to fixed spatial size.
        self._avg_pool_:    AdaptiveAvgPool2d = AdaptiveAvgPool2d(
                                                    output_size = (1, 1) if self._small_stem_ else (7, 7)
                                                )

        # Classification layers.
        self._classifier_:  Sequential =        Sequential(
                                                    Linear(
                                                        in_features =   512 * (1 if self._small_stem_ else 49),
                                                        out_features =  4096
                                                    ),
                                                    ReLU(inplace = True),
                                                    Dropout(p = 0.5),
                                                    Linear(
                                                        in_features =   4096,
                                                        out_features =  4096
                                                    ),
                                                    ReLU(inplace = True),
                                                    Dropout(p = 0.5),
                                                    Linear(
                                                        in_features =   4096,
                                                        out_features =  num_classes
                                                    ),
                                                )

        # Initialize weights.
        self._initialize_weights_()

    # METHODS ======================================================================================

    def forward(self,
        X:                  Tensor,
        return_activations: bool =  False
    ) -> Tensor:
        """# Forward Pass Through Network.

        ## Args:
            * X (Tensor):   Input tensor.

        ## Returns:
            * Tensor:   Output tensor.
        """
        # Feature extraction.
        X_1:    Tensor =    self._features_(X)

        # Adaptive average pooling.
        X_2:    Tensor =    self._avg_pool_(X_1)

        # Flatten.
        X_3:    Tensor =    flatten(input = X_2, start_dim = 1)

        # If returning activations...
        if return_activations:

            # Record activation signals.
            activations:    List[Tensor] =  [
                                                X_2.detach(),
                                                X_3.detach(),
                                            ]


        # Classification.
        output: Tensor =    self._classifier_(X_3)

        # If returning activations, do so.
        if return_activations: return output, activations

        # Otherwise, just return classification.
        return output

    # HELPERS ======================================================================================

    def _initialize_weights_(self) -> None:
        """# Initialize Weights."""
        # For each module in network...
        for m in self.modules():

            # If convolving layer...
            if isinstance(m, Conv2d):

                # Initialize with Kaiming normal distribution.
                kaiming_normal_(tensor = m.weight, mode = "fan_out", nonlinearity = "relu")

                # If bias exists...
                if m.bias is not None:

                    # Initialize bias to zero.
                    constant_(tensor = m.bias, val = 0)

            # If batch normalization layer...
            elif isinstance(m, BatchNorm2d):

                # Initialize weights to 1 and biases to 0.
                constant_(tensor = m.weight, val = 1)
                constant_(tensor = m.bias,   val = 0)

            # If fully connected layer...
            elif isinstance(m, Linear):

                # Initialize with normal distribution.
                normal_(tensor = m.weight, mean = 0, std = 0.01)
                constant_(tensor = m.bias, val = 0)

    @staticmethod
    def _make_layers_(
        config:         List[Union[int, str]],
        in_channels:    int,
        batch_norm:     bool =                  True,
        small_stem:     bool =                  False
    ) -> Sequential:
        """# Build VGG Feature Extraction Layers.

        ## Args:
            * config        (List): Layer configuration list.
            * in_channels   (int):  Number of input channels.
            * batch_norm    (bool): Use batch normalization. Defaults to True.
            * small_stem    (bool): Make layer for images of size 64 or less.

        ## Returns:
            * Sequential:   Feature extraction layers.
        """
        # Initialize list of layers.
        layers:     List[Module] =  []

        # Count max-pool markers so we can skip the last one.
        pool_total: int =           sum(1 for v in config if v == "M")
        pool_seen:  int =           0

        # For each entry in configuration...
        for v in config:

            # If max pooling marker...
            if v == "M":

                # Increment pool count.
                pool_seen += 1

                # # For small inputs, skip the final max-pool to preserve spatial resolution.
                if small_stem and pool_seen == pool_total: continue

                # Add max pooling layer.
                layers.append(MaxPool2d(kernel_size = 2, stride = 2))

            # Otherwise (convolutional layer)...
            else:

                # Add convolving layer.
                layers.append(
                    Conv2d(
                        in_channels =   in_channels,
                        out_channels =  v,
                        kernel_size =   3,
                        padding =       1
                    )
                )

                # If batch normalization is enabled...
                if batch_norm:

                    # Add batch normalization layer.
                    layers.append(BatchNorm2d(num_features = v))

                # Add activation.
                layers.append(ReLU(inplace = True))

                # Update channel count.
                in_channels = v

        # Provide feature extraction layers.
        return Sequential(*layers)
