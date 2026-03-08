"""# gradus.networks.autoencoder

Lightweight autoencoder for use in self-supervised analysis.
"""

__all__ = ["Autoencoder"]

from torch      import Tensor
from torch.nn   import BatchNorm2d, Conv2d, ConvTranspose2d, Module, ReLU, Sigmoid

class Autoencoder(Module):
    """# Auto-Encoder Network"""

    def __init__(self,
        channels:   int =   3
    ):
        """# Instantiate Auto-Encoder.

        ## Args:
            * channels  (int, optional):    Number of input channels. Defaults to 3 (RGB).
        """
        # Initialize module.
        super(Autoencoder, self).__init__()

        # Define encoder layers.
        self._enc_1_:   Conv2d =            Conv2d(
                                                in_channels =       channels,
                                                out_channels =      32,
                                                kernel_size =       3,
                                                stride =            2,
                                                padding =           1
                                            )
        self._enc_2_:   Conv2d =            Conv2d(
                                                in_channels =       32,
                                                out_channels =      64,
                                                kernel_size =       3,
                                                stride =            2,
                                                padding =           1
                                            )
        self._enc_3_:   Conv2d =            Conv2d(
                                                in_channels =       64,
                                                out_channels =      128,
                                                kernel_size =       3,
                                                stride =            2,
                                                padding =           1
                                            )

        # Define decoder layers.
        self._dec_1_:   ConvTranspose2d =   ConvTranspose2d(
                                                in_channels =       128,
                                                out_channels =      64,
                                                kernel_size =       3,
                                                stride =            2,
                                                padding =           1,
                                                output_padding =    1
                                            )
        self._dec_2_:   ConvTranspose2d =   ConvTranspose2d(
                                                in_channels =       64,
                                                out_channels =      32,
                                                kernel_size =       3,
                                                stride =            2,
                                                padding =           1,
                                                output_padding =    1
                                            )
        self._dec_3_:   ConvTranspose2d =   ConvTranspose2d(
                                                in_channels =       32,
                                                out_channels =      channels,
                                                kernel_size =       3,
                                                stride =            2,
                                                padding =           1,
                                                output_padding =    1
                                            )
        
        # Define batch normalization layers.
        self._bn_1_:    BatchNorm2d =       BatchNorm2d(num_features =  32)
        self._bn_2_:    BatchNorm2d =       BatchNorm2d(num_features =  64)
        self._bn_3_:    BatchNorm2d =       BatchNorm2d(num_features = 128)
        self._bn_4_:    BatchNorm2d =       BatchNorm2d(num_features =  64)
        self._bn_5_:    BatchNorm2d =       BatchNorm2d(num_features =  32)

        # Define activations.
        self._relu_:    ReLU =              ReLU()
        self._sigmoid_: Sigmoid =           Sigmoid()

    # METHODS ======================================================================================

    def forward(self,
        X:  Tensor
    ) -> Tensor:
        """# Forward Pass Through Auto-Encoder.

        ## Args:
            * X (Tensor):   Input image tensor, shaped (N, C, H, W).

        ## Returns:
            * Tensor:   Reconstructed image tensor, shaped (N, C, H, W).
        """
        # Encode.
        Z:      Tensor =    self._relu_(self._bn_1_(self._enc_1_(X)))
        Z:      Tensor =    self._relu_(self._bn_2_(self._enc_2_(Z)))
        Z:      Tensor =    self._relu_(self._bn_3_(self._enc_3_(Z)))

        # Decode.
        X_hat:  Tensor =    self._relu_(self._bn_4_(self._dec_1_(Z)))
        X_hat:  Tensor =    self._relu_(self._bn_5_(self._dec_2_(X_hat)))
        X_hat:  Tensor =    self._sigmoid_(self._dec_3_(X_hat))

        # Provide reconstructed image, cropped to original size.
        return X_hat[:, :, :X.shape[2], :X.shape[3]]