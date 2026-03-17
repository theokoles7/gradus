"""# gradus.metrics.complexity.compression_ratio.base

Measurement of image's JPEG compression ratio.
"""

__all__ =   [
                "CompressionRatio",
                "compression_ratio",
            ]

from functools      import cached_property

from numpy.typing   import NDArray
from torch          import Tensor

class CompressionRatio():
    """# Compression Ratio Measurement"""

    def __init__(self,
        # Sample
        sample:     Tensor, *,

        # Calculation parameters
        quality:    int =   95
    ):
        """# Calculate Sample's Compression Ratio.

        ## Args:
            * sample    (Tensor):   Sample whose compression ratio is being measured.
            * quality   (int):      JPEG compression quality (1-100). Defaults to 95.
        """
        # Define properties.
        self._sample_:  Tensor =    sample
        self._quality_: int =       quality

    # PROPERTIES ===================================================================================

    @cached_property
    def compressed_size(self) -> int:
        """# Size of JPEG-Encoded Image (Bytes)"""
        from cv2    import imencode, IMWRITE_JPEG_QUALITY

        # Encode original image to JPEG.
        _, encoded =    imencode(
                            ext =       ".jpg",
                            img =       self.normalized_image,
                            params =    [IMWRITE_JPEG_QUALITY, self._quality_]
                        )

        # Provide size of encoded image.
        return len(encoded)
    
    @cached_property
    def normalized_image(self) -> NDArray:
        """# Sample Normalized to NDArray"""
        from cv2    import cvtColor, COLOR_RGB2BGR
        from numpy  import uint8

        # Convert sample (Tensor) to NDArray.
        image:  NDArray =   self._sample_.detach().cpu().numpy()

        # If this is a 3D image, rranspose shape CHW -> HWC & RGB -> BGR
        if image.ndim == 3:    image = cvtColor(image.transpose(1, 2, 0), COLOR_RGB2BGR)

        # Scale to [0, 255].
        if image.max() <= 1.0: image = (image * 255)

        # Convert values to uint8.
        return image.astype(uint8)
    
    @cached_property
    def original_size(self) -> int:
        """# Size of Original Image (Bytes)"""
        return self.normalized_image.nbytes
    
    @cached_property
    def ratio(self) -> float:
        """# Compression Ratio (Original / Compressed)"""
        return self.original_size / self.compressed_size
    

# QUICK-ACCESS UTILITY =============================================================================

from gradus.metrics.complexity.compression_ratio.__args__   import CompressionRatioConfig
from gradus.registration                                    import register_metric

@register_metric(
    id =        "compression-ratio",
    cls =       CompressionRatio,
    config =    CompressionRatioConfig,
    tags =      ["complexity"]
)
def compression_ratio(
    sample:     Tensor, *,
    quality:    int =   95
) -> float:
    """# Calculate Sample's Compression Ratio.

    ## Args:
        * sample    (Tensor):   Sample whose compression ratio is being measured.
        * quality   (int):      JPEG compression quality (1-100). Defaults to 95.

    ## Returns:
        * float:    Sample's compression ratio.
    """
    return CompressionRatio(**locals()).ratio