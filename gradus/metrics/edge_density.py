"""# gradus.metrics.edge_density

Measurement of image's edge density using Canny edge detection.
"""

__all__ =   [
                "EdgeDensity",
                "edge_density",
            ]

from functools      import cached_property

from numpy.typing   import NDArray
from torch          import Tensor

class EdgeDensity():
    """# Edge Density Measurement"""

    def __init__(self,
        # Sample
        sample: Tensor, *,

        # Calculation parameters
        low:    int =   100,
        high:   int =   200
    ):
        """# Calculate Sample's Edge Density.

        ## Args:
            * sample    (Tensor):   Sample whose edge density is being measured.
            * low       (int):      Canny low threshold. Defaults to 100.
            * high      (int):      Canny high threshold. Defaults to 200.
        """
        # Define properties.
        self._sample_:          Tensor =    sample
        self._low_:             int =       low
        self._high_:            int =       high

    # PROPERTIES ===================================================================================

    @cached_property
    def density(self) -> float:
        """# Sample's Edge Density"""
        return self.edge_count / self.total_pixels

    @cached_property
    def edge_count(self) -> int:
        """# Number of Edges Detected in Image"""
        from numpy import count_nonzero

        return int(count_nonzero(self.edges))
    
    @cached_property
    def edges(self) -> NDArray:
        """# Image Edge Detection"""
        from cv2 import Canny

        return Canny(self.normalized_image, self._low_, self._high_)
    
    @cached_property
    def normalized_image(self) -> NDArray:
        """# Sample Normalized to NDArray"""
        from cv2    import COLOR_RGB2GRAY, cvtColor
        from numpy  import uint8

        # Convert sample (Tensor) to NDArray.
        image:  NDArray =   self._sample_.detach().cpu().numpy()

        # If this is a 3D image, transpose shape CHW -> HWC & convert RGB -> GRAY.
        if image.ndim == 3:    image = cvtColor(image.transpose(1, 2, 0), COLOR_RGB2GRAY)

        # Scale to [0, 255].
        if image.max() <= 1.0: image = (image * 255)

        # Convert values to uint8.
        return image.astype(uint8)
    
    @cached_property
    def total_pixels(self) -> int:
        """# Total Number of Pixels in Image"""
        return self.edges.size


# QUICK-ACCESS UTILITY =============================================================================

def edge_density(
    sample: Tensor, *,
    low:    int =   100,
    high:   int =   200
) -> float:
    """# Calculate Sample's Edge Density.

    ## Args:
        * sample    (Tensor):   Sample whose edge density is being measured.
        * low       (int):      Canny low threshold. Defaults to 100.
        * high      (int):      Canny high threshold. Defaults to 200.

    ## Returns:
        * float:    Fraction of edge pixels. Higher = more complex.
    """
    return EdgeDensity(**locals()).density