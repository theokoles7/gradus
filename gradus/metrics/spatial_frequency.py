"""# gradus.metrics.spatial_frequency

Measurement of image's spatial frequency.
"""

__all__ =   [
                "SpatialFrequency",
                "spatial_frequency",
            ]

from functools  import cached_property
from math       import sqrt

from torch      import Tensor

class SpatialFrequency():
    """# Spatial Frequency Measurement"""

    def __init__(self,
        sample: Tensor
    ):
        """# Calculate Sample's Spatial Frequency.

        ## Args:
            * sample    (Tensor):   Sample whose spatial frequency is being measured.
        """
        # Define properties.
        self._sample_:  Tensor =    sample

    # PROPERTIES ===================================================================================

    @cached_property
    def column_frequency(self) -> float:
        """# RMS of Vertical Differences"""
        # Calculate vertical difference.
        col_diff:   Tensor =    self.image[:, 1:] - self.image[:, :-1]

        # Calculate root mean squared of difference.
        return (col_diff ** 2).mean().item() ** 0.5
    
    @cached_property
    def frequency(self) -> float:
        """# Total Spatial Frequency of Image"""
        return sqrt(self.row_frequency ** 2 + self.column_frequency ** 2)

    @cached_property
    def image(self) -> Tensor:
        """# Sample Normalized for Float Precision"""
        # Copy sample.
        image:  Tensor =    self._sample_

        # If sample is 3D, average to one dimension.
        if image.dim() == 3: image = image.mean(dim = 0)

        # Convert to float for precision.
        return image.float()
    
    @cached_property
    def row_frequency(self) -> float:
        """# RMS of Horizontal Differences"""
        # Calculate horizontal difference.
        row_diff:   Tensor =    self.image[1:, :] - self.image[:-1, :]

        # Calculate root mean squared of difference.
        return (row_diff ** 2).mean().item() ** 0.5


# QUICK-ACCESS UTILITY =============================================================================

def spatial_frequency(
    sample: Tensor
) -> float:
    """# Calculate Sample's Spatial Frequency.

    ## Args:
        * sample    (Tensor):   Sample whose spatial frequency is being measured.

    ## Returns:
        * float:    Sample's overall (row + column) spatial frequency.
    """
    return SpatialFrequency(**locals()).frequency