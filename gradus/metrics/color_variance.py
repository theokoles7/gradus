"""# gradus.metrics.color_variance

Measurement of channel-wise color variance of an image sample.
"""

__all__ =   [
                "ColorVariance",
                "color_variance",
            ]

from functools  import cached_property
from typing     import List

from torch      import Tensor

class ColorVariance():
    """# Color Variance Measurement"""

    def __init__(self,
        sample: Tensor
    ):
        """# Calculate Sample's Color Variance.

        ## Args:
            * sample    (Tensor):   Sample whose color variance is being measured.
        """
        # Define properties.
        self._sample_:  Tensor =    sample

    # PROPERTIES ===================================================================================

    @cached_property
    def channel_variances(self) -> List[float]:
        """# Channel-Wise Pixel Variances"""
        # If sample image is gray-scale, simply calculate single-channel variance.
        if self._sample_.dim() == 2: return [self._sample_.var().item()]

        # Otherwise, calculate variance of each channel (RGB).
        return [self._sample_[c].var().item() for c in range(self._sample_.shape[0])]
    
    @cached_property
    def mean_variance(self) -> float:
        """# Mean of Channel-Wise Variances"""
        return sum(self.channel_variances) / len(self.channel_variances)
        

# QUICK-ACCESS UTILITY =============================================================================

def color_variance(
    sample: Tensor
) -> float:
    """# Calculate Sample's Color Variance.

    ## Args:
        * sample    (Tensor):   Sample whose color variance is being measured.

    ## Returns:
        * float:    Mean color variance of sample.
    """
    return ColorVariance(**locals()).mean_variance