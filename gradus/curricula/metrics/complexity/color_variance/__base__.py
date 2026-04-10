"""# gradus.curricula.metrics.complexity.color_variance.base

Measurement of channel-wise color variance of an image sample.
"""

__all__ = ["ColorVariance"]

from functools                                                      import cached_property
from typing                                                         import List, override, Union

from torch                                                          import device as t_device, Tensor

from gradus.curricula.metrics.complexity.color_variance.__args__    import ColorVarianceConfig
from gradus.registration                                            import register_metric
from gradus.utilities                                               import determine_device

@register_metric(
    id =        "color-variance",
    config =    ColorVarianceConfig,
    tags =      ["complexity"]
)
class ColorVariance():
    """# Color Variance Measurement"""

    def __init__(self,
        sample: Tensor,
        device: Union[str, t_device] =  "auto"
    ):
        """# Calculate Sample's Color Variance.

        ## Args:
            * sample    (Tensor):       Sample whose color variance is being measured.
            * device    (str | device): Torch computation device. Defaults to "auto".
        """
        # Define properties.
        self._device_:  t_device =  determine_device(device)
        self._sample_:  Tensor =    sample.to(self._device_)

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
    
    @override
    @cached_property
    def value(self) -> float:
        """# Mean of Channel-Wise Variances"""
        return self.mean_variance