"""# gradus.curricula.metrics.complexity.object_count.base

Count of objects within an image using Canny edge detection.
"""

__all__ = ["ObjectCount"]

from functools                                                  import cached_property
from typing                                                     import List, override, Union

from numpy.typing                                               import NDArray
from torch                                                      import device as t_device, Tensor

from gradus.curricula.metrics.complexity.object_count.__args__  import ObjectCountConfig
from gradus.registration                                        import register_metric
from gradus.utilities                                           import determine_device

@register_metric(
    id =        "object-count",
    config =    ObjectCountConfig,
    tags =      ["complexity"]
)
class ObjectCount():
    """# Object Count Measurement"""

    def __init__(self,
        # Sampel
        sample: Tensor, *,

        # Calculation parameters
        low:    int =   100,
        high:   int =   200
    ):
        """# Calculate Sample's Object Count.

        ## Args:
            * sample    (Tensor):   Sample whose object conut is being measured.
            * low       (int):      Canny low threshold. Defaults to 100.
            * high      (int):      Canny high threshold. Defaults to 200.
        """
        # Define properties.
        self._sample_:  Tensor =    sample
        self._low_:     int =       low
        self._high_:    int =       high

    # PROPERTIES ===================================================================================
    
    @cached_property
    def edges(self) -> NDArray:
        """# Image Edge Detection"""
        from cv2 import Canny

        return Canny(self.normalized_image, self._low_, self._high_)
    
    @cached_property
    def edges_dilated(self) -> NDArray:
        """"""
        from cv2    import dilate
        from numpy  import ones, uint8

        # Initialize kernel.
        kernel: NDArray =   ones((3, 3), uint8)

        # Dilate edges.
        return dilate(src = self.edges, kernel = kernel, iterations = 1)
    
    @cached_property
    def normalized_image(self) -> NDArray:
        """# Sample Normalized to NDArray"""
        from cv2    import COLOR_RGB2GRAY, cvtColor
        from numpy  import uint8

        # Convert sample (Tensor) to NDArray.
        image:  NDArray =   self._sample_.detach().cpu().numpy()

        # If 2D image...
        if image.ndim == 3:

            # If single channel...
            if image.shape[0] == 1:

                # Squeeze channel dimension.
                image = image.squeeze(0)

            # Otherwise, transpose 3-channel.
            else: image = cvtColor(image.transpose(1, 2, 0), COLOR_RGB2GRAY)

        # Scale to [0, 255].
        if image.max() <= 1.0: image = (image * 255)

        # Convert values to uint8.
        return image.astype(uint8)
    
    @cached_property
    def num_labels(self) -> int:
        """# Number of Objects within Image"""
        from cv2    import connectedComponents

        # Count objects.
        labels, _ = connectedComponents(image = self.edges_dilated)

        # Subtract 1 for background.
        return max(labels - 1, 0)
    
    @override
    @cached_property
    def value(self) -> int:
        """# Number of Objects within Image"""
        return self.num_labels