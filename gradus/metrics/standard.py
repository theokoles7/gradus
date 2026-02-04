"""# gradus.metrics.standard

Foundational image complexity metrics.
"""

__all__ =   [
                "color_variance",
                "compression_ratio",
                "edge_density",
                "spatial_frequency"
            ]

from typing         import Tuple

from cv2            import Canny, COLOR_RGB2GRAY, cvtColor, imencode, IMWRITE_JPEG_QUALITY
from cv2.typing     import MatLike
from numpy          import count_nonzero, diff, mean as np_mean, sqrt as np_sqrt
from numpy.typing   import NDArray


def color_variance(
    image:  NDArray
) -> float:
    """# Compute Color Variance of Image.

    Higher values = more color diversity = more complex.

    ## Args:
        * image (NDArray):  Image being evaluated.

    ## Returns:
        * float:    Image's color variance.
    """
    # If image is not RGB, simply return pixel variance.
    if len(image.shape) != 3: return float(image.var())

    # Compute variance across color channels and return mean variance.
    return float(image.var(axis = (0, 1)).mean())


def compression_ratio(
    image:  NDArray,
) -> float:
    """# Compute Compression Ratio of Image.

    Lower ratio = less compressible = more complex.

    ## Args:
        * image (NDArray):  Image being evaluated.

    ## Returns:
        * float:    Image's compression ratio.
    """
    # Encode image as JPEG to assess compressibility.
    _, encoded_image =          imencode(
                                    ext =       ".jpg",
                                    img =       image,
                                    params =    [IMWRITE_JPEG_QUALITY, 90]
                                )

    # Compute sizes.
    original_size:      int =   image.nbytes
    compressed_size:    int =   encoded_image.nbytes

    # Compute and return compression ratio.
    return original_size / compressed_size


def edge_density(
    image:          NDArray,
    threshold_min:  float =     100.0,
    threshold_max:  float =     200.0
) -> float:
    """# Compute Edge Density of Image.

    Higher values = more edges = more complex.

    ## Args:
        * image         (NDArray):  Image being evaluated.
        * threshold_min (float):    Minimum threshold for Canny edge detection. Defaults to 100.
        * threshold_max (float):    Maximum threshold for Canny edge detection. Defaults to 200.

    ## Returns:
        * float:    Image's edge density.
    """
    # If image is RGB, convert it to grayscale.
    if len(image.shape) == 3: image: MatLike = cvtColor(src = image, code = COLOR_RGB2GRAY)

    # Use Canny edge detection.
    edges:  MatLike =   Canny(
                            image =         image,
                            threshold1 =    threshold_min,
                            threshold2 =    threshold_max
                        )
    
    # Compute density (fraction of edge pixels).
    return count_nonzero(a = edges) / edges.size


def spatial_frequency(
    image:  NDArray
) -> Tuple[float, float, float]:
    """# Compute Spatial Frequency of Image.

    Measures high-frequency content (texture complexity).

    ## Args:
        * image (NDArray):  Image being evaluated.

    ## Returns:
        * float:    Row Frequency.
        * float:    Column Frequency.
        * float:    Overall Frequency.
    """
    # Convert image to grayscale if needed.
    if len(image.shape) == 3: image = cvtColor(src = image, code = COLOR_RGB2GRAY).astype(float)

    # Compute row and column differences.
    RF: float = np_sqrt(np_mean(diff(image, axis = 0) ** 2))
    CF: float = np_sqrt(np_mean(diff(image, axis = 1) ** 2))

    # Compute overall frequency.
    OF: float = np_sqrt(RF ** 2 + CF ** 2)

    # Provide spatial frequency calculations.
    return RF, CF, OF