"""# gradus.metrics.utilities

Image complexity quantification & metrics utilities.
"""

__all__ =   [
                "determine_max_decomposition_level",
                "shannon_entropy_from_energies",
                "to_2d_image",
                "to_float_image",
            ]

from typing                 import Iterable, Optional, Tuple

from numpy                  import asarray, clip, floating, issubdtype, float64 as np_float64, log as np_log
from numpy.typing           import NDArray
from pywt                   import dwt_max_level, Wavelet

from gradus.metrics.types   import ChannelMode

def determine_max_decomposition_level(
    image:      NDArray,
    wavelet:    Wavelet,
    level:      Optional[int]
) -> int:
    """# Determine Maximum Decomposition Level of Image.

    ## Args:
        * image     (NDArray):      Image being evaluated.
        * wavelet   (Wavelet):      Wavelet being used in decomposition.
        * level     (int | None):   Desired decomposition level.

    ## Returns:
        * int:  Desired decomposition level if in range[1, max_level].
    """
    # Determine maximum decomposition level for given wavelet & image.
    max_level:  int =   dwt_max_level(min(image.shape), wavelet.dec_len)

    # If desired level is provided, simply return max level calculated.
    if level is None:   return max_level

    # Otherwise, if requested level is outside of appropriate range...
    if level not in range(1, max_level + 1):

        # Report error.
        raise ValueError(f"Level = {level} outside of range 1-{max_level}")
    
    # Otherwise, requested level is appropriate.
    return level


def shannon_entropy_from_energies(
    energies:   Iterable[float],
    *,
    epsilon:    float =         1e-12,
    normalize:  bool =          True
) -> Tuple[float, float, Tuple[float, ...]]:
    """# Compute Shannon Entropy from List of Energies.

    ## Args:
        * energies  (Iterable[float]):  List of wavelet energies from which Shannon entropy will be 
                                        computed.
        * epsilon   (float):            Epislon value to prevent division by zero. Defaults to 
                                        1e-12.
        * normalize (bool):             Normalize entropy values.. Defaults to True.

    ## Returns:
            * entropy               (float):                Shannon entropy.
            * normalized_entropy    (float):                Normalized Shannon entropy.
            * probabilities         (Tuple[float, ...]):    Probabilities used in entropy 
                                                            calculation.
    """
    # Convert energies to NDArray.
    e:          NDArray =   asarray(list(energies), dtype = np_float64)

    # Compute total energy.
    total:      float =     float(e.sum())

    # If total energy is zero, return zero entropy.
    if total <= 0.0: return 0.0, 0.0, tuple(0.0 for _ in range(len(e)))

    # Compute probabilities.
    p:          NDArray =   clip(e / (total), epsilon, 1.0)

    # Compute Shannon entropy.
    entropy:    float =     float(-(p * np_log(p)).sum())

    # Normalize entropy.
    normalized: float =     entropy / np_log(len(p)) if normalize and len(p) > 1 else entropy

    # Return entropy values.
    return entropy, normalized, tuple(float(prob) for prob in p)


def to_2d_image(
    image:          NDArray,
    channel_mode:   ChannelMode
) -> NDArray:
    """# Convert Image to 2D.

    Convert HxW or HxWxC image to 2D depending on `channel_mode`:
        * `grayscale`:      Expects already 2D image, or averages channels if 3D.
        * `luminance`:      Uses ITU-R BT.601 luma weights for RGB-like inputs.
        * `per_channel`:    Caller handles channels separately.

    ## Returns:
        * NDArray:  Normalized 2D image.
    """
    # If image is already 2D, simply return.
    if image.ndim == 2: return image

    # If image is also not 3D, report complication.
    if image.ndim != 3: raise ValueError(f"Expected 2D/3D image, got shape {image.shape}")

    # Match channel mode.
    match channel_mode:

        # Gray-scale
        case "grayscale":           return image.mean(axis = 2)

        # Luminance
        case "luminance":

            # If there are not at least 3 channels, assume 1.
            if image.shape[2] < 3:  return image.mean(axis = 2)

            # Otherwise, extract channels.
            R, G, B = image[..., 0], image[..., 1], image[..., 2]

            # Convert values.
            return 0.299 * R + 0.587 * G + 0.114 * B
        
        # By-channel should not be handled here.
        case "per_channel": raise   NotImplementedError(
                                        "Per-Channel conversion should be handled elsewhere"
                                    )


def to_float_image(
    image:  NDArray
) -> NDArray:
    """# Convert Image to Float Pixels.

    ## Args:
        * image (NDArray):  Image being converted.

    ## Returns:
        * NDArray:  Converted image, with float pixel values.
    """
    # If image is already floating type, ensure it's 64-bit without changing scale.
    if issubdtype(image.dtype, floating): return image.astype(np_float64, copy = False)

    # Otherwise, simply convert to 64-bit float.
    return image.astype(np_float64)