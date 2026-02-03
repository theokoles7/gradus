"""# gradus.metrics.wavelet_energy

Wavelet energy metric implementation.
"""

__all__ =   [
                "wavelet_energy",
                "wavelet_entropy",
            ]

from typing                     import Dict, List, Optional, Union

from numpy                      import asarray, sum as np_sum
from numpy.typing               import NDArray
from pywt                       import wavedec2, Wavelet

from gradus.metrics.types       import ArrayLike, ChannelMode, WaveletEnergyResult, \
                                       WaveletEntropyResult
from gradus.metrics.utilities   import determine_max_decomposition_level,           \
                                       shannon_entropy_from_energies, to_2d_image,  \
                                       to_float_image


def wavelet_energy(
    image:          ArrayLike,
    *,
    wavelet:        str =           "db2",
    level:          Optional[int] = None,
    mode:           str =           "periodization",
    channel_mode:   ChannelMode =   "luminance",
    normalize:      bool =          False,
    per_level:      bool =          True
) -> Union[WaveletEnergyResult, Dict[str, WaveletEnergyResult]]:
    """# Calculate Wavelet Energy of Image.

    ## Args:
        * image         (ArrayLike):    2D (H,W) or 3D (H,W,C) numpy array.
        * wavelet       (str):          Wavelet kernel (e.g., `db1`, `db2`, `haar`, `sym4`). 
                                        Defaults to "db2".
        * level         (int | None):   Decomposition level. Defaults to None.
        * mode          (str):          Signal extension mode (commonly `periodization`, 
                                        `symmetric`, etc.). Defaults to "periodization".
        * channel_mode  (ChannelMode):  `grayscale` expects already 2D image, or averages channels 
                                        if 3D. `luminance` uses ITU-R BT.601 luma weights for 
                                        RGB-like inputs. `per_channel` handles channels separately.
        * normalize     (bool):         If True, returns energy divided by number of pixels (useful 
                                        for comparing sizes). Defaults to False.
        * per_level     (bool):         If True, includes detail energy by level. Defaults to True.

    ## Returns:
        * WaveletEnergyResult | Dict[str, WaveletEnergyResult]: Wavelet energy calculation (by level).
    """
    # Ensure image is NDArray of 64-bit float pixels.
    image:          NDArray =       to_float_image(asarray(image))

    # If calculating per channel...
    if channel_mode == "per_channel":

        # If image is not RGB image...
        if image.ndim != 3: raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

        # Recurse over each channel.
        return  {f"channel_{c}":    wavelet_energy(
                                        image[..., c],
                                        wavelet =       wavelet,
                                        level =         level,
                                        mode =          mode,
                                        channel_mode =  "grayscale",
                                        normalize =     normalize,
                                        per_level =     per_level
                                    )
                for c in range(image.shape[2])}
    
    # Otherwise, convert image to 2D.
    image:          NDArray =       to_2d_image(image, channel_mode = channel_mode)

    # Initialize wavelet.
    w:              Wavelet =       Wavelet(wavelet)

    # Determine maximum decomposition level.
    level:          int =           determine_max_decomposition_level(
                                        image =     image,
                                        wavelet =   w,
                                        level =     level
                                    )

    # Compute coefficients.
    coeffs:         List[NDArray] = wavedec2(data = image, wavelet = w, level = level, mode = mode)

    # Extract coefficients.
    cA:             NDArray =       coeffs[0]
    detail:         List[NDArray] = coeffs[1:]

    # Approximate energy.
    approx_energy:  float =         float(np_sum(cA * cA))

    # Initialize energy list.
    detail_energy:  List[float] =   []
    detail_total:   float =         0.0

    # For each coefficient...
    for (cH, cV, cD) in detail:

        # Compute energy for each detail coefficient.
        level_energy:   float = float(np_sum(cH * cH) + np_sum(cV * cV) + np_sum(cD * cD))

        # Append to list.
        detail_energy.append(level_energy)

        # Update total energy.
        detail_total += level_energy

    # Update total energy with approximate energy.
    energy_total = approx_energy + detail_total

    # If normalizing...
    if normalize:

        # Determine number of pixels.
        num_pixels:    float =  float(image.shape[0] * image.shape[1])

        # Normalize energies.
        approx_energy   /=  num_pixels
        energy_total    /=  num_pixels
        detail_energy   =   [e / num_pixels for e in detail_energy]

    # Provide wavelet energy calculation.
    return WaveletEnergyResult(
        total_energy =                  energy_total,
        approximate_energy =            approx_energy,
        detail_energy_total =           energy_total - approx_energy,
        detail_energy_by_level =        tuple(detail_energy) if per_level else (),
        approximate_coefficient_shape = tuple(cA.shape),
        levels =                        level,
        wavelet =                       w.name,
        mode =                          mode
    )


def wavelet_entropy(
    image:                  ArrayLike,
    *,
    wavelet:                str =           "db2",
    level:                  Optional[int] = None,
    mode:                   str =           "periodization",
    channel_mode:           ChannelMode =   "luminance",
    include_approximate:    bool =          False,
) -> Union[WaveletEntropyResult, Dict[str, WaveletEntropyResult]]:
    """# Calculate Wavelet Entropy of Image.

    ## Args:
        * image         (ArrayLike):    2D (H,W) or 3D (H,W,C) numpy array.
        * wavelet       (str):          Wavelet kernel (e.g., `db1`, `db2`, `haar`, `sym4`). 
                                        Defaults to "db2".
        * level         (int | None):   Decomposition level. Defaults to None.
        * mode          (str):          Signal extension mode (commonly `periodization`, 
                                        `symmetric`, etc.). Defaults to "periodization".
        * channel_mode  (ChannelMode):  `grayscale` expects already 2D image, or averages channels 
                                        if 3D. `luminance` uses ITU-R BT.601 luma weights for 
                                        RGB-like inputs. `per_channel` handles channels separately.
        * normalize     (bool):         If True, returns normalized entropy. Defaults to True.

    ## Returns:
        * WaveletEntropyResult | Dict[str, WaveletEntropyResult]: Wavelet entropy calculation (by level).
    """
    # Ensure image is NDArray of 64-bit float pixels.
    image:          NDArray =       to_float_image(asarray(image))

    # If calculating per channel...
    if channel_mode == "per_channel":

        # If image is not RGB image...
        if image.ndim != 3: raise ValueError(f"Expected 3-channel image, got shape {image.shape}")

        # Recurse over each channel.
        return  {f"channel_{c}":    wavelet_entropy(
                                        image[..., c],
                                        wavelet =               wavelet,
                                        level =                 level,
                                        mode =                  mode,
                                        channel_mode =          "grayscale",
                                        include_approximate =   include_approximate
                                    )
                for c in range(image.shape[2])}
    
    # Otherwise, convert image to 2D.
    image:          NDArray =       to_2d_image(image, channel_mode = channel_mode)

    # Initialize wavelet.
    w:              Wavelet =       Wavelet(wavelet)

    # Determine maximum decomposition level.
    level:          int =           determine_max_decomposition_level(
                                        image =     image,
                                        wavelet =   w,
                                        level =     level
                                    )
    
    # Compute coefficients.
    coeffs:         List[NDArray] = wavedec2(data = image, wavelet = w, level = level, mode = mode)

    # Extract coefficients.
    cA:             NDArray =       coeffs[0]
    detail:         List[NDArray] = coeffs[1:]

    # Initialize energies list.
    energies:       List[float] =   []

    # If including approximate coefficients, append approximate energy.
    if include_approximate: energies.append(float(np_sum(cA * cA)))

    # For each detail coefficient...
    for (cH, cV, cD) in detail:

        # Append detail energy.
        energies.append(float(np_sum(cH * cH) + np_sum(cV * cV) + np_sum(cD * cD)))

    # Compute Shannon entropy from energies.
    entropy, normalized_entropy, probabilities = shannon_entropy_from_energies(energies)

    # Provide wavelet entropy calculation.
    return WaveletEntropyResult(
        entropy =              entropy,
        normalized_entropy =   normalized_entropy,
        probabilities =        probabilities,
        energies =             tuple(energies),
        levels =               level,
        wavelet =              w.name,
        mode =                 mode
    )