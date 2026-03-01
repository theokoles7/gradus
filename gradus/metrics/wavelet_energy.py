"""# gradus.metrics.wavelet_energy

Measurement of wavelet energy of an image.
"""

__all__ =   [
                "WaveletEnergy",
                "wavelet_energy",
            ]

from functools      import cached_property
from typing         import List

from numpy.typing   import NDArray
from torch          import Tensor

class WaveletEnergy():
    """# Wavelet Energy Measurement"""

    def __init__(self,
        # Sample
        sample:     Tensor, *,

        # Calculation parameters
        wavelet:    str =   "db2",
        level:      int =   None
    ):
        """# Calculate Sample's Wavelet Energy.

        ## Args:
            * sample    (Tensor):   Sample whose wavelet energy is being measured.
            * wavelet   (str):      Wavelet family to use. Defaults to "db2".
            * level     (int):      Decomposition level. Defaults to None (maximum possible).
        """
        # Define properties.
        self._sample_:  Tensor =    sample
        self._wavelet_: str =       wavelet
        self._level_:   int =       level

    # PROPERTIES ===================================================================================

    @cached_property
    def coefficients(self) -> NDArray:
        """# Wavelet Decomposition Coefficients"""
        from pywt import wavedec2

        return wavedec2(self.image, wavelet = self._wavelet_, level = self._level_)

    @cached_property
    def image(self) -> NDArray:
        """# Sample Normalized to NDArray"""
        # Copy sample.
        image:  Tensor =    self._sample_

        # If sample is 3D, average to one dimension.
        if image.dim() == 3: image = image.mean(dim = 0)

        # Convert to NDArray.
        return image.detach().cpu().numpy()
    
    @cached_property
    def level_energies(self) -> List[float]:
        """# Level-Wise Wavelet Energies"""
        from numpy import sum as np_sum

        return  [
                    sum(float(np_sum(d ** 2)) for d in detail_coeffs)
                    for detail_coeffs in self.coefficients[1:]
                ]
    
    @cached_property
    def total_energy(self) -> float:
        """# Sample's Total Wavelet Energy"""
        # Sum energies.
        energy: float = sum(self.level_energies)

        # Provide total energy.
        return 0.0 if energy < 1e-6 else energy


# QUICK-ACCESS UTILITY =============================================================================

def wavelet_energy(
    # Sample
    sample:     Tensor, *,

    # Calculation parameters
    wavelet:    str =   "db2",
    level:      int =   None
) -> float:
    """# Calculate Sample's Wavelet Energy.

    ## Args:
        * sample    (Tensor):   Sample whose wavelet energy is being measured.
        * wavelet   (str):      Wavelet family to use. Defaults to "db2".
        * level     (int):      Decomposition level. Defaults to None (maximum possible).

    ## Returns:
        * float:    Sample's total wavelet energy.
    """
    return WaveletEnergy(**locals()).total_energy