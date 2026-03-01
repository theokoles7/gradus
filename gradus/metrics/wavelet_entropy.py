"""# gradus.metrics.wavelet_entropy

Measurement of wavelet entropy of an image.
"""

__all__ =   [
                "WaveletEntropy",
                "wavelet_entropy",
            ]

from functools      import cached_property
from typing         import List

from numpy          import log2
from numpy.typing   import NDArray
from torch          import Tensor

class WaveletEntropy():
    """# Wavelet Entropy Measurement"""

    def __init__(self,
        # Sample
        sample:     Tensor, *,

        # Calculation parameters
        wavelet:    str =   "db2",
        level:      int =   None
    ):
        """# Calculate Sample's Wavelet Entropy.

        ## Args:
            * sample    (Tensor):   Sample whose wavelet entropy is being measured.
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
    def energy_distribution(self) -> List[float]:
        """# Level-Wise Energy Distribution"""
        # If total energy has been determined to be zero, then distribution will only be zeros.
        if self.total_energy < 1e-6: return [0.0] * len(self.level_energies)

        # Otherwise, normalized distribution.
        return [e / self.total_energy for e in self.level_energies]
    
    @cached_property
    def entropy(self) -> float:
        """# Shannon Entropy of Calculated Energies"""
        return  -sum(
                    p * log2(p)
                    for p in self.energy_distribution
                    if p > 0
                )

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
    def normalized_entropy(self) -> float:
        """# Entropy Normalized by Maximum"""
        # Calculate maximum entropy.
        max_entropy: float = log2(len(self.level_energies)) if len(self.level_energies) > 1 else 1.0

        # Calculate normalized entropy.
        return self.entropy / max_entropy if max_entropy > 0 else 0.0
    
    @cached_property
    def total_energy(self) -> float:
        """# Sample's Total Wavelet Energy"""
        return sum(self.level_energies)


# QUICK-ACCESS UTILITY =============================================================================

def wavelet_entropy(
    # Sample
    sample:     Tensor, *,

    # Calculation parameters
    wavelet:    str =   "db2",
    level:      int =   None
) -> float:
    """# Calculate Sample's Wavelet Entropy.

    ## Args:
        * sample    (Tensor):   Sample whose wavelet entropy is being measured.
        * wavelet   (str):      Wavelet family to use. Defaults to "db2".
        * level     (int):      Decomposition level. Defaults to None (maximum possible).

    ## Returns:
        * float:    Sample's total wavelet entropy.
    """
    return WaveletEntropy(**locals()).normalized_entropy