"""# gradus.curricula.metrics.complexity.wavelet_energy.test

Test suite for wavelet energy complexity metric.
"""

from gradus.curricula.metrics.complexity.wavelet_energy.__base__    import WaveletEnergy

# TESTS ============================================================================================

class TestWaveletEnergyType():
    """# Return Type Tests for Wavelet Energy Metric
    
    Wavelet energy should always be a float.
    """

    def test_return_type(self, test_samples) -> None:
        """# Wavelet energy should return a float."""
        assert  isinstance(WaveletEnergy(test_samples["solid_black"]).value, float), \
                f"Wavelet energy value expected to be of type float."


class TestWaveletEnergySolid():
    """# Solid Color Image Tests
    
    A solid, single-color image has no detail at any decomposition level, so wavelet energy should 
    be zero.
    """

    def test_solid_black_is_zero(self, test_samples) -> None:
        """Solid black image should have zero wavelet energy."""
        assert  WaveletEnergy(test_samples["solid_black"]).value == 0.0,    \
                f"Wavelet energy of solid black image should be 0."

    def test_solid_white_is_zero(self, test_samples) -> None:
        """Solid white image should have zero wavelet energy."""
        assert  WaveletEnergy(test_samples["solid_white"]).value == 0.0,    \
                f"Wavelet energy of solid white image should be 0."

    def test_solid_color_is_zero(self, test_samples) -> None:
        """Any solid color image should have zero wavelet energy."""
        assert  WaveletEnergy(test_samples["solid_red"]).value == 0.0,  \
                f"Wavelet energy of solid red image should be 0."


class TestWaveletEnergyCheckerboard():
    """# Checkerboard Image Tests
    
    A checkerboard image has detail at multiple decomposition levels, so wavelet energy should be 
    positive and greater than that of a solid image.
    """

    def test_checkerboard_is_positive(self, test_samples) -> None:
        """Checkerboard image should have positive wavelet energy."""
        assert  WaveletEnergy(test_samples["checker_bw_2x2"]).value > 0.0,  \
                f"Wavelet energy of checker image is no longer positive."

    def test_checkerboard_greater_than_solid(self, test_samples) -> None:
        """Checkerboard image should have greater wavelet energy than any solid image."""
        assert  WaveletEnergy(test_samples["checker_bw_2x2"]).value >   \
                WaveletEnergy(test_samples["solid_black"]).value,       \
                f"Wavelet energy of checker image should exceed that of a solid image."


class TestWaveletEnergyBounded():
    """# Boundary Tests
    
    Wavelet energy is a sum of squared coefficients, so it should always be non-negative.
    """

    def test_lower_bound_solid(self, test_samples) -> None:
        """Wavelet energy should never be negative."""
        assert  WaveletEnergy(test_samples["solid_black"]).value >= 0.0,    \
                f"Wavelet energy should never be negative."

    def test_lower_bound_checker(self, test_samples) -> None:
        """Wavelet energy should never be negative."""
        assert  WaveletEnergy(test_samples["checker_bw_2x2"]).value >= 0.0, \
                f"Wavelet energy should never be negative."


class TestWaveletEnergyGrayscale():
    """# Grayscale Input Tests"""

    def test_grayscale_returns_float(self, test_samples) -> None:
        """Grayscale (2D) input should still return a float."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the wavelet energy of a 1-channel image is still a float.
        assert  isinstance(WaveletEnergy(gray).value, float),   \
                f"Wavelet energy of a 1-channel image should still be a float."

    def test_grayscale_solid_is_zero(self, test_samples) -> None:
        """Grayscale solid image should have zero wavelet energy."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the wavelet energy of a solid, 1-channel image is zero.
        assert  WaveletEnergy(gray).value == 0.0,   \
                f"Wavelet energy of a solid, 1-channel image should be zero."