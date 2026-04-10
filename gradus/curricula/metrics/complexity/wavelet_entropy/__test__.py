"""# gradus.curricula.metrics.complexity.wavelet_entropy.test

Test suite for wavelet entropy complexity metric.
"""

from gradus.curricula.metrics.complexity.wavelet_entropy.__base__   import WaveletEntropy

# TESTS ============================================================================================

class TestWaveletEntropyType():
    """# Return Type Tests for Wavelet Entropy Metric
    
    Wavelet entropy should always be a float.
    """

    def test_return_type(self, test_samples) -> None:
        """# Wavelet entropy should return a float."""
        assert  isinstance(WaveletEntropy(test_samples["solid_black"]).value, float), \
                f"Wavelet entropy value expected to be of type float."


class TestWaveletEntropySolid():
    """# Solid Color Image Tests
    
    A solid, single-color image has no detail at any decomposition level, so wavelet entropy should 
    be zero.
    """

    def test_solid_black_is_zero(self, test_samples) -> None:
        """Solid black image should have zero wavelet entropy."""
        assert  WaveletEntropy(test_samples["solid_black"]).value == 0.0,   \
                f"Wavelet entropy of solid black image should be 0."

    def test_solid_white_is_zero(self, test_samples) -> None:
        """Solid white image should have zero wavelet entropy."""
        assert  WaveletEntropy(test_samples["solid_white"]).value == 0.0,   \
                f"Wavelet entropy of solid white image should be 0."

    def test_solid_color_is_zero(self, test_samples) -> None:
        """Any solid color image should have zero wavelet entropy."""
        assert  WaveletEntropy(test_samples["solid_red"]).value == 0.0, \
                f"Wavelet entropy of solid red image should be 0."


class TestWaveletEntropyCheckerboard():
    """# Checkerboard Image Tests
    
    A checkerboard image distributes energy across decomposition levels, so wavelet entropy should 
    be positive and greater than that of a solid image.
    """

    def test_checkerboard_is_positive(self, test_samples) -> None:
        """Checkerboard image should have positive wavelet entropy."""
        assert  WaveletEntropy(test_samples["checker_bw_2x2"]).value > 0.0, \
                f"Wavelet entropy of checker image is no longer positive."

    def test_checkerboard_greater_than_solid(self, test_samples) -> None:
        """Checkerboard image should have greater wavelet entropy than any solid image."""
        assert  WaveletEntropy(test_samples["checker_bw_2x2"]).value >  \
                WaveletEntropy(test_samples["solid_black"]).value,      \
                f"Wavelet entropy of checker image should exceed that of a solid image."


class TestWaveletEntropyBounded():
    """# Boundary Tests
    
    Normalized wavelet entropy is bounded to [0.0, 1.0].
    """

    def test_lower_bound_solid(self, test_samples) -> None:
        """Wavelet entropy should never be negative."""
        assert  WaveletEntropy(test_samples["solid_black"]).value >= 0.0,   \
                f"Wavelet entropy should never be negative."

    def test_upper_bound_checker(self, test_samples) -> None:
        """Wavelet entropy should never exceed 1.0."""
        assert  WaveletEntropy(test_samples["checker_bw_2x2"]).value <= 1.0,    \
                f"Wavelet entropy should never exceed 1.0."


class TestWaveletEntropyGrayscale():
    """# Grayscale Input Tests"""

    def test_grayscale_returns_float(self, test_samples) -> None:
        """Grayscale (2D) input should still return a float."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the wavelet entropy of a 1-channel image is still a float.
        assert  isinstance(WaveletEntropy(gray).value, float),  \
                f"Wavelet entropy of a 1-channel image should still be a float."

    def test_grayscale_solid_is_zero(self, test_samples) -> None:
        """Grayscale solid image should have zero wavelet entropy."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the wavelet entropy of a solid, 1-channel image is zero.
        assert  WaveletEntropy(gray).value == 0.0,  \
                f"Wavelet entropy of a solid, 1-channel image should be zero."