"""# gradus.curricula.metrics.complexity.color_variance.test

Test suite for color variance complexity metric.
"""

from gradus.curricula.metrics.complexity.color_variance.__base__    import ColorVariance

# TESTS ============================================================================================

class TestColorVarianceType():
    """# Return Type Tests for Color Variance Metric
    
    Color variance should always be a float.
    """

    def test_return_type(self, test_samples) -> None:
        """# Color variance shoudl return a float."""
        assert  isinstance(ColorVariance(test_samples["solid_black"]).value, float), \
                f"Color variance value expected to be of type float"


class TestColorVarianceSolid():
    """# Solid Color Image Tests
    
    The color variance of any solid, single-color image should simply be zero.
    """

    def test_solid_black_is_zero(self, test_samples):
        """Solid black image should have zero variance."""
        assert  ColorVariance(test_samples["solid_black"]).value == 0.0,    \
                f"Color variance of solid black image should be 0."

    def test_solid_white_is_zero(self, test_samples):
        """Solid white image should have zero variance."""
        assert  ColorVariance(test_samples["solid_white"]).value == 0.0,    \
                f"Color variance of solid white image should be 0."

    def test_solid_color_is_zero(self, test_samples):
        """Any solid color image should have zero variance."""
        assert  ColorVariance(test_samples["solid_red"]).value == 0.0,  \
                f"Color variance of solid red image should be 0."


class TestColorVarianceCheckerboard():
    """# Checkerboard Image Tests
    
    The color variance of any non-single-color image should be grater than zero.
    """

    def test_checkerboard_is_positive(self, test_samples):
        """Checkerboard image should have positive variance."""
        assert  ColorVariance(test_samples["checker_bw_2x2"]).value > 0.0,  \
                f"Color variance of checker image is no longer positive."


class TestColorVarianceGrayscale():
    """# Grayscale Input Tests"""

    def test_grayscale_returns_float(self, test_samples):
        """Grayscale (2D) input should still return a float."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the color variance of a 1-channel image is still a float.
        assert  isinstance(ColorVariance(gray).value, float),   \
                f"Color variance of a 1-channel image should still be a float."

    def test_grayscale_solid_is_zero(self, test_samples):
        """Grayscale solid image should have zero variance."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the color variance of a solid, 1-channel image is zero.
        assert  ColorVariance(gray).value == 0.0,   \
                f"Color variance of a solid, 1-channel image should be zero."