"""# gradus.curricula.metrics.complexity.spatial_frequency.test

Test suite for spatial frequency complexity metric.
"""

from gradus.curricula.metrics.complexity.spatial_frequency.__base__ import SpatialFrequency

# TESTS ============================================================================================

class TestSpatialFrequencyType():
    """# Return Type Tests for Spatial Frequency Metric
    
    Spatial frequency should always be a float.
    """

    def test_return_type(self, test_samples) -> None:
        """# Spatial frequency should return a float."""
        assert  isinstance(SpatialFrequency(test_samples["solid_black"]).value, float), \
                f"Spatial frequency value expected to be of type float."


class TestSpatialFrequencySolid():
    """# Solid Color Image Tests
    
    A solid, single-color image has no pixel-to-pixel differences, so spatial frequency should be 
    zero.
    """

    def test_solid_black_is_zero(self, test_samples) -> None:
        """Solid black image should have zero spatial frequency."""
        assert  SpatialFrequency(test_samples["solid_black"]).value == 0.0, \
                f"Spatial frequency of solid black image should be 0."

    def test_solid_white_is_zero(self, test_samples) -> None:
        """Solid white image should have zero spatial frequency."""
        assert  SpatialFrequency(test_samples["solid_white"]).value == 0.0, \
                f"Spatial frequency of solid white image should be 0."

    def test_solid_color_is_zero(self, test_samples) -> None:
        """Any solid color image should have zero spatial frequency."""
        assert  SpatialFrequency(test_samples["solid_red"]).value == 0.0,   \
                f"Spatial frequency of solid red image should be 0."


class TestSpatialFrequencyCheckerboard():
    """# Checkerboard Image Tests
    
    A checkerboard image has frequent pixel-to-pixel transitions, so spatial
    frequency should be positive and greater than that of a solid image.
    """

    def test_checkerboard_is_positive(self, test_samples) -> None:
        """Checkerboard image should have positive spatial frequency."""
        assert  SpatialFrequency(test_samples["checker_bw_2x2"]).value > 0.0,   \
                f"Spatial frequency of checker image is no longer positive."

    def test_checkerboard_greater_than_solid(self, test_samples) -> None:
        """Checkerboard image should have greater spatial frequency than any solid image."""
        assert  SpatialFrequency(test_samples["checker_bw_2x2"]).value >    \
                SpatialFrequency(test_samples["solid_black"]).value,        \
                f"Spatial frequency of checker image should exceed that of a solid image."

    def test_finer_checkerboard_is_higher(self, test_samples) -> None:
        """Finer checkerboard should have higher spatial frequency than coarser one."""
        assert  SpatialFrequency(test_samples["checker_bw_32x32"]).value >  \
                SpatialFrequency(test_samples["checker_bw_2x2"]).value,     \
                f"Finer checkerboard should have higher spatial frequency than coarser."


class TestSpatialFrequencyBounded():
    """# Boundary Tests
    
    Spatial frequency is always non-negative as it is derived from squared differences.
    """

    def test_lower_bound_solid(self, test_samples) -> None:
        """Spatial frequency should never be negative."""
        assert  SpatialFrequency(test_samples["solid_black"]).value >= 0.0, \
                f"Spatial frequency should never be negative."

    def test_lower_bound_checker(self, test_samples) -> None:
        """Spatial frequency should never be negative."""
        assert  SpatialFrequency(test_samples["checker_bw_2x2"]).value >= 0.0,  \
                f"Spatial frequency should never be negative."


class TestSpatialFrequencyGrayscale():
    """# Grayscale Input Tests"""

    def test_grayscale_returns_float(self, test_samples) -> None:
        """Grayscale (2D) input should still return a float."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the compression ratio of a 1-channel image is still a float.
        assert  isinstance(SpatialFrequency(gray).value, float),    \
                f"Spatial frequency of a 1-channel image should still be a float."

    def test_grayscale_solid_is_zero(self, test_samples) -> None:
        """Grayscale solid image should have zero spatial frequency."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the compression ratio of a solid, 1-channel image is zero.
        assert  SpatialFrequency(gray).value == 0.0,    \
                f"Spatial frequency of a solid, 1-channel image should be zero."