"""# gradus.curricula.metrics.complexity.edge_density.test

Test suite for edge density complexity metric.
"""

from gradus.curricula.metrics.complexity.edge_density.__base__  import EdgeDensity

# TESTS ============================================================================================

class TestEdgeDensityType():
    """# Return Type Tests for Edge Density Metric
    
    Edge density should always be a float.
    """

    def test_return_type(self, test_samples) -> None:
        """# Edge density should return a float."""
        assert  isinstance(EdgeDensity(test_samples["solid_black"]).value, float), \
                f"Edge density value expected to be of type float."


class TestEdgeDensitySolid():
    """# Solid Color Image Tests
    
    A solid, single-color image has no edges, so edge density should be zero.
    """

    def test_solid_black_is_zero(self, test_samples) -> None:
        """Solid black image should have zero edge density."""
        assert  EdgeDensity(test_samples["solid_black"]).value == 0.0,  \
                f"Edge density of solid black image should be 0."

    def test_solid_white_is_zero(self, test_samples) -> None:
        """Solid white image should have zero edge density."""
        assert  EdgeDensity(test_samples["solid_white"]).value == 0.0,  \
                f"Edge density of solid white image should be 0."

    def test_solid_color_is_zero(self, test_samples) -> None:
        """Any solid color image should have zero edge density."""
        assert  EdgeDensity(test_samples["solid_red"]).value == 0.0,    \
                f"Edge density of solid red image should be 0."


class TestEdgeDensityCheckerboard():
    """# Checkerboard Image Tests
    
    A checkerboard image has edges at color boundaries, so edge density should be positive.
    """

    def test_checkerboard_is_positive(self, test_samples) -> None:
        """Checkerboard image should have positive edge density."""
        assert  EdgeDensity(test_samples["checker_bw_2x2"]).value > 0.0,    \
                f"Edge density of checker image is no longer positive."

    def test_checkerboard_greater_than_solid(self, test_samples) -> None:
        """Checkerboard image should have greater edge density than any solid image."""
        assert  EdgeDensity(test_samples["checker_bw_32x32"]).value >   \
                EdgeDensity(test_samples["solid_black"]).value,         \
                f"Edge density of checker image should exceed that of a solid image."


class TestEdgeDensityBounded():
    """# Boundary Tests
    
    Edge density is a fraction of total pixels, so it should always be in [0.0, 1.0].
    """

    def test_lower_bound_solid(self, test_samples) -> None:
        """Edge density should never be negative."""
        assert  EdgeDensity(test_samples["solid_black"]).value >= 0.0,  \
                f"Edge density should never be negative."

    def test_upper_bound_checker(self, test_samples) -> None:
        """Edge density should never exceed 1.0."""
        assert  EdgeDensity(test_samples["checker_bw_2x2"]).value <= 1.0,   \
                f"Edge density should never exceed 1.0."


class TestEdgeDensityGrayscale():
    """# Grayscale Input Tests"""

    def test_grayscale_returns_float(self, test_samples) -> None:
        """Grayscale (2D) input should still return a float."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the edge density of a 1-channel image is still a float.
        assert  isinstance(EdgeDensity(gray).value, float), \
                f"Edge density of a 1-channel image should still be a float."

    def test_grayscale_solid_is_zero(self, test_samples) -> None:
        """Grayscale solid image should have zero edge density."""
        # Take the mean of a solid black image (will be zero).
        gray = test_samples["solid_black"].mean(dim = 0)

        # Ensure that the edge density of a solid, 1-channel image is zero.
        assert  EdgeDensity(gray).value == 0.0, \
                f"Edge density of a solid, 1-channel image should be zero."