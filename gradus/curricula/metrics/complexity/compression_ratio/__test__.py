"""# gradus.curricula.metrics.complexity.compression_ratio.test

Test suite for compression ratio complexity metric.
"""

from gradus.curricula.metrics.complexity.compression_ratio.__base__ import CompressionRatio

# TESTS ============================================================================================

class TestCompressionRatioType():
    """# Return Type Tests for Compression Ratio Metric
    
    Compression ratio should always be a float.
    """

    def test_return_type(self, test_samples) -> None:
        """# Compression ratio should return a float."""
        assert  isinstance(CompressionRatio(test_samples["solid_black"]).value, float), \
                f"Compression ratio value expected to be of type float."


class TestCompressionRatioSolid():
    """# Solid Color Image Tests
    
    Solid images compress extremely well, so their compression ratio should be high.
    """

    def test_solid_black_is_positive(self, test_samples) -> None:
        """Solid black image should have a positive compression ratio."""
        assert  CompressionRatio(test_samples["solid_black"]).value > 0.0,  \
                f"Compression ratio of solid black image should be positive."

    def test_solid_white_is_positive(self, test_samples) -> None:
        """Solid white image should have a positive compression ratio."""
        assert  CompressionRatio(test_samples["solid_white"]).value > 0.0,  \
                f"Compression ratio of solid white image should be positive."

    def test_solid_color_is_positive(self, test_samples) -> None:
        """Any solid color image should have a positive compression ratio."""
        assert  CompressionRatio(test_samples["solid_red"]).value > 0.0,    \
                f"Compression ratio of solid red image should be positive."


class TestCompressionRatioCheckerboard():
    """# Checkerboard Image Tests
    
    Checkerboard images are more complex than solid images and should compress less efficiently, 
    yielding a lower compression ratio than solid images.
    """

    def test_checkerboard_is_positive(self, test_samples) -> None:
        """Checkerboard image should have a positive compression ratio."""
        assert  CompressionRatio(test_samples["checker_bw_32x32"]).value > 0.0, \
                f"Compression ratio of checker image should be positive."

    def test_solid_compresses_better_than_checkerboard(self, test_samples) -> None:
        """Solid image should compress better than a checkerboard image."""
        assert  CompressionRatio(test_samples["solid_black"]).value >       \
                CompressionRatio(test_samples["checker_bw_32x32"]).value,   \
                f"Solid image should have higher compression ratio than checkerboard."


class TestCompressionRatioBounded():
    """# Boundary Tests
    
    Compression ratio is original size divided by compressed size, so it should always be strictly 
    positive.
    """

    def test_lower_bound_solid(self, test_samples) -> None:
        """Compression ratio should always be greater than zero."""
        assert  CompressionRatio(test_samples["solid_black"]).value > 0.0,  \
                f"Compression ratio should always be greater than zero."

    def test_lower_bound_checker(self, test_samples) -> None:
        """Compression ratio should always be greater than zero."""
        assert  CompressionRatio(test_samples["checker_bw_2x2"]).value > 0.0,   \
                f"Compression ratio should always be greater than zero."