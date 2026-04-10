"""# gradus.curricula.metrics.complexity.color_variance.test

Test suite for color variance complexity metric.
"""

from pathlib                import Path
from typing                 import Dict

from PIL.Image              import open as opem_image
from pytest                 import fixture
from torch                  import Tensor
from torchvision.transforms import ToTensor

from gradus.curricula.metrics.complexity.color_variance.__base__    import ColorVariance

# FIXTURES =========================================================================================

@fixture(scope = "module")
def test_samples() -> Dict[str, Tensor]:
    """# Generated Sample Tensors"""
    from gradus.commands.generate_samples.__main__  import generate_samples_entry_point

    # Form samples path.
    samples_path:   Path =  Path(".cache/test-samples")

@fixture(scope = "module")
def to_tensor() -> ToTensor:
    """# ToTensor Transform"""
    return ToTensor()

class TestColorVarianceType():
    """# Return Type Tests for Color Variance Metric"""

    def test_return_type(self, samples) -> None:
        """# Color variance shoudl return a float."""
        assert isinstance(ColorVariance(samples["solid_black"]).value, float)