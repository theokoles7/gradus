"""# gradus.conftest

Test configurations, methods, & fixtures.
"""

from pathlib                import Path
from typing                 import Dict

from PIL.Image              import open as open_image
from pytest                 import fixture
from torch                  import Tensor
from torchvision.transforms import ToTensor

# FIXTURES =========================================================================================

@fixture(scope = "session")
def test_samples() -> Dict[str, Tensor]:
    """# Generated Sample Tensors"""
    from gradus.commands.generate_samples.__main__  import generate_samples_entry_point

    # Form samples path.
    path:       Path =  Path(".cache/test-samples")

    # Load transform.
    transform:  ToTensor =  ToTensor()

    # Generate samples.
    generate_samples_entry_point(size = 64, output_path = path)

    return  {
        "solid_black":          transform(open_image(path / "solid_black.png")),
        "solid_white":          transform(open_image(path / "solid_white.png")),
        "solid_red":            transform(open_image(path / "solid_red.png")),
        "solid_green":          transform(open_image(path / "solid_green.png")),
        "solid_blue":           transform(open_image(path / "solid_blue.png")),
        "checker_bw_2x2":       transform(open_image(path / "checker_black_white_2x2.png")),
        "checker_bw_8x8":       transform(open_image(path / "checker_black_white_8x8.png")),
        "checker_bw_32x32":     transform(open_image(path / "checker_black_white_32x32.png")),
        "checker_bw_64x64":     transform(open_image(path / "checker_black_white_64x64.png")),
        "checker_rc_2x2":       transform(open_image(path / "checker_red_cyan_2x2.png")),
        "checker_rc_32x32":     transform(open_image(path / "checker_red_cyan_32x32.png")),
    }