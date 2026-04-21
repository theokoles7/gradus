"""# gradus.conftest

Test configurations, methods, & fixtures.
"""

from pathlib                import Path
from typing                 import Dict, Generator

from numpy.random           import default_rng
from pandas                 import DataFrame
from PIL.Image              import open as open_image
from pytest                 import fixture, TempPathFactory
from torch                  import Tensor
from torchvision.transforms import ToTensor

# FIXTURES =========================================================================================

@fixture(scope = "session")
def synthetic_scores() -> DataFrame:
    """# Synthetic Scores DataFrame for Rank Testing.
 
    Builds a small, fully deterministic scores DataFrame with known metric values. All rank tests 
    use this fixture so they never depend on pre-computed score files being present on disk.
 
    Shape: 20 samples × 4 metrics, no NaNs, reproducible via seed.
    """
    # Construct default random number generator.
    rng:    Generator = default_rng(seed = 1)
 
    # Define number of samples.
    n:      int =       20

    # Create sample dataframe.
    return  DataFrame({
                "index":                list(range(n)),
                "class":                [str(i % 4) for i in range(n)],
                "saturation-time":      rng.uniform(1.0, 100.0, n),
                "color-variance":       rng.uniform(0.0,   1.0, n),
                "edge-density":         rng.uniform(0.0,   0.5, n),
                "spatial-frequency":    rng.uniform(0.0,   0.3, n),
            })
 
 
@fixture(scope = "session")
def synthetic_scores_path(
    tmp_path_factory:   TempPathFactory,
    synthetic_scores:   DataFrame
) -> "Path":
    """# Write Synthetic Scores to a Temp Parquet File.
 
    Provides a real scores_path that DatasetMetrics can load from,
    for tests that exercise the full DatasetMetrics → Rank pipeline.
    """
    from pathlib    import Path
 
    # Resolve scores path.
    path:   Path =  tmp_path_factory.mktemp("scores") / "test-dataset" / "metric-scores_seed-1.parquet"
    
    # Ensure path exists.
    path.parent.mkdir(parents = True, exist_ok = True)

    # Save to file.
    synthetic_scores.to_parquet(path, index = False)
 
    # Provide path to scores.
    return path.parent


@fixture(scope = "session")
def test_samples() -> Dict[str, Tensor]:
    """# Generated Sample Tensors"""
    from gradus.commands.generate_samples.__main__  import generate_samples_entry_point

    # Form samples path.
    path:       Path =      Path(".cache/test-samples")

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