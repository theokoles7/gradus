"""# gradus.commands.generate_samples.utilities

Utility functions applicable to test sample image generation.
"""

__all__ =   [
                "constrast_pairs",
                "make_checkerboard"
            ]

from typing         import Dict, Tuple

from numpy          import array, indices, uint8, where
from numpy.typing   import NDArray
from PIL.Image      import fromarray, Image


# COLORS ===========================================================================================

constrast_pairs:    Dict[str, Tuple] =  {
                                            "black_white":      ([  0,  0,   0], [255, 255, 255]),
                                            "red_cyan":         ([255,  0,   0], [  0, 255, 255]),
                                            "green_magenta":    ([0,  255,   0], [255,   0, 255]),
                                            "blue_yellow":      ([0,    0, 255], [255, 255,   0])
                                        }


# FUNCTIONS ========================================================================================

def make_checkerboard(
    size:       int,
    tiles:      int,
    color_a:    int,
    color_b:    int
) -> Image:
    """# Generate Checkerboard Image.

    ## Args:
        * size      (int):  Image dimension size.
        * tiles     (int):  Number of tiles on each side.
        * color_a   (int):  First constrasting color.
        * color_b   (int):  Second contrasting color.

    ## Returns:
        * Image:    Checkerboard image.
    """
    # Define row, column coordinates.
    rows, cols =            indices((size, size))

    # Resolve cell size.
    tile_size:  int =       size // tiles

    # Construct grid.
    grid:       NDArray =   (rows // tile_size + cols // tile_size) % 2

    # Generate image.
    return  fromarray(
                obj =   where(
                            grid[:, :, None],
                            array(color_a, dtype = uint8),
                            array(color_b, dtype = uint8)
                        ).astype(uint8),
                mode =  "RGB"
            )