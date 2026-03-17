"""# gradus.commands.generate_samples.main

Main process entry point for generate-samples command.
"""

from gradus.commands.generate_samples.__args__  import GenerateSamplesConfig
from gradus.registration                        import register_command

@register_command(
    id =        "generate-samples",
    config =    GenerateSamplesConfig
)
def generate_samples_entry_point(
    size:           int =   255,
    output_path:    str =   ".test",
    *args,
    **kwargs
) -> None:
    """# Generate Sample Test Images.

    ## Args:
        * size          (int):  Side dimension of images being generated. Defaults to 255.
        * output_path   (str):  Path at which test images will be written. Defaults to "./.test/".
    """
    from logging                                    import Logger
    from pathlib                                    import Path

    from numpy                                      import full, uint8, zeros
    from PIL                                        import Image

    from gradus.commands.generate_samples.utilities import constrast_pairs, make_checkerboard
    from gradus.utilities                           import get_logger

    # Initialize logger.
    logger:         Logger =    get_logger("sample-generation")

    # Resolve output path.
    output_path:    Path =      Path(output_path); output_path.mkdir(parents = True, exist_ok = True)

    # Create solid black & white images.
    Image.fromarray(zeros((size, size, 3),      dtype = uint8)).save(output_path / "black.png")
    Image.fromarray(full( (size, size, 3), 255, dtype = uint8)).save(output_path / "white.png")

    # For each color constrast pair...
    for pair, (color_a, color_b) in constrast_pairs.items():

        # For each tile size...
        for tile_size in [2, 4, 8, 16, 32, 64]:

            # Generate constrasted checkered image.
            make_checkerboard(
                size =      size,
                tiles =     tile_size,
                color_a =   color_a,
                color_b =   color_b
            ).save(output_path / f"{pair}_checker_{tile_size}x{tile_size}.png")

            # Log image generation.
            logger.info(f"""Generated {output_path / f"{pair}_checker_{tile_size}x{tile_size}.png"}""")