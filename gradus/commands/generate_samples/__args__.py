"""# gradus.commands.generate_samples.args

Argument definitions & parsing for generate-samples command.
"""

__all__ = ["GenerateSamplesConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class GenerateSamplesConfig(CommandConfig):
    """# Generate Samples Configuration"""


    def __init__(self):
        """# Instantiate Sample Generate Command Configuration."""
        # Initialize configuration.
        super(GenerateSamplesConfig, self).__init__(
            name =  "generate-samples",
            help =  """Generate sample images for basic complexity testing."""
        )

    # HELPERS ==========================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Dataset Analysis Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "--size",
            dest =      "size",
            type =      int,
            default =   255,
            help =      """Side dimension of images being generated. Defaults to 255."""
        )

        parser.add_argument(
            "--output-path",
            dest =      "output_path",
            type =      str,
            default =   ".test",
            help =      """Path at which test images will be written. Defaults to "./.test/"."""
        )