"""# gradus.commands.score_sample.args

Argument definitions & parsing for score-sample command.
"""

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class ScoreSampleConfig(CommandConfig):
    """# Sample Analysis Configuration"""

    def __init__(self):
        """# Instantiate Sample Analysis Command Configuration."""
        # Initilize configuration.
        super(ScoreSampleConfig, self).__init__(
            name =  "score-sample",
            help =  """Compute image complexity metrics for individual dataset sample."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Sample Analysis Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "sample_path",
            type =      str,
            help =      """Path from which dataset sample can be loaded."""
        )

        parser.add_argument(
            "--metrics",
            dest =      "metrics",
            type =      str,
            nargs =     "+",
            choices =   [
                            "all",
                            "color-variance",
                            "compression-ratio",
                            "edge-density",
                            "spatial-frequency",
                            "time-to-convergence",
                            "time-to-saturation",
                            "wavelet-energy",
                            "wavelet-entropy"
                        ],
            default =   "all",
            help =      """Metric(s) being calculated for sample."""
        )