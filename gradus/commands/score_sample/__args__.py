"""# gradus.commands.score_sample.args

Argument definitions & parsing for score-sample command.
"""

from argparse               import ArgumentParser
from typing                 import List, override

from gradus.configuration   import CommandConfig

class ScoreSampleConfig(CommandConfig):
    """# Sample Analysis Configuration"""

    def __init__(self):
        """# Instantiate Sample Analysis Command Configuration."""
        # Initialize configuration.
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
        from gradus.registration    import METRIC_REGISTRY

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
            choices =   METRIC_REGISTRY.list_entries() + ["all"],
            default =   ["all"],
            help =      """Metric(s) being calculated for sample."""
        )

        parser.add_argument(
            "--seed",
            dest =      "seed",
            type =      int,
            default =   1,
            help =      """Random number generation seed. Defaults to 1."""
        )

        parser.add_argument(
            "--device",
            dest =      "device",
            type =      str,
            choices =   ["auto", "cpu", "cuda"],
            default =   "auto",
            help =      """Torch computation device. Defaults to "auto"."""
        )