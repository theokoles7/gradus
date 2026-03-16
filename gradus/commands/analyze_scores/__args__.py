"""# gradus.commands.analyze_scores.args

Argument definitions & parsing for analyze-scores command.
"""

__all__ = ["AnalyzeScoresConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class AnalyzeScoresConfig(CommandConfig):
    """# Metric Score Analysis Configuration"""

    def __init__(self):
        """# Instantiate Metric Score Analysis Configuration."""
        # Initialize configuration.
        super(AnalyzeScoresConfig, self).__init__(
            name =  "analyze-scores",
            help =  """Compute metric distribution statistics for a scored dataset."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Metric Distribution Analysis Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        from os import listdir

        parser.add_argument(
            "dataset-id",
            dest =      "dataset_id",
            type =      str,
            help =      """Identifier of dataset whose metric distributions are being 
                        calculated."""
        )

        parser.add_argument(
            "--output-path",
            dest =      "output_path",
            type =      str,
            default =   "analysis/datasets",
            help =      """Path at which dataset analysis results will be written. Defaults to 
                        "./analysis/datasets/"."""
        )