"""# gradus.commands.score_dataset.args

Argument definitions & parsing for score-dataset command.
"""

__all__ = ["ScoreDatasetConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class ScoreDatasetConfig(CommandConfig):
    """# Dataset Analysis Configuration"""

    def __init__(self):
        """# Instantiate Dataset Analysis Command Configuration."""
        # Initialize configuration.
        super(ScoreDatasetConfig, self).__init__(
            name =              "score-dataset",
            help =              """Analyze a dataset and compute image complexity metrics.""",
            subparser_title =   "dataset-id",
            subparser_help =    """Identifier of dataset being scored."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Dataset Analysis Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        from gradus.registration    import DATASET_REGISTRY, METRIC_REGISTRY

        parser.add_argument(
            "dataset_id",
            type =      str,
            choices =   DATASET_REGISTRY.list_entries(),
            help =      """Identifier of dataset being analyzed."""
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
            "--output-path",
            dest =      "output_path",
            type =      str,
            default =   ".cache/datasets",
            help =      """Path at which dataset analysis results will be written. Defaults to 
                        "./.cache/datasets/"."""
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