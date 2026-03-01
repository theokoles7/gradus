"""# gradus.commands.analyze_dataset.args

Argument definitions & parsing for score-dataset command.
"""

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
        from gradus.registration    import DATASET_REGISTRY

        parser.add_argument(
            "dataset_id",
            type =      str,
            choices =   DATASET_REGISTRY.list_entries(),
            help =      """Identifier of dataset being analyzed."""
        )
        
        parser.add_argument(
            "--output-path",
            dest =      "output_path",
            type =      str,
            default =   "analyses/datasets",
            help =      """Path at which dataset analysis results will be written. Defaults to 
                        "./analyses/datasets/"."""
        )