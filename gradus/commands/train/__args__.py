"""# gradus.commands.train.args

Argument definitions & parsing for train command.
"""

from argparse               import ArgumentParser, _SubParsersAction
from typing                 import override

from gradus.configuration   import CommandConfig

class TrainConfig(CommandConfig):
    """# Training Process Configuration"""

    def __init__(self):
        """# Instantiate Training Command Configuration."""
        # Initialize configuration.
        super(TrainConfig, self).__init__(
            name =              "train",
            help =              """Train a neural network on a dataset.""",
            subparser_title =   "network-id",
            subparser_help =    """Identifier of neural network being trained."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Training Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        from gradus.registration    import NETWORK_REGISTRY
        
        parser.add_argument(
            "--epochs",
            dest =      "epochs",
            type =      int,
            default =   100,
            help =      """Number of training/validation epochs to administer. Defaults to 100."""
        )

        parser.add_argument(
            "--output-path",
            dest =      "output_path",
            type =      str,
            default =   "results",
            help =      """Path at which training results will be written. Defaults to 
                        "./results/"."""
        )

        parser.add_argument(
            "--seed",
            dest =      "seed",
            type =      int,
            default =   1,
            help =      """Random seed for reproducibility. Defaults to 1."""
        )

        parser.add_argument(
            "--device",
            dest =      "device",
            type =      str,
            choices =   ["auto", "cpu", "cuda"],
            default =   "auto",
            help =      """Hardware device upon which data will be processed. Defaults to "auto"."""
        )

        # Create subparser.
        subparser:  _SubParsersAction = self._create_subparser_(parser = parser)

        # Register neural networks as sub-command.
        NETWORK_REGISTRY.register_configurations(subparser = subparser)