"""# gradus.configuration.metric_config

Sample metric configuration & parsing handler implementation.
"""

__all__ = ["MetricConfig"]

from argparse                       import ArgumentParser
from typing                         import override

from gradus.configuration.protocol  import Config

class MetricConfig(Config):
    """# Sample Metric Configuration & Argument Handler."""

    def __init__(self,
        name:   str,
        help:   str
    ):
        """# Instantiate Metric Configuration.

        ## Args:
            * name  (str):  Sample metric identifier.
            * help  (str):  Description of sample metric.
        """
        # Initialize configuration.
        super(MetricConfig, self).__init__(
            parser_id =         name,
            parser_help =       help
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Metric Arguments.

        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        pass