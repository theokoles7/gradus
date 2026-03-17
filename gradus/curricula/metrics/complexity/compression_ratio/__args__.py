"""# gradus.curricula.metrics.complexity.compression_ratio.args

Argument definitions & parsing for compression-ratio metric.
"""

__all__ = ["CompressionRatioConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class CompressionRatioConfig(MetricConfig):
    """# Compression Ratio Metric Configuration"""

    def __init__(self):
        """# Instantiate Compression Ratio Metric Configuration"""
        super(CompressionRatioConfig, self).__init__(
            name =  "compression-ratio",
            help =  """JPEG compression ratio"""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Color Variance Computation Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "--quality",
            dest =      "quality",
            type =      int,
            default =   95,
            metavar =   "[1-100]",
            help =      """JPEG compression quality (1-100). Higher values yield less compression. 
                        Defaults to 95."""
        )