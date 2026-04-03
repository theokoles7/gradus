"""# gradus.curricula.metrics.complexity.edge_density.args

Argument definitions & parsing for edge-density metric.
"""

__all__ = ["EdgeDensityConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class EdgeDensityConfig(MetricConfig):
    """# Edge Density Metric Configuration"""

    def __init__(self):
        """# Instantiate Edge Density Metric Configuration"""
        super(EdgeDensityConfig, self).__init__(
            name =  "edge-density",
            help =  """Edge density via Canny edge detection."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Edge Density Computation Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "--low", "-L",
            dest =      "low",
            type =      int,
            default =   100,
            help =      """Canny low threshold. Defaults to 100."""
        )

        parser.add_argument(
            "--high", "-H",
            dest =      "high",
            type =      int,
            default =   200,
            help =      """Canny high threshold. Defaults to 200."""
        )