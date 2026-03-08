"""# gradus.metrics.complexity.color_variance.args

Argument definitions & parsing for color-variance metric.
"""

__all__ = ["ColorVarianceConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class ColorVarianceConfig(MetricConfig):
    """# Color Variance Metric Configuration"""

    def __init__(self):
        """# Instantiate Color Variance Metric Configuration"""
        super(ColorVarianceConfig, self).__init__(
            name =  "color-variance",
            help =  """Mean channel-wise pixel variance."""
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
        pass