"""# gradus.curricula.metrics.complexity.spatial_frequency.args

Argument definitions & parsing for spatial-frequency metric.
"""

__all__ = ["SpatialFrequencyConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class SpatialFrequencyConfig(MetricConfig):
    """# Spatial Frequency Metric Configuration"""

    def __init__(self):
        """# Instantiate Spatial Frequency Metric Configuration."""
        super(SpatialFrequencyConfig, self).__init__(
            name =  "spatial-frequency",
            help =  """Root mean squared of row/column pixel differences."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Spatial Frequency Computation Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        pass