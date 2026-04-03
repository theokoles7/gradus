"""# gradus.metrics.complexity.wavelet_energy.args

Argument definitions & parsing for wavelet-energy metric.
"""

__all__ = ["WaveletEnergyConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class WaveletEnergyConfig(MetricConfig):
    """# Wavelet Energy Metric Configuration"""

    def __init__(self):
        """# Instantiate Wavelet Energy Metric Configuration."""
        super(WaveletEnergyConfig, self).__init__(
            name =  "wavelet-energy",
            help =  """Total wavelet decomposition energy."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Wavelet Energy Computation Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "--wavelet", "-W",
            dest =      "wavelet",
            type =      str,
            default =   "db2",
            help =      """Wavelet family to use for decomposition. Defaults to "db2"."""
        )

        parser.add_argument(
            "--level", "-L",
            dest =      "level",
            type =      int,
            default =   None,
            help =      """Decomposition level. Defaults to maximum possible."""
        )