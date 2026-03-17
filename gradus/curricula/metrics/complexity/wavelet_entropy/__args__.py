"""# gradus.metrics.complexity.wavelet_entropy.args

Argument definitions & parsing for wavelet-entropy metric.
"""

__all__ = ["WaveletEntropyConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class WaveletEntropyConfig(MetricConfig):
    """# Wavelet Entropy Metric Configuration"""

    def __init__(self):
        """# Instantiate Wavelet Entropy Metric Configuration."""
        super(WaveletEntropyConfig, self).__init__(
            name =  "wavelet-entropy",
            help =  """Total wavelet decomposition entropy."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Wavelet Entropy Computation Arguments.
        
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