"""# gradus.curricula.metrics.model_informed.saturation_time.args

Argument definitions & parsing for saturation-time metric.
"""

__all__ = ["SaturationTimeConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class SaturationTimeConfig(MetricConfig):
    """# Saturation Time Metric Configuration"""

    def __init__(self,):
        """# Instantiate Saturation Time Metric Configuration."""
        super(SaturationTimeConfig, self).__init__(
            name =  "saturation-time",
            help =  """Time to achieve weight saturation/stability."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Saturation Time Computation Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "--max-iterations",
            dest =      "max_iterations",
            type =      int,
            default =   1000,
            help =      """Maximum number of gradient descent iterations before abandoning 
                        measurement. Defaults to 1000."""
        )

        parser.add_argument(
            "--threshold",
            dest =      "threshold",
            type =      float,
            default =   1e-3,
            help =      """Weight delta (Frobenius norm) threshold below which a layer is considered 
                        stable. Defaults to 1e-3."""
        )

        parser.add_argument(
            "--window",
            dest =      "window",
            type =      int,
            default =   5,
            help =      """Number of consecutive stable iterations required to declare a layer 
                        saturated. Defaults to 5."""
        )

        parser.add_argument(
            "--learning-rate",
            dest =      "learning_rate",
            type =      float,
            default =   0.05,
            help =      """Optimizer learning rate. Defaults to 0.05."""
        )