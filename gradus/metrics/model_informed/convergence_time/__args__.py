"""# gradus.metrics.model_informed.convergence_time.args

Argument definitions & parsing for convergence-time metric.
"""

__all__ = ["ConvergenceTimeConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class ConvergenceTimeConfig(MetricConfig):
    """# Convergence Time Metric Configuration"""

    def __init__(self):
        """# Instantiate Convergence Time Metric Configuration."""
        super(ConvergenceTimeConfig, self).__init__(
            name =  "convergence-time",
            help =  """Time to achieve loss convergence/stability."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Convergence Time Computation Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        parser.add_argument(
            "--max-iterations",
            dest =      "max_iterations",
            type =      int,
            default =   1000,
            help =      """Maximum number of iterations allowed before abandoning measurement 
                        attempt. Defaults to 1,000."""
        )

        parser.add_argument(
            "--threshold",
            dest =      "threshold",
            type =      float,
            default =   1e-3,
            help =      """Threshold under which the loss delta must fall to be considered 
                        "converged". Defaults to 0.001."""
        )

        parser.add_argument(
            "--window",
            dest =      "window",
            type =      int,
            default =   5,
            help =      """Number of consecutive iterations for which loss delta must remain under 
                        threshold to achieve "stable convergence". Defaults to 5."""
        )

        parser.add_argument(
            "--learning-rate",
            dest =      "learning_rate",
            type =      float,
            default =   0.05,
            help =      """Learning rate with which optimizer will be configured. Defaults to 
                        0.05."""
        )