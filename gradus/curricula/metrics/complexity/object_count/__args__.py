"""# gradus.curricula.metrics.complexity.object_count.args

Argument definitions & parsing for object-detection metric.
"""

__all__ = ["ObjectCountConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import MetricConfig

class ObjectCountConfig(MetricConfig):
    """# Object Count Metric Configuration"""

    def __init__(self):
        """# Instantiate Object Count Metric Configuration."""
        super(ObjectCountConfig, self).__init__(
            name =  "object-count",
            help =  """(Estimation of) Number of objects via Canny edge detection + connected 
                    components."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Object Count Computation Arguments.
        
        ## Args:
            * paraser   (ArgumentParser):   Parser to whom arguments will be attributed.
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