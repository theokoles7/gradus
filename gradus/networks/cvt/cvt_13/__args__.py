"""# gradus.networks.cvt.cvt_13.args

Argument definitions & parsing for the CvT-13 neural network.
"""

__all__ = ["CvT13Config"]

from gradus.configuration   import NetworkConfig


class CvT13Config(NetworkConfig):
    """# CvT-13 Network Configuration"""

    def __init__(self):
        """# Instantiate CvT-13 Network Configuration."""
        super(CvT13Config, self).__init__(
            name =  "cvt-13",
            help =  """Convolutional Vision Transformer with 13 transformer blocks."""
        )
