"""# gradus.networks.vgg

VGG (Visual Geometry Group) network architectures.
"""

__all__ =   [
                # Networks
                "VGG11",
                "VGG13",
                "VGG16",
                "VGG19",
            ]

from gradus.networks.vgg.vgg_11 import VGG11
from gradus.networks.vgg.vgg_13 import VGG13
from gradus.networks.vgg.vgg_16 import VGG16
from gradus.networks.vgg.vgg_19 import VGG19
