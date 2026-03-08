"""# gradus.registration

Registry system utilities.
"""

__all__ =   [
                # Registries
                "COMMAND_REGISTRY",
                "DATASET_REGISTRY",
                "METRIC_REGISTRY",
                "NETWORK_REGISTRY",

                # Decorators
                "register_command",
                "register_dataset",
                "register_metric",
                "register_network",
            ]

from gradus.registration.decorators import *
from gradus.registration.registries import *

# Instantiate registries.
COMMAND_REGISTRY:   CommandRegistry =   CommandRegistry()
DATASET_REGISTRY:   DatasetRegistry =   DatasetRegistry()
METRIC_REGISTRY:    MetricRegistry =    MetricRegistry()
NETWORK_REGISTRY:   NetworkRegistry =   NetworkRegistry()