"""# gradus.registration

Registry system utilities.
"""

__all__ =   [
                # Registries
                "COMMAND_REGISTRY",
                "DATASET_REGISTRY",

                # Decorators
                "register_command",
                "register_dataset",
            ]

from gradus.registration.decorators import *
from gradus.registration.registries import *

# Instantiate registries.
COMMAND_REGISTRY:   CommandRegistry =   CommandRegistry()
DATASET_REGISTRY:   DatasetRegistry =   DatasetRegistry()