"""# gradus.registration

Registry system utilities.
"""

__all__ =   [
                # Registries
                "COMMAND_REGISTRY",

                # Decorators
                "register_command",
            ]

from gradus.registration.decorators import *
from gradus.registration.registries import *

# Instantiate registries.
COMMAND_REGISTRY:   CommandRegistry =   CommandRegistry()