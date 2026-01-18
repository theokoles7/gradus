"""# gradus.configuration

Configuration protocol implementations.
"""

__all__ =   [
                # Protocol
                "Config",

                # Concrete
                "CommandConfig",
]

from gradus.configuration.command_config    import CommandConfig
from gradus.configuration.protocol          import Config