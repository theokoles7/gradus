"""# gradus.commands.version.args

Argument definitions & parsing for version command.
"""

__all__ = ["VersionConfig"]

from argparse               import ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class VersionConfig(CommandConfig):
    """# Version Command Configuration"""

    def __init__(self):
        """# Instantiate Version Command Configuration."""
        super(VersionConfig, self).__init__(
            name =  "version",
            help =  """Display package version information."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Parser Arguments.
        
        **NOTE**: Version command does not utilize any arguments.
        """
        pass