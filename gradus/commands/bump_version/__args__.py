"""# gradus.commands.bump_version.args

Arguments definitions & parsing for bump-version command.
"""

__all__ = ["BumpVersionConfig"]

from argparse               import _ArgumentGroup, ArgumentParser
from typing                 import override

from gradus.configuration   import CommandConfig

class BumpVersionConfig(CommandConfig):
    """# Bump Version COmmand Configuration"""

    def __init__(self):
        """# Instantiate Bump Version Command Configuration."""
        super(BumpVersionConfig, self).__init__(
            name =  "bump-version",
            help =  """Increment/update package version."""
        )

    # HELPERS ======================================================================================

    @override
    def _define_arguments_(self,
        parser: ArgumentParser
    ) -> None:
        """# Define Parser Arguments.
        
        ## Args:
            * parser    (ArgumentParser):   Parser to whom arguments will be attributed.
        """
        # BUMP TYPE --------------------------------------------------------------------------------
        bump_type:  _ArgumentGroup =    parser.add_argument_group(
                                            title =         "Bump Type",
                                            description =   """Specify which segment of the version 
                                                            to increment."""
                                        )
        
        bump_type.add_argument(
            "--major",
            dest =      "bump_type",
            action =    "store_const",
            const =     "major",
            help =      """Incompatible API changes."""
        )

        bump_type.add_argument(
            "--minor",
            dest =      "bump_type",
            action =    "store_const",
            const =     "minor",
            help =      """Backward-compatible functionality additions/features."""
        )

        bump_type.add_argument(
            "--patch",
            dest =      "bump_type",
            action =    "store_const",
            const =     "patch",
            help =      """Backward-compatible bug fixes/security patches."""
        )
        
        # GIT ACTIONS ------------------------------------------------------------------------------
        git:        _ArgumentGroup =    parser.add_argument_group(
                                            title =         "Git Actions",
                                            description =   """Indicate Git actions to take on new 
                                                            version/updates."""
                                        )
        
        git.add_argument(
            "--tag",
            dest =      "tag",
            action =    "store_true",
            default =   False,
            help =      """Commit all current changes and tag as release."""
        )

        git.add_argument(
            "--message",
            dest =      "message",
            type =      str,
            default =   "Version bump",
            help =      """Message indicating purpose of version bump."""
        )