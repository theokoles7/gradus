"""# gradus.commands.version.main

Main process entry point for version command.
"""

__all__ = ["version_entry_point"]

from gradus.commands.version.__args__   import VersionConfig
from gradus.registration                import register_command

@register_command(
    id =        "version",
    config =    VersionConfig
)
def version_entry_point(*args, **kwargs) -> None:
    """# Display Package Version Information."""
    # Import banner.
    from gradus.utilities   import BANNER

    # Display banner.
    print(BANNER[1:])