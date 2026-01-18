"""# gradus.configuration.command_config

Command configuration & parsing handler implementation.
"""

__all__ = ["CommandConfig"]

from typing                         import Optional

from gradus.configuration.protocol  import Config

class CommandConfig(Config):
    """# Command Configuration & Argument Handler"""

    def __init__(self,
        name:               str,
        help:               str,
        subparser_title:    Optional[str] = None,
        subparser_help:     Optional[str] = None
    ):
        """# Instantiate Command Configuration.

        ## Args:
            * name              (str):          Command identifier.
            * help              (str):          Description of command's purpose.
            * subparser_title   (str | None):   Name attributed to sub-command objects.
            * subparser_help    (str | None):   Description of sub-command purpose.
        """
        # Initialize configuration.
        super(CommandConfig, self).__init__(
            parser_id =         name,
            parser_help =       help,
            subparser_title =   subparser_title,
            subparser_help =    subparser_help
        )