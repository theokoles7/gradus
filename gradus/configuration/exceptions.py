"""# gradus.configuration.exceptions

Defines various exceptions pertaining to configuration operations.
"""

__all__ =   [
                # Protocol
                "ConfigurationError",

                # Concrete
                "SubParserNotConfiguredError",
            ]

# PROTOCOL =========================================================================================

class ConfigurationError(Exception):
    """# Generic Configuration Error.

    Base exception class for all configuration-related errors.
    """
    pass

# CONCRETE =========================================================================================

class SubParserNotConfiguredError(ConfigurationError):
    """# Sub-Parser Not Configured Error.

    Raised when attempting to create a sub-parser group within a configuration class whose 
    sub-parser parameters are not defined.
    """

    def __init__(self,
        parser_id:  str
    ):
        """# Raise Sub-Parser Not Configured Error.

        ## Args:
            * parsery_id    (str):  Identifier of the entity for whom the configuration class is 
                                    defined.
        """
        super(SubParserNotConfiguredError, self).__init__(
            f"""Sub-parser parameters not configured for {parser_id} config class"""
        )