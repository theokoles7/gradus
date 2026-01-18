"""# gradus.utilities.logging

Package logging utility.
"""

__all__ =   [
                "configure_logger",
                "get_logger"
            ]

from logging            import getLogger, Formatter, Logger, StreamHandler
from logging.handlers   import RotatingFileHandler
from os                 import makedirs
from sys                import stdout


# Declare base logger.
LOGGER: Logger = getLogger(name = "gradus")


def configure_logger(
    logging_level:  str =   "INFO",
    logging_path:   str =   "logs"
) -> Logger:
    """# Configure Loging Utility.

    ## Args:
        * logging_level (str):  Minimum logging level (DEBUG < INFO < WARNING < ERROR < CRITICAL). 
                                Defaults to "INFO".
        * logging_path  (str):  Path at which logs will be written. Defaults to "logs".

    ## Returns:
        * Logger:   Base package logger after configuration.
    """
    # Ensure logging path exists.
    makedirs(name = logging_path, exist_ok = True)
    
    # Declare global logger.
    global LOGGER
    
    # Set logging level.
    LOGGER.setLevel(level = logging_level)
    
    # Define console handler.
    stdout_handler: StreamHandler =         StreamHandler(stream = stdout)
    
    # Define file handler.
    file_handler:   RotatingFileHandler =   RotatingFileHandler(
                                                filename =      f"{logging_path}/gradus.log",
                                                maxBytes =      1048576,
                                                backupCount =   10
                                            )
    
    # Define formats for each handler.
    stdout_handler.setFormatter(fmt = Formatter(fmt = "%(levelname)s | %(name)s | %(message)s"))
    file_handler.setFormatter(  fmt = Formatter(fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    
    # Add handlers to logger.
    LOGGER.addHandler(hdlr = stdout_handler)
    LOGGER.addHandler(hdlr = file_handler)
    
    # Return logger object.
    return LOGGER


def get_logger(
    logger_name:    str
) -> Logger:
    """# Get Child Logger.

    ## Args:
        * logger_name   (str):  Name attributed to child logger.

    ## Returns:
        * Logger:   New child logger.
    """
    # Declare global logger.
    global LOGGER
    
    # Create new child logger.
    return LOGGER.getChild(suffix = logger_name)