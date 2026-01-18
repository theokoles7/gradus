"""# gradus.registration.core.entry

Abstract registration entry protocol.
"""

__all__ = ["Entry"]

from abc                                    import ABC
from argparse                               import _SubParsersAction
from logging                                import Logger
from typing                                 import List, Optional

from gradus.configuration                   import Config
from gradus.registration.core.exceptions    import ParserNotConfiguredError
from gradus.utilities                       import get_logger

class Entry(ABC):
    """# Abstract Registration Entry"""

    def __init__(self,
        id:     str,
        config: Optional[Config] =  None,
        tags:   List[str] =         []
    ):
        """# Instantiate Registration Entry.

        ## Args:
            * id        (str):              Entry ID (seminal entity).
            * config    (Config | None):    Configuration & argument handler.
            * tags      (List[str]):        Taxonomical key words.
        """
        # Initialize logger.
        self.__logger__:    Logger =            get_logger(f"{id}-registration-entry")

        # Define properties.
        self._id_:          str =               id
        self._tags_:        List[str] =         tags
        self._config_:      Optional[Config] =  config

        # Debug registration.
        self.__logger__.debug(f"Registered {self}")

    # PROPERTIES ===================================================================================

    @property
    def config(self) -> Optional[Config]:
        """# Configuration & Argument Handler"""
        return self._config_
    
    @property
    def id(self) -> str:
        """# Entry Identifier"""
        return self._id_
    
    @property
    def tags(self) -> List[str]:
        """# Taxonomical Keywords"""
        return self._tags_
    
    # METHODS ======================================================================================

    def has_tag(self,
        tag:    str
    ) -> bool:
        """# Registration Entry has Tag?

        ## Args:
            * tag   (str):  Tag being queried.

        ## Returns:
            * bool: True if registration entry has specified tag.
        """
        # Debug query.
        self.__logger__.debug(f"{self} has tag {tag}? {tag in self._tags_}")

        # Query tag.
        return tag in self._tags_
    
    def register_configuration(self,
        subparser:  _SubParsersAction
    ) -> None:
        """# Register Configuration & Argument Handler.

        ## Args:
            * subparser (_SubParsersAction):    Sub-parser group of parent under which this entry's 
                                                configuration will be registered.

        ## Raises:
            * ParserNotConfiguredError: If entry was not registered with a configuration & argument 
                                        handler.
        """
        # If entry was not registered with a configuration, report error.
        if self._config_ is None: raise ParserNotConfiguredError(entry_id = self._id_)

        # Debug registration.
        self.__logger__.debug(f"Registering {self} configuration under {subparser.dest}")

        # Register configuration.
        self._config_.register(cls = self._config_, subparser = subparser)

    # DUNDERS ======================================================================================

    def __repr__(self) -> str:
        """# Registration Entry Object Representation"""
        return f"""<{self._id_.upper()}Entry(tags = {self._tags_})>"""