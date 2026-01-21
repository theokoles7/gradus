"""# gradus.registration.core.exceptions

Defines various exceptions pertaining to registration operations.
"""

__all__ =   [
                # Protocol
                "RegistrationError",

                # Concrete
                "DuplicateEntryError",
                "EntryNotFoundError",
                "EntryPointNotConfiguredError",
                "ParserNotConfiguredError",
                "RegistryNotLoadedError"
            ]


# PROTOCOL =========================================================================================

class RegistrationError(Exception):
    """# Generic Registration Error.
    
    Base exception class for all registration-related errors.
    """
    pass

# CONCRETE =========================================================================================

class DuplicateEntryError(RegistrationError):
    """# Duplicate Entry Error.
    
    Raised when attempting to register an entry that already exists.
    """
    
    def __init__(self,
        entry_id:       str,
        registry_id:    str
    ):
        """# Raise Duplicate Entry Error.

        ## Args:
            * entry_id      (str):  Name of entry whose registration was attempted.
            * registry_id   (str):  Registry through which registration was attempted.
        """
        super(DuplicateEntryError, self).__init__(
            f"""Entry "{entry_id}" is already registered in {registry_id} registry"""
        )


class EntryNotFoundError(RegistrationError):
    """# Entry Not Found Error.
    
    Raised when attempting to access an entry that is not registered.
    """
    
    def __init__(self,
        entry_id:       str,
        registry_id:    str
    ):
        """# Raise Entry Not Found Error.

        ## Args:
            * entry_id      (str):  Name of entry whose access was attempted.
            * registry_id   (str):  Registry through which access was attempted.
        """
        super(EntryNotFoundError, self).__init__(
            f"""Entry "{entry_id}" not registered in {registry_id} registry"""
        )
        
        
class EntryPointNotConfiguredError(RegistrationError):
    """# Entry Point Not Configured Error.
    
    Raised when attempting to dispatch to an entry that has no entry point configured.
    """
    
    def __init__(self,
        entry_id:   str
    ):
        """# Raise Entry Point Not Configured Error.

        ## Args:
            * entry_id  (str):  Entry to whom dispatching was attempted.
        """
        super(EntryPointNotConfiguredError, self).__init__(
            f"""Entry "{entry_id}" was not registered with an entry point"""
        )
        
        
class ParserNotConfiguredError(RegistrationError):
    """# Parser Not Configured Error.
    
    Raised when attempting to register arguments for an entry who was not configured with a parser 
    handler.
    """
    
    def __init__(self,
        entry_id:   str
    ):
        """# Raise Parser Not Configured Error.

        ## Args:
            * entry_id  (str):  Entry whose argument parser registration was attempted.
        """
        super(ParserNotConfiguredError, self).__init__(
            f"""Entry "{entry_id}" was not registered with an argument parser handler"""
        )
        

class RegistryNotLoadedError(RegistrationError):
    """# Registry Not Loaded Error.
    
    Raised when attempting to access a registry whose modules have not been loaded yet.
    """
    
    def __init__(self,
        registry_id:    str
    ):
        """# Raise Registry Not Loaded Error.

        ## Args:
            * registry_id   (str):  Registry who is not yet loaded.
        """
        super(RegistryNotLoadedError, self).__init__(
            f"""Operation attempted requires that the {registry_id} registry is loaded"""
        )