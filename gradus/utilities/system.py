"""# grardus.utilities.system

Utility functions for system/hardware information and management.
"""

__all__ =   [
                "get_system_core_count",
            ]

def get_system_core_count() -> int:
    """# Get System Core Count

    ## Returns:
        * (int):  Number of available CPU cores.
    """
    from os import cpu_count

    # Count cores.
    try:                return cpu_count() or 1

    # Should any complications arise, default to 1.
    except Exception:   return 1