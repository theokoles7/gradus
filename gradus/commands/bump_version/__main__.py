"""# gradus.commands.bump_version.main

Main process entry point for bump-version command.
"""

__all__ = ["bump_version_entry_point"]

from typing                                 import Literal

from gradus.commands.bump_version.__args__  import BumpVersionConfig
from gradus.registration                    import register_command

@register_command(
    id =        "bump-version",
    config =    BumpVersionConfig
)
def bump_version_entry_point(
    bump_type:  Literal["major", "minor", "patch"],
    tag:        bool =                              False,
    message:    str =                               "Version bump",
    *args,
    **kwargs
) -> None:
    """# Increment Package Version.

    ## Args:
        * bump_type (str):  Specify which segment of the version to increment.
        * tag       (bool): Commit all changes and tag as release. Defaults to False.
        * message   (strl): Message indicating purpose of version bump.

    ## Bump Types:
        * `major`:  Incompatible PA changes.
        * `minor`:  Backward-compatible functinality additions/features.
        * `patch`:  Backward-compatible bug fixes/security patches.
    """
    from logging            import Logger
    from pathlib            import Path
    from re                 import Match, search, sub
    from subprocess         import CalledProcessError, run

    from gradus.utilities   import get_logger

    # Initialize logger.
    logger: Logger =    get_logger("bump-version")

    try:# Locate metadata file.
        metadata_file:  Path =  Path(__file__).parent.parent.parent / "__meta__.py"

        # If metadata file is missing...
        if not metadata_file.exists():

            # Report error & abort.
            logger.error(f"Could not locate __meta__.py at {metadata_file}"); return
        
        # Read file.
        metadata:       str =   metadata_file.read_text()

        # Read current version.
        old_version:    Match = search(r'__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"', metadata)

        # If current version could not be found...
        if not old_version:

            # Report error & abort.
            logger.error("Could not parse current version from __meta__.py"); return
        
        # Extract each component of current version.
        major, minor, patch =   map(int, old_version.groups())

        # Log current version, prior to update.
        logger.info(f"Current version: {major}.{minor}.{patch}")

        # Match bump type.
        match bump_type:

            # Major
            case "major":   major += 1; minor = 0; patch = 0

            # Minor
            case "minor":   minor += 1; patch = 0

            # Patch
            case "patch":   patch += 1

            # Invalid
            case _:         logger.error(f"Invalid bumpy type: {bump_type}"); return

        # Form new version string.
        new_version:    str =   f"{major}.{minor}.{patch}"

        # Replace version in metadata.
        metadata:       str =   sub(
                                    r'__version__\s*=\s*"[^"]"',
                                    f'__version__ = "{new_version}"',
                                    metadata
                                )
        
        # Write metadata back to file.
        metadata_file.write_text(metadata)

        # Report success.
        logger.info(f"Version successfully updated to v{new_version}")

        # If tagging is requested...
        if tag:

            try:# Stage changes.
                run(["git", "add", "."], check = True)

                # Form version message.
                commit_message: str =   f"v{new_version} {message}"

                # Commit changes.
                run(["git", "commit", "-m", commit_message], check = True)

                # Log action.
                logger.info(f"Committed version bump with message: {commit_message}")

                # Create tag.
                run(["git", "tag", f"v{new_version}", "-m", commit_message], check = True)

                # Report success.
                logger.info(f"Successfully created Git tag: v{new_version}")
                logger.info(f"Remember to push commit & tag to remote repository: git push && git push --tags")

            # If failure occurs...
            except CalledProcessError as e:

                # Report failure.
                logger.error(f"Failed to create git tag: vv{new_version}: {e}")
                logger.warning(f"Metadata file was modified but may not be commited")

    # Report wild card errors.
    except Exception as e: logger.critical(f"Version bump failed: {e}")