"""# gradus.setup

Package setup utility.
"""

from pathlib    import Path
from setuptools import find_packages, setup
from typing     import Any, Dict


# HELPERS ==========================================================================================

def get_long_description() -> str:
    """# Get Long Description.

    ## Returns:
        * str:  README file contents.
    """
    with open(Path(__file__).parent / "README.md", encoding = "utf-8") as f: return f.read()


def get_version() -> str:
    """# Get Package Version.

    ## Returns:
        * str:  Current package version.
    """
    # Initialize metadata mapping.
    metadata:   Dict[str, Any] =    {}

    # Open metadata file.
    with open(Path(__file__).parent / "gradus" / "__meta__.py") as f:

        # Map package data.
        exec(f.read(), metadata)

    # Provide package version.
    return metadata["__version__"]


# SETUP UTILITY ====================================================================================

setup(
    name =                          "gradus",
    version =                       get_version(),
    author =                        "Gabriel C. Trahan",
    author_email =                  "gabriel.trahan1.louisiana.edu",
    description =                   """Experiments in quantifying image complexity and determining 
                                    the efficacy of curriculum learning in image classification 
                                    tasks.""",
    long_description =               get_long_description(),
    long_description_content_type = "text/markdown",
    license =                       "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007",
    license_files =                 ("LICENSE"),
    url =                           "https://github.com/theokoles7/gradus",
    packages =                      find_packages(),
    python_requires =               ">=3.10",
    install_requires =              [
                                        "opencv-python",
                                        "PyWavelets",
                                        "scikit-learn",
                                        "setuptools",
                                        "torch",
                                        "torchvision",
                                        "tqdm",
                                    ],
    entry_points =                  {
                                        "console_scripts":  [
                                                                "gradus=gradus.__main__:gradus_entry_point"
                                                            ],
                                    },
    classifiers =                   [
                                        "Topic :: Scientific/Engineering :: Artificial Intelligence",
                                        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                                        "Intended Audience :: Developers",
                                        "Operating System :: OS Independent",
                                        "Programming Language :: Python :: 3",
                                        "Programming Language :: Python :: 3.10",
                                        "Programming Language :: Python :: 3.11",
                                        "Programming Language :: Python :: 3.12",
                                        "Programming Language :: Python :: 3.13",
                                    ]
)