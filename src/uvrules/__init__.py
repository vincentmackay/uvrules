"""
uvrules: tools for generating uv-complete radio interferometer arrays.

Public API
----------
- AntArray: core container class for antenna positions, commanded uv points, etc.
- add_ant_rules: main RULES algorithm for adding antennas to fulfill commanded uv points.
"""

from __future__ import annotations

from importlib import metadata

# Re-export the main public classes / functions
from .antarray import AntArray
from .algos.rules import add_ant_rules

__all__ = ["AntArray", "add_ant_rules", "__version__"]


def _get_version() -> str:
    """Return the installed package version.

    Falls back to '0.0.0' if the package metadata cannot be found
    (e.g., when running from a source tree without installation).
    """
    try:
        return metadata.version("uvrules")
    except metadata.PackageNotFoundError:
        # Useful during development if the package isn't installed yet.
        return "0.0.0"


__version__ = _get_version()






