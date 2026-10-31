"""Siril-backed prototype; all frontends share the same job API."""

# Acquire before any optional imports or worker startup. Installed direct-Python
# entrypoints need the same protection as the native GUI/MCP launchers.
import sys as _sys

_sys.dont_write_bytecode = True
from .install_lifecycle import ensure_process_lease as _ensure_install_lease

_ensure_install_lease(__file__)

__version__ = "1.0.0rc1"
# The product version shown to users (installer, About dialog); PEP 440 form of the release tag v1.0.0-rc.1.
PRODUCT_VERSION = __version__
