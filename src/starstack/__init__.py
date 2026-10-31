"""Siril-backed prototype; all frontends share the same job API."""

# Acquire before any optional imports or worker startup. Installed direct-Python
# entrypoints need the same protection as the native GUI/MCP launchers.
import sys as _sys

_sys.dont_write_bytecode = True
from .install_lifecycle import ensure_process_lease as _ensure_install_lease

_ensure_install_lease(__file__)

__version__ = "0.2.0rc1"
