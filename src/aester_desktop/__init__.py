# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Aester desktop application (GPL-3.0-or-later). The processing engine is the Apache-2.0 ``starstack`` package."""
import starstack as _engine  # noqa: F401 -- the engine package takes the installation lease before Qt loads
from starstack import i18n as _i18n

__copyright__ = "Copyright (C) 2026 Wang Jian (王俭)"
__license__ = "GPL-3.0-or-later"

_i18n.register_catalogs("aester_desktop.translations_gui", "aester_desktop.translations_dialogs")
