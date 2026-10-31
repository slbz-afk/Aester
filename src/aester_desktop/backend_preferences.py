# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""Recover missing remembered backends from the launcher's installed defaults.

This is a startup migration, not a fallback for an explicit interactive choice.
Keep existing external backends, and keep failures visible when no replacement
is available. In particular, never silently switch the optional finite engine.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from starstack.finite_backend_assets import discover_finite_backend


def _file(value: str) -> bool:
    try:
        return bool(value) and Path(value).expanduser().is_file()
    except (OSError, ValueError):
        return False


def apply_backend_defaults(window, environ=None) -> list[dict[str, str]]:
    """Adopt available defaults only for blank or missing stored component files.

The native launcher has already checked installation/receipt ownership. Do not
search a shared component pointer here: it can belong to another installation.
Persist only migrated keys, leaving all unrelated preferences untouched.
"""
    env = os.environ if environ is None else environ
    changes = []
    settings = window._preferences

    def candidate(field, env_key):
        previous = field.text().strip()
        default = env.get(env_key, '').strip()
        return (previous, default) if not _file(previous) and _file(default) else None

    def adopt(field, key, selected):
        previous, current = selected
        field.setText(current)
        if settings is not None:
            settings.setValue(key, current)
        changes.append({'key': key, 'previous': previous, 'current': current})

    siril = candidate(window.siril_path, 'STARSTACK_SIRIL')
    if siril:
        adopt(window.siril_path, 'siril_path', siril)

    model = candidate(window.star_model_path, 'STARSTACK_STAR_MODEL')
    if model:
        controls = [(window.model_channel_order, 'channel_order'),
                    (window.model_range, 'value_range'),
                    (window.model_layout, 'layout'),
                    (window.model_stitch, 'stitch')]
        try:
            adapter = json.loads(env.get('STARSTACK_STAR_ADAPTER', '{}'))
            indices = [control.findData(adapter.get(key)) for control, key in controls]
        except (ValueError, TypeError, AttributeError):
            indices = []
        # A new model and its full adapter form one choice. Never reuse the old
        # model's channel order/range when accepting a different model file.
        if indices and all(index >= 0 for index in indices):
            for (control, key), index in zip(controls, indices):
                control.setCurrentIndex(index)
                if settings is not None:
                    settings.setValue('model/' + key, control.currentData())
            adopt(window.star_model_path, 'star_model_path', model)

    if settings is not None and changes:
        settings.sync()
    return changes
