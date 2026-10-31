"""Colour-blind-safe annotation colours drawn inside the photo (theme independent).

From outputs/ui-research-20260927/domain (Okabe-Ito / Tol subset): the five colours stay
>= 10 dE2000 apart under protan, deutan and tritan simulation (Machado 2009, severity 1).
Every stroke is cased with ``OUTLINE`` (draw the outline ``OUTLINE_EXTRA`` px wider first).
No Qt dependency: callers build QColor/QPen from these values.
"""
from __future__ import annotations

from types import MappingProxyType


def _rgb(value: str) -> tuple[int, int, int]:
    return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)


SKY = "#66CCEE"
GROUND = "#D55E00"
MILKYWAY = "#CC79A7"
TRAIL = "#009E73"
HORIZON = "#F0E442"
HANDLE = HORIZON
SUGGESTION = SKY
OUTLINE = "#101010"

SKY_RGB = _rgb(SKY)
GROUND_RGB = _rgb(GROUND)
MILKYWAY_RGB = _rgb(MILKYWAY)
TRAIL_RGB = _rgb(TRAIL)
HORIZON_RGB = _rgb(HORIZON)
HANDLE_RGB = HORIZON_RGB
SUGGESTION_RGB = SKY_RGB
OUTLINE_RGB = _rgb(OUTLINE)

# Line styles: "solid", "dash", "dot", "double" (two parallel strokes, after the Suzhou
# star map's Milky Way), with dash patterns in stroke widths (Qt QPen.setDashPattern units).
SOLID, DASH, DOT, DOUBLE = "solid", "dash", "dot", "double"
DASH_PATTERN = (4.0, 3.0)
DOT_PATTERN = (1.0, 2.0)
LINE_WIDTH = 2.0            # logical px, cosmetic
OUTLINE_EXTRA = 2.0         # outline width = LINE_WIDTH + OUTLINE_EXTRA
DOUBLE_GAP = 3.0            # distance between the two Milky Way strokes
HANDLE_RADIUS = 5.0         # round handles, filled HANDLE with OUTLINE ring

OVERLAYS = MappingProxyType({
    "sky": MappingProxyType({"hex": SKY, "rgb": SKY_RGB, "line": DASH}),
    "ground": MappingProxyType({"hex": GROUND, "rgb": GROUND_RGB, "line": SOLID}),
    "milkyway": MappingProxyType({"hex": MILKYWAY, "rgb": MILKYWAY_RGB, "line": DOUBLE}),
    "trail": MappingProxyType({"hex": TRAIL, "rgb": TRAIL_RGB, "line": DOT}),
    "horizon": MappingProxyType({"hex": HORIZON, "rgb": HORIZON_RGB, "line": SOLID}),
    "handle": MappingProxyType({"hex": HANDLE, "rgb": HANDLE_RGB, "line": SOLID}),
    "suggestion": MappingProxyType({"hex": SUGGESTION, "rgb": SUGGESTION_RGB, "line": DASH}),
})
DISTINCT = ("sky", "ground", "milkyway", "trail", "horizon")

# Trail review (2026-09-27): each identity has its own colour, initially from the algorithm
# suggestion and then following the reviewer's choice. The same colour-blind-safe set plus
# white for "unknown"; the HANDLE yellow ring marks the selection. With HANDLE the six colours
# stay >= 13.9 dE2000 apart under normal, protan, deutan and tritan simulation
# (outputs/ui-research-20260927/domain/overlay_palette_check.py method). Line style carries
# the disposition: kept (added back) = solid, omitted = dotted.
TRAIL_UNKNOWN = "#FFFFFF"
TRAIL_IDENTITY = MappingProxyType({
    "meteor": MappingProxyType({"hex": TRAIL, "rgb": TRAIL_RGB}),
    "satellite": MappingProxyType({"hex": SKY, "rgb": SKY_RGB}),
    "aircraft": MappingProxyType({"hex": GROUND, "rgb": GROUND_RGB}),
    "artifact": MappingProxyType({"hex": MILKYWAY, "rgb": MILKYWAY_RGB}),
    "unknown": MappingProxyType({"hex": TRAIL_UNKNOWN, "rgb": _rgb(TRAIL_UNKNOWN)}),
})
TRAIL_SELECTED = HANDLE
TRAIL_KEPT_LINE = SOLID
TRAIL_OMITTED_LINE = DOT


def rgba(name: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """(r, g, b, a) for an overlay name, e.g. a translucent fill of the ground region."""
    r, g, b = OVERLAYS[name]["rgb"]
    return r, g, b, int(alpha)
