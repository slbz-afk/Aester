"""Write a radial lens-distortion model as a WCS + SIP header for Siril's ``register ... -disto=file``.

Proved on Siril 1.4.4 with a synthetic sequence (RUN_LOG §19): Siril accepts a
FITS file carrying ``CTYPE -SIP``, CRPIX/CRVAL/CD, ``A_ORDER``/``A_p_q``,
``B_*``, ``AP_*``, ``BP_*`` (all terms written explicitly, zero included, as
Siril writes them), corrects star positions with the forward polynomials
before fitting the homographies and undistorts the exported frames with the
inverse ones.  CRPIX follows the standard FITS convention (1-based, rows
counted from the bottom); the radial polynomial is invariant under that
flip, so only the centre is converted.

Model (``registration_distortion.RadialDistortion``, TOP-DOWN zero-based
pixel centres), direction ``p_to_u``: ``u = c + m (p - c) (1 + k1 r^2 + k2
r^4)``, ``r = |p - c| / scale``.  In SIP terms (offsets ``U, V`` from CRPIX):
``u' = U + f(U, V)`` with ``f = (m - 1) U + m k1 U (U^2 + V^2) / scale^2 + m k2
U (U^2 + V^2)^2 / scale^4`` and the same for ``V``; the linear ``m - 1`` terms
carry the global magnification that keeps a barrel lens' corners inside the
native box.  The inverse (AP/BP) is then a least-squares polynomial fit of
``p - u`` on a grid — order 5 at most: Siril 1.4.4 crashes on higher orders
(RUN_LOG §22), and order 5 leaves 0.96 px at the extreme corners of the Z8
20 mm model.

Direction ``u_to_p`` (the product default since RUN_LOG §22): the polynomial
acts on the undistorted side, ``p = c + (u - c)(1 + k1 r^2 + k2 r^4)`` with
``r = |u - c| / scale`` and ``u' = c + m (u - c)``; the *inverse* AP/BP are
then exact — ``p = U' + g(U', V')`` with ``g = (1/m - 1) U' + k1 U' (U'^2 +
V'^2) / (m^3 scale^2) + k2 U' (U'^2 + V'^2)^2 / (m^5 scale^4)`` — and the
forward A/B (only used by Siril's own ``register``, which the product no
longer runs) are the least-squares fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .registration_distortion import RadialDistortion

__all__ = ["SIP_HEADER_VERSION", "PIXEL_SCALE_DEG", "DEFAULT_INVERSE_ORDER", "MAX_SIRIL_SIP_ORDER", "sip_forward_coefficients",
           "sip_inverse_coefficients", "sip_cards", "write_wcs_header_file", "fits_card"]

SIP_HEADER_VERSION = "1.2.0"
PIXEL_SCALE_DEG = 0.0125  # 45"/px (Z8 + 20 mm); only the SIP polynomials matter for registration


def fits_card(key: str, value, comment: str = "") -> str:
    if isinstance(value, bool):
        card = f"{key:<8}= {'T' if value else 'F':>20}"
    elif isinstance(value, int):
        card = f"{key:<8}= {value:>20d}"
    elif isinstance(value, float):
        card = f"{key:<8}= {value:>20.10G}"
    else:
        card = f"{key:<8}= '{str(value):<8}'"
    if comment:
        card = f"{card} / {comment}"
    return card[:80].ljust(80)


def _poly_design(u: np.ndarray, v: np.ndarray, order: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    powers = [(p, q) for p in range(order + 1) for q in range(order + 1 - p)]
    return np.stack([u ** p * v ** q for p, q in powers], axis=1), powers


def _exact_radial_terms(k1: float, k2: float, lin: float, c1: float, c2: float) -> tuple[dict, dict, int]:
    """Polynomial of ``x (lin) + c1 x |x|^2 + c2 x |x|^4`` in (p, q) powers for both axes."""
    a = {(3, 0): c1, (1, 2): c1}
    b = {(0, 3): c1, (2, 1): c1}
    if k2 != 0.0:
        a.update({(5, 0): c2, (3, 2): 2 * c2, (1, 4): c2})
        b.update({(0, 5): c2, (2, 3): 2 * c2, (4, 1): c2})
    if lin != 0.0:
        a[(1, 0)] = lin
        b[(0, 1)] = lin
    return a, b, (5 if k2 != 0.0 else 3)


def _fit_polynomial(source_offsets: np.ndarray, delta: np.ndarray, scale: float, order: int) -> tuple[dict, dict, float]:
    """Least-squares polynomial of ``delta`` as a function of the offsets (normalised by ``scale`` for conditioning)."""
    design, powers = _poly_design(source_offsets[:, 0] / scale, source_offsets[:, 1] / scale, order)
    a_n = np.linalg.lstsq(design, delta[:, 0], rcond=None)[0]
    b_n = np.linalg.lstsq(design, delta[:, 1], rcond=None)[0]
    fit_error = float(np.max(np.hypot(design @ a_n - delta[:, 0], design @ b_n - delta[:, 1])))
    a = {pq: c / scale ** (pq[0] + pq[1]) for c, pq in zip(a_n, powers)}   # back to raw pixel offsets
    b = {pq: c / scale ** (pq[0] + pq[1]) for c, pq in zip(b_n, powers)}
    return a, b, fit_error


def _grid(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0, width - 1, 25)
    ys = np.linspace(0, height - 1, 19)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float64)


def sip_forward_coefficients(distortion: RadialDistortion, width: int | None = None, height: int | None = None, order: int = 5) -> dict:
    """A/B coefficients of ``u' = U + f(U, V)``: exact for ``p_to_u``; a least-squares fit (needs ``width``/``height``) for ``u_to_p``."""
    s2, s4 = distortion.scale ** 2, distortion.scale ** 4
    k1, k2, m = distortion.k1, distortion.k2, distortion.magnification
    if distortion.direction == "p_to_u":
        a, b, deg = _exact_radial_terms(k1, k2, m - 1.0, m * k1 / s2, m * k2 / s4)
        return {"A": a, "B": b, "order": deg, "max_fit_error_px": 0.0, "exact": True}
    if width is None or height is None:
        raise ValueError("width and height are needed to fit the forward polynomial of a u_to_p model")
    grid = _grid(width, height)
    delta = distortion.apply(grid) - grid                    # u' - p as a function of p offsets
    a, b, err = _fit_polynomial(grid - distortion.centre, delta, distortion.scale, order)
    return {"A": a, "B": b, "order": order, "max_fit_error_px": err, "exact": False}


def sip_inverse_coefficients(distortion: RadialDistortion, width: int, height: int, order: int = 5) -> dict:
    """AP/BP coefficients of ``p = U' + g(U', V')``: exact for ``u_to_p``; a least-squares fit on a grid for ``p_to_u``."""
    s2, s4 = distortion.scale ** 2, distortion.scale ** 4
    k1, k2, m = distortion.k1, distortion.k2, distortion.magnification
    if distortion.direction == "u_to_p":
        ap, bp, deg = _exact_radial_terms(k1, k2, 1.0 / m - 1.0, k1 / (m ** 3 * s2), k2 / (m ** 5 * s4))
        return {"AP": ap, "BP": bp, "order": deg, "max_fit_error_px": 0.0, "exact": True}
    grid = _grid(width, height)
    u = distortion.apply(grid) - distortion.centre          # undistorted offsets
    delta = grid - distortion.centre - u                     # p - u (offsets)
    ap, bp, err = _fit_polynomial(u, delta, distortion.scale, order)
    return {"AP": ap, "BP": bp, "order": order, "max_fit_error_px": err, "exact": False}


def sip_apply_forward(coeffs: dict, xy_fits: np.ndarray, crpix: tuple[float, float]) -> np.ndarray:
    """Evaluate ``(U + f, V + g) + CRPIX`` for FITS 1-based pixel coordinates (used by tests)."""
    u = xy_fits[:, 0] - crpix[0]
    v = xy_fits[:, 1] - crpix[1]
    f = sum(c * u ** p * v ** q for (p, q), c in coeffs["A"].items())
    g = sum(c * u ** p * v ** q for (p, q), c in coeffs["B"].items())
    return np.column_stack([u + f + crpix[0], v + g + crpix[1]])


DEFAULT_INVERSE_ORDER = 5   # RUN_LOG §22: Siril 1.4.4 crashes on AP/BP orders above 5 (order-9 file: accepted, then crash)
MAX_SIRIL_SIP_ORDER = 5


def sip_cards(distortion: RadialDistortion, width: int, height: int, *, pixel_scale_deg: float = PIXEL_SCALE_DEG,
              inverse_order: int = DEFAULT_INVERSE_ORDER, crpix2_convention: str = "fits_bottom_up") -> list[str]:
    """Complete WCS + SIP header cards for a bottom-up FITS of ``width x height``.

    ``crpix2_convention``: ``fits_bottom_up`` (the one Siril honours, RUN_LOG
    §19) writes CRPIX2 as the 1-based bottom-up row of the distortion centre;
    ``top_down`` is kept for the proof only.  ``inverse_order`` is the degree
    of the least-squares polynomial (AP/BP for ``p_to_u`` models, A/B for
    ``u_to_p`` ones); the other direction is exact.  Orders above 5 are
    written only when asked for explicitly (Siril 1.4.4 crashes on them).
    """
    forward = sip_forward_coefficients(distortion, width, height, inverse_order)
    inverse = sip_inverse_coefficients(distortion, width, height, inverse_order)
    crpix1 = distortion.cx + 1.0
    if crpix2_convention == "fits_bottom_up":
        crpix2 = height - distortion.cy
    elif crpix2_convention == "top_down":
        crpix2 = distortion.cy + 1.0
    else:
        raise ValueError(crpix2_convention)
    cards = [fits_card("WCSAXES", 2), fits_card("CTYPE1", "RA---TAN-SIP"), fits_card("CTYPE2", "DEC--TAN-SIP"),
             fits_card("CUNIT1", "deg"), fits_card("CUNIT2", "deg"),
             fits_card("CRPIX1", float(crpix1), "distortion centre, FITS 1-based"),
             fits_card("CRPIX2", float(crpix2), "distortion centre, FITS 1-based, bottom-up"),
             fits_card("CRVAL1", 180.0, "arbitrary: only the SIP terms matter"), fits_card("CRVAL2", 0.0, "arbitrary"),
             fits_card("CD1_1", -float(pixel_scale_deg)), fits_card("CD1_2", 0.0), fits_card("CD2_1", 0.0), fits_card("CD2_2", float(pixel_scale_deg)),
             fits_card("EQUINOX", 2000.0), fits_card("RADESYS", "ICRS"),
             fits_card("A_ORDER", forward["order"]), fits_card("B_ORDER", forward["order"])]
    order = forward["order"]
    for prefix, coeffs in (("A", forward["A"]), ("B", forward["B"])):
        for total in range(order + 1):
            for p in range(total, -1, -1):
                q = total - p
                cards.append(fits_card(f"{prefix}_{p}_{q}", float(coeffs.get((p, q), 0.0))))
    cards += [fits_card("AP_ORDER", inverse["order"]), fits_card("BP_ORDER", inverse["order"])]
    for prefix, coeffs in (("AP", inverse["AP"]), ("BP", inverse["BP"])):
        for total in range(inverse["order"] + 1):
            for p in range(total, -1, -1):
                q = total - p
                cards.append(fits_card(f"{prefix}_{p}_{q}", float(coeffs.get((p, q), 0.0))))
    cards.append(fits_card("PLTSOLVD", True, "synthetic solution written by starstack (radial lens model)"))
    cards.append(fits_card("DISTOMAG", float(distortion.magnification), "global magnification of the undistorted grid"))
    return cards


def write_wcs_header_file(path: Path, cards: list[str], width: int, height: int) -> None:
    """Header-only FITS like Siril's own ``.wcs`` distortion files (NAXIS 0, IMAGEW/IMAGEH, WCS + SIP)."""
    head = [fits_card("SIMPLE", True, "conforms to FITS standard"), fits_card("BITPIX", 8, "ASCII or bytes array"),
            fits_card("NAXIS", 0, "Minimal header"), fits_card("EXTEND", True), fits_card("IMAGEW", int(width), "Image width in pixels"),
            fits_card("IMAGEH", int(height), "Image height in pixels")]
    header = "".join(head + list(cards) + ["END".ljust(80)])
    header += " " * ((2880 - len(header) % 2880) % 2880)
    Path(path).write_bytes(header.encode("ascii"))
