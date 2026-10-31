"""Parsers for Siril sequence files (``.seq``) and star lists (``.lst``), plus a
registration writer for the sequence file.

Scope: extract the per-image registration homographies that Siril 1.4 stores
in the *original* sequence file after ``register``, the image index binding
recorded by its ``I`` lines, and the star catalogues Siril caches in
``cache/<image>.lst``.  Nothing here interprets coordinate conventions: the
values are returned exactly as written, tagged with their source file and
SHA256, so that a separate, explicitly verified coordinate hypothesis can be
applied by the registration adapter.  ``write_registration`` does the reverse
for the product's own registration (RUN_LOG §22): it replaces the ``H`` of the
``R<layer>`` lines, the selection flags and the ``D<layer>`` distortion line of
an existing sequence file and leaves every other line verbatim, so that
``seqapplyreg`` exports the frames with homographies computed outside Siril.

Facts this module relies on (Siril 1.4.4 sequence file version 6, as seen in
the real ``light_.seq`` of the Z8 smoke job):

* ``S 'name' start_index nb_images nb_selected fixed_len reference_image
  version variable_size fz_flag drizzle`` – ``reference_image`` is a 0-based
  position in the image list.
* ``L n`` – number of layers (3 for RGB).
* ``I index selected [width height]`` – one line per image, in sequence
  order; the file name is ``<name><index zero-padded to fixed_len>.<ext>``.
* ``R<layer> fwhm weighted_fwhm roundness quality background_lvl
  number_of_stars H h00 h01 h02 h10 h11 h12 h20 h21 h22`` – one line per
  image and layer, in image order.  ``H`` maps the frame onto the reference
  frame according to the sirilpy ``Homography`` documentation; row order and
  pixel origin are *not* documented and must be proven experimentally.
* ``M<layer>-<position> …`` – image statistics; kept only as raw numbers.

A registered output sequence (``r_<name>.seq``) carries reset identity
matrices and must not be used as a source of transforms; callers should pass
the original sequence.  The parser records ``all_identity`` so that mistake is
visible.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from . import hashing

__all__ = ["SEQ_PARSER_VERSION", "SequenceError", "parse_seq_text", "read_seq",
           "parse_lst_text", "read_lst", "image_stem", "write_registration_text", "write_registration"]

SEQ_PARSER_VERSION = "1.1.0"

_S_LINE = re.compile(r"^S\s+'(?P<name>[^']*)'\s+(?P<rest>.*)$")


class SequenceError(ValueError):
    """The file is not a sequence/star list this parser understands."""


def _sha256(path: Path) -> str:
    path.read_bytes  # a value without Path.read_bytes fails as before (AttributeError)
    return hashing.sha256_file(path, source="siril_sequence._sha256")


def _numbers(tokens, count, what):
    if len(tokens) < count:
        raise SequenceError(f"{what}: expected {count} numbers, found {len(tokens)}")
    try:
        return [float(t) for t in tokens[:count]]
    except ValueError as exc:
        raise SequenceError(f"{what}: non-numeric token") from exc


def image_stem(name: str, index: int, fixed_len: int) -> str:
    """Siril file stem for image ``index`` of sequence ``name`` (e.g. light_00001)."""
    return f"{name}{index:0{fixed_len}d}"


def parse_seq_text(text: str) -> dict:
    """Parse the text of a Siril ``.seq`` file into a plain dictionary.

    Unknown line types are kept verbatim under ``unparsed`` rather than
    rejected, so that a newer Siril cannot silently change what is read.
    """
    header = None
    layers = None
    images = []
    regdata = {}      # layer -> list of records in image order
    stats = []        # raw M lines
    unparsed = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tag = line.split(None, 1)[0]
        if tag == "S":
            match = _S_LINE.match(line)
            if match is None:
                raise SequenceError("malformed S line")
            values = match.group("rest").split()
            numbers = _numbers(values, 9, "S line")
            if any(v != math.floor(v) for v in numbers):
                raise SequenceError("S line fields must be integers")
            fields = [int(v) for v in numbers]
            header = {"name": match.group("name"), "start_index": fields[0], "nb_images": fields[1],
                      "nb_selected": fields[2], "fixed_len": fields[3], "reference_image": fields[4],
                      "version": fields[5], "variable_size": fields[6], "fz_flag": fields[7],
                      "drizzle": fields[8]}
        elif tag == "L":
            layers = int(_numbers(line.split()[1:], 1, "L line")[0])
        elif tag == "I":
            tokens = line.split()[1:]
            numbers = _numbers(tokens, 2, "I line")
            record = {"index": int(numbers[0]), "selected": bool(int(numbers[1]))}
            if len(tokens) >= 4:
                record["width"], record["height"] = int(float(tokens[2])), int(float(tokens[3]))
            images.append(record)
        elif tag.startswith("R") and tag[1:].isdigit():
            layer = int(tag[1:])
            tokens = line.split()[1:]
            if len(tokens) < 16 or tokens[6] != "H":
                raise SequenceError(f"R{layer} line must contain six quality fields, 'H' and nine coefficients")
            quality = _numbers(tokens[:6], 6, f"R{layer} quality fields")
            h = _numbers(tokens[7:16], 9, f"R{layer} homography")
            if not all(math.isfinite(v) for v in h):
                raise SequenceError(f"R{layer} homography contains non-finite values")
            regdata.setdefault(layer, []).append({
                "fwhm": quality[0], "weighted_fwhm": quality[1], "roundness": quality[2],
                "quality": quality[3], "background_lvl": quality[4], "number_of_stars": int(quality[5]),
                "H": [h[0:3], h[3:6], h[6:9]]})
        elif tag.startswith("M") and "-" in tag:
            layer, position = tag[1:].split("-", 1)
            stats.append({"layer": int(layer), "position": int(position), "values": line.split()[1:]})
        else:
            unparsed.append(raw)
    if header is None:
        raise SequenceError("missing S line")
    if not images:
        raise SequenceError("no I lines")
    if len(images) != header["nb_images"]:
        raise SequenceError(f"S line announces {header['nb_images']} images but {len(images)} I lines found")
    if not 0 <= header["reference_image"] < len(images):
        raise SequenceError("reference_image is outside the image list")
    for layer, records in regdata.items():
        if len(records) != len(images):
            raise SequenceError(f"layer {layer} has {len(records)} R lines for {len(images)} images")
    positions = {}
    for position, record in enumerate(images):
        stem = image_stem(header["name"], record["index"], header["fixed_len"])
        record["position"] = position
        record["stem"] = stem
        record["is_reference"] = position == header["reference_image"]
        positions[stem] = position
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    all_identity = {layer: all(r["H"] == identity for r in records) for layer, records in regdata.items()}
    return {"parser_version": SEQ_PARSER_VERSION, "header": header, "layers": layers, "images": images,
            "regdata": regdata, "all_identity_per_layer": all_identity, "stats": stats,
            "unparsed_lines": unparsed}


def read_seq(path) -> dict:
    """Parse a ``.seq`` file and attach its identity (path, size, SHA256)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    result = parse_seq_text(text)
    result["source"] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


# --------------------------------------------------------------------------- #
# Registration writer                                                         #
# --------------------------------------------------------------------------- #

def _fmt_number(value: float) -> str:
    text = f"{float(value):.12g}"
    return "0" if text in ("-0", "0.0", "-0.0") else text


def write_registration_text(text: str, homographies: dict, *, layer: int = 1, selected: dict | None = None,
                            distortion_file: str | None = None, reference_position: int | None = None,
                            quality: dict | None = None) -> str:
    """Return the sequence text with the ``R<layer>`` homographies replaced.

    ``homographies`` maps image *position* (0-based, sequence order) to a 3x3
    matrix in Siril's own convention (the values to store, as ``read_seq``
    returns them).  ``selected`` maps positions to a selection flag; positions
    missing from ``homographies`` must be deselected and get Siril's own form
    for an unregistered image (all-zero quality fields and matrix).  The six
    quality fields (fwhm, weighted fwhm, roundness, quality, background,
    number of stars) of registered images are kept from the existing line
    unless ``quality`` gives replacements for that position.
    ``distortion_file`` adds or replaces the ``D<layer> 2 <file>`` line that
    makes ``seqapplyreg`` undistort with that WCS+SIP file.  The ``S`` line's
    ``nb_selected`` (and ``reference_image`` when given) are updated.  Every
    other line is left verbatim.  Raises ``SequenceError`` on a malformed file
    or when a registered position has no matrix.
    """
    parsed = parse_seq_text(text)
    n_images = len(parsed["images"])
    selected = dict(selected or {})
    for position in range(n_images):
        selected.setdefault(position, position in homographies)
    for position in range(n_images):
        if selected[position] and position not in homographies:
            raise SequenceError(f"position {position} is selected but has no homography")
    for position, h in homographies.items():
        if not 0 <= position < n_images:
            raise SequenceError(f"homography for position {position} outside the image list")
        values = [float(v) for row in h for v in row]
        if len(values) != 9 or not all(math.isfinite(v) for v in values):
            raise SequenceError(f"homography for position {position} must be a finite 3x3 matrix")
    if reference_position is not None and not 0 <= reference_position < n_images:
        raise SequenceError("reference_position outside the image list")
    out = []
    r_position = 0
    saw_r = False
    d_written = False
    last_i_line = None
    lines = text.splitlines()
    for raw in lines:
        line = raw.strip()
        tag = line.split(None, 1)[0] if line and not line.startswith("#") else None
        if tag == "S":
            match = _S_LINE.match(line)
            values = match.group("rest").split()
            values[2] = str(sum(1 for p in range(n_images) if selected[p]))
            if reference_position is not None:
                values[4] = str(reference_position)
            out.append(f"S '{match.group('name')}' " + " ".join(values))
        elif tag == "I":
            tokens = line.split()
            index = int(float(tokens[1]))
            position = next(im["position"] for im in parsed["images"] if im["index"] == index)
            tokens[2] = "1" if selected[position] else "0"
            out.append(" ".join(tokens))
            last_i_line = len(out) - 1
        elif tag == f"R{layer}":
            saw_r = True
            tokens = line.split()
            if r_position in homographies:
                h = [float(v) for row in homographies[r_position] for v in row]
                fields = tokens[1:7]
                if quality and r_position in quality:
                    fields = [_fmt_number(v) for v in list(quality[r_position])[:6]]
                    if len(fields) != 6:
                        raise SequenceError(f"quality for position {r_position} must have six values")
            else:
                h = [0.0] * 9
                fields = ["0"] * 6
            out.append(f"R{layer} " + " ".join(fields) + " H " + " ".join(_fmt_number(v) for v in h))
            r_position += 1
        elif tag == f"D{layer}":
            if distortion_file is not None:
                out.append(f"D{layer} 2 {distortion_file}")
                d_written = True
            else:
                out.append(raw)
        else:
            out.append(raw)
    if not saw_r:
        raise SequenceError(f"sequence has no R{layer} lines to rewrite")
    if r_position != n_images:
        raise SequenceError(f"layer {layer} has {r_position} R lines for {n_images} images")
    if distortion_file is not None and not d_written:
        insert_at = (last_i_line + 1) if last_i_line is not None else len(out)
        out.insert(insert_at, f"D{layer} 2 {distortion_file}")
    return "\n".join(out) + "\n"


def write_registration(path, homographies: dict, **kwargs) -> dict:
    """Rewrite ``path`` in place (via a temporary file) and return the re-parsed sequence."""
    import os
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    new_text = write_registration_text(text, homographies, **kwargs)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    with hashing.writing(path):   # replaces a sequence file the run ledger may know
        os.replace(tmp, path)
    return read_seq(path)


# --------------------------------------------------------------------------- #
# Star lists                                                                  #
# --------------------------------------------------------------------------- #

_LST_COLUMN_ALIASES = {
    "star#": "index", "layer": "layer", "b": "B", "a": "A", "beta": "beta", "x": "x", "y": "y",
    "fwhmx [px]": "fwhmx_px", "fwhmy [px]": "fwhmy_px", 'fwhmx ["]': "fwhmx_arcsec",
    'fwhmy ["]': "fwhmy_arcsec", "angle": "angle", "rmse": "rmse", "mag": "mag", "sat": "sat",
    "profile": "profile", "ra": "ra", "dec": "dec",
}


def parse_lst_text(text: str) -> dict:
    """Parse a Siril star list as written next to registration caches.

    Coordinates are returned exactly as printed (Siril writes two decimals);
    their origin and row order are not interpreted here.
    """
    lines = text.splitlines()
    comments = [line for line in lines if line.startswith("#")]
    if not comments:
        raise SequenceError("star list has no header comments")
    header_line = None
    for line in comments:
        if "\t" in line and "X" in line and "Y" in line:
            header_line = line
    if header_line is None:
        raise SequenceError("star list has no column header")
    columns = [c.strip() for c in header_line.lstrip("#").strip().split("\t")]
    keys = [_LST_COLUMN_ALIASES.get(c.lower(), c) for c in columns]
    if "x" not in keys or "y" not in keys:
        raise SequenceError("star list columns do not include X and Y")
    parameters = {}
    for line in comments:
        for token in line.lstrip("#").split():
            if "=" in token:
                key, value = token.split("=", 1)
                parameters[key] = value
    declared = None
    match = re.match(r"#\s*(\d+)\s+stars found", comments[0])
    if match:
        declared = int(match.group(1))
    stars = []
    for raw in lines:
        if not raw.strip() or raw.startswith("#"):
            continue
        tokens = [t.strip() for t in raw.split("\t")]
        if len(tokens) != len(keys):
            raise SequenceError(f"star row has {len(tokens)} columns, header has {len(keys)}")
        record = {}
        for key, token in zip(keys, tokens):
            if key in ("index", "layer", "sat"):
                record[key] = int(float(token))
            elif key == "profile":
                record[key] = token
            else:
                try:
                    record[key] = float(token)
                except ValueError:
                    record[key] = token
        if not (math.isfinite(record["x"]) and math.isfinite(record["y"])):
            raise SequenceError("star position is not finite")
        stars.append(record)
    if declared is not None and declared != len(stars):
        raise SequenceError(f"header declares {declared} stars, {len(stars)} rows found")
    return {"parser_version": SEQ_PARSER_VERSION, "declared_count": declared, "parameters": parameters,
            "columns": columns, "stars": stars}


def read_lst(path) -> dict:
    path = Path(path)
    result = parse_lst_text(path.read_text(encoding="utf-8", errors="strict"))
    result["source"] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result
