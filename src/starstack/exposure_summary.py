"""Pure summary of exposure-time metadata; no file or pixel handling."""

import math

__all__ = ["summarize_exposures"]


def _checked_exposure(value, position):
    """Validate one exposure entry and return a float or None."""
    if value is None:
        return None

    if type(value) is int:
        if value <= 0:
            raise ValueError(f"exposures[{position}] must be positive and finite")
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError(
                f"exposures[{position}] cannot be converted to a finite float"
            ) from exc
        if not math.isfinite(result):
            raise ValueError(
                f"exposures[{position}] cannot be converted to a finite float"
            )
        return result

    if type(value) is float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"exposures[{position}] must be positive and finite")
        return value

    raise ValueError(f"exposures[{position}] must be None, int, or float")


def _checked_indices(included_indices, exposure_count):
    """Validate selected indices and return them in caller order."""
    if type(included_indices) is not list:
        raise ValueError("included_indices must be a list")

    seen = set()
    result = []
    for position, index in enumerate(included_indices):
        if type(index) is not int:
            raise ValueError(f"included_indices[{position}] must be an int")
        if not (0 <= index < exposure_count):
            raise ValueError(
                f"included_indices[{position}] index {index} is out of range"
            )
        if index in seen:
            raise ValueError(
                f"included_indices[{position}] index {index} is repeated"
            )
        seen.add(index)
        result.append(index)
    return result


def _fsum(values):
    """Return math.fsum(values), mapping overflow to ValueError."""
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError("exposure total overflowed") from exc
    if not math.isfinite(total):
        raise ValueError("exposure total overflowed")
    return total


def summarize_exposures(exposures: list, included_indices: list) -> dict:
    """Summarize exposure counts and totals without inferring membership."""
    if type(exposures) is not list:
        raise ValueError("exposures must be a list")

    values = []
    unknown_input_count = 0
    for position, value in enumerate(exposures):
        checked = _checked_exposure(value, position)
        values.append(checked)
        if checked is None:
            unknown_input_count += 1

    chosen = _checked_indices(included_indices, len(exposures))

    if unknown_input_count:
        input_total_seconds = None
    else:
        input_total_seconds = _fsum(values)

    included_values = []
    unknown_included_count = 0
    for index in chosen:
        value = values[index]
        if value is None:
            unknown_included_count += 1
        else:
            included_values.append(value)

    if unknown_included_count:
        included_total_seconds = None
    else:
        included_total_seconds = _fsum(included_values)

    return {
        "input_count": len(exposures),
        "included_count": len(included_indices),
        "input_total_seconds": input_total_seconds,
        "included_total_seconds": included_total_seconds,
        "unknown_input_count": unknown_input_count,
        "unknown_included_count": unknown_included_count,
    }
