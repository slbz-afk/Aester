"""Stand-in Siril behaviour for the EFF-D4 stack-settings contract (efficiency P1c §4.2), shared by controlled engines.

A contract stack script carries ``setcpu T'`` / ``set core.mem_mode=1`` / ``set core.mem_amount=B`` right before
``seqclean``; real Siril then prints, untranslated, ``Memory per image: p MB. Max memory: B*1024 MB`` and
``allocating data for T' threads``, and core checks exactly those lines in the attempt's log section.  A controlled
engine calls :func:`stack_log` after it "stacks" so that the product's checks see what Siril would print.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from starstack import stack_settings

SEQUENCE = stack_settings.SEQUENCE_NAME


@contextmanager
def admitted_before_contract(active: bool = True):
    """Submissions inside are admitted as before EFF-D4 (no stack-settings field): a job of an earlier version, which
    keeps its scripts byte for byte.  ``active=False`` admits normally."""
    if not active:
        yield
        return
    original = stack_settings.admission_field
    stack_settings.admission_field = lambda *args, **kwargs: None
    try:
        yield
    finally:
        stack_settings.admission_field = original


def settings_of(commands):
    """(T', B) of a contract script, or None for a script without the settings lines."""
    return stack_settings.without_settings(list(commands))[1]


def sequence_text(count: int, selected=None, name: str = "r_light_", reference: int = 0) -> str:
    """A Siril 1.4.4 sequence file (version 6) of ``count`` images; ``selected``: 1-based selected indices."""
    chosen = set(range(1, count + 1) if selected is None else selected)
    lines = ["#Siril sequence file. Contains list of images, selection, registration data and statistics",
             f"S '{name}' 1 {count} {len(chosen)} 5 {reference} 6 0 0 0", "L 3"]
    lines += [f"I {i} {1 if i in chosen else 0}" for i in range(1, count + 1)]
    return "\n".join(lines) + "\n"


def stack_log(folder: Path, name: str, commands, *, work: Path | None = None, memory_mb=None, threads=None,
              extra: str = "") -> None:
    """Append Siril's two stack lines for a contract script to ``folder/<name>.log`` (nothing for other scripts).

    ``work``: where r_light_.seq and the frames are (default: ``folder/work`` for 02_stack, else ``folder``);
    ``memory_mb`` / ``threads`` override what is printed (to model a Siril that did something else)."""
    found = settings_of(commands)
    if found is None:
        return
    count, budget = found
    work = Path(work) if work is not None else (Path(folder) / "work" if name == "02_stack" else Path(folder))
    _, first = stack_settings.selected_frames(work / SEQUENCE)
    per_frame = stack_settings.frame_mb(*stack_settings.frame_geometry(work / f"{first}.fit"))
    memory = memory_mb if memory_mb is not None else (per_frame, budget * 1024)
    text = (f"running command stack\nMemory per image: {memory[0]} MB. Max memory: {memory[1]} MB\n"
            f"allocating data for {count if threads is None else threads} threads (each 1 MB)\n{extra}")
    with (Path(folder) / f"{name}.log").open("a", encoding="utf-8") as handle:
        handle.write(text)
