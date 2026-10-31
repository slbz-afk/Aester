"""One-shot codemod for the open/closed split (S2). Internal tool; not shipped or exported.

Moves the desktop GUI modules and assets from ``src/starstack`` to the closed package
``src/aester_desktop`` (with ``git mv`` so history follows), then rewrites imports:

* inside moved modules: ``from .X`` (X open) -> ``from starstack.X``; ``from . import a, b`` is split
  into open names (``from starstack import ...``) and closed names (kept relative);
  ``starstack.<closed>`` in strings (e.g. ``-m starstack.gui``) -> ``aester_desktop.<closed>``;
* in tests and scripts: ``starstack.<closed>`` -> ``aester_desktop.<closed>`` (imports, monkeypatch targets,
  ``-m`` strings), ``from starstack import ... <closed> ...`` split into two statements, and the path literals
  ``src/starstack/<closed>.py`` and ``src/starstack/assets`` moved to ``src/aester_desktop``.

Every byte outside the rewritten spans is preserved. Open modules under ``src/starstack`` and ``packaging/``
are not rewritten here (their few references are changed by hand in the same commit).
Run from the repository root: ``python scripts/split/split_desktop.py [--dry-run]``.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

CLOSED = ("gui", "advanced_tools", "experience", "finishing_gui", "region_canvas", "region_dialog",
          "trail_review_dialog", "ui_widgets", "themes", "fonts", "backend_preferences",
          "translations_gui", "translations_dialogs")
OPEN_PKG, CLOSED_PKG = "starstack", "aester_desktop"
ROOT = Path(__file__).resolve().parents[2]
DOTTED = re.compile(r"\bstarstack\.(" + "|".join(CLOSED) + r")\b")
PATHS = [(re.compile(r"src([/\\]{1,2})starstack\1(" + "|".join(CLOSED) + r")\.py"), r"src\1aester_desktop\1\2.py"),
         (re.compile(r"src([/\\]{1,2})starstack\1assets"), r"src\1aester_desktop\1assets")]


def _read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as stream:   # keep CRLF/LF exactly
        return stream.read()


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _git(*args):
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def _spans(text: str, nodes) -> list[tuple[int, int, ast.AST]]:
    """Character spans (start, end, node) of the given statements in ``text``."""
    offsets = [0]                      # line starts, splitting on LF only (as the tokenizer counts lines)
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    offsets.append(len(text))

    def at(lineno, col):
        # ast columns are UTF-8 byte offsets; convert within the line
        line_start = offsets[lineno - 1]
        line = text[line_start:offsets[lineno]]
        return line_start + len(line.encode("utf-8")[:col].decode("utf-8"))
    return [(at(n.lineno, n.col_offset), at(n.end_lineno, n.end_col_offset), n) for n in nodes]


def _alias(a: ast.alias) -> str:
    return a.name + (f" as {a.asname}" if a.asname else "")


def _replace(text: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, new in sorted(edits, reverse=True):
        text = text[:start] + new + text[end:]
    return text


def rewrite_moved(text: str) -> str:
    """Imports inside a module that moved into the closed package."""
    tree = ast.parse(text)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1]
    edits = []
    for start, end, node in _spans(text, nodes):
        segment = text[start:end]
        if node.module is not None:
            if node.module.split(".")[0] in CLOSED:
                continue
            new = re.sub(r"\Afrom\s+\.", f"from {OPEN_PKG}.", segment, count=1)
            assert new != segment, segment
            edits.append((start, end, new))
            continue
        closed = [a for a in node.names if a.name in CLOSED]
        opened = [a for a in node.names if a.name not in CLOSED]
        if not opened:
            continue
        indent = " " * node.col_offset
        parts = [f"from {OPEN_PKG} import " + ", ".join(_alias(a) for a in opened)]
        if closed:
            parts.append(indent + "from . import " + ", ".join(_alias(a) for a in closed))
        edits.append((start, end, "\n".join(parts)))
    text = _replace(text, edits)
    # absolute ``from starstack import a, <closed>`` inside a moved module: closed names become relative
    tree = ast.parse(text)
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module == OPEN_PKG]
    edits = []
    for start, end, node in _spans(text, nodes):
        closed = [a for a in node.names if a.name in CLOSED]
        if not closed:
            continue
        opened = [a for a in node.names if a.name not in CLOSED]
        indent = " " * node.col_offset
        parts = ([f"from {OPEN_PKG} import " + ", ".join(_alias(a) for a in opened)] if opened else [])
        parts.append("from . import " + ", ".join(_alias(a) for a in closed))
        edits.append((start, end, ("\n" + indent).join(parts)))
    text = _replace(text, edits)
    return DOTTED.sub(lambda m: f"{CLOSED_PKG}.{m.group(1)}", text)


def rewrite_user(text: str) -> str:
    """Tests and scripts that use the closed modules."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    edits = []
    if tree is not None:
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module == OPEN_PKG]
        for start, end, node in _spans(text, nodes):
            closed = [a for a in node.names if a.name in CLOSED]
            if not closed:
                continue
            opened = [a for a in node.names if a.name not in CLOSED]
            indent = " " * node.col_offset
            parts = []
            if opened:
                parts.append(f"from {OPEN_PKG} import " + ", ".join(_alias(a) for a in opened))
            parts.append(f"from {CLOSED_PKG} import " + ", ".join(_alias(a) for a in closed))
            edits.append((start, end, ("\n" + indent).join(parts)))
    text = _replace(text, edits)
    text = DOTTED.sub(lambda m: f"{CLOSED_PKG}.{m.group(1)}", text)
    for pattern, replacement in PATHS:
        text = pattern.sub(replacement, text)
    return text


def main(dry_run: bool = False) -> int:
    src, dst = ROOT / "src" / OPEN_PKG, ROOT / "src" / CLOSED_PKG
    missing = [m for m in CLOSED if not (src / f"{m}.py").is_file()]
    if missing and not dry_run:
        raise SystemExit("closed modules not found (already moved?): " + ", ".join(missing))
    if not dry_run:
        dst.mkdir(exist_ok=True)
        for module in CLOSED:
            _git("mv", f"src/{OPEN_PKG}/{module}.py", f"src/{CLOSED_PKG}/{module}.py")
        _git("mv", f"src/{OPEN_PKG}/assets", f"src/{CLOSED_PKG}/assets")
    changed = []
    for module in CLOSED:
        path = (dst if not dry_run else src) / f"{module}.py"
        old = _read(path)
        new = rewrite_moved(old)
        if new != old:
            changed.append(path)
            if not dry_run:
                _write(path, new)
    users = [*sorted((ROOT / "tests").rglob("*.py")), *sorted((ROOT / "scripts").rglob("*.py"))]
    for path in users:
        if "split" in path.relative_to(ROOT).parts:
            continue
        old = _read(path)
        new = rewrite_user(old)
        if new != old:
            changed.append(path)
            if not dry_run:
                _write(path, new)
    for path in changed:
        print(path.relative_to(ROOT))
    print(f"{len(changed)} files {'would change' if dry_run else 'changed'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry-run" in sys.argv[1:]))
