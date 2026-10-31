"""Launch the checked-out NL-1 development tools without installing the project.

Uses only this checkout's source path; does not alter the machine environment.
Host roots remain explicit and overridable. No GIMP process is started here.
"""
from pathlib import Path
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    mode = args.pop(0) if args and args[0] in {"gui", "cli", "mcp"} else "gui"
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    defaults = {"--input-root": root, "--output-root": root / "outputs/retouch-user/photos",
                "--cache-root": root / "outputs/retouch-user/work"}
    prefix = []
    for flag, value in defaults.items():
        if not any(item == flag or item.startswith(flag + "=") for item in args):
            prefix += [flag, str(value)]
    if mode == "gui":
        from aester_desktop.finishing_gui import main as entry
    elif mode == "cli":
        from starstack.finishing_cli import main as entry
    else:
        from starstack.finishing_mcp import main as entry
    return entry(prefix + args)


if __name__ == "__main__":
    raise SystemExit(main())
