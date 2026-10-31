"""Independent local CLI for replayable Aester finishing sessions."""
import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys


def add_host_arguments(parser):
    """Host configuration is parsed only when starting an entrypoint."""
    parser.add_argument("--input-root", help="Allowed source directory (or STARSTACK_INPUT_ROOT)")
    parser.add_argument("--output-root", help="Export directory (or STARSTACK_OUTPUT_ROOT)")
    parser.add_argument("--cache-root", help="Private work directory (or STARSTACK_CACHE_ROOT)")
    parser.add_argument("--gimp", help="Host-selected GIMP executable (or AESTER_GIMP)")


def configured_service(args):
    from .mcp_server import WorkspaceTools
    from .finishing_service import FinishingTools
    workspace = WorkspaceTools(input_root=args.input_root, output_root=args.output_root,
                               cache_root=args.cache_root)
    return FinishingTools(workspace, gimp_path=args.gimp)


def _finite(value):
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a finite number") from exc
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("Expected a finite number")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Aester replayable finishing; local files remain local")
    add_host_arguments(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities", help="List supported finishing capabilities and limitations")
    create = sub.add_parser("create", help="Register an sRGB16 source and a text intention")
    create.add_argument("--source", required=True)
    create.add_argument("--intent", required=True)
    for name in ("get", "add-reference", "add-mask", "add-layers", "draw-mask", "tonal-mask", "revise", "render",
                 "preview", "feedback", "export", "recover"):
        command = sub.add_parser(name)
        command.add_argument("--session", required=True, dest="session_dir")
        if name in {"add-reference", "add-mask"}:
            command.add_argument("--path", required=True)
            command.add_argument("--note", default="")
        if name == "add-reference":
            command.add_argument("--kind", required=True, choices=("original", "generated", "external"))
        if name in {"add-mask", "draw-mask", "tonal-mask"}:
            command.add_argument("--mask-id", required=True)
        if name == "add-mask":
            command.add_argument("--source-sha256", required=True)
        if name == "add-layers":
            for field in ("bundle-id", "background-path", "diffuse-path", "pure-sky-mask-id",
                          "transient-support-mask-id", "source-sha256", "provenance-file"):
                command.add_argument("--" + field, required=True)
        if name == "tonal-mask":
            command.add_argument("--knots", nargs=4, type=_finite, required=True, metavar=("A", "B", "C", "D"))
            command.add_argument("--candidate-id")
            command.add_argument("--region-mask-id")
            command.add_argument("--invert", action="store_true")
        if name == "draw-mask":
            command.add_argument("--polygons-file", required=True, help="UTF-8 JSON list of source-pixel polygons")
            command.add_argument("--feather-px", type=_finite, default=0.0)
            command.add_argument("--invert", action="store_true")
        if name in {"revise", "render"}:
            command.add_argument("--expected-revision", required=True, type=int)
        if name == "revise":
            command.add_argument("--recipe-file", required=True, help="UTF-8 JSON recipe object")
        if name == "render":
            command.add_argument("--timeout", type=_finite, default=600.0,
                                 help="External executor timeout in seconds")
        if name in {"preview", "feedback", "export"}:
            command.add_argument("--candidate-id", required=name != "preview")
        if name == "preview":
            command.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"),
                                 help="Optional [x, y, width, height] crop in source pixels")
        if name == "feedback":
            command.add_argument("--text", required=True)
            command.add_argument("--decision", default="observe", choices=("observe", "revise", "reject"),
                                 help="Recorded as agent feedback; cannot certify human acceptance")
    return parser


def _json_file(value, expected_type):
    def reject_constant(constant):
        raise ValueError(f"JSON must not contain {constant}")
    result = json.loads(Path(value).expanduser().read_text(encoding="utf-8-sig"),
                        parse_constant=reject_constant)
    if type(result) is not expected_type:
        raise ValueError(f"{value}: expected a JSON {'object' if expected_type is dict else 'array'}")
    return result


def _arguments(args):
    parameters = {key: value for key, value in vars(args).items()
                  if key not in {"command", "input_root", "output_root", "cache_root", "gimp",
                                 "recipe_file", "polygons_file", "provenance_file"}}
    if args.command == "revise":
        parameters["recipe"] = _json_file(args.recipe_file, dict)
    elif args.command == "draw-mask":
        parameters["polygons"] = _json_file(args.polygons_file, list)
    elif args.command == "add-layers":
        parameters["provenance"] = _json_file(args.provenance_file, dict)
    return parameters


def main(argv=None, *, service=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parameters = _arguments(args)
        # Keep the one stdout JSON response readable even if a dependency logs.
        with redirect_stdout(sys.stderr):
            selected = configured_service(args) if service is None else service
            try:
                result = getattr(selected, args.command.replace("-", "_"))(**parameters)
            finally:
                if service is None:
                    close = getattr(selected, "close", None)
                    if callable(close):
                        close()
        if type(result) is not dict:
            raise TypeError("Finishing service returned a non-object result")
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except KeyboardInterrupt:
        print("Finishing interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Finishing failed: {exc}", file=sys.stderr)
        return 1
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
