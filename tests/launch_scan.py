"""Static inventory of every process launch in the code base (efficiency P1b W1; used by test_siril_launch_sites.py).

Every load-context reference to a process-launch primitive in src/starstack, packaging and scripts is found with the
module's import aliases resolved, whether it is called or only referenced (an "escaping" reference such as
``runner = subprocess.run`` or ``partial(sp.run, ...)``). The primitives: subprocess.*, os.system/popen/startfile/
spawn*/exec*/posix_spawn*, asyncio.create_subprocess_*, process_guard.guarded_process, any *CreateProcess*/
ShellExecute*/WinExec attribute, and QProcess starts. A primitive is also found when it is reached through another
module's namespace (``siril.guarded_process``, ``core.subprocess.Popen``, ``from .siril import guarded_process``,
``import starstack.siril as s; s.os.startfile``): any canonical name that ends in .guarded_process,
.subprocess.<launcher>, .os.<launcher> or .asyncio.<launcher> counts (P1b review SIRIL-4). Keys are (file, enclosing
scope, canonical name, call|ref), the canonical name without a leading "starstack."; the pinned table in
test_siril_launch_sites.py is compared as a Counter, so a second launch inside an already pinned function fails too.
Aliases are a module-wide union: a local name that shadows an alias is also flagged (errs toward failing).

Regenerate the pinned table after a reviewed change (one command, from the repository root):

    .venv\\Scripts\\python.exe -X utf8 tests\\launch_scan.py --update

It rewrites the SITES block of tests/test_siril_launch_sites.py in place, keeps the comment of every unchanged key,
marks new keys "# NEW: say what it starts" (replace that with what it starts) and prints what was added or removed.
``--print`` only shows the block. A new launch that can start Siril must go through siril.run_script instead.
"""
from collections import Counter
from pathlib import Path
import ast
import re
import sys

SUBPROCESS = {"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}
OS = {"system", "popen", "startfile", "posix_spawn", "posix_spawnp"} | {
    f"{prefix}{suffix}" for prefix in ("spawn", "exec") for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")}
ASYNCIO = {"create_subprocess_exec", "create_subprocess_shell"}
WINAPI_PARTS = {"CreateProcess", "CreateProcessW", "CreateProcessA", "CreateProcessAsUserW", "CreateProcessAsUserA",
                "CreateProcessWithLogonW", "CreateProcessWithTokenW", "ShellExecuteW", "ShellExecuteExW",
                "ShellExecuteA", "ShellExecuteExA", "WinExec"}
LAUNCHERS = ({f"subprocess.{name}" for name in SUBPROCESS} | {f"os.{name}" for name in OS} |
             {f"asyncio.{name}" for name in ASYNCIO} | {"process_guard.guarded_process"})
# The same primitives reached through another module's namespace (siril.subprocess.run, siril.guarded_process): suffixes.
LAUNCHER_SUFFIXES = tuple(sorted({"." + name for name in LAUNCHERS} | {".guarded_process"}))
QPROCESS_STARTS = {"QProcess", "start", "startDetached", "execute", "startCommand"}
PACKAGE_PREFIXES = ("starstack.",)
ROOTS = ("src/starstack", "packaging", "scripts")
REPO = Path(__file__).resolve().parents[1]
TABLE = Path(__file__).resolve().parent / "test_siril_launch_sites.py"
BEGIN, END = "# BEGIN SITES (tests/launch_scan.py --update)", "# END SITES"


def _aliases(tree: ast.AST) -> dict:
    """Name bound in this module -> canonical dotted target (module-wide union; conservative)."""
    alias = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.asname:
                    alias[name.asname] = name.name
                else:
                    head = name.name.split(".")[0]
                    alias[head] = head
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in PACKAGE_PREFIXES:
                if module.startswith(prefix):
                    module = module[len(prefix):]
            if module == "starstack":
                module = ""
            for name in node.names:
                alias[name.asname or name.name] = f"{module}.{name.name}" if module else name.name
    return alias


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if parts:                          # e.g. self.api.CreateProcessW or call().attr: keep the attribute chain
        return "?." + ".".join(reversed(parts))
    return None


def canonical(name: str, alias: dict) -> str:
    head, _, rest = name.partition(".")
    head = alias.get(head, head)
    name = f"{head}.{rest}" if rest else head
    for prefix in PACKAGE_PREFIXES:            # import starstack.siril as s -> 'siril...', like 'from starstack import'
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def is_launcher(name: str) -> bool:
    if name in LAUNCHERS or name.endswith(LAUNCHER_SUFFIXES):
        return True
    parts = name.split(".")
    if any(part in WINAPI_PARTS for part in parts):
        return True
    return "QProcess" in parts and parts[-1] in QPROCESS_STARTS


def scan_source(source: str, rel: str, filename: str = "<source>"):
    """[((rel, scope, canonical, kind), node)] for one module's source."""
    tree = ast.parse(source, filename=filename)
    alias = _aliases(tree)
    annotations = set()
    for node in ast.walk(tree):
        for field in ("annotation", "returns"):
            sub = getattr(node, field, None)
            if sub is not None:
                annotations.update(id(n) for n in ast.walk(sub))
    found = []

    def walk(node, scope, parent):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            outer = list(node.decorator_list)
            if isinstance(node, ast.ClassDef):
                outer += node.bases + [keyword.value for keyword in node.keywords]
            else:
                outer += node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            for sub in outer:                       # evaluated in the enclosing scope
                walk(sub, scope, node)
            for statement in node.body:
                walk(statement, scope + [node.name], node)
            return
        if (isinstance(node, (ast.Attribute, ast.Name)) and isinstance(getattr(node, "ctx", None), ast.Load)
                and not isinstance(parent, ast.Attribute) and id(node) not in annotations):
            name = _dotted(node)
            if name:
                target = name if name.startswith("?.") else canonical(name, alias)
                if is_launcher(target):
                    kind = "call" if isinstance(parent, ast.Call) and parent.func is node else "ref"
                    found.append(((rel, ".".join(scope) or "<module>", target, kind),
                                  parent if kind == "call" else node))
        for child in ast.iter_child_nodes(node):
            walk(child, scope, node)

    walk(tree, [], None)
    return found


def scan_file(path: Path, rel: str):
    return scan_source(path.read_text(encoding="utf-8-sig"), rel, str(path))


def launch_sites(repo: Path = REPO, roots=ROOTS):
    found = []
    for root in roots:
        for path in sorted((Path(repo) / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            found.extend(scan_file(path, path.relative_to(repo).as_posix()))
    return found


def counter(found) -> Counter:
    return Counter(key for key, _ in found)


# --------------------------------------------------------------------------- regeneration

_ENTRY = re.compile(r"^    (\(.*\)): (\d+),\s*(#.*)?$")


def _block_lines(text: str):
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip() == BEGIN)
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == END)
    return lines, start, end


def render(seen: Counter, comments: dict) -> list[str]:
    body = []
    for key in sorted(seen):
        entry = f"    {key!r}: {seen[key]},"
        comment = comments.get(key) or "# NEW: say what it starts"
        body.append(f"{entry}  {comment}")
    return body


def update(table: Path = TABLE, repo: Path = REPO, *, write: bool = True) -> dict:
    text = table.read_bytes().decode("utf-8")
    lines, start, end = _block_lines(text)
    old, comments = Counter(), {}
    for line in lines[start + 1:end]:
        match = _ENTRY.match(line)
        if match:
            key = ast.literal_eval(match.group(1))
            old[key] = int(match.group(2))
            comments[key] = (match.group(3) or "").strip()
    seen = counter(launch_sites(repo))
    new_lines = lines[:start + 1] + render(seen, comments) + lines[end:]
    if write:
        table.write_bytes("\n".join(new_lines).encode("utf-8"))
    return {"added": dict(seen - old), "removed": dict(old - seen), "block": render(seen, comments)}


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--print"
    if mode not in ("--update", "--print"):
        raise SystemExit("usage: launch_scan.py --update | --print")
    result = update(write=mode == "--update")
    print("\n".join(result["block"]))
    print(f"added: {result['added']}\nremoved: {result['removed']}", file=sys.stderr)
