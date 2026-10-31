"""The transient-scan children run with OPENBLAS_NUM_THREADS=1 (efficiency P1a S-5).

That is bit-identical only because every BLAS-capable call the child makes has a result that does not depend on
OpenBLAS's thread count: 2x2 syrk and eigh, two-column gemv, dot products of length <= 5.
(a) pins the inventory of BLAS-capable call sites in the child's import closure: a new or moved site fails this
    test and asks for a re-audit before it ships;
(b) runs the child's code path on synthetic data under several thread counts and requires identical results, with a
    long ddot as the positive control, and checks that the parent-only sites never run in the child;
(c) checks that scan_frames gives every child one shared environment block: this process's, plus the one thread.
"""
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from starstack import imaging, process_guard, transient_scan as scan

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = SRC / "starstack"
BLAS_NAMES = frozenset({"dot", "vdot", "inner", "matmul", "einsum", "tensordot",
                        "convolve", "correlate", "polyfit", "cov", "corrcoef", "multi_dot", "lstsq", "solve",
                        "inv", "pinv", "eig", "eigh", "eigvals", "eigvalsh", "svd", "norm", "det", "slogdet",
                        "qr", "cholesky", "matrix_power"})
# (module, enclosing top-level function, operation) -> count. "@" is any matrix product, "linalg" any linalg access.
CHILD_SITES = {   # on the child path; why the thread count cannot matter
    ("transient_trails", "_axis", "@"): 2,                    # d.T @ d: syrk n=2 (split over 2 outputs); d @ u: gemv_t m=2
    ("transient_trails", "_axis", "eigh"): 1,                 # 2x2 LAPACK
    ("transient_trails", "_axis", "linalg"): 1,
    ("transient_trails", "_straightness", "@"): 2,            # gemv_t m=2: each output one 2-term expression
    ("transient_trails", "find_segments", "@"): 2,            # gemv_t m=2
    ("transient_trails", "_describe", "@"): 1,                # gemv_t m=2
    ("transient_trails", "_relative_position", "@"): 2,       # ddot n=2
    ("transient_trails", "marked_line_relation", "@"): 2,     # n=2 (manual marks only)
    ("transient_trails", "marked_line_relation", "linalg"): 2,
    ("transient_trails", "marked_line_relation", "norm"): 2,  # 2-vectors
    ("transient_trails", "measure_segment", "convolve"): 3,   # dtype dot over a window <= 5
}
PARENT_SITES = {  # in the child's modules but run only in the job worker (add_evidence, _candidate, warp)
    ("fits_crop", "warp_to_reference", "@"): 1,
    ("fits_crop", "warp_to_reference", "inv"): 1,
    ("fits_crop", "warp_to_reference", "linalg"): 1,
    ("transient_stage", "_apply_h", "@"): 1,
    ("transient_stage", "_native_endpoints", "inv"): 1,
    ("transient_stage", "_native_endpoints", "linalg"): 2,
    ("transient_trails", "_line_relation", "@"): 3,
    ("transient_trails", "_periodicity", "dot"): 2,           # long ddot: thread-count dependent, parent only
    ("transient_trails", "_light_curve", "convolve"): 1,
}
IMPORTS = ("import sys; sys.path.insert(0, sys.argv[1]); import starstack.transient_scan_worker; "
           "from starstack.transient_stage import _scan_frame; "
           "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('starstack.'))))")


def child_modules():
    out = subprocess.run([sys.executable, "-B", "-c", IMPORTS, str(SRC)], capture_output=True, text=True, check=True)
    return [line.split(".", 1)[1] for line in out.stdout.split()]


def blas_sites(module):
    found = Counter()

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = scope + [child.name] if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else scope
            op = None
            if isinstance(child, (ast.BinOp, ast.AugAssign)) and isinstance(child.op, ast.MatMult):
                op = "@"
            elif isinstance(child, ast.Call):
                name = getattr(child.func, "attr", None) or getattr(child.func, "id", None)
                op = name if name in BLAS_NAMES else None
            elif isinstance(child, ast.Attribute) and child.attr == "linalg":
                op = "linalg"
            if op:
                found[(module, (inner[0] if inner else "<module>"), op)] += 1
            visit(child, inner)

    visit(ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8")), [])
    return found


def test_the_scan_child_environment_is_this_process_plus_one_blas_thread(monkeypatch):
    assert scan.WORKER_ENVIRONMENT == {"OPENBLAS_NUM_THREADS": "1"}
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "24")
    monkeypatch.setenv("AESTER_HASH_LEDGER", "shadow")
    environment = scan.worker_environment()
    assert environment == {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "24" and environment["AESTER_HASH_LEDGER"] == "shadow"
    assert scan.worker_environment({"A": "1"}) == {"A": "1", "OPENBLAS_NUM_THREADS": "1"}


def test_child_closure_has_no_other_blas_users():
    modules = child_modules()
    assert {"transient_stage", "transient_trails", "fits_crop"} <= set(modules)
    third = subprocess.run([sys.executable, "-B", "-c", IMPORTS.replace(
        "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('starstack.'))))",
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'scipy', 'astropy', 'numba', 'cv2', 'torch'}))"),
        str(SRC)], capture_output=True, text=True, check=True).stdout.strip()
    assert third == "[]", third                        # numpy's OpenBLAS is the only BLAS in the child
    found = Counter()
    for module in modules:
        found.update(blas_sites(module))
    assert dict(found) == {**CHILD_SITES, **PARENT_SITES}, "BLAS-capable call sites changed: re-audit S-5"


CHILD = r'''
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from starstack import transient_trails as tt
from starstack.transient_stage import _scan_frame
args = json.loads(sys.argv[2])
seen = set()
def profile(frame, event, arg):
    if event == "call" and "starstack" in frame.f_code.co_filename:
        seen.add(frame.f_code.co_name)
sys.setprofile(profile)
scanned = _scan_frame(**args)
sys.setprofile(None)
scanned.pop("seconds")
m = hashlib.sha256(json.dumps(scanned, sort_keys=True, allow_nan=True).encode())
for kind, n in [(k, n) for k in ("line", "blob", "far", "vertical", "dup") for n in (3, 4608, 10001, 100003)]:
    rng = np.random.default_rng([n, len(kind)])
    t = rng.uniform(0, 8000, n)
    pts = {"line": np.column_stack([np.floor(100 + t * .83) * 2 + .5, np.floor(200 + t * .55) * 2 + .5]),
           "blob": rng.normal([4000, 2700], [900, 40], (n, 2)),
           "far": 1e7 + rng.normal(0, 1e-3, (n, 2)),
           "vertical": np.column_stack([np.full(n, 1234.5), t]),
           "dup": np.tile([[10.5, 20.5]], (n, 1))}[kind]
    centre, u, *rest = tt._axis(np.ascontiguousarray(pts))
    d = tt._describe(pts, [0], 2)
    for v in (centre, u, rest, tt._straightness(pts, centre, u), d["u"], d["length"], d["line_rms"]):
        m.update(np.asarray(v, dtype=np.float64).tobytes())
x = np.random.default_rng(1).standard_normal(1_000_003)
print(json.dumps({"digest": m.hexdigest(), "control": repr(float(np.dot(x, x))), "seen": sorted(seen),
                  "segments": len(scanned["result"]["segments"])}))
'''


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    folder = tmp_path_factory.mktemp("blas")
    rng = np.random.default_rng(20260930)
    master = np.full((800, 1200, 3), .05, np.float32)
    frame = master + rng.normal(0, .002, master.shape).astype(np.float32)
    for w in range(-1, 2):                                # a 3 px trail across the frame
        xs = np.arange(100, 1000)
        ys = (100 + (xs - 100) * .55).astype(int) + w
        frame[ys, xs] += .05
    imaging.write_linear(folder / "master.fit", master, "blas test")
    imaging.write_linear(folder / "frame.fit", frame, "blas test")
    return {"frame_path": str(folder / "frame.fit"), "master_path": str(folder / "master.fit"),
            "detection": {"bin": 2, "threshold_sigma": 4.0, "min_length_px": 60},
            "grid_marks": [{"x0": 100.0, "y0": 700.0, "x1": 1000.0, "y1": 205.0, "half_width_px": 30}],
            "stats_step": 16}


def run_child(args, threads):
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    if threads:
        env["OPENBLAS_NUM_THREADS"] = threads
    out = subprocess.run([sys.executable, "-B", "-c", CHILD, str(SRC), json.dumps(args)], env=env,
                         capture_output=True, text=True, check=True, timeout=120)
    return json.loads(out.stdout)


def test_child_results_do_not_depend_on_openblas_threads(synthetic):
    runs = {threads: run_child(synthetic, threads) for threads in ("1", "2", None)}
    assert runs["1"]["segments"] >= 2                   # an automatic trail and the manual mark were measured
    assert len({r["digest"] for r in runs.values()}) == 1
    if (os.cpu_count() or 1) > 1:
        assert len({r["control"] for r in runs.values()}) > 1   # the harness does see thread-dependent BLAS
    parent_only = {scope for (_, scope, _) in PARENT_SITES}
    child = {scope for (_, scope, _) in CHILD_SITES}
    seen = set(runs["1"]["seen"])
    assert not seen & parent_only
    assert {"_axis", "measure_segment", "find_segments", "marked_line_relation"} <= seen & child


PAYLOAD = r'''
import json, os, sys
from pathlib import Path
request = Path(sys.argv[1])
for job in json.loads(request.read_text(encoding="utf-8"))["jobs"]:
    scanned = {"result": {"segments": []}, "seconds": 0,
               "environment": {name: os.environ.get(name) for name in
                               ("OPENBLAS_NUM_THREADS", "AESTER_S5_INHERITED", "PYTHONPATH")}}
    pending = Path(job["result"]).with_suffix(".pending")
    pending.write_text(json.dumps({"version": 1, "stem": job["stem"], "scanned": scanned}), encoding="utf-8")
    os.replace(pending, job["result"])
'''


def test_scan_frames_gives_every_child_one_shared_block(tmp_path, monkeypatch):
    if process_guard.guard_kind() == process_guard.GUARD_NONE:
        pytest.skip("no supported process guard")
    payload = tmp_path / "payload.py"
    payload.write_text(PAYLOAD, encoding="utf-8")
    monkeypatch.setattr(scan, "_worker_command", lambda request: [sys.executable, str(payload), str(request)])
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "24")
    monkeypatch.setenv("AESTER_S5_INHERITED", "星空 kept")
    blocks = []
    real = scan.guarded_process

    def recording(args, **kwargs):
        blocks.append(kwargs.get("env"))
        return real(args, **kwargs)

    monkeypatch.setattr(scan, "guarded_process", recording)
    before = dict(os.environ)
    results = scan.scan_frames({"a": {}, "b": {}, "c": {}}, workers=3, output_dir=tmp_path, log=lambda _: None)
    assert dict(os.environ) == before
    assert len(blocks) == 3 and all(block is blocks[0] for block in blocks)
    assert blocks[0] == {**before, "OPENBLAS_NUM_THREADS": "1"}
    for stem in "abc":
        assert results[stem]["environment"] == {"OPENBLAS_NUM_THREADS": "1", "AESTER_S5_INHERITED": "星空 kept",
                                                "PYTHONPATH": os.environ.get("PYTHONPATH")}
