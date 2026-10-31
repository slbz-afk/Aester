"""Resolve fixed Windows wheels from verified PyPI metadata; build-time only (needs network and the project venv).

Usage: python scripts/build_download_manifest.py [--release-id StarStack-Trial-<version>-<YYYYMMDD>-windows-x64]
The root set (ROOTS) is what the trial runtime must import; their pinned dependency closure is resolved from the
installed venv metadata and every wheel URL/sha256 is taken from PyPI's JSON API (docs/PRE_RELEASE_CLOSURE_20260922.md R).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import argparse
import hashlib
import json
import re
import urllib.request

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
VERSION = "3.13.15"
# The runtime's import roots; rawpy joins in 0.2.0 so non-DNG RAW jobs get their colour basis (M/D) in the trial too.
ROOTS = ["numpy", "astropy", "pillow", "tifffile", "pyside6-essentials", "shiboken6", "mcp", "rawpy"]


def product_version() -> str:
    text = (ROOT / "src/starstack/__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def default_release_id(version: str | None = None, date: str | None = None) -> str:
    version = version or product_version()
    date = date or datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"StarStack-Trial-{version}-{date}-windows-x64"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "StarStack-Trial-Builder/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-id", default=default_release_id(), help="试用包标识（默认按产品版本与 UTC 日期生成）")
    parser.add_argument("--out", required=True, help="where to write the resolved wheel closure (merge its wheels into "
                        "packaging/aester_p2/download-manifest.json by hand; that file carries more fields)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    pins = dict(line.split("==") for line in (ROOT / "requirements-lock.txt").read_text().splitlines() if "==" in line)
    pins = {canonicalize_name(k): v for k, v in pins.items()}
    environment = default_environment()
    environment.update(python_version="3.13", python_full_version=VERSION, extra="")
    requested = {}
    pending = [(name, set()) for name in ROOTS]
    while pending:
        raw_name, extras = pending.pop()
        name = canonicalize_name(raw_name)
        if name in requested and extras.issubset(requested[name]):
            continue
        extras = requested.get(name, set()) | extras
        requested[name] = extras
        installed = metadata.distribution(name)
        assert installed.version == pins[name]
        for item in installed.requires or []:
            req = Requirement(item)
            if not req.marker or any(req.marker.evaluate({**environment, "extra": extra}) for extra in {""} | extras):
                dep = canonicalize_name(req.name)
                if pins[dep] not in req.specifier:
                    raise ValueError(f"Unsatisfied pinned requirement: {name}: {req}")
                pending.append((dep, set(req.extras)))
    supported = list(cpython_tags((3, 13), abis=["cp313"], platforms=["win_amd64"]))
    supported += list(compatible_tags((3, 13), interpreter="cp313", platforms=["win_amd64"]))
    rank = {tag: i for i, tag in enumerate(supported)}

    def wheel(name):
        url = f"https://pypi.org/pypi/{name}/{pins[name]}/json"
        data = json.loads(fetch(url))
        options = []
        for item in data["urls"]:
            if item["packagetype"] != "bdist_wheel" or item.get("yanked"):
                continue
            _, _, _, tags = parse_wheel_filename(item["filename"])
            positions = [rank[tag] for tag in tags if tag in rank]
            if positions:
                options.append((min(positions), item))
        if not options:
            raise RuntimeError(f"No compatible Windows x64 CPython 3.13 wheel: {name}")
        item = min(options, key=lambda option: option[0])[1]
        return {"name": name, "version": pins[name], "url": item["url"],
                "filename": item["filename"], "sha256": item["digests"]["sha256"],
                "size": item["size"], "metadata_source": url}

    with ThreadPoolExecutor(max_workers=6) as pool:
        wheels = list(pool.map(wheel, sorted(requested)))
    filename = f"python-{VERSION}-embed-amd64.zip"
    python_url = f"https://www.python.org/ftp/python/{VERSION}/{filename}"
    cache = ROOT / ".cache" / "release-downloads"
    cached_python = cache / filename
    python_bytes = cached_python.read_bytes() if cached_python.exists() else fetch(python_url)
    # Compared with the SHA-256 shown on the official 3.13.15 release page.
    if hashlib.sha256(python_bytes).hexdigest() != "d1f04d990aee1253d8569e8e5104e30fa9f5fa830899f14843448872d936a2cf":
        raise ValueError("Python archive does not match its official release checksum")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / filename).write_bytes(python_bytes)
    value = {"release_id": args.release_id, "platform": "Windows x64", "product_version": product_version(),
             "roots": ROOTS,
             "python": {"version": VERSION, "url": python_url, "filename": filename,
                        "sha256": hashlib.sha256(python_bytes).hexdigest(), "size": len(python_bytes)},
             "wheels": wheels,
             "siril": {"version": "1.4.4", "url": "https://free-astro.org/download/siril-1.4.4-ucrt64_win.zip",
                       "filename": "siril-1.4.4-ucrt64_win.zip", "size": 165994089,
                       "sha256": "ab276bedd3e805d523c535c1fe813fd68ad4f9dd1c1e5367278b6d027e53a6ee"}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(out), "wheels": len(wheels), "python": value["python"],
                      "download_bytes": len(python_bytes) + sum(w["size"] for w in wheels) + 165994089}, indent=2))


if __name__ == "__main__":
    main()
