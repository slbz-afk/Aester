"""Rebuild only the curated 48-frame DNG subset, never the retired 144 set.

The existing 48-frame input directory is the normal development entry. This
explicit rebuilding helper writes a separate compatibility output directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from starstack.process_guard import guarded_process
from starstack.core import atomic_json, now, _sha
from development_assets import BASELINE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()
    assert 1 <= args.count <= 48
    sources = [ROOT / row['original_path'] for row in BASELINE['frames'][:args.count]]
    assert len(sources) == args.count
    exe = ROOT / ".cache/z8-test/dng-portable/{app}/Adobe DNG Converter.exe"
    output = ROOT / "outputs/z8-compatibility-48-20260925/dng"
    output.mkdir(parents=True, exist_ok=True)
    old_hashes = json.loads((ROOT / "docs/diagnostics/z8-source-hashes-20260916.json").read_text())
    pending = [p for p in sources if not (output / (p.stem + ".dng")).exists()]
    log = output.parent / f"dng-convert-{args.count}-{time.time_ns()}.log"
    options = ["-c", "-dng1.4", "-p0", "-d", str(output)]
    command = [str(exe), *options, *map(str, pending)]
    started = time.monotonic()
    if pending:
        with log.open("wb") as stream, guarded_process(command, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW) as process:
            last = -1
            while process.poll() is None:
                count = len(list(output.glob("*.dng")))
                if count != last:
                    print(f"DNG {count}/{args.count}", flush=True);last = count
                if time.monotonic() - started > 1800:
                    raise TimeoutError("DNG test conversion exceeded 30 minutes")
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError(f"DNG Converter returned {process.returncode}; log {log}")
    import tifffile
    records = []
    for source in sources:
        target = output / (source.stem + ".dng")
        assert target.is_file()
        assert _sha(source) == old_hashes[source.name]["sha256"]
        with tifffile.TiffFile(target) as tiff:
            pages = list(tiff.pages)
            for candidate in pages:
                if candidate.pages is not None:
                    pages.extend(candidate.pages)
            page = max(pages, key=lambda p: p.imagewidth * p.imagelength)
            assert page.imagewidth == 8280 and page.imagelength == 5520
            assert int(page.photometric) == 32803, "Expected full-size mosaic CFA, not rendered RGB"
            tags = {name: page.tags[name].value for name in (
                "ImageWidth", "ImageLength", "BitsPerSample", "Compression", "PhotometricInterpretation",
                "DNGVersion", "DNGBackwardVersion", "LossyImageCompression", "ActiveArea", "DefaultCropSize")
                    if name in page.tags}
            for name in ("DNGVersion", "DNGBackwardVersion", "LossyImageCompression"):
                if name not in tags and name in tiff.pages[0].tags:
                    tags[name] = tiff.pages[0].tags[name].value
            tags = {k: (int(v) if isinstance(v, int) else list(v) if isinstance(v, (tuple, bytes)) else v)
                    for k,v in tags.items()}
        records.append({"source": str(source), "source_sha256": old_hashes[source.name]["sha256"],
                        "dng": str(target), "dng_sha256": _sha(target), "bytes": target.stat().st_size,
                        "tags": tags})
    result = {"state": "succeeded", "time": now(), "count": len(records),
              "seconds": round(time.monotonic()-started,3), "exe": str(exe), "exe_sha256": _sha(exe),
              "converter_version": "18.6", "installer_adobe_signature": "Validated before extraction",
              "source_url": "https://download.adobe.com/pub/adobe/dng/win/AdobeDNGConverter_x64_18_6.exe",
              "options": options, "extra_lossy_encoding_requested": False, "raw_sources_unchanged": True,
              "role": "test-only compatibility preprocessing, not prototype-native HE* support",
              "log": str(log) if pending else None, "newly_converted_count": len(pending), "files": records}
    atomic_json(output.parent / f"dng-conversion-{args.count}-{time.time_ns()}.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("files","options")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
