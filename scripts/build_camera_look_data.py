"""Generate ``src/starstack/camera_look_data.py`` from pinned, openly licensed DCP files
(docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md B1).

    python -X utf8 scripts/build_camera_look_data.py <dir with rawtherapee/ and art/ DCP folders> [--check]

Every source file must match its pinned SHA-256 and carry a CC0 / public-domain ProfileCopyright.  The daylight
HueSatMap (the table whose calibration illuminant is D65 or D50) is copied byte for byte as little-endian float32.
``--check`` regenerates in memory and fails if the committed module differs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "starstack" / "camera_look_data.py"
SOURCES = {
    "rawtherapee": {"dir": "rawtherapee", "label": "RawTherapee rtdata/dcpprofiles/{file} @ 6c4cb5964fcea91dd9bdef619d4825712e083cb8"},
    "art": {"dir": "art", "label": "ART (artraweditor/ART) rtdata/dcpprofiles/{file} @ fae1b3bdc311f6f8daaaa3df24577e6c13be8a8e"},
}
OPEN_COPYRIGHTS = {"RawTherapee CC0": "CC0", "public domain": "public domain"}
DAYLIGHT = {21: "D65", 23: "D50"}
# look id, source, file, pinned file SHA-256, donor body, why this file
SPEC = [
    ("sony-ilce7m4-rt-cc0-d50", "rawtherapee", "SONY ILCE-7M4.dcp", "8b209fdcf76b996cd4a663a9a8d784895c7daf4a78415adc37141e8205ef7c4d", "SONY ILCE-7M4",
     "docs/SONY_LOOK_TRIAL_20260926.md: lower ΔE2000 to the camera JPEG and smaller hue shift than the α7R IV table"),
    ("sony-ilce7m3-art-pd-d65", "art", "SONY ILCE-7M3.dcp", "1e9cf6f96b45d14f17891d8961cf63cc895694a3879db25583537be68f6630ab", "SONY ILCE-7M3", "RawTherapee's file is individually copyrighted; ART's is public domain"),
    ("sony-ilce7sm3-rt-pd-d65", "rawtherapee", "SONY ILCE-7SM3.dcp", "4921d6d2e38d9fbc2ec10e61f09fd487c0b6cb08109013590e34c6f0ca7ca443", "SONY ILCE-7SM3", ""),
    ("sony-ilce6400-rt-pd-d50", "rawtherapee", "SONY ILCE-6400.dcp", "9e05c295993f4234b9d320f177b0306e407ba7cfa2a00d8e2d4fdfb9a7fecf64", "SONY ILCE-6400", ""),
    ("nikon-z6-rt-pd-d65", "rawtherapee", "NIKON Z 6.dcp", "690a233382579f96c635d7f330dcc99234bcb1c76a708f2b569435d57dd450df", "NIKON Z 6", ""),
    ("nikon-z6ii-rt-pd-d65", "rawtherapee", "NIKON Z 6_2.dcp", "e843307c021cdd4b4208f4d17183ed0f318cf641a8ce95054db3520bd3f13bc9", "NIKON Z 6_2", ""),
    ("nikon-z7-rt-pd-d65", "rawtherapee", "NIKON Z 7.dcp", "d359225a2fe6fb73385cc58c810c64b542727756f7683174266da8e4931eeff5", "NIKON Z 7", ""),
    ("nikon-d850-rt-pd-d65", "rawtherapee", "NIKON D850.dcp", "0d699c2f7f1c32928ae4635aee22933c625d040a987b552d4e705d5fba6648ec", "NIKON D850", ""),
    ("nikon-d750-art-pd-d65", "art", "NIKON D750.dcp", "9441dc004e2648754813e7f9c3575794cd228720693335915f50be70429c196c", "NIKON D750", "RawTherapee's file is individually copyrighted; ART's is public domain"),
    ("canon-eos6d-art-pd-d65", "art", "CANON EOS 6D.dcp", "e5603d1068c41e5edfe324fc54507c8fa438ce297bfceb5a3a5a83d39f53ac1f", "Canon EOS 6D", "RawTherapee's file is individually copyrighted; ART's is public domain"),
    ("canon-eos5dmk4-rt-pd-d65", "rawtherapee", "Canon EOS 5D Mark IV.dcp", "eacbd3ffb00a0a992890dcbda8137722d453cff7957fb2894747f71e8fb2ce56", "Canon EOS 5D Mark IV", "single-illuminant profile (D65)"),
    ("canon-eosr-rt-pd-d65", "rawtherapee", "Canon EOS R.dcp", "8c708031e6ad308affa70261c88cedd654e5ff1d2ad872af524071dfac4598ef", "Canon EOS R", ""),
    ("canon-eosrp-rt-cc0-d50", "rawtherapee", "Canon EOS RP.dcp", "27e08c7e3c62a5c698fca41402b004a8d076ea0db66ab3516fc0098ecca7027f", "Canon EOS RP", "RawTherapee CC0 preferred over ART's public-domain file (different content)"),
    ("canon-eosr5-rt-pd-d65", "rawtherapee", "Canon EOS R5.dcp", "47dbb2573e1cb78de67ba63a42f47c109457920b7388fcde829a7b026c076b98", "Canon EOS R5", ""),
    ("canon-eosr6-rt-pd-d65", "rawtherapee", "Canon EOS R6.dcp", "98f82e951c5243eded0045c54e5f428112a372338955ea604cc64338baf1996d", "Canon EOS R6", ""),
    ("canon-eosr8-rt-cc0-d50", "rawtherapee", "Canon EOS R8.dcp", "488f264aa090b2a1a2bd636e0361f4e00fd4a86c1d151dbed89fafec7e419bc0", "Canon EOS R8", ""),
]


def parse(data: bytes) -> dict:
    endian = {b"IIRC": "<", b"MMCR": ">"}.get(data[:4])
    if endian is None:
        raise ValueError("not a DCP (IIRC/MMCR)")
    offset = struct.unpack_from(endian + "I", data, 4)[0]
    count = struct.unpack_from(endian + "H", data, offset)[0]
    size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 10: 8, 11: 4}
    tags = {}
    for i in range(count):
        tag, kind, n, value = struct.unpack_from(endian + "HHII", data, offset + 2 + 12 * i)
        length = size[kind] * n
        start = offset + 2 + 12 * i + 8 if length <= 4 else value
        raw = data[start:start + length]
        if kind == 2:
            tags[tag] = raw.split(b"\0")[0].decode("latin1")
        elif kind == 3:
            tags[tag] = list(struct.unpack(endian + "H" * n, raw))
        elif kind == 4:
            tags[tag] = list(struct.unpack(endian + "I" * n, raw))
        elif kind == 11:
            tags[tag] = raw if endian == "<" else struct.pack("<" + "f" * n, *struct.unpack(">" + "f" * n, raw))
        else:
            tags[tag] = raw
    return tags


def build(dcp_root: Path) -> str:
    looks, tables = {}, {}
    for look_id, source, name, pin, donor, why in SPEC:
        path = dcp_root / SOURCES[source]["dir"] / name
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if pin is not None and sha != pin:
            raise ValueError(f"{name}: SHA-256 {sha} ≠ pinned {pin}")
        tags = parse(data)
        licence = OPEN_COPYRIGHTS.get(tags.get(50942))
        if licence is None:
            raise ValueError(f"{name}: ProfileCopyright {tags.get(50942)!r} is not CC0 / public domain")
        if tags.get(50937) != [90, 30, 1] or 51107 in tags:
            raise ValueError(f"{name}: unexpected HueSatMapDims {tags.get(50937)} or encoding tag")
        ill = [tags.get(50778, [None])[0], tags.get(50779, [None])[0]]
        if ill[1] in DAYLIGHT and 50939 in tags:
            tag, illuminant = 50939, ill[1]
        elif ill[0] in DAYLIGHT and ill[1] is None and 50938 in tags:
            tag, illuminant = 50938, ill[0]
        else:
            raise ValueError(f"{name}: no daylight HueSatMap (illuminants {ill})")
        raw = tags[tag]
        if len(raw) != 90 * 30 * 3 * 4:
            raise ValueError(f"{name}: table size {len(raw)}")
        looks[look_id] = {
            "donor": donor, "source": SOURCES[source]["label"].format(file=name), "source_sha256": sha,
            "profile_copyright": tags[50942], "licence": licence, "tag": tag,
            "tag_name": {50938: "HueSatMapData1", 50939: "HueSatMapData2"}[tag],
            "calibration_illuminant": f"{DAYLIGHT[illuminant]} ({illuminant})", "dims": [90, 30, 1],
            "encoding": "linear (ProfileHueSatMapEncoding absent)", "table_sha256": hashlib.sha256(raw).hexdigest(),
        }
        if why:
            looks[look_id]["chosen"] = why
        tables[look_id] = base64.b64encode(raw).decode("ascii")
    lines = ['"""Generated by scripts/build_camera_look_data.py — do not edit by hand.',
             "",
             "Daylight HueSatMap tables (little-endian float32 bytes, base64) of openly licensed DCP profiles, with provenance.",
             '"""', "", "LOOKS = {"]
    for look_id, info in looks.items():
        lines.append(f"    {look_id!r}: {{")
        for key, value in info.items():
            lines.append(f"        {key!r}: {value!r},")
        lines.append("    },")
    lines += ["}", "", "TABLES = {"]
    for look_id, b64 in tables.items():
        lines.append(f"    {look_id!r}: (")
        lines += [f'        "{b64[i:i + 108]}"' for i in range(0, len(b64), 108)]
        lines.append("    ),")
    lines += ["}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dcp_root")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = build(Path(args.dcp_root))
    if args.check:
        same = TARGET.read_bytes() == text.encode("utf-8")
        print("camera_look_data.py up to date" if same else "camera_look_data.py differs from the pinned sources")
        return 0 if same else 1
    TARGET.write_bytes(text.encode("utf-8"))
    print(f"wrote {TARGET} ({len(text)} bytes, {text.count('table_sha256')} looks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
