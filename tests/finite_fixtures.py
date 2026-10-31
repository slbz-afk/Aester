"""Shared fixtures of the efficiency P1c F-3 tests (finite_stack's inventory and q_ write pools, prereg §3.3).

- ``siril_like(...)``: the bytes of a registered frame as Siril 1.4.4 writes it (the F48 fixture's r_light header
  cards, BZERO/BSCALE included; big-endian float32 data unit; 2880-byte padding), with variants for the failure and
  header-growth cases;
- ``make_work(...)``: a strict-support work folder for finite_stack.run (r_light_*.fit, their retained support maps,
  r_light_.seq, the support record and eligible counts), its frames built by ``frame_values``;
- ``FixedPools``: a finite_stack.Pools with fixed plans that keeps every note and shadow call;
- ``f48_sources(n)``: the first ``n`` sources of the read-only F48 reference job, their summed support and work folder.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from starstack import finite_stack as fs, registered_support as rs

BLOCK = 2880


@contextlib.contextmanager
def fixed_checksum_time(stamp="2026-10-06T00:00:00"):
    """astropy re-adds CHECKSUM/DATASUM to an update-mode HDU that had them, with the current time in the comments:
    such a q_ (both bodies; Siril's exports carry none) differs from second to second. Fix the time for comparisons."""
    from astropy.io.fits.hdu import base
    saved = base._ValidHDU._get_timestamp
    base._ValidHDU._get_timestamp = lambda self: stamp
    try:
        yield
    finally:
        base._ValidHDU._get_timestamp = saved


def _cards(width, height, *, bitpix=-32, roworder="BOTTOM-UP", checksum=False, filler=0, number=1):
    cards = [("SIMPLE", True, "file does conform to FITS standard"), ("BITPIX", bitpix, "number of bits per data pixel"),
             ("NAXIS", 3, "number of data axes"), ("NAXIS1", width, "length of data axis 1"),
             ("NAXIS2", height, "length of data axis 2"), ("NAXIS3", 3, "length of data axis 3"),
             ("EXTEND", True, "FITS dataset may contain extensions"),
             ("COMMENT", "FITS (Flexible Image Transport System) format is defined in 'Astronomy", ""),
             ("COMMENT", "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H", ""),
             ("BZERO", 0.0, "Offset data range to that of unsigned short"), ("BSCALE", 1.0, "Default scaling factor"),
             ("MIPS-FLO", 0.0, "Lower visualization cutoff"), ("PROGRAM", "Siril 1.4.4", "Software that created this HDU"),
             ("FILENAME", f"frame{number:05d}.dng", ""), ("DATE", "2026-09-29T20:41:24", "UTC date that FITS file was created"),
             ("DATE-OBS", "2026-08-12T20:13:38", "YYYY-MM-DDThh:mm:ss observation start, UT")]
    if roworder is not None:
        cards.append(("ROWORDER", roworder, "Order of the rows in image array"))
    cards += [("EXPTIME", 10.0, "[s]  Exposure time duration"), ("INSTRUME", "Nikon Z 8", "Instrument name"),
              ("STACKCNT", 1, "Stack frames")]
    if checksum:
        cards += [("CHECKSUM", "9aBDF9aAA9aAF9aA", "HDU checksum"), ("DATASUM", "1234567", "data unit checksum")]
    cards += [(f"FILL{i:03d}", i, "filler") for i in range(filler)]
    return cards


def siril_like(data, *, bitpix=-32, roworder="BOTTOM-UP", checksum=False, filler=0, number=1, extra_hdu=False):
    """The bytes of a Siril-style FITS frame holding ``data`` (3, H, W). ``filler`` adds cards (a header close to a
    block boundary makes the q_ header grow a block: astropy's resize path)."""
    data = np.asarray(data)
    header = "".join(str(fits.Card(*card)) for card in _cards(data.shape[2], data.shape[1], bitpix=bitpix,
                                                              roworder=roworder, checksum=checksum, filler=filler,
                                                              number=number))
    header += "END".ljust(80)
    header += " " * (-len(header) % BLOCK)
    payload = np.asarray(data, ">f4" if bitpix == -32 else ">i2").tobytes()
    payload += b"\0" * (-len(payload) % BLOCK)
    out = header.encode("ascii") + payload
    if extra_hdu:
        out += fits.ImageHDU(np.zeros((2, 2), ">f4")).header.tostring().encode("ascii") + b"\0" * BLOCK
    return out


def frame_values(rng, height, width, valid):
    """A registered frame (3, H, W) in file row order: finite samples with +0, -0, negatives and |x| close to 8 where
    ``valid`` (top-down) holds, 0 where it does not (as Siril leaves missing samples)."""
    data = rng.normal(0.0, 0.6, size=(3, height, width)).astype(np.float32)
    data[:, ::7, ::5] = 0.0
    data[:, 3::11, 2::9] = -0.0
    data[0, 1::13, 1::17] = 7.9921875
    data[2, 2::19, 4::23] = -7.75
    data[:, ~valid[::-1]] = 0.0
    return data


def sequence_text(count, reference):
    head = f"S 'r_light_' 1 {count} {count} 5 {reference} 6 0 0 0\nL 3\n"
    return (head + "".join(f"I {i} 1\n" for i in range(1, count + 1))
            + "R1 2 2 1 1 .03 40 H 1 0 0 0 1 0 0 0 1\n" * count)


def make_work(root, count=7, height=150, width=40, *, seed=0, reference=2, frame_bytes=None, masks=None):
    """A job folder ``root`` with work/ as strict support and prepare_exports leave it for finite_stack.run.
    ``frame_bytes(number, data) -> bytes`` (default siril_like) may vary a frame; ``masks(number) -> valid`` too."""
    rng = np.random.default_rng(seed)
    directory = Path(root)
    work = directory / "work"
    work.mkdir(parents=True)
    (work / "r_light_.seq").write_text(sequence_text(count, reference), encoding="utf-8")
    (work / rs.EXPORT_SEQUENCE).write_text("immutable geometry fixture", encoding="utf-8")
    counts = np.zeros((height, width), dtype=np.uint16)
    frames, arrays = [], []
    for number in range(1, count + 1):
        if masks is not None:
            valid = masks(number)
        else:
            valid = rng.random((height, width)) > 0.15
            valid[-1, -1] = False
            valid[:, :3] = number % 3 != 0          # whole missing columns on some frames
        data = frame_values(rng, height, width, valid)
        path = work / f"r_light_{number:05d}.fit"
        path.write_bytes(siril_like(data, number=number) if frame_bytes is None else frame_bytes(number, data))
        descriptor = rs.save_array(work / f"frame_support_light_{number:05d}.npy", valid)
        frames.append({"stem": f"light_{number:05d}", "supported_pixels": int(valid.sum()),
                       "export_after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "explicit_validity": descriptor})
        counts += valid
        arrays.append(data)
    support = {"state": "exports_filtered", "mode": rs.MODE, "version": rs.SUPPORT_VERSION,
               "sequence_sha256": hashlib.sha256((work / rs.EXPORT_SEQUENCE).read_bytes()).hexdigest(),
               "frames": frames, "shape": [height, width], "eligible": rs.save_array(work / rs.COUNTS, counts)}
    (work / rs.RECORD).write_text(json.dumps(support), encoding="utf-8")
    return SimpleNamespace(directory=directory, work=work, count=count, counts=counts, arrays=arrays)


class FixedPools(fs.Pools):
    """Fixed plans; keeps every note (``notes``: (site, stats, extra)) and shadow call (``shadows``)."""

    def __init__(self, write=None, inventory=None):
        self.write_plan, self.inventory_plan = write or fs.Plan(), inventory or fs.Plan()
        self.notes, self.shadows, self.operations = [], [], []

    def inventory(self, *, units, pixels):
        return self.inventory_plan

    def write(self, *, units, file_bytes, pixels, width):
        self.write_args = {"units": units, "file_bytes": file_bytes, "pixels": pixels, "width": width}
        return self.write_plan

    def operation(self, name):
        self.operations.append(name)
        return super().operation(name)

    def note_pool(self, site, stats, **extra):
        self.notes.append((site, dict(stats), extra))

    def shadow_pool(self, footprint, *, actual, items, frame_pixels):
        self.shadows.append((footprint, actual, items, frame_pixels))

    def stats(self, site):
        return [stats for name, stats, _ in self.notes if name == site][-1]


def streamed(workers, inflight=None):
    return FixedPools(write=fs.Plan(workers, None, fs.STREAMED, inflight))


def sources_of(work):
    """The finite sources (file, validity) of a make_work folder, in sequence order (what the inventory returns)."""
    support = json.loads((Path(work) / rs.RECORD).read_text(encoding="utf-8"))
    return [{"file": "r_light_" + f["stem"][6:] + ".fit", "sha256": f["export_after_sha256"],
             "validity": f["explicit_validity"], "supported_pixels": f["supported_pixels"]}
            for f in support["frames"]]


def q_digests(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).glob("q_*.fit"))}


def fits_manifest(path):
    """A FITS file's cards without DATE, CHECKSUM and DATASUM (P1c §3.3: q_ inherits the export's DATE) and the
    SHA-256 of its data unit's bytes."""
    with fits.open(path, memmap=False, do_not_scale_image_data=True) as hdus:
        cards = [card.image for card in hdus[0].header.cards if card.keyword not in ("DATE", "CHECKSUM", "DATASUM")]
        offset, span = hdus.fileinfo(0)["datLoc"], hdus.fileinfo(0)["datSpan"]
    with open(path, "rb") as stream:
        stream.seek(offset)
        digest = hashlib.sha256(stream.read(span)).hexdigest()
    return {"cards": cards, "data_sha256": digest}


def fake_runtime(folder, monkeypatch):
    """A stand-in finite runtime (an engine file and a manifest) that finite_stack accepts (as tests/test_finite_stack
    does); returns the engine record run() wants."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    executable = folder / "siril-cli.exe"
    executable.write_bytes(b"independent adapter mean oracle")
    manifest = folder / "runtime_manifest.json"
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({"executable": executable.name, "files": {executable.name: digest}}),
                        encoding="utf-8")
    monkeypatch.setattr(fs, "ENGINE_SHA256", digest)
    monkeypatch.setattr(fs, "RUNTIME_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest())
    return fs.describe_engine(str(executable))


class MeanEngine:
    """The finite engine's file contract for run(): the mean of the finite q_ samples, the valid and used counts and
    q_.seq (no normalization or rejection). ``sections`` keeps frame_pool.exclusive_sections() at each launch."""

    def __init__(self, count, reference):
        self.count, self.reference, self.calls, self.sections = count, reference, 0, []

    def __call__(self, executable, directory, name, commands, check):
        from starstack import frame_pool
        self.calls += 1
        self.sections.append(frame_pool.exclusive_sections())
        assert name == "stack" and commands[-1] == fs.STACK and f"setref q {self.reference + 1}" in commands
        x = np.stack([fits.getdata(directory / f"q_{i:05d}.fit").astype(np.float64)
                      for i in range(1, self.count + 1)])
        nvalid = np.isfinite(x).sum(axis=0).astype(np.uint16)
        total = np.where(np.isfinite(x), x, 0).sum(axis=0)
        mean = np.divide(total, nvalid, where=nvalid > 0, out=np.zeros_like(total)).astype(np.float32)
        header = fits.Header({"ROWORDER": "BOTTOM-UP", "STACKCNT": self.count})
        fits.writeto(directory / "finite_master.fit", mean, header)
        fits.writeto(directory / "finite_master_nvalid.fit", nvalid, header)
        fits.writeto(directory / "finite_master_nused.fit", nvalid.copy(), header)
        (directory / "q_.seq").write_text(sequence_text(self.count, self.reference).replace("r_light_", "q_"),
                                          encoding="utf-8")
        check()


def f48_sources(job, count):
    """(work, sources, expected, recorded) of the first ``count`` sources of the F48 reference job's finite receipt
    (read in place, never written): ``sources`` as the inventory lists them (q_<k> is the k-th), ``expected`` the sum
    of their support maps, ``recorded`` the receipt's entries (with the details the write added)."""
    work = Path(job) / "work"
    receipt = json.loads((f48_attempt(job) / "receipt.json").read_text(encoding="utf-8"))
    recorded = receipt["sources"][:count]
    sources = [{key: entry[key] for key in ("file", "sha256", "validity", "supported_pixels")} for entry in recorded]
    expected = None
    for entry in sources:
        mask = rs.load_frame_support(work, entry["validity"])
        expected = mask.astype(np.uint16) if expected is None else expected + mask
    return work, sources, expected, recorded


def f48_inventory_work(job, dest, count):
    """A work folder in ``dest`` for finite_stack's inventory over the F48 job's first ``count`` sources, as the job's
    work folder was when run() started: copies of those r_light frames and support maps (the fixture stays read-only;
    never a link), a sequence of those frames, and the support record in its 'exports_filtered' state (the job has
    since completed it) with their eligible counts."""
    source = Path(job) / "work"
    dest = Path(dest)
    dest.mkdir(parents=True)
    _, sources, expected, _ = f48_sources(job, count)
    for entry in sources:
        for name in (entry["file"], entry["validity"]["file"]):
            with open(source / name, "rb") as reader, open(dest / name, "wb") as writer:
                while chunk := reader.read(64 << 20):
                    writer.write(chunk)
    (dest / rs.EXPORT_SEQUENCE).write_bytes((source / rs.EXPORT_SEQUENCE).read_bytes())
    (dest / "r_light_.seq").write_text(sequence_text(count, count // 2), encoding="utf-8")
    record = json.loads((source / rs.RECORD).read_text(encoding="utf-8"))
    stems = {"light_" + entry["file"][8:-4] for entry in sources}
    record.update(state="exports_filtered", frames=[f for f in record["frames"] if f["stem"] in stems],
                  eligible=rs.save_array(dest / rs.COUNTS, expected))
    (dest / rs.RECORD).write_text(json.dumps(record), encoding="utf-8")
    return dest


def f48_attempt(job):
    """The F48 job's finite attempt folder (its q_ copies)."""
    attempts = sorted(Path(job).glob("finite-*"))
    assert len(attempts) == 1, attempts
    return attempts[0]
