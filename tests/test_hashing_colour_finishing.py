"""P1 hashing in the colour-export and finishing modules (docs/EFFICIENCY_P1_HASHING_PREREG_20260929.md).

Every digest these modules record must be the SHA-256 of the file's bytes, and every product file must be
byte-identical with the ledger on, in shadow mode, off, or outside any run scope.  In-place writers (writes
through a memory map keep the file's modification time) must run inside ``hashing.writing``.
"""
from contextlib import ExitStack, nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageCms
import pytest
import tifffile

from starstack import (color_export as ce, core, finishing_assets, hashing, imaging, master_export as me,
                       photographic as photo, photographic_job as job, profiling, star_layer_export as sle,
                       star_reduction as sr, star_separation as ss)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import color_fixtures as cf  # noqa: E402

MODES = (None, "off", "on", "shadow")
ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
DISPLAY = {"scene_linear": False, "display_encoding_applied": True, "roworder": "TOP-DOWN",
           "color_space": "encoded_sRGB"}


def real(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def age(*paths, seconds=30):
    """Move modification times back so a ledger entry recorded now is not racy (tests run within milliseconds)."""
    for path in paths:
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - seconds * 10**9))


def scoped(mode):
    return nullcontext() if mode is None else hashing.run_scope("colour-finishing-test", mode=mode)


def ledger_volume(directory):
    """Whether the ledger works on the volume holding ``directory`` (local NTFS/ReFS)."""
    probe = Path(directory) / "ledger-probe.bin"
    with hashing.run_scope("probe", mode="on"):
        hashing.write_bytes_with_digest(probe, b"probe", source="probe")
        before = hashing.snapshot()["saved"]["write"]["files"]
        hashing.sha256_file(probe, source="probe")
        hit = hashing.snapshot()["saved"]["write"]["files"] > before
    probe.unlink()
    return hit


def saved(source, kind=None):
    value = profiling.hash_saved_sources().get(source, {})
    return value.get("hash_saved_bytes" if kind is None else f"hash_saved_{kind}_bytes", 0)


def tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


# --------------------------------------------------------------------------- #
# Helpers kept byte-for-byte                                                   #
# --------------------------------------------------------------------------- #

def old_array_identity(array):
    h = hashlib.sha256()
    h.update(str((array.shape, array.dtype.str)).encode('ascii'))
    for y in range(0, array.shape[0], 128):
        h.update(np.ascontiguousarray(array[y:y+128]).view(np.uint8))
    return h.hexdigest()


def failure(call):
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - the exact type and text are compared
        return type(exc), str(exc)
    return None


def test_array_identity_keeps_its_format_errors_and_first_failure(monkeypatch):
    rng = np.random.default_rng(11)
    arrays = [rng.random((300, 7), dtype=np.float32), rng.random((5, 4, 3)).astype(">f8"),
              np.arange(1000, dtype=np.uint16).reshape(40, 25)[:, ::3], np.zeros((0, 4), dtype=np.uint8),
              np.asfortranarray(rng.random((260, 9))), rng.integers(0, 65535, (700, 900, 3), dtype=np.uint16)]
    for array in arrays:
        assert photo.array_identity(array) == old_array_identity(array)
    for bad in (np.float32(1.0), np.zeros((), dtype=np.uint8), [1, 2]):
        assert failure(lambda: photo.array_identity(bad)) == failure(lambda: old_array_identity(bad))
    monkeypatch.setenv(hashing.ENV_WORKERS, "4")
    big = [rng.integers(0, 65535, (600, 1000, 3), dtype=np.uint16) for _ in range(3)]
    named = [("rgb", big[0]), ("sky", big[1].astype(np.float32)), ("protect", big[2] > 30000)]
    assert photo.array_identities(*named) == {k: old_array_identity(v) for k, v in named}
    assert list(photo.array_identities(*named)) == ["rgb", "sky", "protect"]
    # The first failing array in input order is reported, whatever finishes first.
    bad = [("a", big[0]), ("b", [1]), ("c", np.float32(1)), ("d", big[1])]
    assert failure(lambda: photo.array_identities(*bad)) == failure(
        lambda: {k: old_array_identity(v) for k, v in bad})
    before = profiling.hashed_memory_sources().get("photographic.array_identity", {}).get("hashed_memory_bytes", 0)
    photo.array_identities(*named)
    after = profiling.hashed_memory_sources()["photographic.array_identity"]["hashed_memory_bytes"]
    assert after - before == sum(v.nbytes for _, v in named)


def old_master_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def test_file_hash_wrappers_fail_like_the_helpers_they_replace(tmp_path):
    data = tmp_path / "data.bin"
    data.write_bytes(os.urandom(3 << 20))
    for mode in MODES:
        with scoped(mode):
            for wrapper in (me.sha256_file, ce.sha256_file):
                assert wrapper(data) == wrapper(str(data)) == real(data)
                for bad in ("", tmp_path, tmp_path / "missing.bin", 3, None):
                    assert failure(lambda: wrapper(bad)) == failure(lambda: old_master_sha(bad)), (wrapper, bad)
            assert finishing_assets.sha256_file(data) == real(data)
    before = profiling.hashed_sources().get("finishing_assets.sha256_file", {}).get("hashed_files", 0)
    finishing_assets.sha256_file(data)
    assert profiling.hashed_sources()["finishing_assets.sha256_file"]["hashed_files"] == before + 1


def test_finishing_assets_still_imports_in_the_installer_bootstrap(tmp_path):
    """The installer copies finishing_assets.py into a package without hashing/profiling."""
    package = tmp_path / "boot" / "starstack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = Path(finishing_assets.__file__).parent
    for name in ("finishing_components.py", "finishing_assets.py", "process_guard.py"):
        shutil.copyfile(source / name, package / name)
    data = tmp_path / "data.bin"
    data.write_bytes(b"bootstrap" * 1000)
    out = subprocess.run([sys.executable, "-I", "-c",
                          "import sys; sys.path.insert(0, sys.argv[1]); import starstack.finishing_components; "
                          "from starstack import finishing_assets as f; print(f.sha256_file(sys.argv[2]))",
                          str(tmp_path / "boot"), str(data)], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == real(data)


# --------------------------------------------------------------------------- #
# Digest while writing (Pillow encodes to memory, then one sequential write)   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,params,image", [
    ("mask.png", {"optimize": True}, "mask"),
    ("flags.png", {"optimize": True}, "flags"),
    ("photo.jpg", {"quality": 95, "subsampling": 0, "icc_profile": ICC}, "rgb"),
    ("result.jpg", {"quality": 97, "subsampling": 0, "icc_profile": ICC}, "rgb"),
    ("preview.jpg", {"quality": 95, "icc_profile": ICC}, "rgb"),
])
@pytest.mark.parametrize("mode", MODES)
def test_save_image_with_digest_writes_exactly_what_image_save_writes(tmp_path, name, params, image, mode):
    rng = np.random.default_rng(3)
    pixels = {"mask": (rng.random((301, 517)) > .97).astype(np.uint8) * 255,
              "flags": rng.integers(0, 64, (301, 517), dtype=np.uint8),
              "rgb": rng.integers(0, 255, (301, 517, 3), dtype=np.uint8)}[image]
    reference, target = tmp_path / ("reference-" + name), tmp_path / name
    Image.fromarray(pixels).save(reference, **params)
    with scoped(mode):
        digest = me.save_image_with_digest(Image.fromarray(pixels), target, source="test.image", **params)
        assert target.read_bytes() == reference.read_bytes()
        # Outside a scope and with the ledger off it is the plain Image.save (no digest while writing).
        assert digest == (real(target) if mode in ("on", "shadow") else None)
        before = hashing.snapshot()["saved"]["write"]["files"]
        assert me.sha256_file(target) == real(target)
        hit = hashing.snapshot()["saved"]["write"]["files"] - before
    assert hit == (1 if mode in ("on", "shadow") and ledger_volume(tmp_path) else 0)
    # Unknown extension: the same refusal as Image.save, and nothing is written.
    assert failure(lambda: Image.fromarray(pixels).save(tmp_path / "x.unknownext")) == failure(
        lambda: me.save_image_with_digest(Image.fromarray(pixels), tmp_path / "y.unknownext", source="t"))
    assert not (tmp_path / "y.unknownext").exists()


# --------------------------------------------------------------------------- #
# M + D colour export (job worker)                                             #
# --------------------------------------------------------------------------- #

def _colour_export(root):
    dng = cf.write_reference_dng(root / "ref.dng")
    cam, sky, *_ = cf.synthetic_camera_master()
    export = root / "products"
    export.mkdir()
    source = export / "master_linear.fit"
    imaging.write_linear(source, np.flipud(cam), "synthetic camera master")
    sky_path = root / "sky.png"
    Image.fromarray((sky * 255).astype(np.uint8)).save(sky_path)
    age(dng, source, sky_path)
    basis = ce.job_colour_basis(dng)
    artifacts, record = ce.export_job_products(np.flipud(cam), export, lambda: None, basis=basis,
                                               source_fits=source, sky_mask_path=sky_path)
    return export, artifacts, record


def test_colour_export_is_byte_identical_and_records_real_digests_in_every_mode(tmp_path):
    root = tmp_path / "colour"
    runs = {}
    for mode in MODES:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        with scoped(mode):
            before = {name: saved(name, "write") for name in ("master_export.mask_png", "photo_render.jpeg",
                                                               "photo_render.flags")}
            export, artifacts, record = _colour_export(root)
            wrote = {name: saved(name, "write") - value for name, value in before.items()}
        master = json.loads((export / me.RECIPE_FILENAME).read_text(encoding="utf-8"))
        photo_record = json.loads((export / "photo_render.json").read_text(encoding="utf-8"))
        assert record["master"]["sha256"] == master["master_sha256"] == real(master["master"])
        assert master["mask_sha256"] == real(master["mask"])
        assert record["photo"]["tiff_sha256"] == real(record["photo"]["tiff"])
        assert record["photo"]["jpeg_sha256"] == real(record["photo"]["jpeg"])
        assert photo_record["flags_sha256"] == real(photo_record["flags"])
        assert photo_record["metadata"]["source_master"]["sha256"] == real(master["master"])
        assert master["metadata"]["source"]["sha256"] == real(export / "master_linear.fit")
        assert [a["path"] for a in artifacts] == [master["master"], record["photo"]["tiff"], record["photo"]["jpeg"]]
        runs[mode] = tree(root), wrote
    reference = runs[None][0]
    for mode, (files, _) in runs.items():
        assert files.keys() == reference.keys()
        assert [name for name in files if files[name] != reference[name]] == [], mode
    if ledger_volume(tmp_path):
        for mode in ("on", "shadow"):
            wrote = runs[mode][1]
            assert wrote["master_export.mask_png"] > 0 and wrote["photo_render.flags"] > 0
            assert wrote["photo_render.jpeg"] == len(reference[str(Path("products", ce.D_JPEG_NAME))])
    assert all(value == 0 for value in runs[None][1].values()) and all(value == 0 for value in runs["off"][1].values())


# --------------------------------------------------------------------------- #
# Separation, reduction, advice, layers, finishing (workflow worker)           #
# --------------------------------------------------------------------------- #

def _display(path, pixels, extra=None):
    tifffile.imwrite(path, pixels, photometric="rgb", metadata={**DISPLAY, **(extra or {})},
                     extratags=[(34675, "B", len(ICC), ICC, False)])
    return path


def _layer_scene(root):
    h, w = 160, 240
    yy, xx = np.mgrid[:h, :w]
    value = .15 + .06 * np.sin(xx / 31) + .04 * np.exp(-((xx - 120) ** 2 + (yy - 55) ** 2) / 800)
    base = np.rint(value[..., None] * np.array([.88, 1, .95]) * 65535).astype(np.uint16)
    for y, x, amplitude in ((30, 40, 9000), (70, 150, 20000), (45, 200, 5000), (90, 60, 12000)):
        base[y - 1:y + 2, x - 1:x + 2] += np.uint16(amplitude)
    source = base.copy()
    source[:, 100:103] += 1500                      # a trail, inside the protected columns only
    sky = np.full((h, w), 255, np.uint8); sky[120:] = 0
    protect = np.zeros((h, w), np.uint8); protect[:, 99:104] = 255
    galaxy = np.zeros((h, w), np.uint8); galaxy[:, 20:220] = 255
    s = SimpleNamespace(root=root, source=_display(root / "source.tif", source))
    s.base = _display(root / "base.tif", base, {"transient_background_for_sha256": real(s.source)})
    for name, pixels in (("sky", sky), ("protect", protect), ("galaxy", galaxy)):
        setattr(s, name, root / (name + ".png"))
        Image.fromarray(pixels).save(getattr(s, name))
    age(s.source, s.base, s.sky, s.protect, s.galaxy)
    return s


def _chain(root):
    s = _layer_scene(root)
    out = SimpleNamespace(scene=s)
    out.separation = ss.separate_stars(s.source, root / "separation", s.sky, s.protect, transient_background=s.base)
    result = Path(out.separation["artifacts"]["diffuse_srgb16.tif"]["path"]).parent / "result.json"
    out.reduction = sr.reduce_stars(result, root / "reduction", bright_radius=3)
    out.advice = sr.suggest_star_reduction(result, root / "advice")
    out.layers = sle.export_star_layers(result, root / "layers")
    out.milkyway = job.finish_milkyway(s.source, root / "milkyway", s.sky, s.protect, s.galaxy,
                                       transient_background=s.base,
                                       starless_background=out.separation["artifacts"]["diffuse_srgb16.tif"]["path"])
    out.photo = job.finish_photo(s.source, root / "photo", s.sky, s.protect)
    return out


def _digests(out):
    """Every digest the records carry, checked against the file bytes; artifact digests by name."""
    for record in (out.separation, out.milkyway, out.photo):
        assert record["state"] == "succeeded", record
        for path, digest in record["source_hashes"].items():
            assert digest == real(path), path
    for record in (out.reduction, out.advice, out.layers):
        assert record["state"] == "succeeded", record
        for path, digest in record["input_hashes"].items():
            assert digest == real(path), path
    artifacts = {}
    for label, record in (("separation", out.separation), ("reduction", out.reduction), ("layers", out.layers),
                          ("milkyway", out.milkyway), ("photo", out.photo)):
        for name, item in record["artifacts"].items():
            assert item["sha256"] == real(item["path"]) and item["bytes"] == Path(item["path"]).stat().st_size
            artifacts[label, name] = item["sha256"]
    with tifffile.TiffFile(out.reduction["artifacts"]["background_srgb16.tif"]["path"]) as tf:
        bound = json.loads(tf.pages[0].description)["transient_background_for_sha256"]
    assert bound == out.reduction["artifacts"]["star_reduced_srgb16.tif"]["sha256"]
    # This layer embeds the digest of the separation's result.json, which holds paths and timings of its run.
    with tifffile.TiffFile(out.reduction["artifacts"]["stars_reduced_signed_codes.tif"]["path"]) as tf:
        bound = json.loads(tf.pages[0].description)["original_separation_sha256"]
    assert bound == real(Path(out.separation["artifacts"]["diffuse_srgb16.tif"]["path"]).parent / "result.json")
    del artifacts["reduction", "stars_reduced_signed_codes.tif"]
    return artifacts


def test_finishing_chain_records_real_digests_and_identical_products_in_every_mode(tmp_path):
    results = {}
    for mode in MODES:
        root = tmp_path / str(mode)
        root.mkdir()
        with scoped(mode):
            before = {name: saved(name, "ledger") for name in ("star_separation.inputs", "photographic_job.inputs",
                                                                "star_reduction.inputs")}
            out = _chain(root)
            gained = {name: saved(name, "ledger") - value for name, value in before.items()}
        results[mode] = _digests(out), gained, out
    reference = results[None][0]
    assert len(reference) >= 20
    for mode, (artifacts, gained, _) in results.items():
        assert artifacts == reference, mode
    for mode in (None, "off"):
        assert all(value == 0 for value in results[mode][1].values())
    if ledger_volume(tmp_path):
        for mode in ("on", "shadow"):
            gained, out = results[mode][1], results[mode][2]
            inputs = sum(Path(p).stat().st_size for p in out.separation["source_hashes"])
            # Hashed once at the start; the check before publication is answered by the ledger.
            assert gained["star_separation.inputs"] >= inputs
            assert gained["photographic_job.inputs"] > 0 and gained["star_reduction.inputs"] > 0


@pytest.fixture
def watched_writers(monkeypatch):
    """Fail if a writable tifffile map or an in-place description rewrite runs outside hashing.writing()."""
    calls = []
    real_memmap, real_comment = tifffile.memmap, tifffile.tiffcomment

    def busy(path):
        return hashing._norm(path) in hashing._busy

    def memmap(filename, *args, **kwargs):
        result = real_memmap(filename, *args, **kwargs)
        if kwargs.get("mode", "r+") not in ("r", "readonly", "c", "copyonwrite"):
            calls.append(("memmap", Path(filename).name, busy(filename)))
        return result

    def tiffcomment(filename, *args, **kwargs):
        calls.append(("tiffcomment", Path(filename).name, busy(filename)))
        return real_comment(filename, *args, **kwargs)

    monkeypatch.setattr(tifffile, "memmap", memmap)
    monkeypatch.setattr(tifffile, "tiffcomment", tiffcomment)
    return calls


def _layer_stage(root):
    from starstack.finishing_layer_engine import render_layer_stage
    from starstack.finishing_assets import inspect_mask, inspect_source
    rng = np.random.default_rng(7)
    pixels = rng.integers(2000, 60000, (70, 50, 3), dtype=np.uint16)
    images = {}
    for name in ("s", "b", "d"):
        path = root / (name + ".tif")
        tifffile.imwrite(path, pixels, photometric="rgb", metadata=None, extratags=[(34675, "B", len(ICC), ICC, False)])
        images[name] = inspect_source(path, encoding="srgb16")
    masks = {}
    for name in ("pure", "support"):
        path = root / (name + ".tif")
        tifffile.imwrite(path, np.full(pixels.shape[:2], 255, np.uint8), photometric="minisblack", metadata=None)
        masks[name] = inspect_mask(path, images["s"], source_sha256=images["s"]["sha256"])
    bundle = {"schema": "aester-finishing-layer-bundle-1", "id": "test_layers", "bundle_sha256": "a" * 64,
              "source": images["s"], "background": images["b"], "diffuse": images["d"],
              "pure_sky": {"mask_id": "pure", "asset": masks["pure"]},
              "transient_support": {"mask_id": "support", "asset": masks["support"]}}
    params = {"bundle_id": "test_layers", "bundle_sha256": "a" * 64, "star_strength": 0., "milkyway_amount": 0.,
              "milkyway_mask_id": None, "meteor_core_extra": 0., "meteor_glow_amount": 0.}
    target = root / "stage.tif"
    result = render_layer_stage(images["s"], bundle, params, masks, target, tile_rows=16)
    assert result["pixel_sha256"] == hashlib.sha256(tifffile.imread(target).tobytes()).hexdigest()


@pytest.mark.parametrize("mode", [None, "on"])
def test_every_in_place_writer_runs_inside_writing(tmp_path, watched_writers, mode):
    with scoped(mode):
        _digests(_chain(tmp_path))
        _layer_stage(tmp_path)
    names = [name for _, name, _ in watched_writers]
    for expected in ("diffuse_srgb16.tif", "stars_signed_codes.tif", "trails_signed_codes.tif",   # separation
                     "star_reduced_srgb16.tif", "background_srgb16.tif", "stars_reduced_signed_codes.tif",
                     "00_background.tif", "04_trails_add.tif",                                   # layers
                     "milkyway_srgb16.tif", "milkyway_background_srgb16.tif", "milkyway_diffuse_srgb16.tif",
                     "photographic_srgb16.tif", "stage.tif"):
        assert expected in names, expected
    assert ("tiffcomment", "background_srgb16.tif", True) in watched_writers
    assert names.count("stage.tif") == 6                  # the header map and one r+ map per 16-row tile
    assert [call for call in watched_writers if not call[2]] == []


def test_a_memmap_write_outside_writing_is_what_shadow_mode_reports(tmp_path):
    """Writes through a map keep the file's stat, so the ledger relies on writing(); shadow mode catches a miss."""
    path = tmp_path / "layer.tif"
    with hashing.run_scope("shadow-test", mode="shadow"):
        guard = ExitStack()
        mapped = job._new_map(guard, path, shape=(64, 80, 3), dtype=np.uint16, photometric="rgb")
        mapped[:] = 7
        refused = hashing.snapshot()["refused"]["busy"]
        assert hashing.sha256_file(path, source="t") == real(path)       # read, but not remembered while writable
        assert hashing.snapshot()["refused"]["busy"] == refused + 1
        mapped[:] = 9
        mapped.flush()
        job._close_map(mapped)
        guard.close()
        age(path)
        assert hashing.sha256_file(path, source="t") == real(path)
        stat = os.stat(path)
        mapped = tifffile.memmap(path, mode="r+")                      # deliberately unguarded
        mapped[0, 0, 0] = 1
        mapped.flush()
        job._close_map(mapped)
        unchanged = hashing._sig(os.stat(path)) == hashing._sig(stat)
        if unchanged and ledger_volume(tmp_path):
            with pytest.raises(hashing.HashLedgerMismatch):
                hashing.sha256_file(path, source="t")
        assert hashing.sha256_file(path, source="t") == real(path)
        age(path)
        assert hashing.sha256_file(path, source="t") == real(path)     # remembered again
        with hashing.writing(path):                                    # the guarded form
            mapped = tifffile.memmap(path, mode="r+")
            mapped[0, 0, 0] = 2
            mapped.flush()
            job._close_map(mapped)
        mismatches = hashing.snapshot()["shadow"]["mismatches"]
        assert hashing.sha256_file(path, source="t") == real(path)
        assert hashing.snapshot()["shadow"]["mismatches"] == mismatches


# --------------------------------------------------------------------------- #
# Change detection and failure order stay as before                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("how", ["append", "same_size_plain", "same_size_guarded"])
@pytest.mark.parametrize("mode", [None, "on", "shadow"])
def test_inputs_changed_during_processing_are_still_reported(tmp_path, monkeypatch, how, mode):
    s = _layer_scene(tmp_path)
    original = core.atomic_json
    changed = []

    def tamper(path):
        """Change one pixel byte of the source TIFF (it stays readable)."""
        middle = path.stat().st_size // 2
        if how == "append":
            with open(path, "ab") as handle:
                handle.write(b"tamper")
        else:   # an ordinary same-size write (new modification time), or one declared through writing()
            with (hashing.writing(path) if how == "same_size_guarded" else nullcontext()), open(path, "r+b") as handle:
                handle.seek(middle)
                value = handle.read(1)
                handle.seek(middle)
                handle.write(bytes([value[0] ^ 1]))
        changed.append(path)

    def save(path, value, **kwargs):
        result = original(path, value, **kwargs)
        if Path(path).name == "result.json" and value.get("state") == "succeeded" and not changed:
            tamper(s.source)
        return result

    monkeypatch.setattr(core, "atomic_json", save)
    with scoped(mode):
        with pytest.raises(ValueError, match="^input changed during separation$"):
            ss.separate_stars(s.source, tmp_path / "separation", s.sky, s.protect, transient_background=s.base)
        changed.clear()
        age(s.source)
        record = job.finish_photo(s.source, tmp_path / "photo", s.sky, s.protect)
    assert changed and record["state"] == "failed" and record["error"] == "输入在处理期间改变，结果未发布"


def _serial_first_changed(expected):
    return any(old_master_sha(Path(p)) != v for p, v in expected.items())


@pytest.mark.parametrize("module", [sr, sle])
def test_first_changed_keeps_serial_any_semantics(tmp_path, monkeypatch, module):
    monkeypatch.setenv(hashing.ENV_WORKERS, "4")
    files = []
    for index in range(4):
        path = tmp_path / f"f{index}.bin"
        path.write_bytes(os.urandom(3 << 20))
        files.append(path)
    missing = tmp_path / "missing.bin"
    good = {str(p): real(p) for p in files}
    cases = [good,
             {str(files[0]): good[str(files[0])], str(files[1]): "0" * 64, str(missing): "1" * 64,
              str(files[2]): good[str(files[2])]},
             {str(files[0]): good[str(files[0])], str(missing): "1" * 64, str(files[1]): "0" * 64},
             {str(files[3]): "f" * 64, str(files[2]): good[str(files[2])]}]
    for mode in MODES:
        with scoped(mode):
            for case in cases:
                expected = failure(lambda: _serial_first_changed(case))
                got = failure(lambda: module._first_changed(case))
                assert got == expected
                if expected is None:
                    assert (module._first_changed(case) is not None) == _serial_first_changed(case)
