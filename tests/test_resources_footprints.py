"""Governor footprints (efficiency P1b-2 §2.4, §3.4): table rows, scaling, units_within, Siril parts, calibration.

The calibration fixture is the design's 178 frozen points (governor final/calib.json: 165 from 47 kept bench receipts
with Siril data + 13 from 16-frame runs): (part, Siril threads t, pool p, pinned mem GiB or None, frames, measured
peak job commit GB). The in-sample bands of final.md §2.4 must hold for the implemented model:
convert/probe central within [-10 %, +15 %]; every other part measured <= high <= 1.7 x measured.
"""
import pytest

from starstack import core, finite_stack, registered_support, resources, stack_range

GiB, GB = 1024 ** 3, 10 ** 9
FREE_ALLOWANCE = 180e9          # 0.9 x available physical on the dev PC during the benches (never binding)
CALIBRATION = [    ('convert', 16, 16, 16, 20, 16.58), ('probe', 16, 16, 16, 20, 9.53), ('export_stack', 16, 16, 16, 20, 21.35),
    ('convert', 16, 16, 16, 20, 16.85), ('probe', 16, 16, 16, 20, 9.44), ('export_stack', 16, 16, 16, 20, 19.42),
    ('convert', 16, 16, 16, 48, 16.67), ('probe', 16, 16, 16, 48, 9.53), ('export', 16, 16, 16, 48, 18.1),
    ('stack_finite', 16, 16, 16, 48, 17.35), ('convert', 64, 64, None, 48, 50.36), ('probe', 64, 64, None, 48, 30.02),
    ('export_stack', 64, 64, None, 48, 132.68), ('convert', 64, 64, None, 48, 50.71), ('probe', 64, 64, None, 48, 30.44),
    ('export_stack', 64, 64, None, 48, 100.58), ('convert', 64, 64, None, 20, 26.06), ('probe', 64, 64, None, 20, 17.86),
    ('export_stack', 64, 64, None, 20, 47.13), ('convert', 64, 64, None, 20, 26.06), ('probe', 64, 64, None, 20, 17.82),
    ('export', 64, 64, None, 20, 46.04), ('stack_finite', 64, 64, None, 20, 17.56), ('convert', 64, 64, None, 48, 51.08),
    ('probe', 64, 64, None, 48, 29.0), ('export', 64, 64, None, 48, 85.47), ('convert', 16, 64, None, 48, 23.31),
    ('probe', 16, 64, None, 48, 16.09), ('export_stack', 16, 64, None, 48, 45.07), ('convert', 16, 64, 9, 48, 18.74),
    ('probe', 16, 64, 9, 48, 16.09), ('export_stack', 16, 64, 9, 48, 18.0), ('convert', 16, 16, 16, 48, 16.85),
    ('probe', 16, 16, 16, 48, 9.59), ('export', 16, 16, 16, 48, 17.09), ('stack_finite', 16, 16, 16, 48, 17.35),
    ('convert', 64, 64, None, 48, 51.72), ('probe', 64, 64, None, 48, 29.12), ('export', 64, 64, None, 48, 114.05),
    ('stack_finite', 64, 64, None, 48, 31.71), ('convert', 64, 64, None, 48, 49.37), ('probe', 64, 64, None, 48, 30.37),
    ('export', 64, 64, None, 48, 88.35), ('stack_finite', 64, 64, None, 48, 28.49), ('convert', 16, 64, None, 48, 23.21),
    ('probe', 16, 64, None, 48, 15.71), ('export', 16, 64, None, 48, 41.08), ('stack_finite', 16, 64, None, 48, 23.8),
    ('convert', 16, 64, 9, 48, 18.83), ('probe', 16, 64, 9, 48, 16.0), ('export', 16, 64, 9, 48, 17.68),
    ('stack_finite', 16, 64, 9, 48, 16.44), ('convert', 64, 64, None, 48, 50.27), ('probe', 64, 64, None, 48, 28.36),
    ('export', 64, 64, None, 48, 98.06), ('convert', 16, 16, 16, 20, 16.31), ('probe', 16, 16, 16, 20, 9.61),
    ('export_stack', 16, 16, 16, 20, 17.18), ('convert', 16, 16, 16, 20, 16.22), ('probe', 16, 16, 16, 20, 9.53),
    ('export', 16, 16, 16, 20, 18.05), ('stack_finite', 16, 16, 16, 20, 9.35), ('convert', 64, 64, None, 48, 49.18),
    ('probe', 64, 64, None, 48, 30.31), ('export', 64, 64, None, 48, 94.35), ('stack_rej', 64, 64, None, 48, 47.08),
    ('convert', 16, 16, 16, 20, 16.85), ('probe', 16, 16, 16, 20, 9.53), ('export_stack', 16, 16, 16, 20, 17.09),
    ('convert', 64, 64, None, 20, 26.06), ('probe', 64, 64, None, 20, 17.86), ('export', 64, 64, None, 20, 49.33),
    ('stack_finite', 64, 64, None, 20, 18.28), ('convert', 64, 64, None, 20, 26.06), ('probe', 64, 64, None, 20, 17.64),
    ('export', 64, 64, None, 20, 50.43), ('stack_rej', 64, 64, None, 20, 32.0), ('convert', 16, 16, 16, 20, 16.85),
    ('probe', 16, 16, 16, 20, 9.53), ('export_stack', 16, 16, 16, 20, 18.05), ('convert', 16, 16, 16, 48, 16.76),
    ('probe', 16, 16, 16, 48, 9.6), ('export', 16, 16, 16, 48, 17.64), ('stack_finite', 16, 16, 16, 48, 17.35),
    ('convert', 64, 64, None, 48, 50.45), ('probe', 64, 64, None, 48, 28.03), ('export', 64, 64, None, 48, 131.31),
    ('stack_rej', 64, 64, None, 48, 48.54), ('convert', 64, 64, None, 48, 50.27), ('probe', 64, 64, None, 48, 27.58),
    ('export_stack', 64, 64, None, 48, 103.69), ('convert', 64, 64, None, 48, 51.35), ('probe', 64, 64, None, 48, 30.16),
    ('export', 64, 64, None, 48, 95.41), ('stack_finite', 64, 64, None, 48, 28.75), ('convert', 16, 64, None, 48, 23.31),
    ('probe', 16, 64, None, 48, 16.09), ('export_stack', 16, 64, None, 48, 45.57), ('convert', 16, 64, None, 48, 23.03),
    ('probe', 16, 64, None, 48, 15.5), ('export', 16, 64, None, 48, 37.51), ('stack_finite', 16, 64, None, 48, 23.8),
    ('convert', 16, 64, 9, 48, 18.73), ('probe', 16, 64, 9, 48, 16.09), ('export_stack', 16, 64, 9, 48, 18.0),
    ('convert', 16, 64, 9, 48, 18.65), ('probe', 16, 64, 9, 48, 15.79), ('export', 16, 64, 9, 48, 17.54),
    ('stack_finite', 16, 64, 9, 48, 16.44), ('convert', 16, 16, 16, 48, 16.87), ('probe', 16, 16, 16, 48, 9.55),
    ('export', 16, 16, 16, 48, 17.51), ('stack_finite', 16, 16, 16, 48, 17.35), ('convert', 16, 16, 16, 20, 16.32),
    ('probe', 16, 16, 16, 20, 9.39), ('export_stack', 16, 16, 16, 20, 20.12), ('convert', 16, 16, 16, 20, 16.86),
    ('probe', 16, 16, 16, 20, 9.57), ('export_stack', 16, 16, 16, 20, 19.85), ('convert', 64, 64, None, 20, 25.7),
    ('probe', 64, 64, None, 20, 17.86), ('export', 64, 64, None, 20, 60.6), ('stack_finite', 64, 64, None, 20, 17.99),
    ('convert', 64, 64, None, 20, 26.07), ('probe', 64, 64, None, 20, 17.6), ('export', 64, 64, None, 20, 48.79),
    ('stack_finite', 64, 64, None, 20, 17.41), ('convert', 64, 64, None, 48, 51.54), ('probe', 64, 64, None, 48, 28.53),
    ('export_stack', 64, 64, None, 48, 119.27), ('convert', 64, 64, None, 48, 48.89), ('probe', 64, 64, None, 48, 29.01),
    ('export', 64, 64, None, 48, 94.46), ('stack_finite', 64, 64, None, 48, 28.64), ('convert', 64, 64, None, 48, 49.35),
    ('probe', 64, 64, None, 48, 29.81), ('export', 64, 64, None, 48, 104.85), ('stack_rej', 64, 64, None, 48, 47.55),
    ('convert', 16, 64, None, 48, 22.5), ('probe', 16, 64, None, 48, 15.36), ('export_stack', 16, 64, None, 48, 39.63),
    ('convert', 16, 64, None, 48, 23.22), ('probe', 16, 64, None, 48, 15.35), ('export', 16, 64, None, 48, 41.96),
    ('stack_finite', 16, 64, None, 48, 23.8), ('convert', 16, 64, 9, 48, 18.74), ('probe', 16, 64, 9, 48, 16.04),
    ('export_stack', 16, 64, 9, 48, 18.01), ('convert', 16, 64, 9, 48, 18.84), ('probe', 16, 64, 9, 48, 15.82),
    ('export', 16, 64, 9, 48, 18.46), ('stack_finite', 16, 64, 9, 48, 16.44), ('convert', 16, 16, 16, 48, 16.86),
    ('probe', 16, 16, 16, 48, 9.56), ('export', 16, 16, 16, 48, 16.97), ('stack_finite', 16, 16, 16, 48, 17.35),
    ('convert', 64, 64, None, 48, 50.09), ('probe', 64, 64, None, 48, 29.81), ('export_stack', 64, 64, None, 48, 89.65),
    ('convert', 64, 64, None, 48, 49.55), ('probe', 64, 64, None, 48, 29.15), ('export_stack', 64, 64, None, 48, 115.61),
    ('convert', 64, 64, None, 48, 49.01), ('probe', 64, 64, None, 48, 29.83), ('export_stack', 64, 64, None, 48, 101.65),
    ('convert', 64, 64, None, 16, 22.388), ('probe', 64, 64, None, 16, 16.087), ('export', 64, 64, None, 16, 38.201),
    ('stack_rej', 64, 64, None, 16, 33.006), ('stack_rej', 64, 64, None, 16, 30.872), ('convert', 64, 1, None, 16, 13.916),
    ('probe', 64, 1, None, 16, 7.514), ('export', 64, 1, None, 16, 33.849), ('stack_rej', 64, 1, None, 16, 23.829),
    ('stack_rej', 64, 1, None, 16, 23.838), ('stack_finite', 64, 1, None, 16, 7.261), ('stack_finite', 4, 4, None, 16, 4.588),
    ('stack_rej', 4, 4, None, 16, 4.204),
]


def test_the_fixture_has_the_designs_178_points():
    assert len(CALIBRATION) == 178
    counts = {}
    for part, *_ in CALIBRATION:
        counts[part] = counts.get(part, 0) + 1
    assert counts == {"convert": 49, "probe": 49, "export": 28, "stack_finite": 22, "export_stack": 21, "stack_rej": 9}


def _predict(part, t, p, mem, frames):
    allowance = mem * GiB if mem else FREE_ALLOWANCE
    return resources.siril_demand((part,), threads=t, pool=p, frames=frames, frame_pixels=resources.REFERENCE_PIXELS,
                                  allowance=allowance)


@pytest.mark.parametrize("point", CALIBRATION, ids=lambda p: f"{p[0]}-t{p[1]}-p{p[2]}-m{p[3]}-f{p[4]}-{p[5]}")
def test_calibration_band(point):
    part, t, p, mem, frames, measured = point
    predicted = _predict(part, t, p, mem, frames)
    central, high = (value / GB for value in predicted["commit"])
    if part in ("convert", "probe"):
        assert -0.10 <= central / measured - 1 <= 0.15
    else:
        assert measured <= high <= 1.7 * measured
    assert central <= high


def test_in_sample_ranges_match_the_design():
    errors, ratios = {}, {}
    for part, t, p, mem, frames, measured in CALIBRATION:
        central, high = (v / GB for v in _predict(part, t, p, mem, frames)["commit"])
        errors.setdefault(part, []).append(central / measured - 1)
        ratios.setdefault(part, []).append(high / measured)
    assert min(errors["convert"]) > -0.06 and max(errors["convert"]) < 0.03       # design -5.5 % .. +2.1 %
    assert min(errors["probe"]) > -0.01 and max(errors["probe"]) < 0.12           # design -0.7 % .. +11.3 %
    assert max(ratios["export"]) < 1.63 and max(ratios["export_stack"]) < 1.55   # design 1.62 / 1.54
    assert max(ratios["stack_finite"]) < 1.43 and max(ratios["stack_rej"]) < 1.26


def test_pool_is_commit_only_and_follows_p():
    a = _predict("convert", 16, 64, None, 48)
    b = _predict("convert", 16, 1, None, 48)
    assert a["commit"][0] - b["commit"][0] == 63 * resources.SIRIL_POOL_BYTES
    assert a["ws"] == b["ws"]                                                     # F1: pool WS ~ 0.04 GB either way
    idle = resources.siril_demand(("convert",), threads=1, pool=64, frames=1, frame_pixels=1, allowance=None)
    assert idle["commit"][0] / GB == pytest.approx(8.62, abs=0.03)               # F1 idle launch 8.635 / 8.618 GB


def test_the_allowance_bounds_the_in_flight_units():
    unbounded = _predict("export", 16, 64, None, 48)
    bounded = _predict("export", 16, 64, 9, 48)
    assert unbounded["bound_by"] == "units" and bounded["bound_by"] == "allowance"
    assert bounded["commit"][1] / GB == pytest.approx(64 * 0.1342 + 0.03 + 1.2 * 9 * GiB / GB, abs=0.05)
    probe = _predict("probe", 64, 64, 1, 48)
    assert probe["bound_by"] == "units"                                          # probe has no allowance term


def test_a_launch_with_several_parts_takes_the_largest():
    both = resources.siril_demand(("probe", "export_stack"), threads=64, pool=64, frames=48,
                                  frame_pixels=resources.REFERENCE_PIXELS, allowance=FREE_ALLOWANCE)
    alone = _predict("export_stack", 64, 64, None, 48)
    assert both["commit"] == alone["commit"] and both["dominant"] == "export_stack"


def test_pixel_scaling_is_linear():
    full = _predict("convert", 16, 1, None, 48)
    half = resources.siril_demand(("convert",), threads=16, pool=1, frames=48,
                                  frame_pixels=resources.REFERENCE_PIXELS // 2, allowance=FREE_ALLOWANCE)
    pool = resources.SIRIL_POOL_FIXED + resources.SIRIL_POOL_BYTES
    assert (half["commit"][0] - pool) * 2 == pytest.approx(full["commit"][0] - pool, rel=1e-6)


def test_unknown_launches_have_no_prediction():
    assert resources.siril_demand(("other",), threads=4, pool=4, frames=4, frame_pixels=1, allowance=1)["commit"] is None


# --------------------------------------------------------------------------- parts of every product launch

@pytest.mark.parametrize("commands, parts", [
    (["setext fit", "set32bits", "setcompress 0", "cd stage", "convertraw light -debayer -out=../work"], ("convert",)),
    (["setext fit", "set32bits", "setcompress 0", "cd stage", "convert light -out=../work"], ("convert",)),
    (["offline", "setext fit", "set32bits", "setcompress 0", "cd float", "convert frame -debayer -out=../decoded"],
     ("convert",)),
    (list(core.PROBE_SCRIPT), ("probe",)),
    (["cd work", *core.registration_commands(core.DISTORTION_WCS, True)], ("export",)),
    (["cd work", "seqapplyreg light -filter-included"], ("export",)),
    (["cd work", registered_support.EXPORT_COMMAND], ("export",)),
    (core.stack_script(core.DISTORTION_WCS, own_registration=True), ("export_stack",)),
    (["cd work", "seqapplyreg light -filter-included", core.STACK_COMMAND], ("export_stack",)),
    (core.stack_script(None), ("probe", "export_stack")),
    (core.stack_script(core.DISTORTION_WCS), ("probe", "export_stack")),
    (["cd work", "seqclean r_light -stat", core.STACK_COMMAND], ("stack_rej",)),        # also the split L layout (X16)
    (["setext fit", "set32bits", "setcompress 0", "seqclean r_light -stat", stack_range.STACK], ("stack_rej",)),
    (["offline", "setext fit", "set32bits", "setcompress 0", "set gui.icc_autoassignment=0",
      "set gui.icc_autoconversion=0", "setref q 7", finite_stack.STACK], ("stack_finite",)),
    (["requires 1.4.4", "cd work"], ("other",)),
    ([], ("other",)),
])
def test_siril_parts_of_every_product_command_list(commands, parts):
    assert resources.siril_parts(commands, "siril-cli.exe") == parts


# --------------------------------------------------------------------------- the table

CALL_SITE_KEYS = ("py.transient_scan", "py.render_rows", "py.coverage", "py.hash", "py.blas_slot", "py.worker_base")


def test_every_call_site_key_exists_and_rows_have_provenance():
    for key in CALL_SITE_KEYS + tuple("siril." + p for p in resources.SIRIL_PARTS):
        assert key in resources.FOOTPRINTS
    for key, fp in resources.FOOTPRINTS.items():
        assert fp.key == key and fp.provenance.startswith(("measured", "est", "code"))
        assert fp.kind in ("cpu", "gil", "io", "siril")
        assert fp.scale in ("pixels", "width", "frames_width", "blas_threads", "none")   # frames_width: P1c py.median
        assert all(c >= 0 for c in fp.commit_per_unit) and fp.commit_per_unit[0] <= fp.commit_per_unit[1]
    assert resources.FOOTPRINTS_VERSION


def test_x3_rows_follow_the_pool_measurements():
    coverage = resources.footprint_bytes(resources.FOOTPRINTS["py.coverage"], 1, frame_pixels=resources.REFERENCE_PIXELS)
    assert coverage["commit"][0] == 230 * 1024 ** 2                                # pool exp6: unit_bytes 230 MiB
    slot = resources.FOOTPRINTS["py.blas_slot"]
    assert resources.footprint_bytes(slot, 3)["commit"][0] == 3 * 32 * 1024 ** 2
    assert resources.footprint_bytes(slot, 1)["ws"][0] == pytest.approx(0.25e6, rel=0.05)
    assert resources.FOOTPRINTS["py.coverage"].cap == 8 and resources.FOOTPRINTS["py.median"].cap == 4


def test_pixel_width_and_blas_scaling():
    scan = resources.FOOTPRINTS["py.transient_scan"]
    one = resources.footprint_bytes(scan, 1, frame_pixels=resources.REFERENCE_PIXELS)["commit"][0]
    assert one / GB == pytest.approx(1.8, abs=1e-8)              # P1c T-2's charged worst case (P1a typical: 0.55)
    assert resources.footprint_bytes(scan, 1, frame_pixels=resources.REFERENCE_PIXELS // 2)["commit"][0] == \
        pytest.approx(70e6 + (one - 70e6) / 2, abs=1)          # 70 MB a process, the rest scaled by pixels
    rows = resources.FOOTPRINTS["py.render_rows"]
    assert resources.footprint_bytes(rows, 1, width=8280)["commit"][0] / GB == pytest.approx(0.31, abs=0.01)
    assert resources.footprint_bytes(rows, 2, width=4140)["commit"][0] == \
        resources.footprint_bytes(rows, 1, width=8280)["commit"][0]
    base = resources.FOOTPRINTS["py.worker_base"]
    commits = [resources.footprint_bytes(base, 1, blas_threads=b)["commit"][0] / GB for b in (1, 16, 64)]
    assert commits == pytest.approx([0.036, 0.539, 0.808], abs=0.01)             # F6: 0.036 / 0.541 / 0.811 GB


def test_the_scan_row_charges_what_t2_plans_with():
    """py.transient_scan charges each scan process transient_stage.scan_unit_bytes (P1c T-2's measured worst case), so
    the shadow decision recorded next to T-2's process count uses T-2's memory figure; rows without a unit_base keep
    their former values."""
    from starstack import transient_stage
    assert (resources.SCAN_UNIT_BYTES, resources.SCAN_BASE_BYTES, resources.REFERENCE_PIXELS) == \
        (transient_stage.SCAN_UNIT_BYTES, transient_stage.SCAN_BASE_BYTES, transient_stage.REFERENCE_PIXELS)
    scan = resources.FOOTPRINTS["py.transient_scan"]
    for pixels in (64 * 96, 6000 * 4000, resources.REFERENCE_PIXELS, 9504 * 6336, 11648 * 8736):
        for units in (1, 3, 8):
            charged = units * transient_stage.scan_unit_bytes(pixels)
            footprint = resources.footprint_bytes(scan, units, frame_pixels=pixels)
            for value in footprint["commit"] + footprint["ws"]:
                assert value == pytest.approx(charged, abs=units + 1), (pixels, units)
    for key, fp in resources.FOOTPRINTS.items():
        if key != "py.transient_scan":
            assert fp.unit_base == (0.0, 0.0), key


def test_units_within_is_monotone_and_bounded():
    fp = resources.FOOTPRINTS["py.coverage"]
    previous = 0
    for allowance in range(0, 20 * GiB, GiB // 2):
        units = resources.units_within(fp, allowance, allowance, frame_pixels=resources.REFERENCE_PIXELS, cap_items=48)
        assert previous <= units <= 48
        previous = units
    assert resources.units_within(fp, 10 ** 15, 10 ** 15, frame_pixels=resources.REFERENCE_PIXELS, cap_items=8) == 8
    assert resources.units_within(fp, -1, 10 ** 12, frame_pixels=resources.REFERENCE_PIXELS, cap_items=8) == 0
    assert resources.units_within(fp, 10 ** 12, 100, frame_pixels=resources.REFERENCE_PIXELS, cap_items=8) == 0
    assert resources.units_within(fp, None, None, cap_items=5) == 5