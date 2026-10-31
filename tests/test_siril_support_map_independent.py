"""Contract tests with scalar map interpolation, independent of map helpers."""
import math

import numpy as np
import pytest

from starstack.siril_support_map import make_export_mapping


def header():
    return {"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "CRPIX1": 5.25,
            "CRPIX2": 4.5, "AP_ORDER": 3, "BP_ORDER": 3,
            "AP_2_0": .012, "AP_1_1": -.009, "AP_0_1": .007,
            "AP_3_0": -.0003, "BP_0_2": -.01, "BP_2_1": .0005,
            "BP_1_0": .005, "BP_0_0": .001}


def scalar_sip_translation(p, tx, ty, w, h, hdr):
    # A stored pure translation is its own independently derived case.
    xb = np.float32(float(p[0])-tx)
    yb = np.float32((h-1.-float(p[1]))+ty)
    ix, iy = math.floor(float(xb)), math.floor(float(yb))
    if not (0 <= ix <= w-2 and 0 <= iy <= h-2):
        return [-1., float(h)]
    fracx, fracy = np.float32(xb-ix), np.float32(yb-iy)
    one = np.float32(1.)
    weights = ((one-fracx)*(one-fracy), fracx*(one-fracy), (one-fracx)*fracy, fracx*fracy)
    out = []
    for prefix, axis in (("AP", 0), ("BP", 1)):
        values = []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            u, v = ix+dx-(hdr["CRPIX1"]-1.), iy+dy-(hdr["CRPIX2"]-1.)
            base = u if axis == 0 else v
            for total in range(hdr[prefix+"_ORDER"]+1):
                group = sum(hdr.get(f"{prefix}_{power}_{total-power}", 0.)*u**power*v**(total-power)
                            for power in range(total, -1, -1))
                base += group
            values.append(np.float32(base+hdr["CRPIX1" if axis == 0 else "CRPIX2"]-1.))
        v = ((values[0]*weights[0]+values[1]*weights[1])+values[2]*weights[2])+values[3]*weights[3]
        out.append(float(v) if axis == 0 else h-1.-float(v))
    return out


def test_stored_matrix_is_used_without_pixel_origin_conversion():
    h = np.array([[0., -1., 12.], [1., 0., 2.], [0., 0., 1.]])
    points = np.array([[3., 7.], [4.25, 8.5]])
    # source=(destination_y-2,12-destination_x), directly from stored H.
    expected = np.column_stack((points[:, 1]-2., 12.-points[:, 0]))
    np.testing.assert_array_equal(make_export_mapping(h, np.eye(3), 20, 18)(points), expected)


def test_nonidentity_reference_is_composed():
    frame = np.array([[1., 0., 5.], [0., 1., -3.], [0., 0., 1.]])
    ref = np.array([[1., 0., 2.], [0., 1., 4.], [0., 0., 1.]])
    points = np.array([[8., 3.], [9.25, 6.5]])
    np.testing.assert_allclose(make_export_mapping(frame, ref, 20, 18)(points), points+[-3., 7.], atol=1e-13)


@pytest.mark.parametrize("tx,ty", [(0., 0.), (.25, -.5), (-.125, .875), (.01, -.015625)])
def test_sip_interpolates_integer_coordinate_map_with_float_weights(tx, ty):
    h = np.array([[1., 0., tx], [0., 1., ty], [0., 0., 1.]])
    w, height = 14, 11
    y, x = np.indices((height, w))
    points = np.column_stack((x.ravel(), y.ravel())).astype(float)
    hdr = header()
    expected = np.array([scalar_sip_translation(p, tx, ty, w, height, hdr) for p in points])
    actual = make_export_mapping(h, np.eye(3), w, height, hdr)(points)
    np.testing.assert_array_equal(actual, expected)


def test_sip_identity_has_map_domain_exclusion_not_bypass():
    hdr = header()
    hdr.update({key: 0. for key in hdr if key.startswith(("AP_", "BP_")) and not key.endswith("ORDER")})
    mapper = make_export_mapping(np.eye(3), np.eye(3), 8, 6, hdr)
    # Bottom-up map's last row is top row in external coordinates.
    np.testing.assert_array_equal(mapper([[3., 0.], [7., 3.], [3., 1.]]), [[-1., 6.], [-1., 6.], [3., 1.]])


def test_inputs_are_frozen_and_chunks_are_consistent():
    h, ref, hdr = np.eye(3), np.eye(3), header()
    mapper = make_export_mapping(h, ref, 14, 11, hdr)
    p = np.array([[2.2, 4.7], [5., 6.], [8.5, 9.25]])
    expected = mapper(p)
    h[0, 2] = 100.; ref[1, 2] = 100.; hdr['AP_2_0'] = 100.
    np.testing.assert_array_equal(mapper(p), expected)
    np.testing.assert_array_equal(np.vstack([mapper(p[:1]), mapper(p[1:])]), expected)
    assert mapper(np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("bad", [True, 1, 0, -1, 2.5, 32767, 32768])
def test_invalid_dimensions(bad):
    with pytest.raises(ValueError):
        make_export_mapping(np.eye(3), np.eye(3), bad, 8)


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.eye(3, dtype=bool), [[1., 0., 0.], [0., float('nan'), 0.], [0., 0., 1.]], np.ones((2, 2)), np.eye(3, dtype=complex)])
def test_invalid_homographies(bad):
    with pytest.raises(ValueError):
        make_export_mapping(bad, np.eye(3), 8, 8)


@pytest.mark.parametrize("change", [{"CRPIX1": float('nan')}, {"AP_ORDER": 6}, {"AP_ORDER": True}, {"AP_2_0": float('inf')}, {"CTYPE1": "RA---TAN"}])
def test_invalid_sip(change):
    hdr = header(); hdr.update(change)
    with pytest.raises(ValueError):
        make_export_mapping(np.eye(3), np.eye(3), 8, 8, hdr)


@pytest.mark.parametrize("points", [[[1., float('nan')]], [[True, False]], [[1+2j, 0]], [[1., 2., 3.]], [["1", "2"]]])
def test_invalid_points(points):
    with pytest.raises(ValueError):
        make_export_mapping(np.eye(3), np.eye(3), 8, 8)(points)


def test_unbounded_sip_points_do_not_narrow_to_integer():
    mapper = make_export_mapping(np.eye(3), np.eye(3), 8, 8, header())
    with np.errstate(all='raise'):
        actual = mapper([[1e300, -1e300], [-1e300, 1e300]])
    np.testing.assert_array_equal(actual, [[-1., 8.], [-1., 8.]])


def test_projective_horizon_returns_outside_sentinel():
    # inverse denominator x-1, singular at destination x=1.
    inverse = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., -1.]])
    mapper = make_export_mapping(np.linalg.inv(inverse), np.eye(3), 8, 8)
    np.testing.assert_array_equal(mapper([[1., 2.]]), [[-1., 8.]])


@pytest.fixture
def fixed_engine_bundle(tmp_path, monkeypatch):
    """Use separate real files/hashes, without substituting the hash function."""
    import hashlib
    from starstack import registered_support as rs
    executable = tmp_path / 'siril-cli.exe'
    executable.write_bytes(b'controlled executable fixture')
    monkeypatch.setattr(rs, 'SIRIL_SHA256', hashlib.sha256(executable.read_bytes()).hexdigest())
    dependencies = {}
    for name in rs.INTERPOLATION_DLLS:
        path = tmp_path / name
        path.write_bytes(('controlled dependency '+name).encode())
        dependencies[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(rs, 'INTERPOLATION_DLLS', dependencies)
    return rs, {'version': '1.4.4', 'executable': str(executable)}, tmp_path


def test_engine_bundle_accepts_all_matching_files(fixed_engine_bundle):
    rs, engine, _ = fixed_engine_bundle
    rs.require_engine(engine)


@pytest.mark.parametrize('changed', ['siril-cli.exe', 'libopencv_imgproc-411.dll', 'libopencv_core-411.dll'])
def test_each_changed_engine_dependency_is_rejected(fixed_engine_bundle, changed):
    rs, engine, folder = fixed_engine_bundle
    path = folder / changed
    path.write_bytes(path.read_bytes()+b' changed')
    with pytest.raises(ValueError):
        rs.require_engine(engine)


@pytest.mark.parametrize('missing', ['siril-cli.exe', 'libopencv_imgproc-411.dll', 'libopencv_core-411.dll'])
def test_each_missing_engine_dependency_is_rejected(fixed_engine_bundle, missing):
    rs, engine, folder = fixed_engine_bundle
    (folder / missing).unlink()
    with pytest.raises(ValueError):
        rs.require_engine(engine)


def test_wrong_engine_version_rejected_despite_matching_bytes(fixed_engine_bundle):
    rs, engine, _ = fixed_engine_bundle
    engine['version'] = '1.4.5'
    with pytest.raises(ValueError):
        rs.require_engine(engine)


def test_dependency_is_resolved_beside_executable_not_current_directory(fixed_engine_bundle, tmp_path, monkeypatch):
    rs, engine, folder = fixed_engine_bundle
    elsewhere = tmp_path / 'other'
    elsewhere.mkdir()
    for name in rs.INTERPOLATION_DLLS:
        (elsewhere / name).write_bytes((folder / name).read_bytes())
        (folder / name).unlink()
    monkeypatch.chdir(elsewhere)
    with pytest.raises(ValueError):
        rs.require_engine(engine)


def test_missing_executable_description_is_rejected(fixed_engine_bundle):
    rs, engine, _ = fixed_engine_bundle
    engine.pop('executable')
    with pytest.raises(ValueError):
        rs.require_engine(engine)
