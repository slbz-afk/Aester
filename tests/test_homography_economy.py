import math
import numpy as np
import pytest

from starstack.registration_models import apply_matrix, fit_homography


def _normalise_points(p):
    mean = p.mean(axis=0)
    scale = math.sqrt(2.0) / max(1e-12, np.mean(np.hypot(*(p - mean).T)))
    t = np.array([
        [scale, 0.0, -scale * mean[0]],
        [0.0, scale, -scale * mean[1]],
        [0.0, 0.0, 1.0],
    ])
    return t, (p - mean) * scale


def _dlt_system(src, dst):
    s = np.asarray(src, dtype=np.float64)
    d = np.asarray(dst, dtype=np.float64)
    ts, sn = _normalise_points(s)
    td, dn = _normalise_points(d)
    n = len(sn)
    a_mat = np.zeros((2 * n, 9))
    x, y = sn[:, 0], sn[:, 1]
    u, v = dn[:, 0], dn[:, 1]
    a_mat[0::2, 0], a_mat[0::2, 1], a_mat[0::2, 2] = -x, -y, -1.0
    a_mat[0::2, 6], a_mat[0::2, 7], a_mat[0::2, 8] = u * x, u * y, u
    a_mat[1::2, 3], a_mat[1::2, 4], a_mat[1::2, 5] = -x, -y, -1.0
    a_mat[1::2, 6], a_mat[1::2, 7], a_mat[1::2, 8] = v * x, v * y, v
    return a_mat, ts, td


def _full_svd_homography(src, dst):
    a_mat, ts, td = _dlt_system(src, dst)
    _, _, vt = np.linalg.svd(a_mat)
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    if abs(h[2, 2]) < 1e-15:
        raise ValueError('degenerate homography')
    return h / h[2, 2]


def _make_problem(n, seed, noise=0.0, src_scale=1.0, dst_scale=1.0, perspective=0.15):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.0, 1.0, (n, 2))
    h_true = np.array([
        [1.1, -0.2, 0.3],
        [0.15, 1.05, -0.25],
        [perspective, -0.4 * perspective, 0.95],
    ])
    mapped = apply_matrix(h_true, base)
    src = base * src_scale
    dst = mapped * dst_scale
    if noise:
        dst = dst + rng.normal(0.0, noise * dst_scale, dst.shape)
    return src, dst


@pytest.mark.parametrize('n,seed,noise,src_scale,dst_scale,perspective', [
    (4, 11, 0.0, 1.0, 1.0, 0.15),
    (4, 12, 0.0, 1e2, 1e-2, 0.35),
    (5, 21, 1e-6, 1.0, 1.0, 0.10),
    (5, 22, 1e-6, 1e-1, 1e2, 0.40),
    (20, 31, 1e-4, 1.0, 1.0, 0.20),
    (20, 32, 1e-3, 10.0, 0.1, 0.30),
    (100, 41, 1e-4, 1.0, 1.0, 0.15),
    (100, 42, 1e-3, 1e-2, 1e2, 0.25),
])
def test_fit_homography_matches_full_svd_oracle(n, seed, noise, src_scale, dst_scale, perspective):
    src, dst = _make_problem(n, seed, noise, src_scale, dst_scale, perspective)
    h = fit_homography(src, dst)
    h_ref = _full_svd_homography(src, dst)
    assert np.all(np.isfinite(h))
    np.testing.assert_allclose(h, h_ref, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        apply_matrix(h, src),
        apply_matrix(h_ref, src),
        rtol=1e-8,
        atol=1e-8,
    )
    if noise == 0.0:
        np.testing.assert_allclose(apply_matrix(h, src), dst, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize('seed,src_scale,dst_scale,perspective', [
    (301, 1.0, 1.0, 0.25),
    (302, 1e2, 1e-2, 0.35),
])
def test_four_point_exact_nullspace_is_preserved(seed, src_scale, dst_scale, perspective):
    src, dst = _make_problem(4, seed, 0.0, src_scale, dst_scale, perspective)
    a_mat, ts, td = _dlt_system(src, dst)
    h = fit_homography(src, dst)
    h_dlt = td @ h @ np.linalg.inv(ts)
    v = h_dlt.reshape(-1)
    residual = np.linalg.norm(a_mat @ v)
    scale = max(1.0, np.linalg.norm(a_mat) * np.linalg.norm(v))
    assert residual / scale < 1e-10
    h_ref = _full_svd_homography(src, dst)
    np.testing.assert_allclose(h, h_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(apply_matrix(h, src), dst, rtol=1e-8, atol=1e-8)


def test_economy_svd_would_lose_four_point_nullspace():
    src, dst = _make_problem(4, 303, 0.0, 1.0, 1.0, 0.2)
    a_mat, _, _ = _dlt_system(src, dst)
    _, _, vt_full = np.linalg.svd(a_mat)
    _, _, vt_eco = np.linalg.svd(a_mat, full_matrices=False)
    assert np.linalg.norm(a_mat @ vt_full[-1]) < 1e-12
    assert np.linalg.norm(a_mat @ vt_eco[-1]) > 1e-6


def test_degenerate_h33_zero_behavior_parity():
    src = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-0.5, 0.5],
        [0.5, -0.5],
    ])
    h_true = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    dst = apply_matrix(h_true, src)
    try:
        h_new = fit_homography(src, dst)
    except ValueError as exc:
        h_new = None
        assert 'degenerate homography' in str(exc)
    try:
        h_ref = _full_svd_homography(src, dst)
    except ValueError as exc:
        h_ref = None
        assert 'degenerate homography' in str(exc)
    assert (h_new is None) == (h_ref is None)
    if h_new is not None:
        np.testing.assert_allclose(
            apply_matrix(h_new, src),
            apply_matrix(h_ref, src),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            apply_matrix(h_new, src),
            dst,
            rtol=1e-6,
            atol=1e-6,
        )


def test_svd_full_matrices_is_scoped_to_underdetermined_four_point(monkeypatch):
    import starstack.registration_models as rm
    real_svd = np.linalg.svd
    calls = []

    def spy(a, *args, **kwargs):
        calls.append(bool(kwargs.get('full_matrices', True)))
        return real_svd(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, 'svd', spy)
    src, dst = _make_problem(4, 401, 0.0, 1.0, 1.0, 0.15)
    rm.fit_homography(src, dst)
    assert calls == [True]
    calls.clear()
    src, dst = _make_problem(5, 402, 0.0, 1.0, 1.0, 0.15)
    rm.fit_homography(src, dst)
    assert calls == [False]
