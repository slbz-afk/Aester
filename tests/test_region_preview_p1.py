from pathlib import Path
import copy

import numpy as np
import pytest
from PIL import Image
from astropy.io import fits

from starstack import regions, region_preview


@pytest.fixture
def scene(tmp_path):
    inp, cache = tmp_path / 'input', tmp_path / 'cache'
    inp.mkdir(); cache.mkdir()
    source = inp / 'source.NEF'
    source.write_bytes(b'original raw')
    linear = cache / 'reference.fit'
    fits.writeto(linear, np.ones((3, 60, 100), np.float32))
    image = np.zeros((60, 100, 3), np.uint8)
    image[..., 0] = np.arange(60)[:, None] + 20
    image[..., 1] = 35
    image[..., 2] = 60
    path = cache / 'reference.png'
    Image.fromarray(image).save(path)
    reference = {'state': 'succeeded', 'source_path': str(source), 'source_sha256': regions._sha(source),
                 'linear_path': str(linear), 'linear_sha256': regions._sha(linear), 'width': 100, 'height': 60,
                 'preview_path': str(path), 'preview_sha256': regions._sha(path), 'preview_width': 100, 'preview_height': 60}
    project = regions.create_project(reference, [str(source)])
    return project, reference, image, tmp_path / 'output'


def test_overlay_orientation_and_shape_preserve_untouched_sky(scene):
    project, reference, original, out = scene
    project = regions.revise(project, scene_mode='landscape', operations=[{'kind': 'horizon', 'points': [[0, .6], [1, .6]]}])
    before = copy.deepcopy(project)
    result = region_preview.render_preview(project, reference, out)
    image = np.asarray(Image.open(result['preview_path']))
    assert image.shape == original.shape
    assert np.array_equal(image[:34], original[:34])
    # 2026-09-27 拓星: foreground fill changed deliberately from the old #FA8543 to the
    # colour-blind-safe overlay_colors.GROUND #D55E00, at the same 95/255 alpha.
    expected = np.rint(original[50, 20].astype(float) * (1 - 95/255) + np.array([0xD5, 0x5E, 0x00]) * 95/255)
    assert np.max(np.abs(image[50, 20].astype(float) - expected)) <= 1
    assert project == before
    assert result['sha256'] == regions._sha(result['preview_path'])
    assert result['row_order'] == 'TOP-DOWN' and result['width'] == 100 and result['height'] == 60


def test_unknown_milkyway_is_shown_but_never_approved(scene):
    project, reference, original, out = scene
    project = regions.revise(project, scene_mode='sky', milkyway={**project['milkyway'], 'ellipse': [.5,.5,.25,.4]})
    result = region_preview.render_preview(project, reference, out)
    image = np.asarray(Image.open(result['preview_path']))
    assert not np.array_equal(image[30, 50], original[30, 50])
    assert np.array_equal(image[0, 0], original[0, 0])
    assert result['milkyway_status'] == 'unknown' and not project['milkyway']['confirmed']
    assert '不代表' in result['legend']['milkyway']
    again = region_preview.render_preview(project, reference, out)
    assert again['preview_path'] != result['preview_path']
    assert again['sha256'] == result['sha256']


@pytest.mark.parametrize('key,value', [('source_path', 'other.NEF'), ('source_sha256', '0'*64),
                                     ('linear_path', 'other.fit'), ('linear_sha256', '0'*64), ('width', 99), ('height', 59)])
def test_reference_identity_mismatch_stops_before_decoding(scene, monkeypatch, key, value):
    project, reference, _, out = scene
    reference = {**reference, key: value}
    monkeypatch.setattr(Image, 'open', lambda *_: pytest.fail('mismatched reference must not decode'))
    with pytest.raises(ValueError, match='不一致'):
        region_preview.render_preview(project, reference, out)
    assert not out.exists()


def test_tampered_preview_rejected_before_decoder_and_no_outputs(scene, monkeypatch):
    project, reference, _, out = scene
    Path(reference['preview_path']).write_bytes(b'modified preview')
    monkeypatch.setattr(Image, 'open', lambda *_: pytest.fail('unverified bytes must not decode'))
    with pytest.raises(ValueError, match='已改变'):
        region_preview.render_preview(project, reference, out)
    assert not out.exists()


def test_decoder_uses_verified_bytes_despite_file_replacement(scene, monkeypatch):
    from io import BytesIO
    project, reference, original, out = scene
    project = regions.revise(project, scene_mode='sky')
    original_open = Image.open
    invoked = []
    def replace_before_decode(stream, *args, **kwargs):
        assert isinstance(stream, BytesIO)
        invoked.append(True)
        Path(reference['preview_path']).write_bytes(b'replaced after verified read')
        return original_open(stream, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', replace_before_decode)
    result = region_preview.render_preview(project, reference, out)
    assert len(invoked) == 1
    assert np.array_equal(np.asarray(original_open(result['preview_path'])), original)


def test_larger_preview_is_bounded_and_aspect_mismatch_rejected(scene):
    project, reference, _, out = scene
    Image.new('RGB', (2000, 1200), (10,20,30)).save(reference['preview_path'])
    reference.update(preview_sha256=regions._sha(reference['preview_path']), preview_width=2000, preview_height=1200)
    result = region_preview.render_preview(project, reference, out)
    assert (result['width'], result['height']) == (1600, 960)
    Image.new('RGB', (2000, 800), (10,20,30)).save(reference['preview_path'])
    reference.update(preview_sha256=regions._sha(reference['preview_path']), preview_height=800)
    with pytest.raises(ValueError, match='比例'):
        region_preview.render_preview(project, reference, out)


def test_original_folder_output_is_refused(scene):
    project, reference, _, _ = scene
    with pytest.raises(ValueError, match='原片'):
        region_preview.render_preview(project, reference, Path(reference['source_path']).parent / 'preview')
