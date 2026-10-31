"""Optional model consent, exact pinned download and failure atomicity; no real network or inference."""
from pathlib import Path
import copy
import hashlib
import io
import threading
import urllib.request
import zipfile
from unittest.mock import Mock

import pytest

from starstack import components as c, install_lifecycle


BODY = b'local ONNX transport fixture; not a real inference model'
URL = 'https://models.example.test/releases/nox-v1.onnx'


class Response(io.BytesIO):
    def __init__(self, body=BODY, *, url=URL, status=200, headers=None):
        super().__init__(body)
        self.url, self.status, self.headers = url, status, headers or {}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    archives = {}
    for kind, names in [('python', ['python.exe', 'pythonw.exe', 'python313._pth']),
                        ('siril', ['Siril/bin/siril-cli.exe']), ('optional', ['onnxruntime/__init__.py'])]:
        path = tmp_path / (kind + '.zip')
        with zipfile.ZipFile(path, 'w') as archive:
            for name in names:
                archive.writestr(name, b'fixture')
        archives[kind] = path
    data = {'python': {'kind': 'python'}, 'wheels': [], 'siril': {'kind': 'siril'},
            'optional_wheels': [{'kind': 'optional', 'filename': 'ort.whl', 'size': 42}],
            'nox': {'filename': 'nox_color.onnx', 'size': len(BODY),
                    'sha256': hashlib.sha256(BODY).hexdigest(),
                    'upstream_url': 'https://github.com/charvey2718/nox/releases/download/v1.1.0/noxGeneratorColor.pb',
                    'download': {'url': URL, 'redirect_urls': []},
                    'adapter': {'layout': 'nhwc', 'channel_order': 'bgr', 'value_range': '-1..1',
                                'tile': 512, 'stride': 256, 'batch': 1, 'stitch': 'crop'}}}
    monkeypatch.setattr(c, '_manifest', lambda *a: data)
    downloads = []
    def dependency(item, *a):
        downloads.append(item['kind'])
        return archives[item['kind']]
    monkeypatch.setattr(c, '_download', dependency)
    monkeypatch.setattr(c, '_validate', lambda *a: None)  # transport/activation test, not an inference certification
    opener = Mock()
    opener.open.side_effect = lambda *a, **kw: Response()
    monkeypatch.setattr(c.urllib.request, 'build_opener', lambda *a: opener)
    return data, tmp_path / 'components', tmp_path / 'app', opener, downloads


def test_first_launch_info_and_default_install_do_not_download_model(setup):
    data, root, app, opener, downloads = setup
    before = copy.deepcopy(data)
    assert c.nox_install_info(app)['available']
    result = c.install_components(root, app_root=app)
    assert result['model_path'] is None and result['star_backend_options'] == {}
    assert downloads == ['python', 'siril']
    opener.open.assert_not_called()
    assert data == before


@pytest.mark.parametrize('consent', [False, None, 1, 'yes'])
def test_consent_required_before_any_component_or_network_work(setup, consent):
    data, root, app, opener, downloads = setup
    with pytest.raises(ValueError, match='主动同意'):
        c.install_nox_support(root, app_root=app, consent=consent)
    opener.open.assert_not_called()
    assert not root.exists() and not downloads


def test_opt_in_download_installs_runtime_and_binds_all_adapter_fields(setup):
    data, root, app, opener, downloads = setup
    result = c.install_nox_support(root, app_root=app, consent=True)
    assert result['ready'] and Path(result['model_path']).read_bytes() == BODY
    assert result['star_backend_options'] == data['nox']['adapter']
    assert downloads == ['python', 'optional', 'siril']
    assert Path(result['model_path']).with_name('LICENSE.txt').is_file()
    assert opener.open.call_args.args[0].full_url == URL
    c.install_nox_support(root, app_root=app, consent=True)
    assert opener.open.call_count == 1, 'repair reuses the verified existing local model'


def test_absent_url_never_uses_upstream_pb_or_environment(setup, monkeypatch):
    data, root, app, opener, downloads = setup
    data['nox']['download']['url'] = None
    monkeypatch.setenv('AESTER_NOX_URL', URL)
    monkeypatch.setenv('NOX_DOWNLOAD_URL', URL)
    info = c.nox_install_info(app)
    assert not info['available'] and '尚未提供' in info['message']
    with pytest.raises(ValueError, match='尚未提供'):
        c.install_nox_support(root, app_root=app, consent=True)
    assert not downloads and not root.exists()
    opener.open.assert_not_called()


def test_existing_model_works_without_published_download_url(setup, tmp_path):
    data, root, app, opener, downloads = setup
    data['nox']['download']['url'] = None
    model = tmp_path / '用户已有模型.onnx'
    model.write_bytes(BODY)
    result = c.install_nox_support(root, app_root=app, consent=True, model_source=model)
    assert Path(result['model_path']).read_bytes() == BODY
    opener.open.assert_not_called()
    assert model.read_bytes() == BODY


def test_explicit_wrong_local_model_does_not_silently_download(setup, tmp_path):
    data, root, app, opener, downloads = setup
    model = tmp_path / 'wrong.onnx'; model.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='哈希'):
        c.install_nox_support(root, app_root=app, consent=True, model_source=model)
    assert not downloads
    opener.open.assert_not_called()


@pytest.mark.parametrize('url', ['http://models.example.test/a', 'https://user:pw@models.example.test/a',
                                'https://models.example.test:444/a', 'https://models.example.test/a#fragment',
                                'file:///tmp/model.onnx', 'https:///model.onnx'])
def test_invalid_production_source_rejected_before_network(setup, url):
    data, root, app, opener, downloads = setup
    data['nox']['download']['url'] = url
    with pytest.raises(ValueError, match='HTTPS'):
        c.install_nox_support(root, app_root=app, consent=True)
    assert not root.exists() and not downloads
    opener.open.assert_not_called()


def test_tf_source_cannot_be_used_as_converted_onnx_release(setup):
    data, root, app, opener, downloads = setup
    data['nox']['download']['url'] = data['nox']['upstream_url']
    with pytest.raises(ValueError, match='ONNX'):
        c.install_nox_support(root, app_root=app, consent=True)
    opener.open.assert_not_called()


@pytest.mark.parametrize('redirect', ['https://models.example.test/other.onnx', 'https://other.example.test/a',
                                     'http://models.example.test/releases/nox-v1.onnx'])
def test_redirects_require_exact_declared_url(redirect):
    handler = c._ExactHTTPSRedirect([URL])
    with pytest.raises(ValueError, match='exact approved'):
        handler.redirect_request(urllib.request.Request(URL), None, 302, '', {}, redirect)


def test_declared_exact_https_redirect_is_allowed_and_final_response_checked(setup):
    data, root, app, opener, downloads = setup
    mirror = 'https://cdn.example.test/pinned/model.onnx'
    data['nox']['download']['redirect_urls'] = [mirror]
    handler = c._ExactHTTPSRedirect([URL, mirror])
    assert handler.redirect_request(urllib.request.Request(URL), None, 302, '', {}, mirror).full_url == mirror
    opener.open.side_effect = lambda *a, **kw: Response(url=mirror)
    assert c.install_nox_support(root, app_root=app, consent=True)['ready']


def test_unapproved_final_response_on_same_host_is_rejected(setup):
    data, root, app, opener, downloads = setup
    opener.open.side_effect = lambda *a, **kw: Response(url=URL + '?unexpected=1')
    with pytest.raises(ValueError, match='exact approved'):
        c.install_nox_support(root, app_root=app, consent=True)
    assert opener.open.call_count == 1 and not downloads
    assert not (root / 'current.json').exists()


def test_corrupt_hash_never_replaces_current_version(setup):
    data, root, app, opener, downloads = setup
    old = c.install_components(root, app_root=app)
    marker = (root / 'current.json').read_bytes()
    downloads.clear()
    opener.open.side_effect = lambda *a, **kw: Response(b'x' * len(BODY))
    with pytest.raises(ValueError, match='SHA256'):
        c.install_nox_support(root, app_root=app, consent=True)
    assert (root / 'current.json').read_bytes() == marker
    assert Path(old['python']).is_file() and not downloads
    assert not (root / 'downloads/nox_color.onnx').exists()
    assert not (root / 'downloads/nox_color.onnx.part').exists()


def test_connection_failure_retries_same_pinned_source_then_succeeds(setup):
    data, root, app, opener, downloads = setup
    opener.open.side_effect = [ConnectionError('fixture disconnect'), Response()]
    result = c.install_nox_support(root, app_root=app, consent=True)
    assert result['ready'] and Path(result['model_path']).read_bytes() == BODY
    assert [call.args[0].full_url for call in opener.open.call_args_list] == [URL, URL]


def test_model_runtime_validation_failure_preserves_old_then_retries_cached_model(setup, monkeypatch):
    data, root, app, opener, downloads = setup
    c.install_components(root, app_root=app)
    marker = (root / 'current.json').read_bytes()
    def fail(*args):
        raise RuntimeError('fixture model inference validation failed')
    monkeypatch.setattr(c, '_validate', fail)
    with pytest.raises(RuntimeError, match='inference validation'):
        c.install_nox_support(root, app_root=app, consent=True)
    assert (root / 'current.json').read_bytes() == marker
    assert (root / 'downloads/nox_color.onnx').read_bytes() == BODY
    monkeypatch.setattr(c, '_validate', lambda *a: None)
    assert c.install_nox_support(root, app_root=app, consent=True)['ready']
    assert opener.open.call_count == 1


def test_cancel_during_download_preserves_old_pointer_and_retry_reuses_verified_partial(setup):
    data, root, app, opener, downloads = setup
    c.install_components(root, app_root=app)
    marker = (root / 'current.json').read_bytes()
    cancelled = threading.Event()
    def progress(event):
        if event['phase'] == 'download':
            cancelled.set()
    with pytest.raises(RuntimeError, match='取消'):
        c.install_nox_support(root, app_root=app, consent=True, progress=progress, cancelled=cancelled.is_set)
    assert (root / 'current.json').read_bytes() == marker
    assert not (root / 'downloads/nox_color.onnx').exists()
    assert (root / 'downloads/nox_color.onnx.part').read_bytes() == BODY
    assert c.install_nox_support(root, app_root=app, consent=True)['ready']
    assert opener.open.call_count == 1


def test_cancel_after_validation_does_not_activate_new_version(setup):
    data, root, app, opener, downloads = setup
    c.install_components(root, app_root=app)
    marker = (root / 'current.json').read_bytes()
    cancelled = threading.Event()
    def progress(event):
        if event['phase'] == 'validated':
            cancelled.set()
    with pytest.raises(RuntimeError, match='取消'):
        c.install_nox_support(root, app_root=app, consent=True, progress=progress, cancelled=cancelled.is_set)
    assert (root / 'current.json').read_bytes() == marker


@pytest.mark.parametrize('reuse', [False, True])
def test_cancel_from_complete_callback_does_not_activate_new_or_reused_version(setup, reuse):
    data, root, app, opener, downloads = setup
    if reuse:
        c.install_nox_support(root, app_root=app, consent=True)
    # With reuse=True the verified nox version exists, but the active pointer is back on the base version.
    # A stop at its last progress event must not reactivate that earlier nox version either.
    old = c.install_components(root, app_root=app)
    marker = (root / 'current.json').read_bytes()
    cancelled = threading.Event()
    def progress(event):
        if event['phase'] == 'complete':
            cancelled.set()
    with pytest.raises(RuntimeError, match='取消'):
        c.install_nox_support(root, app_root=app, consent=True, progress=progress, cancelled=cancelled.is_set)
    assert cancelled.is_set()
    assert (root / 'current.json').read_bytes() == marker
    assert c.component_status(root)['version'] == old['version']


@pytest.mark.parametrize('reuse', [False, True])
def test_stop_after_commit_boundary_reports_success_for_new_and_reused_version(setup, monkeypatch, reuse):
    data, root, app, opener, downloads = setup
    if reuse:
        c.install_nox_support(root, app_root=app, consent=True)
    c.install_components(root, app_root=app)
    cancelled = threading.Event()
    atomic = c._atomic
    def commit(path, value):
        if path == root / 'current.json':
            cancelled.set()  # arrived after the final check; the atomic commit has started
        return atomic(path, value)
    monkeypatch.setattr(c, '_atomic', commit)
    result = c.install_nox_support(root, app_root=app, consent=True, cancelled=cancelled.is_set)
    assert cancelled.is_set() and result['ready'] and result['model_path']
    assert c.component_status(root)['version'] == result['version']


def test_pending_installer_maintenance_blocks_opt_in_before_download(setup, tmp_path):
    data, root, app, opener, downloads = setup
    installed = tmp_path / 'installed'; app = installed / 'versions/v1'
    app.mkdir(parents=True)
    install_lifecycle.initialize_install_root(installed)
    with install_lifecycle.exclusive_maintenance(installed):
        pass
    with pytest.raises(install_lifecycle.InstallationBusy):
        c.install_nox_support(root, app_root=app, consent=True)
    opener.open.assert_not_called()
    assert not root.exists()


def test_setup_has_no_model_payload_or_selection():
    script = (Path(__file__).parents[1] / 'packaging/aester_p2/Aester.iss').read_text(encoding='utf-8')
    assert 'nox_color.onnx' not in script and 'Name: "nox"' not in script
    worker = (Path(__file__).parents[1] / 'packaging/aester_p2/installer_worker.py').read_text(encoding='utf-8')
    assert 'include_star_support=False' in worker and 'nox=False' in worker
    assert 'parser.add_argument("--nox"' not in worker and 'parser.add_argument("--model"' not in worker
