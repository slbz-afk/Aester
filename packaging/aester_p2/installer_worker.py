"""Extracted install-time worker. stdout is installer progress, never MCP traffic."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import shutil
import traceback
import time
import uuid
import components
import install_ownership as ownership
import install_lifecycle as lifecycle


def _load_progress():
    """The bootstrap runs this script from its folder; tests load it by path."""
    try:
        import installer_progress
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location('installer_progress', Path(__file__).with_name('installer_progress.py'))
        installer_progress = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer_progress)
    return installer_progress


progress_module = _load_progress()
CANCELLED = 3
# Modules the activation probe imports when the application names none (engine-only builds and older test fixtures).
DEFAULT_PROBE = ("starstack.mcp_server", "starstack.workflow")


def activation_probe(app):
    """The modules to import before the launchers switch to this version: the application's own list
    (PACKAGE_CONTENTS.json "activation_probe", which names the desktop app's entry module in official builds),
    else DEFAULT_PROBE. A malformed list is refused rather than executed."""
    contents = Path(app) / "PACKAGE_CONTENTS.json"
    probe = json.loads(contents.read_text(encoding="utf-8")).get("activation_probe") if contents.is_file() else None
    probe = list(probe or DEFAULT_PROBE)
    name = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
    if not probe or not all(isinstance(module, str) and name.fullmatch(module) for module in probe):
        raise ValueError("Invalid activation probe in the application manifest")
    return probe


def _tree_bytes(folder):
    if not folder:
        return 0
    return sum(path.stat().st_size for path in Path(folder).rglob('*') if path.is_file())


def _restore_file(path, content):
    if content is None:
        if path.exists():
            path.unlink()
    else:
        temporary = path.with_name(path.name + ".rollback-" + uuid.uuid4().hex)
        temporary.write_bytes(content)
        temporary.replace(path)


def main():
    """Run one phase. With --status-file, Setup learns the outcome from a file
    (0 done, 1 failed, 3 cancelled) because it no longer blocks on the process."""
    args = _parser().parse_args()
    if args.pid_file:
        progress_module.write_text_atomic(args.pid_file, str(os.getpid()))
    code, state = 1, 'failed'
    try:
        code = _run(args)
        state = 'succeeded' if code == 0 else 'failed'
    except progress_module.InstallCancelled:
        code, state = CANCELLED, 'cancelled'
        print('CANCELLED', flush=True)
    finally:
        if args.status_file:
            progress_module.write_text_atomic(args.status_file, f'{state}\n{code}\n')
    return code


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-root", required=True)
    parser.add_argument("--app-root", required=True)
    parser.add_argument("--cache")
    parser.add_argument('--finite-bundle', help='Reviewed binary/source package extracted by Setup')
    parser.add_argument('--gimp', action='store_true', help='Install/reuse separate official GIMP 3.2.6')
    parser.add_argument('--retouch-skill', action='store_true', help='Prepare application-local skill source/ZIP only')
    parser.add_argument("--result", required=True)
    parser.add_argument("--activate")
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    parser.add_argument('--language', default='zh_CN', help='Installer language for progress text and notices')
    parser.add_argument('--progress-file', help='Overall progress for Setup (percent, headline, detail)')
    parser.add_argument('--status-file', help='Outcome for Setup: state and exit code')
    parser.add_argument('--pid-file', help='This process id, so Setup can tell when it ends')
    parser.add_argument('--cancel-file', help='Setup creates it when the user confirms cancelling')
    # Setup passes it in every mode (from /DOWNLOADSOURCE=); only preparation uses it.
    parser.add_argument('--download-source', choices=components.DOWNLOAD_SOURCE_MODES, default='auto',
                        help='auto: measure the official host and listed mirrors, fastest first; '
                             'official: official URLs only (for GIMP also its own mirror opencolo.mm.fcix.net), without '
                             'measuring. Every file is checked against its pinned SHA-256')
    return parser


def _run(args):
    app = components._safe_root(args.app_root)
    if app.parent.name != 'versions':
        raise ValueError('Application must be an immutable installed version')
    install_root = app.parent.parent
    lifecycle.initialize_install_root(install_root)
    preflight_path = Path(args.result).with_suffix('.preflight.json')
    if args.preflight:
        pending_before = (install_root / lifecycle.PENDING).exists()
        state = install_root / 'install-state.json'
        old_hash = components._digest(state) if state.is_file() else None
        if preflight_path.exists():
            prior = json.loads(preflight_path.read_text(encoding='utf-8'))
            if prior.get('install_root') == str(install_root) and not prior.get('activated'):
                pending_before = prior['pending_before']
                old_hash = prior['old_state_sha256']
        with lifecycle.exclusive_maintenance(install_root, mark_pending=True):
            try:
                components._atomic(preflight_path, {'schema_version': 1, 'install_root': str(install_root),
                    'pending_before': pending_before, 'old_state_sha256': old_hash, 'activated': False})
            except Exception:
                if not pending_before: lifecycle.clear_pending(install_root)
                raise
        return 0
    if args.rollback:
        if preflight_path.exists():
            previous = json.loads(preflight_path.read_text(encoding='utf-8'))
            state = install_root / 'install-state.json'
            current_hash = components._digest(state) if state.is_file() else None
            if (previous.get('install_root') == str(install_root) and not previous.get('pending_before', True)
                    and not previous.get('activated') and not previous.get('activation_started')
                    and current_hash == previous.get('old_state_sha256')):
                with lifecycle.exclusive_maintenance(install_root, mark_pending=False):
                    lifecycle.clear_pending(install_root)
        return 0
    root = components._safe_root(args.component_root)
    root.mkdir(parents=True, exist_ok=True)
    events = components._inside(root, root / "install-events.jsonl")
    tracker = None

    def progress(event):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": time.time(), **event}, ensure_ascii=False) + "\n")
        percent = round(100 * event.get("downloaded", 0) / max(1, event.get("total", 1)))
        # ASCII avoids dependence on the installer process output code page.
        print(f"PROGRESS|{percent}|{event.get('phase', '')} {event.get('filename', '')}", flush=True)
        if tracker is not None:
            tracker.handle(event)  # may raise InstallCancelled at a safe point
    activation_opened = False
    try:
        if args.activate:
            tracker = progress_module.InstallProgress(
                [{'key': 'activate', 'kind': 'timed', 'seconds': 20.0, 'text': 'activate'}],
                language=args.language, path=args.progress_file).start()
            tracker.begin('activate')
            result = json.loads(Path(args.result).read_text(encoding="utf-8"))
            receipt_path = components._inside(root, Path(result["receipt_path"]))
            if components._digest(receipt_path) != result["receipt_sha256"]:
                raise ValueError("Prepared component receipt changed")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["app_root"] != str(Path(args.app_root).absolute()):
                raise ValueError("Prepared application path changed")
            if result.get('finite_backend') is not None:
                components.verify_installed_finite_distribution(root, app_root=app,
                    expected=result['finite_backend'])
            if result.get('retouch_components') is not None:
                components.verify_prepared_retouch_components(root, app_root=app,
                    expected=result['retouch_components'])
            ownership.record_application(app)
            if preflight_path.exists():
                previous = json.loads(preflight_path.read_text(encoding='utf-8'))
                previous['activation_started'] = True
                components._atomic(preflight_path, previous)
            # A retry after an interrupted cleanup has now restored immutable
            # application files and validated the runtime. Reopen launches only
            # while holding the exclusive gate, before the child import probe.
            with lifecycle.exclusive_maintenance(install_root, mark_pending=False):
                lifecycle.clear_pending(install_root)
            activation_opened = True
            subprocess.run([receipt["python"], "-B", "-X", "utf8", "-c", "import " + ", ".join(activation_probe(app))],
                           check=True, capture_output=True, timeout=90, stdin=subprocess.DEVNULL,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            state = {"schema_version": 1, "app_root": receipt["app_root"], "component_root": str(root),
                     "receipt_path": str(receipt_path), "receipt_sha256": result["receipt_sha256"]}
            with components._lock(root):
                # Point the launchers to the prepared version only after imports pass.
                state_path, pointer_path = Path(args.activate), root / "current.json"
                old_state = state_path.read_bytes() if state_path.exists() else None
                old_pointer = pointer_path.read_bytes() if pointer_path.exists() else None
                try:
                    components._atomic(state_path, state)
                    components._atomic(pointer_path, {"version": receipt["version"], "receipt_sha256": result["receipt_sha256"]})
                except Exception:
                    _restore_file(state_path, old_state)
                    _restore_file(pointer_path, old_pointer)
                    raise
            progress({"phase": "activated", "message": "Application and offline runtime are ready"})
            tracker.stop(100)
            if preflight_path.exists():
                previous = json.loads(preflight_path.read_text(encoding='utf-8'))
                previous['activated'] = True
                components._atomic(preflight_path, previous)
        else:
            manifest = json.loads(Path(__file__).with_name("download-manifest.json").read_text(encoding="utf-8"))
            cache = components._safe_root(args.cache or root / "downloads")
            cache.mkdir(parents=True, exist_ok=True)
            gimp_plan = None
            if args.gimp:
                gimp_plan = 'install'
                try:
                    if components._retouch_distribution()._discover(None, probe=False).get('status') != 'not_found':
                        gimp_plan = 'reuse'
                except Exception:
                    pass  # Only the time estimate depends on this guess.
            tracker = progress_module.InstallProgress(progress_module.plan_steps(
                manifest, nox=False, finite_bytes=_tree_bytes(args.finite_bundle),
                retouch_skill=args.retouch_skill, gimp=gimp_plan, cache=cache,
                probe=args.download_source == 'auto'),
                language=args.language, path=args.progress_file, cancel_path=args.cancel_file).start()
            ownership.register_install(install_root, root, cache)
            seed = Path(__file__).with_name(manifest["python"]["filename"])
            if components._valid_file(seed, manifest["python"]):
                with ownership.download_lock(cache):
                    existed = (cache / seed.name).exists()
                    shutil.copyfile(seed, components._inside(cache, cache / seed.name))
                    ownership.claim_download(cache, manifest['python'], app, existed_before=existed)
            # The 'source' event of each measured family (its ranking included) lands in install-events.jsonl.
            result = components.install_components(root, include_star_support=False, progress=progress,
                         cache_dir=args.cache, app_root=args.app_root, manifest=manifest,
                         activate=False, download_source=args.download_source)
            if args.finite_bundle:
                progress({'phase': 'prepare-finite', 'message': 'Verify optional Aester Siril binary and source package'})
                result['finite_backend'] = components.install_finite_distribution(root,
                    source=args.finite_bundle, app_root=app)
                components.verify_installed_finite_distribution(root, app_root=app,
                    expected=result['finite_backend'])
            notices = []
            if args.gimp or args.retouch_skill:
                # Optional external GIMP and the skill copy never fail Aester itself:
                # what happened is recorded and shown, and can be retried later.
                try:
                    prepared = components.install_retouch_components(root,
                        include_gimp=args.gimp, include_skill=args.retouch_skill,
                        app_root=app, cache_dir=args.cache, progress=progress,
                        skill_source=Path(__file__).with_name('retouch-skill') / 'aester-retouch',
                        download_source=args.download_source)
                    components.verify_prepared_retouch_components(root, app_root=app, expected=prepared)
                    result['retouch_components'] = prepared
                    notices += prepared.get('notices', [])
                except Exception as error:
                    notices.append({'code': 'retouch_prepare_failed', 'error': str(error)})
                    progress({'phase': 'retouch-skipped', 'message': str(error)})
            result['notices'] = notices
            components._atomic(Path(args.result), result)
            _write_notices(Path(args.result).with_suffix('.notices.txt'), notices, args.language)
            tracker.stop(100)
        return 0
    except progress_module.InstallCancelled:
        if tracker is not None:
            tracker.stop()
        raise
    except Exception as error:
        if tracker is not None:
            tracker.stop()
        if activation_opened:
            try:
                with lifecycle.exclusive_maintenance(install_root, mark_pending=True):
                    pass
            except lifecycle.InstallationBusy:
                # A launch may have entered the short import-probe window.
                # Do not disturb its lease or delete files. Close future entry
                # while the per-user Inno maintenance mutex still excludes a
                # second installer/uninstaller, and require repair afterwards.
                components._atomic(install_root / lifecycle.PENDING,
                    {'schema_version': 1, 'install_root': str(install_root), 'reason': 'activation_failed'})
        detail = traceback.format_exc()
        if isinstance(error, subprocess.CalledProcessError):
            detail += "\n" + repr(error.stdout) + "\n" + repr(error.stderr)
        components._atomic(root / "install-failure.json", {"error": str(error), "traceback": detail})
        print("FAILED: " + str(error), flush=True)
        return 1


def _write_notices(path, notices, language):
    """Plain text Setup shows once before copying files; absent when all went as planned."""
    lines = []
    if notices:
        try:
            lines = components._retouch_distribution().describe_notices(notices, language)
        except Exception:
            lines = [json.dumps(notice, ensure_ascii=False) for notice in notices]
    if lines:
        progress_module.write_text_atomic(path, '\n\n'.join(lines) + '\n')
    elif path.exists():
        path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
