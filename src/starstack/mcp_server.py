"""Local stdio MCP adapter with host-configured filesystem boundaries."""

import asyncio
import functools
import hashlib
import importlib
import json
import math
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from PIL import Image

from . import core, hashing


def _hash_scoped(label):
    """Run one tool operation inside a hashing run scope: its repeated source checks share one ledger."""
    def decorate(function):
        @functools.wraps(function)
        def scoped(*args, **kwargs):
            with hashing.run_scope(label):
                return function(*args, **kwargs)
        return scoped
    return decorate


class AccessDenied(ValueError):
    """A requested path is outside the host-configured workspace."""


def _linked(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _no_link_components(path: Path) -> None:
    for part in (path, *path.parents):
        if _linked(part):
            raise AccessDenied("不允许符号链接或目录联接；请使用真实目录和文件")


def _bounded(value: str | Path, root: Path, *, exists: bool = True) -> Path:
    text = str(value)
    if not text or "\x00" in text:
        raise AccessDenied("路径不能为空或包含空字符")
    path = Path(text).expanduser()
    if ".." in path.parts:
        raise AccessDenied("路径不能包含上级目录跳转")
    if path.drive and not path.is_absolute():
        raise AccessDenied("不允许相对盘符路径")
    if not path.is_absolute():
        path = root / path
    if not path.is_relative_to(root):
        raise AccessDenied("路径超出此服务允许的目录")
    _no_link_components(path)
    resolved = path.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise AccessDenied("路径超出此服务允许的目录")
    return resolved


def _check_tree(directory: Path) -> None:
    """Reject linked descendants before core can follow them during a mutation."""
    def on_error(error: OSError) -> None:
        raise error

    for current, folders, files in os.walk(directory, followlinks=False, onerror=on_error):
        for name in folders + files:
            path = Path(current) / name
            if _linked(path):
                raise AccessDenied("任务目录内不允许符号链接或目录联接")


def _absolute_record_path(value: str, root: Path) -> Path:
    # Persisted evidence must use one interpretation in this adapter and the
    # backend, regardless of the process working directory.
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise AccessDenied("RAW 证据记录中的路径必须是绝对路径")
    return _bounded(value, root)


class WorkspaceTools:
    """Host-owned roots are configuration, never caller-controlled tool arguments."""

    def __init__(self, input_root: str | Path | None = None,
                 output_root: str | Path | None = None,
                 siril_path: str | Path | None = None, *,
                 stack_engine_path: str | Path | None = None,
                 star_model_path: str | Path | None = None,
                 star_backend_options: dict | None = None,
                 cache_root: str | Path | None = None,
                 dng_converter_path: str | Path | None = None):
        self.input_root = self._root(input_root or os.environ.get("STARSTACK_INPUT_ROOT", "data"))
        products = self._root(output_root or os.environ.get("STARSTACK_OUTPUT_ROOT", "outputs"))
        cache = cache_root or os.environ.get("STARSTACK_CACHE_ROOT")
        self.output_root = self._root(cache) if cache else products
        self.delivery_root = products if cache else None
        if self.input_root == self.output_root:
            raise AccessDenied("素材目录与结果目录必须不同")
        if self.delivery_root is not None:
            if self.delivery_root == self.input_root:
                raise AccessDenied("素材与作品目录必须不同")
            if (self.output_root.is_relative_to(self.delivery_root)
                    or self.delivery_root.is_relative_to(self.output_root)):
                raise AccessDenied("作品与缓存目录不能相互包含")
        self.stack_engine_path = str(Path(stack_engine_path or os.environ["STARSTACK_STACK_ENGINE"]).resolve()) if stack_engine_path or os.environ.get("STARSTACK_STACK_ENGINE") else None
        self.star_model_path = str(Path(star_model_path or os.environ["STARSTACK_STAR_MODEL"]).resolve()) if star_model_path or os.environ.get("STARSTACK_STAR_MODEL") else None
        self.star_backend_options = dict(star_backend_options) if star_backend_options is not None else json.loads(os.environ.get("STARSTACK_STAR_ADAPTER", "{}"))
        if not isinstance(self.star_backend_options, dict):
            raise ValueError("STARSTACK_STAR_ADAPTER 需要模型适配参数对象")
        # Executable configuration belongs to the host, never a tool argument.
        self.siril_path = os.fspath(siril_path) if siril_path is not None else os.environ.get("STARSTACK_SIRIL")
        self.dng_converter_path = (os.fspath(dng_converter_path) if dng_converter_path is not None
                                   else os.environ.get('AESTER_DNG_CONVERTER'))

    @staticmethod
    def _root(value: str | Path) -> Path:
        path = Path(value).expanduser().absolute()
        _no_link_components(path)
        return path.resolve()

    def _output(self) -> Path:
        # Recheck configuration in case a directory was replaced after startup.
        _no_link_components(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        for entry in self.output_root.iterdir():
            if _linked(entry):
                raise AccessDenied("结果根目录内不允许符号链接或目录联接")
        return self.output_root

    def _product_file(self, value: str | Path) -> Path:
        """Advanced render inputs may be in either host-owned output area."""
        try:
            return _bounded(value, self.output_root)
        except AccessDenied:
            if self.delivery_root is None or not Path(value).is_absolute():
                raise
            return _bounded(value, self.delivery_root)

    def _workflow_product(self, value, record) -> Path:
        if not record['config'].get('delivery_root'):
            return _bounded(value, self.output_root)
        from .storage import publication_parent, delivery_file
        parent = publication_parent(record)
        _bounded(parent, self.delivery_root)
        _bounded(value, parent)
        return delivery_file(record, value)

    def _source_files(self, values, *, absolute=False, imports=None) -> list[Path]:
        """Admit originals or exact verified imported outputs, never all cache.

        The memo is operation-local. Persisting it on this service would let a
        later tool call skip integrity checks after a file or record changed.
        """
        if not isinstance(values, list) or not 1 <= len(values) <= 300:
            raise AccessDenied('素材路径须为1至300项列表')
        admissions = {} if imports is None else imports
        candidates = []
        import_root = self.output_root / 'raw-imports'
        for value in values:
            if not isinstance(value, str) or (absolute and not Path(value).is_absolute()):
                raise AccessDenied('素材证据必须使用绝对文件路径')
            path = Path(value).expanduser()
            interpreted = path if path.is_absolute() else self.input_root / path
            if not interpreted.is_relative_to(self.output_root):
                source = _bounded(value, self.input_root)
                candidates.append((source, None))
                continue
            # Cache input is permitted only through an explicit absolute file
            # selection; relative names continue to refer to the original root.
            if not path.is_absolute():
                raise AccessDenied('兼容导入副本须使用返回的绝对文件路径')
            source = _bounded(path, import_root)
            parts = source.relative_to(import_root).parts
            if (len(parts) < 2 or source.suffix.lower() != '.dng'
                    or not re.fullmatch(r'[a-f0-9]{24}(?:-[a-f0-9]{8})?', parts[0])):
                raise AccessDenied('只允许完整兼容导入记录中的DNG副本')
            manifest = _bounded(import_root / parts[0] / 'import.json', import_root)
            candidates.append((source, manifest))
        # All selected path boundaries precede any import-record hash reads.
        from .raw_import import validate_import
        for _, manifest in candidates:
            if manifest is not None and manifest not in admissions:
                record = validate_import(manifest, original_root=self.input_root,
                                         cache_root=self.output_root)
                admissions[manifest] = {Path(p) for p in record['input_files']}
        for source, manifest in candidates:
            if manifest is not None and source not in admissions[manifest]:
                raise AccessDenied('该DNG未列入兼容导入记录')
        return [source for source, _ in candidates]

    def _source_directory(self, value, files) -> Path:
        """A cache directory is context for an explicit list, never a scan root."""
        try:
            candidate = _bounded(value, self.input_root)
            if not candidate.is_relative_to(self.output_root):
                return candidate
        except AccessDenied:
            pass
        if not Path(value).is_absolute():
            raise AccessDenied('兼容素材目录须使用绝对路径')
        candidate = _bounded(value, self.output_root / 'raw-imports')
        if candidate not in {p.parent for p in files}:
            raise AccessDenied('素材目录不是已核查副本的直接父目录')
        return candidate

    def import_photos(self, input_files: list[str], *, force_dng: bool = False) -> dict:
        """Create/reuse local compatibility copies with a host-chosen converter."""
        if type(input_files) is not list or not 1 <= len(input_files) <= 300:
            raise ValueError('请选择1至300张原始照片')
        if type(force_dng) is not bool:
            raise ValueError('force_dng须为bool')
        files = [str(_bounded(p, self.input_root)) for p in input_files]
        from .raw_import import prepare_import
        result = prepare_import(files, self._output(), converter_path=self.dng_converter_path,
                                force_dng=force_dng, allow_converter_discovery=False)
        self._source_files(result['input_files'])
        return result

    def inspect_sources(self, input_dir: str | None = None, *, input_files: list[str] | None = None,
                        min_count: int = 3, _imports=None) -> dict[str, Any]:
        if input_files is not None:
            if not isinstance(input_files, list) or any(not isinstance(p, str) for p in input_files):
                raise ValueError("input_files 必须是文件路径列表")
            admitted = self._source_files(input_files, imports=_imports)
            files = list(map(str, admitted))
            directory = str(self._source_directory(input_dir, admitted)) if input_dir is not None else None
            return core.inspect_sources(directory, input_files=files, min_count=min_count)
        if input_dir is None:
            raise ValueError("请选择照片或素材文件夹")
        directory = _bounded(input_dir, self.input_root)
        if directory.is_relative_to(self.output_root):
            raise AccessDenied('缓存不能作为素材目录扫描，请使用兼容导入返回的文件清单')
        if not directory.is_dir():
            raise ValueError("素材路径必须是文件夹")
        # core scans only immediate RAW/FITS children. Validate before its stat/read.
        for path in directory.iterdir():
            if path.suffix.lower() in core.RAW | core.FITS:
                _bounded(path, self.input_root)
        return core.inspect_sources(str(directory), **({"min_count": min_count} if min_count != 3 else {}))

    def inspect_raw(self, source_path: str) -> dict[str, Any]:
        source = self._source_files([source_path])[0]
        if not source.is_file() or source.suffix.lower() not in core.RAW:
            raise ValueError("需要允许素材目录中的单张 RAW／DNG")
        from .raw_audit import inspect_raw
        return inspect_raw(source, self._output())

    def get_help(self) -> dict[str, Any]:
        from .user_guide import guide_text
        from .i18n import get_language
        title = 'Aester · GUI and MCP guide' if get_language() == 'en' else '星野后期 · GUI 与 MCP 共用指南'
        return {"title": title, "format": "markdown", "content": guide_text()}

    def _region_project(self, value: str, *, verify_sources: bool = True, _imports=None) -> tuple[Path, dict]:
        """Validate paths before the shared validator opens referenced files."""
        path = _bounded(value, self.output_root)
        if not path.is_file() or path.stat().st_size > 16 * 1024**2:
            raise AccessDenied("分区项目文件无效或过大")
        record = json.loads(path.read_text(encoding="utf-8"))
        if type(record) is not dict or type(record.get("source_binding")) is not dict:
            raise ValueError("分区项目格式无效")
        binding = record.get("source_binding", {})
        if type(binding.get("input_files")) is not list or type(binding.get("reference")) is not dict:
            raise ValueError("分区来源格式无效")
        source_paths = []
        for frame in binding.get("input_files", []):
            if type(frame) is not dict:
                raise ValueError("分区素材记录格式无效")
            source_paths.append(frame.get("path", ""))
        ref = binding.get("reference", {})
        admissions = {} if _imports is None else _imports
        self._source_files(source_paths, absolute=True, imports=admissions)
        self._source_files([ref.get('source_path', '')], absolute=True, imports=admissions)
        _absolute_record_path(ref.get("linear_path", ""), self.output_root)
        from .regions import validate_project
        # Validate the exact bytes whose paths were checked; never reread an
        # attacker-replaceable manifest between the path gate and its hash reads.
        return path, validate_project(record, verify_sources=verify_sources) | {"project_path": str(path)}

    @staticmethod
    def _region_presence(milkyway, user_confirmation, previous=None):
        if milkyway is None:
            if user_confirmation is not None:
                raise ValueError("银河确认必须同时提供对应银河区域和判断")
            return None
        if type(milkyway) is not dict:
            raise ValueError("milkyway 必须是对象")
        if not milkyway:
            return None
        if user_confirmation is not None and (type(user_confirmation) is not str
                or not user_confirmation.strip() or len(user_confirmation) > 2000):
            raise ValueError("用户确认须为非空原话，最多2000字")
        from .regions import DRAWING
        selected = {**(previous or {}), **milkyway}
        edited = any(key in milkyway for key in ('ellipse', 'angle_degrees', 'feather'))
        status = milkyway.get('status', ('present' if selected.get('ellipse') else 'absent')
                              if edited else selected.get('status', 'unknown'))
        # A drawing tool is an explicit edit instruction. It never authenticates
        # a human or upgrades the caller's provenance, even if relaying a request.
        selected.update(status=status, authority='agent', reviewer='MCP agent',
                        rationale=user_confirmation or str(milkyway.get('rationale') or 'Explicit region drawing via MCP'),
                        confirmed=status == 'present', selection=DRAWING)
        return selected

    @_hash_scoped("mcp.create_regions")
    def create_regions(self, reference_dir: str, input_files: list[str], *,
                       scene_mode: str = "unknown", operations: list | None = None,
                       milkyway: dict | None = None, user_confirmation: str | None = None) -> dict:
        directory = _bounded(reference_dir, self.output_root)
        _check_tree(directory)
        path = _bounded(directory / "reference.json", directory)
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("参考记录过大")
        reference = json.loads(path.read_text(encoding="utf-8"))
        admissions = {}
        self._source_files([reference.get("source_path", "")], absolute=True, imports=admissions)
        _absolute_record_path(reference.get("linear_path", ""), directory)
        files = self.inspect_sources(input_files=input_files, min_count=1, _imports=admissions)["files"]
        from . import regions
        project = regions.create_project(reference, files)
        presence = self._region_presence(milkyway, user_confirmation, project['milkyway'])
        if presence is not None:
            presence = {**project["milkyway"], **presence}
        project = regions.revise(project, scene_mode=scene_mode, operations=operations, milkyway=presence)
        if scene_mode == 'sky' and operations is None:
            project = regions.clear_landscape(project)
        project = regions.normalize_marks(project)
        return regions.save_project(project, self._output())

    @_hash_scoped("mcp.update_regions")
    def update_regions(self, regions_path: str, *, scene_mode: str | None = None,
                       operations: list | None = None, milkyway: dict | None = None,
                       user_confirmation: str | None = None) -> dict:
        _, project = self._region_project(regions_path)
        from . import regions
        presence = self._region_presence(milkyway, user_confirmation, project['milkyway'])
        if presence is not None:
            presence = {**project["milkyway"], **presence}
        revised = regions.revise(project, scene_mode=scene_mode, operations=operations, milkyway=presence)
        if scene_mode == 'sky' and operations is None:
            revised = regions.clear_landscape(revised)
        revised = regions.normalize_marks(revised)
        return regions.save_project(revised, self._output())

    @_hash_scoped("mcp.draw_regions")
    def draw_regions(self, reference_dir: str, *, input_files: list[str] | None = None,
                     regions_path: str | None = None, horizon: list | None = None,
                     ground_strokes: list | None = None, sky_strokes: list | None = None,
                     brush_radius: float = .018, milkyway_ellipse: list | None = None,
                     milkyway_angle: float | None = None, milkyway_feather: float | None = None,
                     clear_ground: bool = False, clear_milkyway: bool = False) -> dict:
        """Shared drawing vocabulary; prior immutable revisions provide undo."""
        from . import regions
        if regions_path:
            project = self._region_project(regions_path)[1]
            if input_files is not None:
                files = self.inspect_sources(input_files=input_files, min_count=1)['files']
                regions.validate_project(project, input_files=files)
        else:
            if input_files is None:
                raise ValueError('新建分区需要 input_files')
            project = self.create_regions(reference_dir, input_files)
        if clear_ground:
            project = regions.clear_landscape(project)
        if clear_milkyway:
            project = regions.clear_milkyway(project)
        operations = list(project['operations'])
        if horizon is not None:
            operations.append({'kind': 'horizon', 'points': horizon})
        for target, strokes in [('ground', ground_strokes), ('sky', sky_strokes)]:
            if strokes is not None:
                if not isinstance(strokes, list):
                    raise ValueError('笔画须为归一化坐标折线列表')
                for points in strokes:
                    operations.append({'kind': 'paint', 'target': target, 'points': points, 'radius': brush_radius})
        project = regions.normalize_marks(regions.revise(project, operations=operations))
        if milkyway_ellipse is not None or milkyway_angle is not None or milkyway_feather is not None:
            ellipse = milkyway_ellipse if milkyway_ellipse is not None else project['milkyway']['ellipse']
            if ellipse is None:
                raise ValueError('请先绘制银河椭圆')
            project = regions.draw_milkyway(project, ellipse, angle_degrees=milkyway_angle,
                feather=milkyway_feather, authority='agent', reviewer='MCP agent',
                rationale='Explicit region drawing via MCP')
        saved = regions.save_project(project, self._output())
        preview = self.preview_regions(saved['project_path'], reference_dir)
        return {'project_path': saved['project_path'], 'revision': saved['revision'],
                'scene_mode': regions.scene_from_marks(saved), 'milkyway_enabled': regions.milkyway_active(saved),
                'project': saved, 'preview': preview}

    def get_regions(self, regions_path: str) -> dict:
        return self._region_project(regions_path)[1]

    def preview_regions(self, regions_path: str, reference_dir: str) -> dict:
        admissions = {}
        _, project = self._region_project(regions_path, _imports=admissions)
        directory = _bounded(reference_dir, self.output_root)
        _check_tree(directory)
        path = _bounded(directory / "reference.json", directory)
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("参考记录过大")
        reference = json.loads(path.read_text(encoding="utf-8"))
        if type(reference) is not dict:
            raise ValueError("参考记录格式无效")
        self._source_files([reference.get("source_path", "")], absolute=True, imports=admissions)
        _absolute_record_path(reference.get("linear_path", ""), directory)
        _absolute_record_path(reference.get("preview_path", ""), directory)
        from .region_preview import render_preview
        return render_preview(project, reference, self._output())

    def _review_session(self, workflow_dir: str, session_path: str) -> tuple[Path, Path]:
        directory, record = self._workflow(workflow_dir)
        if not record.get("stack_job_dir"):
            raise ValueError("工作流尚无堆栈作业")
        child, _ = self._job(record["stack_job_dir"])
        path = _bounded(session_path, child)
        if path.parent != child / "review" or not path.name.startswith("draft-"):
            raise AccessDenied("复核草稿不属于当前作业")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("job_dir") != str(child) or raw.get("session_path") != str(path):
            raise AccessDenied("复核草稿内的作业绑定不匹配")
        return directory, path

    def open_trail_review(self, workflow_dir: str) -> dict:
        _, record = self._workflow(workflow_dir)
        if not record.get("stack_job_dir"):
            raise ValueError("请先等堆栈完成")
        directory, _ = self._job(record["stack_job_dir"])
        from .trail_review_session import open_review
        return open_review(str(directory))

    def save_trail_review(self, workflow_dir: str, session_path: str, decisions: dict,
                          expected_draft_revision: str, export_trail_layer: bool | None = None) -> dict:
        if export_trail_layer is not None and type(export_trail_layer) is not bool:
            raise ValueError("export_trail_layer 必须是 bool")
        _, path = self._review_session(workflow_dir, session_path)
        from .trail_review_session import save_review
        options = None if export_trail_layer is None else {"export_trail_layer": export_trail_layer}
        return save_review(str(path), decisions, reviewer="agent", expected_draft_revision=expected_draft_revision,
                           options=options)

    def trail_candidate_context(self, workflow_dir: str, session_path: str, candidate_id: str) -> dict:
        _, path = self._review_session(workflow_dir, session_path)
        from .trail_review_session import candidate_context
        return candidate_context(str(path), candidate_id)

    def apply_trail_review(self, workflow_dir: str, session_path: str, expected_draft_revision: str) -> dict:
        directory, path = self._review_session(workflow_dir, session_path)
        from .workflow_edit import apply_review
        if not isinstance(expected_draft_revision, str) or not expected_draft_revision:
            raise ValueError("应用复核需要刚保存的草稿版本")
        return apply_review(str(directory), str(path), expected_draft_revision=expected_draft_revision)

    def _workflow(self, value: str) -> tuple[Path, dict]:
        directory = _bounded(value, self.output_root)
        if not directory.name.startswith("workflow-"):
            raise AccessDenied("只允许结果范围中的 workflow- 作业")
        _check_tree(directory)
        record_path = _bounded(directory / "workflow.json", directory)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if _absolute_record_path(record.get("workflow_dir", ""), self.output_root) != directory:
            raise AccessDenied("工作流所属目录不匹配")
        config = record["config"]
        request = record["request"]
        if request.get("config") != config or request.get("input_dir") != config.get("input_dir"):
            raise AccessDenied("工作流配置与冻结请求不一致")
        admissions = {}
        admitted_sources = self._source_files(request.get('input_files', []), absolute=True, imports=admissions)
        self._source_directory(config['input_dir'], admitted_sources)
        if _absolute_record_path(config["output_root"], self.output_root) != directory.parent:
            raise AccessDenied("工作流输出根不匹配")
        if config.get('delivery_root'):
            if (self.delivery_root is None or not Path(config['delivery_root']).is_absolute()
                    or _bounded(config['delivery_root'], self.delivery_root, exists=False) != self.delivery_root):
                raise AccessDenied("作品目录不是宿主配置")
            if request.get('delivery_root') != config['delivery_root']:
                raise AccessDenied("作品目录与冻结请求不一致")
        if config.get('delivery_format') is not None:
            from .storage import VERSION, IMAGE_VERSION
            if (not config.get('delivery_root') or config['delivery_format'] not in (VERSION, IMAGE_VERSION)
                    or request.get('delivery_format') != config['delivery_format']):
                raise AccessDenied("作品格式与冻结请求不一致")
        # The user-facing directory contains images only. New integrity
        # receipts live in this job's cache, never in arbitrary host files.
        if record.get('publication_receipts'):
            from .storage import IMAGE_VERSION, publication_parent, delivery_receipt
            if config.get('delivery_format') != IMAGE_VERSION:
                raise AccessDenied("旧作品记录不能混入新的发布凭据")
            publication = publication_parent(record)
            _bounded(publication, self.delivery_root)
            for destination, descriptor in record['publication_receipts'].items():
                target = _absolute_record_path(destination, publication)
                if target.parent != publication or not target.name.startswith('delivery-'):
                    raise AccessDenied("作品发布位置不属于当前工作流")
                receipt_path = _absolute_record_path(descriptor['path'], directory / 'publication-receipts')
                if receipt_path.parent != directory / 'publication-receipts':
                    raise AccessDenied("作品凭据必须属于当前工作流")
                delivery_receipt(record, target)
        if config.get('delivery_format') == 'aester-delivery-2' and record.get('state') == 'succeeded':
            from .storage import delivery_receipt
            required = {record.get('delivery_dir'), record.get('base_delivery_dir')}
            required.update(item.get('delivery_dir') for item in record.get('review_deliveries', []))
            if not record.get('delivery_dir'):
                raise AccessDenied("已完成工作流缺少作品位置")
            for target in required - {None}:
                # Status reads validate small cached receipts. Full image hash
                # verification remains mandatory before review/cache deletion.
                delivery_receipt(record, target)
        for name, configured in (("siril_path", self.siril_path), ("stack_engine_path", self.stack_engine_path), ("star_model_path", self.star_model_path)):
            selected = config.get(name)
            if name == "siril_path" and configured is None:
                configured = core.doctor().get("executable")
            # A workflow may omit an optional host model, but may never choose a different executable/model.
            if selected is not None and (configured is None or Path(selected).resolve() != Path(configured).resolve()):
                raise AccessDenied("工作流引擎或模型不是宿主配置")
        if config.get("sky_mask_path"):
            self._workflow_mask(config["sky_mask_path"])
        allowed = set(admitted_sources)
        if config.get("regions_path"):
            region_path, region_project = self._region_project(config["regions_path"], verify_sources=False, _imports=admissions)
            if request.get("regions_path") != str(region_path):
                raise AccessDenied("分区路径与冻结请求不一致")
            allowed.add(region_path)
            region_assets = request.get("region_assets") or {}
            if region_assets.get("source_binding") != region_project.get("source_binding"):
                raise AccessDenied("分区来源与冻结素材不一致")
            for key in ("project_path", "sky_mask_path", "milkyway_mask_path"):
                if region_assets.get(key):
                    allowed.add(_absolute_record_path(region_assets[key], self.output_root))
            allowed.add(_absolute_record_path(region_project["source_binding"]["reference"]["linear_path"], self.output_root))
        for name, item in request.get("assets", {}).items():
            if name not in ("siril_path", "stack_engine_path", "star_model_path", "sky_mask_path"):
                raise AccessDenied("工作流含未知宿主资源")
            if item is not None:
                if not config.get(name) or item.get("path") != config[name]:
                    raise AccessDenied("工作流资源与配置不匹配")
                allowed.add(Path(item["path"]).resolve())
        for item in request.get("files", []):
            if not Path(item["path"]).is_absolute() or Path(item["path"]).resolve() not in allowed:
                raise AccessDenied("工作流含未授权输入文件")
        for checkpoint_name, checkpoint in record.get("checkpoints", {}).items():
            for item in checkpoint.get("files", []):
                if not Path(item['path']).is_absolute():
                    raise AccessDenied('检查点路径须为绝对路径')
                if checkpoint_name == 'delivery':
                    self._workflow_product(item['path'], record)
                else:
                    _absolute_record_path(item["path"], self.output_root)
        if record.get("stack_job_dir"):
            stack = _absolute_record_path(record["stack_job_dir"], directory)
            if stack.parent != directory / "stack" or not stack.name.startswith("job-"):
                raise AccessDenied("堆栈子作业不属于此工作流")
            _check_tree(stack)
            _bounded(stack / "job.json", stack)
            child = core.load_job(stack)
            if child.get("job_dir") != str(stack):
                raise AccessDenied("堆栈子作业所属目录不匹配")
            from .workflow import validate_stack_binding
            validate_stack_binding(record, child)
        for artifact in record.get("artifacts", []):
            self._workflow_product(artifact['path'], record)
        return directory, record

    def _workflow_mask(self, value: str) -> Path:
        try:
            return _bounded(value, self.input_root)
        except AccessDenied:
            # prepare_reference writes masks into output_root; relative mask names still mean input_root.
            if not Path(value).is_absolute():
                raise
            return _bounded(value, self.output_root)

    @_hash_scoped("mcp.submit_workflow")
    def submit_workflow(self, input_dir: str | None = None, sky_mask_path: str | None = None, *,
                        input_files: list[str] | None = None, trails_enabled: bool = False,
                        stars_enabled: bool = False, color_adjustments: dict | None = None,
                        star_reduction: bool | None = None, background_strength: float | None = None,
                        idempotency_key: str | None = None, stack_range_mode: str = "legacy",
                        regions_path: str | None = None, milkyway_enabled: bool = False,
                        milkyway_strength: float = .6, milkyway_offset: float = 0.,
                        image_rotation: int | None = None, trail_default: str | None = None,
                        save_starless: bool = False, save_star_layer: bool = False,
                        jpeg_quality: int | None = None) -> dict:
        if jpeg_quality is not None:
            from .jpeg_estimate import validate_quality
            validate_quality(jpeg_quality)
        if image_rotation is not None:
            from .orientation import normalize_rotation
            image_rotation = normalize_rotation(image_rotation)
        if type(trails_enabled) is not bool or type(stars_enabled) is not bool:
            raise ValueError("trails_enabled 与 stars_enabled 必须是 bool")
        if type(save_starless) is not bool or type(save_star_layer) is not bool:
            raise ValueError("save_starless 与 save_star_layer 必须是 bool")
        if trail_default is not None and trail_default not in ("meteor", "review_only"):
            raise ValueError("trail_default 须为 meteor、review_only 或省略")
        # Compatibility alias for existing agent prompts. New prompts use the
        # same stars_enabled field as the GUI and Python API.
        if star_reduction is not None:
            if type(star_reduction) is not bool:
                raise ValueError("star_reduction 必须是 bool")
            if stars_enabled and not star_reduction:
                raise ValueError("star_reduction 与 stars_enabled 冲突")
            stars_enabled = star_reduction
        if stars_enabled and not self.star_model_path:
            raise ValueError("宿主尚未配置去星模型；请配置后再启用星点整理")
        sources = self.inspect_sources(input_dir, **({"input_files": input_files} if input_files is not None else {}))
        mask = self._workflow_mask(sky_mask_path) if sky_mask_path else None
        region = self._region_project(regions_path)[0] if regions_path else None
        output = self._output()
        if idempotency_key is not None:
            if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 200:
                raise ValueError("幂等标识须为 1–200 个字符")
            ident = hashlib.sha256(idempotency_key.encode()).hexdigest()[:24]
            existing = output / f"workflow-{ident}"
            if existing.exists():
                self._workflow(str(existing))
        from . import workflow
        result = workflow.submit_workflow(sources["input_dir"], str(output), siril_path=self.siril_path,
                    sky_mask_path=str(mask) if mask else None, stack_engine_path=self.stack_engine_path,
                    star_model_path=self.star_model_path if stars_enabled else None,
                    backend_options=self.star_backend_options if stars_enabled else None,
                    background_strength=background_strength, idempotency_key=idempotency_key,
                    stack_range_mode=stack_range_mode,
                    input_files=sources["files"] if input_files is not None else None,
                    trails_enabled=trails_enabled, stars_enabled=stars_enabled,
                    color_adjustments=color_adjustments,
                    **({'image_rotation': image_rotation} if image_rotation is not None else {}),
                    **({'trail_default': trail_default} if trail_default is not None else {}),
                    **({'save_starless': True} if save_starless else {}),
                    **({'save_star_layer': True} if save_star_layer else {}),
                    **({'jpeg_quality': jpeg_quality} if jpeg_quality is not None else {}),
                    **({'delivery_root': str(self.delivery_root)} if self.delivery_root is not None else {}),
                    **({"regions_path": str(region), "milkyway_enabled": milkyway_enabled,
                        "milkyway_strength": milkyway_strength, "milkyway_offset": milkyway_offset}
                       if region is not None or milkyway_enabled or milkyway_offset != 0 else {}))
        return self.workflow_status(result["workflow_dir"])

    def workflow_status(self, workflow_dir: str) -> dict:
        directory, _ = self._workflow(workflow_dir)
        from .workflow import workflow_status
        result = workflow_status(str(directory))
        for artifact in result.get("artifacts", []):
            self._workflow_product(artifact['path'], result)
        return result

    def inspect_workflow_cache(self, workflow_dir: str) -> dict:
        """Read-only: what the desktop 作品与缓存 page could free for this job. Deletion is never offered here.

        Admitted by the job folder alone (``_workflow_folder``), not by ``_workflow``: that one admits the
        originals too, and users move their RAWs, which the desktop cleanup by design does not require.
        """
        directory = self._workflow_folder(workflow_dir)
        from .cache_cleanup import inspect_workflow_cache
        return inspect_workflow_cache(str(directory))

    def _workflow_folder(self, value: str) -> Path:
        """A ``workflow-`` folder directly in this host's cache root, link-free, whose record names this folder,
        this root and (when it has one) this host's photographs root. Nothing outside the folder is admitted."""
        directory = _bounded(value, self.output_root)
        if not directory.name.startswith("workflow-") or directory.parent != self.output_root:
            raise AccessDenied("只允许结果范围中的 workflow- 作业")
        _check_tree(directory)
        record = json.loads(_bounded(directory / "workflow.json", directory).read_text(encoding="utf-8"))
        if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
            raise AccessDenied("工作流配置与冻结请求不一致")
        if _absolute_record_path(record.get("workflow_dir", ""), self.output_root) != directory:
            raise AccessDenied("工作流所属目录不匹配")
        config = record["config"]
        if _absolute_record_path(config.get("output_root", ""), self.output_root) != directory.parent:
            raise AccessDenied("工作流输出根不匹配")
        if config.get("delivery_root") and (
                self.delivery_root is None or not Path(config["delivery_root"]).is_absolute()
                or _bounded(config["delivery_root"], self.delivery_root, exists=False) != self.delivery_root):
            raise AccessDenied("作品目录不是宿主配置")
        return directory

    def cancel_workflow(self, workflow_dir: str) -> dict:
        directory, _ = self._workflow(workflow_dir)
        from .workflow import cancel_workflow
        cancel_workflow(str(directory))
        return self.workflow_status(str(directory))

    def resume_workflow(self, workflow_dir: str) -> dict:
        directory, _ = self._workflow(workflow_dir)
        from .workflow import resume_workflow
        resume_workflow(str(directory))
        return self.workflow_status(str(directory))

    def protect_trace(self, audit_dir: str, rectangle: list[int],
                      classification: str = "unknown", note: str = "") -> dict[str, Any]:
        directory = _bounded(audit_dir, self.output_root)
        if directory.parent != self.output_root or not directory.name.startswith("raw-audit-"):
            raise AccessDenied("只允许结果根目录下的 RAW 检查记录")
        _check_tree(directory)
        manifest = _bounded(directory / "audit.json", directory)
        if manifest.stat().st_size > 2 * 1024**2:
            raise ValueError("RAW 检查记录过大")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if _absolute_record_path(record.get("audit_dir", ""), self.output_root) != directory:
            raise AccessDenied("RAW 检查记录所属目录不匹配")
        self._source_files([record.get("source", {}).get("path", "")], absolute=True)
        for artifact in record.get("artifacts", {}).values():
            _absolute_record_path(artifact.get("path", ""), directory)
        from .transients import record_candidate
        return record_candidate(directory, self._output(), rectangle=rectangle,
                                classification=classification, note=note)

    def _job(self, value: str) -> tuple[Path, dict[str, Any]]:
        directory = _bounded(value, self.output_root)
        if not directory.name.startswith("job-"):
            raise AccessDenied("仅允许访问结果根目录中的 job- 任务文件夹")
        if directory.parent != self.output_root:
            workflow_dir = directory.parent.parent
            if directory.parent.name != "stack" or not workflow_dir.is_relative_to(self.output_root):
                raise AccessDenied("堆栈子作业不属于输出根中的工作流")
            _, workflow = self._workflow(str(workflow_dir))
            if workflow.get("stack_job_dir") != str(directory):
                raise AccessDenied("工作流未登记这个堆栈子作业")
        if not directory.is_dir():
            raise ValueError("任务路径必须是文件夹")
        _check_tree(directory)
        _bounded(directory / "job.json", directory)
        job = core.load_job(str(directory))
        if _bounded(job.get("job_dir", ""), self.output_root) != directory:
            raise AccessDenied("任务记录的所属目录不匹配")
        return directory, job

    def _artifacts(self, directory: Path, job: dict[str, Any]) -> list[dict[str, Any]]:
        items = []
        for item in job.get("artifacts", []):
            path = _bounded(item["path"], directory)
            if not path.is_file():
                raise ValueError("产物不是普通文件")
            items.append({"path": str(path), "kind": item["kind"], "label": item["label"]})
        return items

    def _status(self, directory: Path, job: dict[str, Any]) -> dict[str, Any]:
        diagnostics = job.get("diagnostics") or {}
        input_count = diagnostics.get("input_count")
        stacked_count = diagnostics.get("stacked_count")
        # Older jobs recorded the first two counts only. Derive the difference
        # without editing their manifest; missing engine counts stay unknown.
        skipped_count = (input_count - stacked_count
                         if type(input_count) is int and type(stacked_count) is int else None)
        return {"job_dir": str(directory),
                **{key: job.get(key) for key in ("state", "stage", "progress", "message", "error")},
                "diagnostics": {"input_count": input_count, "stacked_count": stacked_count,
                                "skipped_count": skipped_count,
                                **({'exposure': diagnostics['exposure']} if 'exposure' in diagnostics else {}),
                                **({'stack_range': diagnostics['stack_range']} if 'stack_range' in diagnostics else {})},
                "recovery_blocked": bool(job.get("recovery_blocked", False)),
                "artifacts": self._artifacts(directory, job),
                "limitations": job.get("limitations", [])}

    def submit_job(self, input_dir: str, sky_mask_path: str | None = None,
                   idempotency_key: str | None = None, support_mode: str = "legacy",
                   stack_range_mode: str = "legacy", force_restart: bool = False,
                   image_rotation: int | None = None) -> dict[str, Any]:
        if image_rotation is not None:
            from .orientation import normalize_rotation
            image_rotation = normalize_rotation(image_rotation)
        if type(force_restart) is not bool:
            raise ValueError("force_restart 须为 bool")
        if force_restart and idempotency_key is None:
            raise ValueError("force_restart 需要 idempotency_key（只对同一提交编号的已结束作业重跑）")
        sources = self.inspect_sources(input_dir)
        mask = _bounded(sky_mask_path, self.input_root) if sky_mask_path else None
        if mask and (not mask.is_file() or mask.suffix.lower() != ".png"):
            raise ValueError("天空蒙版必须是允许素材目录内的 PNG 文件")
        output = self._output()
        if idempotency_key is not None:
            if not idempotency_key or len(idempotency_key) > 200:
                raise ValueError("幂等标识须为 1–200 个字符")
            ident = hashlib.sha256(idempotency_key.encode()).hexdigest()[:24]
            existing = output / f"job-{ident}"
            if existing.exists():
                self._job(str(existing))
        result = core.submit_job(sources["input_dir"], str(output),
                                 siril_path=self.siril_path,
                                 sky_mask_path=str(mask) if mask else None,
                                 idempotency_key=idempotency_key, support_mode=support_mode,
                                 stack_range_mode=stack_range_mode, force_restart=force_restart,
                                 **({'image_rotation': image_rotation} if image_rotation is not None else {}),
                                 **({"stack_engine_path": self.stack_engine_path} if stack_range_mode == "finite_validity_v1" else {}))
        directory, job = self._job(result["job_dir"])
        return self._status(directory, job)

    def job_status(self, job_dir: str) -> dict[str, Any]:
        directory, job = self._job(job_dir)
        return self._status(directory, job)

    def prepare_reference(self, input_dir: str | None = None, *, input_files: list[str] | None = None,
                          color_adjustments: dict | None = None, image_rotation: int | None = None) -> dict[str, Any]:
        if image_rotation is not None:
            from .orientation import normalize_rotation
            image_rotation = normalize_rotation(image_rotation)
        admissions = {}
        sources = self.inspect_sources(input_dir, input_files=input_files, min_count=1, _imports=admissions)
        output = self._output()
        reference = importlib.import_module(".references", package=__package__)
        extra = {}
        if input_files is not None:
            extra["input_files"] = sources["files"]
        if color_adjustments is not None:
            extra["color_adjustments"] = color_adjustments
        if image_rotation is not None:
            extra["image_rotation"] = image_rotation
        result = reference.prepare_reference(sources["input_dir"], str(output), self.siril_path, **extra)
        directory = _bounded(result["reference_dir"], output)
        if directory.parent != output or not directory.name.startswith("reference-") or not directory.is_dir():
            raise AccessDenied("参考图必须保存在结果根目录下独立的 reference- 文件夹")
        _check_tree(directory)
        result = {**result, "reference_dir": str(directory)}
        for key in ("preview_path", "mask_template_path", "linear_path"):
            path = _bounded(result[key], directory)
            if not path.is_file():
                raise ValueError("参考图产物不是普通文件")
            result[key] = str(path)
        for key in ("staged_path", "log_path"):
            if key in result:
                result[key] = str(_bounded(result[key], directory))
        source = self._source_files([result["source_path"]], absolute=True, imports=admissions)[0]
        if source != Path(sources["files"][0]).resolve():
            raise AccessDenied("参考图原片必须是已检查序列的首帧")
        result["source_path"] = str(source)
        return result

    def rotate_reference(self, reference_dir: str, image_rotation: int) -> dict[str, Any]:
        """Rotate an admitted reference into a new immutable reference directory."""
        from .orientation import normalize_rotation
        rotation = normalize_rotation(image_rotation)
        directory = _bounded(reference_dir, self.output_root)
        _check_tree(directory)
        path = _bounded(directory / 'reference.json', directory)
        if not path.is_file() or path.stat().st_size > 16 * 1024**2:
            raise ValueError('参考记录无效或过大')
        record = json.loads(path.read_text(encoding='utf-8'))
        if type(record) is not dict or record.get('state') != 'succeeded':
            raise ValueError('不是已完成的参考图')
        if _absolute_record_path(record.get('reference_dir', ''), self.output_root) != directory:
            raise AccessDenied('参考图目录绑定不一致')
        admissions = {}
        source = self._source_files([record.get('source_path', '')], absolute=True, imports=admissions)[0]
        for key in ('linear_path', 'preview_path', 'mask_template_path', 'staged_path', 'log_path'):
            if key in record:
                _absolute_record_path(record[key], directory)
        cache = record.get('color_cache')
        if cache is not None:
            if type(cache) is not dict:
                raise ValueError('参考颜色缓存格式无效')
            for item in (cache, cache.get('interactive')):
                if item is not None:
                    if type(item) is not dict:
                        raise ValueError('参考颜色缓存格式无效')
                    for key in ('linear', 'saturated'):
                        _absolute_record_path(item.get(key, ''), directory)
        reference = importlib.import_module('.references', package=__package__)
        # The shared implementation verifies hashes from these exact admitted
        # bytes, rather than deciding file access from a second manifest read.
        result = reference.rotate_reference(str(directory), image_rotation=rotation, _record=record)
        output_dir = _bounded(result['reference_dir'], self.output_root)
        if output_dir.parent != directory.parent or not output_dir.name.startswith('reference-'):
            raise AccessDenied('旋转参考图必须保存在同级独立 reference- 文件夹')
        _check_tree(output_dir)
        for key in ('linear_path', 'preview_path', 'mask_template_path'):
            artifact = _absolute_record_path(result[key], output_dir)
            if not artifact.is_file():
                raise ValueError('旋转参考图产物不是普通文件')
        if Path(result['source_path']).resolve() != source:
            raise AccessDenied('旋转参考图不能改变原片身份')
        return result

    def render_job_preview(self, job_dir: str, red_gain: float = 1.0,
                           blue_gain: float = 1.0, exposure_ev: float = 0.0) -> dict[str, Any]:
        for value, label, low, high in ((red_gain, "红通道", .25, 4),
                                       (blue_gain, "蓝通道", .25, 4),
                                       (exposure_ev, "曝光", -3, 3)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{label} 必须为 {low} 到 {high} 之间的有限数值")
        self._output()
        directory, job = self._job(job_dir)
        self._artifacts(directory, job)
        renderer = importlib.import_module(".preview", package=__package__)
        result = renderer.render_job_preview(str(directory), red_gain=red_gain,
                                             blue_gain=blue_gain, exposure_ev=exposure_ev)
        rendered = _bounded(result["directory"], directory)
        if rendered.parent != directory / "preview-variations" or not rendered.is_dir():
            raise AccessDenied("新预览必须位于此任务的 preview-variations 子目录")
        _check_tree(rendered)
        recipe = _bounded(result["recipe_path"], rendered)
        if not recipe.is_file():
            raise ValueError("预览配方不是普通文件")
        items = self._artifacts(rendered, result)
        checked = {**result, "directory": str(rendered), "recipe_path": str(recipe),
                   "artifacts": [{**item, **safe} for item, safe in zip(result["artifacts"], items)]}
        if "source_path" in checked:
            checked["source_path"] = str(_bounded(checked["source_path"], directory))
        if "job_dir" in checked and _bounded(checked["job_dir"], self.output_root) != directory:
            raise AccessDenied("预览配方的所属任务不匹配")
        return checked

    def coordinate_background(self, job_dir: str, samples: str | None = None,
                              recipe: str | None = None, mode: str | None = None,
                              strength: float | None = None, red_gain: float | None = None,
                              blue_gain: float | None = None) -> dict[str, Any]:
        self._output()
        directory, job = self._job(job_dir)
        self._artifacts(directory, job)
        # User-supplied samples live within the host's output workspace;
        # recipes must be from this job. No model-selected external paths.
        sample_path = _bounded(samples, self.output_root) if samples is not None else None
        recipe_path = _bounded(recipe, directory) if recipe is not None else None
        renderer = importlib.import_module('.background_job', package=__package__)
        result = renderer.render_background(str(directory), sample_path, recipe_path,
                                            mode=mode, strength=strength, red_gain=red_gain, blue_gain=blue_gain)
        output = _bounded(result['directory'], directory)
        if output.parent != directory/'background-variations':
            raise AccessDenied('背景派生必须保存到本作业独立目录')
        _check_tree(output)
        _bounded(result['recipe_path'], output)
        self._artifacts(output, result)
        return result

    def cancel_job(self, job_dir: str) -> dict[str, Any]:
        directory, _ = self._job(job_dir)
        if directory.parent != self.output_root:
            raise ValueError("这是完整工作流的子作业，请使用 cancel_workflow 停止整个流程")
        core.cancel_job(str(directory))
        result = self.job_status(str(directory))
        result["cancel_requested"] = (directory / "cancel.request").exists()
        return result

    def resume_job(self, job_dir: str) -> dict[str, Any]:
        self._output()
        directory, job = self._job(job_dir)
        if directory.parent != self.output_root:
            raise ValueError("这是完整工作流的子作业，请使用 resume_workflow 恢复整个流程")
        self._source_files([source['path'] for source in job.get('inputs', [])], absolute=True)
        if job.get("mask"):
            _bounded(job["mask"]["path"], self.input_root)
        for records in job.get("checkpoint_files", {}).values():
            for record in records:
                _bounded(record["path"], directory)
        self._artifacts(directory, job)
        # A persisted executable is not an unrestricted execute-program capability.
        trusted = core.doctor(self.siril_path) if self.siril_path else core.doctor()
        saved_exe = job.get("engine", {}).get("executable")
        if not trusted.get("available") or not saved_exe:
            raise ValueError("未找到此 MCP 服务配置的 Siril，无法恢复")
        if Path(saved_exe).resolve() != Path(trusted["executable"]).resolve():
            raise AccessDenied("此任务的 Siril 与 MCP 服务配置路径不同；请通过桌面界面恢复")
        core.resume_job(str(directory))
        return self.job_status(str(directory))

    def list_transients(self, job_dir: str, needs_review: bool = False) -> dict[str, Any]:
        """Read current review evidence; filtering never changes any disposition."""
        if type(needs_review) is not bool:
            raise ValueError("needs_review 须为 bool")
        directory, job = self._job(job_dir)
        from .transient_stage import read_record, summary as transient_summary
        from .cli import candidate_view
        record = read_record(directory / "work", job=job)
        if record is None:
            return {"job_dir": str(directory), "transients": None, "message": "这个作业没有亮迹记录（在亮迹保护之前完成）"}
        current_summary = transient_summary(record)
        views = []
        for c in record["candidates"]:
            view = candidate_view(c)
            if view.get("review_png"):
                png = _bounded(view["review_png"], directory)
                view["review_png"] = str(png) if png.is_file() else None
            views.append(view)
        if needs_review:
            by_id = {v["id"]: v for v in views}
            views = [by_id[cid] for cid in current_summary["recall_review"]["candidate_ids"]]
        return {"job_dir": str(directory), "summary": current_summary, "policy": record.get("policy"), "candidates": views,
                "candidate_filter": "needs_review" if needs_review else "all", "summary_scope": "all_candidates",
                "returned_candidates": len(views),
                "related_candidate_groups": record.get("related_candidate_groups"), "application_by_source": record.get("application_by_source"),
                "artifacts": [a for a in self._artifacts(directory, job) if a.get("label", "").startswith("含亮迹")],
                "classes": ["meteor", "satellite", "aircraft", "artifact", "unknown"], "decisions": ["keep", "omit"]}

    def set_transients(self, job_dir: str, labels: dict[str, str | None] | None = None, overrides: dict[str, str | None] | None = None,
                       policy: dict[str, str] | None = None, marks: list[dict[str, Any]] | None = None,
                       layer_style: str | None = None) -> dict[str, Any]:
        """Marks search by default; protect:true requests preservation without relabelling.

        An override omit cancels add-back. Eligibility is not complete recovery;
        list_transients returns unmet requests and actual application evidence.
        """
        directory, job = self._job(job_dir)
        if directory.parent != self.output_root:
            _, workflow = self._workflow(str(directory.parent.parent))
            if workflow.get("state") != "succeeded":
                raise ValueError("完整出图完成后才能复核并重合成亮迹")
            frozen = workflow.get("checkpoints", {}).get("stack", {}).get("files", [])
            if any(Path(item["path"]).resolve() == directory / "job.json" for item in frozen):
                raise ValueError("此早期工作流冻结了任务索引，不能原地重做亮迹；请以新版本创建独立作业")
        if layer_style is not None and layer_style not in ("source_residual", "photographic_v3"):
            raise ValueError("未知亮迹渲染版本")
        for mapping, name in ((labels, "labels"), (overrides, "overrides"), (policy, "policy")):
            if mapping is not None and (not isinstance(mapping, dict) or len(mapping) > 500
                                        or any(not isinstance(k, str) or (v is not None and not isinstance(v, str)) for k, v in mapping.items())):
                raise ValueError(f"{name} 须为 字符串 -> 字符串/空 的映射")
        if marks is not None:
            if not isinstance(marks, list) or len(marks) > 100:
                raise ValueError("marks 须为不超过 100 项的列表")
            for mark in marks:
                if not isinstance(mark, dict) or not isinstance(mark.get("input"), str):
                    raise ValueError("每个标记须含 input（作业输入的文件名）与 x0 y0 x1 y1")
                if "protect" in mark and type(mark["protect"]) is not bool:
                    raise ValueError("mark.protect 须为 bool；true 请求保护，省略或 false 仅搜索")
        self._artifacts(directory, job)
        options = {"layer_style": layer_style} if layer_style is not None else {}
        core.recompose_transients(str(directory), labels=labels, overrides=overrides, policy=policy, marks=marks,
                                  reviewer="agent", **options)
        return self.list_transients(str(directory))

    def finish_photo(self, source: str, sky_mask: str, protected_mask: str,
                     mode: str = "photographic", detail_strength: float = 1.,
                     star_strength: float = 1., ground_strength: float = 0.,
                     jpeg_quality: int | None = None) -> dict[str, Any]:
        """Explicit local stack derivative, from files inside the output root."""
        paths = [_bounded(p, self.output_root) for p in (source, sky_mask, protected_mask)]
        root = self._output()
        from .photographic_job import finish_photo
        result = finish_photo(*[str(paths[0]), str(root), str(paths[1]), str(paths[2])],
            mode=mode, detail_strength=detail_strength, star_strength=star_strength,
            ground_strength=ground_strength, **({"jpeg_quality": jpeg_quality} if jpeg_quality is not None else {}))
        directory = _bounded(result["processing_directory"], root)
        for artifact in result.get("artifacts", {}).values():
            _bounded(artifact["path"], directory)
        return result

    def finish_milkyway(self, source: str, sky_mask: str, protected_mask: str, milkyway_mask: str,
                        brightness: float = .020, contrast: float = .60, structure: float = .90,
                        color: float = .35, strength: float = .60,
                        transient_background: str | None = None,
                        starless_background: str | None = None,
                        milkyway_evidence: str | None = None,
                        jpeg_quality: int | None = None) -> dict[str, Any]:
        """Local presentation through explicit masks, all inside the output root."""
        paths = [_bounded(p, self.output_root) for p in (source, sky_mask, protected_mask, milkyway_mask)]
        paired_options = ({"transient_background": str(_bounded(transient_background, self.output_root))}
                          if transient_background is not None else {})
        if starless_background is not None:
            paired_options["starless_background"] = str(_bounded(starless_background, self.output_root))
        if milkyway_evidence is not None:
            paired_options["milkyway_evidence"] = str(_bounded(milkyway_evidence, self.output_root))
        if jpeg_quality is not None:
            paired_options["jpeg_quality"] = jpeg_quality
        root = self._output()
        from .photographic_job import finish_milkyway
        result = finish_milkyway(str(paths[0]), str(root), str(paths[1]), str(paths[2]), str(paths[3]),
            brightness=brightness, contrast=contrast, structure=structure, color=color, strength=strength,
            **paired_options)
        directory = _bounded(result["processing_directory"], root)
        for artifact in result.get("artifacts", {}).values():
            _bounded(artifact["path"], directory)
        return result

    def export_star_layers(self, separation_result: str) -> dict[str, Any]:
        """Export editable TIFF layers; all referenced paths stay in this root."""
        from .star_layer_export import export_star_layers
        result = export_star_layers(str(_bounded(separation_result,self.output_root)),str(self._output()),
                                    allowed_root=self.output_root)
        directory = _bounded(result['processing_directory'],self.output_root)
        for artifact in result.get('artifacts',{}).values():
            _bounded(artifact['path'],directory)
        return result

    def reduce_stars(self, separation_result: str, strength: float = .55,
                     bright_start: float = .10, bright_end: float = .45,
                     milkyway_mask: str | None = None, milkyway_evidence: str | None = None,
                     milkyway_relief: float = .5, bright_radius: int | None = None,
                     background_strength: float | None = None,
                     milkyway_offset: float = -.2, jpeg_quality: int | None = None) -> dict[str, Any]:
        """Reduce faint display-domain stars from a bounded, verified layer record."""
        for value, name, low, high in ((background_strength, "background_strength", 0, 1),
                                        (milkyway_offset, "milkyway_offset", -1, 1)):
            if value is None and name == "background_strength":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} 须为 {low} 至 {high} 的有限数值")
        record = _bounded(separation_result, self.output_root)
        mask = None if milkyway_mask is None else str(_bounded(milkyway_mask, self.output_root))
        evidence = (None if milkyway_evidence is None
                    else str(_bounded(milkyway_evidence, self.output_root)))
        root = self._output()
        from .star_reduction import reduce_stars
        range_options = ({"background_strength": background_strength, "milkyway_offset": milkyway_offset}
                         if background_strength is not None else {})
        if jpeg_quality is not None:
            range_options["jpeg_quality"] = jpeg_quality
        result = reduce_stars(str(record), str(root), strength=strength,
            bright_start=bright_start, bright_end=bright_end,
            milkyway_mask=mask, milkyway_evidence=evidence,
            milkyway_relief=milkyway_relief, bright_radius=bright_radius,
            allowed_root=self.output_root, **range_options)
        directory = _bounded(result['processing_directory'], root)
        for artifact in result.get('artifacts', {}).values():
            _bounded(artifact['path'], directory)
        return result

    def suggest_star_reduction(self, separation_result: str,
                               milkyway_mask: str | None = None,
                               milkyway_evidence: str | None = None) -> dict[str, Any]:
        """Report a density-based starting point without rendering a derivative."""
        record = _bounded(separation_result, self.output_root)
        mask = None if milkyway_mask is None else str(_bounded(milkyway_mask, self.output_root))
        evidence = (None if milkyway_evidence is None
                    else str(_bounded(milkyway_evidence, self.output_root)))
        root = self._output()
        from .star_reduction import suggest_star_reduction
        result = suggest_star_reduction(str(record), str(root), milkyway_mask=mask,
            milkyway_evidence=evidence, allowed_root=self.output_root)
        directory = _bounded(result['processing_directory'], root)
        for artifact in result.get('artifacts', {}).values():
            _bounded(artifact['path'], directory)
        return result

    def _reference_path(self, value: str) -> Path:
        """References must stay in a host-approved input, work or product root."""
        try:
            return _bounded(value, self.input_root)
        except AccessDenied:
            return self._product_file(value)

    def export_master(self, source: str, reference_dng: str | None = None, recipe: str | None = None,
                      wb: str = "daylight", gamut: str = "rec2020", dcp_profile: str | None = None) -> dict[str, Any]:
        """M · scene-referred linear master from a finished job's camera-RGB master (package doc §2.1)."""
        from .color_products import export_master
        source_path = _bounded(source, self.output_root)
        root = self._output()
        reference = None if reference_dng is None else str(self._reference_path(reference_dng))
        recipe_path = None if recipe is None else str(_bounded(recipe, self.output_root))
        profile = None if dcp_profile is None else str(self._reference_path(dcp_profile))
        result = export_master(str(source_path), str(root), reference_dng=reference, recipe=recipe_path, wb_mode=wb,
                               gamut=gamut, dcp_profile=profile)
        _bounded(result["directory"], root)
        return result

    def render_photo(self, master: str, sky_mask: str | None = None, wb: str = "as_shot", contrast: float = 12.0,
                     saturation: float = 1.0, target: float = .22, candidates: bool = False,
                     shadow_smoothing: bool = True, look: str = "auto", dcp_profile: str | None = None) -> dict[str, Any]:
        """D · finished photo from an M master with the single display transform (package doc §2.2, recipe photo-default-1)."""
        from .color_products import render_photo
        master_path = self._product_file(master)
        root = self._output()
        mask = None if sky_mask is None else str(self._reference_path(sky_mask))
        profile = None if dcp_profile is None else str(self._reference_path(dcp_profile))
        result = render_photo(str(master_path), str(root), sky_mask=mask, wb_mode=wb, contrast=contrast, saturation=saturation,
                              target=target, candidates=candidates, shadow_smoothing=shadow_smoothing, look=look, dcp_profile=profile)
        _bounded(result["directory"], root)
        return result

    def preview(self, job_dir: str, artifact_index: int = 0) -> dict[str, Any]:
        directory, job = self._job(job_dir)
        images = [item for item in self._artifacts(directory, job) if item["kind"] == "jpeg"]
        if artifact_index < 0 or artifact_index >= len(images):
            raise ValueError("预览序号不存在；只可选择此任务已登记的 JPG，序号从 0 开始")
        item = images[artifact_index]
        path = Path(item["path"])
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise AccessDenied("预览只允许 JPG 产物")
        with Image.open(path) as picture:
            if picture.format != "JPEG":
                raise AccessDenied("预览文件内容不是 JPG")
            width, height = picture.size
        return {**item, "job_dir": str(directory), "artifact_index": artifact_index,
                "width": width, "height": height, "bytes": path.stat().st_size,
                "mime_type": "image/jpeg", "transport": "local_path_only",
                "message": "仅返回本机产物路径与尺寸，未传输图片或原始素材内容"}


def create_server(workspace: WorkspaceTools | None = None):
    """Construct official SDK v1 FastMCP; importing path checks needs no SDK."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import CallToolResult, ImageContent, TextContent
    from pydantic import Field
    from contextlib import asynccontextmanager
    import anyio
    from .finishing_service import FinishingTools
    from .finishing_mcp import register_tools as register_finishing_tools
    from .mcp_replies import json_native, structured_tools
    from .retouch_integration import get_retouch_info, build_retouch_prompt, read_retouch_skill

    # Windows DLL initialization can hang when NumPy is first imported from a
    # worker while stdio threads are alive. Load the core's FITS reader here.
    importlib.import_module("astropy.io.fits")
    service = workspace or WorkspaceTools()
    finishing = FinishingTools(service)

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:
            with anyio.CancelScope(shield=True):
                await asyncio.to_thread(finishing.close)

    server = FastMCP("StarStack", log_level="WARNING", lifespan=lifespan, instructions=(
        "用户请求自然语言修图时，先 retouch_setup、finishing_capabilities，建议加载/安装aester-retouch技能；技能非强制，不自动写客户端全局目录。"
        "技能可通过aester://retouch/skill/SKILL.md及references资源读取；与堆栈工具同属此服务，无需另接开发服务器。"
        "GIMP 3.2.6仅为修图选装；缺失时引导至GUI「组件与提示词管理」（旧称“组件管理”），不能阻断基础堆栈。finishing_*从16位sRGB成片TIFF建立会话，配方迭代/预览后另存JPEG/TIFF/XCF。"
        "本地星野后期。get_help 返回与GUI相同指南。RAW压缩不受支持时可import_photos生成兼容DNG，使用宿主配置转换器且原片保留；后续沿用返回的input_files。完整出图先 inspect_sources，再 submit_workflow；保存 workflow_dir，用 workflow_status 查询，cancel_workflow/resume_workflow 停止/恢复。"
        "仅堆栈也可 submit_job，保存 job_dir，使用 job_status 查询。"
        "input_dir、sky_mask_path 相对 STARSTACK_INPUT_ROOT；job_dir/workflow_dir 相对宿主工作根（有 STARSTACK_CACHE_ROOT 时使用它，否则沿用 STARSTACK_OUTPUT_ROOT）。"
        "配置了缓存根时，完整作业作品另存于 STARSTACK_OUTPUT_ROOT；请沿用返回的绝对任务和作品路径。"
        "inspect_workflow_cache 只读报告已完成任务可释放的中间文件；删除只能由用户在桌面“设置 → 作品与缓存”确认。"
        "也可传这些根目录内的绝对路径。输出根目录由宿主固定；不支持调用者选择。"
        "prepare_reference 生成首帧完整预览与全白编辑模板（非分割结果）。"
        "render_job_preview 从已完成任务生成独立的通道平衡/曝光预览，不改原任务和线性母版。"
        "inspect_raw 可检查单张RAW并另存解码CFA；protect_trace 保存人工指定区域的单帧证据，尚不合成流星。"
        "list_transients/set_transients 查看与修改亮迹候选的标签、处置与策略，基础成片不主动加回亮迹层，仍可能有堆栈残留。"
        "完整出图 trail_default=meteor 时，基础成片完成后自动另存加回量测可靠流星（算法建议）的版本；其余候选仍待复核。"
        "preview 只返回本机 JPG 路径；preview_regions默认也仅给路径，include_image=true会将小尺寸分区叠色图返回当前MCP客户端，须符合用户的图片访问授权。RAW已接母版/成片颜色，严格浮点全流程与自动天空分割仍未全部验收。"))
    # Every tool declares an object output schema and replies with structuredContent in JSON-native types.
    tool = structured_tools(server)

    @server.resource("aester://retouch/skill/{section}")
    def retouch_skill(section: str) -> str:
        """Read the bundled SKILL.md. Installing it in the Agent client is optional."""
        return read_retouch_skill(section)

    @server.resource("aester://retouch/skill/references/{name}")
    def retouch_skill_reference(name: str) -> str:
        """Read one reviewed retouch skill reference; no arbitrary files."""
        return read_retouch_skill("references/" + name)

    @tool()
    async def retouch_setup() -> dict[str, Any]:
        """自然语言修图试用引导：只读检测GIMP与技能包，返回加载提示词；技能推荐不强制，不安装/执行/上传。"""
        def inspect():
            from .i18n import get_language
            return {**get_retouch_info(gimp_path=finishing.gimp_path),
                    "prompt": build_retouch_prompt(gimp_path=finishing.gimp_path, language=get_language())}
        return await asyncio.to_thread(inspect)

    @tool()
    async def get_help() -> dict[str, Any]:
        """读取GUI与MCP共用的离线指南：步骤、参数含义、产物与限制相同。"""
        return service.get_help()

    @tool()
    async def import_photos(input_files: list[str],
                            force_dng: Annotated[bool, Field(strict=True)] = False) -> dict[str, Any]:
        """将不兼容RAW导入为无损CFA DNG副本，原片不动；只用宿主配置的转换器。"""
        return await asyncio.to_thread(service.import_photos, input_files, force_dng=force_dng)

    @tool()
    async def submit_workflow(input_dir: str | None = None, sky_mask_path: str | None = None,
                             input_files: list[str] | None = None,
                             trails_enabled: Annotated[bool, Field(strict=True)] = False,
                             stars_enabled: Annotated[bool, Field(strict=True)] = False,
                             color_adjustments: dict[str, Any] | None = None,
                             star_reduction: Annotated[bool, Field(strict=True)] | None = None,
                             background_strength: Annotated[float, Field(ge=0, le=1)] | None = None,
                             idempotency_key: str | None = None,
                             stack_range_mode: Literal["legacy", "positive_power2_v1"] = "legacy",
                             regions_path: str | None = None,
                             milkyway_enabled: Annotated[bool, Field(strict=True)] = False,
                             milkyway_strength: Annotated[float, Field(strict=True, ge=0, le=1)] = .6,
                             milkyway_offset: Annotated[float, Field(strict=True, ge=-1, le=1)] = 0.,
                             image_rotation: Annotated[int, Field(strict=True)] | None = None,
                             trail_default: Literal["meteor", "review_only"] | None = None,
                             save_starless: Annotated[bool, Field(strict=True)] = False,
                             save_star_layer: Annotated[bool, Field(strict=True)] = False,
                             jpeg_quality: Annotated[int, Field(strict=True, ge=60, le=100)] | None = None) -> dict[str, Any]:
        """提交后台完整出图：堆栈→母版/成片→可选分层/缩星。返回workflow_dir用于状态/取消/恢复。

        input_files 按指定顺序只处理所选照片，不扫描同目录其余文件。效果默认关闭。
        stars_enabled=true使用宿主模型并要求天空蒙版；强度null采用建议，0保持，1尝试完整移星。
        trails_enabled=true检测亮迹。trail_default="meteor"（与界面一致）时，基础成片完成后自动另存加回量测可靠流星候选的版本
        （delivery-trails-<n>kept-*，算法建议、不认证身份；带质量标记或未量测的不加回），基础成片保留，其余候选待复核；
        省略或"review_only"为旧行为：只检测，候选不会自动加回。
        save_starless/save_star_layer（需stars_enabled）另存无星图 starless_srgb16.tif、星点加/减层 stars_add/stars_subtract_srgb16.tif（编码sRGB，无星−减层+加层=缩星前成片）。
        color_adjustments={warmth,tint,exposure_ev} 与界面共用，范围分别±100、±100、±4；均0沿用默认。
        star_reduction 为旧版 stars_enabled 别名。
        image_rotation 为全部素材相对解码像素顺时针0/90/180/270度；省略时采用分区方向或首帧拍摄方向，分区与显式方向必须相同。
        jpeg_quality（60–100整数，界面默认95）为本任务全部JPEG（final.jpg及之后的含亮迹版本）的质量，4:4:4色度，冻结在提交中；
        数值越高越清晰、文件越大（q95约为q90的1.6倍，q100约为q95的3.4倍）。省略时各JPEG保持原有编码：基础成片95/4:4:4，
        缩星成片95/4:2:0，银河局部增强成片及亮迹版本97/4:4:4，FITS成片94/4:2:0。TIFF与母版不受影响。
        引擎/模型只能由宿主配置；配置浮点引擎时自动用finite_validity_v1。缺银河依据不增强。
        """
        return await asyncio.to_thread(service.submit_workflow, input_dir, sky_mask_path,
                star_reduction=star_reduction, background_strength=background_strength,
                idempotency_key=idempotency_key, stack_range_mode=stack_range_mode,
                input_files=input_files, trails_enabled=trails_enabled,
                stars_enabled=stars_enabled, color_adjustments=color_adjustments,
                regions_path=regions_path, milkyway_enabled=milkyway_enabled,
                milkyway_strength=milkyway_strength, milkyway_offset=milkyway_offset,
                image_rotation=image_rotation, trail_default=trail_default,
                save_starless=save_starless, save_star_layer=save_star_layer,
                **({"jpeg_quality": jpeg_quality} if jpeg_quality is not None else {}))

    @tool()
    async def create_regions(reference_dir: str, input_files: list[str], scene_mode: str = "unknown",
                             operations: list[dict] | None = None, milkyway: dict | None = None,
                             user_confirmation: str | None = None) -> dict[str, Any]:
        """保存共享分区项目；建议优先用 draw_regions 绘制并预览。horizon/paint 为0..1左上坐标；银河ellipse=[cx,cy,rx,ry]中心可[-4,5]、半径[.001,8]，angle_degrees±180、feather[.1,.95]。显式绘制即可生效，始终记录agent来源，不需要再确认。显式status=unknown/absent不增强。返回project_path直接用于submit_workflow(regions_path)。"""
        return await asyncio.to_thread(service.create_regions, reference_dir, input_files,
                    scene_mode=scene_mode, operations=operations, milkyway=milkyway,
                    user_confirmation=user_confirmation)

    @tool()
    async def update_regions(regions_path: str, scene_mode: str | None = None,
                             operations: list[dict] | None = None, milkyway: dict | None = None,
                             user_confirmation: str | None = None) -> dict[str, Any]:
        """另存区域新修订；operations替换完整操作列表，旧版本保留可撤销。银河显式编辑生效并记录agent来源；ellipse=null清除，不需要重复确认。"""
        return await asyncio.to_thread(service.update_regions, regions_path, scene_mode=scene_mode,
                    operations=operations, milkyway=milkyway, user_confirmation=user_confirmation)

    @tool()
    async def draw_regions(reference_dir: str, input_files: list[str] | None = None,
                           regions_path: str | None = None, horizon: list[list[float]] | None = None,
                           ground_strokes: list[list[list[float]]] | None = None,
                           sky_strokes: list[list[list[float]]] | None = None,
                           brush_radius: float = .018, milkyway_ellipse: list[float] | None = None,
                           milkyway_angle: float | None = None, milkyway_feather: float | None = None,
                           clear_ground: bool = False, clear_milkyway: bool = False) -> dict[str, Any]:
        """Draw foreground and Milky Way on the same calibrated reference used by the GUI.

        New: supply input_files and reference_dir from prepare_reference. Edit: supply regions_path.
        horizon is a left-to-right [[x,y],...] line (0..1, origin top-left); everything below is ground.
        ground_strokes/sky_strokes are lists of brush polylines; brush_radius is a fraction of the short image edge.
        milkyway_ellipse=[cx,cy,rx,ry] supports off-frame centers [-4,5] and radii [.001,8];
        milkyway_angle is degrees [-180,180], feather [.1,.95]. Explicit drawing enables its region;
        no drawing means no foreground/Milky Way, no second confirmation. All MCP edits record agent origin.
        clear_ground/clear_milkyway remove prior marks. Every edit saves a new revision: reuse an old
        regions_path to undo. Returns preview image path and project_path for submit_workflow;
        preview_regions(include_image=true) returns the overlay image to the client when authorized.
        """
        return await asyncio.to_thread(service.draw_regions, reference_dir, input_files=input_files,
            regions_path=regions_path, horizon=horizon, ground_strokes=ground_strokes,
            sky_strokes=sky_strokes, brush_radius=brush_radius, milkyway_ellipse=milkyway_ellipse,
            milkyway_angle=milkyway_angle, milkyway_feather=milkyway_feather,
            clear_ground=clear_ground, clear_milkyway=clear_milkyway)

    @tool()
    async def get_regions(regions_path: str) -> dict[str, Any]:
        """读取并核验已保存区域、原片身份和存在依据，与GUI共用项目。"""
        return await asyncio.to_thread(service.get_regions, regions_path)

    @tool()
    async def preview_regions(regions_path: str, reference_dir: str,
                              include_image: Annotated[bool, Field(strict=True)] = False
                              ) -> Annotated[CallToolResult, dict[str, Any]]:
        """检查分区叠色：橙色地景（实线边界）、粉紫双线银河。仅预览，不改变分区或启用未知旧选区。默认返回本机路径。include_image=true向当前MCP客户端返回最长1600像素的图片，调用须符合用户图片访问授权。"""
        receipt = await asyncio.to_thread(service.preview_regions, regions_path, reference_dir)
        if not include_image:
            return receipt
        import base64
        payload = Path(receipt["preview_path"]).read_bytes()
        if hashing.sha256_bytes(payload, source="mcp_server.preview_regions") != receipt["sha256"]:
            raise ValueError("分区叠色图在返回前已改变")
        # The image rides next to the receipt; the text block and structuredContent carry the same JSON object, and
        # the declared schema is the one of the path-only reply.
        receipt = json_native(receipt)
        return CallToolResult(content=[
            TextContent(type="text", text=json.dumps(receipt, ensure_ascii=False)),
            ImageContent(type="image", mimeType="image/png", data=base64.b64encode(payload).decode("ascii"))],
            structuredContent=receipt)

    @tool()
    async def open_trail_review(workflow_dir: str) -> dict[str, Any]:
        """打开当前完整任务的亮迹候选和草稿，含整图位置、建议身份/处置与来源；保留session_path、base_revision和draft_revision用于后续复核。未知不是卫星。

        application_policy_version=meteor-default-keep-1 时量测可靠的流星默认保留；每条候选含 identity/identity_source
        （human/agent/algorithm）、default_keep、effective_keep（含已存草稿，明确覆盖优先）与 keep_source；options.export_trail_layer；
        preview_path 优先当前含亮迹成片，base_preview_path 为不含亮迹的基础成片。"""
        return await asyncio.to_thread(service.open_trail_review, workflow_dir)

    @tool()
    async def trail_candidate_context(workflow_dir: str, session_path: str, candidate_id: str) -> dict[str, Any]:
        """查看该候选所在曝光与前后实际曝光的相同位置裁块；无邻帧时明确报告缺失。用于查连续轨迹，不以没有链接证明是流星。"""
        return await asyncio.to_thread(service.trail_candidate_context, workflow_dir, session_path, candidate_id)

    @tool()
    async def save_trail_review(workflow_dir: str, session_path: str, decisions: dict,
                                expected_draft_revision: str,
                                export_trail_layer: Annotated[bool, Field(strict=True)] | None = None) -> dict[str, Any]:
        """保存整批复核草稿、不重算：decisions={候选ID:{label,override,note}}，可只传某字段，label=meteor/satellite/aircraft/artifact/unknown/null，override=keep/omit/null。流星默认规则下改为流星即默认保留、改为其他身份即取消默认保留；omit总是省略；null撤销覆盖。替换整个草稿决定列表，记录来源agent而不冒充人工。export_trail_layer=true时应用后另存亮迹图层（RGBA16，叠在基础成片上精确重现含亮迹成片），null保持原选项。过期draft_revision拒绝覆盖。"""
        return await asyncio.to_thread(service.save_trail_review, workflow_dir, session_path,
                                       decisions, expected_draft_revision, export_trail_layer)

    @tool()
    async def apply_trail_review(workflow_dir: str, session_path: str, expected_draft_revision: str) -> dict[str, Any]:
        """一次应用已保存整批决定并另存完整成片（delivery-trails-<n>kept-*），继承该任务颜色、银河及缩星，原母版与基础成片保留。一条都不保留或像素无变化时不另存新文件夹（published=false，identical_to_base=true）。返回kept_count/kept_ids/published/delivery_dir/trail_layer_path。仅已校色RAW支持；失败可重试同草稿，不能保证未量测候选已完整恢复。较耗时。"""
        return await asyncio.to_thread(service.apply_trail_review, workflow_dir, session_path, expected_draft_revision)

    @tool()
    async def workflow_status(workflow_dir: str) -> dict[str, Any]:
        """查看完整工作流状态、阶段、错误与产物；堆栈结束不等于后期结束（trail_addback阶段仍在自动加回流星）。

        delivery_dir 为当前应显示的成片（自动加回流星时为含亮迹版本），base_delivery_dir 为不含亮迹的基础成片；
        trail_review 含 automatic_addback、default_policy、kept_count/kept_ids、kept_by{algorithm,human,agent}、pending_count；
        extras 列出另存的无星图、星点层与亮迹图层（role/path/sha256）。逐条身份与保留来源用 open_trail_review。"""
        return await asyncio.to_thread(service.workflow_status, workflow_dir)

    @tool()
    async def inspect_workflow_cache(workflow_dir: str) -> dict[str, Any]:
        """只读检查已完成任务的工作缓存：可释放的逐帧中间文件（分组、文件数、字节）、保留字节、状态与不能清理的原因；不删除任何文件。

        state 为 cleanable/partial/cleaned/nothing/running/refused；cache_cleaned=true 时 workflow_status 也会注明。
        清理只能由用户在桌面界面“设置 → 作品与缓存”确认后执行；清理后作品与原片保留，但不能再复核亮迹、恢复或重做亮迹合成。"""
        return await asyncio.to_thread(service.inspect_workflow_cache, workflow_dir)

    @tool()
    async def cancel_workflow(workflow_dir: str) -> dict[str, Any]:
        """请求停止整个工作流，包括尚在运行的堆栈；等待状态进入cancelled。"""
        return await asyncio.to_thread(service.cancel_workflow, workflow_dir)

    @tool()
    async def resume_workflow(workflow_dir: str) -> dict[str, Any]:
        """核对冻结输入/参数后恢复未完成工作流，保留失败证据与已完成产物。"""
        return await asyncio.to_thread(service.resume_workflow, workflow_dir)

    @tool()
    async def inspect_sources(input_dir: str | None = None, input_files: list[str] | None = None) -> dict[str, Any]:
        """检查允许范围内的 RAW/FITS 序列；input_files 明确选择并保留顺序，原片只读。"""
        return await asyncio.to_thread(service.inspect_sources, input_dir, input_files=input_files)

    @tool()
    async def inspect_raw(source_path: str) -> dict[str, Any]:
        """检查单张允许的 RAW/DNG，另存 CFA 和数值报告；不显影、不猜测逆转机内处理。"""
        return await asyncio.to_thread(service.inspect_raw, source_path)

    @tool()
    async def protect_trace(
            audit_dir: str, rectangle: list[Annotated[int, Field(strict=True, ge=0)]],
            classification: str = "unknown", note: str = "") -> dict[str, Any]:
        """保存人工亮迹证据；rectangle=[x,y,width,height]，零起点、原生CFA左上坐标。无自动检测或合成。"""
        return await asyncio.to_thread(service.protect_trace, audit_dir, rectangle, classification, note)

    @tool()
    async def submit_job(input_dir: str, sky_mask_path: str | None = None,
                         idempotency_key: str | None = None, support_mode: str = "legacy",
                         stack_range_mode: Literal["legacy", "positive_power2_v1", "finite_validity_v1"] = "legacy",
                         force_restart: Annotated[bool, Field(strict=True)] = False,
                         image_rotation: Annotated[int, Field(strict=True)] | None = None) -> dict[str, Any]:
        """提交持久后台任务并返回 job_dir。重试请复用幂等标识；无蒙版时仅堆栈天空。

        stack_range_mode 默认 legacy；positive_power2_v1 保留额外高光范围，单通道零值和负值照常堆栈（暗端与 legacy 相同在0处截断），遇非有限值停止并提示。
        force_restart 只配合 idempotency_key：同一编号的作业已结束（成功/失败/取消，含崩溃）时把旧目录改名保留并重跑；运行中或素材不同仍拒绝。
        image_rotation 为显式整组顺时针0/90/180/270度，默认保持旧原生方向；蒙版必须对应同方向。日常出图优先用 submit_workflow。
        """
        restart = {"force_restart": True} if force_restart else {}     # off: the call shape stays exactly as before
        if image_rotation is not None:
            restart['image_rotation'] = image_rotation
        return await asyncio.to_thread(service.submit_job, input_dir, sky_mask_path, idempotency_key, support_mode,
                                       stack_range_mode=stack_range_mode, **restart)

    @tool()
    async def prepare_reference(input_dir: str | None = None, input_files: list[str] | None = None,
                                color_adjustments: dict[str, Any] | None = None,
                                image_rotation: Annotated[int, Field(strict=True)] | None = None) -> dict[str, Any]:
        """为首帧生成颜色预览与全白模板；现代文件选择自动采用拍摄方向。可指定所有素材相对解码像素顺时针0/90/180/270度。模板并非自动分割结果。"""
        return await asyncio.to_thread(service.prepare_reference, input_dir,
                                       input_files=input_files, color_adjustments=color_adjustments,
                                       image_rotation=image_rotation)

    @tool()
    async def rotate_reference(reference_dir: str,
                               image_rotation: Annotated[int, Field(strict=True)]) -> dict[str, Any]:
        """将首帧方向设为相对解码像素顺时针0/90/180/270度；另存参考，不改原片。之后用新reference_dir重新绘制地景/银河，submit_workflow传相同image_rotation或让分区自动绑定。"""
        return await asyncio.to_thread(service.rotate_reference, reference_dir, image_rotation)

    @tool()
    async def render_job_preview(
            job_dir: str,
            red_gain: Annotated[float, Field(strict=True, ge=.25, le=4)] = 1.0,
            blue_gain: Annotated[float, Field(strict=True, ge=.25, le=4)] = 1.0,
            exposure_ev: Annotated[float, Field(strict=True, ge=-3, le=3)] = 0.0) -> dict[str, Any]:
        """生成独立新预览，不改原任务。有后期母版的作业：按成片显示变换出一张 JPG + 16 位 TIFF（增益偏移机内白平衡，曝光作用于中间调锚点）；旧作业：三张 JPG 与显示 TIFF。通道增益0.25–4，曝光−3–3 EV。"""
        return await asyncio.to_thread(service.render_job_preview, job_dir, red_gain, blue_gain, exposure_ev)

    @tool()
    async def coordinate_background(
            job_dir: str, samples: str | None = None, recipe: str | None = None,
            mode: str | None = None,
            strength: Annotated[float, Field(strict=True, ge=0, le=1)] | None = None,
            red_gain: Annotated[float, Field(strict=True, ge=.25, le=4)] | None = None,
            blue_gain: Annotated[float, Field(strict=True, ge=.25, le=4)] | None = None) -> dict[str, Any]:
        """背景样本PNG位于输出根内；另存堆栈背景派生，或给recipe回放。非相机校色或哈勃色。"""
        return await asyncio.to_thread(service.coordinate_background, job_dir, samples, recipe,
                                       mode, strength, red_gain, blue_gain)

    @tool()
    async def job_status(job_dir: str) -> dict[str, Any]:
        """读取同一持久任务的真实状态、0–100 进度和已登记产物。"""
        return await asyncio.to_thread(service.job_status, job_dir)

    @tool()
    async def finish_photo(source: str, sky_mask: str, protected_mask: str,
            mode: Literal["photographic", "natural"] = "photographic",
            detail_strength: Annotated[float, Field(strict=True, ge=0, le=2)] = 1.,
            star_strength: Annotated[float, Field(strict=True, ge=0, le=2)] = 1.,
            ground_strength: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.,
            jpeg_quality: Annotated[int, Field(strict=True, ge=60, le=100)] | None = None) -> dict[str, Any]:
        """实验全天空／星点／地景调整，摄影审美未通过。需输出根内16位sRGB成片与同尺寸天空／保护蒙版；无生成式AI，不接受线性母版。jpeg_quality（60–100整数）为另存JPEG的质量（4:4:4），省略时为97。"""
        return await asyncio.to_thread(service.finish_photo, source, sky_mask, protected_mask,
            mode, detail_strength, star_strength, ground_strength,
            **({"jpeg_quality": jpeg_quality} if jpeg_quality is not None else {}))

    @tool()
    async def finish_milkyway(source: str, sky_mask: str, protected_mask: str, milkyway_mask: str,
            brightness: Annotated[float, Field(strict=True, ge=0, le=.06)] = .020,
            contrast: Annotated[float, Field(strict=True, ge=0, le=1.5)] = .60,
            structure: Annotated[float, Field(strict=True, ge=0, le=2)] = .90,
            color: Annotated[float, Field(strict=True, ge=0, le=1)] = .35,
            strength: Annotated[float, Field(strict=True, ge=0, le=1)] = .60,
            transient_background: str | None = None,
            starless_background: str | None = None,
            milkyway_evidence: str | None = None,
            jpeg_quality: Annotated[int, Field(strict=True, ge=60, le=100)] | None = None) -> dict[str, Any]:
        """需输出根内16位sRGB成片及同尺寸银河／天空／保护蒙版。银河蒙版不是存在证据；需milkyway_evidence给出绑定源/蒙版的人工复核JSON，缺省或无可见银河时原样跳过，不得自动伪造确认或回退旧全天空增强。可选transient_background先增强背景再保留亮迹差值；starless_background为已绑定的实验分层。仅展示，非物理恢复。jpeg_quality（60–100整数）为另存JPEG的质量（4:4:4），省略时为97。"""
        paired_options = ({"transient_background": transient_background}
                          if transient_background is not None else {})
        if starless_background is not None:
            paired_options["starless_background"] = starless_background
        if milkyway_evidence is not None:
            paired_options["milkyway_evidence"] = milkyway_evidence
        if jpeg_quality is not None:
            paired_options["jpeg_quality"] = jpeg_quality
        return await asyncio.to_thread(service.finish_milkyway, source, sky_mask, protected_mask, milkyway_mask,
            brightness, contrast, structure, color, strength, **paired_options)

    @tool()
    async def export_star_layers(separation_result: str) -> dict[str, Any]:
        """导出独立背景、恒星正负与亮迹正负RGB16 TIFF图层包，无银河增强或重新推理。分层记录及所有引用须在输出根内。不是单个PSD，编辑器读回未认证；保留精确signed残差。"""
        return await asyncio.to_thread(service.export_star_layers,separation_result)

    @tool()
    async def reduce_stars(separation_result: str,
            strength: Annotated[float, Field(strict=True, ge=0, le=1)] = .55,
            bright_start: Annotated[float, Field(strict=True, ge=0, le=1)] = .10,
            bright_end: Annotated[float, Field(strict=True, ge=0, le=1)] = .45,
            milkyway_mask: str | None = None, milkyway_evidence: str | None = None,
            milkyway_relief: Annotated[float, Field(strict=True, ge=0, le=1)] = .5,
            bright_radius: Annotated[int, Field(strict=True, ge=0, le=32)] | None = None,
            background_strength: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] | None = None,
            milkyway_offset: Annotated[float, Field(strict=True, ge=-1, le=1, allow_inf_nan=False)] = -.2,
            jpeg_quality: Annotated[int, Field(strict=True, ge=60, le=100)] | None = None) -> dict[str, Any]:
        """从已验证分层另存成片。jpeg_quality（60–100整数）为star_reduced.jpg的质量（4:4:4色度）；省略时保持旧编码（95、4:2:0）。background_strength 为新全范围背景力度：0不移星，1移除安全范围内已分离星层，高力度渐进放开亮星保护；未提供则沿用旧strength弱星压暗。milkyway_offset为银河相对背景的偏移−1至1，负值减弱，通过渐变蒙版平滑过渡。显示域调整不是测光星等或几何去卷积；独立亮迹、地景、安全边界及可表示余量仍受保护，1不保证后端真正无残星。分层及所有引用须在输出根内。银河需绑定源/蒙版的存在复核JSON，不得自动伪造人工确认。可先调用suggest_star_reduction获取建议，建议不自动执行。"""
        range_options = ({"background_strength": background_strength, "milkyway_offset": milkyway_offset}
                         if background_strength is not None else {})
        if jpeg_quality is not None:
            range_options["jpeg_quality"] = jpeg_quality
        return await asyncio.to_thread(service.reduce_stars, separation_result,
            strength=strength, bright_start=bright_start, bright_end=bright_end,
            milkyway_mask=milkyway_mask, milkyway_evidence=milkyway_evidence,
            milkyway_relief=milkyway_relief, bright_radius=bright_radius, **range_options)

    @tool()
    async def suggest_star_reduction(separation_result: str,
            milkyway_mask: str | None = None, milkyway_evidence: str | None = None) -> dict[str, Any]:
        """读取已验证分层中的星体密度，报告recommendation（available、background_strength、milkyway_offset、milkyway_strength、reason）与metrics，保存可追溯建议记录，不自动渲染或改变图像。用户或agent可调整建议后调用reduce_stars。分层及所有引用须在输出根内；银河差别建议需绑定源/蒙版的人工存在复核，不得伪造确认；缺依据不阻止普通背景建议。"""
        return await asyncio.to_thread(service.suggest_star_reduction, separation_result,
            milkyway_mask=milkyway_mask, milkyway_evidence=milkyway_evidence)

    @tool()
    async def cancel_job(job_dir: str) -> dict[str, Any]:
        """请求停止处理；结果 cancel_requested 不代表工作进程已退出，请继续查询。"""
        return await asyncio.to_thread(service.cancel_job, job_dir)

    @tool()
    async def resume_job(job_dir: str) -> dict[str, Any]:
        """恢复失败或已取消任务；由核心检查原片身份和已完成阶段完整性。"""
        return await asyncio.to_thread(service.resume_job, job_dir)

    @tool()
    async def list_transients(job_dir: str, needs_review: Annotated[bool, Field(strict=True)] = False) -> dict[str, Any]:
        """列出已完成任务的亮迹候选：来源帧、量测、建议标签与证据、人工标签、处置；基础成片不主动加回亮迹层，仍可能有堆栈残留，含亮迹成片另出。"""
        return await asyncio.to_thread(service.list_transients, job_dir, needs_review)

    @tool()
    async def set_transients(job_dir: str, labels: dict[str, str | None] | None = None, overrides: dict[str, str | None] | None = None,
                             policy: dict[str, str] | None = None, marks: list[dict[str, Any]] | None = None,
                             layer_style: Literal["source_residual", "photographic_v3"] | None = None) -> dict[str, Any]:
        """改人工标签（候选ID→meteor/satellite/aircraft/artifact/unknown/空）、逐条处置（→keep/omit/空）、策略（类别→keep/omit）或加人工标记
        （{input, x0, y0, x1, y1, protect?: bool}，源帧原生坐标）。标记默认只搜索；protect:true 请求保护但不改类别，
        override omit 撤回加回。查看 protection_status/application 了解未兑现或部分结果；只重做亮迹层与含亮迹成片，运行中拒绝。"""
        return await asyncio.to_thread(service.set_transients, job_dir, labels, overrides, policy, marks, layer_style)

    @tool()
    async def export_master(source: str, reference_dng: str | None = None, recipe: str | None = None,
                            wb: Literal["daylight", "as_shot"] = "daylight",
                            gamut: Literal["rec2020", "prophoto", "srgb"] = "rec2020",
                            dcp_profile: str | None = None) -> dict[str, Any]:
        """从已完成作业（作业目录、products/<uuid> 或 master_linear.fit）另存后期母版 M：float32 场景参照线性、默认线性 Rec.2020、嵌 ICC、不裁切、记录黑白位/白平衡/矩阵、同源同配方逐位可重放。颜色底子取参考帧 DNG（素材根或输出根内）或已有配方 JSON。输出到结果根内新目录 master-<uuid>。"""
        return await asyncio.to_thread(service.export_master, source, reference_dng, recipe, wb, gamut, dcp_profile)

    @tool()
    async def render_photo(master: str, sky_mask: str | None = None,
                           wb: Literal["as_shot", "daylight", "master"] = "as_shot",
                           contrast: Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)] = 12.0,
                           saturation: Annotated[float, Field(strict=True, ge=0, le=2, allow_inf_nan=False)] = 1.0,
                           target: Annotated[float, Field(strict=True, ge=.01, le=.95, allow_inf_nan=False)] = .22,
                           candidates: Annotated[bool, Field(strict=True)] = False,
                           shadow_smoothing: Annotated[bool, Field(strict=True)] = True,
                           look: Literal["auto", "none", "dcp_hsm"] = "auto", dcp_profile: str | None = None) -> dict[str, Any]:
        """从 M 后期母版用唯一的保色相显示变换另存成片 D：16 位 sRGB TIFF 与同源 JPEG（默认配方 photo-default-1：机内白平衡、k=12、饱和度 1、阴影色度平滑；观感 auto = 该机型登记的默认观感，与作业成片一致）。candidates=true 另存缩小候选集供挑选。母版须在结果根内；天空蒙版可在素材根或结果根。不改母版，不改作业。"""
        return await asyncio.to_thread(service.render_photo, master, sky_mask, wb, contrast, saturation, target, candidates,
                                       shadow_smoothing, look, dcp_profile)

    @tool()
    async def preview(job_dir: str, artifact_index: int = 0) -> dict[str, Any]:
        """返回第 N 个已登记 JPG 的本机路径和尺寸；序号从 0 开始，不传图片内容。"""
        return await asyncio.to_thread(service.preview, job_dir, artifact_index)

    return register_finishing_tools(server, finishing)


def main() -> int:
    # No print: stdout belongs exclusively to the SDK's stdio protocol.
    create_server().run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
