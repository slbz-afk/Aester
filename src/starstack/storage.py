"""Separate durable photographs from disposable processing files.

Only the optional delivery_root route uses this module. All publication paths
are derived from the frozen workflow identity, never from mutable UI fields.
Cleanup is deliberately a read-only plan; no function here removes a cache.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import stat
import uuid

from . import hashing
from .records import atomic_json

VERSION = "aester-delivery-1"
IMAGE_VERSION = "aester-delivery-2"
# Optional images a v2 delivery may declare in addition to its three base
# images (file name -> role). Anything else in a v2 folder is rejected.
EXTRA_IMAGES = {"starless_srgb16.tif": "starless", "stars_add_srgb16.tif": "stars_add",
                "stars_subtract_srgb16.tif": "stars_subtract", "trails_layer_rgba16.tif": "trails_layer"}
BASE_IMAGE_ROLES = {"master_scene_linear.tif": "master", "master_input_linear.tif": "master",
                    "final_srgb16.tif": "final_tiff", "final.jpg": "final_jpeg"}
DELIVERY_ROLES = frozenset({*BASE_IMAGE_ROLES.values(), *EXTRA_IMAGES.values(), "sidecar"})


def delivery_role(name):
    """The role of a delivered file name; JSON/PNG/text sidecars are 'sidecar'."""
    return BASE_IMAGE_ROLES.get(name) or EXTRA_IMAGES.get(name) or "sidecar"


def default_cache_root():
    """Per-user, non-roaming work area; does not create it."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "Aester" / "Cache"


def _no_links(path):
    """Reject symlinks/junctions, including redirected ancestors before writes."""
    path = Path(os.path.abspath(Path(path).expanduser()))
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        # The name-surrogate bit covers junctions and symlinks. Cloud-provider
        # placeholders (e.g. hydrated OneDrive files) are also reparse points,
        # but do not redirect the path and must not be rejected as links.
        redirected = bool(getattr(info, "st_reparse_tag", 0) & 0x20000000)
        if stat.S_ISLNK(info.st_mode) or redirected:
            raise ValueError("作品／缓存路径不能经过软链接或目录联接：" + str(part))
    return path


def _overlap(a, b):
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def admit_delivery_root(delivery_root, cache_root, input_files):
    root = _no_links(delivery_root).resolve()
    cache = _no_links(cache_root).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("作品目录不是文件夹")
    if _overlap(root, cache):
        raise ValueError("作品目录与工作缓存必须互不嵌套")
    for source in input_files:
        parent = Path(source).resolve(strict=True).parent
        if root == parent or root.is_relative_to(parent):
            raise ValueError("作品目录必须位于原片文件夹之外")
    return str(root)


def publication_parent(record, *, create=False):
    directory = Path(record["workflow_dir"]).resolve(strict=True)
    requested = record["request"].get("delivery_root")
    if requested is None:
        return directory
    root = Path(admit_delivery_root(requested, directory.parent, record["request"]["input_files"]))
    # Same idempotency key in different cache roots must never share ownership.
    identity = hashlib.sha256((str(directory) + "\0" + record["signature"]).encode()).hexdigest()[:20]
    parent = root / ("aester-" + identity)
    _no_links(parent)
    if create:
        parent.mkdir(parents=True, exist_ok=True)
        _no_links(parent)
    return parent


def delivery_file(record, path):
    """Admit only this workflow's delivery children, not the whole Pictures root."""
    parent = publication_parent(record)
    candidate = _no_links(path).resolve(strict=True)
    if not candidate.is_relative_to(parent):
        raise ValueError("作品文件不属于本工作流")
    relative = candidate.relative_to(parent)
    if len(relative.parts) < 2 or not relative.parts[0].startswith("delivery-"):
        raise ValueError("作品文件必须位于已发布的版本目录")
    if not candidate.is_file():
        raise ValueError("作品不是普通文件")
    return candidate


def _copy_verified(source, destination, check):
    """Exclusive, fsynced copy of a frozen file with cancellation before every 8 MiB chunk.

    In a hash run scope (not "off"): the bytes copied are hashed during the
    copy and must equal the frozen digest, and the destination is read back
    for real; the source needs no separate re-verification after the copy.
    Otherwise the former copy: read the copy back, then verify the source again.
    """
    from . import workflow as wf
    source = Path(source).resolve(strict=True)
    frozen = wf._file(source)
    if not hashing.digest_writes():
        with hashing.writing(destination):
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                while True:
                    check()
                    chunk = incoming.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    outgoing.write(chunk)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        check()
        copied = wf._file(destination)
        if (copied["sha256"], copied["bytes"]) != (frozen["sha256"], frozen["bytes"]):
            raise ValueError("作品复制校验失败")
        wf._verify_files([frozen])
        return frozen
    copied = hashing.copy_with_digest(source, destination, source="storage._copy_verified",
                                      expected=frozen["sha256"], check=check, exclusive=True, fsync=True)
    if (copied.dst_sha256, Path(destination).stat().st_size) != (frozen["sha256"], frozen["bytes"]):
        raise ValueError("作品复制校验失败")
    return frozen


def deliver_image(source, destination, *, role, check, display_transform=None, portable_master=False):
    """The single place that writes a file into the works folder.

    ``role`` names what the file is (master, final_tiff, final_jpeg, starless,
    stars_add, stars_subtract, trails_layer, sidecar) so that a size-capped
    edition can later limit delivered images here, in one function.
    """
    if role not in DELIVERY_ROLES:
        raise ValueError("未知作品图像角色：" + str(role))
    if portable_master:
        from . import master_export as me
        return me.package_portable_master(source, destination, display_transform=display_transform, check=check)
    return _copy_verified(source, destination, check)


def stage_delivery(record, sources, target_name, *, check=lambda: None):
    """Copy a small deliverable set beside its final target, without publishing.

    sources maps plain filenames to {path,kind,label}. Pixel files are copied
    unchanged. JSON sidecars receive relative references for copied assets;
    originals remain as provenance sidecars with their original hashes.
    """
    from . import workflow as wf
    if record["request"].get("delivery_format") == IMAGE_VERSION:
        return _stage_images(record, sources, target_name, check=check)
    if not record["request"].get("delivery_root"):
        raise ValueError("分离式发布需要 delivery_root")
    if Path(target_name).name != target_name or not target_name.startswith("delivery-"):
        raise ValueError("无效的作品版本名称")
    parent = publication_parent(record, create=True)
    target = parent / target_name
    if target.exists():
        raise FileExistsError(target)
    stage = parent / (".partial-" + target_name + "-" + uuid.uuid4().hex[:8])
    stage.mkdir()
    copies = []
    entries = dict(sources)
    try:
        mapping = {}
        for name, entry in sources.items():
            if Path(name).name != name or name in {".", "..", "delivery.json", "README.txt"}:
                raise ValueError("作品文件名必须是普通文件名")
            check()
            frozen = deliver_image(entry["path"], stage / name, role=delivery_role(name), check=check)
            mapping[frozen["path"]] = name
            copies.append({"file": name, "source": frozen})
        # Master TIFF embeds its full colour basis and a *relative* saturation
        # mask name. Its pixels/metadata remain byte-identical. Historical RAW
        # and FITS references are provenance, not runtime dependencies.
        def portable(value):
            if isinstance(value, str):
                return mapping.get(value, value)
            if isinstance(value, list):
                return [portable(item) for item in value]
            if isinstance(value, dict):
                return {key: portable(item) for key, item in value.items()}
            return value
        for name in ("master_recipe.json", "photo_render.json"):
            path = stage / name
            if path.exists():
                original = name.replace(".json", ".original.json")
                original_path = stage / original
                path.rename(original_path)
                data = json.loads(original_path.read_text(encoding="utf-8"))
                if name == "master_recipe.json":
                    master_name = next((n for n in ("master_scene_linear.tif", "master_input_linear.tif") if n in sources), None)
                    if master_name and data.get("master"):
                        if data.get("master_sha256") != wf._file(stage / master_name)["sha256"]:
                            raise ValueError("母版配方与作品母版不一致")
                        mapping[data["master"]] = master_name
                    if "camera_saturation_mask.png" in sources and data.get("mask"):
                        if data.get("mask_sha256") != wf._file(stage / "camera_saturation_mask.png")["sha256"]:
                            raise ValueError("母版配方与饱和蒙版不一致")
                        mapping[data["mask"]] = "camera_saturation_mask.png"
                with hashing.writing(path):
                    atomic_json(path, portable(data))
                entries[original] = {"kind": "json", "label": "原始处理记录（溯源）"}
        (stage / "README.txt").write_text(
            "Aester 作品\n\nfinal.jpg：可直接分享。\nfinal_srgb16.tif：带 sRGB 色彩描述的 16 位成片，可继续后期。\n"
            "master_scene_linear.tif：带线性色彩描述的浮点母版，保留高光和负值；普通查看器中较暗属于正常现象。\n"
            "请整体保留本文件夹，尤其母版旁的 camera_saturation_mask.png 和配方。配方中的作品路径相对此文件夹。\n"
            "*.original.json 及 TIFF 内的原 RAW/FITS 路径仅用于溯源；作品查看与母版显色不依赖这些原路径。\n"
            "删除工作缓存后，作品仍可查看和外部编辑；Aester 的亮迹复核、修改分区及工作流恢复需要保留缓存和原片。\n"
            "此目录不是 RAW 备份。\n", encoding="utf-8")
        entries["README.txt"] = {"kind": "text", "label": "作品使用说明"}
        records = wf._files([stage / name for name in entries], source="storage.stage_delivery")
        artifacts = [{**copied, "path": str(target / name), "name": name,
                      "kind": item["kind"], "label": item["label"]}
                     for copied, (name, item) in zip(records, entries.items())]
        receipt = {"version": VERSION, "workflow_signature": record["signature"],
                   "workflow_dir": record["workflow_dir"], "files": [
                       {"file": item["name"], "sha256": item["sha256"], "bytes": item["bytes"]} for item in artifacts],
                   "source_copies": copies, "paths_relative_to": "this delivery directory",
                   "cache_required_for_workflow_edits": True, "originals_included": False}
        atomic_json(stage / "delivery.json", receipt)
        artifacts.append({**wf._file(stage / "delivery.json"), "path": str(target / "delivery.json"),
                          "name": "delivery.json", "kind": "json", "label": "作品完整性记录"})
        check()
        _no_links(stage)
        _no_links(target)
        for item in copies:
            copied_name = item["file"]
            if copied_name in {"master_recipe.json", "photo_render.json"}:
                copied_name = copied_name.replace(".json", ".original.json")
            wf._verify_files([item["source"], {**item["source"], "path": str(stage / copied_name)}])
        return {"stage": stage, "target": target, "artifacts": artifacts}
    except BaseException as exc:
        atomic_json(stage / "failure.json", {"state": "unpublished", "error": str(exc)})
        raise


def verify_staged(package):
    """Last check immediately before a same-volume atomic directory rename."""
    from . import workflow as wf
    stage, target = _no_links(package["stage"]), _no_links(package["target"])
    if stage.parent != target.parent or target.exists():
        raise ValueError("待发布作品的位置已改变")
    wf._verify_files([{**item, "path": str(stage / item["name"])} for item in package["artifacts"]])
    if package.get("receipt"):
        wf._verify_files([package["receipt"]])
        receipt = json.loads(Path(package["receipt"]["path"]).read_text(encoding="utf-8"))
        verify_delivery(stage, receipt=receipt)


def verify_delivery(directory, *, receipt=None, hashes=True):
    """Verify a movable delivered folder without needing cache or original RAW.

    ``hashes=False`` (a cheap listing check, e.g. the 作品与缓存 job list): the same record and
    admission checks, then each file only by its recorded size, nothing read. Anything that acts on
    the result (deletes cache files) verifies with ``hashes=True``.
    """
    from . import workflow as wf
    root = _no_links(directory).resolve(strict=True)
    if receipt is None:
        receipt = json.loads((root / "delivery.json").read_text(encoding="utf-8"))
    if receipt.get("version") not in (VERSION, IMAGE_VERSION):
        raise ValueError("未知作品记录")
    names = [item["file"] for item in receipt["files"]]
    if len(set(names)) != len(names):
        raise ValueError("作品记录含重复文件")
    if receipt["version"] == IMAGE_VERSION:
        expected = {"final.jpg", "final_srgb16.tif", receipt.get("master_file")}
        extras = receipt.get("extras", [])
        if (not isinstance(extras, list) or any(not isinstance(item, dict) or set(item) != {"file", "role"}
                                                or EXTRA_IMAGES.get(item.get("file")) != item.get("role") for item in extras)
                or len({item["file"] for item in extras}) != len(extras)):
            raise ValueError("图像作品记录的附加图像不在允许列表内")
        if receipt.get("master_file") not in {"master_scene_linear.tif", "master_input_linear.tif"} \
                or set(names) != expected | {item["file"] for item in extras}:
            raise ValueError("图像作品记录必须恰好声明三份图像")
    # Formerly one file at a time: admit it (plain name, no links, inside the
    # folder), then wf._verify_files([{**item, "path": admitted path}]).  Every
    # file is admitted first and the admitted ones are verified together (in
    # parallel); an admission failure, or a file that goes away while earlier
    # ones are hashed, is raised where the one-file-at-a-time loop raised it.
    admitted = {}

    def admit(item):
        name = item["file"]
        if Path(name).name != name:
            raise ValueError("作品记录包含越界路径")
        path = _no_links(root / name).resolve(strict=True)
        if path.parent != root:
            raise ValueError("作品记录包含越界文件")
        admitted[id(item)] = path
        return path

    if not hashes:
        for item in receipt["files"]:
            path = admit(item)
            if not (path.is_file() and path.stat().st_size == item["bytes"]):
                raise wf._changed({"path": str(path)})
        return receipt
    hashing.verify_entries(receipt["files"], source="workflow._verify_files", path=admit,
                           error=lambda item: wf._changed({"path": str(admitted[id(item)])}))
    return receipt


def _stage_images(record, sources, target_name, *, check):
    """Publishable images only; all receipts and provenance stay in cache."""
    from . import workflow as wf, master_export as me
    if not record["request"].get("delivery_root") or Path(target_name).name != target_name or not target_name.startswith("delivery-"):
        raise ValueError("无效的图像作品发布位置")
    parent = publication_parent(record, create=True)
    target = parent / target_name
    if target.exists():
        raise FileExistsError(target)
    stage = parent / (".partial-" + target_name + "-" + uuid.uuid4().hex[:8])
    stage.mkdir()
    cache = _no_links(Path(record["workflow_dir"]) / "publication-receipts")
    cache.mkdir(exist_ok=True)
    receipt_path = cache / (target_name + ".json")
    master = next((n for n in ("master_scene_linear.tif", "master_input_linear.tif") if n in sources), None)
    if master is None or not {"final.jpg", "final_srgb16.tif"}.issubset(sources):
        raise ValueError("图像交付缺少母版或成片")
    frozen = dict(zip(sources, wf._files(sources, prepare=lambda name: sources[name]["path"],
                                         source="storage._stage_images")))
    extras = [name for name in EXTRA_IMAGES if name in sources]
    selected = [master, "final_srgb16.tif", "final.jpg", *extras]
    try:
        packaged = None
        display = None
        if "photo_render.json" in sources:
            display = json.loads(Path(sources["photo_render.json"]["path"]).read_text(encoding="utf-8")).get("metadata", {}).get("display_transform")
        elif "review_delivery.json" in sources:
            display = json.loads(Path(sources["review_delivery.json"]["path"]).read_text(encoding="utf-8")).get("display_transform")
        for name in selected:
            check()
            if name == "master_scene_linear.tif":
                packaged = deliver_image(sources[name]["path"], stage/name, role="master", check=check,
                                         display_transform=display, portable_master=True)
            else:
                deliver_image(sources[name]["path"], stage/name, role=delivery_role(name), check=check)
                wf._verify_files([{**frozen[name], "path": str(stage/name)}])
        wf._verify_files(list(frozen.values()))
        records = wf._files([stage/name for name in selected], source="storage._stage_images")
        artifacts = [{**copied, "path": str(target/name), "name": name,
                      "kind": sources[name]["kind"], "label": sources[name]["label"]}
                     for copied, name in zip(records, selected)]
        receipt = {"version": IMAGE_VERSION, "workflow_signature": record["signature"],
                   "workflow_dir": record["workflow_dir"], "delivery_dir": str(target), "master_file": master,
                   "files": [{"file": a["name"], "sha256": a["sha256"], "bytes": a["bytes"]} for a in artifacts],
                   "source_copies": [{"file": n, "source": f} for n, f in frozen.items()],
                   "portable_master": packaged, "cache_required_for_workflow_edits": True, "originals_included": False}
        if extras:
            # Old receipts without this key still require exactly three images.
            receipt["extras"] = [{"file": name, "role": EXTRA_IMAGES[name]} for name in extras]
        atomic_json(receipt_path, receipt)
        verify_delivery(stage, receipt=receipt)
        check()
        return {"stage": stage, "target": target, "artifacts": artifacts, "receipt": wf._file(receipt_path)}
    except BaseException as exc:
        atomic_json(cache / (target_name + ".failure.json"), {"state": "unpublished", "error": str(exc), "stage": str(stage)})
        raise


def delivery_receipt(record, directory):
    """Load a v2 manifest only from its frozen cache descriptor, never guess."""
    from . import workflow as wf
    directory = _no_links(directory).resolve(strict=True)
    if record["request"].get("delivery_format") != IMAGE_VERSION:
        return None
    descriptor = record.get("publication_receipts", {}).get(str(directory))
    if not isinstance(descriptor, dict):
        raise ValueError("图像作品缺少工作缓存中的完整性记录")
    cache = _no_links(Path(record["workflow_dir"]) / "publication-receipts").resolve(strict=True)
    path = _no_links(descriptor["path"]).resolve(strict=True)
    if path.parent != cache:
        raise ValueError("作品完整性记录不在工作流缓存内")
    wf._verify_files([descriptor])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("version") != IMAGE_VERSION or receipt.get("workflow_signature") != record["signature"] \
            or receipt.get("workflow_dir") != record["workflow_dir"] or receipt.get("delivery_dir") != str(directory):
        raise ValueError("作品完整性记录与工作流绑定不符")
    return receipt


def plan_cache_cleanup(workflow_dir, *, cache_root):
    """Read-only deletion proposal for one terminal, separately delivered job.

    The caller must communicate that workflow edits become unavailable. This
    API does not perform deletion and does not authorise deletion of an entire
    user-selected cache root, sibling job, photograph, or original exposure.
    """
    from . import workflow as wf
    from .locking import is_locked
    directory = _no_links(workflow_dir).resolve(strict=True)
    root = _no_links(cache_root).resolve(strict=True)
    if directory == root or not directory.is_relative_to(root) or not directory.name.startswith("workflow-"):
        raise ValueError("只能清理指定缓存根内的单个工作流")
    record = wf._read(directory)
    if record["state"] != "succeeded" or not record["request"].get("delivery_root"):
        raise ValueError("只为已完成且作品独立保存的工作流生成清理计划")
    if any(is_locked(directory / name) for name in ("worker.lock", "edit.lock", "metadata.lock")):
        raise ValueError("工作流正在使用，不能清理")
    if record.get("stack_job_dir") and is_locked(Path(record["stack_job_dir"]) / "worker.lock"):
        raise ValueError("堆栈正在使用，不能清理")
    wf._verify(record)
    preserved = {record["delivery_dir"], record.get("base_delivery_dir", record["delivery_dir"])}
    preserved.update(item["delivery_dir"] for item in record.get("review_deliveries", []))
    for path in preserved:
        actual = _no_links(path).resolve(strict=True)
        if _overlap(actual, directory):
            raise ValueError("作品仍位于缓存内")
        verify_delivery(actual, receipt=delivery_receipt(record, actual))
    for path in record["request"]["input_files"]:
        if Path(path).resolve(strict=True).is_relative_to(directory):
            raise ValueError("缓存内包含原片，不能清理")
    size = 0
    count = 0
    for path in directory.rglob("*"):
        _no_links(path)
        if path.is_file():
            size += path.stat().st_size
            count += 1
    return {"version": "aester-cache-cleanup-plan-1", "delete_root": str(directory),
            "cache_root": str(root), "bytes": size, "files": count, "executed": False,
            "preserved_deliveries": sorted(preserved), "preserved_originals": record["request"]["input_files"],
            "loses_workflow_resume_and_review": True, "requires_explicit_confirmation": True}
