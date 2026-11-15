"""raw_import.import_sources_for: original file names from the import records in the cache, read only."""
import builtins
import io
import json
import os
from pathlib import Path

import pytest

from starstack import raw_import as ri
from test_raw_import import converter, sources  # noqa: F401 - fixtures

KEY = "0123456789abcdef01234567"


def write_record(cache, key=KEY, count=3, *, shots=None, change=None):
    """A synthetic import record in <cache>/raw-imports/<key>/import.json, shaped as prepare_import writes it.

    No DNG or original file is created: the lookup never opens them. Returns (folder, record, input paths).
    """
    folder = Path(cache) / "raw-imports" / key
    folder.mkdir(parents=True, exist_ok=True)
    shots = Path(shots or Path(cache).parent / "shots")
    sources = [{"path": str(shots / f"DSC_{n:04d}.NEF"), "sha256": f"{n:064x}", "bytes": 1,
                "visible_shape": [4, 6]} for n in range(1, count + 1)]
    outputs = [{"relative_path": os.path.join(f"{n:04d}", f"DSC_{n:04d}.dng"), "sha256": f"d{n:063x}", "bytes": 1}
               for n in range(1, count + 1)]
    record = {"schema_version": 1, "state": "ready", "converted": True, "sources": sources,
              "originals": [s["path"] for s in sources], "converter": str(shots / "converter.exe"),
              "converter_sha256": "e" * 64, "recipe": "cfa-lossless-dng14-v1", "outputs": outputs,
              "input_files": [str(folder / o["relative_path"]) for o in outputs],
              "reason": "unsupported_native_compression", "unsupported_sources": [s["path"] for s in sources],
              "cache_hit": False, "caution": "A decoded DNG is a compatibility copy"}
    if change:
        change(record)
    (folder / "import.json").write_text(json.dumps(record), encoding="utf-8")
    return folder, record, list(record.get("input_files") or [])


def expected(record, inputs):
    return {os.path.normcase(os.path.abspath(path)): {"file": Path(source["path"]).name, "sha256": source["sha256"],
                                                      "input_sha256": output["sha256"]}
            for path, source, output in zip(inputs, record["sources"], record["outputs"])}


def tree(root):
    return sorted((str(p.relative_to(root)), p.is_dir(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in Path(root).rglob("*"))


@pytest.fixture
def opened(monkeypatch):
    """Every path opened through open()/Path.open while the lookup runs."""
    paths = []
    real = io.open

    def recording(file, *args, **kwargs):
        paths.append(os.path.normcase(os.path.abspath(os.fspath(file))) if not isinstance(file, int) else file)
        return real(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", recording)
    monkeypatch.setattr(builtins, "open", recording)
    monkeypatch.setattr(ri, "_sha", lambda path: pytest.fail(f"hashed {path}"))
    return paths


def test_valid_record_names_each_original_reading_only_the_record(tmp_path, opened):
    cache = tmp_path / "cache"
    folder, record, inputs = write_record(cache)
    before = tree(tmp_path)
    opened.clear()
    found = ri.import_sources_for(inputs + [str(tmp_path / "elsewhere" / "frame.dng")], str(cache))
    assert found == expected(record, inputs)
    assert opened == [os.path.normcase(str(folder / "import.json"))]
    assert tree(tmp_path) == before                         # nothing written, no lock or folder


def test_inputs_are_matched_by_path_and_a_subset_returns_a_subset(tmp_path):
    cache = tmp_path / "cache"
    _, record, inputs = write_record(cache)
    first = inputs[1].upper() if os.name == "nt" else inputs[1]        # Windows paths ignore case
    found = ri.import_sources_for([first, Path(inputs[2])], str(cache).upper() if os.name == "nt" else cache)
    assert list(found.values()) == list(expected(record, inputs).values())[1:]


def test_round_trip_with_prepare_import_and_originals_moved_away(sources, converter, tmp_path,  # noqa: F811
                                                                 request):
    cache = tmp_path / "cache"
    result = ri.prepare_import(list(map(str, sources)), cache, converter_path=converter[0])
    for path in sources:
        path.unlink()                                       # moved or deleted since: names come from the record
    opened = request.getfixturevalue("opened")              # from here on: no hashing, opens recorded
    before = tree(tmp_path)
    found = ri.import_sources_for(result["input_files"], cache)
    assert found == {os.path.normcase(os.path.abspath(path)): {"file": Path(source["path"]).name,
                                                              "sha256": source["sha256"],
                                                              "input_sha256": output["sha256"]}
                     for path, source, output in zip(result["input_files"], result["sources"], result["outputs"])}
    assert len(found) == 2 and opened == [os.path.normcase(result["manifest_path"])] and tree(tmp_path) == before


def test_one_record_read_per_key_folder(tmp_path, opened):
    cache = tmp_path / "cache"
    first, record_a, inputs_a = write_record(cache, count=5)
    second, record_b, inputs_b = write_record(cache, key="fedcba9876543210fedcba98", count=2,
                                              shots=tmp_path / "other shots")
    opened.clear()
    again = [path.upper() for path in inputs_a] if os.name == "nt" else inputs_a     # same folder, other spelling
    found = ri.import_sources_for(inputs_a + inputs_b + again, cache)
    assert found == {**expected(record_a, inputs_a), **expected(record_b, inputs_b)}
    assert sorted(opened) == sorted(os.path.normcase(str(folder / "import.json")) for folder in (first, second))


def test_missing_cache_and_inputs_outside_it_give_nothing_and_create_nothing(tmp_path):
    cache = tmp_path / "cache"
    assert ri.import_sources_for([str(cache / "raw-imports" / KEY / "0001" / "DSC_0001.dng")], cache) == {}
    assert not cache.exists()
    _, _, inputs = write_record(tmp_path / "other cache")
    assert ri.import_sources_for(inputs, cache) == {}                     # another cache's record
    assert ri.import_sources_for([], cache) == {}
    assert ri.import_sources_for([None, 3, "", str(cache / "raw-imports" / "lonely.dng")], cache) == {}


def test_record_must_sit_directly_in_its_key_folder(tmp_path):
    cache = tmp_path / "cache"
    folder, record, inputs = write_record(cache)
    assert len(ri.import_sources_for(inputs, cache)) == 3
    (folder / "0001").mkdir()
    deeper = folder / "0001" / "import.json"
    (folder / "import.json").replace(deeper)
    assert ri.import_sources_for(inputs, cache) == {}
    deeper.replace(cache / "raw-imports" / "import.json")
    assert ri.import_sources_for(inputs, cache) == {}


def _link_folder(link, target):
    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        else:
            os.symlink(target, link, target_is_directory=True)
    except (OSError, AttributeError, NotImplementedError) as error:
        pytest.skip(f"cannot create a directory link here: {error}")


def test_links_or_junctions_on_the_record_path_are_skipped(tmp_path):
    cache = tmp_path / "cache"
    folder, record, inputs = write_record(cache)
    moved = tmp_path / "moved key folder"
    folder.replace(moved)
    _link_folder(folder, moved)                             # the key folder is a junction (or link)
    assert (folder / "import.json").is_file()
    assert ri.import_sources_for(inputs, cache) == {}
    real, record, inputs = write_record(tmp_path / "real cache", key="fedcba9876543210fedcba98")
    assert len(ri.import_sources_for(inputs, tmp_path / "real cache")) == 3
    linked = tmp_path / "linked cache"
    _link_folder(linked, tmp_path / "real cache")           # the cache folder itself is linked
    through = [str(linked / Path(path).relative_to(tmp_path / "real cache")) for path in inputs]
    assert ri.import_sources_for(through, linked) == {}


def test_a_linked_record_file_is_skipped(tmp_path):
    cache = tmp_path / "cache"
    folder, record, inputs = write_record(cache)
    target = tmp_path / "record.json"
    (folder / "import.json").replace(target)
    try:
        os.symlink(target, folder / "import.json")
    except OSError as error:
        pytest.skip(f"cannot create a file symbolic link here: {error}")
    assert ri.import_sources_for(inputs, cache) == {}


def test_a_linked_dng_folder_inside_the_record_folder_is_skipped(tmp_path):
    cache = tmp_path / "cache"
    folder, record, inputs = write_record(cache)
    (tmp_path / "elsewhere").mkdir()
    _link_folder(folder / "0002", tmp_path / "elsewhere")
    assert ri.import_sources_for(inputs, cache) == {}


def _broken(name):
    def digest_upper(r): r["sources"][0]["sha256"] = "A" * 64
    def digest_short(r): r["outputs"][1]["sha256"] = "d" * 63
    def schema(r): r["schema_version"] = 2
    def recipe(r): r["recipe"] = "rendered-rgb"
    def relative_source(r): r["sources"][0]["path"] = r["originals"][0] = "shots/DSC_0001.NEF"
    def not_raw(r): r["sources"][0]["path"] = r["originals"][0] = r["sources"][0]["path"][:-4] + ".jpg"
    def escape(r): r["outputs"][0]["relative_path"] = os.path.join("..", "x.dng")
    def absolute(r): r["outputs"][0]["relative_path"] = r["input_files"][0]
    def not_dng(r): r["outputs"][0]["relative_path"] = os.path.join("0001", "DSC_0001.tif")
    def mapping(r): r["input_files"][0], r["input_files"][1] = r["input_files"][1], r["input_files"][0]
    def originals(r): r["originals"][0] = r["originals"][1]
    def duplicate(r): r["outputs"][1] = dict(r["outputs"][0]); r["input_files"][1] = r["input_files"][0]
    def uneven(r): r["outputs"].pop()
    def empty(r): r.update(sources=[], outputs=[], input_files=[], originals=[])
    def too_many(r):
        for key in ("sources", "outputs", "input_files", "originals"):
            r[key] = r[key] * 101
    def not_list(r): r["outputs"] = {"0": r["outputs"][0]}
    def not_dict(r): r.clear(); r["schema_version"] = 1
    return locals()[name]


@pytest.mark.parametrize("name", ["digest_upper", "digest_short", "schema", "recipe", "relative_source", "not_raw",
                                  "escape", "absolute", "not_dng", "mapping", "originals", "duplicate", "uneven",
                                  "empty", "too_many", "not_list", "not_dict"])
def test_invalid_records_are_skipped_not_raised(tmp_path, name):
    cache = tmp_path / "cache"
    _, _, inputs = write_record(cache, change=_broken(name))
    assert ri.import_sources_for(inputs, cache) == {}


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe\x00", b"[1, 2]", b"[" * 100000 + b"]" * 100000],
                         ids=["not json", "not utf-8", "a list", "nested too deep"])
def test_unreadable_records_are_skipped(tmp_path, content):
    cache = tmp_path / "cache"
    folder, _, inputs = write_record(cache)
    (folder / "import.json").write_bytes(content)
    assert ri.import_sources_for(inputs, cache) == {}


def test_oversize_record_is_skipped_and_a_valid_one_beside_it_still_counts(tmp_path):
    cache = tmp_path / "cache"
    big, _, big_inputs = write_record(cache)
    text = (big / "import.json").read_text(encoding="utf-8")
    (big / "import.json").write_text(text[:-1] + " " * (2 * 1024 ** 2) + "}", encoding="utf-8")
    assert json.loads((big / "import.json").read_text(encoding="utf-8"))["schema_version"] == 1
    _, record, inputs = write_record(cache, key="fedcba9876543210fedcba98", shots=tmp_path / "other")
    assert ri.import_sources_for(big_inputs + inputs, cache) == expected(record, inputs)
    (big / "import.json").write_text(text[:-1] + " " * (2 * 1024 ** 2 - len(text.encode())) + "}", encoding="utf-8")
    assert (big / "import.json").stat().st_size == 2 * 1024 ** 2
    assert len(ri.import_sources_for(big_inputs, cache)) == 3          # exactly 2 MB is allowed


def test_a_folder_named_import_json_or_a_missing_record_is_skipped(tmp_path):
    cache = tmp_path / "cache"
    folder, _, inputs = write_record(cache)
    (folder / "import.json").unlink()
    assert ri.import_sources_for(inputs, cache) == {}
    (folder / "import.json").mkdir()
    assert ri.import_sources_for(inputs, cache) == {}


@pytest.mark.parametrize("arguments", [("a.dng", "cache"), (None, "cache"), ([], None), ([], ""), ([], 3)])
def test_wrong_argument_types_raise(arguments):
    with pytest.raises(ValueError):
        ri.import_sources_for(*arguments)
