"""Independent acceptance of the worker catalog contract; no fitter imports."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest

from starstack import holdout_catalog as catalog


def fixture():
    return {
        "version": "1.0.0", "width": 80, "height": 60,
        "reference": "reference", "identity_basis": "synthetic_truth",
        "stars": [{"id": "train_z", "role": "train"},
                  {"id": "holdout_z", "role": "holdout"},
                  {"id": "train_a", "role": "train"},
                  {"id": "holdout_a", "role": "holdout"}],
        "frames": [
            {"id": "target", "observations": [
                {"star_id": "holdout_a", "x": 0, "y": 59},
                {"star_id": "train_z", "x": 70, "y": 11.25}]},
            {"id": "reference", "observations": [
                {"star_id": "train_z", "x": 79, "y": 59},
                {"star_id": "holdout_z", "x": 13.5, "y": 17.25},
                {"star_id": "train_a", "x": 0, "y": 0},
                {"star_id": "holdout_a", "x": 42, "y": 31}]},
            {"id": "empty", "observations": []}],
    }


def assign(path, value):
    doc = fixture()
    owner = doc
    for key in path[:-1]:
        owner = owner[key]
    owner[path[-1]] = value
    return doc


class IntChild(int):
    pass


class FloatChild(float):
    pass


class StrChild(str):
    pass


class DictChild(dict):
    pass


class ListChild(list):
    pass


class CanonicalAcceptance(unittest.TestCase):
    def test_exact_encoding_oracle(self):
        values = [None, True, False, 0, -3, 1.25, -0.0, "中文\ud800",
                  [1, "雪", {"b": False, "a": None}],
                  {"星": [3.5, -2], "a": {"y": True, "x": None}}]
        for value in values:
            with self.subTest(value=value):
                expected = hashlib.sha256(json.dumps(
                    value, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
                self.assertEqual(catalog.canonical_sha256(value), expected)

    def test_dict_order_stable_but_list_order_matters(self):
        self.assertEqual(catalog.canonical_sha256({"b": 2, "a": 1}),
                         catalog.canonical_sha256({"a": 1, "b": 2}))
        self.assertNotEqual(catalog.canonical_sha256([1, 2]),
                            catalog.canonical_sha256([2, 1]))

    def test_rejects_non_json_and_non_builtin(self):
        bad = [float("nan"), float("inf"), -float("inf"), (1, 2), {1: "x"},
               {"x": b"bytes"}, {"x": set()}, object(), IntChild(1),
               FloatChild(1.0), StrChild("x"), DictChild(a=1), ListChild([1])]
        for index, value in enumerate(bad):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    catalog.canonical_sha256(value)

    def test_cycles_rejected_shared_containers_allowed(self):
        cyclic_list = []
        cyclic_list.append(cyclic_list)
        cyclic_dict = {}
        cyclic_dict["self"] = cyclic_dict
        for value in (cyclic_list, cyclic_dict):
            with self.assertRaises(ValueError):
                catalog.canonical_sha256(value)
        shared = [1, 2]
        self.assertEqual(catalog.canonical_sha256([shared, shared]),
                         catalog.canonical_sha256([[1, 2], [1, 2]]))

    def test_actual_encoder_limit_raises_valueerror(self):
        with self.assertRaises(ValueError):
            catalog.canonical_sha256(10 ** 5000)
        nested = []
        for _ in range(1500):
            nested = [nested]
        try:
            encoded = json.dumps(nested, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=True, allow_nan=False).encode("utf-8")
        except (ValueError, RecursionError):
            with self.assertRaises(ValueError):
                catalog.canonical_sha256(nested)
        else:
            self.assertEqual(catalog.canonical_sha256(nested), hashlib.sha256(encoded).hexdigest())


class CatalogAcceptance(unittest.TestCase):
    def test_normalization_and_no_mutation(self):
        source = fixture()
        original = copy.deepcopy(source)
        result = catalog.validate_catalog(source)
        self.assertEqual(catalog.CATALOG_VERSION, "1.0.0")
        self.assertEqual(source, original)
        self.assertEqual([s["id"] for s in result["stars"]],
                         ["holdout_a", "holdout_z", "train_a", "train_z"])
        self.assertEqual([f["id"] for f in result["frames"]],
                         ["empty", "reference", "target"])
        for frame in result["frames"]:
            self.assertEqual([o["star_id"] for o in frame["observations"]],
                             sorted(o["star_id"] for o in frame["observations"]))
            for obs in frame["observations"]:
                self.assertIs(type(obs["x"]), float)
                self.assertIs(type(obs["y"]), float)
        result["stars"][0]["id"] = "modified"
        result["frames"][1]["observations"][0]["x"] = -100
        result["frames"][0]["observations"].append({"x": "changed"})
        self.assertEqual(source, original)

    def test_normalized_order_independent(self):
        a = fixture()
        b = copy.deepcopy(a)
        b["stars"].reverse()
        b["frames"].reverse()
        for frame in b["frames"]:
            frame["observations"].reverse()
        self.assertEqual(catalog.validate_catalog(a), catalog.validate_catalog(b))

    def test_external_provenance_is_allowed_declaration(self):
        doc = fixture()
        doc["identity_basis"] = "external_verified"
        self.assertEqual(catalog.validate_catalog(doc)["identity_basis"], "external_verified")

    def test_required_fields_and_unknown_keys(self):
        for key in fixture():
            with self.subTest(missing=key):
                doc = fixture()
                del doc[key]
                with self.assertRaises(ValueError):
                    catalog.validate_catalog(doc)
        paths = [(), ("stars", 0), ("frames", 0), ("frames", 0, "observations", 0)]
        for path in paths:
            with self.subTest(extra_at=path):
                doc = fixture()
                owner = doc
                for key in path:
                    owner = owner[key]
                owner["unexpected"] = doc  # reject unknown key before traversing the cycle
                with self.assertRaises(ValueError):
                    catalog.validate_catalog(doc)
        for path in [("stars", 0), ("frames", 0), ("frames", 0, "observations", 0)]:
            obj = fixture()
            owner = obj
            for key in path:
                owner = owner[key]
            for name in list(owner):
                with self.subTest(path=path, missing=name):
                    doc = fixture()
                    node = doc
                    for key in path:
                        node = node[key]
                    del node[name]
                    with self.assertRaises(ValueError):
                        catalog.validate_catalog(doc)

    def test_wrong_shapes_and_values_always_valueerror(self):
        cases = [
            (("version",), "2.0.0"), (("version",), []),
            (("width",), True), (("height",), False),
            (("width",), 1), (("height",), -1), (("width",), 80.0),
            (("width",), IntChild(80)), (("height",), IntChild(60)),
            (("width",), 10 ** 5000),
            (("reference",), "missing"), (("reference",), " "),
            (("identity_basis",), "siril_cache"), (("identity_basis",), []),
            (("stars",), []), (("stars",), {}), (("stars", 0), None),
            (("stars", 0, "id"), "\t\n"), (("stars", 0, "id"), 0),
            (("stars", 0, "role"), "validation"), (("stars", 0, "role"), []),
            (("frames",), []), (("frames",), {}), (("frames", 0), []),
            (("frames", 0, "id"), " "), (("frames", 0, "id"), None),
            (("frames", 0, "observations"), {}),
            (("frames", 0, "observations", 0), []),
            (("frames", 0, "observations", 0, "star_id"), "unknown"),
            (("frames", 0, "observations", 0, "star_id"), []),
        ]
        for index, (path, value) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    catalog.validate_catalog(assign(path, value))
        for value in (None, [], True, 3, "catalog", (fixture(),)):
            with self.assertRaises(ValueError):
                catalog.validate_catalog(value)

    def test_invalid_numeric_observations(self):
        for axis in ("x", "y"):
            values = [True, False, None, "1", [], float("nan"), float("inf"),
                      -float("inf"), -0.0001, IntChild(1), FloatChild(1), 10 ** 5000]
            values += [79.00001 if axis == "x" else 59.00001]
            for index, value in enumerate(values):
                with self.subTest(axis=axis, case=index):
                    with self.assertRaises(ValueError):
                        catalog.validate_catalog(assign(("frames", 0, "observations", 0, axis), value))

    def test_duplicate_and_missing_identity_constraints(self):
        variants = []
        doc = fixture()
        doc["stars"][1]["id"] = doc["stars"][0]["id"]
        variants.append(doc)
        doc = fixture()
        doc["frames"][0]["id"] = "reference"
        variants.append(doc)
        doc = fixture()
        doc["frames"][0]["observations"].append(copy.deepcopy(doc["frames"][0]["observations"][0]))
        variants.append(doc)
        doc = fixture()
        doc["frames"][1]["observations"].pop()
        variants.append(doc)
        doc = fixture()
        doc["frames"] = [doc["frames"][1]]
        variants.append(doc)
        for role in ("train", "holdout"):
            doc = fixture()
            for star in doc["stars"]:
                star["role"] = role
            variants.append(doc)
        for index, doc in enumerate(variants):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    catalog.validate_catalog(doc)

    def test_aliases_do_not_escape_and_cycles_are_rejected(self):
        doc = fixture()
        obs = doc["frames"][0]["observations"][0]
        doc["frames"][2]["observations"].append(obs)
        result = catalog.validate_catalog(doc)
        a = next(f for f in result["frames"] if f["id"] == "target")["observations"][0]
        b = next(f for f in result["frames"] if f["id"] == "empty")["observations"][0]
        self.assertIsNot(a, b)
        doc = fixture()
        doc["frames"][0]["observations"] = doc["frames"]
        with self.assertRaises(ValueError):
            catalog.validate_catalog(doc)


class ViewSeparationAcceptance(unittest.TestCase):
    def test_exact_keys_and_empty_frames_preserved(self):
        for role, make_view in (("train", catalog.training_view), ("holdout", catalog.holdout_view)):
            view = make_view(fixture())
            self.assertEqual(set(view), {"version", "width", "height", "reference", "identity_basis", "role", "stars", "frames"})
            self.assertEqual(view["role"], role)
            self.assertEqual(view["stars"], [f"{role}_a", f"{role}_z"])
            self.assertEqual([f["id"] for f in view["frames"]], ["empty", "reference", "target"])
            self.assertEqual(view["frames"][0]["observations"], [])
            for frame in view["frames"]:
                for obs in frame["observations"]:
                    self.assertIn(obs["star_id"], view["stars"])

    def test_holdout_coordinate_poison_does_not_change_train_fingerprint(self):
        a = fixture()
        b = copy.deepcopy(a)
        for frame in b["frames"]:
            for obs in frame["observations"]:
                if obs["star_id"].startswith("holdout_"):
                    obs["x"] = 79 - obs["x"]
                    obs["y"] = 59 - obs["y"]
        before = catalog.canonical_sha256(catalog.training_view(a))
        after = catalog.canonical_sha256(catalog.training_view(b))
        self.assertEqual(before, after)
        self.assertNotEqual(catalog.canonical_sha256(catalog.holdout_view(a)),
                            catalog.canonical_sha256(catalog.holdout_view(b)))
        self.assertNotIn("holdout_a", json.dumps(catalog.training_view(a)))
        self.assertNotIn("holdout_z", json.dumps(catalog.training_view(a)))

    def test_outputs_do_not_share_memory(self):
        source = fixture()
        original = copy.deepcopy(source)
        train = catalog.training_view(source)
        holdout = catalog.holdout_view(source)
        train["stars"].append("another")
        train["frames"][1]["observations"][0]["x"] = -1
        holdout["frames"][0]["observations"].append({})
        self.assertEqual(source, original)
        self.assertEqual(catalog.training_view(source)["stars"], ["train_a", "train_z"])

    def test_views_validate_entire_catalog_and_are_not_catalog_schema(self):
        malformed = assign(("frames", 0, "observations", 0, "x"), float("nan"))
        for make_view in (catalog.training_view, catalog.holdout_view):
            with self.assertRaises(ValueError):
                make_view(malformed)
            with self.assertRaises(ValueError):
                catalog.validate_catalog(make_view(fixture()))


if __name__ == "__main__":
    unittest.main()
