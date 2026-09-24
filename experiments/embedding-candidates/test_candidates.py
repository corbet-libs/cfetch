import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from candidates import CATALOG, external_weight_contract, load_candidate, verify_files
from probe import BUCKET, pool, reference_vectors


class CandidateTests(unittest.TestCase):
    def test_catalog_has_pinned_distinct_semantics(self):
        catalog = json.loads(CATALOG.read_bytes())
        for name in catalog["models"]:
            row = load_candidate(name)
            self.assertRegex(row["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(row["files"]["model.onnx"]["sha256"], r"^[0-9a-f]{64}$")
            tokenizer = row["files"]["tokenizer.json"]
            self.assertTrue("sha256" in tokenizer or "git_blob_sha1" in tokenizer)
            if "sha256" in tokenizer:
                self.assertRegex(tokenizer["sha256"], r"^[0-9a-f]{64}$")
            if "git_blob_sha1" in tokenizer:
                self.assertRegex(tokenizer["git_blob_sha1"], r"^[0-9a-f]{40}$")
        modern = load_candidate("ModernBertEmbedLarge")
        mxbai = load_candidate("MxbaiEmbedLargeV1")
        self.assertEqual((modern["max_tokens"], mxbai["max_tokens"]), (8192, 512))
        self.assertNotEqual(modern["pooling"], mxbai["pooling"])
        self.assertNotEqual(modern["query_prefix"], mxbai["query_prefix"])

    def test_changed_weights_and_symlinks_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = b"weights"
            (root / "model.onnx").write_bytes(data)
            candidate = {"files": {"model.onnx": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
            verify_files(root, candidate)
            (root / "model.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_files(root, candidate)
            (root / "real").write_bytes(data)
            (root / "model.onnx").unlink()
            (root / "model.onnx").symlink_to("real")
            with self.assertRaisesRegex(ValueError, "file type"):
                verify_files(root, candidate)

    def test_external_weights_are_pinned_local_files_with_bounded_ranges(self):
        from onnx import helper, TensorProto
        tensor = TensorProto(name="weights", data_type=TensorProto.FLOAT,
                             dims=[2], data_location=TensorProto.EXTERNAL)
        for key, value in (("location", "weights.bin"), ("offset", "4"), ("length", "8")):
            tensor.external_data.add(key=key, value=value)
        model = helper.make_model(helper.make_graph([], "external", [], [], [tensor]))
        files = {"weights.bin": {"bytes": 12, "sha256": "a" * 64}}
        self.assertEqual(external_weight_contract(model, files),
                         {"files": ["weights.bin"], "tensors": 1})
        for field, value in (("location", "../weights.bin"), ("location", "/weights.bin"),
                             ("location", "other.bin"), ("offset", "-1"),
                             ("offset", "12"), ("length", "9")):
            bad = copy.deepcopy(model)
            for item in bad.graph.initializer[0].external_data:
                if item.key == field:
                    item.value = value
            with self.assertRaises(ValueError):
                external_weight_contract(bad, files)
        with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
            external_weight_contract(model, {"weights.bin": {"bytes": 12}})

    def test_nested_external_tensor_cannot_bypass_weight_checks(self):
        from onnx import helper, TensorProto
        tensor = TensorProto(name="weights", data_type=TensorProto.FLOAT,
                             dims=[1], data_location=TensorProto.EXTERNAL)
        tensor.external_data.add(key="location", value="unlisted.bin")
        node = helper.make_node("Constant", [], ["out"], value=tensor)
        model = helper.make_model(helper.make_graph([node], "nested", [], []))
        with self.assertRaisesRegex(ValueError, "verified local artifact"):
            external_weight_contract(model, {})

    def test_padding_never_enters_mean(self):
        values = np.full((1, BUCKET, 2), 1000.0)
        values[0, 0] = [2, 0]
        values[0, 1] = [0, 2]
        result = pool(values, [1, 1] + [0] * (BUCKET - 2), {"dimensions": 2, "pooling": "masked-mean"})
        np.testing.assert_allclose(result, [2 ** -0.5, 2 ** -0.5], atol=1e-6)

    def test_cls_is_not_mean(self):
        values = np.full((1, BUCKET, 2), 1000.0)
        values[0, 0] = [0, 2]
        self.assertEqual(pool(values, [1] * BUCKET, {"dimensions": 2, "pooling": "cls"}), [0, 1])

    def test_wrong_width_and_bad_vectors_refused(self):
        contract = {"dimensions": 1024, "pooling": "cls"}
        for values in (np.zeros((1, BUCKET, 768)), np.zeros((1, BUCKET, 1024)),
                       np.full((1, BUCKET, 1024), np.nan)):
            with self.assertRaises(ValueError):
                pool(values, [1] * BUCKET, contract)

    def test_reference_requires_exact_inputs_model_and_cpu(self):
        identity = {"model": "a", "inputs_sha256": "input-a"}
        reference = {"identity": identity, "device": "CPU", "execution_devices": ["CPU"],
                     "vectors": [[1, 0], [0, 1], [1, 0]]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.json"
            path.write_text(json.dumps(reference))
            reference_vectors(path, identity, 2)
            for bad in ({**identity, "model": "b"}, {**identity, "inputs_sha256": "input-b"}):
                with self.assertRaisesRegex(ValueError, "another"):
                    reference_vectors(path, bad, 2)
            bad = copy.deepcopy(reference)
            bad["execution_devices"] = ["GPU.0"]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "CPU"):
                reference_vectors(path, identity, 2)


if __name__ == "__main__":
    unittest.main()
