import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

spec = importlib.util.spec_from_file_location("export_plain_model", Path(__file__).with_name("export-plain-model.py"))
export_plain_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_plain_model)


def tiny_model(rng):
    table = rng.normal(size=(64, 16)).astype(np.float32)
    weights = rng.normal(size=(16, 8)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["table", "ids"], ["rows"], axis=0),
            helper.make_node("MatMul", ["rows", "weights"], ["out"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [None])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, 8])],
        [numpy_helper.from_array(table, "table"), numpy_helper.from_array(weights, "weights")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)]), table, weights


class QuantizeTests(unittest.TestCase):
    def test_symmetric_int8_is_per_slice_and_round_half_even(self):
        weights = np.array([[1.0, -2.0], [0.5, 4.0], [0.0, 0.0]], np.float32)
        q, scale = export_plain_model.symmetric_int8(weights, axis=1)
        self.assertEqual(q.dtype, np.int8)
        np.testing.assert_allclose(scale, [1.0 / 127, 4.0 / 127])
        self.assertEqual(int(np.abs(q).max()), 127)
        np.testing.assert_array_equal(q[:, 1], np.rint(weights[:, 1] / scale[1]).astype(np.int8))
        q_rows, row_scale = export_plain_model.symmetric_int8(weights, axis=0)
        self.assertEqual(float(row_scale[2]), 1.0)  # an all-zero row keeps a finite scale
        np.testing.assert_array_equal(q_rows[2], [0, 0])

    def test_rewrite_uses_standard_operators_and_preserves_outputs(self):
        rng = np.random.default_rng(7)
        model, table, weights = tiny_model(rng)
        ids = np.array([0, 5, 63, 5], np.int64)
        expected = table[ids] @ weights
        quantized = export_plain_model.quantize(model, min_matmul=1, min_table=1)
        onnx.checker.check_model(quantized)
        ops = [n.op_type for n in quantized.graph.node]
        self.assertIn("DequantizeLinear", ops)
        self.assertEqual({n.domain for n in quantized.graph.node}, {""})
        names = {i.name for i in quantized.graph.initializer}
        self.assertNotIn("table", names)
        self.assertNotIn("weights", names)
        self.assertEqual(numpy_helper.to_array(next(i for i in quantized.graph.initializer if i.name == "weights_q8")).dtype, np.int8)
        (actual,) = ReferenceEvaluator(quantized).run(None, {"ids": ids})
        cosine = (actual * expected).sum(axis=1) / np.linalg.norm(actual, axis=1) / np.linalg.norm(expected, axis=1)
        self.assertGreater(float(cosine.min()), 0.999)

    def test_pack_manifest_names_exactly_the_loader_files(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            graph, tokenizer = root / "graph", root / "tok"
            graph.mkdir()
            tokenizer.mkdir()
            for name in ("model.onnx", "model.onnx_data"):
                (graph / name).write_bytes(name.encode())
            for name in export_plain_model.PACK_FILES:
                (tokenizer / name).write_text("{}")
            digest = export_plain_model.pack(graph, tokenizer, root / "pack")
            manifest = (root / "pack" / "manifest.json").read_text()
            self.assertEqual(len(digest), 64)
            self.assertIn(export_plain_model.PIPELINE, manifest)
            self.assertEqual(len(__import__("json").loads(manifest)["files"]), 6)
            with self.assertRaises(FileExistsError):
                export_plain_model.pack(graph, tokenizer, root / "pack")


if __name__ == "__main__":
    unittest.main()
