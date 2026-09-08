from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from packages.openvino import convert


class ConversionContractTests(unittest.TestCase):
    def test_empty_attention_rows_gain_only_their_diagonal(self) -> None:
        class FakeRows:
            def __init__(self, rows):
                self.rows = rows

            def any(self, *, dim, keepdim):
                self_outer = self

                class FakeColumns:
                    def __init__(self, values):
                        self.values = values

                    def __invert__(self):
                        return FakeColumns([not value for value in self.values])

                    def __and__(self, matrix):
                        return FakeRows(
                            [
                                [enabled and value for value in row]
                                for enabled, row in zip(self.values, matrix.rows)
                            ]
                        )

                if dim != -1 or keepdim is not True:
                    raise AssertionError("attention repair must inspect the last axis")
                return FakeColumns([any(row) for row in self_outer.rows])

            def __or__(self, other):
                return FakeRows(
                    [
                        [left or right for left, right in zip(left_row, right_row)]
                        for left_row, right_row in zip(self.rows, other.rows)
                    ]
                )

        allowed = FakeRows(
            [
                [True, False, False],
                [False, False, False],
                [False, True, False],
            ]
        )
        diagonal = FakeRows(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        )
        repaired = convert._ensure_nonempty_attention_rows(allowed, diagonal)
        self.assertEqual(
            repaired.rows,
            [
                [True, False, False],
                [False, True, False],
                [False, True, False],
            ],
        )

    def test_backbone_attention_backend_is_frozen_to_sdpa(self) -> None:
        calls = []
        loaded_config = SimpleNamespace(use_bidirectional_attention=True, sliding_window=257)

        class FakeModule:
            def register_buffer(self, name, value):
                setattr(self, name, value)

            def eval(self):
                return self

            def parameters(self):
                return []

        class FakeWeight:
            def to(self, **kwargs):
                return self

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append((args, kwargs))
                return SimpleNamespace(config=loaded_config)

        fake_torch = ModuleType("torch")
        fake_torch.float32 = "float32"
        fake_nn = ModuleType("torch.nn")
        fake_nn.Module = FakeModule
        fake_functional = ModuleType("torch.nn.functional")
        fake_nn.functional = fake_functional
        fake_torch.nn = fake_nn
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = FakeAutoModel
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "torch": fake_torch,
                    "torch.nn": fake_nn,
                    "torch.nn.functional": fake_functional,
                    "transformers": fake_transformers,
                },
            ),
            mock.patch.object(convert, "_load_dense_weight", return_value=FakeWeight()),
        ):
            convert.build_torch_pipeline(Path("unused-source"))
            self.assertEqual(len(calls), 1)
            # The serialized 512-token width must be converted once by the
            # upstream config. Accepting it here doubled the local radius.
            for invalid in (512, 129):
                loaded_config.sliding_window = invalid
                with self.assertRaisesRegex(convert.ConversionError, "radius 257"):
                    convert.build_torch_pipeline(Path("unused-source"))

        self.assertEqual(calls[0][1]["attn_implementation"], "sdpa")

    def test_masked_mean_selects_real_tokens_before_reduction(self) -> None:
        calls = []

        class FakeMask:
            def unsqueeze(self, axis):
                calls.append(("unsqueeze", axis))
                return self

            def bool(self):
                calls.append(("bool",))
                return "real-token-mask"

            def sum(self, **kwargs):
                calls.append(("mask-sum", kwargs))
                return FakeDenominator()

        class FakeDenominator:
            def to(self, dtype):
                calls.append(("denominator-to", dtype))
                return self

            def clamp_min(self, value):
                calls.append(("clamp-min", value))
                return "safe-denominator"

        class FakeEmbeddings:
            dtype = "embedding-dtype"

        class FakeSelected:
            def sum(self, **kwargs):
                calls.append(("selected-sum", kwargs))
                return FakeNumerator()

        class FakeNumerator:
            def __truediv__(self, denominator):
                calls.append(("divide", denominator))
                return "pooled"

        embeddings = FakeEmbeddings()
        fake_torch = SimpleNamespace(
            zeros_like=lambda value: calls.append(("zeros-like", value)) or "zeros",
            where=lambda condition, value, zero: calls.append(
                ("where", condition, value, zero)
            )
            or FakeSelected(),
        )
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            result = convert.masked_mean(embeddings, FakeMask())

        self.assertEqual(result, "pooled")
        where_index = next(
            index for index, call in enumerate(calls) if call[0] == "where"
        )
        reduction_index = next(
            index for index, call in enumerate(calls) if call[0] == "selected-sum"
        )
        self.assertLess(where_index, reduction_index)
        self.assertIn(("where", "real-token-mask", embeddings, "zeros"), calls)
        self.assertIn(("selected-sum", {"dim": 1}), calls)
        self.assertIn(("mask-sum", {"dim": 1, "keepdim": True}), calls)

    def test_torch_export_binds_one_bounded_sequence_symbol_to_both_inputs(
        self,
    ) -> None:
        dimensions = []
        calls = []

        class FakeDim:
            def __init__(self, name, **bounds):
                self.name = name
                self.bounds = bounds
                dimensions.append(self)

        def fake_export(*args, **kwargs):
            calls.append((args, kwargs))
            return "exported-program"

        fake_torch = SimpleNamespace(
            export=SimpleNamespace(Dim=FakeDim, export=fake_export)
        )
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            result = convert.export_torch_pipeline(
                "pipeline", "example-ids", "example-mask"
            )

        self.assertEqual(result, "exported-program")
        self.assertEqual(len(dimensions), 1)
        self.assertEqual(dimensions[0].name, "sequence")
        self.assertEqual(dimensions[0].bounds, {"min": 1, "max": 2048})
        args, kwargs = calls[0]
        self.assertEqual(args, ("pipeline", ("example-ids", "example-mask")))
        id_shapes, mask_shapes = kwargs["dynamic_shapes"]
        self.assertIs(id_shapes[1], dimensions[0])
        self.assertIs(mask_shapes[1], dimensions[0])
        self.assertIs(kwargs["strict"], False)

    def test_unit_reduction_matmuls_are_rewritten_exactly(self) -> None:
        class FakeDimension:
            def __init__(self, minimum, maximum=None):
                self.minimum = minimum
                self.maximum = minimum if maximum is None else maximum

            @property
            def is_static(self):
                return self.minimum == self.maximum

            def get_length(self):
                if not self.is_static:
                    raise AssertionError("dynamic dimension has no static length")
                return self.minimum

            def get_min_length(self):
                return self.minimum

            def get_max_length(self):
                return self.maximum

            def signature(self):
                return (self.minimum, self.maximum)

        class FakeRank:
            def __init__(self, length):
                self.length = length
                self.is_static = True

            def get_length(self):
                return self.length

        class FakeShape:
            def __init__(self, *dimensions):
                self.dimensions = dimensions
                self.rank = FakeRank(len(dimensions))

            def __getitem__(self, index):
                return self.dimensions[index]

            def __eq__(self, other):
                return isinstance(other, FakeShape) and [
                    value.signature() for value in self.dimensions
                ] == [value.signature() for value in other.dimensions]

        class FakePort:
            def __init__(self, shape, element_type="f32"):
                self.shape = shape
                self.element_type = element_type
                self.replacement = None

            def get_partial_shape(self):
                return self.shape

            def get_element_type(self):
                return self.element_type

            def replace(self, replacement):
                self.replacement = replacement

        class FakeNode:
            def __init__(self, name):
                self.name = name
                self.left = FakePort(
                    FakeShape(
                        FakeDimension(1), FakeDimension(128), FakeDimension(1)
                    )
                )
                self.right = FakePort(
                    FakeShape(
                        FakeDimension(1),
                        FakeDimension(1),
                        FakeDimension(0, -1),
                    )
                )
                self.result = FakePort(
                    FakeShape(
                        FakeDimension(1),
                        FakeDimension(128),
                        FakeDimension(0, -1),
                    )
                )

            def get_type_name(self):
                return "MatMul"

            def get_friendly_name(self):
                return self.name

            def get_attributes(self):
                return {"transpose_a": False, "transpose_b": False}

            def input_value(self, index):
                return (self.left, self.right)[index]

            def output(self, index):
                self.assert_zero(index)
                return self.result

            @staticmethod
            def assert_zero(index):
                if index != 0:
                    raise AssertionError("fixture has one output")

        class FakeMultiply:
            def __init__(self, left, right):
                self.name = None
                self.result = FakePort(
                    FakeShape(
                        FakeDimension(1),
                        FakeDimension(128),
                        FakeDimension(0, -1),
                    )
                )

            def set_friendly_name(self, name):
                self.name = name

            def output(self, index):
                FakeNode.assert_zero(index)
                return self.result

        class FakeModel:
            def __init__(self, nodes):
                self.nodes = nodes
                self.validated = False

            def get_ordered_ops(self):
                return self.nodes

            def validate_nodes_and_infer_types(self):
                self.validated = True

        current_names = {
            "bmm/aten.bmm.default/MatMul",
            "bmm_1/aten.bmm.default/MatMul",
        }
        self.assertEqual(convert.DEGENERATE_OUTER_PRODUCT_NODES, current_names)
        nodes = [FakeNode(name) for name in sorted(current_names)]
        similar_node = FakeNode("unrelated/aten.bmm.default/MatMul")
        model = FakeModel([*nodes, similar_node])
        fake_openvino = ModuleType("openvino")
        fake_openvino.__path__ = []
        fake_ops = ModuleType("openvino.opset13")
        fake_ops.multiply = FakeMultiply
        with mock.patch.dict(
            "sys.modules",
            {"openvino": fake_openvino, "openvino.opset13": fake_ops},
        ):
            convert.rewrite_unit_reduction_matmuls(model)

        self.assertTrue(model.validated)
        self.assertTrue(all(node.result.replacement is not None for node in nodes))
        self.assertIsNone(similar_node.result.replacement)
        missing = FakeModel(nodes[:1])
        with (
            mock.patch.dict(
                "sys.modules",
                {"openvino": fake_openvino, "openvino.opset13": fake_ops},
            ),
            self.assertRaisesRegex(convert.ConversionError, "lacks the two frozen"),
        ):
            convert.rewrite_unit_reduction_matmuls(missing)

        stale_nodes = [
            FakeNode("bmm_22/aten.bmm.default/MatMul"),
            FakeNode("bmm_23/aten.bmm.default/MatMul"),
        ]
        stale = FakeModel(stale_nodes)
        with (
            self.subTest("stale rotary names are never accepted as a fallback"),
            mock.patch.dict(
                "sys.modules",
                {"openvino": fake_openvino, "openvino.opset13": fake_ops},
            ),
            self.assertRaisesRegex(convert.ConversionError, "lacks the two frozen"),
        ):
            convert.rewrite_unit_reduction_matmuls(stale)
        self.assertFalse(stale.validated)
        self.assertTrue(all(node.result.replacement is None for node in stale_nodes))

    def test_cli_keeps_failure_diagnostic_out_of_result_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(
                convert,
                "convert",
                side_effect=convert.ConversionError("safe conversion failure"),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            status = convert.main(
                [
                    "--source-dir",
                    "unused-source",
                    "--legal-dir",
                    "unused-legal",
                    "--output-dir",
                    "unused-output",
                ]
            )
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("safe conversion failure", stderr.getvalue())

    def test_source_file_verification_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "one.bin").write_bytes(b"exact")
            expected = {"one.bin": hashlib.sha256(b"exact").hexdigest()}
            convert.verify_source_files(root, expected)
            (root / "one.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(convert.ConversionError, "digest mismatch"):
                convert.verify_source_files(root, expected)

    def test_semantic_source_requires_mean_dense_dense_normalize_contract(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            documents = {
                "modules.json": convert.EXPECTED_MODULES,
                "1_Pooling/config.json": convert.EXPECTED_POOLING,
                "2_Dense/config.json": convert.EXPECTED_DENSE_2,
                "3_Dense/config.json": convert.EXPECTED_DENSE_3,
                "sentence_bert_config.json": convert.EXPECTED_SENTENCE_BERT,
                "config.json": {
                    "architectures": ["Gemma3TextModel"],
                    "dtype": "float32",
                    "hidden_size": 768,
                    "max_position_embeddings": 2048,
                    "model_type": "gemma3_text",
                    "num_hidden_layers": 24,
                    "pad_token_id": 0,
                    "sliding_window": 512,
                    "use_bidirectional_attention": True,
                },
            }
            for relative, document in documents.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(document), encoding="utf-8")
            convert.validate_semantic_source(root)
            broken = dict(convert.EXPECTED_DENSE_3)
            broken["activation_function"] = "torch.nn.modules.activation.Tanh"
            (root / "3_Dense/config.json").write_text(
                json.dumps(broken), encoding="utf-8"
            )
            with self.assertRaisesRegex(convert.ConversionError, "frozen mean"):
                convert.validate_semantic_source(root)


if __name__ == "__main__":
    unittest.main()
