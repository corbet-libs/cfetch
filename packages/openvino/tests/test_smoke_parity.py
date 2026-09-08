from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

from packages.openvino import convert, reference, smoke_parity


class FrozenFixtureTokenizer:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def encode(self, text: str, *, add_special_tokens: bool):
        if add_special_tokens:
            raise AssertionError("fixture tokenization must add BOS/EOS explicitly")
        if text.startswith("task: search result | query: "):
            prefix_tokens = 7
        elif text.startswith("title: none | text: "):
            prefix_tokens = 6
        else:
            raise AssertionError("unexpected semantic fixture prefix")
        topic_tokens = text.count("cat") + text.count("music")
        return SimpleNamespace(
            ids=[0] * (prefix_tokens + topic_tokens + self.offset)
        )


class SemanticFixtureSmokeTests(unittest.TestCase):
    def test_every_fixture_input_reaches_its_exact_bucket(self) -> None:
        results = smoke_parity.validate_sequence_semantic_fixture(
            FrozenFixtureTokenizer()
        )
        self.assertEqual(
            [row["bucket"] for row in results],
            [32, 64, 128, 257, 512, 1024, 2048],
        )
        self.assertTrue(
            all(row["token_counts"] == [row["bucket"]] * 3 for row in results)
        )

    def test_fixture_refuses_tokenizer_contract_drift(self) -> None:
        with self.assertRaisesRegex(
            smoke_parity.ParityError, "expected three exact 32-token inputs"
        ):
            smoke_parity.validate_sequence_semantic_fixture(
                FrozenFixtureTokenizer(offset=1)
            )

    def test_long_probes_have_real_tokens_beyond_the_window_and_no_truncation(self) -> None:
        tokenizer = SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=[10, 11, 12, 13]),
            id_to_token=lambda token: str(token) if token in [10, 11, 12, 13] else None,
        )
        fixtures = smoke_parity.long_token_fixtures(tokenizer)
        self.assertEqual([(bucket, len(ids)) for _, bucket, ids in fixtures], [(512, 258), (2048, 1946)])
        for _, _, ids in fixtures:
            self.assertEqual((ids[0], ids[-1]), (2, 1))
            self.assertTrue(all(token in [10, 11, 12, 13] for token in ids[1:-1]))
            self.assertNotIn(0, ids, "long fixture tokens are actual input, not padding")
        for invalid in [[], [0], [2], [-1], [True], [999]]:
            tokenizer.encode = lambda text, **kwargs: SimpleNamespace(ids=invalid)
            with self.assertRaisesRegex(smoke_parity.ParityError, "valid non-special"):
                smoke_parity.long_token_fixtures(tokenizer)

    def test_reference_uses_fresh_upstream_model_and_ordinary_mask(self) -> None:
        fake_torch = ModuleType("torch")
        fake_torch.float32 = "float32"
        fake_nn = ModuleType("torch.nn")
        functional = ModuleType("torch.nn.functional")
        functional.linear = mock.Mock(side_effect=[mock.MagicMock(), mock.MagicMock()])
        functional.normalize = mock.Mock(return_value=object())
        fake_nn.functional = functional
        fake_torch.nn = fake_nn
        backbone = mock.MagicMock()
        backbone.cpu.return_value = backbone
        backbone.eval.return_value = backbone
        backbone.config = SimpleNamespace(use_bidirectional_attention=True, sliding_window=257)
        backbone.parameters.return_value = []
        auto_model = mock.Mock()
        auto_model.from_pretrained.return_value = backbone
        weights = []
        for shape in [(3072, 768), (768, 3072)]:
            weight = mock.MagicMock()
            weight.shape = shape
            weight.to.return_value = weight
            weights.append(weight)
        loader = mock.Mock(side_effect=[{"linear.weight": weight} for weight in weights])
        modules = {
            "torch": fake_torch,
            "torch.nn": fake_nn,
            "torch.nn.functional": functional,
            "transformers": SimpleNamespace(AutoModel=auto_model),
            "safetensors": ModuleType("safetensors"),
            "safetensors.torch": SimpleNamespace(load_file=loader),
        }
        with (
            mock.patch.dict("sys.modules", modules),
            mock.patch.object(reference, "verify_source_files") as verify,
            mock.patch.object(reference, "validate_semantic_source") as validate,
            mock.patch.object(convert, "build_torch_pipeline", side_effect=AssertionError("converter cannot be the oracle")),
            mock.patch.object(convert, "safe_bidirectional_attention_masks", side_effect=AssertionError("mask must remain upstream")),
            mock.patch.object(convert, "masked_mean", side_effect=AssertionError("pooling must remain independent")),
        ):
            source = Path("independent-source").resolve()
            pipeline = reference.build_upstream(source)
            verify.assert_called_once_with(source)
            validate.assert_called_once_with(source)
            auto_model.from_pretrained.assert_called_once_with(
                str(source), local_files_only=True, trust_remote_code=False,
                torch_dtype="float32", attn_implementation="sdpa",
            )
            ids, mask = mock.MagicMock(), mock.MagicMock()
            self.assertIs(pipeline(ids, mask), functional.normalize.return_value)
            backbone.assert_called_once_with(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)
            mask.unsqueeze.assert_called_once_with(-1)
            self.assertIs(functional.linear.call_args_list[0].args[1], weights[0])
            self.assertIs(functional.linear.call_args_list[1].args[1], weights[1])
            for invalid in [512, 129]:
                backbone.config.sliding_window = invalid
                with self.assertRaisesRegex(convert.ConversionError, "radius 257"):
                    reference.build_upstream(source)

    def test_smoke_compares_actual_long_inputs_to_the_independent_reference(self) -> None:
        # Exercise run() without loading a model. The simulated bad graph
        # matches every padded-short case and diverges only on real long input.
        class Array:
            def __init__(self, values):
                self.values = values
                self.shape = (len(values), len(values[0])) if isinstance(values[0], list) else (len(values),)

            def __getitem__(self, index):
                return Array(self.values[index])

            def tolist(self):
                return self.values

            def astype(self, *args, **kwargs):
                return self

            def tobytes(self, **kwargs):
                return json.dumps(self.values).encode()

        class Tensor:
            def __init__(self, array):
                self.array = array

            def detach(self):
                return self

            def cpu(self):
                return self

            def to(self, *args):
                return self

            def numpy(self):
                return self.array

        tokenizer = SimpleNamespace(
            no_truncation=mock.Mock(), no_padding=mock.Mock(),
            encode=lambda text, **kwargs: SimpleNamespace(ids=[10, 11, 12, 13]),
            id_to_token=lambda token: str(token),
        )
        for divergent_long in [False, True]:
            compiled = mock.MagicMock()
            compiled.output.return_value = "embedding"

            def candidate(inputs):
                count = sum(inputs["attention_mask"].values[0])
                first = -1.0 if divergent_long and count > 257 else 1.0
                return {"embedding": Array([[first] + [0.0] * 767])}

            compiled.side_effect = candidate
            core = mock.Mock(available_devices=["CPU"])
            core.compile_model.return_value = compiled
            oracle = mock.Mock(return_value=Tensor(Array([[1.0] + [0.0] * 767])))
            modules = {
                "numpy": SimpleNamespace(
                    int64="int64", float32="float32",
                    asarray=lambda values, **kwargs: values if isinstance(values, Array) else Array(values),
                ),
                "openvino": SimpleNamespace(Core=lambda: core),
                "torch": SimpleNamespace(
                    __version__="test-torch", float32="float32",
                    no_grad=nullcontext, from_numpy=Tensor,
                ),
                "transformers": SimpleNamespace(__version__="test-transformers"),
                "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: tokenizer)),
            }
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "artifact-manifest.json").write_text("{}")
                artifact = SimpleNamespace(
                    tokenizer_json=path / "tokenizer.json", graph_xml=path / "model.xml",
                    graph_bin=path / "model.bin", input_ids_name="input_ids",
                    attention_mask_name="attention_mask", output_name="embedding",
                )
                with (
                    mock.patch.dict("sys.modules", modules),
                    mock.patch.object(smoke_parity, "load_artifact", return_value=artifact),
                    mock.patch.object(smoke_parity, "validate_sequence_semantic_fixture", return_value=[]),
                    mock.patch.object(reference, "build_upstream", return_value=oracle) as build,
                ):
                    if divergent_long:
                        with self.assertRaisesRegex(smoke_parity.ParityError, "window-boundary-258.*cosine"):
                            smoke_parity.run(path, path)
                    else:
                        report = smoke_parity.run(path, path)
                        build.assert_called_once_with(path.resolve())
                        self.assertEqual(len(report["cases"]), len(smoke_parity.CASES) + 2)
                        long = report["cases"][-2:]
                        self.assertEqual([(row["bucket"], row["token_count"]) for row in long], [(512, 258), (2048, 1946)])
                        self.assertEqual(report["reference"]["implementation"], reference.DESCRIPTION)
                        self.assertEqual(oracle.call_count, len(report["cases"]))
                        self.assertTrue(all(len(row["reference_f32_sha256"]) == 64 for row in long))


if __name__ == "__main__":
    unittest.main()
