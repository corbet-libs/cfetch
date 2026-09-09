"""Focused conversion preflight/metadata tests; no models or native inference."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import convert_canonical as conversion


class CanonicalConversionTests(unittest.TestCase):
    def reader(self, *, fields=None, dense_2=True, shape=None, dtype=None):
        values = {**conversion.EXPECTED_FIELDS, **(fields or {})}
        values["tokenizer.ggml.tokens"] = ["<pad>", "<eos>", "<bos>"]
        def get_field(name):
            if name not in values:
                return None
            value = values[name]
            return SimpleNamespace(contents=lambda index=None: value if index is None else value[index])
        f32 = SimpleNamespace(name="F32")
        tensors = [SimpleNamespace(name="dense_3.weight", shape=[3072, 768], tensor_type=f32)]
        if dense_2:
            tensors.append(SimpleNamespace(name="dense_2.weight", shape=shape or [768, 3072],
                                           tensor_type=dtype or f32))
        return SimpleNamespace(get_field=get_field, tensors=tensors), f32

    def test_metadata_rejects_missing_heads_wrong_shapes_mask_tokens_and_quantization(self):
        reader, f32 = self.reader()
        self.assertEqual(conversion.validate_reader(reader, f32)["errors"], [])
        cases = [
            {"dense_2": False}, {"shape": [3072, 768]},
            {"fields": {"gemma-embedding.attention.sliding_window": 257}},
            {"fields": {"tokenizer.ggml.add_eos_token": 1}},
            {"fields": {"gemma-embedding.pooling_type": 0}},
            {"dtype": SimpleNamespace(name="Q8_0")},
        ]
        for case in cases:
            with self.subTest(case=case):
                reader, f32 = self.reader(**case)
                diagnostics = conversion.validate_reader(reader, f32)
                self.assertTrue(diagnostics["errors"])
                self.assertIn("gemma-embedding.attention.sliding_window", diagnostics["fields"])

    def test_requirements_include_checks_exact_installed_versions_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements"
            requirements.mkdir()
            (root / conversion.REQUIREMENTS).write_text(
                "-r ./base.txt\n--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.11.0\n")
            (requirements / "base.txt").write_text("numpy~=1.26.4\ntransformers==4.57.6\n")
            versions = {"torch": "2.11.0+cpu", "numpy": "1.26.4", "transformers": "4.57.6",
                        "packaging": "25.0", "safetensors": "0.6.2", "huggingface-hub": "0.35.0",
                        "tokenizers": "0.22.0"}
            with patch.object(conversion.importlib.metadata, "version", side_effect=versions.__getitem__), patch.object(
                    conversion.subprocess, "Popen", side_effect=AssertionError("must not convert")):
                result = conversion.check_dependencies(root)
                self.assertEqual(result["torch"]["installed"], "2.11.0+cpu")
                versions["torch"] = "2.8.0"
                with self.assertRaisesRegex(conversion.ConversionError, "torch==2.11.0: installed 2.8.0"):
                    conversion.check_dependencies(root)

    def test_git_blob_comparison_rejects_changes_even_when_status_is_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"convert_hf_to_gguf.py", "conversion/base.py", "conversion/gemma.py",
                     "gguf-py/gguf/gguf_reader.py", conversion.REQUIREMENTS}
            tree = []
            for name in sorted(files):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = b"pinned source\n"
                path.write_bytes(data)
                blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                tree.append(f"100644 blob {blob}\t{name}\0".encode())
            def git(_root, *args):
                if args == ("rev-parse", "--show-toplevel"):
                    return str(root).encode() + b"\n"
                if args == ("rev-parse", "HEAD"):
                    return conversion.REVISION.encode() + b"\n"
                if args[0] == "status":
                    return b""
                return b"".join(tree)
            with patch.object(conversion, "git", side_effect=git):
                self.assertEqual(set(conversion.verify_converter(root)), files)
                (root / "conversion/gemma.py").write_text("changed despite clean status\n")
                with self.assertRaisesRegex(conversion.ConversionError, "pinned converter bytes differ"):
                    conversion.verify_converter(root)

    def test_existing_output_including_dangling_symlink_is_never_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.symlink_to(root / "missing", target_is_directory=True)
            with patch.object(conversion.subprocess, "Popen", side_effect=AssertionError("must not convert")):
                with self.assertRaisesRegex(conversion.ConversionError, "must be fresh"):
                    conversion.convert(root, root, output, 30)
                self.assertTrue(output.is_symlink())


if __name__ == "__main__":
    unittest.main()
