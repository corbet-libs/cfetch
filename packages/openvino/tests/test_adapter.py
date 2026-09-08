from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from packages.openvino import adapter
from packages.openvino.manifest import (
    Artifact,
    HostBinding,
    HostFileBinding,
    PackageManifest,
    Scope,
)


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        if text == "over-limit":
            return list(range(2049))
        return list(range(33 if text == "bucket-64" else 3))


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int], int]] = []

    def embed(self, input_ids, attention_mask, bucket):
        self.calls.append((list(input_ids), list(attention_mask), bucket))
        return [1.0] + [0.0] * 767

    def runtime_evidence(self, bucket):
        return {
            "bucket": bucket,
            "requested_device": "NPU",
            "execution_devices": ["NPU"],
            "execution_devices_source": "compiled_model.get_property(EXECUTION_DEVICES)",
            "device_properties": {},
            "device_properties_source": "core.get_property",
        }

    def host_evidence(self):
        return {
            "system": "Linux",
            "machine": "x86_64",
            "kernel_release": "test-kernel",
            "files": [],
        }


class FakeSigner:
    public_key_hex = "f" * 64

    def sign(self, message: bytes) -> bytes:
        return hashlib.sha512(message).digest()


def scope() -> Scope:
    return Scope(
        package_state="candidate",
        scope_id="intel-test-npu",
        backend="openvino",
        transport="supervised-local",
        runtime="openvino test",
        compiler="openvino test static buckets",
        package_target="linux-x86_64",
        artifact_source="google/embeddinggemma-300m@57c266a740f537b4dc058e1b0cda161fd15afa75",
        artifact_sha256="1" * 64,
        internal_precision="fp16-hardware-compute",
        device_class="npu",
        device="test-intel-npu",
        openvino_device="NPU",
        openvino_compile_config={},
        required_openvino_properties={
            "FULL_DEVICE_NAME": "Test Intel NPU",
            "DEVICE_ARCHITECTURE": "test-npu-architecture",
            "NPU_DRIVER_VERSION": 1,
            "NPU_COMPILER_VERSION": 2,
        },
        required_execution_devices=("NPU",),
        required_host=HostBinding(
            system="Linux",
            machine="x86_64",
            kernel_release="test-kernel",
            files=(HostFileBinding(Path("/usr/lib/test-driver.so"), "8" * 64),),
        ),
        placement_evidence_sha256="2" * 64,
        sequence_capability_evidence_sha256="3" * 64,
        performance_evidence_sha256="4" * 64,
        compatibility_report_sha256=None,
        attestation_public_key="5" * 64,
        attestation_private_key_file=Path("unused.key"),
        accelerated_placement=True,
    )


def package(selected_scope: Scope) -> PackageManifest:
    artifact = Artifact(
        root=Path("."),
        manifest_path=Path("artifact-manifest.json"),
        manifest_sha256=selected_scope.artifact_sha256,
        graph_xml=Path("model.xml"),
        graph_bin=Path("model.bin"),
        tokenizer_json=Path("tokenizer.json"),
        input_ids_name="input_ids",
        attention_mask_name="attention_mask",
        output_name="embedding",
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        files=(
            Path("model.xml"),
            Path("model.bin"),
            Path("tokenizer.json"),
        ),
        conversion_versions={
            "openvino": "test",
            "safetensors": "test",
            "torch": "test",
            "transformers": "test",
        },
    )
    return PackageManifest(
        path=Path("package-manifest.json"),
        artifact=artifact,
        scopes={selected_scope.scope_id: selected_scope},
        dependency_versions={},
        runtime_manifest_sha256="9" * 64,
        package_state="candidate",
    )


class AdapterContractTests(unittest.TestCase):
    def setUp(self) -> None:
        class TestGovernor:
            policy_sha256 = "1" * 64
            @contextmanager
            def operation(self, kind, bucket):
                yield SimpleNamespace(deadline_ns=time.monotonic_ns() + 5_000_000_000)

        governor_patch = patch.object(adapter, "from_installation", return_value=TestGovernor())
        governor_patch.start()
        self.addCleanup(governor_patch.stop)
        hash_host_file = adapter._host_file_sha256
        file_patch = patch.object(adapter, "_host_file_sha256", side_effect=lambda path:
            "1" * 64 if path == Path(adapter.GOVERNOR_POLICY_PATH) else hash_host_file(path))
        file_patch.start()
        self.addCleanup(file_patch.stop)
        self.scope = scope()
        self.engine = FakeEngine()
        self.package = package(self.scope)
        self.signer = FakeSigner()
        self.service = adapter.EmbeddingService(
            self.package,
            FakeTokenizer(),
            lambda _package, _scope: self.engine,
            {self.scope.scope_id: self.signer},
        )

    def test_execution_devices_normalizes_scalar_and_sequence_forms(self) -> None:
        self.assertEqual(adapter._normalize_execution_devices("NPU"), ("NPU",))
        self.assertEqual(
            adapter._normalize_execution_devices(["GPU.0", "GPU.1"]),
            ("GPU.0", "GPU.1"),
        )

    def test_execution_devices_rejects_invalid_raw_values(self) -> None:
        invalid_values = (
            "",
            (),
            ["NPU", ""],
            ["NPU", 1],
            b"NPU",
            bytearray(b"NPU"),
            memoryview(b"NPU"),
            {"NPU"},
            iter(("NPU",)),
            None,
        )
        for raw in invalid_values:
            with self.subTest(raw=raw), self.assertRaises((TypeError, ValueError)):
                adapter._normalize_execution_devices(raw)

    @staticmethod
    def request_body(inputs: list[str], requested_scope: str = "intel-test-npu") -> bytes:
        return json.dumps(
            {
                "model": "google/embeddinggemma-300m",
                "dimensions": 768,
                "input": inputs,
                "cfetch_requested_scope_id": requested_scope,
            },
            separators=(",", ":"),
        ).encode()

    def test_service_right_pads_to_smallest_bucket_without_truncation(self) -> None:
        response_body, signer = self.service.response_for(
            self.request_body(["short", "bucket-64"])
        )
        response = json.loads(response_body)
        self.assertIs(signer, self.signer)
        self.assertEqual([row["token_count"] for row in response["data"]], [3, 33])
        self.assertEqual([row["sequence_bucket"] for row in response["data"]], [32, 64])
        self.assertEqual([call[2] for call in self.engine.calls], [32, 64])
        self.assertEqual(self.engine.calls[0][1], [1, 1, 1] + [0] * 29)
        self.assertTrue(all(row["truncated"] is False for row in response["data"]))
        runtime = response["cfetch_runtime_evidence"]
        self.assertEqual(set(runtime), {
            "schema_version",
            "provider",
            "scope_id",
            "host",
            "host_source",
            "bucket_results",
        })
        self.assertEqual(
            [result["bucket"] for result in runtime["bucket_results"]], [32, 64]
        )
        self.assertEqual(response["cfetch_execution"]["scope_id"], "intel-test-npu")
        self.assertEqual(
            response["cfetch_execution"]["transport"], "supervised-local"
        )
        self.assertEqual(response["cfetch_execution"]["package_state"], "candidate")
        self.assertIsNone(
            response["cfetch_execution"]["compatibility_report_sha256"]
        )

    def test_service_releases_cached_native_engines_explicitly(self) -> None:
        self.service.response_for(self.request_body(["short"]))
        self.assertEqual(list(self.service._engines), ["intel-test-npu"])
        self.service.close()
        self.assertEqual(self.service._engines, {})

    def test_wrong_requested_scope_is_rejected(self) -> None:
        with self.assertRaisesRegex(adapter.RequestError, "exact scope"):
            self.service.response_for(
                self.request_body(["short"], requested_scope="intel-test-cpu")
            )

    def test_overlength_input_is_rejected_not_truncated(self) -> None:
        with self.assertRaisesRegex(adapter.RequestError, "truncation is forbidden"):
            self.service.response_for(self.request_body(["over-limit"]))
        self.assertEqual(self.engine.calls, [])

    def test_auth_stdin_is_one_exact_32_byte_hex_credential(self) -> None:
        bearer = "a" * 64
        self.assertEqual(
            adapter.parse_auth_line(io.BytesIO(f'{{"bearer":"{bearer}"}}\n'.encode())),
            bearer,
        )
        with self.assertRaises(adapter.RequestError):
            adapter.parse_auth_line(io.BytesIO(b'{"bearer":"ABC"}\n'))

    def test_http_response_signature_binds_nonce_and_exact_bodies(self) -> None:
        bearer = "a" * 64
        signer = FakeSigner()
        server = adapter.AdapterServer(
            ("127.0.0.1", 0), self.service, bearer
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = self.request_body(["short"])
            nonce_hex = "b" * 64
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request(
                "POST",
                "/v1/embeddings",
                body=body,
                headers={
                    "Authorization": f"Bearer {bearer}",
                    "Content-Type": "application/json",
                    "X-Cfetch-Attestation-Nonce": nonce_hex,
                },
            )
            response = connection.getresponse()
            response_body = response.read()
            self.assertEqual(response.status, 200)
            expected = hashlib.sha512(
                adapter.attestation_message(bytes.fromhex(nonce_hex), body, response_body)
            ).hexdigest()
            self.assertEqual(
                response.getheader("X-Cfetch-Attestation-Signature"), expected
            )
            self.assertEqual(json.loads(response_body)["data"][0]["index"], 0)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_http_requires_bearer_and_nonce(self) -> None:
        server = adapter.AdapterServer(
            ("127.0.0.1", 0), self.service, "a" * 64
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request(
                "POST",
                "/v1/embeddings",
                body=self.request_body(["short"]),
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(connection.getresponse().status, 401)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_readiness_is_bounded_and_names_all_plan_scopes_in_order(self) -> None:
        gpu = replace(
            self.scope,
            scope_id="intel-test-gpu",
            device_class="gpu",
            openvino_device="GPU",
            required_openvino_properties={
                "FULL_DEVICE_NAME": "Test Intel GPU",
                "DEVICE_ARCHITECTURE": "test-gpu-architecture",
                "GPU_UARCH_VERSION": "test-uarch",
                "GPU_DEVICE_ID": "0x0000",
            },
        )
        cpu = replace(
            self.scope,
            scope_id="intel-test-cpu",
            device_class="cpu",
            openvino_device="CPU",
            required_openvino_properties={
                "FULL_DEVICE_NAME": "Test Intel CPU",
                "DEVICE_ARCHITECTURE": "intel64",
            },
        )
        plan_package = replace(
            self.package,
            scopes={
                self.scope.scope_id: self.scope,
                gpu.scope_id: gpu,
                cpu.scope_id: cpu,
            },
        )
        server = adapter.AdapterServer(
            ("127.0.0.1", 0), self.service, "a" * 64
        )
        try:
            ready = adapter.readiness_document(server, plan_package)
            self.assertEqual(ready["schema_version"], 1)
            self.assertRegex(ready["url"], r"^http://127\.0\.0\.1:[1-9][0-9]*/v1$")
            self.assertEqual(
                ready["scope_ids"],
                ["intel-test-npu", "intel-test-gpu", "intel-test-cpu"],
            )
            self.assertLess(len(json.dumps(ready)), 512)
        finally:
            server.server_close()

    def test_scope_initialization_failure_has_the_only_retryable_shape(self) -> None:
        def unavailable(_package, _scope):
            raise RuntimeError("vendor detail must stay on stderr")

        service = adapter.EmbeddingService(
            self.package,
            FakeTokenizer(),
            unavailable,
            {self.scope.scope_id: FakeSigner()},
        )
        server = adapter.AdapterServer(("127.0.0.1", 0), service, "a" * 64)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = self.request_body(["short"])
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=2
            )
            connection.request(
                "POST",
                "/v1/embeddings",
                body=body,
                headers={
                    "Authorization": f"Bearer {'a' * 64}",
                    "Content-Type": "application/json",
                    "X-Cfetch-Attestation-Nonce": "b" * 64,
                },
            )
            response = connection.getresponse()
            response_body = response.read()
            self.assertEqual(response.status, 503)
            self.assertEqual(
                response_body,
                b'{"error":{"code":"scope_unavailable","scope_id":"intel-test-npu",'
                b'"message":"requested admitted scope could not initialize or execute"}}',
            )
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_physical_properties_must_exactly_match_admitted_scope(self) -> None:
        class FakeCore:
            def __init__(self, values):
                self.values = values
                self.calls = []

            def get_property(self, device, name):
                self.calls.append((device, name))
                if name == "SUPPORTED_PROPERTIES":
                    return list(self.values)
                return self.values[name]

        indexed_scope = replace(
            self.scope, required_execution_devices=("NPU.0",)
        )
        matching = FakeCore(dict(indexed_scope.required_openvino_properties))
        adapter.validate_openvino_properties(matching, indexed_scope)
        self.assertEqual(
            matching.calls,
            [("NPU.0", "SUPPORTED_PROPERTIES")]
            + [
                ("NPU.0", name)
                for name in indexed_scope.required_openvino_properties
            ],
        )

        wrong = dict(indexed_scope.required_openvino_properties)
        wrong["NPU_DRIVER_VERSION"] = 99
        with self.assertRaisesRegex(RuntimeError, "did not match the admitted"):
            adapter.validate_openvino_properties(FakeCore(wrong), indexed_scope)

    def test_host_binding_requires_exact_kernel_and_file_bytes(self) -> None:
        with unittest.mock.patch.object(
            adapter.platform, "system", return_value="Linux"
        ), unittest.mock.patch.object(
            adapter.platform, "machine", return_value="x86_64"
        ), unittest.mock.patch.object(
            adapter.platform, "release", return_value="test-kernel"
        ), unittest.mock.patch.object(
            adapter, "_host_file_sha256", return_value="8" * 64
        ):
            adapter.validate_host_binding(self.scope)

        with unittest.mock.patch.object(
            adapter.platform, "system", return_value="Linux"
        ), unittest.mock.patch.object(
            adapter.platform, "machine", return_value="x86_64"
        ), unittest.mock.patch.object(
            adapter.platform, "release", return_value="wrong-kernel"
        ):
            with self.assertRaisesRegex(RuntimeError, "host identity did not match"):
                adapter.validate_host_binding(self.scope)

    def test_host_preflight_uses_typed_live_properties_and_all_static_compiles(
        self,
    ) -> None:
        class BucketModel:
            bucket = 0

            def reshape(self, shapes):
                self.bucket = next(iter(shapes.values()))[1]

        class Model:
            def clone(self):
                return BucketModel()

        class Compiled:
            def __init__(self, execution_devices):
                self.execution_devices = execution_devices

            def get_property(self, name):
                self.assert_property = name
                return self.execution_devices

        class Core:
            available_devices = ["CPU", "GPU.0", "NPU.0"]

            def __init__(self):
                self.properties = dict(self_scope.required_openvino_properties)
                self.compile_calls = []
                self.property_calls = []

            def get_property(self, device, name):
                self.property_calls.append((device, name))
                if name == "SUPPORTED_PROPERTIES":
                    return list(self.properties)
                return self.properties[name]

            def read_model(self, *, model, weights):
                self.graph_paths = (model, weights)
                return Model()

            def compile_model(self, model, device, config):
                self.compile_calls.append((model.bucket, device, config))
                return Compiled(["NPU.0"])

        self_scope = self.scope
        core = Core()
        runtime = {
            "runtime_manifest_sha256": "9" * 64,
            "dependency_versions": {
                "cryptography": "test",
                "numpy": "test",
                "openvino": "test",
                "tokenizers": "test",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            host_file = Path(directory) / "libnpu-test.so"
            host_file.write_bytes(b"driver bytes")
            with patch.object(
                adapter, "HOST_FILE_PREFIXES", (Path(directory),)
            ), patch.object(
                adapter.platform, "system", return_value="Linux"
            ), patch.object(
                adapter.platform, "machine", return_value="x86_64"
            ), patch.object(
                adapter.platform, "release", return_value="test-kernel"
            ):
                result = adapter.collect_host_preflight(
                    self.package.artifact,
                    runtime,
                    "npu",
                    "NPU",
                    {"PERFORMANCE_HINT": "LATENCY"},
                    [host_file],
                    core=core,
                )
                wrong_type = Core()
                wrong_type.properties["NPU_DRIVER_VERSION"] = "1"
                with self.assertRaisesRegex(RuntimeError, "has type str"):
                    adapter.collect_host_preflight(
                        self.package.artifact,
                        runtime,
                        "npu",
                        "NPU",
                        {},
                        [host_file],
                        core=wrong_type,
                    )
                drifting = Core()

                def compile_with_drift(model, device, config):
                    drifting.compile_calls.append((model.bucket, device, config))
                    selected = "NPU.1" if model.bucket == 2048 else "NPU.0"
                    return Compiled([selected])

                drifting.compile_model = compile_with_drift
                with self.assertRaisesRegex(RuntimeError, "changed between"):
                    adapter.collect_host_preflight(
                        self.package.artifact,
                        runtime,
                        "npu",
                        "NPU",
                        {},
                        [host_file],
                        core=drifting,
                    )
                property_drifting = Core()
                property_round = 0

                def property_with_drift(device, name):
                    nonlocal property_round
                    property_drifting.property_calls.append((device, name))
                    if name == "SUPPORTED_PROPERTIES":
                        property_round += 1
                        return list(property_drifting.properties)
                    value = property_drifting.properties[name]
                    if name == "NPU_DRIVER_VERSION" and property_round == 2:
                        return value + 1
                    return value

                property_drifting.get_property = property_with_drift
                with self.assertRaisesRegex(RuntimeError, "properties changed"):
                    adapter.collect_host_preflight(
                        self.package.artifact,
                        runtime,
                        "npu",
                        "NPU",
                        {},
                        [host_file],
                        core=property_drifting,
                    )
                host_before = {
                    "system": "Linux",
                    "machine": "x86_64",
                    "kernel_release": "test-kernel",
                    "files": [{"path": str(host_file), "sha256": "1" * 64},
                              {"path": str(adapter.GOVERNOR_POLICY_PATH), "sha256": "1" * 64}],
                }
                host_after = {
                    **host_before,
                    "files": [{"path": str(host_file), "sha256": "2" * 64},
                              {"path": str(adapter.GOVERNOR_POLICY_PATH), "sha256": "1" * 64}],
                }
                with patch.object(
                    adapter,
                    "_preflight_host_binding",
                    side_effect=(host_before, host_after),
                ), self.assertRaisesRegex(RuntimeError, "host binding changed"):
                    adapter.collect_host_preflight(
                        self.package.artifact,
                        runtime,
                        "npu",
                        "NPU",
                        {},
                        [host_file],
                        core=Core(),
                    )

        self.assertEqual(result["purpose"], adapter.PREFLIGHT_PURPOSE)
        self.assertEqual(result["required_openvino_properties"], core.properties)
        self.assertEqual(result["required_execution_devices"], ["NPU.0"])
        self.assertEqual(result["openvino_property_device"], "NPU.0")
        self.assertEqual(
            result["host_binding_source"],
            "operator-selected-paths-sha256-before-and-after-compilation",
        )
        self.assertEqual(
            [device for device, _name in core.property_calls],
            ["NPU.0"] * (2 * (len(core.properties) + 1)),
        )
        self.assertEqual(
            [row["bucket"] for row in result["bucket_results"]],
            list(adapter.SEQUENCE_BUCKETS),
        )
        self.assertEqual(
            core.compile_calls,
            [
                (bucket, "NPU", {"PERFORMANCE_HINT": "LATENCY"})
                for bucket in adapter.SEQUENCE_BUCKETS
            ],
        )
        raw = adapter.canonical_preflight_output(result)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(
            raw,
            (
                json.dumps(
                    json.loads(raw),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
        )

    def test_host_preflight_rejects_noncanonical_config_and_device_drift(self) -> None:
        self.assertEqual(
            adapter.parse_preflight_compile_config('{"A":1,"B":true}'),
            {"A": 1, "B": True},
        )
        for value in ('{ "A":1}', '{"B":1,"A":2}', '{"A":null}', "\ud800"):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                adapter.parse_preflight_compile_config(value)
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            adapter.collect_host_preflight(
                self.package.artifact,
                {
                    "runtime_manifest_sha256": "9" * 64,
                    "dependency_versions": {
                        "cryptography": "test",
                        "numpy": "test",
                        "openvino": "test",
                        "tokenizers": "test",
                    },
                },
                "npu",
                "GPU",
                {},
                [Path("/usr/lib/libtest.so")],
                core=object(),
            )

    def test_host_preflight_main_uses_raw_runtime_without_package_inventory(self) -> None:
        runtime = {
            "schema_version": 1,
            "runtime_manifest_sha256": "9" * 64,
            "dependency_versions": {
                "cryptography": "test",
                "numpy": "test",
                "openvino": "test",
                "tokenizers": "test",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host_file = root / "driver.so"
            host_file.write_bytes(b"driver")
            output_bytes = io.BytesIO()
            output = io.TextIOWrapper(output_bytes, encoding="utf-8")
            environment = dict(os.environ)
            environment.pop("CFETCH_PACKAGE_INVENTORY_SHA256", None)
            with patch.dict(os.environ, environment, clear=True), patch.object(
                adapter, "runtime_self_check", return_value=runtime
            ) as runtime_check, patch.object(
                adapter, "load_artifact", return_value=self.package.artifact
            ) as artifact_loader, patch.object(
                adapter,
                "collect_host_preflight",
                return_value={"schema_version": 1, "purpose": adapter.PREFLIGHT_PURPOSE},
            ), patch.object(
                adapter, "verify_package_inventory"
            ) as inventory_check, patch.object(
                adapter.sys, "stdout", output
            ):
                result = adapter.main(
                    [
                        "host-preflight",
                        "--runtime-manifest-sha256",
                        "9" * 64,
                        "--artifact-dir",
                        str(root),
                        "--artifact-manifest-sha256",
                        "1" * 64,
                        "--device-class",
                        "npu",
                        "--device",
                        "NPU",
                        "--compile-config-json",
                        "{}",
                        "--host-file",
                        str(host_file),
                    ]
                )
                output.flush()
            self.assertEqual(result, 0)
            self.assertEqual(
                output_bytes.getvalue(),
                (
                    '{"purpose":"physical-probe-scope-config-input-not-admission-evidence",'
                    '"schema_version":1}\n'
                ).encode(),
            )
            self.assertEqual(runtime_check.call_count, 2)
            self.assertEqual(
                runtime_check.call_args_list,
                [unittest.mock.call("9" * 64), unittest.mock.call("9" * 64)],
            )
            self.assertEqual(artifact_loader.call_count, 2)
            inventory_check.assert_not_called()

    def test_runtime_check_uses_raw_runtime_without_package_inventory(self) -> None:
        runtime = {
            "schema_version": 1,
            "runtime_manifest_sha256": "9" * 64,
            "dependency_versions": {
                "cryptography": "test",
                "numpy": "test",
                "openvino": "test",
                "tokenizers": "test",
            },
        }
        environment = dict(os.environ)
        environment.pop("CFETCH_PACKAGE_INVENTORY_SHA256", None)
        output = io.StringIO()
        with patch.dict(os.environ, environment, clear=True), patch.object(
            adapter, "runtime_self_check", return_value=runtime
        ) as runtime_check, patch.object(
            adapter, "verify_package_inventory"
        ) as inventory_check, patch.object(
            adapter.sys, "stdout", output
        ):
            result = adapter.main(["runtime-check"])
        self.assertEqual(result, 0)
        self.assertEqual(
            output.getvalue(), json.dumps(runtime, separators=(",", ":")) + "\n"
        )
        runtime_check.assert_called_once_with(None)
        inventory_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
