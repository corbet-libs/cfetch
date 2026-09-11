#!/usr/bin/env python3
"""Hosted selection uses Crow's exact commands and never admits private probes."""
import importlib.util
import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("hosted", Path(__file__).with_name("hosted.py"))
hosted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosted)


class HostedSelectionTests(unittest.TestCase):
    def config(self):
        return {"checks": {name: {"kind": "commands", "commands": [["bash", "scripts/ci-check.sh", name]]}
                           for name in hosted.PUBLIC_CHECKS}}

    def test_selects_exact_declared_commands_in_requested_order(self):
        self.assertEqual(hosted.selected_checks("governor, catalog", self.config()), ["governor", "catalog"])

    def test_rejects_private_unknown_empty_and_duplicate_checks_before_work(self):
        for value in ("ort-foundation-cpu", "ort-foundation-build", "catalog,unknown", "", "catalog,", "catalog,catalog"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hosted.selected_checks(value, self.config())

    def test_rejects_command_drift_from_crow(self):
        config = self.config()
        config["checks"]["rust"]["commands"] = [["cargo", "test"]]
        with self.assertRaises(ValueError):
            hosted.selected_checks("rust", config)

    def test_model_free_checks_do_not_request_rust_or_policy_installs(self):
        self.assertEqual(hosted.flags(["catalog", "governor"]),
                         {"rust": False, "profile": False, "licenses": False, "ci_config": False})

    def test_profile_and_licenses_request_only_their_prerequisites(self):
        self.assertEqual(hosted.flags(["profile"]),
                         {"rust": False, "profile": True, "licenses": False, "ci_config": False})
        self.assertTrue(hosted.flags(["licenses"])["rust"])

    def test_failure_receipt_preserves_source_and_never_becomes_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps({"commit": "a" * 40, "checks": ["catalog"], "status": "prepared"}))
            with mock.patch.object(hosted, "receipt_path", return_value=path), mock.patch.dict(os.environ, {"CHECK_OUTCOME": "failure"}), contextlib.redirect_stdout(io.StringIO()):
                hosted.finish()
            self.assertEqual(json.loads(path.read_text()), {"commit": "a" * 40, "checks": ["catalog"], "status": "failure"})

    def test_automatic_mapping_binds_workflow_selection_and_one_job_allocation(self):
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / ".ci/providers.toml").read_text())
        github = config["github"]
        workflow = root / ".github/workflows" / github["workflow"]
        self.assertEqual(github["workflow_sha256"], hashlib.sha256(workflow.read_bytes()).hexdigest())
        self.assertEqual(github["checks"], ["catalog", "governor"])
        self.assertTrue(github["secret_free"])
        self.assertTrue(github["free_eligible"])
        self.assertEqual(github["resources"], {
            "jobs": 1, "test_threads": 1, "minimum_available_mb": 8192, "timeout": 900})
        text = workflow.read_text()
        automatic = text.split("  automatic:\n", 1)[1].split("\n  checks:\n", 1)[0]
        self.assertIn("uses: corbet-labs/ccid/.github/workflows/reusable-check.yml@" + github["reusable_workflow_revision"], automatic)
        self.assertIn("if: github.event.repository.private == false && inputs.request_id != ''", automatic)
        self.assertIn("setup_rust: false", automatic)
        for name, value in {"ci_jobs": 1, "ci_test_threads": 1, "ci_min_available_mb": 8192, "ci_timeout": 900}.items():
            self.assertIn(f"      {name}: {value}\n", automatic)
        for name in ("source_commit", "checks", "request_id", "dependency_snapshot", "tool_revision",
                     "tool_run_id", "tool_asset_id", "tool_archive_sha256", "tool_binary_sha256",
                     "manifest_sha256", "config_sha256", "tool_bootstrap_sha256"):
            self.assertIn(name + ": ${{ inputs." + name + " }}", automatic)
        self.assertNotIn("secrets:", automatic)
        for dependency in config["dependencies"]["files"]:
            self.assertTrue((root / dependency).is_file())

    def test_manual_and_automatic_selected_jobs_are_exclusive(self):
        workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/selected.yml").read_text()
        manual = workflow.split("\n  checks:\n", 1)[1]
        self.assertIn("if: github.event.repository.private == false && inputs.request_id == ''", manual)
        self.assertIn("format('ccid/{0}', inputs.request_id)", workflow)
        self.assertIn("cancel-in-progress: ${{ inputs.request_id == '' }}", workflow)
        self.assertIn("REQUESTED_CHECKS: ${{ inputs.checks }}", manual)


if __name__ == "__main__":
    unittest.main()
