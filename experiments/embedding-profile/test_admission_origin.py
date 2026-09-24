"""Dependency-free regression checks for the canonical artifact origin."""

import unittest

from admission_transaction import _registry_entry
from cross_backend_eval import (
    REPORT_BACKEND_BINDING_FIELDS,
    validate_admission_cache_url,
    validate_measurement_evidence_url,
)


class AdmissionOriginTests(unittest.TestCase):
    def test_package_and_registry_share_the_refreshed_admission_identity(self) -> None:
        from packages.openvino.manifest import ADMISSION_POLICY_SHA256 as package_policy
        from profile_identity import ADMISSION_POLICY_SHA256 as registry_policy

        self.assertEqual(package_policy, registry_policy)

    def test_transaction_outputs_are_accepted_only_at_the_canonical_origin(self) -> None:
        digest = "a" * 64
        row = _registry_entry(
            {field: "fixture" for field in REPORT_BACKEND_BINDING_FIELDS},
            digest, digest, digest, "admission-v1",
        )
        for field, suffix, validate in (
            ("admission_cache_url", ".npz", validate_admission_cache_url),
            ("measurement_evidence_url", ".zip", validate_measurement_evidence_url),
        ):
            with self.subTest(field=field):
                expected = (
                    "https://github.com/corbet-libs/cfetch/releases/download/"
                    f"admission-v1/{digest}{suffix}"
                )
                self.assertEqual(row[field], expected)
                validate("fixture", row[field], digest)
                for rejected in (
                    expected.replace("corbet-libs", "corbet-labs"),
                    expected.replace("corbet-libs", "another-owner"),
                    expected.replace(digest, "b" * 64),
                    expected + "?redirect=1",
                ):
                    with self.subTest(url=rejected), self.assertRaisesRegex(
                        ValueError, "content-addressed cfetch GitHub release URL"
                    ):
                        validate("fixture", rejected, digest)


if __name__ == "__main__":
    unittest.main()
