//! The executable release catalog.
//!
//! This file deliberately knows only about artifacts that exist. Future
//! accelerator work belongs in the design until its backend is linked, its
//! build is green, and its archive is produced. That keeps runtime advice,
//! release automation, and package generators from promising different
//! products.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};

const CATALOG_JSON: &str = include_str!("../release/variants.json");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema_version: u32,
    pub variants: Vec<ReleaseVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReleaseVariant {
    pub id: String,
    pub os: String,
    pub arch: String,
    pub runner: String,
    pub target: String,
    pub binary: String,
    pub archive: String,
    pub backend: String,
    pub cargo_features: String,
}

impl Catalog {
    pub fn parse(input: &str) -> anyhow::Result<Self> {
        let catalog: Catalog =
            serde_json::from_str(input).context("parse release/variants.json")?;
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema_version == 1,
            "unsupported variant catalog schema {}",
            self.schema_version
        );
        anyhow::ensure!(!self.variants.is_empty(), "variant catalog is empty");

        let mut ids = BTreeSet::new();
        let mut endpoint_targets = BTreeSet::new();
        for variant in &self.variants {
            anyhow::ensure!(
                ids.insert(&variant.id),
                "duplicate variant id {}",
                variant.id
            );
            anyhow::ensure!(
                variant
                    .id
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_'),
                "variant id {} contains unsupported characters",
                variant.id
            );
            anyhow::ensure!(
                ["linux", "mac", "win"].contains(&variant.os.as_str()),
                "variant {} has unknown OS {}",
                variant.id,
                variant.os
            );
            anyhow::ensure!(
                ["x86_64", "aarch64"].contains(&variant.arch.as_str()),
                "variant {} has unknown architecture {}",
                variant.id,
                variant.arch
            );
            anyhow::ensure!(
                !variant.runner.is_empty(),
                "variant {} has no CI runner",
                variant.id
            );
            anyhow::ensure!(
                !variant.binary.is_empty(),
                "variant {} has no binary name",
                variant.id
            );
            anyhow::ensure!(
                ["tar.gz", "zip"].contains(&variant.archive.as_str()),
                "variant {} has unknown archive format {}",
                variant.id,
                variant.archive
            );
            anyhow::ensure!(
                matches!(
                    variant.backend.as_str(),
                    "endpoint" | "local" | "cpu" | "lexical"
                ),
                "variant {} claims unsupported backend {}",
                variant.id,
                variant.backend
            );
            anyhow::ensure!(
                if variant.backend == "cpu" {
                    variant.cargo_features == "embedded-embeddings"
                } else {
                    variant.cargo_features.is_empty()
                },
                "variant {} has features inconsistent with its backend",
                variant.id
            );
            let expected_name = match variant.backend.as_str() {
                "local" => "-cfetch-local-",
                "cpu" => "-cfetch-cpu-",
                "lexical" => "-cfetch-cli-",
                _ => "-cfetch-remote-",
            };
            anyhow::ensure!(
                variant.id.contains(expected_name),
                "variant {} backend {} must contain {} in its id",
                variant.id,
                variant.backend,
                expected_name
            );
            if matches!(variant.backend.as_str(), "endpoint" | "cpu" | "lexical") {
                anyhow::ensure!(
                    endpoint_targets.insert((variant.os.clone(), variant.arch.clone())),
                    "multiple {} {} variants claim the portable CPU or lexical backend",
                    variant.os,
                    variant.arch
                );
            }
        }
        Ok(())
    }
}

pub fn catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| {
        Catalog::parse(CATALOG_JSON).expect("embedded release variant catalog must be valid")
    })
}

pub fn os_token() -> &'static str {
    if cfg!(target_os = "macos") {
        "mac"
    } else if cfg!(target_os = "windows") {
        "win"
    } else {
        "linux"
    }
}

pub fn arch_token() -> &'static str {
    if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else {
        std::env::consts::ARCH
    }
}

/// The artifact an installer can truthfully offer for this target.
pub fn recommended_release() -> Option<&'static ReleaseVariant> {
    build_id()
        .and_then(|id| catalog().variants.iter().find(|variant| variant.id == id))
        .or_else(|| {
            // A source build has no package identity and therefore cannot
            // assume that a device-specific local payload is installed.
            catalog().variants.iter().find(|variant| {
                variant.os == os_token()
                    && variant.arch == arch_token()
                    && matches!(variant.backend.as_str(), "endpoint" | "cpu" | "lexical")
            })
        })
}

/// Identity injected by the release/package build. A developer build remains
/// unidentified instead of borrowing an artifact identity it may not match.
pub fn build_id() -> Option<&'static str> {
    option_env!("CFETCH_VARIANT")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_catalog_is_valid() {
        catalog().validate().unwrap();
    }

    #[test]
    fn cpu_packages_enable_the_embedded_engine() {
        for variant in &catalog().variants {
            if variant.backend == "cpu" {
                assert_eq!(variant.cargo_features, "embedded-embeddings");
                assert!(variant.id.contains("-cfetch-cpu-"));
            }
        }
        let mut broken = catalog().clone();
        broken.variants[0].cargo_features.clear();
        assert!(broken.validate().is_err());
    }

    #[test]
    fn catalog_refuses_a_backend_that_has_no_package_contract() {
        let fake = CATALOG_JSON.replacen("\"backend\": \"cpu\"", "\"backend\": \"coreml\"", 1);
        assert!(Catalog::parse(&fake)
            .unwrap_err()
            .to_string()
            .contains("unsupported backend"));
    }

    #[test]
    fn catalog_accepts_a_package_local_out_of_process_variant() {
        let mut local = catalog().clone();
        let mut variant = local.variants[0].clone();
        variant.id = "linux-cfetch-local-test-one-x86_64".into();
        variant.backend = "local".into();
        variant.cargo_features.clear();
        local.variants.push(variant.clone());
        variant.id = "linux-cfetch-local-test-two-x86_64".into();
        local.variants.push(variant);
        local.validate().unwrap();
    }

    #[test]
    fn lunar_lake_local_variant_exists_but_does_not_replace_portable_linux() {
        let local = catalog()
            .variants
            .iter()
            .find(|variant| variant.id == "linux-cfetch-local-intel-lunar-lake-x86_64")
            .expect("the first exact local package needs an activation target");
        assert_eq!(local.backend, "local");
        assert_eq!(
            (local.os.as_str(), local.arch.as_str()),
            ("linux", "x86_64")
        );

        let portable = catalog()
            .variants
            .iter()
            .find(|variant| {
                variant.os == "linux"
                    && variant.arch == "x86_64"
                    && matches!(variant.backend.as_str(), "endpoint" | "cpu" | "lexical")
            })
            .expect("the portable CPU or lexical variant remains the default");
        assert_eq!(portable.id, "linux-cfetch-cpu-x86_64");
    }

    #[test]
    fn every_supported_target_has_exactly_one_portable_endpoint_release() {
        for os in ["linux", "mac", "win"] {
            for arch in ["x86_64", "aarch64"] {
                let matches = catalog()
                    .variants
                    .iter()
                    .filter(|variant| {
                        variant.os == os
                            && variant.arch == arch
                            && matches!(variant.backend.as_str(), "endpoint" | "cpu" | "lexical")
                    })
                    .count();
                assert_eq!(matches, 1, "supported target {os}/{arch}");
            }
        }
    }
}
