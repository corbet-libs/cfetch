//! Typed release boundary for target-local inference packages.
//!
//! A release build may select a local package only by its exact
//! `CFETCH_VARIANT` identity. OS and architecture discovery is deliberately
//! not a substitute: two archives for the same target can contain different
//! admitted artifacts. The registry is empty today, so this module preserves
//! endpoint-only behavior while making the first activation fail closed.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use anyhow::Context as _;
use serde::Deserialize;
use serde::de::{self, MapAccess, Visitor};

const REGISTRY_JSON: &str = include_str!("../release/inference-backends.json");
const REQUIRED_SELECTION: &str = "first available admitted scope in NPU, GPU, accelerated CPU order; each signed request and response is bound to the requested scope id";
const REQUIRED_SEQUENCE_BUCKETS: &[usize] = &[32, 64, 128, 257, 512, 1024, 2048];

/// The sibling process that owns target-native runtime initialization.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatcherArtifactV1 {
    pub binary: String,
    pub sha256: String,
}

/// One package-local way to obtain an exact admitted model artifact.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRecipeV1 {
    pub artifact_sha256: String,
    pub install_source: String,
}

/// Archive format of the immutable target-local inference payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum PackageFormatV1 {
    #[serde(rename = "tar.gz")]
    TarGz,
    #[serde(rename = "zip")]
    Zip,
}

impl PackageFormatV1 {
    fn extension(self) -> &'static str {
        match self {
            Self::TarGz => "tar.gz",
            Self::Zip => "zip",
        }
    }
}

/// A validated local package plan for one exact cfetch release archive.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalPackagePlanV1 {
    pub package_id: String,
    pub release_variant_id: String,
    pub os: String,
    pub arch: String,
    pub device_families: Vec<String>,
    pub package_url: String,
    pub package_sha256: String,
    pub package_manifest_sha256: String,
    pub package_format: PackageFormatV1,
    pub dispatcher: DispatcherArtifactV1,
    pub ordered_scope_ids: Vec<String>,
    #[serde(deserialize_with = "deserialize_unique_recipes")]
    pub artifact_recipes: BTreeMap<String, ArtifactRecipeV1>,
    pub selection: String,
    pub remote_fallback: String,

    /// Derived from the one global report shared by the admitted cohort.
    #[serde(skip)]
    pub compatibility_report: String,
    /// Digest of the newest global report carried by every admitted scope.
    #[serde(skip)]
    pub compatibility_report_sha256: String,
}

#[derive(Debug, Deserialize)]
struct RegistryV1 {
    schema_version: u32,
    profile_id: String,
    profile_status: String,
    shared_identity: SharedIdentityV1,
    selection_order: Vec<DeviceClass>,
    cpu_requirement: String,
    remote_policy: String,
    admission: AdmissionIdentityV1,
    local_packages: Vec<LocalPackagePlanV1>,
    admitted_backends: Vec<AdmittedBackendV1>,
}

#[derive(Debug, Deserialize)]
struct SharedIdentityV1 {
    profile_manifest_sha256: String,
}

#[derive(Debug, Deserialize)]
struct AdmissionIdentityV1 {
    policy_manifest_sha256: String,
    implementation_bundle_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "lowercase")]
enum DeviceClass {
    Npu,
    Gpu,
    Cpu,
}

impl DeviceClass {
    const ORDER: [Self; 3] = [Self::Npu, Self::Gpu, Self::Cpu];
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AdmittedBackendV1 {
    profile_manifest_sha256: String,
    admission_policy_sha256: String,
    scope_id: String,
    transport: crate::embedding_profile::ExecutionTransport,
    backend: String,
    runtime: String,
    compiler: String,
    package_target: String,
    artifact_source: String,
    device_class: DeviceClass,
    device: String,
    artifact_sha256: String,
    internal_precision: String,
    placement_evidence_sha256: String,
    supported_max_tokens: usize,
    supported_sequence_buckets: Vec<usize>,
    supported_max_batch_size: usize,
    sequence_capability_evidence_sha256: String,
    performance_evidence_sha256: String,
    admission_cache_url: String,
    admission_cache_sha256: String,
    measurement_evidence_url: String,
    measurement_evidence_sha256: String,
    compatibility_report: String,
    compatibility_report_sha256: String,
    attestation_public_key: String,
    accelerated_placement: bool,
}

struct UniqueRecipesVisitor;

impl<'de> Visitor<'de> for UniqueRecipesVisitor {
    type Value = BTreeMap<String, ArtifactRecipeV1>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an object with unique execution-scope keys")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut recipes = BTreeMap::new();
        while let Some((scope_id, recipe)) = map.next_entry()? {
            if recipes.insert(scope_id, recipe).is_some() {
                return Err(de::Error::custom(
                    "artifact_recipes contains a duplicate scope id",
                ));
            }
        }
        Ok(recipes)
    }
}

fn deserialize_unique_recipes<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, ArtifactRecipeV1>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserializer.deserialize_map(UniqueRecipesVisitor)
}

#[derive(Clone, Copy)]
struct ExpectedIdentity<'a> {
    profile_id: &'a str,
    profile_status: &'a str,
    profile_manifest_sha256: &'a str,
    admission_policy_sha256: &'a str,
}

impl ExpectedIdentity<'static> {
    const fn embedded() -> Self {
        Self {
            profile_id: crate::embedding_profile::PROFILE_ID,
            profile_status: crate::embedding_profile::PROFILE_STATUS,
            profile_manifest_sha256: crate::embedding_profile::PROFILE_MANIFEST_SHA256,
            admission_policy_sha256: crate::embedding_profile::ADMISSION_POLICY_SHA256,
        }
    }
}

/// Return the local package plan for this exact release artifact.
///
/// Developer builds and endpoint-only variants return `None`. A nonempty
/// registry is validated in full before any plan is returned, so an invalid
/// package for another target cannot hide behind the current target selection.
pub fn selected_local_package_plan() -> anyhow::Result<Option<LocalPackagePlanV1>> {
    select_local_package_plan(
        REGISTRY_JSON,
        crate::variant::catalog(),
        crate::variant::build_id(),
        ExpectedIdentity::embedded(),
        Some((crate::variant::os_token(), crate::variant::arch_token())),
    )
}

fn select_local_package_plan(
    registry_json: &str,
    catalog: &crate::variant::Catalog,
    build_variant: Option<&str>,
    expected: ExpectedIdentity<'_>,
    compiled_target: Option<(&str, &str)>,
) -> anyhow::Result<Option<LocalPackagePlanV1>> {
    let mut registry: RegistryV1 = serde_json::from_str(registry_json)
        .context("parse embedded local inference package registry")?;
    validate_registry(&mut registry, catalog, expected)?;

    if registry.local_packages.is_empty() || build_variant.is_none() {
        return Ok(None);
    }
    let build_variant = build_variant.expect("checked above");
    let Some(plan) = registry
        .local_packages
        .into_iter()
        .find(|plan| plan.release_variant_id == build_variant)
    else {
        return Ok(None);
    };

    if let Some((os, arch)) = compiled_target {
        anyhow::ensure!(
            plan.os == os && plan.arch == arch,
            "local package {} targets {}/{}, but this binary targets {}/{}",
            plan.package_id,
            plan.os,
            plan.arch,
            os,
            arch
        );
    }
    Ok(Some(plan))
}

fn validate_registry(
    registry: &mut RegistryV1,
    catalog: &crate::variant::Catalog,
    expected: ExpectedIdentity<'_>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        registry.schema_version == 1,
        "unsupported local inference registry schema {}",
        registry.schema_version
    );
    anyhow::ensure!(
        registry.profile_id == expected.profile_id
            && registry.profile_status == expected.profile_status
            && registry.shared_identity.profile_manifest_sha256 == expected.profile_manifest_sha256
            && registry.admission.policy_manifest_sha256 == expected.admission_policy_sha256
            && registry.admission.implementation_bundle_sha256
                == crate::embedding_profile::ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
        "local inference registry does not match this binary's frozen profile and admission policy"
    );
    anyhow::ensure!(
        registry.selection_order == DeviceClass::ORDER,
        "local inference selection_order must be exactly npu, gpu, cpu"
    );
    anyhow::ensure!(
        registry.cpu_requirement == "accelerated",
        "local inference CPU fallback must require accelerated placement"
    );
    anyhow::ensure!(
        registry.remote_policy == "explicit-only",
        "local inference registry cannot provide an implicit remote fallback"
    );
    anyhow::ensure!(
        registry.local_packages.is_empty() == registry.admitted_backends.is_empty(),
        "local_packages and admitted_backends must activate atomically"
    );
    if registry.local_packages.is_empty() {
        return Ok(());
    }
    anyhow::ensure!(
        registry.profile_status == "active",
        "a nonempty local package registry requires an active embedding profile"
    );

    let admitted = validate_admitted_backends(&registry.admitted_backends, expected)?;
    let report = admitted
        .values()
        .next()
        .map(|scope| {
            (
                scope.compatibility_report.clone(),
                scope.compatibility_report_sha256.clone(),
            )
        })
        .context("a nonempty package registry has no admitted backends")?;

    let mut package_ids = BTreeSet::new();
    let mut release_variant_ids = BTreeSet::new();
    let mut packaged_scope_ids = BTreeSet::new();
    for plan in &mut registry.local_packages {
        validate_package(
            plan,
            catalog,
            &admitted,
            &mut package_ids,
            &mut release_variant_ids,
        )?;
        packaged_scope_ids.extend(plan.ordered_scope_ids.iter().cloned());
        plan.compatibility_report.clone_from(&report.0);
        plan.compatibility_report_sha256.clone_from(&report.1);
    }
    for (scope_id, scope) in admitted {
        anyhow::ensure!(
            scope.transport != crate::embedding_profile::ExecutionTransport::SupervisedLocal
                || packaged_scope_ids.contains(scope_id),
            "admitted supervised-local scope {scope_id} is not delivered by any local package"
        );
    }
    Ok(())
}

fn validate_admitted_backends<'a>(
    scopes: &'a [AdmittedBackendV1],
    expected: ExpectedIdentity<'_>,
) -> anyhow::Result<BTreeMap<&'a str, &'a AdmittedBackendV1>> {
    let mut admitted = BTreeMap::new();
    let mut attestation_keys = BTreeSet::new();
    let mut report_binding: Option<(&str, &str)> = None;

    for scope in scopes {
        validate_scope_id(&scope.scope_id).with_context(|| {
            format!("admitted backend has invalid scope id {:?}", scope.scope_id)
        })?;
        anyhow::ensure!(
            admitted.insert(scope.scope_id.as_str(), scope).is_none(),
            "duplicate admitted scope id {}",
            scope.scope_id
        );
        anyhow::ensure!(
            scope.profile_manifest_sha256 == expected.profile_manifest_sha256
                && scope.admission_policy_sha256 == expected.admission_policy_sha256,
            "admitted scope {} does not match the registry profile and admission policy",
            scope.scope_id
        );
        for (field, value) in [
            ("artifact_sha256", scope.artifact_sha256.as_str()),
            (
                "placement_evidence_sha256",
                scope.placement_evidence_sha256.as_str(),
            ),
            (
                "sequence_capability_evidence_sha256",
                scope.sequence_capability_evidence_sha256.as_str(),
            ),
            (
                "performance_evidence_sha256",
                scope.performance_evidence_sha256.as_str(),
            ),
            (
                "admission_cache_sha256",
                scope.admission_cache_sha256.as_str(),
            ),
            (
                "measurement_evidence_sha256",
                scope.measurement_evidence_sha256.as_str(),
            ),
            (
                "compatibility_report_sha256",
                scope.compatibility_report_sha256.as_str(),
            ),
        ] {
            validate_sha256(value).with_context(|| {
                format!("admitted scope {} has invalid {field}", scope.scope_id)
            })?;
        }
        validate_ed25519_key(&scope.attestation_public_key).with_context(|| {
            format!(
                "admitted scope {} has invalid attestation_public_key",
                scope.scope_id
            )
        })?;
        anyhow::ensure!(
            attestation_keys.insert(scope.attestation_public_key.as_str()),
            "admitted scopes must use unique attestation public keys"
        );
        for (field, value) in [
            ("backend", scope.backend.as_str()),
            ("runtime", scope.runtime.as_str()),
            ("compiler", scope.compiler.as_str()),
            ("package_target", scope.package_target.as_str()),
            ("artifact_source", scope.artifact_source.as_str()),
            ("device", scope.device.as_str()),
            ("internal_precision", scope.internal_precision.as_str()),
            ("admission_cache_url", scope.admission_cache_url.as_str()),
            (
                "measurement_evidence_url",
                scope.measurement_evidence_url.as_str(),
            ),
        ] {
            validate_nonempty(field, value, &scope.scope_id)?;
        }
        anyhow::ensure!(
            scope.accelerated_placement
                && scope.supported_max_tokens == crate::embedding_profile::MAX_TOKENS
                && scope.supported_sequence_buckets == REQUIRED_SEQUENCE_BUCKETS
                && scope.supported_max_batch_size == crate::embedding_profile::MAX_WIRE_BATCH_SIZE,
            "admitted scope {} lacks complete accelerated 2048-token, seven-bucket, batch-64 capability",
            scope.scope_id
        );
        validate_report_reference(
            &scope.compatibility_report,
            &scope.compatibility_report_sha256,
        )
        .with_context(|| {
            format!(
                "admitted scope {} has an invalid compatibility report binding",
                scope.scope_id
            )
        })?;
        let binding = (
            scope.compatibility_report.as_str(),
            scope.compatibility_report_sha256.as_str(),
        );
        if let Some(expected_binding) = report_binding {
            anyhow::ensure!(
                binding == expected_binding,
                "all admitted scopes must reference the same newest compatibility report"
            );
        } else {
            report_binding = Some(binding);
        }
    }
    Ok(admitted)
}

fn validate_package(
    plan: &LocalPackagePlanV1,
    catalog: &crate::variant::Catalog,
    admitted: &BTreeMap<&str, &AdmittedBackendV1>,
    package_ids: &mut BTreeSet<String>,
    release_variant_ids: &mut BTreeSet<String>,
) -> anyhow::Result<()> {
    validate_slug(&plan.package_id, 128)
        .with_context(|| format!("invalid local package id {:?}", plan.package_id))?;
    anyhow::ensure!(
        package_ids.insert(plan.package_id.clone()),
        "duplicate local package id {}",
        plan.package_id
    );
    anyhow::ensure!(
        release_variant_ids.insert(plan.release_variant_id.clone()),
        "release variant {} has more than one local package",
        plan.release_variant_id
    );

    let variant = catalog
        .variants
        .iter()
        .find(|variant| variant.id == plan.release_variant_id)
        .with_context(|| {
            format!(
                "local package {} references unknown release variant {}",
                plan.package_id, plan.release_variant_id
            )
        })?;
    anyhow::ensure!(
        variant.backend == "local",
        "local package {} cannot bind endpoint-only release variant {}",
        plan.package_id,
        plan.release_variant_id
    );
    anyhow::ensure!(
        plan.os == variant.os && plan.arch == variant.arch,
        "local package {} target {}/{} does not match release variant {} target {}/{}",
        plan.package_id,
        plan.os,
        plan.arch,
        plan.release_variant_id,
        variant.os,
        variant.arch
    );
    anyhow::ensure!(
        ["linux", "mac", "win"].contains(&plan.os.as_str())
            && ["x86_64", "aarch64"].contains(&plan.arch.as_str()),
        "local package {} has unsupported target {}/{}",
        plan.package_id,
        plan.os,
        plan.arch
    );
    validate_dispatcher(&plan.dispatcher, &variant.binary, &plan.os).with_context(|| {
        format!(
            "local package {} has an invalid dispatcher",
            plan.package_id
        )
    })?;
    validate_package_archive(plan)
        .with_context(|| format!("local package {} has an invalid archive", plan.package_id))?;
    anyhow::ensure!(
        plan.selection == REQUIRED_SELECTION,
        "local package {} has an unsupported selection policy",
        plan.package_id
    );
    anyhow::ensure!(
        plan.remote_fallback == "none",
        "local package {} must not provide a remote fallback",
        plan.package_id
    );

    anyhow::ensure!(
        !plan.device_families.is_empty(),
        "local package {} has no device families",
        plan.package_id
    );
    let device_families: BTreeSet<_> = plan.device_families.iter().map(String::as_str).collect();
    anyhow::ensure!(
        device_families.len() == plan.device_families.len()
            && plan.device_families.iter().all(|family| {
                !family.is_empty()
                    && family.trim() == family
                    && !family.chars().any(char::is_control)
            }),
        "local package {} device families must be nonempty and unique",
        plan.package_id
    );

    let mut scope_ids = BTreeSet::new();
    let mut classes = Vec::with_capacity(plan.ordered_scope_ids.len());
    let mut scope_devices = BTreeSet::new();
    for scope_id in &plan.ordered_scope_ids {
        anyhow::ensure!(
            scope_ids.insert(scope_id.as_str()),
            "local package {} repeats scope {}",
            plan.package_id,
            scope_id
        );
        let scope = admitted.get(scope_id.as_str()).with_context(|| {
            format!(
                "local package {} references non-admitted scope {}",
                plan.package_id, scope_id
            )
        })?;
        anyhow::ensure!(
            scope.transport == crate::embedding_profile::ExecutionTransport::SupervisedLocal,
            "local package {} scope {} must use supervised-local transport",
            plan.package_id,
            scope_id
        );
        classes.push(scope.device_class);
        scope_devices.insert(scope.device.as_str());
    }
    validate_device_class_order(&classes).with_context(|| {
        format!(
            "local package {} has an invalid scope order",
            plan.package_id
        )
    })?;
    anyhow::ensure!(
        scope_devices == device_families,
        "local package {} device_families must exactly name its admitted scope devices",
        plan.package_id
    );

    let recipe_ids: BTreeSet<_> = plan.artifact_recipes.keys().map(String::as_str).collect();
    anyhow::ensure!(
        recipe_ids == scope_ids,
        "local package {} artifact recipes must be keyed exactly to its ordered scopes",
        plan.package_id
    );
    for (scope_id, recipe) in &plan.artifact_recipes {
        validate_sha256(&recipe.artifact_sha256).with_context(|| {
            format!(
                "local package {} recipe {} has an invalid artifact sha256",
                plan.package_id, scope_id
            )
        })?;
        anyhow::ensure!(
            recipe.artifact_sha256 == admitted[scope_id.as_str()].artifact_sha256,
            "local package {} recipe {} does not match its admitted artifact",
            plan.package_id,
            scope_id
        );
        anyhow::ensure!(
            !recipe.install_source.is_empty()
                && recipe.install_source.trim() == recipe.install_source
                && !recipe.install_source.chars().any(char::is_control),
            "local package {} recipe {} has an invalid install_source",
            plan.package_id,
            scope_id
        );
    }
    Ok(())
}

fn validate_device_class_order(classes: &[DeviceClass]) -> anyhow::Result<()> {
    anyhow::ensure!(!classes.is_empty(), "ordered_scope_ids cannot be empty");
    let mut seen = BTreeSet::new();
    let mut previous = None;
    for class in classes {
        if let Some(previous) = previous {
            anyhow::ensure!(
                *class >= previous,
                "ordered scopes must follow NPU, GPU, accelerated CPU order"
            );
        }
        seen.insert(*class);
        previous = Some(*class);
    }
    let required: BTreeSet<_> = DeviceClass::ORDER.into_iter().collect();
    anyhow::ensure!(
        seen == required,
        "ordered scopes must contain admitted NPU, GPU, and accelerated CPU fallbacks"
    );
    Ok(())
}

fn validate_dispatcher(
    dispatcher: &DispatcherArtifactV1,
    cfetch_binary: &str,
    os: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !dispatcher.binary.is_empty()
            && dispatcher.binary != "."
            && dispatcher.binary != ".."
            && dispatcher.binary.trim() == dispatcher.binary
            && !dispatcher.binary.contains('/')
            && !dispatcher.binary.contains('\\')
            && !dispatcher.binary.chars().any(char::is_control)
            && Path::new(&dispatcher.binary)
                .file_name()
                .and_then(|name| name.to_str())
                == Some(dispatcher.binary.as_str()),
        "dispatcher binary must be a plain sibling basename"
    );
    anyhow::ensure!(
        dispatcher.binary != cfetch_binary,
        "dispatcher binary must be distinct from the cfetch binary"
    );
    anyhow::ensure!(
        (os == "win") == dispatcher.binary.ends_with(".exe"),
        "dispatcher binary extension does not match package OS"
    );
    validate_sha256(&dispatcher.sha256).context("invalid dispatcher sha256")
}

fn validate_package_archive(plan: &LocalPackagePlanV1) -> anyhow::Result<()> {
    validate_sha256(&plan.package_sha256).context("invalid package sha256")?;
    validate_sha256(&plan.package_manifest_sha256)
        .context("invalid package manifest sha256")?;
    let prefix = "https://github.com/corbet-libs/cfetch/releases/download/";
    let relative = plan
        .package_url
        .strip_prefix(prefix)
        .context("package_url must use the cfetch GitHub release origin")?;
    let (tag, filename) = relative
        .split_once('/')
        .context("package_url must contain a release tag and digest-named file")?;
    anyhow::ensure!(
        !tag.is_empty()
            && !tag.chars().any(char::is_control)
            && filename
                == format!(
                    "{}.{}",
                    plan.package_sha256,
                    plan.package_format.extension()
                ),
        "package_url must end in the package digest and declared archive format"
    );
    Ok(())
}

fn validate_report_reference(path: &str, digest: &str) -> anyhow::Result<()> {
    validate_sha256(digest)?;
    anyhow::ensure!(
        path == format!("release/admission/{digest}.json"),
        "report path must be the digest-named release/admission JSON file"
    );
    Ok(())
}

pub(crate) fn validate_scope_id(value: &str) -> anyhow::Result<()> {
    validate_slug(value, 128)
}

fn validate_slug(value: &str, max_len: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.len() <= max_len,
        "slug must contain 1 through {max_len} characters"
    );
    let mut previous_separator = false;
    for character in value.chars() {
        let separator = matches!(character, '.' | '_' | '-');
        anyhow::ensure!(
            character.is_ascii_lowercase() || character.is_ascii_digit() || separator,
            "slug contains an unsupported character"
        );
        anyhow::ensure!(
            !(separator && previous_separator),
            "slug separators must be single"
        );
        previous_separator = separator;
    }
    anyhow::ensure!(
        !previous_separator
            && value.chars().next().is_some_and(
                |character| character.is_ascii_lowercase() || character.is_ascii_digit()
            ),
        "slug must begin and end with a letter or digit"
    );
    Ok(())
}

fn validate_sha256(value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "value must be a 64-character lowercase hexadecimal SHA-256"
    );
    Ok(())
}

fn validate_ed25519_key(value: &str) -> anyhow::Result<()> {
    validate_sha256(value)
}

fn validate_nonempty(field: &str, value: &str, scope_id: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty() && value.trim() == value && !value.chars().any(char::is_control),
        "admitted scope {scope_id} has invalid {field}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    const REPORT_SHA256: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn local_catalog() -> crate::variant::Catalog {
        crate::variant::Catalog {
            schema_version: 1,
            variants: vec![crate::variant::ReleaseVariant {
                id: "linux-cfetch-local-x86_64".to_string(),
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                runner: "local-hardware".to_string(),
                target: String::new(),
                binary: "cfetch".to_string(),
                archive: "tar.gz".to_string(),
                backend: "local".to_string(),
                cargo_features: String::new(),
            }],
        }
    }

    fn identity() -> ExpectedIdentity<'static> {
        ExpectedIdentity {
            profile_id: crate::embedding_profile::PROFILE_ID,
            profile_status: "active",
            profile_manifest_sha256: crate::embedding_profile::PROFILE_MANIFEST_SHA256,
            admission_policy_sha256: crate::embedding_profile::ADMISSION_POLICY_SHA256,
        }
    }

    fn scope(scope_id: &str, device_class: &str, device: &str, digit: char) -> Value {
        let digest = digit.to_string().repeat(64);
        json!({
            "profile_manifest_sha256": crate::embedding_profile::PROFILE_MANIFEST_SHA256,
            "admission_policy_sha256": crate::embedding_profile::ADMISSION_POLICY_SHA256,
            "scope_id": scope_id,
            "transport": "supervised-local",
            "backend": "native-test",
            "runtime": "runtime-1",
            "compiler": "compiler-1",
            "package_target": "linux-x86_64-test",
            "artifact_source": format!("package://{scope_id}"),
            "device_class": device_class,
            "device": device,
            "artifact_sha256": digest,
            "internal_precision": "target-native",
            "placement_evidence_sha256": "b".repeat(64),
            "supported_max_tokens": 2048,
            "supported_sequence_buckets": REQUIRED_SEQUENCE_BUCKETS,
            "supported_max_batch_size": 64,
            "sequence_capability_evidence_sha256": "c".repeat(64),
            "performance_evidence_sha256": "d".repeat(64),
            "admission_cache_url": format!("https://example.invalid/{scope_id}.npz"),
            "admission_cache_sha256": "e".repeat(64),
            "measurement_evidence_url": format!("https://example.invalid/{scope_id}.zip"),
            "measurement_evidence_sha256": "f".repeat(64),
            "compatibility_report": format!("release/admission/{REPORT_SHA256}.json"),
            "compatibility_report_sha256": REPORT_SHA256,
            "attestation_public_key": match device_class {
                "npu" => "1".repeat(64),
                "gpu" => "2".repeat(64),
                _ => "3".repeat(64),
            },
            "accelerated_placement": true
        })
    }

    fn valid_registry() -> Value {
        json!({
            "schema_version": 1,
            "profile_id": crate::embedding_profile::PROFILE_ID,
            "profile_status": "active",
            "shared_identity": {
                "profile_manifest_sha256": crate::embedding_profile::PROFILE_MANIFEST_SHA256
            },
            "selection_order": ["npu", "gpu", "cpu"],
            "cpu_requirement": "accelerated",
            "remote_policy": "explicit-only",
            "admission": {
                "policy_manifest_sha256": crate::embedding_profile::ADMISSION_POLICY_SHA256,
                "implementation_bundle_sha256": crate::embedding_profile::ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
            },
            "local_packages": [{
                "package_id": "linux-intel-lunar-lake-v1",
                "release_variant_id": "linux-cfetch-local-x86_64",
                "os": "linux",
                "arch": "x86_64",
                "device_families": [
                    "intel-lunar-lake-npu",
                    "intel-arc-140v",
                    "x86-64-avx2"
                ],
                "package_url": format!(
                    "https://github.com/corbet-libs/cfetch/releases/download/v1/{}.tar.gz",
                    "7".repeat(64)
                ),
                "package_sha256": "7".repeat(64),
                "package_manifest_sha256": "8".repeat(64),
                "package_format": "tar.gz",
                "dispatcher": {
                    "binary": "cfetch-inference",
                    "sha256": "9".repeat(64)
                },
                "ordered_scope_ids": ["intel-npu", "intel-gpu", "intel-cpu"],
                "artifact_recipes": {
                    "intel-npu": {
                        "artifact_sha256": "4".repeat(64),
                        "install_source": "sibling:artifacts/intel-npu"
                    },
                    "intel-gpu": {
                        "artifact_sha256": "5".repeat(64),
                        "install_source": "sibling:artifacts/intel-gpu"
                    },
                    "intel-cpu": {
                        "artifact_sha256": "6".repeat(64),
                        "install_source": "sibling:artifacts/intel-cpu"
                    }
                },
                "selection": REQUIRED_SELECTION,
                "remote_fallback": "none"
            }],
            "admitted_backends": [
                scope("intel-npu", "npu", "intel-lunar-lake-npu", '4'),
                scope("intel-gpu", "gpu", "intel-arc-140v", '5'),
                scope("intel-cpu", "cpu", "x86-64-avx2", '6')
            ]
        })
    }

    fn select(registry: &Value) -> anyhow::Result<Option<LocalPackagePlanV1>> {
        select_local_package_plan(
            &serde_json::to_string(registry).unwrap(),
            &local_catalog(),
            Some("linux-cfetch-local-x86_64"),
            identity(),
            Some(("linux", "x86_64")),
        )
    }

    #[test]
    fn embedded_empty_registry_selects_no_local_package() {
        assert_eq!(selected_local_package_plan().unwrap(), None);
    }

    #[test]
    fn valid_exact_variant_returns_a_typed_plan() {
        let plan = select(&valid_registry()).unwrap().unwrap();
        assert_eq!(plan.dispatcher.binary, "cfetch-inference");
        assert_eq!(
            plan.ordered_scope_ids,
            ["intel-npu", "intel-gpu", "intel-cpu"]
        );
        assert_eq!(plan.compatibility_report_sha256, REPORT_SHA256);
        assert_eq!(
            plan.compatibility_report,
            format!("release/admission/{REPORT_SHA256}.json")
        );
    }

    #[test]
    fn selection_never_falls_back_by_os_or_arch() {
        let registry = valid_registry();
        let selected = select_local_package_plan(
            &serde_json::to_string(&registry).unwrap(),
            &local_catalog(),
            Some("linux-cfetch-remote-x86_64"),
            identity(),
            Some(("linux", "x86_64")),
        )
        .unwrap();
        assert_eq!(selected, None);
    }

    #[test]
    fn rejects_missing_or_reordered_device_class_fallbacks() {
        let mut missing = valid_registry();
        missing["local_packages"][0]["ordered_scope_ids"] = json!(["intel-npu", "intel-gpu"]);
        missing["local_packages"][0]["artifact_recipes"]
            .as_object_mut()
            .unwrap()
            .remove("intel-cpu");
        missing["local_packages"][0]["device_families"] =
            json!(["intel-lunar-lake-npu", "intel-arc-140v"]);
        assert!(
            format!("{:#}", select(&missing).unwrap_err())
                .contains("NPU, GPU, and accelerated CPU")
        );

        let mut reordered = valid_registry();
        reordered["local_packages"][0]["ordered_scope_ids"] =
            json!(["intel-gpu", "intel-npu", "intel-cpu"]);
        assert!(
            format!("{:#}", select(&reordered).unwrap_err())
                .contains("NPU, GPU, accelerated CPU order")
        );
    }

    #[test]
    fn rejects_incomplete_or_unaccelerated_scope_capability() {
        for (field, replacement) in [
            ("supported_max_tokens", json!(1024)),
            ("supported_sequence_buckets", json!([32, 64, 128])),
            ("supported_max_batch_size", json!(63)),
            ("accelerated_placement", json!(false)),
        ] {
            let mut registry = valid_registry();
            registry["admitted_backends"][0][field] = replacement;
            assert!(
                select(&registry)
                    .unwrap_err()
                    .to_string()
                    .contains("complete accelerated")
            );
        }
    }

    #[test]
    fn rejects_remote_attested_scope_inside_local_package() {
        let mut registry = valid_registry();
        registry["admitted_backends"][0]["transport"] = json!("remote-attested");
        assert!(
            select(&registry)
                .unwrap_err()
                .to_string()
                .contains("must use supervised-local transport")
        );
    }

    #[test]
    fn supervised_local_scopes_must_ship_in_a_package() {
        let mut registry = valid_registry();
        let mut orphan = scope("orphan-npu", "npu", "other-npu", '0');
        orphan["attestation_public_key"] = json!("7".repeat(64));
        registry["admitted_backends"]
            .as_array_mut()
            .unwrap()
            .push(orphan.clone());
        assert!(
            select(&registry)
                .unwrap_err()
                .to_string()
                .contains("not delivered by any local package")
        );

        orphan["transport"] = json!("remote-attested");
        registry["admitted_backends"].as_array_mut().unwrap().pop();
        registry["admitted_backends"]
            .as_array_mut()
            .unwrap()
            .push(orphan);
        select(&registry).unwrap();
    }

    #[test]
    fn rejects_recipe_scope_or_artifact_drift() {
        let mut extra = valid_registry();
        extra["local_packages"][0]["artifact_recipes"]["not-a-scope"] = json!({
            "artifact_sha256": "7".repeat(64),
            "install_source": "sibling:artifacts/extra"
        });
        assert!(
            select(&extra)
                .unwrap_err()
                .to_string()
                .contains("keyed exactly")
        );

        let mut changed = valid_registry();
        changed["local_packages"][0]["artifact_recipes"]["intel-gpu"]["artifact_sha256"] =
            json!("7".repeat(64));
        assert!(
            select(&changed)
                .unwrap_err()
                .to_string()
                .contains("does not match its admitted artifact")
        );
    }

    #[test]
    fn rejects_split_report_generation_and_remote_fallback() {
        let mut split = valid_registry();
        split["admitted_backends"][2]["compatibility_report_sha256"] = json!("8".repeat(64));
        split["admitted_backends"][2]["compatibility_report"] =
            json!(format!("release/admission/{}.json", "8".repeat(64)));
        assert!(
            select(&split)
                .unwrap_err()
                .to_string()
                .contains("same newest compatibility report")
        );

        let mut remote = valid_registry();
        remote["local_packages"][0]["remote_fallback"] = json!("configured-endpoint");
        assert!(
            select(&remote)
                .unwrap_err()
                .to_string()
                .contains("must not provide a remote fallback")
        );
    }

    #[test]
    fn rejects_dispatcher_paths_and_duplicate_package_identity() {
        let mut path = valid_registry();
        path["local_packages"][0]["dispatcher"]["binary"] = json!("bin/cfetch-inference");
        assert!(format!("{:#}", select(&path).unwrap_err()).contains("plain sibling basename"));

        let mut duplicate = valid_registry();
        let second = duplicate["local_packages"][0].clone();
        duplicate["local_packages"]
            .as_array_mut()
            .unwrap()
            .push(second);
        assert!(
            select(&duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate local package id")
        );
    }

    #[test]
    fn rejects_non_content_addressed_package_archive() {
        let mut registry = valid_registry();
        registry["local_packages"][0]["package_url"] =
            json!("https://github.com/corbet-libs/cfetch/releases/download/v1/inference.tar.gz");
        assert!(
            format!("{:#}", select(&registry).unwrap_err())
                .contains("package digest and declared archive format")
        );

        let mut retired = valid_registry();
        retired["local_packages"][0]["package_url"] = json!(format!(
            "https://github.com/corbet-labs/cfetch/releases/download/v1/{}.tar.gz",
            "7".repeat(64)
        ));
        assert!(
            format!("{:#}", select(&retired).unwrap_err()).contains("cfetch GitHub release origin")
        );

        let mut foreign = valid_registry();
        foreign["local_packages"][0]["package_url"] = json!(format!(
            "https://example.invalid/releases/{}.tar.gz",
            "7".repeat(64)
        ));
        assert!(
            format!("{:#}", select(&foreign).unwrap_err()).contains("cfetch GitHub release origin")
        );
    }

    #[test]
    fn rejects_endpoint_variant_binding_and_target_mismatch() {
        let registry = valid_registry();
        let mut endpoint_catalog = local_catalog();
        endpoint_catalog.variants[0].backend = "endpoint".to_string();
        let error = select_local_package_plan(
            &serde_json::to_string(&registry).unwrap(),
            &endpoint_catalog,
            Some("linux-cfetch-local-x86_64"),
            identity(),
            Some(("linux", "x86_64")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("endpoint-only release variant"));

        let mut mismatch = valid_registry();
        mismatch["local_packages"][0]["arch"] = json!("aarch64");
        assert!(
            select(&mismatch)
                .unwrap_err()
                .to_string()
                .contains("does not match")
        );
    }
}
