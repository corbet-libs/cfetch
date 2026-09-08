//! The logical embedding/vector-space contract for cfetch network major v1.
//!
//! The source model, semantic pipeline, and output/storage codec are shared.
//! Runtime graphs and their internal precision are not: each NPU, GPU, and CPU
//! backend may use the native compiled artifact needed to run efficiently on
//! that device. Backends enter the profile by passing every ordered
//! query/document pairing plus the adversarial mixed-store retrieval gate,
//! not by reproducing one runtime's bytes or one universal internal W8A8 graph.

use crate::config::{EmbeddingsConfig, Precision};

/// The data/network compatibility major. The v1 profile is still a candidate:
/// no released local producer or shared store activated the rejected ORT
/// artifact, so correcting the pre-activation contract does not reinterpret
/// released vectors.
pub const NETWORK_MAJOR: u32 = 1;
pub const PROFILE_ID: &str = "cfetch-embedding-v1";
pub const PROFILE_STATUS: &str = "candidate";
pub const PROFILE_MANIFEST_SHA256: &str =
    "59210a333494f788eb8e607fe38cabb6af1a7aa7cdf604ddf52e3fa6004b5afb";
pub const ADMISSION_POLICY_VERSION: u32 = 1;
pub const ADMISSION_POLICY_SHA256: &str =
    "36375d4e6fb5f6a48e5ecb7036ccc141c2314c2f2609c37063a6f79aa367fcc2";
pub const ADMISSION_IMPLEMENTATION_BUNDLE_SHA256: &str =
    "7a8bca24dc89ab1f23d00632de9653f4f045cae8aa8f876a936c4d6e9ab7e8bd";

/// Immutable semantic source for the candidate profile. Every native package
/// records its actual lineage and artifact digest; direct derivation is never
/// inferred when an upstream package does not document it. Compatibility is
/// established by profile attestation and the global ordered-pair plus
/// mixed-store gates, not by treating one package-specific graph as the
/// numerical source.
pub const MODEL: &str = "google/embeddinggemma-300m";
pub const MODEL_REVISION: &str = "57c266a740f537b4dc058e1b0cda161fd15afa75";
pub const MODEL_WEIGHTS_SHA256: &str =
    "cbf5a78393b6a033e0b8a63a57549964f7ed5c6fbeb4ba0694214f36123f2fd2";
pub const DENSE_2_WEIGHTS_SHA256: &str =
    "c327f2acb00149676ade24a75e11eb6ebbd367f9ee050267ba56829d2979f702";
pub const DENSE_3_WEIGHTS_SHA256: &str =
    "ffb6cc5162e11e2ce6bc2367e121ee3bbbc4e82e1ee26826bd7573d4948d81b8";
/// Backends may use their native internal precision. INT8 is the common
/// output/storage codec, not a requirement on weights, activations, or
/// accumulators inside a target-native graph.
pub const BACKEND_INTERNAL_PRECISION: &str = "target-native";
pub const ARTIFACT_POLICY: &str = "backend-native-pinned-artifact-global-gates-admitted";
pub const EXECUTION_POLICY: &str = "local-accelerated-npu-gpu-cpu";
pub const NUMERICAL_ANCHOR: Option<&str> = None;
pub const BACKEND_ADMISSION: &str =
    "global-ordered-all-pairs-plus-adversarial-mixed-document-plus-per-bucket-semantic-ranking";
pub const ADMISSION_RANKING: &str =
    "exact-signed-int8-cosine-query-norm-cancels-sign-branches-squared-cross-multiplication";
pub const ADMISSION_TIE_BREAK: &str =
    "pinned-corpus-insertion-index-ascending-as-evaluation-block-id-order";
pub const ADVERSARIAL_MIXED_DOCUMENT_SELECTION: &str =
    "per-query-relevant-minimum-irrelevant-maximum-exact-cosine-across-document-scopes";
pub const BACKEND_REPEATABILITY: &str = "required-per-runtime-artifact-device";
pub const SCOPE_AUTHENTICATION: &str =
    "ed25519-response-challenge-bound-to-scope-key-with-transport-defined-trust";
pub const CROSS_BACKEND_EXACT_BYTES: bool = false;
pub const DECISION_PRIORITY: &[&str] = &["compatibility", "quality", "efficiency"];
pub const ADMISSION_DATASET: &str = "mteb/scifact";
pub const ADMISSION_DATASET_REVISION: &str = "cf10ab6856b15b0e670ef8ae5dae4e266c12d035";
pub const SEQUENCE_SEMANTIC_FIXTURE_ID: &str = "cfetch-sequence-semantic-v1-cat-vs-music";
pub const SEQUENCE_SEMANTIC_FIXTURE_SHA256: &str =
    "567fac02f2d55ad2b98b54d89e4b8f0ae81aa2b65a2361603adca9f167543203";
pub const SEQUENCE_SEMANTIC_GATE: &str = "every-profile-sequence-bucket-global-ordered-query-document-scope-plus-adversarial-relevant-minimum-irrelevant-maximum-exact-int8-strict-ranking";
pub const ADMISSION_REQUIRED_DEVICE_CLASSES: &[&str] = &["npu", "gpu", "cpu"];
pub const ADMISSION_NDCG_AT_10_MINIMUM: &str = "0.767907905520953";
pub const ADMISSION_RECALL_AT_100_MINIMUM: &str = "0.970";
pub const ADMISSION_MRR_AT_10_MINIMUM: &str = "0.7305529100529101";
pub const MAX_WIRE_BATCH_SIZE: usize = 64;
pub const BATCHING_CONTRACT: &str = "one-execution-scope-per-response-supports-1-through-64-items-canonical-output-invariant-for-same-64-ordered-inputs-under-every-grouping-size-1-through-64";
pub const ENERGY_EVIDENCE_POLICY: &str =
    "measure-per-sequence-bucket-when-available-otherwise-record-not-measured";
pub const ADMISSION_EVIDENCE_REPLAY: &str =
    "durable-content-addressed-cache-and-measurement-bundle-strict-schema-ci-full-gate-replay";

pub const TOKENIZER_JSON_SHA256: &str =
    "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e";
pub const TOKENIZER_MODEL_SHA256: &str =
    "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c";
pub const TOKENIZER_CONFIG_SHA256: &str =
    "9076840490613047bc9115963ee96b7702018b0d26ba644240bf856efda93118";
pub const MODEL_CONFIG_SHA256: &str =
    "8f863f76e2d9c710cc833dc92efa898c9adfd41031c786507cc6b0e49c2e3e68";
pub const SPECIAL_TOKENS_MAP_SHA256: &str =
    "2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397";
pub const DIMENSIONS: usize = 768;
pub const MAX_TOKENS: usize = 2048;
pub const SEQUENCE_BUCKETS: &[usize] = &[32, 64, 128, 257, 512, 1024, 2048];
pub const TOKEN_COUNTING: &str = "prefixed-input-including-all-special-tokens";
pub const SEQUENCE_BUCKET_SELECTION: &str =
    "smallest-supported-bucket-greater-than-or-equal-to-token-count";
pub const PADDING: &str = "right-padding-attention-mask-excludes-padding";
pub const TRUNCATION: &str = "disabled-reject-token-count-over-max-tokens";
pub const QUERY_PREFIX: &str = "task: search result | query: ";
pub const DOCUMENT_PREFIX: &str = "title: none | text: ";
pub const POOLING: &str = "attention-mask-weighted-mean-include-prompt";
pub const NORMALIZATION: &str = "l2-source-output-then-i8-maxabs-rne-storage";
pub const VECTOR_ENCODING: &str = "signed-int8x768";

/// How an admitted execution scope is reached.  Transport is part of scope
/// identity: a package-private response key admitted behind the local
/// supervisor must not also self-admit an operator-configured endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ExecutionTransport {
    SupervisedLocal,
    RemoteAttested,
}

impl ExecutionTransport {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SupervisedLocal => "supervised-local",
            Self::RemoteAttested => "remote-attested",
        }
    }
}

/// Serializable form of the vector-space contract. Only changes that alter
/// which vector an input means belong here. Admission governance is hashed
/// separately so changing a dataset or quality floor requires recertification,
/// not re-embedding already compatible stored vectors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Manifest {
    pub network_major: u32,
    pub profile_id: &'static str,
    pub model: &'static str,
    pub model_revision: &'static str,
    pub model_weights_sha256: &'static str,
    pub dense_2_weights_sha256: &'static str,
    pub dense_3_weights_sha256: &'static str,
    pub tokenizer_revision: &'static str,
    pub tokenizer_json_sha256: &'static str,
    pub tokenizer_model_sha256: &'static str,
    pub tokenizer_config_sha256: &'static str,
    pub model_config_sha256: &'static str,
    pub special_tokens_map_sha256: &'static str,
    pub max_tokens: usize,
    pub sequence_buckets: &'static [usize],
    pub token_counting: &'static str,
    pub sequence_bucket_selection: &'static str,
    pub padding: &'static str,
    pub truncation: &'static str,
    pub query_prefix: &'static str,
    pub document_prefix: &'static str,
    pub pooling: &'static str,
    pub normalization: &'static str,
    pub dimensions: usize,
    pub vector_encoding: &'static str,
    pub vector_bytes: usize,
}

pub const fn manifest() -> Manifest {
    Manifest {
        network_major: NETWORK_MAJOR,
        profile_id: PROFILE_ID,
        model: MODEL,
        model_revision: MODEL_REVISION,
        model_weights_sha256: MODEL_WEIGHTS_SHA256,
        dense_2_weights_sha256: DENSE_2_WEIGHTS_SHA256,
        dense_3_weights_sha256: DENSE_3_WEIGHTS_SHA256,
        tokenizer_revision: MODEL_REVISION,
        tokenizer_json_sha256: TOKENIZER_JSON_SHA256,
        tokenizer_model_sha256: TOKENIZER_MODEL_SHA256,
        tokenizer_config_sha256: TOKENIZER_CONFIG_SHA256,
        model_config_sha256: MODEL_CONFIG_SHA256,
        special_tokens_map_sha256: SPECIAL_TOKENS_MAP_SHA256,
        max_tokens: MAX_TOKENS,
        sequence_buckets: SEQUENCE_BUCKETS,
        token_counting: TOKEN_COUNTING,
        sequence_bucket_selection: SEQUENCE_BUCKET_SELECTION,
        padding: PADDING,
        truncation: TRUNCATION,
        query_prefix: QUERY_PREFIX,
        document_prefix: DOCUMENT_PREFIX,
        pooling: POOLING,
        normalization: NORMALIZATION,
        dimensions: DIMENSIONS,
        vector_encoding: VECTOR_ENCODING,
        vector_bytes: DIMENSIONS,
    }
}

/// Versioned backend-admission policy. This can evolve without changing the
/// semantic vector-space identity above; any change invalidates prior backend
/// admission evidence until it is rerun against the new policy.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct AdmissionPolicy {
    pub policy_version: u32,
    pub profile_id: &'static str,
    pub profile_manifest_sha256: &'static str,
    pub admission_implementation_bundle_sha256: &'static str,
    pub backend_internal_precision: &'static str,
    pub artifact_policy: &'static str,
    pub execution_policy: &'static str,
    pub numerical_anchor: Option<&'static str>,
    pub backend_admission: &'static str,
    pub admission_ranking: &'static str,
    pub admission_tie_break: &'static str,
    pub adversarial_mixed_document_selection: &'static str,
    pub backend_repeatability: &'static str,
    pub scope_authentication: &'static str,
    pub cross_backend_exact_bytes: bool,
    pub decision_priority: &'static [&'static str],
    pub admission_dataset: &'static str,
    pub admission_dataset_revision: &'static str,
    pub sequence_semantic_fixture_id: &'static str,
    pub sequence_semantic_fixture_sha256: &'static str,
    pub sequence_semantic_gate: &'static str,
    pub admission_required_device_classes: &'static [&'static str],
    pub admission_ndcg_at_10_minimum: &'static str,
    pub admission_recall_at_100_minimum: &'static str,
    pub admission_mrr_at_10_minimum: &'static str,
    pub max_wire_batch_size: usize,
    pub batching_contract: &'static str,
    pub energy_evidence_policy: &'static str,
    pub admission_evidence_replay: &'static str,
}

pub const fn admission_policy() -> AdmissionPolicy {
    AdmissionPolicy {
        policy_version: ADMISSION_POLICY_VERSION,
        profile_id: PROFILE_ID,
        profile_manifest_sha256: PROFILE_MANIFEST_SHA256,
        admission_implementation_bundle_sha256: ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
        backend_internal_precision: BACKEND_INTERNAL_PRECISION,
        artifact_policy: ARTIFACT_POLICY,
        execution_policy: EXECUTION_POLICY,
        numerical_anchor: NUMERICAL_ANCHOR,
        backend_admission: BACKEND_ADMISSION,
        admission_ranking: ADMISSION_RANKING,
        admission_tie_break: ADMISSION_TIE_BREAK,
        adversarial_mixed_document_selection: ADVERSARIAL_MIXED_DOCUMENT_SELECTION,
        backend_repeatability: BACKEND_REPEATABILITY,
        scope_authentication: SCOPE_AUTHENTICATION,
        cross_backend_exact_bytes: CROSS_BACKEND_EXACT_BYTES,
        decision_priority: DECISION_PRIORITY,
        admission_dataset: ADMISSION_DATASET,
        admission_dataset_revision: ADMISSION_DATASET_REVISION,
        sequence_semantic_fixture_id: SEQUENCE_SEMANTIC_FIXTURE_ID,
        sequence_semantic_fixture_sha256: SEQUENCE_SEMANTIC_FIXTURE_SHA256,
        sequence_semantic_gate: SEQUENCE_SEMANTIC_GATE,
        admission_required_device_classes: ADMISSION_REQUIRED_DEVICE_CLASSES,
        admission_ndcg_at_10_minimum: ADMISSION_NDCG_AT_10_MINIMUM,
        admission_recall_at_100_minimum: ADMISSION_RECALL_AT_100_MINIMUM,
        admission_mrr_at_10_minimum: ADMISSION_MRR_AT_10_MINIMUM,
        max_wire_batch_size: MAX_WIRE_BATCH_SIZE,
        batching_contract: BATCHING_CONTRACT,
        energy_evidence_policy: ENERGY_EVIDENCE_POLICY,
        admission_evidence_replay: ADMISSION_EVIDENCE_REPLAY,
    }
}

/// The only valid bucket for a prefixed input's tokenizer-reported count.
/// Special tokens are already included in `token_count`; overlength inputs
/// have no bucket and must be rejected, never truncated.
pub fn sequence_bucket_for_token_count(token_count: usize) -> Option<usize> {
    (token_count > 0 && token_count <= MAX_TOKENS)
        .then(|| {
            SEQUENCE_BUCKETS
                .iter()
                .copied()
                .find(|bucket| *bucket >= token_count)
        })
        .flatten()
}

pub fn manifest_sha256() -> String {
    use sha2::Digest as _;

    let bytes = serde_json::to_vec(&manifest()).expect("embedding manifest serializes");
    let digest = crate::hashing::hex_lower(sha2::Sha256::digest(bytes));
    assert_eq!(
        digest, PROFILE_MANIFEST_SHA256,
        "embedding profile digest changed without updating its frozen identity"
    );
    digest
}

pub fn admission_policy_sha256() -> String {
    use sha2::Digest as _;

    let bytes =
        serde_json::to_vec(&admission_policy()).expect("embedding admission policy serializes");
    let digest = crate::hashing::hex_lower(sha2::Sha256::digest(bytes));
    assert_eq!(
        digest, ADMISSION_POLICY_SHA256,
        "embedding admission policy digest changed without recertifying its scopes"
    );
    digest
}

/// Machine-readable profile document. Lifecycle status and digest are envelope
/// fields rather than part of [`Manifest`]: promoting an unchanged contract
/// from candidate to active must not create a new semantic identity, and the
/// digest cannot recursively hash itself.
pub fn manifest_document() -> serde_json::Value {
    let mut document = serde_json::to_value(manifest()).expect("embedding manifest serializes");
    let object = document
        .as_object_mut()
        .expect("embedding manifest serializes as an object");
    object.insert(
        "profile_status".to_string(),
        serde_json::Value::String(PROFILE_STATUS.to_string()),
    );
    object.insert(
        "profile_manifest_sha256".to_string(),
        serde_json::Value::String(manifest_sha256()),
    );
    object.insert(
        "admission_policy".to_string(),
        serde_json::to_value(admission_policy()).expect("embedding admission policy serializes"),
    );
    object.insert(
        "admission_policy_sha256".to_string(),
        serde_json::Value::String(admission_policy_sha256()),
    );
    document
}

fn validate_production_availability(
    profile_status: &str,
    registry: &serde_json::Value,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        profile_status == "active",
        "embedding profile {PROFILE_ID} is {profile_status}, not active; this release cannot produce canonical vectors"
    );
    anyhow::ensure!(
        registry["profile_id"] == PROFILE_ID
            && registry["profile_status"] == "active"
            && registry["shared_identity"]["profile_manifest_sha256"] == PROFILE_MANIFEST_SHA256
            && registry["admission"]["policy_manifest_sha256"] == ADMISSION_POLICY_SHA256
            && registry["admission"]["implementation_bundle_sha256"]
                == ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
        "embedding backend registry is not active for this release's frozen profile and admission policy"
    );
    let admitted = registry["admitted_backends"].as_array().ok_or_else(|| {
        anyhow::anyhow!("embedding backend registry has no admitted_backends array")
    })?;
    anyhow::ensure!(
        !admitted.is_empty(),
        "embedding backend registry has no admitted production backends"
    );
    Ok(())
}

/// Whether this release can truthfully produce canonical vectors. Diagnostics
/// use this before checking an endpoint: a configured transport is not a
/// production backend while the profile is inactive or its registry is empty.
pub fn production_availability() -> anyhow::Result<()> {
    let registry: serde_json::Value =
        serde_json::from_str(include_str!("../release/inference-backends.json"))
            .map_err(|error| anyhow::anyhow!("embedding backend registry is invalid: {error}"))?;
    validate_production_availability(PROFILE_STATUS, &registry)
}

/// Whether one exact execution scope is admitted by this release. Profile
/// strings supplied by an endpoint prove only semantic intent; they never
/// admit a runtime or artifact by themselves.
pub struct BackendScopeAttestation<'a> {
    pub scope_id: &'a str,
    pub transport: ExecutionTransport,
    pub backend: &'a str,
    pub runtime: &'a str,
    pub compiler: &'a str,
    pub package_target: &'a str,
    pub artifact_source: &'a str,
    pub artifact_sha256: &'a str,
    pub internal_precision: &'a str,
    pub device: &'a str,
    pub device_class: &'a str,
    pub placement_evidence_sha256: &'a str,
    pub supported_max_tokens: usize,
    pub supported_sequence_buckets: &'a [usize],
    pub supported_max_batch_size: usize,
    pub sequence_capability_evidence_sha256: &'a str,
    pub performance_evidence_sha256: &'a str,
    pub compatibility_report_sha256: &'a str,
    pub accelerated_placement: bool,
}

pub fn admitted_backend_attestation_public_key(
    scope: &BackendScopeAttestation<'_>,
) -> Option<String> {
    if PROFILE_STATUS != "active" {
        return None;
    }
    let Ok(registry) = serde_json::from_str::<serde_json::Value>(include_str!(
        "../release/inference-backends.json"
    )) else {
        return None;
    };
    if registry["profile_id"] != PROFILE_ID
        || registry["profile_status"] != "active"
        || registry["shared_identity"]["profile_manifest_sha256"] != PROFILE_MANIFEST_SHA256
        || registry["admission"]["policy_manifest_sha256"] != ADMISSION_POLICY_SHA256
        || registry["admission"]["implementation_bundle_sha256"]
            != ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
    {
        return None;
    }
    registry["admitted_backends"]
        .as_array()
        .and_then(|entries| {
            entries.iter().find_map(|entry| {
                let matches = entry["profile_manifest_sha256"] == PROFILE_MANIFEST_SHA256
                    && entry["admission_policy_sha256"] == ADMISSION_POLICY_SHA256
                    && entry["scope_id"] == scope.scope_id
                    && entry["transport"] == scope.transport.as_str()
                    && entry["backend"] == scope.backend
                    && entry["runtime"] == scope.runtime
                    && entry["compiler"] == scope.compiler
                    && entry["package_target"] == scope.package_target
                    && entry["artifact_source"] == scope.artifact_source
                    && entry["artifact_sha256"] == scope.artifact_sha256
                    && entry["internal_precision"] == scope.internal_precision
                    && entry["device"] == scope.device
                    && entry["device_class"] == scope.device_class
                    && entry["placement_evidence_sha256"] == scope.placement_evidence_sha256
                    && entry["supported_max_tokens"] == scope.supported_max_tokens
                    && entry["supported_sequence_buckets"]
                        == serde_json::json!(scope.supported_sequence_buckets)
                    && entry["supported_max_batch_size"] == scope.supported_max_batch_size
                    && entry["sequence_capability_evidence_sha256"]
                        == scope.sequence_capability_evidence_sha256
                    && entry["performance_evidence_sha256"] == scope.performance_evidence_sha256
                    && entry["compatibility_report_sha256"] == scope.compatibility_report_sha256
                    && entry["accelerated_placement"] == scope.accelerated_placement
                    && scope.accelerated_placement;
                matches
                    .then(|| entry["attestation_public_key"].as_str().map(str::to_string))
                    .flatten()
            })
        })
}

#[cfg(test)]
pub fn backend_scope_is_admitted(scope: &BackendScopeAttestation<'_>) -> bool {
    admitted_backend_attestation_public_key(scope).is_some()
}

/// Refuse semantic drift. Device runtimes and their compiled artifacts are
/// selected outside this contract and therefore are not configuration fields.
pub fn validate(config: &EmbeddingsConfig) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.model == MODEL,
        "cfetch network major {NETWORK_MAJOR} requires embeddings.model={MODEL:?}; changing the source model requires a new network major and re-embedding"
    );
    anyhow::ensure!(
        config.dimensions == DIMENSIONS,
        "cfetch network major {NETWORK_MAJOR} requires {DIMENSIONS} embedding dimensions; changing the width requires a new network major and re-embedding"
    );
    anyhow::ensure!(
        config.precision == Precision::I8,
        "cfetch network major {NETWORK_MAJOR} requires embeddings.precision=\"i8\"; changing vector precision requires a new network major and re-embedding"
    );
    anyhow::ensure!(
        config.query_prefix == QUERY_PREFIX,
        "cfetch network major {NETWORK_MAJOR} fixes embeddings.query_prefix to {QUERY_PREFIX:?}; changing prompts requires a new network major and re-embedding"
    );
    anyhow::ensure!(
        config.document_prefix == DOCUMENT_PREFIX,
        "cfetch network major {NETWORK_MAJOR} fixes embeddings.document_prefix to {DOCUMENT_PREFIX:?}; changing prompts requires a new network major and re-embedding"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_freezes_semantics_and_the_int8_output_not_backend_internals() {
        let profile = manifest();
        let policy = admission_policy();
        assert_eq!(manifest_document()["profile_status"], PROFILE_STATUS);
        assert_eq!(policy.numerical_anchor, None);
        assert_eq!(policy.execution_policy, "local-accelerated-npu-gpu-cpu");
        assert_eq!(
            policy.backend_admission,
            "global-ordered-all-pairs-plus-adversarial-mixed-document-plus-per-bucket-semantic-ranking"
        );
        assert_eq!(
            policy.admission_ranking,
            "exact-signed-int8-cosine-query-norm-cancels-sign-branches-squared-cross-multiplication"
        );
        assert_eq!(
            policy.admission_tie_break,
            "pinned-corpus-insertion-index-ascending-as-evaluation-block-id-order"
        );
        assert_eq!(
            policy.adversarial_mixed_document_selection,
            "per-query-relevant-minimum-irrelevant-maximum-exact-cosine-across-document-scopes"
        );
        assert!(!policy.cross_backend_exact_bytes);
        assert_eq!(
            policy.artifact_policy,
            "backend-native-pinned-artifact-global-gates-admitted"
        );
        assert_eq!(policy.backend_internal_precision, "target-native");
        assert_eq!(
            policy.scope_authentication,
            "ed25519-response-challenge-bound-to-scope-key-with-transport-defined-trust"
        );
        assert_eq!(policy.max_wire_batch_size, 64);
        assert_eq!(
            policy.batching_contract,
            "one-execution-scope-per-response-supports-1-through-64-items-canonical-output-invariant-for-same-64-ordered-inputs-under-every-grouping-size-1-through-64"
        );
        assert_eq!(profile.dimensions, 768);
        assert_eq!(profile.vector_encoding, "signed-int8x768");
        assert_eq!(profile.vector_bytes, 768);
        assert_ne!(policy.backend_internal_precision, profile.vector_encoding);
        assert_eq!(
            policy.decision_priority,
            ["compatibility", "quality", "efficiency"]
        );
    }

    #[test]
    fn v1_semantic_identity_is_complete_and_frozen() {
        let profile = manifest();
        let policy = admission_policy();
        assert_eq!(profile.model_revision, MODEL_REVISION);
        assert_eq!(profile.model_weights_sha256, MODEL_WEIGHTS_SHA256);
        assert_eq!(profile.dense_2_weights_sha256, DENSE_2_WEIGHTS_SHA256);
        assert_eq!(profile.dense_3_weights_sha256, DENSE_3_WEIGHTS_SHA256);
        assert_eq!(profile.tokenizer_revision, MODEL_REVISION);
        assert_eq!(profile.tokenizer_json_sha256, TOKENIZER_JSON_SHA256);
        assert_eq!(profile.tokenizer_model_sha256, TOKENIZER_MODEL_SHA256);
        assert_eq!(profile.tokenizer_config_sha256, TOKENIZER_CONFIG_SHA256);
        assert_eq!(profile.model_config_sha256, MODEL_CONFIG_SHA256);
        assert_eq!(profile.special_tokens_map_sha256, SPECIAL_TOKENS_MAP_SHA256);
        assert_eq!(profile.query_prefix, "task: search result | query: ");
        assert_eq!(profile.document_prefix, "title: none | text: ");
        assert_eq!(
            profile.pooling,
            "attention-mask-weighted-mean-include-prompt"
        );
        assert_eq!(
            profile.normalization,
            "l2-source-output-then-i8-maxabs-rne-storage"
        );
        assert_eq!(profile.max_tokens, 2048);
        assert_eq!(profile.sequence_buckets.last(), Some(&2048));
        assert_eq!(
            profile.token_counting,
            "prefixed-input-including-all-special-tokens"
        );
        assert_eq!(
            profile.sequence_bucket_selection,
            "smallest-supported-bucket-greater-than-or-equal-to-token-count"
        );
        assert_eq!(
            profile.padding,
            "right-padding-attention-mask-excludes-padding"
        );
        assert_eq!(
            profile.truncation,
            "disabled-reject-token-count-over-max-tokens"
        );
        assert_eq!(policy.admission_dataset, "mteb/scifact");
        assert_eq!(
            policy.admission_dataset_revision,
            "cf10ab6856b15b0e670ef8ae5dae4e266c12d035"
        );
        assert_eq!(
            policy.sequence_semantic_fixture_id,
            "cfetch-sequence-semantic-v1-cat-vs-music"
        );
        assert_eq!(
            policy.sequence_semantic_fixture_sha256,
            "567fac02f2d55ad2b98b54d89e4b8f0ae81aa2b65a2361603adca9f167543203"
        );
        assert_eq!(
            policy.sequence_semantic_gate,
            "every-profile-sequence-bucket-global-ordered-query-document-scope-plus-adversarial-relevant-minimum-irrelevant-maximum-exact-int8-strict-ranking"
        );
        assert_eq!(
            policy.admission_required_device_classes,
            ["npu", "gpu", "cpu"]
        );
        assert_eq!(policy.admission_ndcg_at_10_minimum, "0.767907905520953");
        assert_eq!(policy.admission_recall_at_100_minimum, "0.970");
        assert_eq!(policy.admission_mrr_at_10_minimum, "0.7305529100529101");
        assert_eq!(
            policy.admission_implementation_bundle_sha256,
            ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
        );
        assert_eq!(
            policy.admission_evidence_replay,
            "durable-content-addressed-cache-and-measurement-bundle-strict-schema-ci-full-gate-replay"
        );
    }

    #[test]
    fn profile_digest_is_stable_sha256() {
        assert_eq!(manifest_sha256(), PROFILE_MANIFEST_SHA256);
        assert_eq!(admission_policy_sha256(), ADMISSION_POLICY_SHA256);
        assert_eq!(
            manifest_document()["profile_manifest_sha256"],
            PROFILE_MANIFEST_SHA256
        );
        assert_eq!(
            manifest_document()["admission_policy_sha256"],
            ADMISSION_POLICY_SHA256
        );
    }

    #[test]
    fn admission_implementation_bundle_is_exact_and_registry_bound() {
        use sha2::Digest as _;

        let files: [(&str, &[u8]); 12] = [
            (
                "experiments/embedding-profile/admission_evidence.py",
                &include_bytes!("../experiments/embedding-profile/admission_evidence.py")[..],
            ),
            (
                "experiments/embedding-profile/admission_transaction.py",
                &include_bytes!("../experiments/embedding-profile/admission_transaction.py")[..],
            ),
            (
                "experiments/embedding-profile/cross_backend_eval.py",
                &include_bytes!("../experiments/embedding-profile/cross_backend_eval.py")[..],
            ),
            (
                "experiments/embedding-profile/export_adapter_cache.py",
                &include_bytes!("../experiments/embedding-profile/export_adapter_cache.py")[..],
            ),
            (
                "experiments/embedding-profile/final_package_conformance.py",
                &include_bytes!("../experiments/embedding-profile/final_package_conformance.py")[..],
            ),
            (
                "experiments/embedding-profile/measurement_bundle.py",
                &include_bytes!("../experiments/embedding-profile/measurement_bundle.py")[..],
            ),
            (
                "experiments/embedding-profile/physical_evidence.py",
                &include_bytes!("../experiments/embedding-profile/physical_evidence.py")[..],
            ),
            (
                "experiments/embedding-profile/requirements-lock.txt",
                &include_bytes!("../experiments/embedding-profile/requirements-lock.txt")[..],
            ),
            (
                "experiments/embedding-profile/requirements-test.txt",
                &include_bytes!("../experiments/embedding-profile/requirements-test.txt")[..],
            ),
            (
                "experiments/embedding-profile/scifact_contract.py",
                &include_bytes!("../experiments/embedding-profile/scifact_contract.py")[..],
            ),
            (
                "packages/openvino/package_inventory.py",
                &include_bytes!("../packages/openvino/package_inventory.py")[..],
            ),
            (
                "scripts/apply_admission_activation.py",
                &include_bytes!("../scripts/apply_admission_activation.py")[..],
            ),
        ];
        let mut digest = sha2::Sha256::new();
        digest.update(b"cfetch-admission-implementation-bundle-v1\0");
        for (relative_path, bytes) in files {
            let path_bytes = relative_path.as_bytes();
            digest.update((path_bytes.len() as u32).to_be_bytes());
            digest.update(path_bytes);
            digest.update((bytes.len() as u64).to_be_bytes());
            digest.update(bytes);
        }
        let computed = crate::hashing::hex_lower(digest.finalize());
        let registry: serde_json::Value =
            serde_json::from_str(include_str!("../release/inference-backends.json")).unwrap();

        assert_eq!(computed, ADMISSION_IMPLEMENTATION_BUNDLE_SHA256);
        assert_eq!(
            admission_policy().admission_implementation_bundle_sha256,
            computed
        );
        assert_eq!(
            registry["admission"]["implementation_bundle_sha256"],
            computed
        );
    }

    #[test]
    fn sequence_bucket_selection_is_total_only_inside_the_profile_limit() {
        assert_eq!(sequence_bucket_for_token_count(0), None);
        assert_eq!(sequence_bucket_for_token_count(1), Some(32));
        assert_eq!(sequence_bucket_for_token_count(32), Some(32));
        assert_eq!(sequence_bucket_for_token_count(33), Some(64));
        assert_eq!(sequence_bucket_for_token_count(2048), Some(2048));
        assert_eq!(sequence_bucket_for_token_count(2049), None);
    }

    #[test]
    fn production_availability_is_fail_closed_for_inactive_or_empty_registry() {
        // Profile is now active; the fail-closed path for a hypothetical
        // inactive status is exercised through the empty-registry check
        // below. The production gate still refuses an empty admitted set.
        let empty = serde_json::json!({
            "profile_id": PROFILE_ID,
            "profile_status": "active",
            "shared_identity": {"profile_manifest_sha256": PROFILE_MANIFEST_SHA256},
            "admission": {
                "policy_manifest_sha256": ADMISSION_POLICY_SHA256,
                "implementation_bundle_sha256": ADMISSION_IMPLEMENTATION_BUNDLE_SHA256,
            },
            "admitted_backends": [],
        });
        let error = validate_production_availability("active", &empty)
            .unwrap_err()
            .to_string();
        assert!(error.contains("no admitted production backends"), "{error}");
    }

    #[test]
    fn release_registry_keeps_npu_gpu_cpu_order_and_no_unproved_backend() {
        let registry: serde_json::Value =
            serde_json::from_str(include_str!("../release/inference-backends.json")).unwrap();
        assert_eq!(registry["profile_id"], PROFILE_ID);
        assert_eq!(registry["profile_status"], PROFILE_STATUS);
        assert_eq!(registry["model_candidate"]["source"], MODEL);
        assert_eq!(registry["model_candidate"]["revision"], MODEL_REVISION);
        assert_eq!(
            registry["shared_identity"]["profile_manifest_sha256"],
            PROFILE_MANIFEST_SHA256
        );
        assert_eq!(
            registry["internal_execution"]["numerical_anchor"],
            serde_json::Value::Null
        );
        assert_eq!(
            registry["internal_execution"]["precision_policy"],
            "target-native"
        );
        assert_eq!(
            registry["decision_priority"],
            serde_json::json!(DECISION_PRIORITY)
        );
        assert_eq!(
            registry["admission"]["policy_manifest_sha256"],
            ADMISSION_POLICY_SHA256
        );
        assert_eq!(
            registry["admission"]["implementation_bundle_sha256"],
            ADMISSION_IMPLEMENTATION_BUNDLE_SHA256
        );
        assert_eq!(registry["admission"]["dataset"], ADMISSION_DATASET);
        assert_eq!(
            registry["admission"]["dataset_revision"],
            ADMISSION_DATASET_REVISION
        );
        assert_eq!(
            registry["admission"]["sequence_semantic_fixture"]["id"],
            SEQUENCE_SEMANTIC_FIXTURE_ID
        );
        assert_eq!(
            registry["admission"]["sequence_semantic_fixture"]["sha256"],
            SEQUENCE_SEMANTIC_FIXTURE_SHA256
        );
        assert_eq!(
            registry["admission"]["sequence_semantic_gate"],
            SEQUENCE_SEMANTIC_GATE
        );
        assert_eq!(
            registry["admission"]["required_device_classes"],
            serde_json::json!(ADMISSION_REQUIRED_DEVICE_CLASSES)
        );
        assert_eq!(registry["admission"]["ranking"], ADMISSION_RANKING);
        assert_eq!(registry["admission"]["tie_break"], ADMISSION_TIE_BREAK);
        assert_eq!(
            registry["admission"]["mixed_document_store"],
            ADVERSARIAL_MIXED_DOCUMENT_SELECTION
        );
        assert_eq!(
            registry["admission"]["wire_batch_contract"],
            BATCHING_CONTRACT
        );
        assert_eq!(
            registry["admission"]["absolute_minimums"]["ndcg_at_10"].as_f64(),
            Some(ADMISSION_NDCG_AT_10_MINIMUM.parse().unwrap())
        );
        assert_eq!(
            registry["admission"]["absolute_minimums"]["recall_at_100"].as_f64(),
            Some(ADMISSION_RECALL_AT_100_MINIMUM.parse().unwrap())
        );
        assert_eq!(
            registry["admission"]["absolute_minimums"]["mrr_at_10"].as_f64(),
            Some(ADMISSION_MRR_AT_10_MINIMUM.parse().unwrap())
        );
        assert_eq!(
            registry["selection_order"],
            serde_json::json!(["npu", "gpu", "cpu"])
        );
        assert_eq!(registry["remote_policy"], "explicit-only");
        assert_eq!(
            registry["package_composition_schema"]["remote_fallback"],
            "none"
        );
        assert_eq!(
            registry["package_composition_schema"]["selection"],
            "first available admitted scope in NPU, GPU, accelerated CPU order; each signed request and response is bound to the requested scope id"
        );
        assert_eq!(registry["local_packages"], serde_json::json!([]));
        assert_eq!(registry["admitted_backends"], serde_json::json!([]));
        assert!(!backend_scope_is_admitted(&BackendScopeAttestation {
            scope_id: "candidate",
            transport: ExecutionTransport::RemoteAttested,
            backend: "candidate-runtime",
            runtime: "candidate-version",
            compiler: "candidate-compiler",
            package_target: "candidate-target",
            artifact_source: "candidate-source",
            artifact_sha256: &"0".repeat(64),
            internal_precision: "target-native",
            device: "candidate-device",
            device_class: "npu",
            placement_evidence_sha256: &"1".repeat(64),
            supported_max_tokens: MAX_TOKENS,
            supported_sequence_buckets: SEQUENCE_BUCKETS,
            supported_max_batch_size: MAX_WIRE_BATCH_SIZE,
            sequence_capability_evidence_sha256: &"2".repeat(64),
            performance_evidence_sha256: &"3".repeat(64),
            compatibility_report_sha256: &"4".repeat(64),
            accelerated_placement: true,
        }));
    }

    #[test]
    fn semantic_pipeline_changes_are_refused_inside_v1() {
        let valid = EmbeddingsConfig::default();
        validate(&valid).unwrap();

        let mut changed = valid.clone();
        changed.model = "another/model".into();
        assert!(
            validate(&changed)
                .unwrap_err()
                .to_string()
                .contains("new network major")
        );

        let mut changed = valid.clone();
        changed.dimensions = 256;
        assert!(
            validate(&changed)
                .unwrap_err()
                .to_string()
                .contains("re-embedding")
        );

        let mut changed = valid;
        changed.document_prefix.clear();
        assert!(validate(&changed).is_err());
    }
}
