//! Canonical request/response core for the package-local native adapter.
//!
//! This is staged integration, not an admitted producer. There is deliberately
//! no HTTP/CLI entrypoint: the native package/runtime-closure loader must be
//! ported before `from_installation` can construct a service. Test fixtures
//! cannot enter a production build. The wire contract and profile stay fixed.

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context as _, ensure};
use ed25519_dalek::Signer as _;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest as _, Sha256};

use crate::embedding_profile as profile;
use crate::inference_governor::Governor;
use crate::native_adapter::{Adapter, NativeFailure};
use crate::native_worker::{Command, Identity, Operation, PROTOCOL_VERSION, Pooling, Response};

const MAX_BODY: usize = 8 * 1024 * 1024;
const MAX_TOKENIZER: usize = 128 * 1024 * 1024;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

trait TextTokenizer {
    /// Input already contains its query/document prefix. This layer adds only
    /// the frozen BOS/EOS tokens and never adds a second instruction.
    fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>>;
}

struct PinnedTokenizer(tokenizers::Tokenizer);

impl PinnedTokenizer {
    fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        ensure!(
            bytes.len() <= MAX_TOKENIZER,
            "tokenizer exceeds its byte limit"
        );
        ensure!(
            digest(bytes) == profile::TOKENIZER_JSON_SHA256,
            "tokenizer is not the frozen Gemma tokenizer"
        );
        let mut tokenizer = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|error| anyhow::anyhow!("decode pinned tokenizer: {error}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!("disable tokenizer truncation: {error}"))?;
        tokenizer.with_padding(None);
        for (token, expected) in [("<pad>", 0), ("<bos>", 2), ("<eos>", 1)] {
            ensure!(
                tokenizer.token_to_id(token) == Some(expected),
                "frozen tokenizer special-token identity changed"
            );
        }
        Ok(Self(tokenizer))
    }
}

impl TextTokenizer for PinnedTokenizer {
    fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>> {
        let encoded = self
            .0
            .encode(text, false)
            .map_err(|error| anyhow::anyhow!("tokenize canonical input: {error}"))?;
        Ok(std::iter::once(2)
            .chain(encoded.get_ids().iter().copied().map(i64::from))
            .chain(std::iter::once(1))
            .collect())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    model: String,
    input: crate::embed::BoundedVec<String, { profile::MAX_WIRE_BATCH_SIZE }>,
    dimensions: usize,
    cfetch_requested_scope_id: String,
}

#[derive(Debug)]
struct PreparedInput {
    token_count: usize,
    bucket: usize,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
}

struct Scope {
    id: String,
    execution: serde_json::Value,
    identity: Identity,
    compile: Operation,
    signer: ed25519_dalek::SigningKey,
}

impl Scope {
    fn validate_shape(&self) -> anyhow::Result<()> {
        ensure!(
            self.execution["scope_id"] == self.id,
            "scope identity changed"
        );
        ensure!(
            self.identity.pipeline_sha256 == profile::PROFILE_MANIFEST_SHA256
                && self.identity.dimensions == profile::DIMENSIONS
                && self.identity.pooling == Pooling::SentenceL2
                && self.identity.output_name == "embedding",
            "native graph must implement the complete frozen Gemma pooling/dense pipeline"
        );
        let class = match self.identity.device {
            crate::inference_governor::Device::Npu => "npu",
            crate::inference_governor::Device::Gpu => "gpu",
            crate::inference_governor::Device::Cpu => "cpu",
        };
        ensure!(
            self.execution["device_class"] == class,
            "native device differs from the attested scope"
        );
        ensure!(
            matches!(self.compile, Operation::Compile { .. }),
            "scope lacks a compile recipe"
        );
        Command {
            schema_version: PROTOCOL_VERSION,
            request_id: "validate".into(),
            identity: self.identity.clone(),
            operation: self.compile.clone(),
        }
        .validate()
    }

    fn validate_admission(&self) -> anyhow::Result<()> {
        self.validate_shape()?;
        let key: String = self
            .signer
            .verifying_key()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        crate::embed::validate_native_serving_scope(&self.execution, &self.id, &key)
    }
}

/// A confirmed controlled failure affects one scope. Every uncertain outcome
/// latches this entire service, including CPU, until explicit recovery.
#[derive(Debug)]
enum Failure {
    Rejected(String),
    ScopeUnavailable(String),
    HardStop(anyhow::Error),
}

impl Failure {
    fn wire_error(&self) -> (u16, serde_json::Value) {
        match self {
            Self::ScopeUnavailable(scope) => (
                503,
                json!({"error": {
                    "code": "scope_unavailable", "scope_id": scope,
                    "message": "requested admitted scope could not initialize or execute"
                }}),
            ),
            Self::Rejected(message) => (
                400,
                json!({"error": {
                    "code": "invalid_request", "message": message
                }}),
            ),
            Self::HardStop(_) => (
                500,
                json!({"error": {
                    "code": "native_hard_stop", "message": "native inference requires operator review"
                }}),
            ),
        }
    }
}

struct SignedReply {
    body: Vec<u8>,
    signature: String,
}

trait Engine {
    fn embed(&mut self, scope: &Scope, input: &PreparedInput) -> Result<Vec<f32>, NativeFailure>;
    fn preferred_bucket(&self, _scope: &Scope) -> Option<usize> {
        None
    }
    fn stop(&mut self) -> anyhow::Result<()>;
}

trait Worker {
    fn execute(&mut self, command: &Command) -> Result<Response, NativeFailure>;
    fn stop(&mut self) -> anyhow::Result<()>;
}

trait WorkerFactory {
    type Worker: Worker;
    fn spawn(&mut self, identity: &Identity, bucket: usize) -> anyhow::Result<Self::Worker>;
}

struct GovernedWorker {
    adapter: Adapter,
    governor: Arc<Governor>,
    evidence: PathBuf,
}

impl Worker for GovernedWorker {
    fn execute(&mut self, command: &Command) -> Result<Response, NativeFailure> {
        self.adapter
            .execute(&self.governor, command, &self.evidence)
    }
    fn stop(&mut self) -> anyhow::Result<()> {
        self.adapter.stop()
    }
}

struct GovernedFactory {
    governor: Arc<Governor>,
    evidence: PathBuf,
}

impl WorkerFactory for GovernedFactory {
    type Worker = GovernedWorker;
    fn spawn(&mut self, identity: &Identity, bucket: usize) -> anyhow::Result<Self::Worker> {
        Ok(GovernedWorker {
            adapter: Adapter::spawn(identity.clone(), bucket)?,
            governor: Arc::clone(&self.governor),
            evidence: self.evidence.clone(),
        })
    }
}

struct CachedWorker<W> {
    scope: String,
    identity: Identity,
    compile_sha256: String,
    bucket: usize,
    worker: W,
}

/// At most one compiled native worker is resident. Changing bucket, runtime,
/// device or profile requires confirmed termination before spawning another.
/// Every compile and inference still goes through the installed host governor.
struct NativeEngine<F: WorkerFactory> {
    factory: F,
    cached: Option<CachedWorker<F::Worker>>,
}

impl<F: WorkerFactory> NativeEngine<F> {
    fn command(scope: &Scope, operation: Operation) -> Command {
        Command {
            schema_version: PROTOCOL_VERSION,
            request_id: rand::random::<[u8; 16]>()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
            identity: scope.identity.clone(),
            operation,
        }
    }

    fn worker(&mut self, scope: &Scope, bucket: usize) -> Result<&mut F::Worker, NativeFailure> {
        let mut operation = scope.compile.clone();
        let Operation::Compile { bucket: target, .. } = &mut operation else {
            return Err(NativeFailure::HardStop(anyhow::anyhow!(
                "invalid compile recipe"
            )));
        };
        *target = bucket;
        let compile_sha256 = digest(
            &serde_json::to_vec(&operation)
                .map_err(|error| NativeFailure::HardStop(error.into()))?,
        );
        let same = self.cached.as_ref().is_some_and(|cached| {
            cached.scope == scope.id
                && cached.identity == scope.identity
                && cached.compile_sha256 == compile_sha256
                && cached.bucket == bucket
        });
        if !same {
            if let Some(cached) = &mut self.cached {
                // Keep the owner if cleanup fails. Never create a second worker.
                cached.worker.stop().map_err(NativeFailure::HardStop)?;
            }
            self.cached = None;
            let mut worker = self
                .factory
                .spawn(&scope.identity, bucket)
                .map_err(NativeFailure::HardStop)?;
            worker.execute(&Self::command(scope, operation))?;
            self.cached = Some(CachedWorker {
                scope: scope.id.clone(),
                identity: scope.identity.clone(),
                compile_sha256,
                bucket,
                worker,
            });
        }
        Ok(&mut self.cached.as_mut().expect("compiled worker").worker)
    }
}

impl<F: WorkerFactory> Engine for NativeEngine<F> {
    fn preferred_bucket(&self, scope: &Scope) -> Option<usize> {
        self.cached
            .as_ref()
            .filter(|cached| cached.scope == scope.id && cached.identity == scope.identity)
            .map(|cached| cached.bucket)
    }

    fn stop(&mut self) -> anyhow::Result<()> {
        if let Some(cached) = &mut self.cached {
            cached.worker.stop()?;
        }
        self.cached = None;
        Ok(())
    }

    fn embed(&mut self, scope: &Scope, input: &PreparedInput) -> Result<Vec<f32>, NativeFailure> {
        scope.validate_shape().map_err(NativeFailure::HardStop)?;
        let response = self.worker(scope, input.bucket)?.execute(&Self::command(
            scope,
            Operation::Infer {
                input_ids: input.input_ids.clone(),
                attention_mask: input.attention_mask.clone(),
                token_type_ids: None,
            },
        ))?;
        response
            .output
            .context("native worker omitted its vector")
            .map_err(NativeFailure::HardStop)
    }
}

struct ServingCore<T: TextTokenizer, E: Engine> {
    tokenizer: T,
    engine: E,
    scopes: Vec<Scope>,
    unavailable: BTreeSet<String>,
    stopped: bool,
}

impl ServingCore<PinnedTokenizer, NativeEngine<GovernedFactory>> {
    fn from_installation() -> anyhow::Result<Self> {
        // This check cannot be replaced by a manifest's own claim of admission.
        crate::local_inference::selected_local_package_plan()?
            .context("no admitted package-local native producer in this release")?;
        anyhow::bail!(
            "native package/runtime closure verification and serving transport are not installed"
        )
    }
}

impl<T: TextTokenizer, E: Engine> ServingCore<T, E> {
    #[cfg(test)]
    fn fixture(tokenizer: T, engine: E, scopes: Vec<Scope>) -> Self {
        for scope in &scopes {
            scope.validate_shape().unwrap();
        }
        Self {
            tokenizer,
            engine,
            scopes,
            unavailable: BTreeSet::new(),
            stopped: false,
        }
    }

    fn handle(&mut self, body: &[u8], nonce: &[u8; 32]) -> Result<SignedReply, Failure> {
        if self.stopped {
            return Err(Failure::HardStop(anyhow::anyhow!(
                "native serving is latched after an uncertain outcome"
            )));
        }
        match self.response(body, nonce) {
            Err(Failure::HardStop(error)) => {
                self.stopped = true;
                // An idle cached worker can exist even when tokenization or
                // output validation fails before/after the next native call.
                match self.engine.stop() {
                    Ok(()) => Err(Failure::HardStop(error)),
                    Err(cleanup) => Err(Failure::HardStop(error.context(format!(
                        "native worker cleanup was not confirmed: {cleanup:#}"
                    )))),
                }
            }
            result => result,
        }
    }

    fn response(&mut self, body: &[u8], nonce: &[u8; 32]) -> Result<SignedReply, Failure> {
        if body.len() > MAX_BODY {
            return Err(Failure::Rejected("request exceeds byte limit".into()));
        }
        let request: Request =
            serde_json::from_slice(body).map_err(|error| Failure::Rejected(error.to_string()))?;
        if request.model != profile::MODEL || request.dimensions != profile::DIMENSIONS {
            return Err(Failure::Rejected(
                "request does not name the frozen Gemma vector profile".into(),
            ));
        }
        let texts = request.input.as_slice();
        if !(1..=profile::MAX_WIRE_BATCH_SIZE).contains(&texts.len()) {
            return Err(Failure::Rejected(
                "request must contain 1 through 64 inputs".into(),
            ));
        }
        let scope = self
            .scopes
            .iter()
            .find(|scope| scope.id == request.cfetch_requested_scope_id)
            .ok_or_else(|| Failure::Rejected("unknown package-local scope".into()))?;
        if self.unavailable.contains(&scope.id) {
            return Err(Failure::ScopeUnavailable(scope.id.clone()));
        }

        // Prepare the entire batch before any native work: an oversize final
        // row must not consume accelerator work or publish a partial response.
        let mut prepared = Vec::with_capacity(texts.len());
        for text in texts {
            let mut ids = self.tokenizer.encode(text).map_err(Failure::HardStop)?;
            if ids.is_empty() || ids.iter().any(|id| *id < 0) {
                return Err(Failure::HardStop(anyhow::anyhow!(
                    "tokenizer produced invalid IDs"
                )));
            }
            let count = ids.len();
            let bucket = profile::SEQUENCE_BUCKETS
                .iter()
                .copied()
                .find(|bucket| *bucket >= count)
                .ok_or_else(|| {
                    Failure::Rejected(format!(
                        "prefixed input contains {count} tokens; truncation is forbidden"
                    ))
                })?;
            ids.resize(bucket, 0);
            let mut mask = vec![1; count];
            mask.resize(bucket, 0);
            prepared.push(PreparedInput {
                token_count: count,
                bucket,
                input_ids: ids,
                attention_mask: mask,
            });
        }

        // Group shapes to avoid recompiling for alternating bucket rows.
        // Response positions remain the original request indices.
        let preferred = self.engine.preferred_bucket(scope);
        let mut order: Vec<usize> = (0..prepared.len()).collect();
        order.sort_by_key(|index| {
            let bucket = prepared[*index].bucket;
            (Some(bucket) != preferred, bucket, *index)
        });
        let mut rows = vec![serde_json::Value::Null; prepared.len()];
        for index in order {
            let input = &prepared[index];
            let vector = match self.engine.embed(scope, input) {
                Ok(vector) => vector,
                Err(NativeFailure::Controlled(_completed)) => {
                    self.unavailable.insert(scope.id.clone());
                    return Err(Failure::ScopeUnavailable(scope.id.clone()));
                }
                Err(NativeFailure::HardStop(error)) => return Err(Failure::HardStop(error)),
            };
            let norm: f64 = vector.iter().map(|v| f64::from(*v).powi(2)).sum();
            if vector.len() != profile::DIMENSIONS
                || !vector.iter().all(|v| v.is_finite())
                || (norm - 1.0).abs() > 1e-4
            {
                return Err(Failure::HardStop(anyhow::anyhow!(
                    "native output is not a finite normalized canonical vector"
                )));
            }
            rows[index] = json!({"index": index, "embedding": vector, "token_count": input.token_count,
                "sequence_bucket": input.bucket, "truncated": false, "cfetch_scope_id": scope.id});
        }
        let response = serde_json::to_vec(&json!({
            "model": profile::MODEL, "cfetch_profile": profile::PROFILE_ID,
            "cfetch_profile_manifest_sha256": profile::PROFILE_MANIFEST_SHA256,
            "cfetch_admission_policy_sha256": profile::ADMISSION_POLICY_SHA256,
            "cfetch_model_revision": profile::MODEL_REVISION,
            "cfetch_execution": scope.execution, "data": rows
        }))
        .map_err(|error| Failure::HardStop(error.into()))?;
        if response.len() > MAX_BODY {
            return Err(Failure::HardStop(anyhow::anyhow!(
                "response exceeds byte limit"
            )));
        }
        let signature = scope
            .signer
            .sign(&crate::embed::attestation_message(nonce, body, &response));
        Ok(SignedReply {
            body: response,
            signature: signature
                .to_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_governor::Device;
    use crate::native_worker::Precision;
    use std::sync::Mutex;

    #[derive(Clone, Default)]
    struct FixtureTokenizer(Arc<Mutex<Vec<String>>>, Arc<std::sync::atomic::AtomicBool>);
    impl TextTokenizer for FixtureTokenizer {
        fn encode(&self, text: &str) -> anyhow::Result<Vec<i64>> {
            ensure!(
                !self.1.swap(false, std::sync::atomic::Ordering::SeqCst),
                "fixture tokenizer failed"
            );
            self.0.lock().unwrap().push(text.into());
            Ok(std::iter::once(2)
                .chain(text.chars().map(|c| i64::from(u32::from(c))))
                .chain(std::iter::once(1))
                .collect())
        }
    }

    fn scope(device: Device) -> Scope {
        let (class, key) = match device {
            Device::Npu => ("npu", 1),
            Device::Gpu => ("gpu", 2),
            Device::Cpu => ("cpu", 3),
        };
        let signer = ed25519_dalek::SigningKey::from_bytes(&[key; 32]);
        Scope {
            id: class.into(),
            signer,
            execution: json!({"scope_id": class, "transport":"supervised-local", "device_class":class,
                "backend":"openvino", "runtime":"fixture-runtime", "compiler":"fixture-compiler",
                "package_target":"linux-x86_64", "artifact_source":"fixture", "device":class,
                "artifact_sha256":"a".repeat(64), "internal_precision":"fp32",
                "placement_evidence_sha256":"b".repeat(64), "supported_max_tokens":profile::MAX_TOKENS,
                "supported_sequence_buckets":profile::SEQUENCE_BUCKETS, "supported_max_batch_size":profile::MAX_WIRE_BATCH_SIZE,
                "sequence_capability_evidence_sha256":"c".repeat(64), "performance_evidence_sha256":"d".repeat(64),
                "compatibility_report_sha256":"e".repeat(64), "accelerated_placement":true}),
            identity: Identity {
                device,
                model_sha256: "a".repeat(64),
                weights_sha256: Some("b".repeat(64)),
                runtime_build: "fixture-runtime".into(),
                pipeline_sha256: profile::PROFILE_MANIFEST_SHA256.into(),
                output_name: "embedding".into(),
                pooling: Pooling::SentenceL2,
                dimensions: profile::DIMENSIONS,
            },
            compile: Operation::Compile {
                model_path: "/fixture/model.xml".into(),
                weights_path: Some("/fixture/model.bin".into()),
                runtime_library_path: "/fixture/runtime.so".into(),
                runtime_library_sha256: "c".repeat(64),
                bucket: 32,
                precision: Precision::F32,
                threads: 1,
            },
        }
    }

    fn body(scope: &str, texts: &[&str]) -> Vec<u8> {
        serde_json::to_vec(
            &json!({"model":profile::MODEL,"dimensions":profile::DIMENSIONS,
            "cfetch_requested_scope_id":scope,"input":texts}),
        )
        .unwrap()
    }

    fn vector() -> Vec<f32> {
        let mut value = vec![0.0; profile::DIMENSIONS];
        value[0] = 1.0;
        value
    }

    type RecordedCall = (String, usize, Vec<i64>, Vec<i64>);

    #[derive(Default)]
    struct FixtureEngine {
        calls: Vec<RecordedCall>,
        failures: std::collections::VecDeque<NativeFailure>,
        output: Option<Vec<f32>>,
        stops: usize,
    }
    impl Engine for FixtureEngine {
        fn stop(&mut self) -> anyhow::Result<()> {
            self.stops += 1;
            Ok(())
        }
        fn embed(
            &mut self,
            scope: &Scope,
            input: &PreparedInput,
        ) -> Result<Vec<f32>, NativeFailure> {
            self.calls.push((
                scope.id.clone(),
                input.bucket,
                input.input_ids.clone(),
                input.attention_mask.clone(),
            ));
            if let Some(error) = self.failures.pop_front() {
                return Err(error);
            }
            Ok(self.output.clone().unwrap_or_else(vector))
        }
    }
    fn service(engine: FixtureEngine) -> ServingCore<FixtureTokenizer, FixtureEngine> {
        ServingCore::fixture(
            FixtureTokenizer::default(),
            engine,
            vec![scope(Device::Npu), scope(Device::Gpu), scope(Device::Cpu)],
        )
    }

    #[test]
    fn text_and_special_tokens_keep_existing_prefixes_and_signed_wire_identity() {
        let mut core = service(FixtureEngine::default());
        let text = format!("{}Grüezi 世界", profile::QUERY_PREFIX);
        let request = body("cpu", &[&text]);
        let nonce = [7; 32];
        let response = core.handle(&request, &nonce).unwrap();
        assert_eq!(*core.tokenizer.0.lock().unwrap(), [text.clone()]);
        let call = &core.engine.calls[0];
        let count = text.chars().count() + 2;
        assert_eq!(call.2[0], 2);
        assert_eq!(call.2[count - 1], 1);
        assert!(call.2[count..].iter().all(|id| *id == 0));
        assert_eq!(call.3.iter().sum::<i64>(), count as i64);
        let parsed: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(
            parsed["cfetch_profile_manifest_sha256"],
            profile::PROFILE_MANIFEST_SHA256
        );
        assert_eq!(parsed["data"][0]["token_count"], count);
        assert_eq!(parsed["data"][0]["truncated"], false);
        let key: String = core.scopes[2]
            .signer
            .verifying_key()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        crate::embed::verify_execution_signature(
            &key,
            &response.signature,
            &nonce,
            &request,
            &response.body,
        )
        .unwrap();
        assert!(
            crate::embed::verify_execution_signature(
                &key,
                &response.signature,
                &[8; 32],
                &request,
                &response.body
            )
            .is_err()
        );
        assert!(
            crate::embed::verify_execution_signature(
                &key,
                &response.signature,
                &nonce,
                &body("gpu", &[&text]),
                &response.body
            )
            .is_err()
        );
        let mut changed = response.body;
        changed.push(b' ');
        assert!(
            crate::embed::verify_execution_signature(
                &key,
                &response.signature,
                &nonce,
                &request,
                &changed
            )
            .is_err()
        );
    }

    #[test]
    fn all_sequence_boundaries_use_smallest_bucket_without_truncation() {
        let mut core = service(FixtureEngine::default());
        for (index, bucket) in profile::SEQUENCE_BUCKETS.iter().copied().enumerate() {
            let text = "x".repeat(bucket - 2);
            core.handle(&body("cpu", &[&text]), &[0; 32]).unwrap();
            assert_eq!(core.engine.calls.last().unwrap().1, bucket);
            if let Some(next) = profile::SEQUENCE_BUCKETS.get(index + 1) {
                let text = "x".repeat(bucket - 1);
                core.handle(&body("cpu", &[&text]), &[0; 32]).unwrap();
                assert_eq!(core.engine.calls.last().unwrap().1, *next);
            }
        }
        let calls = core.engine.calls.len();
        assert!(matches!(
            core.handle(
                &body("cpu", &["valid", &"x".repeat(profile::MAX_TOKENS - 1)]),
                &[0; 32]
            ),
            Err(Failure::Rejected(_))
        ));
        assert_eq!(
            core.engine.calls.len(),
            calls,
            "oversize last row must reject before any native work"
        );
    }

    #[test]
    fn batch_order_and_scope_are_preserved_and_more_than_64_is_rejected() {
        let mut core = service(FixtureEngine::default());
        let texts: Vec<String> = (0..64)
            .map(|i| format!("{}document {i}", profile::DOCUMENT_PREFIX))
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let reply = core.handle(&body("cpu", &refs), &[1; 32]).unwrap();
        let response: serde_json::Value = serde_json::from_slice(&reply.body).unwrap();
        for (index, row) in response["data"].as_array().unwrap().iter().enumerate() {
            assert_eq!(row["index"], index);
            assert_eq!(row["cfetch_scope_id"], "cpu");
        }
        assert_eq!(*core.tokenizer.0.lock().unwrap(), texts);
        assert!(matches!(
            core.handle(&body("cpu", &["x"; 65]), &[1; 32]),
            Err(Failure::Rejected(_))
        ));
        assert_eq!(core.engine.calls.len(), 64);
    }

    #[test]
    fn controlled_failure_disables_only_its_scope_and_preserves_ordered_attempts() {
        let mut engine = FixtureEngine::default();
        engine
            .failures
            .push_back(crate::native_adapter::fixture_controlled_failure());
        engine
            .failures
            .push_back(crate::native_adapter::fixture_controlled_failure());
        let mut core = service(engine);
        for id in ["npu", "gpu"] {
            let error = core.handle(&body(id, &["text"]), &[0; 32]).err().unwrap();
            let (status, wire) = error.wire_error();
            assert_eq!(status, 503);
            assert_eq!(wire["error"]["code"], "scope_unavailable");
            assert_eq!(wire["error"]["scope_id"], id);
        }
        core.handle(&body("cpu", &["text"]), &[0; 32]).unwrap();
        assert!(matches!(
            core.handle(&body("npu", &["text"]), &[0; 32]),
            Err(Failure::ScopeUnavailable(_))
        ));
        assert_eq!(
            core.engine
                .calls
                .iter()
                .map(|c| c.0.as_str())
                .collect::<Vec<_>>(),
            ["npu", "gpu", "cpu"]
        );
    }

    #[test]
    fn uncertain_intent_protocol_or_artifact_error_stops_every_scope_including_cpu() {
        for detail in [
            "pending intent",
            "worker crash",
            "artifact changed",
            "checkpoint failed",
            "cleanup failed",
        ] {
            let mut engine = FixtureEngine::default();
            engine
                .failures
                .push_back(NativeFailure::HardStop(anyhow::anyhow!(detail)));
            let mut core = service(engine);
            let error = core
                .handle(&body("npu", &["text"]), &[0; 32])
                .err()
                .unwrap();
            assert_eq!(error.wire_error().0, 500);
            match error {
                Failure::HardStop(error) => assert_eq!(error.to_string(), detail),
                _ => panic!("hard failure was weakened"),
            }
            for next in ["gpu", "cpu", "npu"] {
                assert!(matches!(
                    core.handle(&body(next, &["text"]), &[0; 32]),
                    Err(Failure::HardStop(_))
                ));
            }
            assert_eq!(core.engine.calls.len(), 1);
            assert_eq!(core.engine.stops, 1);
        }
    }

    #[test]
    fn malformed_output_never_emits_a_signed_response_or_falls_back() {
        let mut nan = vector();
        nan[1] = f32::NAN;
        for output in [
            vec![1.0],
            vec![0.0; profile::DIMENSIONS],
            vec![1.0; profile::DIMENSIONS],
            nan,
        ] {
            let mut core = service(FixtureEngine {
                output: Some(output),
                ..Default::default()
            });
            assert!(matches!(
                core.handle(&body("npu", &["text"]), &[0; 32]),
                Err(Failure::HardStop(_))
            ));
            assert!(matches!(
                core.handle(&body("cpu", &["text"]), &[0; 32]),
                Err(Failure::HardStop(_))
            ));
            assert_eq!(core.engine.calls.len(), 1);
            assert_eq!(core.engine.stops, 1);
        }
    }

    #[test]
    fn profile_and_request_refusals_do_not_touch_native_workers() {
        let mut core = service(FixtureEngine::default());
        let good: serde_json::Value = serde_json::from_slice(&body("cpu", &["text"])).unwrap();
        for (field, value) in [
            ("model", json!("Qwen/Qwen3-Embedding-0.6B")),
            ("dimensions", json!(1024)),
            ("cfetch_requested_scope_id", json!("unregistered")),
            ("input", json!([])),
            ("extra", json!(true)),
        ] {
            let mut request = good.clone();
            request[field] = value;
            assert!(matches!(
                core.handle(&serde_json::to_vec(&request).unwrap(), &[0; 32]),
                Err(Failure::Rejected(_))
            ));
        }
        assert!(core.engine.calls.is_empty());
        assert!(!core.stopped);
        core.handle(&body("cpu", &["text"]), &[0; 32]).unwrap();
    }

    #[test]
    fn production_constructor_and_scope_admission_remain_closed() {
        assert!(ServingCore::from_installation().is_err());
        let mut canonical = scope(Device::Cpu);
        assert!(
            canonical.validate_admission().is_err(),
            "fixture key cannot admit a backend"
        );
        canonical.identity.pipeline_sha256 = "d".repeat(64);
        assert!(canonical.validate_shape().is_err());
        canonical.identity.pipeline_sha256 = profile::PROFILE_MANIFEST_SHA256.into();
        canonical.identity.pooling = Pooling::LastTokenL2;
        assert!(canonical.validate_shape().is_err());
        assert!(PinnedTokenizer::from_bytes(br#"{"version":"1.0"}"#).is_err());
    }

    #[derive(Default)]
    struct WorkerLog {
        events: Vec<String>,
        fail_stop: bool,
    }
    struct FakeWorker(Arc<Mutex<WorkerLog>>);
    impl Worker for FakeWorker {
        fn execute(&mut self, command: &Command) -> Result<Response, NativeFailure> {
            command.validate().map_err(NativeFailure::HardStop)?;
            let compile = matches!(command.operation, Operation::Compile { .. });
            self.0
                .lock()
                .unwrap()
                .events
                .push(if compile { "compile" } else { "infer" }.into());
            Ok(Response {
                schema_version: PROTOCOL_VERSION,
                request_id: Some(command.request_id.clone()),
                request_sha256: "f".repeat(64),
                identity: Some(command.identity.clone()),
                runtime_build: Some(command.identity.runtime_build.clone()),
                execution_devices: vec!["CPU".into()],
                output: match &command.operation {
                    Operation::Compile { .. } => None,
                    Operation::Infer { input_ids, .. } => {
                        let mut output = vec![0.0; profile::DIMENSIONS];
                        output[usize::try_from(input_ids[1]).unwrap() % profile::DIMENSIONS] = 1.0;
                        Some(output)
                    }
                },
                error: None,
            })
        }
        fn stop(&mut self) -> anyhow::Result<()> {
            let mut log = self.0.lock().unwrap();
            log.events.push("stop".into());
            ensure!(!log.fail_stop, "cannot confirm child death");
            Ok(())
        }
    }
    struct FakeFactory(Arc<Mutex<WorkerLog>>);
    impl WorkerFactory for FakeFactory {
        type Worker = FakeWorker;
        fn spawn(&mut self, _: &Identity, bucket: usize) -> anyhow::Result<Self::Worker> {
            self.0
                .lock()
                .unwrap()
                .events
                .push(format!("spawn-{bucket}"));
            Ok(FakeWorker(Arc::clone(&self.0)))
        }
    }
    fn input(bucket: usize) -> PreparedInput {
        let mut ids = vec![0; bucket];
        ids[..2].copy_from_slice(&[2, 1]);
        let mut mask = vec![0; bucket];
        mask[..2].fill(1);
        PreparedInput {
            token_count: 2,
            bucket,
            input_ids: ids,
            attention_mask: mask,
        }
    }

    #[test]
    fn compiled_worker_reuse_is_exact_and_eviction_confirms_child_death_first() {
        let log = Arc::new(Mutex::new(WorkerLog::default()));
        let mut engine = NativeEngine {
            factory: FakeFactory(Arc::clone(&log)),
            cached: None,
        };
        let mut cpu = scope(Device::Cpu);
        engine.embed(&cpu, &input(32)).unwrap();
        engine.embed(&cpu, &input(32)).unwrap();
        engine.embed(&cpu, &input(64)).unwrap();
        cpu.identity.runtime_build = "other-runtime".into();
        engine.embed(&cpu, &input(64)).unwrap();
        engine.embed(&scope(Device::Gpu), &input(64)).unwrap();
        assert_eq!(
            log.lock().unwrap().events,
            [
                "spawn-32", "compile", "infer", "infer", "stop", "spawn-64", "compile", "infer",
                "stop", "spawn-64", "compile", "infer", "stop", "spawn-64", "compile", "infer"
            ]
        );
    }

    #[test]
    fn unconfirmed_cache_eviction_cannot_spawn_another_worker() {
        let log = Arc::new(Mutex::new(WorkerLog::default()));
        let mut engine = NativeEngine {
            factory: FakeFactory(Arc::clone(&log)),
            cached: None,
        };
        engine.embed(&scope(Device::Cpu), &input(32)).unwrap();
        log.lock().unwrap().fail_stop = true;
        assert!(matches!(
            engine.embed(&scope(Device::Gpu), &input(32)),
            Err(NativeFailure::HardStop(_))
        ));
        assert_eq!(
            log.lock().unwrap().events,
            ["spawn-32", "compile", "infer", "stop"]
        );
        assert_eq!(
            engine.cached.as_ref().unwrap().scope,
            "cpu",
            "keep the unresolved owner"
        );
    }

    #[test]
    fn alternating_bucket_batch_compiles_each_shape_once_and_preserves_vector_order() {
        let log = Arc::new(Mutex::new(WorkerLog::default()));
        let engine = NativeEngine {
            factory: FakeFactory(Arc::clone(&log)),
            cached: None,
        };
        let mut core = ServingCore::fixture(
            FixtureTokenizer::default(),
            engine,
            vec![scope(Device::Cpu)],
        );
        let texts: Vec<String> = (0..64)
            .map(|i| {
                if i % 2 == 0 {
                    "a".repeat(10)
                } else {
                    "b".repeat(40)
                }
            })
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        for expected_compiles in [2, 1] {
            log.lock().unwrap().events.clear();
            let reply = core.handle(&body("cpu", &refs), &[1; 32]).unwrap();
            let parsed: serde_json::Value = serde_json::from_slice(&reply.body).unwrap();
            for (i, row) in parsed["data"].as_array().unwrap().iter().enumerate() {
                let (count, bucket, component) = if i % 2 == 0 {
                    (12, 32, 97)
                } else {
                    (42, 64, 98)
                };
                assert_eq!(row["index"], i);
                assert_eq!(row["token_count"], count);
                assert_eq!(row["sequence_bucket"], bucket);
                assert_eq!(row["embedding"][component], 1.0);
            }
            let events = &log.lock().unwrap().events;
            assert_eq!(
                events.iter().filter(|event| *event == "compile").count(),
                expected_compiles
            );
            assert_eq!(events.iter().filter(|event| *event == "infer").count(), 64);
        }
    }

    #[test]
    fn tokenizer_failure_stops_idle_worker_and_failed_cleanup_retains_its_owner() {
        for cleanup_fails in [false, true] {
            let log = Arc::new(Mutex::new(WorkerLog::default()));
            let engine = NativeEngine {
                factory: FakeFactory(Arc::clone(&log)),
                cached: None,
            };
            let mut core = ServingCore::fixture(
                FixtureTokenizer::default(),
                engine,
                vec![scope(Device::Cpu)],
            );
            core.handle(&body("cpu", &["text"]), &[0; 32]).unwrap();
            log.lock().unwrap().fail_stop = cleanup_fails;
            core.tokenizer
                .1
                .store(true, std::sync::atomic::Ordering::SeqCst);
            assert!(matches!(
                core.handle(&body("cpu", &["next"]), &[0; 32]),
                Err(Failure::HardStop(_))
            ));
            assert_eq!(core.engine.cached.is_some(), cleanup_fails);
            assert!(matches!(
                core.handle(&body("cpu", &["retry"]), &[0; 32]),
                Err(Failure::HardStop(_))
            ));
            assert_eq!(
                log.lock().unwrap().events,
                ["spawn-32", "compile", "infer", "stop"]
            );
        }
    }
}
