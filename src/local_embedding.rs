//! Offline CPU embeddings from a content-verified model pack. The pack's
//! identity includes graph, weights, tokenizer and pipeline. It never enters
//! the old cross-device vector namespace or downloads a moving model revision.

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

pub const PIPELINE: &str = "embeddinggemma-local-cpu-v1-fastembed6.0.2-ort2.0.0rc13";
const FILES: &[&str] = &[
    "model.onnx",
    "model.onnx_data",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    pipeline: String,
    model: String,
    files: BTreeMap<String, FileIdentity>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    bytes: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Qualification {
    pipeline: String,
    manifest_sha256: String,
    reference_cosines: Vec<f64>,
    token_counts: Vec<usize>,
    semantic_ordering_passed: bool,
}

fn manifest(dir: &Path) -> anyhow::Result<(Manifest, String)> {
    let bytes = std::fs::read(dir.join("manifest.json")).context("read local model manifest")?;
    ensure!(
        bytes.len() <= 16 * 1024,
        "local model manifest is too large"
    );
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(
        manifest.pipeline == PIPELINE,
        "unsupported local embedding pipeline"
    );
    ensure!(
        manifest.model == crate::embedding_profile::MODEL,
        "local model must be EmbeddingGemma"
    );
    ensure!(
        manifest.files.len() == FILES.len()
            && FILES.iter().all(|n| manifest.files.contains_key(*n)),
        "local model manifest must name exactly the six required files"
    );
    for identity in manifest.files.values() {
        ensure!(
            identity.bytes > 0 && identity.bytes <= 2 * 1024 * 1024 * 1024,
            "local model file has an invalid size"
        );
        ensure!(
            identity.sha256.len() == 64
                && identity
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
            "local model file has an invalid digest"
        );
    }
    Ok((manifest, format!("{:x}", Sha256::digest(&bytes))))
}

pub fn profile_id(dir: &Path) -> anyhow::Result<String> {
    Ok(format!("local-cpu-v1-{}", manifest(dir)?.1))
}

/// Cheap readiness check; full file verification happens once when loading.
pub fn available(dir: &Path) -> anyhow::Result<()> {
    ensure!(
        cfg!(feature = "embedded-embeddings"),
        "this binary needs the embedded-embeddings feature for local vectors"
    );
    let (_, digest) = manifest(dir)?;
    let proof: Qualification = serde_json::from_slice(
        &std::fs::read(dir.join("qualification.json"))
            .context("local model has not passed `cfetch qualify-model`")?,
    )?;
    ensure!(
        proof.pipeline == PIPELINE && proof.manifest_sha256 == digest,
        "local model qualification belongs to another pipeline or manifest"
    );
    ensure!(
        proof.semantic_ordering_passed
            && proof.reference_cosines.len() == proof.token_counts.len()
            && proof
                .reference_cosines
                .iter()
                .all(|v| v.is_finite() && *v >= 0.999)
            && proof.token_counts.contains(&13)
            && proof.token_counts.iter().any(|n| *n >= 1900),
        "local model qualification is incomplete or failed"
    );
    Ok(())
}

#[cfg(feature = "embedded-embeddings")]
mod cpu {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    pub struct Model {
        model: fastembed::TextEmbedding,
        ready_at: Instant,
    }
    type Handle = Arc<Mutex<Model>>;
    static CACHE: OnceLock<Mutex<BTreeMap<String, Handle>>> = OnceLock::new();

    fn read_checked(dir: &Path, name: &str, manifest: &Manifest) -> anyhow::Result<Vec<u8>> {
        let expected = &manifest.files[name];
        let path = dir.join(name);
        let meta = std::fs::symlink_metadata(&path)?;
        ensure!(
            meta.is_file() && meta.len() == expected.bytes,
            "local model {name}: size or file type changed"
        );
        let bytes = std::fs::read(&path)?;
        ensure!(
            bytes.len() as u64 == expected.bytes
                && format!("{:x}", Sha256::digest(&bytes)) == expected.sha256,
            "local model {name}: content digest changed"
        );
        Ok(bytes)
    }

    pub fn load(dir: &Path, require_qualification: bool) -> anyhow::Result<Handle> {
        if require_qualification {
            available(dir)?;
        }
        let (manifest, digest) = manifest(dir)?;
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .map_err(|_| anyhow::anyhow!("local model cache poisoned"))?;
        if let Some(model) = cache.get(&digest) {
            return Ok(model.clone());
        }
        let tokenizer = fastembed::TokenizerFiles {
            tokenizer_file: read_checked(dir, "tokenizer.json", &manifest)?,
            config_file: read_checked(dir, "config.json", &manifest)?,
            special_tokens_map_file: read_checked(dir, "special_tokens_map.json", &manifest)?,
            tokenizer_config_file: read_checked(dir, "tokenizer_config.json", &manifest)?,
        };
        let mut source = fastembed::UserDefinedEmbeddingModel::new(
            read_checked(dir, "model.onnx", &manifest)?,
            tokenizer,
        )
        .with_external_initializer(
            "model.onnx_data".into(),
            read_checked(dir, "model.onnx_data", &manifest)?,
        )
        .with_pooling(fastembed::Pooling::Mean);
        source.output_key = Some(fastembed::OutputKey::ByName("sentence_embedding"));
        let mut model = fastembed::TextEmbedding::try_new_from_user_defined(
            source,
            fastembed::InitOptionsUserDefined::new()
                .with_max_length(2048)
                .with_intra_threads(2),
        )?;
        // Count untruncated tokens before every call; never claim full coverage
        // of an input which the library silently shortened.
        model
            .tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        let model = Arc::new(Mutex::new(Model {
            model,
            ready_at: Instant::now(),
        }));
        cache.insert(digest, model.clone());
        Ok(model)
    }

    impl Model {
        fn tokens(&self, text: &str) -> anyhow::Result<usize> {
            Ok(self
                .model
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
                .len())
        }
        pub fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
            for text in texts {
                ensure!(
                    text.len() <= 128 * 1024 && self.tokens(text)? <= 2048,
                    "local embedding input exceeds 2048 tokens or 128 KiB; source text was not truncated"
                );
            }
            let mut vectors = Vec::with_capacity(texts.len());
            for text in texts {
                // Serial inputs give stable semantics independent of batch padding.
                // Two CPU threads and equal active/cooldown time bound background load.
                std::thread::sleep(self.ready_at.saturating_duration_since(Instant::now()));
                let started = Instant::now();
                let output = self.model.embed([*text], Some(1));
                self.ready_at = Instant::now() + started.elapsed();
                let mut output = output?;
                ensure!(
                    output.len() == 1,
                    "local model returned an unexpected row count"
                );
                let vector = output.remove(0);
                ensure!(
                    vector.len() == 768 && crate::vectors::degenerate(&vector).is_none(),
                    "local model returned invalid output"
                );
                vectors.push(vector);
            }
            Ok(vectors)
        }
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Reference {
        text: String,
        tokens: usize,
        vector: Vec<f32>,
    }

    fn cosine(a: &[f32], b: &[f32]) -> anyhow::Result<f64> {
        ensure!(
            a.len() == b.len()
                && crate::vectors::degenerate(a).is_none()
                && crate::vectors::degenerate(b).is_none(),
            "invalid reference vector"
        );
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        Ok(dot / norm(a) / norm(b))
    }

    pub fn qualify(dir: &Path, reference: &Path) -> anyhow::Result<()> {
        let reference = std::fs::read(reference)?;
        ensure!(
            reference.len() < 2 * 1024 * 1024,
            "qualification input too large"
        );
        let references: Vec<Reference> = serde_json::from_slice(&reference)?;
        ensure!(
            (2..=8).contains(&references.len()),
            "qualification needs 2..8 canonical reference rows"
        );
        let loaded = load(dir, false)?;
        let mut model = loaded
            .lock()
            .map_err(|_| anyhow::anyhow!("local model poisoned"))?;
        let mut proof = Qualification {
            pipeline: PIPELINE.into(),
            manifest_sha256: manifest(dir)?.1,
            reference_cosines: Vec::new(),
            token_counts: Vec::new(),
            semantic_ordering_passed: false,
        };
        for row in references {
            let tokens = model.tokens(&row.text)?;
            ensure!(
                tokens == row.tokens,
                "tokenizer differs from canonical reference"
            );
            let vectors = model.embed(&[&row.text])?;
            proof
                .reference_cosines
                .push(cosine(&vectors[0], &row.vector)?);
            proof.token_counts.push(tokens);
        }
        let texts = [
            format!("{}cat animal", crate::embedding_profile::QUERY_PREFIX),
            format!(
                "{}A cat is a small domesticated feline animal.",
                crate::embedding_profile::DOCUMENT_PREFIX
            ),
            format!(
                "{}A violin is a musical instrument with four strings.",
                crate::embedding_profile::DOCUMENT_PREFIX
            ),
        ];
        let vectors = model.embed(&texts.iter().map(String::as_str).collect::<Vec<_>>())?;
        proof.semantic_ordering_passed =
            cosine(&vectors[0], &vectors[1])? > cosine(&vectors[0], &vectors[2])?;
        ensure!(
            proof.semantic_ordering_passed
                && proof.reference_cosines.iter().all(|v| *v >= 0.999)
                && proof.token_counts.contains(&13)
                && proof.token_counts.iter().any(|n| *n >= 1900),
            "local model failed semantic or short/long canonical parity: {}",
            serde_json::to_string(&proof)?
        );
        let bytes = serde_json::to_vec_pretty(&proof)?;
        use std::io::Write as _;
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join("qualification.json"))?;
        output.write_all(&bytes)?;
        output.sync_all()?;
        println!("{}", String::from_utf8(bytes)?);
        // Avoid a model object keeping inference active after this explicit check.
        model.ready_at = Instant::now() + Duration::ZERO;
        Ok(())
    }
}
#[cfg(feature = "embedded-embeddings")]
pub use cpu::{Model, load, qualify};
