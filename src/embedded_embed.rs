//! Isolated model diagnostics and query-local reranking via FastEmbed.
//!
//! Catalogue embeddings do not implement cfetch's admitted semantic pipeline.
//! Even a matching model name and width do not prove compatible vectors.
//! This module never supplies `EmbedClient` or writes the shared vector store.

#![cfg(feature = "embedded-embeddings")]

use std::path::PathBuf;

/// Default reranking model: Jina reranker v2 multilingual.
/// Cross-encoder that scores query-document pairs locally.
#[allow(dead_code)]
pub const DEFAULT_RERANKER_MODEL: fastembed::RerankerModel =
    fastembed::RerankerModel::JINARerankerV2BaseMultiligual;

/// Human-readable name of the default reranker (for status output).
pub const DEFAULT_RERANKER_MODEL_NAME: &str = "jina-reranker-v2-base-multilingual";

/// Where fastembed caches downloaded models.
pub fn cache_dir() -> PathBuf {
    // FastEmbed's HF transport gives HF_HOME precedence over with_cache_dir.
    // Inspection and loading must agree on the directory actually used.
    std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| crate::paths::state_dir().join("models"))
}

/// Resolve only exact catalogue variant names. In particular, quantized
/// variants must not collapse into their similarly named floating models.
pub fn parse_model(name: &str) -> Result<fastembed::EmbeddingModel, String> {
    available_models()
        .into_iter()
        .find(|info| format!("{:?}", info.model).eq_ignore_ascii_case(name))
        .map(|info| info.model)
        .ok_or_else(|| format!("unknown diagnostic model {name:?}; run `cfetch embed-model list`"))
}

/// An in-process embedding backend using fastembed.
/// Handles model download (first use), tokenization, and inference.
pub struct EmbeddedEmbedder {
    model: fastembed::TextEmbedding,
}

impl EmbeddedEmbedder {
    /// Loads exactly the requested diagnostic model, downloading on first use.
    pub fn load(model_name: fastembed::EmbeddingModel) -> anyhow::Result<Self> {
        let model = fastembed::TextEmbedding::try_new(diagnostic_options(model_name))
            .map_err(|e| anyhow::anyhow!("load embedding model: {e}"))?;
        Ok(Self { model })
    }

    /// Runs the selected catalogue pipeline. Width and pooling are model
    /// specific; its tokenizer may truncate. This is never canonical output.
    pub fn embed(&mut self, texts: Vec<String>) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model
            .embed(texts, None)
            .map_err(|e| anyhow::anyhow!("embed: {e}"))
    }
}

fn diagnostic_options(model: fastembed::EmbeddingModel) -> fastembed::TextInitOptions {
    fastembed::TextInitOptions::new(model)
        .with_cache_dir(cache_dir())
        .with_show_download_progress(true)
}

/// An in-process reranker using fastembed.
/// Scores query-document pairs with a local cross-encoder.
#[allow(dead_code)]
pub struct EmbeddedReranker {
    model: fastembed::TextRerank,
}

#[allow(dead_code)]
impl EmbeddedReranker {
    /// Loads the reranker model, downloading it on first use.
    pub fn load() -> anyhow::Result<Self> {
        let model = fastembed::TextRerank::try_new(
            fastembed::RerankInitOptions::new(DEFAULT_RERANKER_MODEL)
                .with_cache_dir(cache_dir())
                .with_show_download_progress(true),
        )
        .map_err(|e| anyhow::anyhow!("load reranker: {e}"))?;
        Ok(Self { model })
    }

    /// Reranks documents against a query. Returns (index, score) sorted by
    /// relevance (best first). If `return_text` is true, includes content.
    pub fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        return_text: bool,
    ) -> anyhow::Result<Vec<(usize, f32, Option<String>)>> {
        let results = self
            .model
            .rerank(query, documents, return_text, None)
            .map_err(|e| anyhow::anyhow!("rerank: {e}"))?;
        Ok(results
            .into_iter()
            .map(|r| (r.index, r.score, r.document))
            .collect())
    }

    /// Scores every document against the query, one score per input document
    /// in INPUT order — the same contract as `RerankClient::rank`.
    pub fn rank(&mut self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<f32>> {
        let results = self
            .model
            .rerank(query, documents, false, None)
            .map_err(|e| anyhow::anyhow!("rerank: {e}"))?;
        let mut scores = vec![f32::MIN; documents.len()];
        for r in results {
            if let Some(slot) = scores.get_mut(r.index) {
                *slot = r.score;
            }
        }
        Ok(scores)
    }
}

/// Use the linked library's catalogue so every listed variant is selectable.
pub fn available_models() -> Vec<fastembed::ModelInfo<fastembed::EmbeddingModel>> {
    let mut models = fastembed::TextEmbedding::list_supported_models();
    models.sort_by_key(|info| format!("{:?}", info.model));
    models
}

/// Lists all available reranker models.
#[allow(dead_code)]
pub fn available_rerankers() -> Vec<(&'static str, &'static str)> {
    vec![
        ("BGERerankerBase", "English cross-encoder"),
        ("BGERerankerV2M3", "Multilingual cross-encoder"),
        ("JinaRerankerV2BaseMultilingual", "Multilingual (default)"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_matches_the_runtime_override_or_state_default() {
        let expected = std::env::var("HF_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| crate::paths::state_dir().join("models"));
        assert_eq!(cache_dir(), expected);
    }

    #[test]
    fn every_listed_model_selects_its_own_runtime_options() {
        for info in available_models() {
            let requested = parse_model(&format!("{:?}", info.model)).unwrap();
            assert_eq!(diagnostic_options(requested).model_name, info.model);
        }
    }

    #[test]
    fn quantized_models_remain_distinct_and_substring_matches_are_refused() {
        use fastembed::EmbeddingModel::{EmbeddingGemma300M, EmbeddingGemma300MQ4};
        assert_eq!(
            parse_model("EmbeddingGemma300M").unwrap(),
            EmbeddingGemma300M
        );
        assert_eq!(
            parse_model("embeddinggemma300mq4").unwrap(),
            EmbeddingGemma300MQ4
        );
        assert!(parse_model("EmbeddingGemma300M-made-up").is_err());
    }
}
