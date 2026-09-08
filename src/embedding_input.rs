//! Deterministic document payloads, before the encoder's document prefix.
//!
//! Citation identity belongs to the exact statement body. Vector identity
//! belongs to the rendered payload together with its `VectorSpec`, which
//! separately pins the prefix and encoder contract.

use sha2::Digest as _;

/// Includes enclosing headings and, for table rows, the full table header.
/// The caller supplies context derived from the same privacy-filtered source
/// as the body. Neither payload component is normalized or truncated here.
pub fn render(body: &str, context: &str) -> String {
    if context.is_empty() {
        body.to_string()
    } else {
        format!("{context}\n\n{body}")
    }
}

/// Hashes the exact unprefixed payload under a separate, versioned domain.
/// The shared store's `VectorSpec` binds the prefix, so `(spec, hash)` names
/// the complete encoder input without coupling the catalog to one spec.
pub fn hash(payload: &str) -> String {
    let mut hash = sha2::Sha256::new();
    hash.update(b"cfetch-document-payload-v1\0");
    hash.update(payload.as_bytes());
    crate::hashing::hex_lower(hash.finalize())
}
