//! Native package schema 2. The compiled client is the admission authority;
//! the independently built sibling rechecks the parent's exact private permit.
//! The final client is outside `inference/`, so this inventory has no hash cycle.

use anyhow::{Context as _, ensure};
use serde::Deserialize as _;
use serde_json::{Map, Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::embedding_profile as profile;
use crate::inference_governor::{Device, Policy};
use crate::native_worker::{
    ExecutionExpectation, Identity, Operation, PinnedFile, Pooling, Precision,
};

pub(crate) const PAYLOAD_DIRECTORY: &str = "inference";
const PACKAGE_MANIFEST: &str = "package-manifest.json";
const RUNTIME_MANIFEST: &str = "runtime-manifest.json";
const MAX_JSON: u64 = 1024 * 1024;
const MAX_RUNTIME_JSON: u64 = 16 * MAX_JSON;
const MAX_FILES: usize = 4096;
const MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const POLICY_FILE: &str = "/var/lib/cfetch/inference/policy.json";

pub(crate) struct VerifiedScope {
    pub(crate) id: String,
    pub(crate) execution: Value,
    pub(crate) identity: Identity,
    pub(crate) compile: Operation,
    pub(crate) signer: ed25519_dalek::SigningKey,
}

pub(crate) struct VerifiedPackage {
    pub(crate) scopes: Vec<VerifiedScope>,
    pub(crate) tokenizer: Vec<u8>,
    pub(crate) policy_sha256: String,
    pub(crate) request_budget: Duration,
    dispatcher: PinnedFile,
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// serde's typed structs reject duplicate fields; these retained artifact
/// schemas have dynamic property maps, so reject duplicates at every level
/// before performing any semantic validation or converting them to Value.
struct UniqueJson(Value);
impl<'de> serde::Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|value| UniqueJson(Value::Number(value)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJson(value.into()))
            }
            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut result = Vec::new();
                while let Some(UniqueJson(value)) = seq.next_element()? {
                    result.push(value);
                }
                Ok(UniqueJson(Value::Array(result)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut result = Map::new();
                while let Some((key, UniqueJson(value))) = map.next_entry::<String, UniqueJson>()? {
                    if result.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate JSON key"));
                    }
                }
                Ok(UniqueJson(Value::Object(result)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

fn parse(raw: &[u8]) -> anyhow::Result<Value> {
    let mut decoder = serde_json::Deserializer::from_slice(raw);
    let result = UniqueJson::deserialize(&mut decoder)?.0;
    decoder.end()?;
    Ok(result)
}

fn object<'a>(value: &'a Value, fields: &[&str]) -> anyhow::Result<&'a Map<String, Value>> {
    let result = value
        .as_object()
        .context("manifest field must be an object")?;
    ensure!(
        result.len() == fields.len() && fields.iter().all(|field| result.contains_key(*field)),
        "manifest object fields differ from the exact native schema"
    );
    Ok(result)
}
fn string<'a>(map: &'a Map<String, Value>, key: &str) -> anyhow::Result<&'a str> {
    let value = map
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("{key} must be a string"))?;
    ensure!(
        !value.is_empty() && value.len() <= 4096 && !value.chars().any(char::is_control),
        "invalid {key} string"
    );
    Ok(value)
}
fn hash<'a>(map: &'a Map<String, Value>, key: &str) -> anyhow::Result<&'a str> {
    let value = string(map, key)?;
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid {key} digest"
    );
    Ok(value)
}
fn array<'a>(map: &'a Map<String, Value>, key: &str) -> anyhow::Result<&'a Vec<Value>> {
    map.get(key)
        .and_then(Value::as_array)
        .with_context(|| format!("{key} must be an array"))
}
fn count(map: &Map<String, Value>, key: &str) -> anyhow::Result<u64> {
    map.get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("{key} must be an unsigned integer"))
}

fn read(path: &Path, limit: u64) -> anyhow::Result<Vec<u8>> {
    let fd = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    let file = std::fs::File::from(fd);
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && (1..=limit).contains(&metadata.len()),
        "manifest input must be a bounded regular file"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        !bytes.is_empty() && bytes.len() as u64 == metadata.len(),
        "manifest changed while being read"
    );
    Ok(bytes)
}

fn relative(root: &Path, name: &str) -> anyhow::Result<PathBuf> {
    ensure!(
        !name.is_empty()
            && name.len() <= 512
            && !name.contains('\\')
            && !name.chars().any(char::is_control)
            && name
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "noncanonical package-relative path"
    );
    let mut current = root.to_path_buf();
    for part in name.split('/') {
        current.push(part);
        ensure!(
            !std::fs::symlink_metadata(&current)?.is_symlink(),
            "package contains a symlink"
        );
    }
    ensure!(current.is_file(), "package entry is not a regular file");
    Ok(current)
}

fn scan(root: &Path) -> anyhow::Result<BTreeSet<String>> {
    let mut result = BTreeSet::new();
    let mut directories = vec![root.to_path_buf()];
    let mut visited = 0usize;
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            visited += 1;
            ensure!(
                visited <= MAX_FILES * 2,
                "package entry count exceeds native bound"
            );
            let kind = entry.file_type()?;
            ensure!(!kind.is_symlink(), "package contains a symlink");
            if kind.is_dir() {
                directories.push(entry.path());
                continue;
            }
            ensure!(kind.is_file(), "package contains a nonregular entry");
            let path = entry.path();
            let name = path
                .strip_prefix(root)?
                .to_str()
                .context("package path is not UTF-8")?;
            if name == PACKAGE_MANIFEST || name == RUNTIME_MANIFEST {
                continue;
            }
            relative(root, name)?;
            result.insert(name.to_string());
            ensure!(result.len() <= MAX_FILES, "package contains too many files");
        }
    }
    Ok(result)
}

fn inventory(root: &Path, files: &[Value]) -> anyhow::Result<BTreeMap<String, PinnedFile>> {
    ensure!(
        (1..=MAX_FILES).contains(&files.len()),
        "invalid native inventory length"
    );
    let mut result = BTreeMap::new();
    let mut previous = String::new();
    let mut total = 0u64;
    for entry in files {
        let entry = object(entry, &["path", "sha256", "bytes", "executable"])?;
        let name = string(entry, "path")?;
        ensure!(
            name > previous.as_str() && name != PACKAGE_MANIFEST && name != RUNTIME_MANIFEST,
            "native inventory must be unique, sorted and exclude its two root manifests"
        );
        previous = name.to_string();
        let file = PinnedFile {
            path: relative(root, name)?,
            sha256: hash(entry, "sha256")?.into(),
            bytes: count(entry, "bytes")?,
            executable: entry["executable"]
                .as_bool()
                .context("inventory executable must be boolean")?,
        };
        total = total
            .checked_add(file.bytes)
            .context("native inventory byte overflow")?;
        ensure!(
            total <= MAX_BYTES,
            "native package exceeds its byte ceiling"
        );
        file.verify()?;
        result.insert(name.into(), file);
    }
    ensure!(
        scan(root)? == result.keys().cloned().collect(),
        "package contains missing or unbound files"
    );
    Ok(result)
}

fn pinned<'a>(
    files: &'a BTreeMap<String, PinnedFile>,
    name: &str,
) -> anyhow::Result<&'a PinnedFile> {
    files
        .get(name)
        .with_context(|| format!("required file is not in the immutable inventory: {name}"))
}

fn source_files() -> Value {
    json!({
        "model.safetensors":profile::MODEL_WEIGHTS_SHA256,
        "2_Dense/model.safetensors":profile::DENSE_2_WEIGHTS_SHA256,
        "3_Dense/model.safetensors":profile::DENSE_3_WEIGHTS_SHA256,
        "tokenizer.json":profile::TOKENIZER_JSON_SHA256,"tokenizer.model":profile::TOKENIZER_MODEL_SHA256,
        "tokenizer_config.json":profile::TOKENIZER_CONFIG_SHA256,"config.json":profile::MODEL_CONFIG_SHA256,
        "special_tokens_map.json":profile::SPECIAL_TOKENS_MAP_SHA256,
        "modules.json":"5b5649645fb756dad1a8e2efe7872d3bb32bc00b93c95f276dd17f474eedccdc",
        "sentence_bert_config.json":"5ea26221ce733ace29a3897360e7c6ac8816b2ca0f7306657d69e594fece7325",
        "1_Pooling/config.json":"35bbd47d7fdf1e378db6130bcc668b09d1aa67a7bbf7c8f89a9c71f4cc8ebcc6",
        "2_Dense/config.json":"0661e5e0b67b8f8408ab31ab5d073a78972fc1dc24a49992a64796557e4f9e53",
        "3_Dense/config.json":"8c4575c49353d63fb907878856ba94384635c3b2711fd5b7439e7f71888c66fc"
    })
}
const LEGAL_FILES: &[(&str, &str)] = &[
    (
        "GEMMA_TERMS.txt",
        "b3609ee6ac1616e087bb5fe53356202eff274aecde124ab1e065fac5e0eb1f2e",
    ),
    (
        "GEMMA_PROHIBITED_USE_POLICY.txt",
        "14a821208c6a174b08c942112c132a4a802af4efd100ba5ec4086dd9822bc698",
    ),
    (
        "MODEL_USE_RESTRICTIONS.txt",
        "88838c95fef08ecd2e4e8f35e5227137562d0c7b95f9120e5cc678ce8d809bd9",
    ),
    (
        "MODEL_MODIFICATIONS.txt",
        "4a49079058f0989a9a9eab3a936f16beee2388c1e3802b43cc4edc421f650aed",
    ),
    (
        "NOTICE",
        "66f856d7da72797f528fca46b7c80634ab481f917bfe020960e123d84b19f75f",
    ),
];

struct Artifact {
    model: PinnedFile,
    weights: PinnedFile,
    tokenizer: PinnedFile,
    openvino_version: String,
}

fn artifact(
    root: &Path,
    name: &str,
    expected_hash: &str,
    files: &BTreeMap<String, PinnedFile>,
) -> anyhow::Result<Artifact> {
    let bound = pinned(files, name)?;
    ensure!(
        bound.sha256 == expected_hash,
        "artifact manifest differs from the package binding"
    );
    let raw = read(&relative(root, name)?, MAX_JSON)?;
    ensure!(
        digest(&raw) == expected_hash,
        "artifact manifest changed during validation"
    );
    let document = parse(&raw)?;
    let doc = object(
        &document,
        &[
            "schema_version",
            "artifact_format",
            "source",
            "semantic_pipeline",
            "sequence_buckets",
            "graph",
            "tokenizer",
            "legal",
            "files",
            "conversion",
        ],
    )?;
    ensure!(
        count(doc, "schema_version")? == 1
            && string(doc, "artifact_format")? == "openvino-ir-dynamic-sequence-static-buckets-v1",
        "unsupported canonical artifact schema"
    );
    ensure!(
        doc["source"]
            == json!({"model":profile::MODEL,"revision":profile::MODEL_REVISION,
        "acquisition":{"repository":"unsloth/embeddinggemma-300m","revision":"bfa3c846ac738e62aa61806ef9112d34acb1dc5a","mode":"public-byte-identical-mirror"},
        "files":source_files()}),
        "artifact source is not the exact frozen Gemma revision"
    );
    ensure!(
        doc["semantic_pipeline"]
            == json!({"dimensions":profile::DIMENSIONS,"pooling":profile::POOLING,
        "dense_2":"linear-768x3072-identity","dense_3":"linear-3072x768-identity","normalization":"l2",
        "truncation":"disabled","padding":"right-attention-mask-excludes-padding"})
            && doc["sequence_buckets"] == json!(profile::SEQUENCE_BUCKETS),
        "artifact semantic pipeline changed"
    );
    ensure!(
        doc["legal"]
            == json!({"terms_url":"https://ai.google.dev/gemma/terms",
        "prohibited_use_policy_url":"https://ai.google.dev/gemma/prohibited_use_policy",
        "terms_file":"GEMMA_TERMS.txt","prohibited_use_policy_file":"GEMMA_PROHIBITED_USE_POLICY.txt",
        "use_restrictions_file":"MODEL_USE_RESTRICTIONS.txt","modifications_file":"MODEL_MODIFICATIONS.txt","notice_file":"NOTICE"}),
        "artifact legal contract changed"
    );
    let conversion = object(
        &doc["conversion"],
        &[
            "recipe",
            "export",
            "weight_storage",
            "openvino",
            "safetensors",
            "torch",
            "transformers",
        ],
    )?;
    ensure!(
        string(conversion, "recipe")? == "packages/openvino/convert.py"
            && string(conversion, "export")?
                == "torch-export-bounded-dynamic-sequence-1-to-2048-unit-reduction-matmul-rewrite-v1"
            && matches!(string(conversion, "weight_storage")?, "f16" | "f32"),
        "artifact conversion recipe changed"
    );
    for key in ["openvino", "safetensors", "torch", "transformers"] {
        string(conversion, key)?;
    }
    let graph = object(
        &doc["graph"],
        &["xml", "bin", "input_ids", "attention_mask", "output"],
    )?;
    ensure!(
        string(graph, "input_ids")? == "input_ids"
            && string(graph, "attention_mask")? == "attention_mask"
            && string(graph, "output")? == "embedding",
        "native graph does not expose the canonical complete embedding pipeline"
    );
    let tokenizer = object(
        &doc["tokenizer"],
        &[
            "json",
            "sha256",
            "pad_token",
            "pad_token_id",
            "bos_token",
            "bos_token_id",
            "eos_token",
            "eos_token_id",
            "add_bos_token",
            "add_eos_token",
        ],
    )?;
    let expected = json!({"json":string(tokenizer,"json")?,"sha256":profile::TOKENIZER_JSON_SHA256,
        "pad_token":"<pad>","pad_token_id":0,"bos_token":"<bos>","bos_token_id":2,"eos_token":"<eos>","eos_token_id":1,
        "add_bos_token":true,"add_eos_token":true});
    ensure!(
        doc["tokenizer"] == expected,
        "artifact tokenizer contract changed"
    );
    let parent = Path::new(name)
        .parent()
        .context("artifact has no directory")?;
    let mut artifacts = BTreeMap::new();
    for entry in array(doc, "files")? {
        let entry = object(entry, &["path", "sha256", "bytes"])?;
        let relative_name = string(entry, "path")?;
        relative(&root.join(parent), relative_name)?;
        let key = parent
            .join(relative_name)
            .to_str()
            .context("artifact path is not UTF-8")?
            .to_string();
        let file = pinned(files, &key)?;
        ensure!(
            file.sha256 == hash(entry, "sha256")? && file.bytes == count(entry, "bytes")?,
            "artifact bytes differ from full inventory"
        );
        ensure!(
            artifacts
                .insert(relative_name.to_string(), file.clone())
                .is_none(),
            "duplicate artifact file"
        );
    }
    let mut required: BTreeSet<String> =
        LEGAL_FILES.iter().map(|(name, _)| (*name).into()).collect();
    for name in [
        string(graph, "xml")?,
        string(graph, "bin")?,
        string(tokenizer, "json")?,
    ] {
        ensure!(
            required.insert(name.into()),
            "artifact graph/tokenizer/legal files overlap"
        );
    }
    ensure!(
        artifacts.keys().cloned().collect::<BTreeSet<_>>() == required,
        "artifact file set is not exact"
    );
    for (name, hash) in LEGAL_FILES {
        ensure!(
            artifacts[*name].sha256 == *hash,
            "artifact legal bytes changed"
        );
    }
    let tokenizer = artifacts[string(tokenizer, "json")?].clone();
    ensure!(
        tokenizer.sha256 == profile::TOKENIZER_JSON_SHA256,
        "artifact tokenizer bytes changed"
    );
    Ok(Artifact {
        model: artifacts[string(graph, "xml")?].clone(),
        weights: artifacts[string(graph, "bin")?].clone(),
        tokenizer,
        openvino_version: string(conversion, "openvino")?.into(),
    })
}

struct Runtime {
    files: BTreeMap<String, PinnedFile>,
    dispatcher: PinnedFile,
    library: PinnedFile,
    build: String,
    openvino_version: String,
    plugins: BTreeMap<String, PinnedFile>,
}

fn runtime(root: &Path, expected: &str) -> anyhow::Result<Runtime> {
    let raw = read(&relative(root, RUNTIME_MANIFEST)?, MAX_RUNTIME_JSON)?;
    ensure!(
        digest(&raw) == expected,
        "native runtime manifest digest changed"
    );
    let value = parse(&raw)?;
    let doc = object(
        &value,
        &[
            "schema_version",
            "format",
            "target",
            "source_revision",
            "cargo_lock_sha256",
            "rustc",
            "openvino_version",
            "openvino_build",
            "dispatcher",
            "openvino_library",
            "plugin_configuration",
            "plugins",
            "files",
        ],
    )?;
    ensure!(
        count(doc, "schema_version")? == 2
            && string(doc, "format")? == "cfetch-native-openvino-v2"
            && string(doc, "target")? == "x86_64-unknown-linux-gnu",
        "unsupported native runtime target or schema"
    );
    let revision = string(doc, "source_revision")?;
    ensure!(
        revision.len() == 40
            && revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid native source revision"
    );
    hash(doc, "cargo_lock_sha256")?;
    ensure!(
        string(doc, "rustc")?.starts_with("rustc "),
        "native runtime lacks compiler provenance"
    );
    let files = inventory(root, array(doc, "files")?)?;
    let dispatcher_name = string(doc, "dispatcher")?;
    ensure!(
        !dispatcher_name.contains('/'),
        "dispatcher must be a payload-root basename"
    );
    let dispatcher = pinned(&files, dispatcher_name)?.clone();
    ensure!(dispatcher.executable, "native dispatcher is not executable");
    let library = pinned(&files, string(doc, "openvino_library")?)?.clone();
    ensure!(
        library.path != dispatcher.path,
        "native runtime library overlaps dispatcher"
    );
    ensure!(
        string(doc, "plugin_configuration")? == "single-device-absolute-library-v1",
        "native runtime must use the fixed single-device plugin configuration"
    );
    let plugin_doc = object(&doc["plugins"], &["NPU", "GPU", "CPU"])?;
    let mut plugins = BTreeMap::new();
    let mut plugin_paths = BTreeSet::new();
    for device in ["NPU", "GPU", "CPU"] {
        let plugin = pinned(&files, string(plugin_doc, device)?)?.clone();
        ensure!(
            plugin.path != library.path
                && plugin.path != dispatcher.path
                && plugin_paths.insert(plugin.path.clone()),
            "native device plugins must be separate pinned libraries"
        );
        plugins.insert(device.into(), plugin);
    }
    Ok(Runtime {
        files,
        dispatcher,
        library,
        build: string(doc, "openvino_build")?.into(),
        openvino_version: string(doc, "openvino_version")?.into(),
        plugins,
    })
}

fn measured(path: &Path, expected: &str) -> anyhow::Result<PinnedFile> {
    use std::os::unix::fs::PermissionsExt as _;
    let metadata = std::fs::symlink_metadata(path)?;
    let file = PinnedFile {
        path: path.to_path_buf(),
        sha256: expected.into(),
        bytes: metadata.len(),
        executable: metadata.permissions().mode() & 0o111 != 0,
    };
    file.verify()?;
    Ok(file)
}

fn host(value: &Value) -> anyhow::Result<(Vec<PinnedFile>, String)> {
    let doc = object(value, &["system", "machine", "kernel_release", "files"])?;
    let uname = rustix::system::uname();
    ensure!(
        string(doc, "system")? == "Linux"
            && string(doc, "machine")? == "x86_64"
            && uname.sysname().to_bytes() == b"Linux"
            && uname.machine().to_bytes() == b"x86_64"
            && uname.release().to_bytes() == string(doc, "kernel_release")?.as_bytes(),
        "native package host/kernel binding changed"
    );
    let entries = array(doc, "files")?;
    ensure!(
        (3..=16).contains(&entries.len()),
        "host closure must bind policy and both C++ runtime dependencies"
    );
    let mut seen = BTreeSet::new();
    let mut files = Vec::new();
    let mut policy = None;
    for entry in entries {
        let entry = object(entry, &["path", "sha256"])?;
        let path = Path::new(string(entry, "path")?);
        ensure!(
            path == Path::new(POLICY_FILE)
                || [
                    "/usr/lib",
                    "/usr/lib64",
                    "/lib",
                    "/lib64",
                    "/opt/intel",
                    "/nix/store"
                ]
                .iter()
                .any(|prefix| path.starts_with(prefix)),
            "host dependency is outside the declared library directories"
        );
        ensure!(seen.insert(path.to_path_buf()), "duplicate host dependency");
        let file = measured(path, hash(entry, "sha256")?)?;
        if path == Path::new(POLICY_FILE) {
            policy = Some(file.sha256.clone());
        }
        files.push(file);
    }
    for soname in ["libstdc++.so.6", "libgcc_s.so.1"] {
        ensure!(
            files.iter().any(|file| file
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(
                    |name| name.strip_prefix(soname).is_some_and(|tail| tail.is_empty()
                        || tail
                            .strip_prefix('.')
                            .is_some_and(|suffix| suffix
                                .split('.')
                                .all(|part| !part.is_empty()
                                    && part.bytes().all(|b| b.is_ascii_digit()))))
                )),
            "host closure lacks resolved {soname}"
        );
    }
    Ok((
        files,
        policy.context("host closure must pin the installed governor policy")?,
    ))
}

fn request_budget(raw: &[u8]) -> anyhow::Result<Duration> {
    let policy: Policy = serde_json::from_slice(raw)?;
    policy.validate(Path::new("/var/lib/cfetch/inference"))?;
    let compile = u128::from(policy.lock_wait_ns)
        + u128::from(policy.operations.compile.max_duration_ns)
        + 2_000_000_000;
    let inference =
        u128::from(policy.lock_wait_ns) + u128::from(policy.operations.inference.max_duration_ns);
    let total = profile::SEQUENCE_BUCKETS.len() as u128 * compile
        + profile::MAX_WIRE_BATCH_SIZE as u128 * inference
        + 10_000_000_000;
    Ok(Duration::from_nanos(
        u64::try_from(total).context("native request budget overflow")?,
    ))
}

fn signing_key(file: &PinnedFile, expected: &str) -> anyhow::Result<ed25519_dalek::SigningKey> {
    let raw = read(&file.path, 65)?;
    ensure!(
        digest(&raw) == file.sha256,
        "attestation key changed after closure verification"
    );
    let raw = raw.strip_suffix(b"\n").unwrap_or(&raw);
    ensure!(
        raw.len() == 64
            && raw
                .iter()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)),
        "invalid private attestation key encoding"
    );
    let mut seed = [0u8; 32];
    for (byte, pair) in seed.iter_mut().zip(raw.as_chunks::<2>().0) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair)?, 16)?;
    }
    let key = ed25519_dalek::SigningKey::from_bytes(&seed);
    let actual: String = key
        .verifying_key()
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    ensure!(
        actual == expected,
        "private attestation key does not match the bound scope key"
    );
    Ok(key)
}

const SCOPE_FIELDS: &[&str] = &[
    "scope_id",
    "backend",
    "transport",
    "runtime",
    "compiler",
    "package_target",
    "artifact_source",
    "artifact_sha256",
    "internal_precision",
    "device_class",
    "device",
    "openvino_device",
    "native_compile",
    "required_openvino_properties",
    "required_execution_devices",
    "required_host",
    "placement_evidence_sha256",
    "supported_max_tokens",
    "supported_sequence_buckets",
    "supported_max_batch_size",
    "sequence_capability_evidence_sha256",
    "performance_evidence_sha256",
    "compatibility_report_sha256",
    "attestation_public_key",
    "attestation_private_key_file",
    "accelerated_placement",
];
const EXECUTION_FIELDS: &[&str] = &[
    "scope_id",
    "backend",
    "transport",
    "runtime",
    "compiler",
    "package_target",
    "artifact_source",
    "artifact_sha256",
    "internal_precision",
    "device_class",
    "device",
    "placement_evidence_sha256",
    "supported_max_tokens",
    "supported_sequence_buckets",
    "supported_max_batch_size",
    "sequence_capability_evidence_sha256",
    "performance_evidence_sha256",
    "compatibility_report_sha256",
    "accelerated_placement",
];

fn execution_expectation(
    entry: &Map<String, Value>,
    device: Device,
) -> anyhow::Result<ExecutionExpectation> {
    let properties = entry["required_openvino_properties"]
        .as_object()
        .context("device properties must be an object")?;
    let mut actual = BTreeMap::new();
    for (name, value) in properties {
        let text = if name.starts_with("NPU_") {
            ensure!(
                value.as_i64().is_some() || value.as_u64().is_some(),
                "NPU version properties must be JSON integers"
            );
            value.to_string()
        } else {
            value
                .as_str()
                .context("physical device properties must be strings")?
                .to_string()
        };
        actual.insert(name.clone(), text);
    }
    let result = ExecutionExpectation {
        properties: actual,
        devices: serde_json::from_value(entry["required_execution_devices"].clone())?,
    };
    result.validate(device)?;
    Ok(result)
}

fn permitted_scopes(entries: &[Value], ordered: &[String]) -> anyhow::Result<()> {
    ensure!(
        entries.len() == ordered.len()
            && entries
                .iter()
                .zip(ordered)
                .all(|(entry, id)| entry.get("scope_id").and_then(Value::as_str)
                    == Some(id.as_str())),
        "native package scope list differs from the exact parent permit"
    );
    Ok(())
}

fn load(root: &Path, expected: &str, ordered: &[String]) -> anyhow::Result<VerifiedPackage> {
    ensure!(
        root.is_absolute()
            && std::fs::canonicalize(root)?.as_os_str() == root.as_os_str()
            && root.file_name().and_then(|name| name.to_str()) == Some(PAYLOAD_DIRECTORY),
        "native payload must be the canonical inference directory"
    );
    ensure!(
        !ordered.is_empty() && ordered.len() <= crate::local_adapter::MAX_SCOPES,
        "invalid permitted native scope count"
    );
    let mut ids = BTreeSet::new();
    for id in ordered {
        crate::local_inference::validate_scope_id(id)?;
        ensure!(ids.insert(id), "duplicate permitted scope");
    }
    let raw = read(&relative(root, PACKAGE_MANIFEST)?, MAX_JSON)?;
    ensure!(
        digest(&raw) == expected,
        "native package differs from the parent's exact binding"
    );
    let value = parse(&raw)?;
    let doc = object(
        &value,
        &[
            "schema_version",
            "package_state",
            "profile_id",
            "profile_manifest_sha256",
            "admission_policy_sha256",
            "model",
            "model_revision",
            "artifact_manifest",
            "artifact_manifest_sha256",
            "runtime_manifest_sha256",
            "scopes",
        ],
    )?;
    ensure!(
        count(doc, "schema_version")? == 2 && string(doc, "package_state")? == "release",
        "native serving requires an exact release package"
    );
    for (key, value) in [
        ("profile_id", profile::PROFILE_ID),
        ("profile_manifest_sha256", profile::PROFILE_MANIFEST_SHA256),
        ("admission_policy_sha256", profile::ADMISSION_POLICY_SHA256),
        ("model", profile::MODEL),
        ("model_revision", profile::MODEL_REVISION),
    ] {
        ensure!(
            string(doc, key)? == value,
            "native package changed the frozen {key}"
        );
    }
    permitted_scopes(array(doc, "scopes")?, ordered)?;
    let runtime = runtime(root, hash(doc, "runtime_manifest_sha256")?)?;
    let artifact_hash = hash(doc, "artifact_manifest_sha256")?;
    let artifact = artifact(
        root,
        string(doc, "artifact_manifest")?,
        artifact_hash,
        &runtime.files,
    )?;
    ensure!(
        artifact.openvino_version == runtime.openvino_version,
        "conversion and native runtime OpenVINO versions differ"
    );
    let mut closure: Vec<PinnedFile> = runtime.files.values().cloned().collect();
    closure.push(measured(&root.join(PACKAGE_MANIFEST), expected)?);
    closure.push(measured(
        &root.join(RUNTIME_MANIFEST),
        hash(doc, "runtime_manifest_sha256")?,
    )?);
    let entries = array(doc, "scopes")?;
    ensure!(
        entries.len() == ordered.len(),
        "manifest scopes differ from the parent permit"
    );
    let mut keys = BTreeSet::new();
    let mut key_files = BTreeSet::new();
    let mut classes = BTreeSet::new();
    let mut previous_rank = 0;
    let mut policy_sha256: Option<String> = None;
    let mut scopes = Vec::new();
    for (entry, permitted) in entries.iter().zip(ordered) {
        let entry = object(entry, SCOPE_FIELDS)?;
        let id = string(entry, "scope_id")?;
        ensure!(
            id == permitted,
            "native scope ordering differs from compiled admission"
        );
        let class = string(entry, "device_class")?;
        let (device, device_name, rank) = match class {
            "npu" => (Device::Npu, "NPU", 0),
            "gpu" => (Device::Gpu, "GPU", 1),
            "cpu" => (Device::Cpu, "CPU", 2),
            _ => anyhow::bail!("invalid native device class"),
        };
        ensure!(
            rank >= previous_rank,
            "native scopes must be in NPU, GPU, CPU order"
        );
        previous_rank = rank;
        classes.insert(class);
        ensure!(
            string(entry, "backend")? == "openvino"
                && string(entry, "transport")? == "supervised-local"
                && string(entry, "openvino_device")? == device_name,
            "native scope cannot select an aggregate or fallback device"
        );
        ensure!(
            hash(entry, "artifact_sha256")? == artifact_hash
                && count(entry, "supported_max_tokens")? == profile::MAX_TOKENS as u64
                && entry["supported_sequence_buckets"] == json!(profile::SEQUENCE_BUCKETS)
                && count(entry, "supported_max_batch_size")? == profile::MAX_WIRE_BATCH_SIZE as u64
                && entry["accelerated_placement"] == true,
            "native scope lacks the full admitted shape/placement contract"
        );
        for field in [
            "placement_evidence_sha256",
            "sequence_capability_evidence_sha256",
            "performance_evidence_sha256",
            "compatibility_report_sha256",
        ] {
            hash(entry, field)?;
        }
        let compile = object(&entry["native_compile"], &["precision", "threads"])?;
        let precision: Precision = serde_json::from_value(compile["precision"].clone())?;
        ensure!(
            precision != Precision::F16,
            "frozen Gemma forbids float16 activation compute"
        );
        let threads = u32::try_from(count(compile, "threads")?)?;
        let execution = execution_expectation(entry, device)?;
        let (host_files, policy) = host(&entry["required_host"])?;
        if let Some(expected) = &policy_sha256 {
            ensure!(
                expected == &policy,
                "native scopes disagree on the global governor"
            );
        } else {
            policy_sha256 = Some(policy);
        }
        let mut scope_closure = closure.clone();
        scope_closure.extend(host_files);
        let public = hash(entry, "attestation_public_key")?;
        let key_file = pinned(
            &runtime.files,
            string(entry, "attestation_private_key_file")?,
        )?;
        ensure!(
            keys.insert(public) && key_files.insert(&key_file.path),
            "native scopes must have unique signing keys"
        );
        let signer = signing_key(key_file, public)?;
        let identity = Identity {
            device,
            model_sha256: artifact.model.sha256.clone(),
            weights_sha256: Some(artifact.weights.sha256.clone()),
            pipeline_sha256: profile::PROFILE_MANIFEST_SHA256.into(),
            runtime_build: runtime.build.clone(),
            dimensions: profile::DIMENSIONS,
            pooling: Pooling::SentenceL2,
            output_name: "embedding".into(),
            execution,
        };
        let compile = Operation::Compile {
            model_path: artifact.model.path.clone(),
            weights_path: Some(artifact.weights.path.clone()),
            runtime_library_path: runtime.library.path.clone(),
            runtime_library_sha256: runtime.library.sha256.clone(),
            plugin_library_path: runtime.plugins[device_name].path.clone(),
            plugin_library_sha256: runtime.plugins[device_name].sha256.clone(),
            bucket: profile::SEQUENCE_BUCKETS[0],
            precision,
            threads,
            closure: scope_closure,
        };
        crate::native_worker::Command {
            schema_version: crate::native_worker::PROTOCOL_VERSION,
            request_id: "verify-package".into(),
            identity: identity.clone(),
            operation: compile.clone(),
        }
        .validate()?;
        let mut attestation = Map::new();
        attestation.insert("package_state".into(), "release".into());
        for field in EXECUTION_FIELDS {
            attestation.insert((*field).into(), entry[*field].clone());
        }
        for field in [
            "runtime",
            "compiler",
            "package_target",
            "artifact_source",
            "internal_precision",
            "device",
        ] {
            string(entry, field)?;
        }
        scopes.push(VerifiedScope {
            id: id.into(),
            execution: Value::Object(attestation),
            identity,
            compile,
            signer,
        });
    }
    ensure!(
        classes == BTreeSet::from(["npu", "gpu", "cpu"]),
        "native package must preserve the full NPU, GPU, CPU cohort"
    );
    let policy_sha256 = policy_sha256.context("package lacks a governor binding")?;
    let policy_raw = read(Path::new(POLICY_FILE), MAX_JSON)?;
    ensure!(
        digest(&policy_raw) == policy_sha256,
        "governor policy changed while loading native package"
    );
    let tokenizer = read(&artifact.tokenizer.path, 128 * MAX_JSON)?;
    ensure!(
        digest(&tokenizer) == profile::TOKENIZER_JSON_SHA256,
        "tokenizer changed after native inventory verification"
    );
    Ok(VerifiedPackage {
        scopes,
        tokenizer,
        policy_sha256,
        request_budget: request_budget(&policy_raw)?,
        dispatcher: runtime.dispatcher,
    })
}

/// The final client alone owns compiled admission. This runs once when a
/// supervised owner is created. The sibling revalidates its private permit
/// and complete closure on every spawn; compilation rehashes under a lease.
pub(crate) fn validate_parent(
    root: &Path,
    plan: &crate::local_inference::LocalPackagePlanV1,
) -> anyhow::Result<Duration> {
    let package = load(root, &plan.package_manifest_sha256, &plan.ordered_scope_ids)?;
    ensure!(
        package.dispatcher.path == root.join(&plan.dispatcher.binary)
            && package.dispatcher.sha256 == plan.dispatcher.sha256,
        "native dispatcher differs from compiled admission"
    );
    for scope in &package.scopes {
        let public: String = scope
            .signer
            .verifying_key()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        crate::embed::validate_native_serving_scope(&scope.execution, &scope.id, &public)?;
    }
    Ok(package.request_budget)
}

/// The separately built sibling uses only the private parent permit. Reading
/// its own compiled plan would create a binary/manifest hash cycle. A direct
/// invocation cannot add scopes to the final client's compiled admission.
pub(crate) fn from_startup(
    permit: &crate::native_http::StartupPermit,
) -> anyhow::Result<VerifiedPackage> {
    let executable = std::env::current_exe()?;
    let root = executable
        .parent()
        .context("native sibling has no payload directory")?;
    let package = load(
        root,
        &permit.package_manifest_sha256,
        &permit.ordered_scope_ids,
    )?;
    ensure!(
        package.dispatcher.path == executable,
        "running native sibling is not the inventoried dispatcher"
    );
    package.dispatcher.verify()?;
    Ok(package)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt as _;

    fn payload() -> (tempfile::TempDir, PathBuf) {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join(PAYLOAD_DIRECTORY);
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join(PACKAGE_MANIFEST), b"{}").unwrap();
        std::fs::write(root.join(RUNTIME_MANIFEST), b"{}").unwrap();
        (temporary, root)
    }
    fn file(root: &Path, name: &str, bytes: &[u8], executable: bool) -> Value {
        let path = root.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, bytes).unwrap();
        std::fs::set_permissions(
            path,
            std::fs::Permissions::from_mode(if executable { 0o755 } else { 0o644 }),
        )
        .unwrap();
        json!({"path":name,"sha256":digest(bytes),"bytes":bytes.len(),"executable":executable})
    }
    #[test]
    fn dynamic_manifest_maps_reject_nested_duplicates_and_trailing_json() {
        for raw in [
            br#"{"a":1,"a":2}"#.as_slice(),
            br#"{"scope":{"device":1,"device":1}}"#,
            br#"{}{}"#,
            br#"{"n":NaN}"#,
        ] {
            assert!(parse(raw).is_err());
        }
        assert_eq!(
            parse(br#"{"a":[true,null,2,"x"]}"#).unwrap(),
            json!({"a":[true,null,2,"x"]})
        );
    }
    #[test]
    fn payload_inventory_is_exact_and_final_client_stays_outside() {
        let (temporary, root) = payload();
        std::fs::write(
            temporary.path().join("cfetch"),
            b"independently built final client",
        )
        .unwrap();
        let files = vec![
            file(&root, "model.bin", b"weights", false),
            file(&root, "native", b"sibling", true),
        ];
        assert_eq!(inventory(&root, &files).unwrap().len(), 2);
        std::fs::write(root.join("extra.so"), b"ambient").unwrap();
        assert!(inventory(&root, &files).is_err());
        std::fs::remove_file(root.join("extra.so")).unwrap();
        std::fs::remove_file(root.join("model.bin")).unwrap();
        assert!(inventory(&root, &files).is_err());
    }
    #[test]
    fn payload_inventory_rejects_content_mode_symlink_and_order_drift() {
        let (_temporary, root) = payload();
        let files = vec![file(&root, "a", b"a", false), file(&root, "b", b"b", true)];
        inventory(&root, &files).unwrap();
        assert!(inventory(&root, &[files[1].clone(), files[0].clone()]).is_err());
        std::fs::write(root.join("a"), b"z").unwrap();
        assert!(inventory(&root, &files).is_err());
        std::fs::write(root.join("a"), b"a").unwrap();
        std::fs::set_permissions(root.join("b"), std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(inventory(&root, &files).is_err());
        std::fs::remove_file(root.join("a")).unwrap();
        std::os::unix::fs::symlink(root.join("b"), root.join("a")).unwrap();
        assert!(inventory(&root, &files).is_err());
    }
    #[test]
    fn paths_and_inventory_never_allow_parent_escape_or_self_hash() {
        let (_temporary, root) = payload();
        for path in [
            "../outside",
            "/etc/passwd",
            "a//b",
            "a/./b",
            "a/../b",
            "a\\b",
        ] {
            assert!(relative(&root, path).is_err());
        }
        let manifest_entry =
            json!({"path":PACKAGE_MANIFEST,"sha256":digest(b"{}"),"bytes":2,"executable":false});
        assert!(inventory(&root, &[manifest_entry]).is_err());
    }
    #[test]
    fn parent_digest_order_and_closed_candidate_are_enforced_before_native_loading() {
        let (_temporary, root) = payload();
        assert!(load(&root, &"0".repeat(64), &["cpu".into()]).is_err());
        let scopes = vec![
            json!({"scope_id":"npu"}),
            json!({"scope_id":"gpu"}),
            json!({"scope_id":"cpu"}),
        ];
        permitted_scopes(&scopes, &["npu".into(), "gpu".into(), "cpu".into()]).unwrap();
        assert!(permitted_scopes(&scopes, &["cpu".into(), "gpu".into(), "npu".into()]).is_err());
        assert!(permitted_scopes(&scopes, &["npu".into()]).is_err());
        let value = json!({"schema_version":2,"package_state":"candidate","profile_id":profile::PROFILE_ID,
            "profile_manifest_sha256":profile::PROFILE_MANIFEST_SHA256,"admission_policy_sha256":profile::ADMISSION_POLICY_SHA256,
            "model":profile::MODEL,"model_revision":profile::MODEL_REVISION,"artifact_manifest":"model/manifest.json",
            "artifact_manifest_sha256":"a".repeat(64),"runtime_manifest_sha256":"b".repeat(64),"scopes":scopes});
        let raw = serde_json::to_vec(&value).unwrap();
        std::fs::write(root.join(PACKAGE_MANIFEST), &raw).unwrap();
        let error = load(
            &root,
            &digest(&raw),
            &["npu".into(), "gpu".into(), "cpu".into()],
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("release package"));
        assert!(
            crate::local_inference::selected_local_package_plan()
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn runtime_binds_independently_built_sibling_plugins_and_provenance() {
        let (_temporary, root) = payload();
        let files = vec![
            file(&root, "cpu.so", b"cpu", false),
            file(&root, "gpu.so", b"gpu", false),
            file(&root, "native", b"independent sibling", true),
            file(&root, "npu.so", b"npu", false),
            file(&root, "runtime.so", b"runtime", false),
        ];
        let mut value = json!({"schema_version":2,"format":"cfetch-native-openvino-v2","target":"x86_64-unknown-linux-gnu",
            "source_revision":"a".repeat(40),"cargo_lock_sha256":"b".repeat(64),"rustc":"rustc fixture",
            "openvino_version":"fixture-version","openvino_build":"fixture-build","dispatcher":"native",
            "openvino_library":"runtime.so","plugin_configuration":"single-device-absolute-library-v1",
            "plugins":{"NPU":"npu.so","GPU":"gpu.so","CPU":"cpu.so"},"files":files});
        let check = |value: &Value| {
            let raw = serde_json::to_vec(value).unwrap();
            std::fs::write(root.join(RUNTIME_MANIFEST), &raw).unwrap();
            runtime(&root, &digest(&raw))
        };
        let verified = check(&value).unwrap();
        assert_eq!(verified.dispatcher.path, root.join("native"));
        assert_eq!(verified.plugins.len(), 3);
        value["plugins"]["NPU"] = "outside.so".into();
        assert!(check(&value).is_err());
        value["plugins"]["NPU"] = "npu.so".into();
        value["schema_version"] = 1.into();
        assert!(check(&value).is_err());
    }
    #[test]
    fn attestation_key_must_match_both_inventory_and_declared_public_identity() {
        let (_temporary, root) = payload();
        let key = ed25519_dalek::SigningKey::from_bytes(&[7; 32]);
        let public: String = key
            .verifying_key()
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let seed = format!("{}\n", "07".repeat(32));
        let entry = file(&root, "key", seed.as_bytes(), false);
        let pin = inventory(&root, &[entry]).unwrap().remove("key").unwrap();
        signing_key(&pin, &public).unwrap();
        assert!(signing_key(&pin, &"a".repeat(64)).is_err());
        std::fs::write(root.join("key"), "08".repeat(32)).unwrap();
        assert!(signing_key(&pin, &public).is_err());
    }
    #[test]
    fn typed_device_properties_preserve_numeric_npu_versions_and_exact_device() {
        let mut value = json!({"required_openvino_properties":{"DEVICE_ARCHITECTURE":"fixture","FULL_DEVICE_NAME":"fixture-npu",
            "NPU_COMPILER_VERSION":123,"NPU_DRIVER_VERSION":456},"required_execution_devices":["NPU.0"]});
        let expectation = execution_expectation(value.as_object().unwrap(), Device::Npu).unwrap();
        assert_eq!(expectation.properties["NPU_DRIVER_VERSION"], "456");
        value["required_openvino_properties"]["NPU_DRIVER_VERSION"] = "456".into();
        assert!(execution_expectation(value.as_object().unwrap(), Device::Npu).is_err());
        value["required_openvino_properties"]["NPU_DRIVER_VERSION"] = 456.into();
        value["required_execution_devices"] = json!(["GPU.0"]);
        assert!(execution_expectation(value.as_object().unwrap(), Device::Npu).is_err());
    }
    #[test]
    fn policy_budget_is_finite_and_includes_all_shapes_and_rows() {
        let limits = json!({"max_operations":100,"max_charged_buckets":100000,"max_duration_ns":1000000000u64,
            "minimum_cooldown_ns":1000000000u64,"cooldown_numerator":1,"cooldown_denominator":1});
        let mut policy = json!({"schema_version":2,"namespace":"cfetch-host-inference-v2","epoch_id":"fixture",
            "state_directory":"/var/lib/cfetch/inference","allowed_devices":["CPU"],"lock_wait_ns":2000000000u64,
            "operations":{"compile":limits,"inference":limits}});
        assert_eq!(
            request_budget(&serde_json::to_vec(&policy).unwrap()).unwrap(),
            Duration::from_secs(7 * 5 + 64 * 3 + 10)
        );
        policy["lock_wait_ns"] = json!(u64::MAX);
        assert!(request_budget(&serde_json::to_vec(&policy).unwrap()).is_err());
        policy["lock_wait_ns"] = 1.into();
        policy["state_directory"] = "/tmp/another-governor".into();
        assert!(request_budget(&serde_json::to_vec(&policy).unwrap()).is_err());
    }
}
