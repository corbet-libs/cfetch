//! A supervised, sequential OpenVINO worker. Its parent owns the durable
//! governor lease before sending each command and checkpoints every reply.
//! Merely starting this process never loads a native runtime or model.

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::io::{BufRead as _, Read as _, Write as _};
use std::path::{Path, PathBuf};

use crate::inference_governor::Device;

pub const PROTOCOL_VERSION: u32 = 3;
pub const MAX_COMMAND_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_RESPONSE_BYTES: usize = 128 * 1024;
const MAX_TOKENS: usize = 32768;
const MAX_DIMENSIONS: usize = 4096;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    MeanMaskL2,
    ClsL2,
    SentenceL2,
    LastTokenL2,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub device: Device,
    pub model_sha256: String,
    pub weights_sha256: Option<String>,
    pub runtime_build: String,
    pub pipeline_sha256: String,
    pub output_name: String,
    pub pooling: Pooling,
    pub dimensions: usize,
    pub execution: ExecutionExpectation,
}

/// Physical properties belong to the admitted scope, never to runtime
/// discovery or a free-form AUTO device request. Numeric C-API properties are
/// canonical decimal strings; the package loader preserves their typed source.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExecutionExpectation {
    #[serde(deserialize_with = "unique_properties")]
    pub properties: BTreeMap<String, String>,
    pub devices: Vec<String>,
}

fn unique_properties<'de, D>(deserializer: D) -> Result<BTreeMap<String, String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, String>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("at most four unique device property strings")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut result = BTreeMap::new();
            while let Some((key, value)) = map.next_entry()? {
                if result.len() == 4 || result.insert(key, value).is_some() {
                    return Err(serde::de::Error::custom(
                        "duplicate or excess device property",
                    ));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Visitor)
}

impl ExecutionExpectation {
    pub fn validate(&self, device: Device) -> anyhow::Result<()> {
        let mut names = vec!["DEVICE_ARCHITECTURE", "FULL_DEVICE_NAME"];
        match device {
            Device::Cpu => (),
            Device::Gpu => names.extend(["GPU_DEVICE_ID", "GPU_UARCH_VERSION"]),
            Device::Npu => names.extend(["NPU_COMPILER_VERSION", "NPU_DRIVER_VERSION"]),
        }
        names.sort_unstable();
        ensure!(
            self.properties.keys().map(String::as_str).eq(names),
            "scope property keys do not match its device class"
        );
        for (name, value) in &self.properties {
            ensure!(identifier(value, 4096), "invalid expected device property");
            if name.starts_with("NPU_") {
                let integer: i128 = value
                    .parse()
                    .context("NPU version property must be an integer")?;
                ensure!(
                    integer >= i128::from(i64::MIN)
                        && integer <= i128::from(u64::MAX)
                        && integer.to_string() == *value,
                    "noncanonical NPU version property"
                );
            }
        }
        ensure!(
            self.devices.len() == 1 && execution_devices(&self.devices[0], device)? == self.devices,
            "scope must bind one exact physical execution device"
        );
        Ok(())
    }

    pub fn verify(
        &self,
        properties: &BTreeMap<String, String>,
        devices: &[String],
    ) -> anyhow::Result<()> {
        ensure!(
            &self.properties == properties,
            "native device properties differ from the admitted scope"
        );
        ensure!(
            self.devices == devices,
            "native execution device differs from the admitted scope"
        );
        Ok(())
    }
}

/// A file in the immutable package or explicit host dependency closure. The
/// native worker verifies all entries while its parent holds the compile lease.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PinnedFile {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    pub executable: bool,
}

impl PinnedFile {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.path.is_absolute()
                && self.path.to_str().is_some_and(|s| identifier(s, 4096))
                && self.path.components().all(|part| matches!(
                    part,
                    std::path::Component::RootDir | std::path::Component::Normal(_)
                )),
            "closure file path is not canonical and absolute"
        );
        ensure!(
            valid_hash(&self.sha256) && (1..=32 * 1024 * 1024 * 1024).contains(&self.bytes),
            "invalid pinned closure file digest or size"
        );
        Ok(())
    }
    pub fn verify(&self) -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt as _;
        self.validate()?;
        ensure!(
            self.path.canonicalize()?.as_os_str() == self.path.as_os_str(),
            "pinned closure path contains a symlink or noncanonical component"
        );
        let metadata = std::fs::symlink_metadata(&self.path)?;
        ensure!(
            metadata.is_file()
                && metadata.len() == self.bytes
                && (metadata.permissions().mode() & 0o111 != 0) == self.executable,
            "pinned closure file type, size or mode changed"
        );
        verify_file(&self.path, &self.sha256, self.bytes)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Compile {
        model_path: PathBuf,
        weights_path: Option<PathBuf>,
        runtime_library_path: PathBuf,
        runtime_library_sha256: String,
        plugin_library_path: PathBuf,
        plugin_library_sha256: String,
        closure: Vec<PinnedFile>,
        bucket: usize,
        precision: Precision,
        threads: u32,
    },
    Infer {
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        token_type_ids: Option<Vec<i64>>,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Command {
    pub schema_version: u32,
    pub request_id: String,
    pub identity: Identity,
    pub operation: Operation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub schema_version: u32,
    pub request_id: Option<String>,
    pub request_sha256: String,
    pub identity: Option<Identity>,
    pub runtime_build: Option<String>,
    pub execution_devices: Vec<String>,
    #[serde(deserialize_with = "unique_properties")]
    pub execution_properties: BTreeMap<String, String>,
    pub output: Option<Vec<f32>>,
    pub error: Option<WorkerFailure>,
}

/// ScopeUnavailable has no production emitter until a vendor API supplies a
/// proven typed absence signal. An arbitrary native error is always HardStop.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerFailureKind {
    ScopeUnavailable,
    HardStop,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerFailure {
    pub kind: WorkerFailureKind,
    pub message: String,
}

fn hard_failure(error: &anyhow::Error) -> WorkerFailure {
    WorkerFailure {
        kind: WorkerFailureKind::HardStop,
        message: bounded_error(&format!("{error:#}")),
    }
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn bounded_error(message: &str) -> String {
    if message.is_empty() {
        return "native operation failed".to_owned();
    }
    let mut end = message.len().min(2048);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn identifier(value: &str, limit: usize) -> bool {
    !value.is_empty() && value.len() <= limit && !value.chars().any(char::is_control)
}
impl Command {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema_version == PROTOCOL_VERSION,
            "unsupported native worker protocol"
        );
        ensure!(
            identifier(&self.request_id, 128),
            "invalid request identity"
        );
        let id = &self.identity;
        id.execution.validate(id.device)?;
        ensure!(
            valid_hash(&id.model_sha256) && valid_hash(&id.pipeline_sha256),
            "invalid artifact or pipeline digest"
        );
        if let Some(hash) = &id.weights_sha256 {
            ensure!(valid_hash(hash), "invalid weights digest");
        }
        ensure!(
            identifier(&id.runtime_build, 256) && identifier(&id.output_name, 256),
            "invalid runtime or output identity"
        );
        ensure!(
            (1..=MAX_DIMENSIONS).contains(&id.dimensions),
            "invalid output dimensions"
        );
        if let Operation::Compile {
            model_path,
            weights_path,
            runtime_library_path,
            runtime_library_sha256,
            plugin_library_path,
            plugin_library_sha256,
            closure,
            bucket,
            threads,
            ..
        } = &self.operation
        {
            ensure!(
                (1..=MAX_TOKENS).contains(bucket),
                "invalid static token bucket"
            );
            ensure!(
                (1..=2).contains(threads),
                "native worker permits at most two compute threads"
            );
            ensure!(
                weights_path.is_some() == id.weights_sha256.is_some(),
                "weights path and digest must be paired"
            );
            ensure!(
                valid_hash(runtime_library_sha256) && valid_hash(plugin_library_sha256),
                "invalid native runtime or plugin digest"
            );
            ensure!(
                (1..=4096).contains(&closure.len()),
                "native closure file count outside bounds"
            );
            let mut paths = std::collections::BTreeSet::new();
            let mut total = 0u64;
            for file in closure {
                file.validate()?;
                ensure!(paths.insert(&file.path), "duplicate pinned closure path");
                total = total
                    .checked_add(file.bytes)
                    .context("closure byte count overflow")?;
                ensure!(
                    total <= 64 * 1024 * 1024 * 1024,
                    "native closure exceeds byte limit"
                );
            }
            for (path, hash, maximum) in [
                (model_path, &id.model_sha256, 16 * 1024 * 1024 * 1024),
                (
                    runtime_library_path,
                    runtime_library_sha256,
                    512 * 1024 * 1024,
                ),
                (
                    plugin_library_path,
                    plugin_library_sha256,
                    1024 * 1024 * 1024,
                ),
            ]
            .into_iter()
            .chain(
                weights_path
                    .iter()
                    .zip(id.weights_sha256.iter())
                    .map(|(path, hash)| (path, hash, 32 * 1024 * 1024 * 1024)),
            ) {
                ensure!(
                    closure.iter().any(|file| &file.path == path
                        && &file.sha256 == hash
                        && file.bytes <= maximum),
                    "native graph, weights, C runtime and device plugin must be bound in the full closure"
                );
            }
            for path in [model_path, runtime_library_path, plugin_library_path]
                .into_iter()
                .chain(weights_path.iter())
            {
                ensure!(
                    path.is_absolute() && path.to_str().is_some_and(|p| identifier(p, 4096)),
                    "artifact path must be an absolute bounded UTF-8 path"
                );
            }
        } else if let Operation::Infer {
            input_ids,
            attention_mask,
            token_type_ids,
        } = &self.operation
        {
            validate_inputs(
                input_ids,
                attention_mask,
                token_type_ids.as_deref(),
                input_ids.len(),
            )?;
        }
        Ok(())
    }
}

/// Hash a bounded stream, rather than loading potentially large weights into
/// memory. The parent has already verified the complete artifact manifest,
/// including external ONNX references; this confirms the named files again.
fn verify_file(path: &Path, expected: &str, maximum: u64) -> anyhow::Result<()> {
    let descriptor = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )
    .context("open pinned native artifact")?;
    let mut file = std::fs::File::from(descriptor);
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() > 0 && metadata.len() <= maximum,
        "native artifact size outside bounds"
    );
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut count = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .context("artifact length overflow")?;
        ensure!(count <= maximum, "native artifact grew beyond bounds");
        hash.update(&buffer[..n]);
    }
    let actual: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    ensure!(
        count == metadata.len() && actual == expected,
        "pinned native artifact digest mismatch"
    );
    Ok(())
}

fn validate_inputs(
    ids: &[i64],
    mask: &[i64],
    types: Option<&[i64]>,
    bucket: usize,
) -> anyhow::Result<()> {
    ensure!(
        bucket > 0 && bucket <= MAX_TOKENS && ids.len() == bucket && mask.len() == bucket,
        "input tensors must match the compiled bucket"
    );
    ensure!(ids.iter().all(|v| *v >= 0), "negative input token id");
    ensure!(
        mask.iter().all(|v| *v == 0 || *v == 1) && mask.contains(&1),
        "invalid or empty attention mask"
    );
    if let Some(types) = types {
        ensure!(
            types.len() == bucket && types.iter().all(|v| *v >= 0),
            "invalid token type tensor"
        );
    }
    Ok(())
}

fn execution_devices(raw: &str, requested: Device) -> anyhow::Result<Vec<String>> {
    let name = match requested {
        Device::Cpu => "CPU",
        Device::Gpu => "GPU",
        Device::Npu => "NPU",
    };
    let devices: Vec<String> = raw
        .split(|c: char| c.is_ascii_whitespace() || ",[]".contains(c))
        .filter(|p| !p.is_empty())
        .map(str::to_owned)
        .collect();
    ensure!(
        devices.len() == 1,
        "runtime did not prove single-device execution"
    );
    let actual = &devices[0];
    let matches = actual == name
        || actual
            .strip_prefix(name)
            .and_then(|s| s.strip_prefix('.'))
            .is_some_and(|suffix| !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()));
    ensure!(
        matches,
        "runtime execution device differs from the requested device"
    );
    Ok(devices)
}

fn pool(
    values: &[f32],
    shape: &[i64],
    mask: &[i64],
    identity: &Identity,
) -> anyhow::Result<Vec<f32>> {
    let width = identity.dimensions;
    ensure!(
        width > 0 && width <= MAX_DIMENSIONS && !mask.is_empty() && mask.len() <= MAX_TOKENS,
        "invalid pooling dimensions"
    );
    ensure!(
        mask.iter().all(|v| *v == 0 || *v == 1) && mask.contains(&1),
        "invalid pooling attention mask"
    );
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "native output contains non-finite values"
    );
    let mut output = vec![0f64; width];
    if identity.pooling == Pooling::SentenceL2 {
        ensure!(
            shape == [1, width as i64] && values.len() == width,
            "pooled output shape mismatch"
        );
        for (out, value) in output.iter_mut().zip(values) {
            *out = f64::from(*value);
        }
    } else {
        ensure!(
            shape == [1, mask.len() as i64, width as i64]
                && values.len()
                    == mask
                        .len()
                        .checked_mul(width)
                        .context("output size overflow")?,
            "token output shape mismatch"
        );
        let selected = match identity.pooling {
            Pooling::ClsL2 => {
                ensure!(mask[0] == 1, "CLS token is masked");
                Some(0)
            }
            Pooling::LastTokenL2 => mask.iter().rposition(|m| *m == 1),
            Pooling::MeanMaskL2 => None,
            Pooling::SentenceL2 => unreachable!(),
        };
        if let Some(index) = selected {
            for (out, value) in output
                .iter_mut()
                .zip(&values[index * width..(index + 1) * width])
            {
                *out = f64::from(*value);
            }
        } else {
            let count = mask.iter().filter(|m| **m == 1).count() as f64;
            for (row, active) in values.chunks_exact(width).zip(mask) {
                if *active == 1 {
                    for (out, value) in output.iter_mut().zip(row) {
                        *out += f64::from(*value) / count;
                    }
                }
            }
        }
    }
    let norm = output.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "native output has zero or invalid norm"
    );
    let result: Vec<f32> = output.into_iter().map(|v| (v / norm) as f32).collect();
    ensure!(
        result.iter().all(|v| v.is_finite()),
        "normalized output is non-finite"
    );
    Ok(result)
}

/// Reject inherited configuration that can replace the audited runtime or
/// redirect plugin discovery. Values are never included in diagnostics.
fn loader_override(name: &str) -> bool {
    name.starts_with("LD_")
        || name.starts_with("DYLD_")
        || name.starts_with("OPENVINO_")
        || name.starts_with("OCL_ICD_")
        || matches!(
            name,
            "NIX_LD"
                | "NIX_LD_LIBRARY_PATH"
                | "GLIBC_TUNABLES"
                | "OPENCL_VENDOR_PATH"
                | "ZE_ENABLE_ALT_DRIVERS"
        )
}

pub(crate) fn reject_loader_overrides() -> anyhow::Result<()> {
    for (name, _) in std::env::vars_os() {
        if let Some(name) = name.to_str() {
            ensure!(
                !loader_override(name),
                "ambient native loader override is forbidden: {name}"
            );
        }
    }
    Ok(())
}

fn plugin_configuration(device: Device, path: &Path) -> anyhow::Result<String> {
    let path = path.to_str().context("plugin path is not UTF-8")?;
    ensure!(
        Path::new(path).is_absolute() && identifier(path, 4096),
        "plugin path is not absolute and bounded"
    );
    let escaped = path
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let device = match device {
        Device::Cpu => "CPU",
        Device::Gpu => "GPU",
        Device::Npu => "NPU",
    };
    Ok(format!(
        "<ie><plugins><plugin name=\"{device}\" location=\"{escaped}\"/></plugins></ie>\n"
    ))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    bytes: u64,
    mode: u32,
    modified: (i64, i64),
    changed: (i64, i64),
}
impl FileIdentity {
    fn read(path: &Path) -> anyhow::Result<Self> {
        use std::os::unix::fs::MetadataExt as _;
        ensure!(
            path.canonicalize()?.as_os_str() == path.as_os_str(),
            "mapped closure path is no longer canonical"
        );
        let metadata = std::fs::symlink_metadata(path)?;
        ensure!(
            metadata.is_file(),
            "mapped closure path is no longer a file"
        );
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            mode: metadata.mode(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }
}

struct ExecutableClosure(BTreeMap<PathBuf, FileIdentity>);
impl ExecutableClosure {
    fn capture(files: &[PinnedFile]) -> anyhow::Result<Self> {
        let mut result = BTreeMap::new();
        for file in files {
            let before = FileIdentity::read(&file.path)?;
            file.verify()?;
            ensure!(
                FileIdentity::read(&file.path)? == before,
                "closure identity changed while hashing"
            );
            ensure!(
                result.insert(file.path.clone(), before).is_none(),
                "duplicate mapped closure path"
            );
        }
        Ok(Self(result))
    }

    fn verify(&self) -> anyhow::Result<()> {
        const MAX_MAPS_BYTES: u64 = 4 * 1024 * 1024;
        let mut raw = Vec::new();
        std::fs::File::open("/proc/self/maps")?
            .take(MAX_MAPS_BYTES + 1)
            .read_to_end(&mut raw)?;
        ensure!(
            raw.len() as u64 <= MAX_MAPS_BYTES,
            "executable mapping inventory exceeds its bound"
        );
        self.verify_text(std::str::from_utf8(&raw).context("mapping inventory is not UTF-8")?)
    }

    fn verify_text(&self, raw: &str) -> anyhow::Result<()> {
        // A previously hashed dependency can be unmapped or mapped without
        // executable permissions. Its identity still binds signed host evidence.
        // Check every entry at each boundary without rehashing model weights.
        for (path, bound) in &self.0 {
            ensure!(
                FileIdentity::read(path)? == *bound,
                "pinned dependency changed after closure verification"
            );
        }
        ensure!(!raw.is_empty(), "empty executable mapping inventory");
        for (number, line) in raw.lines().enumerate() {
            ensure!(number < 32768, "too many process mappings");
            let mut rest = line;
            let mut fields = [""; 5];
            for field in &mut fields {
                rest = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
                let end = rest
                    .find(|c: char| c.is_ascii_whitespace())
                    .unwrap_or(rest.len());
                *field = &rest[..end];
                ensure!(!field.is_empty(), "incomplete executable mapping record");
                rest = &rest[end..];
            }
            let permissions = fields[1].as_bytes();
            ensure!(permissions.len() == 4, "invalid mapping permissions");
            if permissions[2] != b'x' {
                continue;
            }
            let inode: u64 = fields[4].parse().context("invalid mapped inode")?;
            let path = rest.trim_start_matches(|c: char| c.is_ascii_whitespace());
            if inode == 0 {
                // Kernel code pages and anonymous JIT allocations are not file
                // dependencies. Named/deleted file-backed mappings are not exempt.
                ensure!(
                    path.is_empty()
                        || matches!(path, "[vdso]" | "[vsyscall]")
                        || (path.starts_with("[anon:") && path.ends_with(']')),
                    "unknown executable mapping without an inode"
                );
                continue;
            }
            ensure!(
                !path.ends_with(" (deleted)")
                    && !path.contains('\\')
                    && Path::new(path).is_absolute(),
                "deleted, escaped or unnamed executable file mapping"
            );
            let (major, minor) = fields[3].split_once(':').context("invalid mapped device")?;
            let major = u32::from_str_radix(major, 16).context("invalid mapped device major")?;
            let minor = u32::from_str_radix(minor, 16).context("invalid mapped device minor")?;
            let bound = self
                .0
                .get(Path::new(path))
                .context("executable mapping is outside the audited closure")?;
            ensure!(
                bound.inode == inode
                    && rustix::fs::major(bound.device) == major
                    && rustix::fs::minor(bound.device) == minor,
                "executable mapping inode/device differs from the audited closure"
            );
            ensure!(
                FileIdentity::read(Path::new(path))? == *bound,
                "mapped dependency changed after closure verification"
            );
        }
        Ok(())
    }
}

#[cfg(feature = "native-openvino")]
mod native {
    use super::*;
    use openvino::{
        Core, DeviceType, ElementType, PartialShape, PropertyKey, RwPropertyKey, Shape, Tensor,
    };

    pub(super) struct Session {
        compiled: openvino::CompiledModel,
        _model: openvino::Model,
        _core: Core,
        _plugin_config: tempfile::NamedTempFile,
        executable_closure: ExecutableClosure,
        pub identity: Identity,
        pub devices: Vec<String>,
        pub runtime_build: String,
        pub properties: BTreeMap<String, String>,
        bucket: usize,
        input_names: Vec<String>,
    }
    impl Session {
        pub(super) fn compile(command: &Command) -> anyhow::Result<Self> {
            let Operation::Compile {
                model_path,
                weights_path,
                runtime_library_path,
                runtime_library_sha256: _,
                plugin_library_path,
                plugin_library_sha256: _,
                closure,
                bucket,
                precision,
                threads,
            } = &command.operation
            else {
                anyhow::bail!("expected compile command")
            };
            reject_loader_overrides()?;
            let executable_closure = ExecutableClosure::capture(closure)?;
            executable_closure.verify()?;
            let mut plugin_config = tempfile::NamedTempFile::new()?;
            plugin_config.write_all(
                plugin_configuration(command.identity.device, plugin_library_path)?.as_bytes(),
            )?;
            plugin_config.as_file().sync_all()?;
            // This is the first native call. The parent must hold its compile
            // lease before sending this command, including runtime/model loading.
            // Loading the exact file first prevents discovery of another system
            // runtime; Core::new_with_config reuses the already-loaded binding.
            openvino_sys::library::load_from(runtime_library_path.clone())
                .map_err(anyhow::Error::msg)
                .context("load pinned OpenVINO C library")?;
            executable_closure.verify()?;
            let mut core = Core::new_with_config(
                plugin_config
                    .path()
                    .to_str()
                    .context("private plugin config path is not UTF-8")?,
            )
            .context("initialize the exact pinned device plugin")?;
            executable_closure.verify()?;
            let runtime_build = openvino::version().build_number;
            ensure!(
                runtime_build == command.identity.runtime_build,
                "OpenVINO runtime build differs from pinned identity"
            );
            let device = match command.identity.device {
                Device::Cpu => DeviceType::CPU,
                Device::Gpu => DeviceType::GPU,
                Device::Npu => DeviceType::NPU,
            };
            let mut properties = BTreeMap::new();
            for name in command.identity.execution.properties.keys() {
                let value = core.get_property(&device, &PropertyKey::Other(name.clone().into()))?;
                properties.insert(name.clone(), value);
            }
            ensure!(
                properties == command.identity.execution.properties,
                "native device properties differ from the admitted scope"
            );
            core.set_property(&device, &RwPropertyKey::HintPerformanceMode, "LATENCY")?;
            core.set_property(&device, &RwPropertyKey::HintNumRequests, "1")?;
            core.set_property(
                &device,
                &RwPropertyKey::HintInferencePrecision,
                match precision {
                    Precision::F32 => "f32",
                    Precision::F16 => "f16",
                    Precision::Bf16 => "bf16",
                },
            )?;
            if command.identity.device == Device::Cpu {
                core.set_property(
                    &device,
                    &RwPropertyKey::InferenceNumThreads,
                    &threads.to_string(),
                )?;
                core.set_property(&device, &RwPropertyKey::NumStreams, "1")?;
            }
            let mut model = core.read_model_from_file(
                model_path.to_str().context("model path is not UTF-8")?,
                weights_path.as_ref().and_then(|p| p.to_str()).unwrap_or(""),
            )?;
            let input_count = model.get_inputs_len()?;
            ensure!(
                (2..=3).contains(&input_count),
                "unsupported native model input count"
            );
            let mut input_names = Vec::new();
            for index in 0..input_count {
                let input = model.get_input_by_index(index)?;
                let name = input.get_name()?;
                ensure!(
                    ["input_ids", "attention_mask", "token_type_ids"].contains(&name.as_str())
                        && !input_names.contains(&name),
                    "unsupported or duplicate native input name"
                );
                ensure!(
                    input.get_element_type()? == ElementType::I64,
                    "native inputs must be int64 tensors"
                );
                input_names.push(name);
            }
            ensure!(
                input_names.iter().any(|n| n == "input_ids")
                    && input_names.iter().any(|n| n == "attention_mask"),
                "model lacks required token inputs"
            );
            let static_shape = PartialShape::new_static(2, &[1, *bucket as i64])?;
            for name in &input_names {
                model.reshape_input_by_name(name, &static_shape)?;
            }
            executable_closure.verify()?;
            let compiled = core
                .compile_model(&model, device)
                .context("compile explicitly selected native device")?;
            let raw = compiled
                .get_property(&PropertyKey::Other("EXECUTION_DEVICES".into()))?
                .into_owned();
            let devices = execution_devices(&raw, command.identity.device)?;
            command.identity.execution.verify(&properties, &devices)?;
            compiled
                .get_output_by_name(&command.identity.output_name)
                .context("resolve pinned native output")?;
            executable_closure.verify()?;
            Ok(Self {
                compiled,
                _model: model,
                _core: core,
                _plugin_config: plugin_config,
                executable_closure,
                identity: command.identity.clone(),
                devices,
                runtime_build,
                properties,
                bucket: *bucket,
                input_names,
            })
        }
        pub(super) fn infer(&mut self, command: &Command) -> anyhow::Result<Vec<f32>> {
            ensure!(
                command.identity == self.identity,
                "native worker identity changed after compilation"
            );
            let Operation::Infer {
                input_ids,
                attention_mask,
                token_type_ids,
            } = &command.operation
            else {
                anyhow::bail!("expected inference command")
            };
            validate_inputs(
                input_ids,
                attention_mask,
                token_type_ids.as_deref(),
                self.bucket,
            )?;
            ensure!(
                token_type_ids.is_some() == self.input_names.iter().any(|n| n == "token_type_ids"),
                "token type input presence differs from compiled graph"
            );
            self.executable_closure.verify()?;
            let mut request = self.compiled.create_infer_request()?;
            let shape = Shape::new(&[1, self.bucket as i64])?;
            let mut tensors = Vec::new();
            for name in &self.input_names {
                let values = match name.as_str() {
                    "input_ids" => input_ids.as_slice(),
                    "attention_mask" => attention_mask.as_slice(),
                    "token_type_ids" => token_type_ids
                        .as_deref()
                        .context("missing token type tensor")?,
                    _ => anyhow::bail!("unknown compiled input"),
                };
                let mut tensor = Tensor::new(ElementType::I64, &shape)?;
                tensor.get_data_mut::<i64>()?.copy_from_slice(values);
                request.set_tensor(name, &tensor)?;
                tensors.push(tensor);
            }
            request
                .infer()
                .context("native inference on explicit device")?;
            let actual = self
                .compiled
                .get_property(&PropertyKey::Other("EXECUTION_DEVICES".into()))?;
            ensure!(
                execution_devices(actual.as_ref(), self.identity.device)? == self.devices,
                "execution placement changed during inference"
            );
            let output = request.get_tensor(&self.identity.output_name)?;
            ensure!(
                output.get_element_type()? == ElementType::F32,
                "native output must be float32"
            );
            let shape = output.get_shape()?;
            ensure!(
                output.get_size()?
                    <= self
                        .bucket
                        .checked_mul(self.identity.dimensions)
                        .context("output bound overflow")?,
                "native output exceeds declared dimensions"
            );
            let result = pool(
                output.get_data::<f32>()?,
                shape.get_dimensions(),
                attention_mask,
                &self.identity,
            )?;
            drop(output);
            drop(shape);
            drop(request);
            drop(tensors);
            self.executable_closure.verify()?;
            Ok(result)
        }
    }
}

fn read_command(reader: &mut impl std::io::BufRead) -> anyhow::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    reader
        .take((MAX_COMMAND_BYTES + 1) as u64)
        .read_until(b'\n', &mut line)?;
    if line.is_empty() {
        return Ok(None);
    }
    ensure!(
        line.len() <= MAX_COMMAND_BYTES && line.last() == Some(&b'\n'),
        "native command exceeds frame limit or lacks newline"
    );
    line.pop();
    Ok(Some(line))
}

#[cfg(target_os = "linux")]
fn assert_parent(expected: u32) -> anyhow::Result<()> {
    ensure!(
        expected > 1
            && rustix::process::getppid().map(|p| p.as_raw_nonzero().get() as u32)
                == Some(expected),
        "native worker parent identity changed"
    );
    Ok(())
}

/// Keep protocol bytes on a private descriptor before a vendor runtime can
/// write to stdout. Native diagnostics go to stderr, never into JSON framing.
#[cfg(all(target_os = "linux", feature = "native-openvino"))]
fn protocol_writer() -> anyhow::Result<std::fs::File> {
    let protocol = rustix::io::fcntl_dupfd_cloexec(std::io::stdout(), 3)?;
    rustix::stdio::dup2_stdout(std::io::stderr())?;
    Ok(std::fs::File::from(protocol))
}

/// Internal entry point. EOF exits; parent death kills this exact worker even
/// while a native call blocks. No child or grandchild is spawned here.
#[cfg(all(target_os = "linux", feature = "native-openvino"))]
pub fn run_stdio(expected_parent_pid: u32) -> anyhow::Result<()> {
    rustix::process::set_parent_process_death_signal(Some(rustix::process::Signal::KILL))?;
    assert_parent(expected_parent_pid)?;
    reject_loader_overrides()?;
    let mut reader = std::io::BufReader::new(std::io::stdin().lock());
    let mut writer = protocol_writer()?;
    let mut session: Option<native::Session> = None;
    let mut compile_attempted = false;
    let mut failed = false;
    while let Some(line) = read_command(&mut reader)? {
        assert_parent(expected_parent_pid)?;
        let command: Command =
            serde_json::from_slice(&line).context("decode native worker command")?;
        command.validate()?;
        let mut response = Response {
            schema_version: PROTOCOL_VERSION,
            request_id: Some(command.request_id.clone()),
            request_sha256: sha256(&line),
            identity: Some(command.identity.clone()),
            runtime_build: None,
            execution_devices: Vec::new(),
            execution_properties: BTreeMap::new(),
            output: None,
            error: None,
        };
        let result = (|| -> anyhow::Result<()> {
            ensure!(
                !failed,
                "native worker latched unavailable after a failed operation"
            );
            match &command.operation {
                Operation::Compile { .. } => {
                    ensure!(
                        !compile_attempted,
                        "native worker permits exactly one compile attempt"
                    );
                    compile_attempted = true;
                    session = Some(native::Session::compile(&command)?);
                }
                Operation::Infer { .. } => {
                    response.output = Some(
                        session
                            .as_mut()
                            .context("model has not been compiled")?
                            .infer(&command)?,
                    );
                }
            }
            Ok(())
        })();
        if let Some(active) = &session {
            response.runtime_build = Some(active.runtime_build.clone());
            response.execution_devices = active.devices.clone();
            response.execution_properties = active.properties.clone();
        }
        if let Err(error) = result {
            failed = true;
            response.error = Some(hard_failure(&error));
        }
        assert_parent(expected_parent_pid)?;
        let bytes = serde_json::to_vec(&response)?;
        ensure!(
            bytes.len() < MAX_RESPONSE_BYTES,
            "native response exceeds frame bound"
        );
        writer.write_all(&bytes)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "native-openvino")))]
pub fn run_stdio(_expected_parent_pid: u32) -> anyhow::Result<()> {
    anyhow::bail!("native OpenVINO worker is unavailable in this target package")
}

#[cfg(test)]
pub(crate) fn fixture_execution(device: Device) -> ExecutionExpectation {
    let mut properties: BTreeMap<String, String> = [
        ("DEVICE_ARCHITECTURE".into(), "fixture-architecture".into()),
        ("FULL_DEVICE_NAME".into(), "fixture-device".into()),
    ]
    .into_iter()
    .collect();
    let physical = match device {
        Device::Cpu => "CPU",
        Device::Gpu => {
            properties.insert("GPU_DEVICE_ID".into(), "0x0000".into());
            properties.insert("GPU_UARCH_VERSION".into(), "fixture-uarch".into());
            "GPU.0"
        }
        Device::Npu => {
            properties.insert("NPU_COMPILER_VERSION".into(), "1".into());
            properties.insert("NPU_DRIVER_VERSION".into(), "1".into());
            "NPU"
        }
    };
    ExecutionExpectation {
        properties,
        devices: vec![physical.into()],
    }
}

#[cfg(test)]
pub(crate) fn fixture_closure(entries: &[(&str, &str)]) -> Vec<PinnedFile> {
    entries
        .iter()
        .map(|(path, hash)| PinnedFile {
            path: (*path).into(),
            sha256: (*hash).into(),
            bytes: 1,
            executable: false,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loader_override_names_and_exact_one_device_xml_are_explicit() {
        for name in [
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "LD_AUDIT",
            "GLIBC_TUNABLES",
            "OPENVINO_INSTALL_DIR",
            "OCL_ICD_VENDORS",
            "ZE_ENABLE_ALT_DRIVERS",
            "NIX_LD_LIBRARY_PATH",
        ] {
            assert!(loader_override(name), "{name}");
        }
        for name in ["PATH", "HOME", "OMP_NUM_THREADS", "RAYON_NUM_THREADS"] {
            assert!(!loader_override(name), "{name}");
        }
        let xml = plugin_configuration(Device::Cpu, Path::new("/package/a&b/plugin\".so")).unwrap();
        assert_eq!(
            xml,
            "<ie><plugins><plugin name=\"CPU\" location=\"/package/a&amp;b/plugin&quot;.so\"/></plugins></ie>\n"
        );
        assert!(!xml.contains("GPU"));
        assert!(!xml.contains("NPU"));
        assert!(plugin_configuration(Device::Cpu, Path::new("relative.so")).is_err());
    }

    fn mapped_row(path: &Path, identity: &FileIdentity) -> String {
        format!(
            "1000-2000 r-xp 00000000 {:02x}:{:02x} {}           {}\n",
            rustix::fs::major(identity.device),
            rustix::fs::minor(identity.device),
            identity.inode,
            path.display()
        )
    }

    #[test]
    fn executable_file_mappings_require_bound_bytes_inode_and_unchanged_identity() {
        use std::os::unix::fs::PermissionsExt as _;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().canonicalize().unwrap().join("runtime.so");
        std::fs::write(&path, b"runtime").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let file = PinnedFile {
            path: path.clone(),
            sha256: sha256(b"runtime"),
            bytes: 7,
            executable: false,
        };
        let closure = ExecutableClosure::capture(std::slice::from_ref(&file)).unwrap();
        let identity = FileIdentity::read(&path).unwrap();
        let row = mapped_row(&path, &identity);
        assert!(closure.verify_text(&row).is_ok());
        let unbound = directory.path().canonicalize().unwrap().join("unbound.so");
        assert!(
            closure
                .verify_text(&mapped_row(&unbound, &identity))
                .is_err()
        );
        let mut wrong = identity.clone();
        wrong.inode += 1;
        assert!(closure.verify_text(&mapped_row(&path, &wrong)).is_err());
        wrong = identity.clone();
        wrong.device += 1;
        assert!(closure.verify_text(&mapped_row(&path, &wrong)).is_err());
        assert!(
            closure
                .verify_text(&format!("{} (deleted)\n", row.trim_end()))
                .is_err()
        );
        // Replacing a dependency with identical bytes still changes the inode.
        let replacement = directory.path().join("replacement");
        std::fs::write(&replacement, b"runtime").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        assert!(closure.verify_text(&row).is_err());
        assert!(
            ExecutableClosure::capture(&[PinnedFile {
                sha256: "0".repeat(64),
                ..file
            }])
            .is_err()
        );
    }

    #[test]
    fn unmapped_and_nonexecutable_pinned_dependencies_must_remain_unchanged() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory
            .path()
            .canonicalize()
            .unwrap()
            .join("host-dependency.bin");
        std::fs::write(&path, b"original").unwrap();
        let file = PinnedFile {
            path: path.clone(),
            sha256: sha256(b"original"),
            bytes: 8,
            executable: false,
        };
        let closure = ExecutableClosure::capture(&[file]).unwrap();
        let unmapped = "1000-2000 r-xp 00000000 00:00 0 [vdso]\n";
        let nonexecutable =
            mapped_row(&path, &FileIdentity::read(&path).unwrap()).replace("r-xp", "r--p");
        assert!(closure.verify_text(unmapped).is_ok());
        assert!(closure.verify_text(&nonexecutable).is_ok());
        // The same path and size must not conceal changed bytes, even when
        // neither executable-mapping branch examines the dependency.
        std::fs::write(&path, b"modified").unwrap();
        assert!(closure.verify_text(unmapped).is_err());
        assert!(closure.verify_text(&nonexecutable).is_err());
    }

    #[test]
    fn executable_mapping_parser_distinguishes_kernel_and_anonymous_memory_from_files() {
        let closure = ExecutableClosure(BTreeMap::new());
        assert!(closure.verify_text("1000-2000 r-xp 00000000 00:00 0\n2000-3000 r-xp 00000000 00:00 0 [vdso]\n3000-4000 rwxp 00000000 00:00 0 [anon:jit]\n").is_ok());
        for row in [
            "",
            "bad\n",
            "1000-2000 r-xp 00000000 00:00 0 /memfd:runtime (deleted)\n",
            "1000-2000 r-xp 00000000 00:00 1 /unknown.so\n",
            "1000-2000 r-xp 00000000 00:00 0 [unknown]\n",
        ] {
            assert!(closure.verify_text(row).is_err(), "{row}");
        }
    }

    #[test]
    fn exact_device_expectations_reject_missing_or_duplicate_properties_and_other_devices() {
        for device in [Device::Cpu, Device::Gpu, Device::Npu] {
            let expected = fixture_execution(device);
            expected.validate(device).unwrap();
            let mut changed = expected.clone();
            changed.properties.remove("DEVICE_ARCHITECTURE");
            assert!(changed.validate(device).is_err());
            changed = expected.clone();
            changed.devices.push(expected.devices[0].clone());
            assert!(changed.validate(device).is_err());
            let mut actual = expected.properties.clone();
            actual.insert("FULL_DEVICE_NAME".into(), "another physical device".into());
            assert!(expected.verify(&actual, &expected.devices).is_err());
            assert!(
                expected
                    .verify(&expected.properties, &["GPU.9".into()])
                    .is_err()
            );
        }
        assert!(serde_json::from_str::<ExecutionExpectation>(
            r#"{"properties":{"FULL_DEVICE_NAME":"one","FULL_DEVICE_NAME":"two"},"devices":["CPU"]}"#
        ).is_err());
        let response = serde_json::json!({"schema_version": PROTOCOL_VERSION, "request_id": null,
            "request_sha256": "a".repeat(64), "identity": null, "runtime_build": null,
            "execution_devices": [], "execution_properties": {}, "output": null, "error": null});
        let duplicate = response.to_string().replace("\"execution_properties\":{}",
            "\"execution_properties\":{\"DEVICE_ARCHITECTURE\":\"first\",\"DEVICE_ARCHITECTURE\":\"second\"}");
        assert!(serde_json::from_str::<Response>(&duplicate).is_err());
    }

    #[test]
    fn closure_files_are_content_size_mode_and_path_bound_before_native_loading() {
        use std::os::unix::fs::PermissionsExt as _;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().canonicalize().unwrap().join("runtime.so");
        std::fs::write(&path, b"runtime").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let pinned = PinnedFile {
            path: path.clone(),
            sha256: sha256(b"runtime"),
            bytes: 7,
            executable: false,
        };
        pinned.verify().unwrap();
        std::fs::write(&path, b"changed").unwrap();
        assert!(pinned.verify().is_err());
        std::fs::write(&path, b"runtime").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(pinned.verify().is_err());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        let alias = directory.path().join("alias.so");
        std::os::unix::fs::symlink(&path, &alias).unwrap();
        assert!(
            PinnedFile {
                path: alias,
                ..pinned.clone()
            }
            .verify()
            .is_err()
        );
        std::fs::write(&path, b"runtime longer").unwrap();
        assert!(pinned.verify().is_err());
    }
    #[cfg(all(target_os = "linux", feature = "native-openvino"))]
    #[test]
    fn vendor_stdout_cannot_corrupt_protocol() {
        const FLAG: &str = "CFETCH_TEST_PROTOCOL_WRITER_CHILD";
        if std::env::var_os(FLAG).is_some() {
            let mut protocol = super::protocol_writer().unwrap();
            rustix::io::write(std::io::stdout(), b"vendor diagnostic\n").unwrap();
            protocol.write_all(b"private protocol frame\n").unwrap();
            protocol.flush().unwrap();
            std::process::exit(0);
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native_worker::tests::vendor_stdout_cannot_corrupt_protocol",
                "--nocapture",
            ])
            .env(FLAG, "1")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("protocol isolation child exceeded deadline");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains("private protocol frame\n"));
        assert!(!stdout.contains("vendor diagnostic"));
        assert_eq!(output.stderr, b"vendor diagnostic\n");
    }

    #[test]
    fn native_error_text_never_grants_scope_fallback() {
        for text in [
            "device unavailable",
            "NOT_FOUND",
            "model hash mismatch",
            "runtime changed",
        ] {
            let failure = hard_failure(&anyhow::anyhow!(text));
            assert_eq!(failure.kind, WorkerFailureKind::HardStop);
        }
        assert!(
            serde_json::from_str::<WorkerFailure>(r#"{"message":"device unavailable"}"#).is_err()
        );
        assert!(
            serde_json::from_str::<WorkerFailure>(r#"{"kind":"unknown","message":"unavailable"}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<WorkerFailure>(r#""device unavailable""#).is_err());
    }

    fn identity(pooling: Pooling) -> Identity {
        Identity {
            device: Device::Cpu,
            model_sha256: "a".repeat(64),
            weights_sha256: None,
            runtime_build: "test-runtime".into(),
            pipeline_sha256: "b".repeat(64),
            output_name: "embedding".into(),
            pooling,
            dimensions: 2,
            execution: fixture_execution(Device::Cpu),
        }
    }
    #[test]
    fn pooling_respects_mask_and_selected_semantics() {
        let values = [3., 0., 0., 4., 900., 900.];
        let mask = [1, 1, 0];
        assert_eq!(
            pool(&values, &[1, 3, 2], &mask, &identity(Pooling::ClsL2)).unwrap(),
            [1., 0.]
        );
        assert_eq!(
            pool(&values, &[1, 3, 2], &mask, &identity(Pooling::LastTokenL2)).unwrap(),
            [0., 1.]
        );
        let mean = pool(&values, &[1, 3, 2], &mask, &identity(Pooling::MeanMaskL2)).unwrap();
        assert!((mean[0] - 0.6).abs() < 1e-6 && (mean[1] - 0.8).abs() < 1e-6);
        assert_eq!(
            pool(&[3., 4.], &[1, 2], &mask, &identity(Pooling::SentenceL2)).unwrap(),
            [0.6, 0.8]
        );
    }
    #[test]
    fn invalid_outputs_fail_before_publication() {
        for values in [[0., 0.], [f32::NAN, 1.], [f32::INFINITY, 1.]] {
            assert!(pool(&values, &[1, 2], &[1], &identity(Pooling::SentenceL2)).is_err());
        }
        assert!(pool(&[1., 2.], &[2], &[1], &identity(Pooling::SentenceL2)).is_err());
        assert!(pool(&[1., 2.], &[1, 2], &[0], &identity(Pooling::SentenceL2)).is_err());
    }
    #[test]
    fn exact_device_evidence_rejects_fallback() {
        assert_eq!(execution_devices("GPU.0", Device::Gpu).unwrap(), ["GPU.0"]);
        for raw in ["AUTO", "GPU CPU", "GPU.0,CPU", "GPU.foo", "NPU", ""] {
            assert!(execution_devices(raw, Device::Gpu).is_err());
        }
    }
    #[test]
    fn tensors_cannot_truncate_or_change_compiled_bucket() {
        assert!(validate_inputs(&[1, 2], &[1, 0], None, 2).is_ok());
        assert!(validate_inputs(&[1], &[1], None, 2).is_err());
        assert!(validate_inputs(&[1, 2], &[0, 0], None, 2).is_err());
        assert!(validate_inputs(&[1, 2], &[1, 2], None, 2).is_err());
        assert!(validate_inputs(&[1, 2], &[1, 0], Some(&[0]), 2).is_err());
        let mut command = Command {
            schema_version: PROTOCOL_VERSION,
            request_id: "request-2".into(),
            identity: identity(Pooling::MeanMaskL2),
            operation: Operation::Infer {
                input_ids: vec![1, 2],
                attention_mask: vec![1, 0],
                token_type_ids: None,
            },
        };
        assert!(command.validate().is_ok());
        if let Operation::Infer { input_ids, .. } = &mut command.operation {
            input_ids[0] = -1;
        }
        assert!(command.validate().is_err());
    }
    #[test]
    fn framing_is_bounded_and_requires_complete_lines() {
        let mut source = std::io::Cursor::new(b"{}\n{}\n");
        assert_eq!(read_command(&mut source).unwrap().unwrap(), b"{}");
        assert_eq!(read_command(&mut source).unwrap().unwrap(), b"{}");
        assert!(read_command(&mut source).unwrap().is_none());
        assert!(read_command(&mut std::io::Cursor::new(b"{}")).is_err());
        assert!(
            read_command(&mut std::io::Cursor::new(vec![b'x'; MAX_COMMAND_BYTES + 1])).is_err()
        );
    }
    #[test]
    fn artifact_digest_is_enforced_without_native_runtime() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"pinned-artifact").unwrap();
        assert!(verify_file(file.path(), &sha256(b"pinned-artifact"), 128).is_ok());
        assert!(verify_file(file.path(), &"0".repeat(64), 128).is_err());
        assert!(verify_file(file.path(), &sha256(b"pinned-artifact"), 2).is_err());
    }
    #[test]
    fn controlled_error_limit_counts_utf8_bytes() {
        let message = bounded_error(&"🧠".repeat(2048));
        assert_eq!(message.len(), 2048);
        assert!(!bounded_error("").is_empty());
    }

    #[test]
    fn precision_tokens_are_explicit() {
        assert_eq!(serde_json::to_string(&Precision::Bf16).unwrap(), "\"bf16\"");
        assert!(serde_json::from_str::<Precision>("\"AUTO\"").is_err());
    }

    #[test]
    fn compile_identity_pairs_weights_and_bounds_resources() {
        let mut command = Command {
            schema_version: PROTOCOL_VERSION,
            request_id: "request-1".into(),
            identity: identity(Pooling::MeanMaskL2),
            operation: Operation::Compile {
                model_path: PathBuf::from("/models/pinned.onnx"),
                weights_path: None,
                runtime_library_path: PathBuf::from("/runtime/libopenvino_c.so"),
                runtime_library_sha256: "d".repeat(64),
                plugin_library_path: PathBuf::from("/runtime/libopenvino_cpu_plugin.so"),
                plugin_library_sha256: "e".repeat(64),
                closure: fixture_closure(&[
                    ("/models/pinned.onnx", &"a".repeat(64)),
                    ("/runtime/libopenvino_c.so", &"d".repeat(64)),
                    ("/runtime/libopenvino_cpu_plugin.so", &"e".repeat(64)),
                ]),
                bucket: 128,
                precision: Precision::F32,
                threads: 2,
            },
        };
        assert!(command.validate().is_ok());
        let mut missing = serde_json::to_value(&command).unwrap();
        missing["operation"]
            .as_object_mut()
            .unwrap()
            .remove("plugin_library_path");
        assert!(serde_json::from_value::<Command>(missing).is_err());
        let original = command.operation.clone();
        if let Operation::Compile { closure, .. } = &mut command.operation {
            closure.pop();
        }
        assert!(command.validate().is_err());
        command.operation = original;
        command.identity.weights_sha256 = Some("c".repeat(64));
        assert!(command.validate().is_err());
        command.identity.weights_sha256 = None;
        if let Operation::Compile {
            runtime_library_sha256,
            ..
        } = &mut command.operation
        {
            *runtime_library_sha256 = "invalid".into();
        }
        assert!(command.validate().is_err());
        if let Operation::Compile {
            runtime_library_sha256,
            ..
        } = &mut command.operation
        {
            *runtime_library_sha256 = "d".repeat(64);
        }
        if let Operation::Compile { threads, .. } = &mut command.operation {
            *threads = 3;
        }
        assert!(command.validate().is_err());
    }
}
