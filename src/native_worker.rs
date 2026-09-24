//! A supervised, sequential OpenVINO worker. Its parent owns the durable
//! governor lease before sending each command and checkpoints every reply.
//! Merely starting this process never loads a native runtime or model.

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::io::{BufRead as _, Read as _, Write as _};
use std::path::{Path, PathBuf};

use crate::inference_governor::Device;

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
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Compile {
        model_path: PathBuf,
        weights_path: Option<PathBuf>,
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
    pub output: Option<Vec<f32>>,
    pub error: Option<String>,
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
            self.schema_version == 1,
            "unsupported native worker protocol"
        );
        ensure!(
            identifier(&self.request_id, 128),
            "invalid request identity"
        );
        let id = &self.identity;
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
            for path in std::iter::once(model_path).chain(weights_path.iter()) {
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
    let mut file = std::fs::File::open(path).context("open pinned native artifact")?;
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
        pub identity: Identity,
        pub devices: Vec<String>,
        pub runtime_build: String,
        bucket: usize,
        input_names: Vec<String>,
    }
    impl Session {
        pub(super) fn compile(command: &Command) -> anyhow::Result<Self> {
            let Operation::Compile {
                model_path,
                weights_path,
                bucket,
                precision,
                threads,
            } = &command.operation
            else {
                anyhow::bail!("expected compile command")
            };
            verify_file(
                model_path,
                &command.identity.model_sha256,
                16 * 1024 * 1024 * 1024,
            )?;
            if let (Some(path), Some(digest)) = (weights_path, &command.identity.weights_sha256) {
                verify_file(path, digest, 32 * 1024 * 1024 * 1024)?;
            }
            // This is the first native call. The parent must hold its compile
            // lease before sending this command, including model loading.
            let mut core = Core::new().context("load OpenVINO runtime")?;
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
            let compiled = core
                .compile_model(&model, device)
                .context("compile explicitly selected native device")?;
            let raw = compiled
                .get_property(&PropertyKey::Other("EXECUTION_DEVICES".into()))?
                .into_owned();
            let devices = execution_devices(&raw, command.identity.device)?;
            compiled
                .get_output_by_name(&command.identity.output_name)
                .context("resolve pinned native output")?;
            Ok(Self {
                compiled,
                _model: model,
                _core: core,
                identity: command.identity.clone(),
                devices,
                runtime_build,
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
            pool(
                output.get_data::<f32>()?,
                shape.get_dimensions(),
                attention_mask,
                &self.identity,
            )
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

/// Internal entry point. EOF exits; parent death kills this exact worker even
/// while a native call blocks. No child or grandchild is spawned here.
#[cfg(all(target_os = "linux", feature = "native-openvino"))]
pub fn run_stdio(expected_parent_pid: u32) -> anyhow::Result<()> {
    rustix::process::set_parent_process_death_signal(Some(rustix::process::Signal::KILL))?;
    assert_parent(expected_parent_pid)?;
    let mut reader = std::io::BufReader::new(std::io::stdin().lock());
    let mut writer = std::io::stdout().lock();
    let mut session: Option<native::Session> = None;
    let mut compile_attempted = false;
    let mut failed = false;
    while let Some(line) = read_command(&mut reader)? {
        assert_parent(expected_parent_pid)?;
        let command: Command =
            serde_json::from_slice(&line).context("decode native worker command")?;
        command.validate()?;
        let mut response = Response {
            schema_version: 1,
            request_id: Some(command.request_id.clone()),
            request_sha256: sha256(&line),
            identity: Some(command.identity.clone()),
            runtime_build: None,
            execution_devices: Vec::new(),
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
        }
        if let Err(error) = result {
            failed = true;
            response.error = Some(bounded_error(&format!("{error:#}")));
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
mod tests {
    use super::*;
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
        for values in ([0., 0.], [f32::NAN, 1.], [f32::INFINITY, 1.]) {
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
            schema_version: 1,
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
            schema_version: 1,
            request_id: "request-1".into(),
            identity: identity(Pooling::MeanMaskL2),
            operation: Operation::Compile {
                model_path: PathBuf::from("/models/pinned.onnx"),
                weights_path: None,
                bucket: 128,
                precision: Precision::F32,
                threads: 2,
            },
        };
        assert!(command.validate().is_ok());
        command.identity.weights_sha256 = Some("c".repeat(64));
        assert!(command.validate().is_err());
        command.identity.weights_sha256 = None;
        if let Operation::Compile { threads, .. } = &mut command.operation {
            *threads = 3;
        }
        assert!(command.validate().is_err());
    }
}
