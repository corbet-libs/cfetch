//! Bounded parent transport for the native worker. The governor is held across
//! command writes, synchronous native work, response validation and fsynced
//! checkpoint publication. It never admits a vector profile or resets policy.

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::path::Path;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};

use crate::inference_governor::{Checkpoint, Governor, Kind};
use crate::native_worker::{
    Command, Identity, MAX_COMMAND_BYTES, MAX_RESPONSE_BYTES, Operation, Response,
};

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn response_frame(input: &mut impl std::io::BufRead) -> anyhow::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    input
        .take(MAX_RESPONSE_BYTES as u64 + 1)
        .read_until(b'\n', &mut bytes)?;
    ensure!(
        bytes.len() <= MAX_RESPONSE_BYTES && bytes.last() == Some(&b'\n'),
        "native response is oversized, truncated or missing its frame terminator"
    );
    bytes.pop();
    ensure!(!bytes.is_empty(), "empty native response");
    Ok(bytes)
}
fn validate_response(
    command: &Command,
    request_hash: &str,
    response: &Response,
) -> anyhow::Result<()> {
    ensure!(
        response.schema_version == 1
            && response.request_id.as_deref() == Some(&command.request_id)
            && response.request_sha256 == request_hash
            && response.identity.as_ref() == Some(&command.identity),
        "native response does not match its exact request and profile"
    );
    if let Some(error) = &response.error {
        ensure!(
            !error.is_empty() && error.len() <= 4096 && response.output.is_none(),
            "invalid controlled native error"
        );
        return Ok(());
    }
    ensure!(
        response.runtime_build.as_deref() == Some(&command.identity.runtime_build),
        "native response runtime differs from the request"
    );
    let name = match command.identity.device {
        crate::inference_governor::Device::Cpu => "CPU",
        crate::inference_governor::Device::Gpu => "GPU",
        crate::inference_governor::Device::Npu => "NPU",
    };
    ensure!(
        response.execution_devices.len() == 1,
        "native reply lacks single-device placement"
    );
    let selected = &response.execution_devices[0];
    ensure!(
        selected == name
            || selected
                .strip_prefix(name)
                .and_then(|v| v.strip_prefix('.'))
                .is_some_and(|v| !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit())),
        "native reply executed on another device"
    );
    match &command.operation {
        Operation::Compile { .. } => ensure!(
            response.output.is_none(),
            "compile returned an unexpected vector"
        ),
        Operation::Infer { .. } => {
            let output = response
                .output
                .as_ref()
                .context("native inference lacks a vector")?;
            ensure!(
                output.len() == command.identity.dimensions && output.iter().all(|v| v.is_finite()),
                "native vector width or values are invalid"
            );
            let norm: f64 = output.iter().map(|v| f64::from(*v).powi(2)).sum();
            ensure!(
                (norm - 1.0).abs() <= 1e-4,
                "native vector is not L2 normalized"
            );
        }
    }
    Ok(())
}
fn bounded_file(path: &Path, limit: u64) -> anyhow::Result<Vec<u8>> {
    let fd = rustix::fs::open(
        path,
        rustix::fs::OFlags::RDONLY
            | rustix::fs::OFlags::NOFOLLOW
            | rustix::fs::OFlags::NONBLOCK
            | rustix::fs::OFlags::CLOEXEC,
        rustix::fs::Mode::empty(),
    )?;
    let file = File::from(fd);
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= limit,
        "input must be a bounded regular file"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "input grew beyond its size bound"
    );
    Ok(bytes)
}

fn persist(directory: &Path, request_hash: &str, bytes: &[u8]) -> anyhow::Result<()> {
    let path = directory.join(format!("{request_hash}.json"));
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(bytes)?;
            file.sync_all()?;
            File::open(directory)?.sync_all()?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let metadata = std::fs::symlink_metadata(&path)?;
            ensure!(
                metadata.is_file()
                    && metadata.len() == bytes.len() as u64
                    && bounded_file(&path, MAX_RESPONSE_BYTES as u64)? == bytes,
                "checkpoint already exists with different bytes"
            );
            File::open(&path)?.sync_all()?;
            File::open(directory)?.sync_all()?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn bounded_cpu(raw: &str) -> anyhow::Result<bool> {
    let fields: Vec<_> = raw.split_whitespace().collect();
    ensure!(fields.len() == 2, "invalid cgroup CPU limit");
    let period: u64 = fields[1].parse()?;
    ensure!(period > 0, "invalid cgroup CPU period");
    if fields[0] == "max" {
        return Ok(false);
    }
    let quota: u64 = fields[0].parse()?;
    ensure!(quota > 0, "invalid cgroup CPU quota");
    Ok(u128::from(quota) <= 2 * u128::from(period))
}
fn bounded_memory(raw: &str) -> anyhow::Result<bool> {
    if raw.trim() == "max" {
        return Ok(false);
    }
    let bytes: u64 = raw.trim().parse()?;
    ensure!(bytes > 0, "invalid cgroup memory limit");
    Ok(bytes <= 8 * 1024 * 1024 * 1024)
}
fn require_resource_ceiling() -> anyhow::Result<()> {
    let cgroups = std::fs::read_to_string("/proc/self/cgroup")?;
    let relative = cgroups
        .lines()
        .find_map(|line| line.strip_prefix("0::/"))
        .context("native inference requires cgroup-v2 resource budgets")?;
    ensure!(
        !Path::new(relative)
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir)),
        "invalid process cgroup path"
    );
    let root = Path::new("/sys/fs/cgroup");
    let own = root.join(relative);
    let (mut cpu, mut memory) = (false, false);
    for path in own.ancestors().take_while(|path| path.starts_with(root)) {
        for name in ["cpu.max", "memory.max"] {
            match std::fs::read_to_string(path.join(name)) {
                Ok(raw) => {
                    if name == "cpu.max" {
                        cpu |= bounded_cpu(&raw)?;
                    } else {
                        memory |= bounded_memory(&raw)?;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
                Err(error) => return Err(error.into()),
            }
        }
        if cpu && memory {
            return Ok(());
        }
    }
    anyhow::bail!("native inference requires inherited ceilings of at most two CPUs and 8 GiB")
}

pub struct Adapter {
    child: Child,
    requests: SyncSender<Vec<u8>>,
    replies: Receiver<anyhow::Result<Vec<u8>>>,
    identity: Identity,
    bucket: usize,
    compiled: bool,
    unavailable: bool,
}
impl Adapter {
    pub fn spawn(identity: Identity, bucket: usize) -> anyhow::Result<Self> {
        let mut child = ProcessCommand::new(std::env::current_exe()?)
            .args([
                "native-worker",
                "--parent-pid",
                &std::process::id().to_string(),
            ])
            .env("OMP_NUM_THREADS", "2")
            .env("OPENBLAS_NUM_THREADS", "2")
            .env("MKL_NUM_THREADS", "2")
            .env("RAYON_NUM_THREADS", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let setup = (|| {
            let mut stdin = child.stdin.take().context("native worker stdin missing")?;
            let stdout = child
                .stdout
                .take()
                .context("native worker stdout missing")?;
            let (requests, receive_request) = mpsc::sync_channel::<Vec<u8>>(1);
            let (send_reply, replies) = mpsc::sync_channel(1);
            std::thread::Builder::new()
                .name("native-worker-io".into())
                .spawn(move || {
                    let mut output = BufReader::new(stdout);
                    while let Ok(mut raw) = receive_request.recv() {
                        raw.push(b'\n');
                        let reply = (|| {
                            stdin.write_all(&raw)?;
                            stdin.flush()?;
                            response_frame(&mut output)
                        })();
                        let failed = reply.is_err();
                        if send_reply.send(reply).is_err() || failed {
                            break;
                        }
                    }
                })?;
            Ok::<_, anyhow::Error>((requests, replies))
        })();
        match setup {
            Ok((requests, replies)) => Ok(Self {
                child,
                requests,
                replies,
                identity,
                bucket,
                compiled: false,
                unavailable: false,
            }),
            Err(error) => {
                crate::local_adapter::terminate(&mut child)?;
                Err(error)
            }
        }
    }
    pub fn execute(
        &mut self,
        governor: &Governor,
        command: &Command,
        evidence: &Path,
    ) -> anyhow::Result<Response> {
        ensure!(
            !self.unavailable,
            "native worker is unavailable; no automatic retry"
        );
        command.validate()?;
        ensure!(command.identity == self.identity, "worker profile changed");
        let kind = match &command.operation {
            Operation::Compile { bucket, .. } => {
                ensure!(
                    !self.compiled && *bucket == self.bucket,
                    "worker was already compiled or bucket changed"
                );
                Kind::Compile
            }
            Operation::Infer {
                input_ids,
                attention_mask,
                token_type_ids,
            } => {
                ensure!(
                    self.compiled
                        && input_ids.len() == self.bucket
                        && attention_mask.len() == self.bucket
                        && token_type_ids
                            .as_ref()
                            .is_none_or(|v| v.len() == self.bucket),
                    "inference must match the compiled static bucket"
                );
                Kind::Inference
            }
        };
        let raw = serde_json::to_vec(command)?;
        ensure!(
            raw.len() < MAX_COMMAND_BYTES,
            "native command exceeds framing bound"
        );
        let request_hash = digest(&raw);
        require_resource_ceiling()?;
        let lease = governor.begin(kind, command.identity.device, self.bucket.try_into()?)?;
        let outcome = (|| {
            self.requests
                .try_send(raw)
                .context("native worker request channel unavailable")?;
            let response_bytes =
                lease.supervise(&mut self.child, || match self.replies.try_recv() {
                    Ok(Ok(bytes)) => Ok(Some(bytes)),
                    Ok(Err(error)) => Err(error),
                    Err(TryRecvError::Empty) => Ok(None),
                    Err(TryRecvError::Disconnected) => {
                        anyhow::bail!("native worker I/O channel closed")
                    }
                })?;
            let response: Response = serde_json::from_slice(&response_bytes)?;
            validate_response(command, &request_hash, &response)?;
            // This fsync precedes completion/intent clearing. A crash on either
            // side of that boundary leaves evidence or a blocking pending intent.
            persist(evidence, &request_hash, &response_bytes)?;
            if response.error.is_some() {
                // A controlled failure permits fallback only after the idle
                // worker has exited. Failed cleanup leaves the intent pending.
                crate::local_adapter::terminate(&mut self.child)?;
            }
            lease.complete(Checkpoint {
                request_sha256: request_hash,
                result_sha256: digest(&response_bytes),
            })?;
            Ok::<_, anyhow::Error>(response)
        })();
        match outcome {
            Ok(response) => {
                if response.error.is_some() {
                    self.unavailable = true;
                } else if kind == Kind::Compile {
                    self.compiled = true;
                }
                Ok(response)
            }
            Err(error) => {
                self.unavailable = true;
                crate::local_adapter::terminate(&mut self.child)?;
                Err(error)
            }
        }
    }
}
impl Drop for Adapter {
    fn drop(&mut self) {
        if let Err(error) = crate::local_adapter::terminate(&mut self.child) {
            eprintln!("cfetch native worker cleanup: {error:#}");
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProbePlan {
    pub schema_version: u32,
    pub commands: Vec<Command>,
}

/// An explicit, finite diagnostic. No production configuration or vector index
/// is changed. A completed inference checkpoint is reused only for identical
/// request bytes; a fresh worker still pays a new compile lease after restart.
pub fn probe(plan_path: &Path, evidence: &Path, policy_sha256: String) -> anyhow::Result<()> {
    let plan: ProbePlan = serde_json::from_slice(&bounded_file(plan_path, 32 * 1024 * 1024)?)?;
    ensure!(
        plan.schema_version == 1 && (2..=65).contains(&plan.commands.len()),
        "invalid finite probe plan"
    );
    let Operation::Compile { bucket, .. } = &plan.commands[0].operation else {
        anyhow::bail!("probe must start with a compile command");
    };
    for (index, command) in plan.commands.iter().enumerate() {
        command.validate()?;
        ensure!(
            command.identity == plan.commands[0].identity,
            "probe cannot mix vector profiles or devices"
        );
        ensure!(
            index == 0 || matches!(command.operation, Operation::Infer { .. }),
            "probe has another compile command"
        );
    }
    match std::fs::create_dir(evidence) {
        Ok(()) => (),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => (),
        Err(error) => return Err(error.into()),
    }
    ensure!(
        std::fs::symlink_metadata(evidence)?.is_dir(),
        "probe evidence cannot be a symlink"
    );
    File::open(evidence)?.sync_all()?;
    let parent = evidence
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    File::open(parent)?.sync_all()?;
    let governor = Governor::installed(policy_sha256)?;
    let mut adapter = Adapter::spawn(plan.commands[0].identity.clone(), *bucket)?;
    for command in &plan.commands {
        let request_hash = digest(&serde_json::to_vec(command)?);
        let saved = evidence.join(format!("{request_hash}.json"));
        if matches!(command.operation, Operation::Infer { .. }) && saved.exists() {
            let metadata = std::fs::symlink_metadata(&saved)?;
            ensure!(
                metadata.is_file() && metadata.len() <= MAX_RESPONSE_BYTES as u64,
                "invalid saved checkpoint"
            );
            let response: Response =
                serde_json::from_slice(&bounded_file(&saved, MAX_RESPONSE_BYTES as u64)?)?;
            validate_response(command, &request_hash, &response)?;
            ensure!(
                response.error.is_none(),
                "previous probe operation failed; explicit new plan required"
            );
            continue;
        }
        let response = adapter.execute(&governor, command, evidence)?;
        ensure!(
            response.error.is_none(),
            "native probe failed: {}",
            response.error.as_deref().unwrap_or("")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_governor::Device;
    use crate::native_worker::Pooling;
    fn command() -> Command {
        Command {
            schema_version: 1,
            request_id: "q1".into(),
            identity: Identity {
                device: Device::Gpu,
                model_sha256: "a".repeat(64),
                weights_sha256: None,
                pipeline_sha256: "b".repeat(64),
                runtime_build: "reviewed-runtime".into(),
                output_name: "tokens".into(),
                pooling: Pooling::ClsL2,
                dimensions: 2,
            },
            operation: Operation::Infer {
                input_ids: vec![1; 64],
                attention_mask: vec![1; 64],
                token_type_ids: None,
            },
        }
    }
    fn response(command: &Command) -> Response {
        Response {
            schema_version: 1,
            request_id: Some(command.request_id.clone()),
            request_sha256: digest(&serde_json::to_vec(command).unwrap()),
            identity: Some(command.identity.clone()),
            runtime_build: Some(command.identity.runtime_build.clone()),
            execution_devices: vec!["GPU.0".into()],
            output: Some(vec![1.0, 0.0]),
            error: None,
        }
    }
    #[test]
    fn bounded_frames_reject_unterminated_and_oversized_data() {
        assert_eq!(
            response_frame(&mut std::io::Cursor::new(b"{}\n")).unwrap(),
            b"{}"
        );
        assert!(response_frame(&mut std::io::Cursor::new(b"{}")).is_err());
        assert!(response_frame(&mut std::io::Cursor::new(b"\n")).is_err());
        assert!(
            response_frame(&mut std::io::Cursor::new(vec![
                b'x';
                MAX_RESPONSE_BYTES + 1
            ]))
            .is_err()
        );
    }
    #[test]
    fn response_cannot_substitute_profile_device_or_request() {
        let command = command();
        let good = response(&command);
        validate_response(&command, &good.request_sha256, &good).unwrap();
        for variant in 0..5 {
            let mut bad = good.clone();
            match variant {
                0 => bad.identity.as_mut().unwrap().pipeline_sha256 = "c".repeat(64),
                1 => bad.execution_devices = vec!["CPU".into()],
                2 => bad.request_id = Some("other".into()),
                3 => bad.runtime_build = Some("other".into()),
                _ => bad.output = Some(vec![0.0, 0.0]),
            }
            assert!(validate_response(&command, &good.request_sha256, &bad).is_err());
        }
    }
    #[test]
    fn controlled_error_has_no_vector_but_retains_exact_request_identity() {
        let command = command();
        let mut result = response(&command);
        result.error = Some("device unavailable".into());
        assert!(validate_response(&command, &result.request_sha256, &result).is_err());
        result.output = None;
        result.runtime_build = None;
        result.execution_devices.clear();
        validate_response(&command, &result.request_sha256, &result).unwrap();
        result.identity = None;
        assert!(validate_response(&command, &result.request_sha256, &result).is_err());
    }
    #[test]
    fn checkpoints_never_overwrite_changed_or_symlinked_evidence() {
        let root = tempfile::tempdir().unwrap();
        let hash = "d".repeat(64);
        persist(root.path(), &hash, b"exact result").unwrap();
        persist(root.path(), &hash, b"exact result").unwrap();
        assert!(persist(root.path(), &hash, b"different").is_err());
        let alias = "e".repeat(64);
        std::os::unix::fs::symlink(
            root.path().join(format!("{hash}.json")),
            root.path().join(format!("{alias}.json")),
        )
        .unwrap();
        assert!(persist(root.path(), &alias, b"exact result").is_err());
        assert_eq!(
            std::fs::read(root.path().join(format!("{hash}.json"))).unwrap(),
            b"exact result"
        );
    }
    #[test]
    fn cpu_ceiling_cannot_be_unlimited_or_more_than_two_cores() {
        assert!(bounded_cpu("200000 100000\n").unwrap());
        assert!(bounded_cpu("100000 100000").unwrap());
        assert!(!bounded_cpu("max 100000").unwrap());
        assert!(!bounded_cpu("200001 100000").unwrap());
        assert!(bounded_cpu("0 100000").is_err());
        assert!(bounded_cpu("200000 0").is_err());
        assert!(bounded_cpu("200000 100000 extra").is_err());
    }
    #[test]
    fn memory_ceiling_and_plan_reads_are_bounded() {
        assert!(bounded_memory("8589934592\n").unwrap());
        assert!(!bounded_memory("8589934593").unwrap());
        assert!(!bounded_memory("max").unwrap());
        assert!(bounded_memory("0").is_err());
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("plan.json");
        std::fs::write(&path, b"12345").unwrap();
        assert_eq!(bounded_file(&path, 5).unwrap(), b"12345");
        assert!(bounded_file(&path, 4).is_err());
        let link = root.path().join("alias");
        std::os::unix::fs::symlink(path, &link).unwrap();
        assert!(bounded_file(&link, 5).is_err());
    }
}
