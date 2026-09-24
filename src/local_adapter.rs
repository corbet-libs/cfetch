//! Supervision boundary for a package-local inference adapter.
//!
//! cfetch owns process identity and lifetime. The target adapter owns only its
//! native runtime. A random bearer travels over stdin, the child binds an
//! ephemeral loopback port, and EOF tells it that its parent is gone.

use std::io::{BufRead as _, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use sha2::Digest as _;

pub(crate) const MAX_STARTUP_BYTES: usize = 8192;
pub(crate) const MAX_SCOPES: usize = 16;

const READY_TIMEOUT: Duration = Duration::from_secs(20);
const MAX_READY_BYTES: usize = 16 * 1024;
const TERMINATE_TIMEOUT: Duration = Duration::from_secs(2);
const EXIT_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterEndpoint {
    pub base_url: String,
    pub authorization: String,
}

#[derive(Debug, Clone)]
pub struct AdapterLaunch {
    pub binary: PathBuf,
    pub sha256: String,
    pub package_manifest: PathBuf,
    pub package_manifest_sha256: String,
    pub ordered_scope_ids: Vec<String>,
}

struct RunningAdapter {
    child: Child,
    /// Kept open for the whole child lifetime. The adapter treats EOF as a
    /// parent-death signal and shuts down instead of becoming an orphan.
    _stdin: Option<ChildStdin>,
    endpoint: AdapterEndpoint,
}

pub struct AdapterSupervisor {
    launch: AdapterLaunch,
    running: Option<RunningAdapter>,
    /// A failed startup whose child could not be confirmed dead. Retaining
    /// it prevents another launch and lets Drop retry bounded cleanup.
    cleanup_pending: Option<Child>,
    failed: bool,
}

#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadyLine {
    schema_version: u32,
    url: String,
    scope_ids: Vec<String>,
}

impl AdapterSupervisor {
    pub fn new(launch: AdapterLaunch) -> anyhow::Result<Self> {
        validate_launch(&launch)?;
        Ok(Self {
            launch,
            running: None,
            cleanup_pending: None,
            failed: false,
        })
    }

    pub fn ordered_scope_ids(&self) -> &[String] {
        &self.launch.ordered_scope_ids
    }

    /// Starts lazily. An unexplained child exit latches the entire native
    /// owner; it never authorizes another scope or an automatic restart.
    pub fn endpoint(&mut self) -> anyhow::Result<AdapterEndpoint> {
        anyhow::ensure!(
            !self.failed,
            "package-local adapter is unavailable; native supervisor is latched"
        );
        if let Some(child) = &self.cleanup_pending {
            anyhow::bail!(
                "package-local adapter startup cleanup is incomplete for PID {}; refusing another launch",
                child.id()
            );
        }
        if self.child_exited()? {
            self.failed = true;
            anyhow::bail!(
                "package-local adapter exited unexpectedly; native supervisor is latched"
            );
        }
        if self.running.is_none() {
            anyhow::ensure!(
                !self.failed,
                "package-local adapter is unavailable; native supervisor is latched"
            );
            self.running = Some(spawn_adapter(&self.launch, &mut self.cleanup_pending)?);
        }
        Ok(self
            .running
            .as_ref()
            .expect("adapter was started")
            .endpoint
            .clone())
    }

    /// All uncertain native/transport/attestation outcomes latch the owner.
    /// Confirming death is cleanup, never permission to attempt another scope.
    pub fn abort_after_failure(&mut self) -> anyhow::Result<()> {
        self.failed = true;
        self.stop()
            .context("stop package-local adapter after hard failure")
    }

    fn child_exited(&mut self) -> anyhow::Result<bool> {
        let Some(running) = self.running.as_mut() else {
            return Ok(false);
        };
        match running
            .child
            .try_wait()
            .context("inspect package-local adapter")?
        {
            Some(_) => {
                self.running.take();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn stop(&mut self) -> anyhow::Result<()> {
        if let Some(running) = self.running.as_mut() {
            // Closing stdin first gives a well-behaved adapter a clean exit.
            running._stdin.take();
            // Keep the owned child on failure: an uncertain exit must not
            // make the next endpoint call start a second adapter.
            terminate(&mut running.child)?;
        }
        self.running.take();
        if let Some(child) = self.cleanup_pending.as_mut() {
            terminate(child)?;
        }
        self.cleanup_pending.take();
        Ok(())
    }
}

impl Drop for AdapterSupervisor {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            eprintln!("cfetch: package-local adapter cleanup failed: {error:#}");
        }
    }
}

fn validate_launch(launch: &AdapterLaunch) -> anyhow::Result<()> {
    validate_sha256(&launch.sha256, "package-local adapter digest")?;
    validate_sha256(
        &launch.package_manifest_sha256,
        "package-local root manifest digest",
    )?;
    anyhow::ensure!(
        !launch.ordered_scope_ids.is_empty() && launch.ordered_scope_ids.len() <= MAX_SCOPES,
        "package-local adapter needs at least one admitted scope"
    );
    anyhow::ensure!(
        launch
            .package_manifest
            .file_name()
            .and_then(|name| name.to_str())
            == Some("package-manifest.json")
            && launch.binary.parent() == launch.package_manifest.parent(),
        "package-local root manifest must be package-manifest.json beside the adapter"
    );
    let mut unique = std::collections::BTreeSet::new();
    for id in &launch.ordered_scope_ids {
        crate::local_inference::validate_scope_id(id)?;
        anyhow::ensure!(unique.insert(id), "duplicate package-local scope");
    }
    validate_regular_file(&launch.binary, "package-local adapter")?;
    validate_regular_file(&launch.package_manifest, "package-local root manifest")?;
    let actual = file_sha256(&launch.binary)?;
    anyhow::ensure!(
        actual == launch.sha256,
        "package-local adapter digest mismatch: package plan requires {}, found {actual}",
        launch.sha256
    );
    let actual_manifest = file_sha256(&launch.package_manifest)?;
    anyhow::ensure!(
        actual_manifest == launch.package_manifest_sha256,
        "package-local root manifest digest mismatch: package plan requires {}, found {actual_manifest}",
        launch.package_manifest_sha256
    );
    Ok(())
}

fn validate_sha256(value: &str, label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{label} must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

fn validate_regular_file(path: &Path, label: &str) -> anyhow::Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .with_context(|| format!("inspect {label} {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        "{label} must be a regular non-symlink file"
    );
    Ok(())
}

fn file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut input = std::fs::File::open(path)
        .with_context(|| format!("open package-local adapter {}", path.display()))?;
    let mut digest = sha2::Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = input
            .read(&mut buffer)
            .with_context(|| format!("hash package-local adapter {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(crate::hashing::hex_lower(digest.finalize()))
}

fn spawn_adapter(
    launch: &AdapterLaunch,
    cleanup_pending: &mut Option<Child>,
) -> anyhow::Result<RunningAdapter> {
    // Re-check immediately before execution. Construction and first use may
    // be separated by a long-running daemon's lifetime.
    validate_launch(launch)?;
    // Unit transport fixtures execute shell scripts under Cargo's injected
    // loader environment; the CLI integration tests exercise this production
    // guard in the real binary using an explicitly controlled child environment.
    #[cfg(all(target_os = "linux", feature = "native-openvino", not(test)))]
    crate::native_worker::reject_loader_overrides()?;
    let bearer = hex(&rand::random::<[u8; 32]>());
    let mut child = std::process::Command::new(&launch.binary)
        .args([
            "native-serve",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--auth-stdin",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("start package-local adapter {}", launch.binary.display()))?;
    let startup: anyhow::Result<(ChildStdin, ReadyLine)> = (|| {
        let mut stdin = child.stdin.take().context("adapter stdin was not piped")?;
        let secret_line = serde_json::to_vec(&serde_json::json!({
            "schema_version": 3, "purpose": "production", "bearer": bearer,
            "package_manifest_sha256": launch.package_manifest_sha256,
            "ordered_scope_ids": launch.ordered_scope_ids,
        }))?;
        anyhow::ensure!(
            secret_line.len() < MAX_STARTUP_BYTES,
            "native startup permit exceeds its bound"
        );
        stdin
            .write_all(&secret_line)
            .context("write package-local adapter authentication")?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;

        let stdout = child
            .stdout
            .take()
            .context("adapter stdout was not piped")?;
        let (sender, receiver) = mpsc::sync_channel(1);
        std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(stdout);
            let mut bytes = Vec::new();
            let result = reader
                .by_ref()
                .take((MAX_READY_BYTES + 1) as u64)
                .read_until(b'\n', &mut bytes)
                .map(|_| bytes);
            let _ = sender.send(result);
        });
        let ready_bytes = match receiver.recv_timeout(READY_TIMEOUT) {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(error)) => return Err(error).context("read package-local adapter readiness"),
            Err(_) => anyhow::bail!("timed out waiting for package-local adapter readiness"),
        };
        anyhow::ensure!(
            ready_bytes.len() <= MAX_READY_BYTES && ready_bytes.ends_with(b"\n"),
            "package-local adapter readiness line is missing or exceeds its bound"
        );
        let ready: ReadyLine = serde_json::from_slice(&ready_bytes)
            .context("parse package-local adapter readiness")?;
        validate_ready_line(&ready, &launch.ordered_scope_ids)?;
        Ok((stdin, ready))
    })();
    match startup {
        Ok((stdin, ready)) => Ok(RunningAdapter {
            child,
            _stdin: Some(stdin),
            endpoint: AdapterEndpoint {
                base_url: ready.url,
                authorization: format!("Bearer {bearer}"),
            },
        }),
        Err(error) => {
            if let Err(cleanup_error) = terminate(&mut child) {
                *cleanup_pending = Some(child);
                return Err(error).context(format!(
                    "package-local adapter startup cleanup failed: {cleanup_error:#}"
                ));
            }
            Err(error)
        }
    }
}

fn validate_ready_line(ready: &ReadyLine, expected_scopes: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(
        ready.schema_version == 1,
        "unsupported adapter readiness schema"
    );
    anyhow::ensure!(
        ready.scope_ids == expected_scopes,
        "package-local adapter readiness scopes do not exactly match its package plan"
    );
    let rest = ready
        .url
        .strip_prefix("http://127.0.0.1:")
        .context("package-local adapter must advertise IPv4 loopback HTTP")?;
    let (port, path) = rest
        .split_once('/')
        .context("package-local adapter readiness URL has no path")?;
    let port: u16 = port
        .parse()
        .context("package-local adapter readiness port is invalid")?;
    anyhow::ensure!(
        port != 0 && path == "v1",
        "package-local adapter readiness URL must end in /v1"
    );
    Ok(())
}

pub(crate) fn terminate(child: &mut Child) -> anyhow::Result<()> {
    let pid = child.id();
    if child
        .try_wait()
        .with_context(|| format!("inspect package-local adapter PID {pid} before termination"))?
        .is_some()
    {
        return Ok(());
    }
    if let Err(error) = child.kill() {
        // The process may have exited between inspection and the kill.
        if child
            .try_wait()
            .with_context(|| {
                format!("inspect package-local adapter PID {pid} after failed termination")
            })?
            .is_some()
        {
            return Ok(());
        }
        return Err(error).with_context(|| {
            format!("terminate package-local adapter PID {pid}; process may still be alive")
        });
    }
    wait_for_owned_exit(pid, TERMINATE_TIMEOUT, || {
        child.try_wait().map(|status| status.is_some())
    })
}

fn wait_for_owned_exit(
    pid: u32,
    timeout: Duration,
    mut exited: impl FnMut() -> std::io::Result<bool>,
) -> anyhow::Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        if exited().with_context(|| {
            format!("inspect package-local adapter PID {pid} during termination")
        })? {
            return Ok(());
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        anyhow::ensure!(
            !remaining.is_zero(),
            "timed out confirming package-local adapter PID {pid} exit after {} ms; process may still be alive",
            timeout.as_millis()
        );
        std::thread::sleep(EXIT_POLL_INTERVAL.min(remaining));
    }
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn termination_poll_times_out_and_reports_the_owned_pid() {
        let started = Instant::now();
        let error = wait_for_owned_exit(123, Duration::ZERO, || Ok(false)).unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(1));
        let message = error.to_string();
        assert!(
            message.contains("PID 123") && message.contains("timed out"),
            "{message}"
        );
        assert!(message.contains("process may still be alive"), "{message}");
    }

    #[test]
    fn termination_poll_reports_inspection_errors_without_retrying() {
        let mut inspections = 0;
        let error = wait_for_owned_exit(123, TERMINATE_TIMEOUT, || {
            inspections += 1;
            Err(std::io::Error::other("fake inspection failure"))
        })
        .unwrap_err();
        assert_eq!(inspections, 1);
        let message = format!("{error:#}");
        assert!(
            message.contains("PID 123") && message.contains("fake inspection failure"),
            "{message}"
        );
    }

    #[cfg(unix)]
    fn idle_child() -> Child {
        // exec preserves the exact child PID and leaves no shell descendant.
        std::process::Command::new("sh")
            .args(["-c", "exec sleep 60"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn termination_reaps_only_the_owned_child_and_is_repeatable() {
        let mut owned = idle_child();
        let mut other = idle_child();
        let started = Instant::now();
        let result = (|| -> anyhow::Result<()> {
            terminate(&mut owned)?;
            anyhow::ensure!(owned.try_wait()?.is_some(), "owned child was not reaped");
            anyhow::ensure!(other.try_wait()?.is_none(), "another child was terminated");
            terminate(&mut owned)?;
            Ok(())
        })();
        // Clean both exact children before asserting, including failure paths.
        let owned_cleanup = terminate(&mut owned);
        let other_cleanup = terminate(&mut other);
        result.unwrap();
        owned_cleanup.unwrap();
        other_cleanup.unwrap();
        assert!(started.elapsed() < TERMINATE_TIMEOUT + Duration::from_secs(1));
    }

    #[cfg(unix)]
    fn fake_launch(script: &str) -> (tempfile::TempDir, AdapterLaunch) {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let binary = directory.path().join("fake-adapter");
        let package_manifest = directory.path().join("package-manifest.json");
        std::fs::write(&binary, script).unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(&package_manifest, "{}\n").unwrap();
        let launch = AdapterLaunch {
            sha256: file_sha256(&binary).unwrap(),
            binary,
            package_manifest_sha256: file_sha256(&package_manifest).unwrap(),
            package_manifest,
            ordered_scope_ids: vec!["cpu-scope".into()],
        };
        (directory, launch)
    }

    #[cfg(unix)]
    #[test]
    fn malformed_readiness_cleans_up_the_fake_adapter_within_the_bound() {
        let (_directory, launch) = fake_launch(
            "#!/bin/sh\nIFS= read -r secret\nprintf '%s\\n' 'invalid readiness'\nexec sleep 60\n",
        );
        let mut supervisor = AdapterSupervisor::new(launch).unwrap();
        let started = Instant::now();
        let error = supervisor.endpoint().unwrap_err();
        assert!(format!("{error:#}").contains("parse package-local adapter readiness"));
        assert!(supervisor.running.is_none());
        assert!(supervisor.cleanup_pending.is_none());
        assert!(!supervisor.failed);
        assert!(started.elapsed() < TERMINATE_TIMEOUT + Duration::from_secs(1));
    }

    #[cfg(unix)]
    #[test]
    fn incomplete_startup_cleanup_blocks_another_launch_without_consuming_a_restart() {
        let (_directory, launch) = fake_launch("#!/bin/sh\nexit 0\n");
        let mut supervisor = AdapterSupervisor::new(launch).unwrap();
        supervisor.cleanup_pending = Some(idle_child());
        let pid = supervisor.cleanup_pending.as_ref().unwrap().id();
        let error = supervisor.endpoint().unwrap_err().to_string();
        assert!(
            error.contains(&format!("PID {pid}")) && error.contains("refusing another launch"),
            "{error}"
        );
        assert!(supervisor.running.is_none());
        assert!(!supervisor.failed);
        assert!(
            supervisor
                .cleanup_pending
                .as_mut()
                .unwrap()
                .try_wait()
                .unwrap()
                .is_none()
        );
        supervisor.stop().unwrap();
        assert!(supervisor.cleanup_pending.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn unexpected_child_death_latches_without_another_scope_or_restart() {
        let (directory, launch) = fake_launch(
            "#!/bin/sh\nIFS= read -r secret\nprintf '%s\\n' \"$$\" >> \"$0.starts\"\nprintf '%s\\n' '{\"schema_version\":1,\"url\":\"http://127.0.0.1:43123/v1\",\"scope_ids\":[\"cpu-scope\"]}'\nwhile IFS= read -r command; do :; done\n",
        );
        let mut supervisor = AdapterSupervisor::new(launch).unwrap();
        supervisor.endpoint().unwrap();
        terminate(&mut supervisor.running.as_mut().unwrap().child).unwrap();
        for _ in 0..3 {
            assert!(
                supervisor
                    .endpoint()
                    .unwrap_err()
                    .to_string()
                    .contains("latched")
            );
            assert!(supervisor.running.is_none());
        }
        let starts = std::fs::read_to_string(directory.path().join("fake-adapter.starts")).unwrap();
        assert_eq!(
            starts.lines().count(),
            1,
            "an uncertain child exit cannot permit another native operation"
        );
    }

    #[cfg(unix)]
    #[test]
    fn hard_latch_blocks_reuse_even_while_cleanup_retains_a_live_owner() {
        let (_directory, launch) = fake_launch("#!/bin/sh\nexit 0\n");
        let mut supervisor = AdapterSupervisor::new(launch).unwrap();
        supervisor.running = Some(RunningAdapter {
            child: idle_child(),
            _stdin: None,
            endpoint: AdapterEndpoint {
                base_url: "http://127.0.0.1:43123/v1".into(),
                authorization: "fixture".into(),
            },
        });
        supervisor.failed = true;
        let pid = supervisor.running.as_ref().unwrap().child.id();
        for _ in 0..2 {
            assert!(
                supervisor
                    .endpoint()
                    .unwrap_err()
                    .to_string()
                    .contains("latched")
            );
            let child = &mut supervisor.running.as_mut().unwrap().child;
            assert_eq!(child.id(), pid);
            assert!(child.try_wait().unwrap().is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn private_startup_permit_binds_schema_manifest_and_ordered_scopes() {
        let (directory, launch) = fake_launch(
            "#!/bin/sh\n[ \"$1\" = native-serve ] || exit 2\nIFS= read -r secret\nprintf '%s\\n' \"$secret\" > \"$0.permit\"\nprintf '%s\\n' '{\"schema_version\":1,\"url\":\"http://127.0.0.1:43123/v1\",\"scope_ids\":[\"cpu-scope\"]}'\ncat >/dev/null\n",
        );
        let expected = launch.package_manifest_sha256.clone();
        let mut supervisor = AdapterSupervisor::new(launch).unwrap();
        supervisor.endpoint().unwrap();
        let raw = std::fs::read(directory.path().join("fake-adapter.permit")).unwrap();
        let permit: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(permit.as_object().unwrap().len(), 5);
        assert_eq!(permit["schema_version"], 3);
        assert_eq!(permit["purpose"], "production");
        assert_eq!(permit["package_manifest_sha256"], expected);
        assert_eq!(
            permit["ordered_scope_ids"],
            serde_json::json!(["cpu-scope"])
        );
        assert_eq!(permit["bearer"].as_str().unwrap().len(), 64);
        assert!(raw.len() <= MAX_STARTUP_BYTES);
    }

    #[test]
    fn readiness_is_exact_and_loopback_only() {
        let scopes = vec!["npu-scope".to_string(), "gpu-scope".to_string()];
        validate_ready_line(
            &ReadyLine {
                schema_version: 1,
                url: "http://127.0.0.1:43123/v1".into(),
                scope_ids: scopes.clone(),
            },
            &scopes,
        )
        .unwrap();
        for url in [
            "https://127.0.0.1:43123/v1",
            "http://localhost:43123/v1",
            "http://127.0.0.1:0/v1",
            "http://127.0.0.1:43123/other",
        ] {
            let ready = ReadyLine {
                schema_version: 1,
                url: url.into(),
                scope_ids: scopes.clone(),
            };
            assert!(
                validate_ready_line(&ready, &scopes).is_err(),
                "accepted {url}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn supervisor_hashes_starts_authenticates_and_stops_a_sibling() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let adapter = directory.path().join("fake-adapter");
        let package_manifest = directory.path().join("package-manifest.json");
        std::fs::write(
            &adapter,
            "#!/bin/sh\nIFS= read -r secret\nprintf '%s\\n' '{\"schema_version\":1,\"url\":\"http://127.0.0.1:43123/v1\",\"scope_ids\":[\"npu-scope\",\"gpu-scope\",\"cpu-scope\"]}'\ncat >/dev/null\n",
        )
        .unwrap();
        std::fs::write(&package_manifest, "{\"schema_version\":1}\n").unwrap();
        std::fs::set_permissions(&adapter, std::fs::Permissions::from_mode(0o700)).unwrap();
        let launch = AdapterLaunch {
            sha256: file_sha256(&adapter).unwrap(),
            binary: adapter,
            package_manifest_sha256: file_sha256(&package_manifest).unwrap(),
            package_manifest,
            ordered_scope_ids: vec!["npu-scope".into(), "gpu-scope".into(), "cpu-scope".into()],
        };
        let mut supervisor = AdapterSupervisor::new(launch).unwrap();
        let endpoint = supervisor.endpoint().unwrap();
        assert_eq!(endpoint.base_url, "http://127.0.0.1:43123/v1");
        assert!(endpoint.authorization.starts_with("Bearer "));
        assert_eq!(endpoint.authorization.len(), "Bearer ".len() + 64);
        let started = Instant::now();
        supervisor.abort_after_failure().unwrap();
        assert!(supervisor.failed);
        assert!(supervisor.running.is_none());
        assert!(supervisor.cleanup_pending.is_none());
        assert!(started.elapsed() < TERMINATE_TIMEOUT + Duration::from_secs(1));
        for _ in 0..3 {
            assert!(
                supervisor
                    .endpoint()
                    .unwrap_err()
                    .to_string()
                    .contains("latched")
            );
            assert!(supervisor.running.is_none());
        }
        drop(supervisor);
    }

    #[cfg(unix)]
    #[test]
    fn supervisor_refuses_root_manifest_drift_before_launch() {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempfile::tempdir().unwrap();
        let adapter = directory.path().join("fake-adapter");
        let package_manifest = directory.path().join("package-manifest.json");
        std::fs::write(&adapter, "#!/bin/sh\nexit 0\n").unwrap();
        std::fs::set_permissions(&adapter, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(&package_manifest, "{\"schema_version\":1}\n").unwrap();
        let launch = AdapterLaunch {
            binary: adapter.clone(),
            sha256: file_sha256(&adapter).unwrap(),
            package_manifest: package_manifest.clone(),
            package_manifest_sha256: "0".repeat(64),
            ordered_scope_ids: vec!["npu-scope".into()],
        };
        let error = AdapterSupervisor::new(launch).err().unwrap().to_string();
        assert!(error.contains("root manifest digest mismatch"), "{error}");
    }
}
