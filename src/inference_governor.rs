//! Host-wide admission for one native compile or inference operation.
//!
//! Provisioning is privileged and external. This module never creates the
//! namespace, resets a budget, clears a failed intent, or accepts a new boot.
//! A lease owns the permanent flock inode until its durable completion. Native
//! calls run in an owned child; an unconfirmed exit leaves the intent in place.
//! This is a cooperative load governor, not a sandbox for hostile group members.

use anyhow::{Context as _, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Component, Path};
use std::process::Child;
use std::time::Duration;

const MAX_BYTES: u64 = 16 * 1024;
const POLL: Duration = Duration::from_millis(10);
pub const DIRECTORY: &str = "/var/lib/cfetch/inference";

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum Device {
    Cpu,
    Gpu,
    Npu,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Compile,
    Inference,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_operations: u64,
    pub max_charged_buckets: u64,
    pub max_duration_ns: u64,
    pub minimum_cooldown_ns: u64,
    pub cooldown_numerator: u64,
    pub cooldown_denominator: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Operations<T> {
    pub compile: T,
    pub inference: T,
}
impl<T> Operations<T> {
    fn get(&self, kind: Kind) -> &T {
        match kind {
            Kind::Compile => &self.compile,
            Kind::Inference => &self.inference,
        }
    }
    fn get_mut(&mut self, kind: Kind) -> &mut T {
        match kind {
            Kind::Compile => &mut self.compile,
            Kind::Inference => &mut self.inference,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Policy {
    pub schema_version: u32,
    pub namespace: String,
    pub epoch_id: String,
    pub state_directory: String,
    pub allowed_devices: Vec<Device>,
    pub lock_wait_ns: u64,
    pub operations: Operations<Limits>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Usage {
    pub operations: u64,
    pub charged_buckets: u64,
}

/// An already durable result, identified by its exact input and result bytes.
/// The caller must fsync that result before committing this checkpoint.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub request_sha256: String,
    pub result_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Completed {
    pub kind: Kind,
    pub device: Device,
    pub bucket: u64,
    pub total_operations: u64,
    pub started_ns: u64,
    pub finished_ns: u64,
    pub cooldown_until_ns: u64,
    pub checkpoint: Checkpoint,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub schema_version: u32,
    pub policy_sha256: String,
    pub epoch_id: String,
    pub boot_id: String,
    pub last_observed_ns: u64,
    pub not_before_ns: u64,
    pub last_completed: Option<Completed>,
    pub usage: Operations<Usage>,
}

fn digest(raw: &[u8]) -> String {
    Sha256::digest(raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn boot_valid(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit() && !b.is_ascii_uppercase()
            }
        })
}
fn bucket_valid(value: u64) -> bool {
    [32, 64, 128, 257, 512, 1024, 2048, 4096, 8192, 16384, 32768].contains(&value)
}
fn add(a: u64, b: u64) -> anyhow::Result<u64> {
    a.checked_add(b).context("governor counter overflow")
}
impl Limits {
    fn cooldown(&self, elapsed: u64) -> anyhow::Result<u64> {
        ensure!(self.cooldown_denominator > 0, "zero cooldown denominator");
        let ratio = u128::from(elapsed) * u128::from(self.cooldown_numerator);
        let proportional = ratio.div_ceil(u128::from(self.cooldown_denominator));
        add(self.minimum_cooldown_ns, u64::try_from(proportional)?)
    }
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.max_operations > 0
                && self.max_charged_buckets > 0
                && self.max_duration_ns > 0
                && self.minimum_cooldown_ns > 0,
            "governor requires positive finite limits"
        );
        let cooldown = self.cooldown(self.max_duration_ns)?;
        ensure!(
            cooldown >= self.max_duration_ns,
            "governor exceeds 50% duty cycle"
        );
        add(self.max_duration_ns, cooldown)?;
        Ok(())
    }
}
impl Policy {
    fn validate(&self, path: &Path) -> anyhow::Result<()> {
        ensure!(
            self.schema_version == 2 && self.namespace == "cfetch-host-inference-v2",
            "unsupported native governor policy"
        );
        ensure!(
            Path::new(&self.state_directory) == path,
            "governor namespace mismatch"
        );
        ensure!(
            !self.epoch_id.is_empty()
                && self.epoch_id.len() <= 128
                && self
                    .epoch_id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
            "invalid governor epoch"
        );
        ensure!(
            self.lock_wait_ns > 0 && !self.allowed_devices.is_empty(),
            "empty governor admission"
        );
        for (index, device) in self.allowed_devices.iter().enumerate() {
            ensure!(
                !self.allowed_devices[..index].contains(device),
                "duplicate device admission"
            );
        }
        self.operations.compile.validate()?;
        self.operations.inference.validate()
    }
}

impl State {
    /// Provisioning data only: callers must install it explicitly as root.
    pub fn initial(
        policy_sha256: String,
        epoch_id: String,
        boot_id: String,
        now: u64,
    ) -> anyhow::Result<Self> {
        ensure!(
            hash(&policy_sha256) && boot_valid(&boot_id),
            "invalid governor initial identity"
        );
        Ok(Self {
            schema_version: 2,
            policy_sha256,
            epoch_id,
            boot_id,
            last_observed_ns: now,
            not_before_ns: now,
            last_completed: None,
            usage: Operations {
                compile: Usage::default(),
                inference: Usage::default(),
            },
        })
    }
    fn total(&self) -> anyhow::Result<u64> {
        add(
            self.usage.compile.operations,
            self.usage.inference.operations,
        )
    }
    fn validate(
        &self,
        policy: &Policy,
        expected_hash: &str,
        boot: &str,
        now: u64,
    ) -> anyhow::Result<()> {
        ensure!(
            self.schema_version == 2
                && self.policy_sha256 == expected_hash
                && self.epoch_id == policy.epoch_id,
            "governor policy or epoch changed"
        );
        ensure!(
            boot_valid(&self.boot_id) && self.boot_id == boot,
            "boot changed; external recovery required"
        );
        ensure!(
            self.last_observed_ns <= now && self.not_before_ns >= self.last_observed_ns,
            "governor clock or cooldown is inconsistent"
        );
        for kind in [Kind::Compile, Kind::Inference] {
            let usage = self.usage.get(kind);
            let limits = policy.operations.get(kind);
            ensure!(
                u128::from(usage.operations) * 32 <= u128::from(usage.charged_buckets)
                    && u128::from(usage.charged_buckets) <= u128::from(usage.operations) * 32768,
                "inconsistent charged bucket count"
            );
            ensure!(
                usage.operations <= limits.max_operations
                    && usage.charged_buckets <= limits.max_charged_buckets,
                "persisted governor budget exceeds policy"
            );
        }
        if let Some(done) = &self.last_completed {
            let limits = policy.operations.get(done.kind);
            ensure!(
                bucket_valid(done.bucket)
                    && policy.allowed_devices.contains(&done.device)
                    && done.total_operations == self.total()?
                    && self.usage.get(done.kind).operations > 0,
                "invalid completed operation"
            );
            ensure!(
                done.started_ns <= done.finished_ns
                    && done.finished_ns == self.last_observed_ns
                    && done.cooldown_until_ns == self.not_before_ns,
                "completion clock mismatch"
            );
            let elapsed = done.finished_ns - done.started_ns;
            ensure!(
                elapsed <= limits.max_duration_ns
                    && add(done.finished_ns, limits.cooldown(elapsed)?)? == done.cooldown_until_ns,
                "completed duration or cooldown violates policy"
            );
            ensure!(
                hash(&done.checkpoint.request_sha256) && hash(&done.checkpoint.result_sha256),
                "invalid completion checkpoint"
            );
        } else {
            ensure!(
                self.total()? == 0,
                "charged operation lacks durable completion"
            );
        }
        Ok(())
    }
}

fn now() -> anyhow::Result<u64> {
    let value = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    add(
        u64::try_from(value.tv_sec)?
            .checked_mul(1_000_000_000)
            .context("monotonic overflow")?,
        u64::try_from(value.tv_nsec)?,
    )
}
fn boot() -> anyhow::Result<String> {
    let value = std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
    let value = value.trim();
    ensure!(boot_valid(value), "invalid boot identity");
    Ok(value.into())
}
fn read(file: &mut File) -> anyhow::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(0))?;
    let mut raw = Vec::new();
    file.take(MAX_BYTES + 1).read_to_end(&mut raw)?;
    ensure!(
        raw.len() as u64 <= MAX_BYTES,
        "governor record exceeds size bound"
    );
    Ok(raw)
}
fn write(file: &mut File, raw: &[u8]) -> anyhow::Result<()> {
    ensure!(
        raw.len() as u64 <= MAX_BYTES,
        "governor record exceeds size bound"
    );
    file.seek(SeekFrom::Start(0))?;
    file.write_all(raw)?;
    file.set_len(raw.len() as u64)?;
    file.sync_all()?;
    Ok(())
}
fn write_json(file: &mut File, value: &impl Serialize) -> anyhow::Result<()> {
    write(file, &serde_json::to_vec(value)?)
}

fn directory(path: &Path) -> anyhow::Result<std::os::fd::OwnedFd> {
    use rustix::fs::{FileType, Mode, OFlags, fstat, open, openat};
    ensure!(path.is_absolute(), "governor directory must be absolute");
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let mut fd = open("/", flags, Mode::empty())?;
    let check = |fd: &std::os::fd::OwnedFd| -> anyhow::Result<()> {
        let meta = fstat(fd)?;
        ensure!(
            meta.st_uid == 0
                && FileType::from_raw_mode(meta.st_mode) == FileType::Directory
                && meta.st_mode & 0o022 == 0,
            "governor ancestry must be root-owned and nonwritable"
        );
        Ok(())
    };
    check(&fd)?;
    for component in path.components() {
        match component {
            Component::RootDir => (),
            Component::Normal(name) => {
                fd = openat(&fd, name, flags, Mode::empty())?;
                check(&fd)?;
            }
            _ => anyhow::bail!("invalid governor directory component"),
        }
    }
    Ok(fd)
}
fn file(dir: &std::os::fd::OwnedFd, name: &str, writable: bool) -> anyhow::Result<File> {
    use rustix::fs::{FileType, Mode, OFlags, fstat, openat};
    let access = if writable {
        OFlags::RDWR
    } else {
        OFlags::RDONLY
    };
    let fd = openat(
        dir,
        name,
        access | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::CLOEXEC,
        Mode::empty(),
    )?;
    let meta = fstat(&fd)?;
    ensure!(
        meta.st_uid == 0
            && meta.st_gid == fstat(dir)?.st_gid
            && meta.st_nlink == 1
            && FileType::from_raw_mode(meta.st_mode) == FileType::RegularFile,
        "governor files must be root-owned regular single-link files in the installation group"
    );
    ensure!(
        meta.st_mode & 0o002 == 0
            && (writable || meta.st_mode & 0o022 == 0)
            && meta.st_size >= 0
            && meta.st_size as u64 <= MAX_BYTES,
        "unsafe governor file mode or size"
    );
    Ok(File::from(fd))
}

pub struct Governor {
    policy_sha256: String,
}
pub struct Lease {
    _lock: File,
    state_file: File,
    intent_file: File,
    state: State,
    limits: Limits,
    kind: Kind,
    device: Device,
    bucket: u64,
    started_ns: u64,
    pub deadline_ns: u64,
}
impl Governor {
    pub fn installed(policy_sha256: String) -> anyhow::Result<Self> {
        ensure!(hash(&policy_sha256), "invalid pinned policy digest");
        Ok(Self { policy_sha256 })
    }
    pub fn begin(&self, kind: Kind, device: Device, bucket: u64) -> anyhow::Result<Lease> {
        ensure!(bucket_valid(bucket), "invalid final sequence bucket");
        let path = Path::new(DIRECTORY);
        let dir = directory(path)?;
        let mut policy_file = file(&dir, "policy.json", false)?;
        let lock = file(&dir, "operation.lock", true)?;
        let mut state_file = file(&dir, "state.json", true)?;
        let mut intent_file = file(&dir, "intent.json", true)?;
        let raw = read(&mut policy_file)?;
        ensure!(
            digest(&raw) == self.policy_sha256,
            "pinned governor policy changed"
        );
        let policy: Policy = serde_json::from_slice(&raw)?;
        policy.validate(path)?;
        ensure!(
            policy.allowed_devices.contains(&device),
            "device excluded from governor policy"
        );
        let wait_deadline = add(now()?, policy.lock_wait_ns)?;
        loop {
            match rustix::fs::flock(&lock, rustix::fs::FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => break,
                Err(rustix::io::Errno::WOULDBLOCK) => {
                    ensure!(now()? < wait_deadline, "governor lock wait exhausted");
                    std::thread::sleep(POLL);
                }
                Err(error) => return Err(error.into()),
            }
        }
        ensure!(
            read(&mut intent_file)?.is_empty(),
            "pending native intent; external recovery required"
        );
        ensure!(
            digest(&read(&mut policy_file)?) == self.policy_sha256,
            "policy changed while waiting"
        );
        let mut state: State = serde_json::from_slice(&read(&mut state_file)?)?;
        let current_boot = boot()?;
        state.validate(&policy, &self.policy_sha256, &current_boot, now()?)?;
        ensure!(
            state.not_before_ns <= wait_deadline,
            "cooldown exceeds bounded governor wait"
        );
        while now()? < state.not_before_ns {
            std::thread::sleep(POLL);
        }
        let limits = policy.operations.get(kind).clone();
        let usage = state.usage.get_mut(kind);
        ensure!(
            usage.operations < limits.max_operations
                && add(usage.charged_buckets, bucket)? <= limits.max_charged_buckets,
            "provisioned epoch budget exhausted"
        );
        let started_ns = now()?;
        ensure!(started_ns <= wait_deadline, "governor lock wait exhausted");
        let deadline_ns = add(started_ns, limits.max_duration_ns)?;
        usage.operations = add(usage.operations, 1)?;
        usage.charged_buckets = add(usage.charged_buckets, bucket)?;
        let intent = serde_json::json!({"schema_version": 2, "policy_sha256": self.policy_sha256,
            "epoch_id": policy.epoch_id, "boot_id": current_boot, "kind": kind,
            "device": device, "bucket": bucket, "operation_id": state.total()?,
            "intent_ns": started_ns, "deadline_ns": deadline_ns, "pid": std::process::id()});
        write_json(&mut intent_file, &intent)?;
        state.last_observed_ns = started_ns;
        write_json(&mut state_file, &state)?;
        ensure!(now()? < deadline_ns, "deadline expired before native entry");
        Ok(Lease {
            _lock: lock,
            state_file,
            intent_file,
            state,
            limits,
            kind,
            device,
            bucket,
            started_ns,
            deadline_ns,
        })
    }
}
impl Lease {
    /// Poll an already-owned worker and a nonblocking bounded response channel.
    /// The worker must not start its native call before this lease is obtained.
    /// Errors retain the durable intent; another backend must not bypass it.
    pub fn supervise<T>(
        &self,
        child: &mut Child,
        mut response: impl FnMut() -> anyhow::Result<Option<T>>,
    ) -> anyhow::Result<T> {
        let outcome = (|| {
            loop {
                ensure!(
                    now()? < self.deadline_ns,
                    "native operation timed out; external recovery required"
                );
                if let Some(value) = response()? {
                    ensure!(now()? < self.deadline_ns, "native response missed deadline");
                    return Ok(value);
                }
                ensure!(
                    child.try_wait()?.is_none(),
                    "native worker exited without a response; external recovery required"
                );
                std::thread::sleep(POLL);
            }
        })();
        if outcome.is_err() {
            // This also covers clock/read errors and late or malformed replies.
            // Never allow another launch merely because the wait loop returned.
            crate::local_adapter::terminate(child)?;
        }
        outcome
    }

    pub fn complete(mut self, checkpoint: Checkpoint) -> anyhow::Result<()> {
        ensure!(
            hash(&checkpoint.request_sha256) && hash(&checkpoint.result_sha256),
            "invalid durable checkpoint"
        );
        let finished_ns = now()?;
        ensure!(
            finished_ns >= self.started_ns && finished_ns <= self.deadline_ns,
            "native operation violated its deadline"
        );
        ensure!(
            boot()? == self.state.boot_id,
            "boot changed during native operation"
        );
        let cooldown_until_ns = add(
            finished_ns,
            self.limits.cooldown(finished_ns - self.started_ns)?,
        )?;
        self.state.last_observed_ns = finished_ns;
        self.state.not_before_ns = cooldown_until_ns;
        self.state.last_completed = Some(Completed {
            kind: self.kind,
            device: self.device,
            bucket: self.bucket,
            total_operations: self.state.total()?,
            started_ns: self.started_ns,
            finished_ns,
            cooldown_until_ns,
            checkpoint,
        });
        write_json(&mut self.state_file, &self.state)?;
        write(&mut self.intent_file, b"")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const BOOT: &str = "11111111-1111-1111-1111-111111111111";
    fn limits() -> Limits {
        Limits {
            max_operations: 10,
            max_charged_buckets: 640,
            max_duration_ns: 100,
            minimum_cooldown_ns: 1,
            cooldown_numerator: 1,
            cooldown_denominator: 1,
        }
    }
    fn policy() -> Policy {
        Policy {
            schema_version: 2,
            namespace: "cfetch-host-inference-v2".into(),
            epoch_id: "explicit-window".into(),
            state_directory: DIRECTORY.into(),
            allowed_devices: vec![Device::Cpu, Device::Gpu],
            lock_wait_ns: 1000,
            operations: Operations {
                compile: limits(),
                inference: limits(),
            },
        }
    }
    fn initial() -> State {
        State::initial("a".repeat(64), policy().epoch_id, BOOT.into(), 10).unwrap()
    }
    fn completed() -> State {
        let mut state = initial();
        state.usage.inference = Usage {
            operations: 1,
            charged_buckets: 64,
        };
        state.last_observed_ns = 30;
        state.not_before_ns = 41;
        state.last_completed = Some(Completed {
            kind: Kind::Inference,
            device: Device::Gpu,
            bucket: 64,
            total_operations: 1,
            started_ns: 20,
            finished_ns: 30,
            cooldown_until_ns: 41,
            checkpoint: Checkpoint {
                request_sha256: "b".repeat(64),
                result_sha256: "c".repeat(64),
            },
        });
        state
    }
    #[test]
    fn explicit_policy_has_no_npu_admission() {
        let mut row = policy();
        row.validate(Path::new(DIRECTORY)).unwrap();
        assert!(!row.allowed_devices.contains(&Device::Npu));
        row.allowed_devices.push(Device::Gpu);
        assert!(row.validate(Path::new(DIRECTORY)).is_err());
    }
    #[test]
    fn policy_rejects_other_namespace_and_unbounded_values() {
        let mut row = policy();
        row.schema_version = 1;
        assert!(row.validate(Path::new(DIRECTORY)).is_err());
        row = policy();
        row.lock_wait_ns = 0;
        assert!(row.validate(Path::new(DIRECTORY)).is_err());
        row = policy();
        row.allowed_devices.clear();
        assert!(row.validate(Path::new(DIRECTORY)).is_err());
        assert!(
            policy()
                .validate(Path::new("/tmp/alternative-budget"))
                .is_err()
        );
    }
    #[test]
    fn cooldown_enforces_worst_case_half_duty_cycle() {
        let mut row = limits();
        row.validate().unwrap();
        row.cooldown_numerator = 0;
        assert!(row.validate().is_err());
        row.minimum_cooldown_ns = row.max_duration_ns;
        row.validate().unwrap();
        row.cooldown_denominator = 0;
        assert!(row.validate().is_err());
    }
    #[test]
    fn cooldown_rounds_up_and_refuses_overflow() {
        let mut row = limits();
        row.cooldown_denominator = 3;
        assert_eq!(row.cooldown(4).unwrap(), 3);
        row.cooldown_numerator = u64::MAX;
        assert!(row.cooldown(u64::MAX).is_err());
        assert!(add(u64::MAX, 1).is_err());
    }
    #[test]
    fn state_never_resets_on_restart_boot_or_policy_change() {
        initial()
            .validate(&policy(), &"a".repeat(64), BOOT, 10)
            .unwrap();
        let state = completed();
        state
            .validate(&policy(), &"a".repeat(64), BOOT, 100)
            .unwrap();
        assert!(
            state
                .validate(&policy(), &"b".repeat(64), BOOT, 100)
                .is_err()
        );
        assert!(
            state
                .validate(
                    &policy(),
                    &"a".repeat(64),
                    "22222222-2222-2222-2222-222222222222",
                    100
                )
                .is_err()
        );
        assert!(
            state
                .validate(&policy(), &"a".repeat(64), BOOT, 29)
                .is_err()
        );
    }
    #[test]
    fn charged_without_completion_requires_external_recovery() {
        let mut state = initial();
        state.usage.inference = Usage {
            operations: 1,
            charged_buckets: 64,
        };
        assert!(
            state
                .validate(&policy(), &"a".repeat(64), BOOT, 100)
                .is_err()
        );
    }
    #[test]
    fn completion_cannot_hide_device_budget_or_cooldown_changes() {
        for change in 0..5 {
            let mut state = completed();
            match change {
                0 => state.last_completed.as_mut().unwrap().device = Device::Npu,
                1 => state.usage.inference.operations = 11,
                2 => state.usage.inference.charged_buckets = 641,
                3 => state.last_completed.as_mut().unwrap().cooldown_until_ns = 40,
                _ => {
                    state
                        .last_completed
                        .as_mut()
                        .unwrap()
                        .checkpoint
                        .result_sha256 = "wrong".into()
                }
            }
            assert!(
                state
                    .validate(&policy(), &"a".repeat(64), BOOT, 100)
                    .is_err()
            );
        }
    }
    #[test]
    fn duplicate_policy_fields_and_old_python_schema_are_refused() {
        assert!(
            serde_json::from_str::<Usage>(
                r#"{"operations":1,"operations":0,"charged_buckets":64}"#
            )
            .is_err()
        );
        assert!(serde_json::from_str::<Operations<Usage>>(r#"{"compile":{"operations":0,"charged_buckets":0},"inference":{"operations":0,"charged_buckets":0},"inference":{"operations":0,"charged_buckets":0}}"#).is_err());
        let mut old = serde_json::to_value(policy()).unwrap();
        old["schema_version"] = 1.into();
        let old: Policy = serde_json::from_value(old).unwrap();
        assert!(old.validate(Path::new(DIRECTORY)).is_err());
    }
    #[test]
    fn secure_namespace_refuses_temporary_directory_and_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        assert!(directory(dir.path()).is_err());
        let link = dir.path().join("alias");
        std::os::unix::fs::symlink("/", &link).unwrap();
        assert!(directory(&link).is_err());
        // Open the test directory directly to exercise NOFOLLOW independently
        // of its deliberately invalid production ancestry/ownership.
        let fd = rustix::fs::open(
            dir.path(),
            rustix::fs::OFlags::RDONLY | rustix::fs::OFlags::DIRECTORY,
            rustix::fs::Mode::empty(),
        )
        .unwrap();
        let error = file(&fd, "alias", false).unwrap_err();
        assert_eq!(
            error.downcast_ref::<rustix::io::Errno>(),
            Some(&rustix::io::Errno::LOOP)
        );
    }
    #[test]
    fn torn_or_oversized_state_never_parses_as_initial_state() {
        assert!(serde_json::from_slice::<State>(b"{\"schema_version\":2").is_err());
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(&vec![b'x'; MAX_BYTES as usize + 1]).unwrap();
        assert!(read(&mut file).is_err());
    }
    fn lease(duration: Duration) -> Lease {
        let mut intent_file = tempfile::tempfile().unwrap();
        intent_file.write_all(b"durable pending intent").unwrap();
        intent_file.sync_all().unwrap();
        let started_ns = now().unwrap();
        let mut state = initial();
        state.boot_id = boot().unwrap();
        state.usage.inference = Usage {
            operations: 1,
            charged_buckets: 64,
        };
        let mut budget = limits();
        budget.max_duration_ns = duration.as_nanos().try_into().unwrap();
        Lease {
            _lock: tempfile::tempfile().unwrap(),
            state_file: tempfile::tempfile().unwrap(),
            intent_file,
            state,
            limits: budget,
            kind: Kind::Inference,
            device: Device::Gpu,
            bucket: 64,
            started_ns,
            deadline_ns: started_ns + u64::try_from(duration.as_nanos()).unwrap(),
        }
    }
    #[test]
    fn timeout_reaps_only_owned_child_and_retains_intent() {
        let mut owned = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let mut other = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let mut lease = lease(Duration::from_millis(50));
        let result = lease.supervise::<()>(&mut owned, || Ok(None));
        let owned_exited = owned.try_wait().unwrap().is_some();
        let other_alive = other.try_wait().unwrap().is_none();
        // Clean test processes before assertions, including regression failures.
        crate::local_adapter::terminate(&mut owned).unwrap();
        crate::local_adapter::terminate(&mut other).unwrap();
        assert!(result.is_err());
        assert!(owned_exited && other_alive);
        assert!(!read(&mut lease.intent_file).unwrap().is_empty());
    }
    #[test]
    fn worker_crash_mid_operation_retains_pending_intent() {
        let mut child = std::process::Command::new("sh")
            .args(["-c", "sleep 0.03; exit 7"])
            .spawn()
            .unwrap();
        let mut lease = lease(Duration::from_secs(1));
        let result = lease.supervise::<()>(&mut child, || Ok(None));
        assert!(result.is_err());
        assert_eq!(child.try_wait().unwrap().unwrap().code(), Some(7));
        assert_eq!(
            read(&mut lease.intent_file).unwrap(),
            b"durable pending intent"
        );
        assert!(read(&mut lease.state_file).unwrap().is_empty());
    }
    #[test]
    fn malformed_worker_reply_reaps_owned_child() {
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let mut lease = lease(Duration::from_secs(1));
        assert!(
            lease
                .supervise::<()>(&mut child, || anyhow::bail!("malformed reply"))
                .is_err()
        );
        assert!(child.try_wait().unwrap().is_some());
        assert!(!read(&mut lease.intent_file).unwrap().is_empty());
    }
    #[test]
    fn checkpoint_is_persisted_before_pending_intent_is_cleared() {
        let lease = lease(Duration::from_secs(1));
        let mut state_file = lease.state_file.try_clone().unwrap();
        let mut intent_file = lease.intent_file.try_clone().unwrap();
        let checkpoint = Checkpoint {
            request_sha256: "d".repeat(64),
            result_sha256: "e".repeat(64),
        };
        lease.complete(checkpoint.clone()).unwrap();
        let state: State = serde_json::from_slice(&read(&mut state_file).unwrap()).unwrap();
        assert_eq!(
            state.last_completed.unwrap().checkpoint.result_sha256,
            checkpoint.result_sha256
        );
        assert!(read(&mut intent_file).unwrap().is_empty());
    }
    #[test]
    fn rejected_checkpoint_preserves_pending_intent() {
        let lease = lease(Duration::from_secs(1));
        let mut intent_file = lease.intent_file.try_clone().unwrap();
        assert!(
            lease
                .complete(Checkpoint {
                    request_sha256: "invalid".into(),
                    result_sha256: "e".repeat(64)
                })
                .is_err()
        );
        assert!(!read(&mut intent_file).unwrap().is_empty());
    }
}
