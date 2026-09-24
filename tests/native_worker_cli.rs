#![cfg(all(target_os = "linux", feature = "native-openvino"))]

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn worker(parent: u32, loader_override: bool) -> std::process::ExitStatus {
    let mut command = Command::new(env!("CARGO_BIN_EXE_cfetch"));
    // Cargo injects a loader search path for its own test process. The fixture
    // explicitly creates a clean child; production rejects rather than strips it.
    command.env_clear();
    if loader_override {
        command.env(
            "OPENVINO_INSTALL_DIR",
            "/nonexistent-native-worker-test-runtime",
        );
    }
    let mut child = command
        .args(["native-worker", "--parent-pid", &parent.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("idle native worker did not exit within its deadline");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn empty_input_exits_without_loading_native_runtime() {
    assert!(worker(std::process::id(), false).success());
}
#[test]
fn wrong_parent_is_rejected_before_runtime_initialization() {
    assert!(!worker(u32::MAX, false).success());
}

#[test]
fn inherited_loader_override_is_rejected_even_before_a_command() {
    assert!(!worker(std::process::id(), true).success());
}
