//! Filesystem locations.
//!
//! Mutable indexes, sockets and heartbeat belong to the local runtime.
//! Markdown belongs to the configured brain and selected mind. Physical
//! placement and file sharing are outside cfetch's contract.
//!
//! Every location keeps the same override environment variable on every
//! platform (`HOME`, `CFETCH_STATE_DIR`, `CFETCH_CONFIG`, `CFETCH_BRAIN`);
//! only the DEFAULT layout differs — XDG-shaped on unix, `%LOCALAPPDATA%` /
//! `%APPDATA%` on Windows. Both layouts are compiled everywhere so the tests
//! prove both on any runner; only one is reachable at runtime.

use std::path::{Path, PathBuf};

#[cfg(unix)]
pub fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| "/".into()))
}

/// `HOME` first — it is the documented override and what the test harness
/// sets — then the Windows profile variables, which is where a real Windows
/// session actually keeps it.
#[cfg(windows)]
pub fn home() -> PathBuf {
    for key in ["HOME", "USERPROFILE"] {
        if let Some(v) = std::env::var_os(key).filter(|v| !v.is_empty()) {
            return PathBuf::from(v);
        }
    }
    if let (Some(drive), Some(path)) = (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH"))
    {
        let mut p = PathBuf::from(drive);
        p.push(path);
        return p;
    }
    PathBuf::from("C:\\")
}

/// XDG-shaped state location (Linux, macOS).
#[cfg_attr(windows, allow(dead_code))]
fn xdg_state_dir(home: &Path) -> PathBuf {
    home.join(".local/state/cfetch")
}

/// Windows state location: per-machine, NON-roaming — the SQLite index must
/// never follow a roaming profile onto a network share.
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_state_dir(home: &Path, local_app_data: Option<&Path>) -> PathBuf {
    match local_app_data {
        Some(d) => d.join("cfetch"),
        None => home.join("AppData/Local/cfetch"),
    }
}

/// XDG-shaped config location (Linux, macOS).
#[cfg_attr(windows, allow(dead_code))]
fn xdg_config_path(home: &Path) -> PathBuf {
    home.join(".config/cfetch/config.json")
}

/// Windows config location: roaming, like every other per-user tool config.
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_config_path(home: &Path, app_data: Option<&Path>) -> PathBuf {
    match app_data {
        Some(d) => d.join("cfetch/config.json"),
        None => home.join("AppData/Roaming/cfetch/config.json"),
    }
}

/// Per-host mutable state: heartbeat, derived indexes, daemon socket fallback.
/// Nothing of record lives here.
pub fn state_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("CFETCH_STATE_DIR") {
        return PathBuf::from(d);
    }
    #[cfg(unix)]
    {
        xdg_state_dir(&home())
    }
    #[cfg(windows)]
    {
        windows_state_dir(&home(), std::env::var_os("LOCALAPPDATA").map(PathBuf::from).as_deref())
    }
}

/// Daemon socket: runtime dir when available (tmpfs, cleaned on logout),
/// state dir otherwise. Unix only — Windows has no socket file; the loopback
/// endpoint that replaces it there lives in [`crate::ipc`].
#[cfg(unix)]
pub fn socket_path() -> PathBuf {
    if let Some(rt) = std::env::var_os("XDG_RUNTIME_DIR") {
        let rt = PathBuf::from(rt);
        if rt.is_dir() {
            return rt.join("cfetch.sock");
        }
    }
    state_dir().join("daemon.sock")
}

pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("CFETCH_CONFIG") {
        return PathBuf::from(p);
    }
    #[cfg(unix)]
    {
        xdg_config_path(&home())
    }
    #[cfg(windows)]
    {
        windows_config_path(&home(), std::env::var_os("APPDATA").map(PathBuf::from).as_deref())
    }
}

/// Codex's public configuration/state root. `CODEX_HOME` is the supported
/// override used by the CLI, IDE, app server and installers; an empty value
/// behaves like an unset value.
pub fn codex_home() -> PathBuf {
    std::env::var_os("CODEX_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".codex"))
}

/// Claude Code's per-project native auto-memory stores live under here.
/// cfetch indexes them read-only; it never writes to the native store.
pub fn native_projects_root() -> PathBuf {
    home().join(".claude/projects")
}

/// Codex session transcripts. Unlike Claude's project store this is nested by
/// date, so transcript discovery walks recursively.
pub fn codex_sessions_root() -> PathBuf {
    codex_home().join("sessions")
}

/// Gemini CLI session transcripts, grouped by project.
pub fn gemini_sessions_root() -> PathBuf {
    home().join(".gemini/tmp")
}

/// Cursor agent transcripts, grouped by workspace.
pub fn cursor_sessions_root() -> PathBuf {
    home().join(".cursor/projects")
}

/// The shared brain tree (source of truth, git-tracked markdown).
pub fn default_brain_root() -> PathBuf {
    match std::env::var_os("CFETCH_BRAIN") {
        Some(p) => PathBuf::from(p),
        None => home().join("agents"),
    }
}

/// The SHARED vector artifact store, inside the brain tree: vectors are
/// derived from shared content, so they are computed once per storage group
/// and read by every host that can reach the tree — never recomputed per
/// host, and never a per-host database's private property.
pub fn shared_vector_dir(brain_root: &std::path::Path) -> PathBuf {
    brain_root.join("scratch/cfetch/vectors")
}

/// Append-only JSONL streams of record (ring-6 exhaust, injection ledger).
/// Under `logs/`, which the index excludes wholesale — the streams are record,
/// not recallable prose, and a per-tool-call append must never churn the
/// serving daemon's watcher.
pub fn logs_dir(brain_root: &Path) -> PathBuf {
    brain_root.join("logs/cfetch")
}

/// Ring-5 provisional evidence belongs to the selected mind.
pub fn staging_dir(brain_root: &Path) -> PathBuf {
    mind_dir(brain_root).join("memories")
}

/// A mind may serve several machines. CFETCH_MIND selects its directory
/// name independently of the host identity used to attribute log records.
pub fn mind_id() -> String {
    std::env::var("CFETCH_MIND").unwrap_or_else(|_| host_id())
}

pub fn validate_mind_id() -> anyhow::Result<()> {
    let id = mind_id();
    anyhow::ensure!(
        !id.is_empty() && id != "." && id != ".." && id != "models"
            && id.bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.')),
        "CFETCH_MIND must be one directory name (letters, digits, '.', '_' or '-'), excluding '.', '..' and 'models'"
    );
    Ok(())
}

pub fn mind_dir(brain_root: &Path) -> PathBuf {
    brain_root.join("mind").join(mind_id())
}

/// Tool configuration inside the tree — shared by every host that mounts it,
/// versioned with the content it describes. Hidden because it sits beside
/// `.git/` and is machine-written, and because the walker skips hidden paths
/// so no rule has to remember to exclude it.
pub fn tree_config_path(brain_root: &Path) -> PathBuf {
    brain_root.join(".cfetch/config.json")
}

/// Identity stamped into this host's stream file names and staged candidates.
/// `CFETCH_HOST` overrides it — a container or a test needs to be able to say
/// who it is without renaming the machine.
pub fn host_id() -> String {
    let raw = std::env::var("CFETCH_HOST")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(hostname);
    crate::jsonl::sanitize_host(raw.trim())
}

/// This machine's name. `uname(2)` is portable across the unixes (Linux,
/// macOS, BSD) where `/proc` is not; Windows has neither, and `COMPUTERNAME`
/// is the equivalent its session manager always sets.
pub fn hostname() -> String {
    #[cfg(unix)]
    let node = {
        let uname = rustix::system::uname();
        uname.nodename().to_string_lossy().trim().to_string()
    };
    #[cfg(windows)]
    let node = std::env::var("COMPUTERNAME").unwrap_or_default().trim().to_string();
    if !node.is_empty() {
        return node;
    }
    std::env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xdg_layout_is_unchanged() {
        let home = Path::new("/home/agent");
        assert_eq!(xdg_state_dir(home), PathBuf::from("/home/agent/.local/state/cfetch"));
        assert_eq!(xdg_config_path(home), PathBuf::from("/home/agent/.config/cfetch/config.json"));
    }

    #[test]
    fn windows_layout_prefers_the_shell_folders_and_falls_back_under_the_profile() {
        let home = Path::new("C:/Users/agent");
        assert_eq!(
            windows_state_dir(home, Some(Path::new("C:/Users/agent/AppData/Local"))),
            PathBuf::from("C:/Users/agent/AppData/Local/cfetch")
        );
        assert_eq!(
            windows_state_dir(home, None),
            PathBuf::from("C:/Users/agent/AppData/Local/cfetch"),
            "a missing LOCALAPPDATA lands in the same place, never in the XDG tree"
        );
        assert_eq!(
            windows_config_path(home, Some(Path::new("C:/Users/agent/AppData/Roaming"))),
            PathBuf::from("C:/Users/agent/AppData/Roaming/cfetch/config.json")
        );
        assert_eq!(
            windows_config_path(home, None),
            PathBuf::from("C:/Users/agent/AppData/Roaming/cfetch/config.json")
        );
    }

    #[test]
    fn the_index_never_defaults_into_a_roaming_profile() {
        // Roaming would put a SQLite database on a network share, which is
        // exactly what the per-host state dir exists to avoid.
        let p = windows_state_dir(Path::new("C:/Users/agent"), None);
        assert!(!p.to_string_lossy().contains("Roaming"), "{p:?}");
    }
}
