//! The daemon's LOCAL control channel — the transport hooks and the CLI use
//! to reach the warm daemon on this machine.
//!
//! On unix that is the unix socket it has always been: the socket file's mode
//! is the access control, so a local request carries no credential and the
//! bytes on the wire are exactly the caller's request object.
//!
//! Windows has no socket file. A loopback TCP listener is the portable
//! equivalent, but a TCP port is reachable by every process on the machine,
//! whatever user it runs as — so the Windows local channel is gated by a
//! PER-DAEMON bearer token, checked with the same `serve::token_eq`
//! comparison the serving TCP listener already uses. Address and token are
//! published together in ONE atomically replaced state-dir file, so a client
//! can never pair a freshly bound port with a dead daemon's token.
//!
//! `LOCAL_REQUIRES_TOKEN` is the single fact the rest of the daemon reads:
//! whether this platform's local transport is access-controlled by the
//! operating system or by cfetch.
//!
//! A Windows client reads the published endpoint twice per call — once to
//! connect, once to stamp the token. A daemon restart between the two would
//! pair a new port with an old token; the request is then refused, which
//! every caller of `daemon::call_req` already tolerates (it returns `None`
//! and the hook falls back). Nothing is left inconsistent, so the race is
//! documented rather than engineered away.

use std::borrow::Cow;

/// True when the local transport is NOT access-controlled by the operating
/// system and cfetch must gate it with its own bearer token.
pub const LOCAL_REQUIRES_TOKEN: bool = cfg!(windows);

/// Endpoint file body: address on the first line, token on the second.
///
/// One file, written once, replaced atomically — never two files a client
/// could read a fresh half and a stale half of.
// Compiled on every platform so the tests below prove the Windows wire format
// on any runner; reachable at runtime only where the local channel is TCP.
#[cfg_attr(not(windows), allow(dead_code))]
fn render_endpoint(addr: &str, token: &str) -> String {
    format!("{addr}\n{token}\n")
}

/// Parses [`render_endpoint`]. `None` for anything short of BOTH a non-empty
/// address and a non-empty token — a half-written file must read as "no
/// daemon", never as an unauthenticated endpoint.
#[cfg_attr(not(windows), allow(dead_code))]
fn parse_endpoint(raw: &str) -> Option<(String, String)> {
    let mut lines = raw.lines();
    let addr = lines.next()?.trim();
    let token = lines.next()?.trim();
    if addr.is_empty() || token.is_empty() {
        return None;
    }
    Some((addr.to_string(), token.to_string()))
}

/// A fresh bearer token for one daemon's local channel: 128 bits, hex.
///
/// Entropy without a new dependency: the standard library seeds every
/// `RandomState` from the operating system's CSPRNG (`ProcessPrng` on
/// Windows), and SipHash under an unknown key is a pseudo-random function —
/// so hashing fixed inputs under two independently constructed states yields
/// bits an attacker cannot predict. Process id and wall-clock nanoseconds are
/// mixed in as defence in depth, never as the only source.
#[cfg_attr(not(windows), allow(dead_code))]
fn new_token() -> String {
    use std::hash::{BuildHasher as _, Hasher as _};
    let mix = |salt: u64| -> u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(salt);
        h.write_u32(std::process::id());
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        h.finish()
    };
    format!(
        "{:016x}{:016x}",
        mix(0x9e37_79b9_7f4a_7c15),
        mix(0xbf58_476d_1ce4_e5b9)
    )
}

/// The credential a fresh daemon should publish for its local channel:
/// `None` where the operating system already gates the transport.
///
/// Minted BEFORE the listener binds so the daemon's request context can carry
/// it without reordering anything else in the boot sequence.
pub fn new_local_token() -> Option<String> {
    if LOCAL_REQUIRES_TOKEN {
        Some(new_token())
    } else {
        None
    }
}

#[cfg(unix)]
mod imp {
    use std::io;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::PathBuf;
    use std::time::Duration;

    use crate::paths;

    /// One accepted local connection.
    pub type Stream = UnixStream;

    /// The bound local channel.
    pub struct Listener {
        inner: UnixListener,
        path: PathBuf,
    }

    /// Where the local channel lives, for status output and error messages.
    pub fn describe() -> String {
        paths::socket_path().display().to_string()
    }

    /// Binds the local channel. A stale socket file from a dead daemon is
    /// cleared; a live one is never stolen.
    pub fn listen(_token: Option<String>) -> anyhow::Result<Listener> {
        let sock = paths::socket_path();
        if let Some(dir) = sock.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if sock.exists() {
            if UnixStream::connect(&sock).is_ok() {
                anyhow::bail!("daemon already running on {}", sock.display());
            }
            std::fs::remove_file(&sock)?;
        }
        let inner = UnixListener::bind(&sock)?;
        // Bind does not choose a mode: the default comes from the process
        // umask, and "socket mode is the access control" only holds if the
        // mode is ours. Pin it to owner-only regardless of environment; a
        // chmod that cannot be applied is a socket not worth serving on.
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&sock, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(Listener { inner, path: sock })
    }

    impl Listener {
        pub fn incoming(&self) -> impl Iterator<Item = io::Result<Stream>> + '_ {
            self.inner.incoming()
        }

        pub fn describe(&self) -> String {
            self.path.display().to_string()
        }

        /// Removes the published endpoint on a clean shutdown.
        pub fn cleanup(&self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Client connect with the caller's deadline on both directions.
    pub fn connect(timeout: Duration) -> Option<Stream> {
        use rustix::net::{AddressFamily, SocketAddrUnix, SocketFlags, SocketType};
        // A full Unix listener backlog must not turn connect into an
        // unbounded wait. Local sockets either connect immediately or report
        // unavailable; callers can retry without leaving background threads.
        let socket = rustix::net::socket_with(
            AddressFamily::UNIX,
            SocketType::STREAM,
            SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
            None,
        )
        .ok()?;
        let address = SocketAddrUnix::new(paths::socket_path()).ok()?;
        rustix::net::connect(&socket, &address).ok()?;
        let stream = UnixStream::from(socket);
        stream.set_nonblocking(false).ok()?;
        stream.set_read_timeout(Some(timeout)).ok()?;
        stream.set_write_timeout(Some(timeout)).ok()?;
        Some(stream)
    }

    /// Nudges the accept loop so it observes a shutdown flag.
    pub fn wake() {
        let _ = UnixStream::connect(paths::socket_path());
    }

    /// The local channel needs no credential here, so the request goes out
    /// byte-for-byte as the caller built it.
    pub fn authenticate(body: &serde_json::Value) -> super::Cow<'_, serde_json::Value> {
        super::Cow::Borrowed(body)
    }
}

#[cfg(windows)]
mod imp {
    use std::io;
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use crate::paths;

    /// One accepted local connection.
    pub type Stream = TcpStream;

    fn endpoint_file() -> PathBuf {
        paths::state_dir().join("daemon.endpoint")
    }

    fn published() -> Option<(String, String)> {
        super::parse_endpoint(&std::fs::read_to_string(endpoint_file()).ok()?)
    }

    /// Atomic replace so a reader never sees a half-written endpoint.
    /// `rename` maps to `MoveFileExW(MOVEFILE_REPLACE_EXISTING)` here, so an
    /// existing endpoint file from a dead daemon is overwritten, not refused.
    fn write_atomic(path: &Path, content: &str) -> io::Result<()> {
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, content)?;
        std::fs::rename(&tmp, path)
    }

    /// Where the local channel lives, for status output and error messages.
    pub fn describe() -> String {
        match published() {
            Some((addr, _)) => format!("tcp {addr} (loopback, token-gated)"),
            None => format!("no endpoint published at {}", endpoint_file().display()),
        }
    }

    /// The bound local channel.
    pub struct Listener {
        inner: TcpListener,
        addr: String,
        file: PathBuf,
    }

    /// Binds the local channel on an ephemeral loopback port and publishes
    /// (address, token) for clients.
    ///
    /// The "already running" refusal is decided by a full ping ROUND TRIP,
    /// not by a bare connect: a stale endpoint file may name a port some
    /// unrelated process has since been given, and only an answer that parses
    /// as our protocol proves a cfetch daemon is on the other end.
    pub fn listen(token: Option<String>) -> anyhow::Result<Listener> {
        std::fs::create_dir_all(paths::state_dir())?;
        if crate::daemon::call("ping", Duration::from_millis(300)).is_some_and(|r| r.ok) {
            anyhow::bail!("daemon already running on {}", describe());
        }
        let Some(token) = token else {
            anyhow::bail!("the loopback control channel requires a bearer token");
        };
        let inner = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))?;
        let addr = inner.local_addr()?.to_string();
        let file = endpoint_file();
        write_atomic(&file, &super::render_endpoint(&addr, &token))?;
        Ok(Listener { inner, addr, file })
    }

    impl Listener {
        pub fn incoming(&self) -> impl Iterator<Item = io::Result<Stream>> + '_ {
            self.inner.incoming()
        }

        pub fn describe(&self) -> String {
            format!("tcp {} (loopback, token-gated)", self.addr)
        }

        /// Removes the published endpoint on a clean shutdown.
        pub fn cleanup(&self) {
            let _ = std::fs::remove_file(&self.file);
        }
    }

    /// Client connect with the caller's deadline on both directions.
    pub fn connect(timeout: Duration) -> Option<Stream> {
        let (addr, _) = published()?;
        let sock: SocketAddr = addr.parse().ok()?;
        let stream = TcpStream::connect_timeout(&sock, timeout).ok()?;
        stream.set_read_timeout(Some(timeout)).ok()?;
        stream.set_write_timeout(Some(timeout)).ok()?;
        Some(stream)
    }

    /// Nudges the accept loop so it observes a shutdown flag.
    pub fn wake() {
        let _ = connect(Duration::from_millis(200));
    }

    /// Stamps the published token onto the request: loopback TCP is not
    /// access-controlled, so every local request is a credentialed request.
    pub fn authenticate(body: &serde_json::Value) -> super::Cow<'_, serde_json::Value> {
        let Some((_, token)) = published() else {
            return super::Cow::Borrowed(body);
        };
        let mut owned = body.clone();
        match owned.as_object_mut() {
            Some(map) => {
                map.insert("token".to_string(), serde_json::Value::String(token));
                super::Cow::Owned(owned)
            }
            None => super::Cow::Borrowed(body),
        }
    }
}

pub use imp::{Stream, authenticate, connect, describe, listen, wake};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_local_channel_is_token_gated_exactly_where_the_transport_is_shared() {
        // Unix sockets are access-controlled by file mode; a loopback TCP
        // port is reachable by every local process and is not.
        assert_eq!(LOCAL_REQUIRES_TOKEN, cfg!(windows));
        assert_eq!(new_local_token().is_some(), cfg!(windows));
    }

    #[test]
    fn endpoint_round_trips() {
        let rendered = render_endpoint("127.0.0.1:54321", "0123456789abcdef");
        let (addr, token) = parse_endpoint(&rendered).expect("round trip");
        assert_eq!(addr, "127.0.0.1:54321");
        assert_eq!(token, "0123456789abcdef");
    }

    #[test]
    fn a_half_written_endpoint_reads_as_no_daemon() {
        // Never as an endpoint without a credential.
        assert!(parse_endpoint("").is_none());
        assert!(
            parse_endpoint("127.0.0.1:1\n").is_none(),
            "address alone is not an endpoint"
        );
        assert!(
            parse_endpoint("127.0.0.1:1\n\n").is_none(),
            "an empty token is not a token"
        );
        assert!(
            parse_endpoint("\ntoken\n").is_none(),
            "an empty address is not an address"
        );
    }

    #[test]
    fn tokens_are_128_bit_hex_and_never_a_constant() {
        let a = new_token();
        let b = new_token();
        assert_eq!(a.len(), 32, "128 bits of hex");
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()), "hex only: {a}");
        assert_ne!(a, b, "a per-daemon token must not be a constant");
    }
}
