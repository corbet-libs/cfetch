//! Private, bounded loopback transport for the admitted native package adapter.
//! HTTP never owns native execution: one dedicated thread serializes the core.
//! Parent loss or an uncertain execution deadline terminates this adapter;
//! the native child's parent-death guard leaves any durable intent pending.

use anyhow::{Context as _, ensure};
use http_body_util::{BodyExt as _, Full};
use hyper::body::{Bytes, Incoming};
use hyper::{Request, Response};
use hyper_util::rt::{TokioIo, TokioTimer};
use serde::Deserialize;
use std::convert::Infallible;
use std::future::Future as _;
use std::io::{BufRead as _, Read as _, Write as _};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Semaphore, oneshot, watch};
use tokio::time::{Instant, Sleep};

use crate::local_adapter::{MAX_SCOPES, MAX_STARTUP_BYTES};
use crate::native_serving::{Failure, NativeService, SignedReply};
const MAX_HEADERS: usize = 32;
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_BODY: usize = 8 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(10);
const STOP_GRACE: Duration = Duration::from_secs(2);
const NONCE_HEADER: &str = "x-cfetch-attestation-nonce";
const SIGNATURE_HEADER: &str = "x-cfetch-attestation-signature";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StartupPermit {
    schema_version: u32,
    bearer: String,
    pub(crate) package_manifest_sha256: String,
    pub(crate) ordered_scope_ids: Vec<String>,
}

fn lower_hex(value: &str, bytes: usize) -> bool {
    value.len() == bytes * 2
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl StartupPermit {
    fn parse(raw: &[u8]) -> anyhow::Result<Self> {
        ensure!(
            raw.len() <= MAX_STARTUP_BYTES && raw.last() == Some(&b'\n'),
            "invalid startup frame"
        );
        // A derived struct rejects duplicate fields, unlike a generic JSON map.
        let permit: Self =
            serde_json::from_slice(raw).map_err(|_| anyhow::anyhow!("invalid startup permit"))?;
        ensure!(permit.schema_version == 2, "unsupported startup schema");
        ensure!(lower_hex(&permit.bearer, 32), "invalid startup bearer");
        ensure!(
            lower_hex(&permit.package_manifest_sha256, 32),
            "invalid parent-bound manifest digest"
        );
        ensure!(
            (1..=MAX_SCOPES).contains(&permit.ordered_scope_ids.len()),
            "invalid scope cohort size"
        );
        let mut seen = std::collections::BTreeSet::new();
        for id in &permit.ordered_scope_ids {
            crate::local_inference::validate_scope_id(id).context("invalid scope identity")?;
            ensure!(seen.insert(id), "duplicate scope identity");
        }
        Ok(permit)
    }
}

trait Service: Send + 'static {
    fn scope_ids(&self) -> Vec<String>;
    /// Derived from the pinned governor's lock wait, operation budgets and
    /// supported maximum batch/shapes, not from the HTTP input timeout.
    fn request_budget(&self) -> Duration;
    fn handle(&mut self, body: &[u8], nonce: &[u8; 32]) -> Result<SignedReply, Failure>;
    fn stop(&mut self) -> anyhow::Result<()>;
}

impl Service for NativeService {
    fn scope_ids(&self) -> Vec<String> {
        self.scope_ids()
    }
    fn request_budget(&self) -> Duration {
        self.request_budget()
    }
    fn handle(&mut self, body: &[u8], nonce: &[u8; 32]) -> Result<SignedReply, Failure> {
        self.handle(body, nonce)
    }
    fn stop(&mut self) -> anyhow::Result<()> {
        self.stop()
    }
}

#[derive(Clone)]
struct Stop {
    requested: Arc<AtomicBool>,
    fatal: watch::Sender<Option<&'static str>>,
}
impl Stop {
    fn trip(&self, reason: &'static str) {
        self.requested.store(true, Ordering::Release);
        self.fatal.send_if_modified(|value| {
            if value.is_none() {
                *value = Some(reason);
                true
            } else {
                false
            }
        });
    }
}

/// Dropping an HTTP future does not cancel a vendor call. It instead makes
/// the process owner stop the entire adapter under the bounded exit contract.
struct ObservedExecution {
    stop: Stop,
    armed: bool,
}
impl Drop for ObservedExecution {
    fn drop(&mut self) {
        if self.armed {
            self.stop.trip("native execution observer disappeared");
        }
    }
}

struct Job {
    body: Vec<u8>,
    nonce: [u8; 32],
    reply: oneshot::Sender<Result<SignedReply, Failure>>,
    // Held by the actual owner, not by the cancelable HTTP future.
    _slot: tokio::sync::OwnedSemaphorePermit,
}

struct Shared {
    authorization: String,
    jobs: mpsc::SyncSender<Job>,
    slot: Arc<Semaphore>,
    budget: Duration,
    stop: Stop,
}

type HttpResponse = Response<Full<Bytes>>;

fn response(status: u16, body: Vec<u8>, signature: Option<&str>) -> HttpResponse {
    let mut out = Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("content-length", body.len().to_string())
        .header("connection", "close");
    if let Some(signature) = signature {
        out = out.header(SIGNATURE_HEADER, signature);
    }
    out.body(Full::new(Bytes::from(body)))
        .expect("fixed status and validated headers")
}

fn rejection(status: u16, message: &str) -> HttpResponse {
    response(
        status,
        serde_json::to_vec(&serde_json::json!({"error": message})).expect("string JSON"),
        None,
    )
}

fn single<'a>(headers: &'a hyper::HeaderMap, name: &str) -> Option<&'a str> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?;
    values.next().is_none().then_some(value)
}

fn request_headers<B>(
    request: &Request<B>,
    expected: &str,
) -> Result<(usize, [u8; 32]), (u16, &'static str)> {
    if request.method() != hyper::Method::POST
        || request.uri().path_and_query().map(|p| p.as_str()) != Some("/v1/embeddings")
    {
        return Err((404, "not found"));
    }
    let headers = request.headers();
    if !single(headers, "authorization")
        .is_some_and(|value| crate::serve::token_eq(value, expected))
    {
        return Err((401, "unauthorized"));
    }
    // Never poll a body with Expect: Hyper could emit an interim 100 response,
    // starting the final-write timer before the policy-bounded native work.
    if headers.contains_key("expect") {
        return Err((417, "interim responses are not accepted"));
    }
    if headers.contains_key("transfer-encoding") {
        return Err((400, "transfer encoding is not accepted"));
    }
    let raw_length =
        single(headers, "content-length").ok_or((400, "one Content-Length is required"))?;
    if raw_length.is_empty() || !raw_length.bytes().all(|b| b.is_ascii_digit()) {
        return Err((400, "invalid Content-Length"));
    }
    let length: usize = raw_length
        .parse()
        .map_err(|_| (400, "invalid Content-Length"))?;
    if !(1..=MAX_BODY).contains(&length) {
        return Err((413, "request body exceeds limit"));
    }
    if !single(headers, "content-type").is_some_and(|value| {
        value
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case("application/json")
    }) {
        return Err((400, "Content-Type must be application/json"));
    }
    let nonce = single(headers, NONCE_HEADER)
        .filter(|value| lower_hex(value, 32))
        .ok_or((400, "one lowercase attestation nonce is required"))?;
    let mut decoded = [0; 32];
    for (out, pair) in decoded.iter_mut().zip(nonce.as_bytes().as_chunks::<2>().0) {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        *out = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok((length, decoded))
}

async fn bounded_body<B>(
    mut body: B,
    length: usize,
    budget: Duration,
) -> Result<Vec<u8>, (u16, &'static str)>
where
    B: hyper::body::Body<Data = Bytes> + Unpin,
{
    let read = async {
        let mut bytes = Vec::with_capacity(length);
        while let Some(frame) = body.frame().await {
            let frame = frame.map_err(|_| ())?;
            let data = frame.into_data().map_err(|_| ())?;
            if bytes.len().saturating_add(data.len()) > length {
                return Err(());
            }
            bytes.extend_from_slice(&data);
        }
        if bytes.len() != length {
            return Err(());
        }
        Ok(bytes)
    };
    match tokio::time::timeout(budget, read).await {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(())) => Err((400, "request body does not match Content-Length")),
        Err(_) => Err((408, "request body deadline exceeded")),
    }
}

async fn handle(
    request: Request<Incoming>,
    shared: Arc<Shared>,
) -> Result<HttpResponse, Infallible> {
    let (length, nonce) = match request_headers(&request, &shared.authorization) {
        Ok(value) => value,
        Err((status, message)) => return Ok(rejection(status, message)),
    };
    let bytes = match bounded_body(request.into_body(), length, IO_TIMEOUT).await {
        Ok(bytes) => bytes,
        Err((status, message)) => return Ok(rejection(status, message)),
    };
    if shared.stop.requested.load(Ordering::Acquire) {
        return Ok(rejection(503, "adapter is stopping"));
    }
    let slot = match Arc::clone(&shared.slot).try_acquire_owned() {
        Ok(slot) => slot,
        Err(_) => return Ok(rejection(429, "native executor is busy")),
    };
    let (reply, receiver) = oneshot::channel();
    if shared
        .jobs
        .try_send(Job {
            body: bytes,
            nonce,
            reply,
            _slot: slot,
        })
        .is_err()
    {
        shared.stop.trip("native executor channel failed");
        return Ok(rejection(500, "native executor unavailable"));
    }
    let mut observed = ObservedExecution {
        stop: shared.stop.clone(),
        armed: true,
    };
    let result = tokio::time::timeout(shared.budget, receiver).await;
    observed.armed = false;
    let result = match result {
        Ok(Ok(result)) => result,
        Ok(Err(_)) => {
            shared
                .stop
                .trip("native executor exited without a response");
            return Ok(rejection(500, "native execution stopped"));
        }
        Err(_) => {
            shared.stop.trip("native batch execution deadline exceeded");
            return Ok(rejection(500, "native execution stopped"));
        }
    };
    match result {
        Ok(reply) if reply.body.len() <= MAX_BODY && lower_hex(&reply.signature, 64) => {
            Ok(response(200, reply.body, Some(&reply.signature)))
        }
        Ok(_) => {
            shared.stop.trip("native reply exceeded its wire contract");
            Ok(rejection(500, "invalid native reply"))
        }
        Err(error) => {
            if matches!(error, Failure::HardStop(_)) {
                shared.stop.trip("native core reported a hard stop");
            }
            let (status, body) = error.wire_error();
            let bytes = serde_json::to_vec(&body).expect("error JSON");
            if bytes.len() > MAX_BODY {
                return Ok(rejection(status, "request could not be completed"));
            }
            Ok(response(status, bytes, None))
        }
    }
}

/// Hyper normalizes identical duplicate Content-Length fields. The existing
/// protocol forbids them, so inspect bounded raw headers with Hyper's own
/// locked parser before forwarding their final bytes. No body is parsed here.
fn raw_headers(bytes: &[u8]) -> std::io::Result<bool> {
    let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
    let mut request = httparse::Request::new(&mut headers);
    let status = request
        .parse(bytes)
        .map_err(|_| std::io::Error::other("invalid HTTP headers"))?;
    let httparse::Status::Complete(end) = status else {
        if bytes.len() >= MAX_HEADER_BYTES {
            return Err(std::io::Error::other("HTTP headers exceed limit"));
        }
        return Ok(false);
    };
    if end > MAX_HEADER_BYTES {
        return Err(std::io::Error::other("HTTP headers exceed limit"));
    }
    if request
        .headers
        .iter()
        .filter(|h| h.name.eq_ignore_ascii_case("content-length"))
        .count()
        > 1
    {
        return Err(std::io::Error::other(
            "duplicate Content-Length is not accepted",
        ));
    }
    Ok(true)
}

struct GuardedIo {
    stream: TcpStream,
    headers: Option<Vec<u8>>,
    write_deadline: Option<Pin<Box<Sleep>>>,
}
impl AsyncRead for GuardedIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        output: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        let mut bytes = [0; 8192];
        let limit = bytes.len().min(output.remaining());
        if limit == 0 {
            return Poll::Ready(Ok(()));
        }
        let mut incoming = ReadBuf::new(&mut bytes[..limit]);
        match Pin::new(&mut this.stream).poll_read(cx, &mut incoming) {
            Poll::Ready(Ok(())) => {
                let bytes = incoming.filled();
                if let Some(headers) = &mut this.headers {
                    headers.extend_from_slice(bytes);
                    match raw_headers(headers) {
                        Ok(true) => this.headers = None,
                        Ok(false) => (),
                        Err(error) => return Poll::Ready(Err(error)),
                    }
                }
                output.put_slice(bytes);
                Poll::Ready(Ok(()))
            }
            other => other,
        }
    }
}
impl GuardedIo {
    fn write_ready(&mut self, cx: &mut Context<'_>) -> std::io::Result<()> {
        let deadline = self
            .write_deadline
            .get_or_insert_with(|| Box::pin(tokio::time::sleep(IO_TIMEOUT)));
        if deadline.as_mut().poll(cx).is_ready() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "HTTP response deadline exceeded",
            ));
        }
        Ok(())
    }
}
impl AsyncWrite for GuardedIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        if let Err(error) = this.write_ready(cx) {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut this.stream).poll_write(cx, bytes)
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if this.write_deadline.is_some()
            && let Err(error) = this.write_ready(cx)
        {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut this.stream).poll_flush(cx)
    }
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if this.write_deadline.is_some()
            && let Err(error) = this.write_ready(cx)
        {
            return Poll::Ready(Err(error));
        }
        Pin::new(&mut this.stream).poll_shutdown(cx)
    }
}

async fn serve<S: Service>(
    mut service: S,
    permit: StartupPermit,
    mut parent_eof: watch::Receiver<bool>,
    ready: impl FnOnce(&[u8]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    ensure!(
        service.scope_ids() == permit.ordered_scope_ids,
        "loaded scopes differ from parent permit"
    );
    let budget = service.request_budget();
    ensure!(
        !budget.is_zero() && Instant::now().checked_add(budget).is_some(),
        "invalid native batch budget"
    );
    ensure!(!*parent_eof.borrow(), "parent already closed its lifeline");
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).await?;
    let address = listener.local_addr()?;
    let (fatal, mut fatal_rx) = watch::channel(None);
    let stop = Stop {
        requested: Arc::new(AtomicBool::new(false)),
        fatal,
    };
    let (jobs, receiver) = mpsc::sync_channel::<Job>(1);
    let (done, mut completion) = oneshot::channel();
    let worker_stop = stop.clone();
    // Deliberately not spawn_blocking: dropping a Tokio runtime must never
    // wait indefinitely for a vendor-blocked core thread.
    std::thread::Builder::new()
        .name("native-serving-owner".into())
        .spawn(move || {
            while !worker_stop.requested.load(Ordering::Acquire) {
                match receiver.recv_timeout(Duration::from_millis(25)) {
                    Ok(job) => {
                        if worker_stop.requested.load(Ordering::Acquire) {
                            break;
                        }
                        let result = service.handle(&job.body, &job.nonce);
                        let hard = matches!(&result, Err(Failure::HardStop(_)));
                        let _ = job.reply.send(result);
                        if hard {
                            worker_stop.trip("native core reported a hard stop");
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => (),
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            let _ = done.send(service.stop());
        })?;
    let shared = Arc::new(Shared {
        authorization: format!("Bearer {}", permit.bearer),
        jobs,
        slot: Arc::new(Semaphore::new(1)),
        budget,
        stop: stop.clone(),
    });
    let document = serde_json::to_vec(
        &serde_json::json!({"schema_version":1,"url":format!("http://127.0.0.1:{}/v1",address.port()),"scope_ids":permit.ordered_scope_ids}),
    )?;
    ready(&document)?;
    let connections = Arc::new(Semaphore::new(4));
    let mut tasks = tokio::task::JoinSet::new();
    let mut completed = None;
    let result = loop {
        tokio::select! {
            biased;
            changed = parent_eof.changed() => {
                if changed.is_err() || *parent_eof.borrow() { break Ok(()); }
            }
            _ = fatal_rx.changed() => { break Err(anyhow::anyhow!(fatal_rx.borrow().unwrap_or("native transport stopped"))); }
            result = &mut completion => {
                completed = Some(result);
                break Err(anyhow::anyhow!("native owner exited before transport shutdown"));
            }
            // Drain completed tasks before accepting more sockets. Otherwise
            // a continuously ready listener could grow completed task storage.
            _ = tasks.join_next(), if !tasks.is_empty() => (),
            result = listener.accept() => {
                let (stream, _) = match result { Ok(value) => value, Err(error) => break Err(error.into()) };
                let Ok(connection) = Arc::clone(&connections).try_acquire_owned() else { drop(stream); continue; };
                let shared = Arc::clone(&shared);
                tasks.spawn(async move {
                    let _connection = connection;
                    let io = GuardedIo { stream, headers: Some(Vec::new()), write_deadline: None };
                    let mut builder = hyper::server::conn::http1::Builder::new();
                    builder.keep_alive(false).max_headers(MAX_HEADERS).max_buf_size(MAX_HEADER_BYTES).timer(TokioTimer::new()).header_read_timeout(IO_TIMEOUT);
                    let _ = builder.serve_connection(TokioIo::new(io), hyper::service::service_fn(move |request| handle(request, Arc::clone(&shared)))).await;
                });
            }
        }
    };
    stop.requested.store(true, Ordering::Release);
    drop(listener);
    tasks.abort_all();
    drop(shared);
    let cleanup = match completed {
        Some(result) => result,
        None => tokio::time::timeout(STOP_GRACE, completion)
            .await
            .context("native owner did not stop within grace")?,
    };
    cleanup.context("native owner exited without cleanup proof")??;
    result
}

/// This entry point owns an entire supervised adapter process. It never
/// returns to a caller that could accidentally keep an uncertain core alive.
pub(crate) fn run_stdio() -> ! {
    run_stdio_with(NativeService::from_startup)
}

fn run_stdio_with<S: Service>(load: impl FnOnce(&StartupPermit) -> anyhow::Result<S>) -> ! {
    let result = (|| -> anyhow::Result<()> {
        let (auth_tx, auth_rx) = mpsc::sync_channel(1);
        let (eof_tx, eof_rx) = watch::channel(false);
        std::thread::Builder::new()
            .name("native-parent-lifeline".into())
            .spawn(move || {
                let mut input = std::io::BufReader::new(std::io::stdin());
                let mut raw = Vec::new();
                let parsed = input
                    .by_ref()
                    .take(MAX_STARTUP_BYTES as u64 + 1)
                    .read_until(b'\n', &mut raw)
                    .map_err(anyhow::Error::from)
                    .and_then(|_| StartupPermit::parse(&raw));
                if auth_tx.send(parsed).is_err() {
                    return;
                }
                let mut discard = [0; 4096];
                loop {
                    match input.read(&mut discard) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => (),
                    }
                }
                eof_tx.send_replace(true);
                // Also bounds loss during synchronous installation validation or
                // readiness publication, before the async loop can observe EOF.
                std::thread::sleep(STOP_GRACE);
                std::process::exit(1);
            })?;
        let permit = auth_rx
            .recv_timeout(IO_TIMEOUT)
            .context("startup authorization deadline exceeded")??;
        let service = load(&permit)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(serve(service, permit, eof_rx, |document| {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(document)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
            Ok(())
        }))
    })();
    let code = match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("native adapter stopped: {error:#}");
            1
        }
    };
    // No join of a blocked owner thread. Its owned native child receives
    // PDEATHSIG; an in-flight durable intent is deliberately not reconciled.
    std::process::exit(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn permit_bytes() -> Vec<u8> {
        let mut raw = serde_json::to_vec(&serde_json::json!({"schema_version":2,"bearer":"a".repeat(64),"package_manifest_sha256":"b".repeat(64),"ordered_scope_ids":["cpu"]})).unwrap();
        raw.push(b'\n');
        raw
    }

    fn request() -> Request<()> {
        Request::builder()
            .method("POST")
            .uri("/v1/embeddings")
            .header("authorization", format!("Bearer {}", "a".repeat(64)))
            .header("content-length", "2")
            .header("content-type", "application/json")
            .header(NONCE_HEADER, "b".repeat(64))
            .body(())
            .unwrap()
    }

    fn wire(port: u16, token: char) -> Vec<u8> {
        format!("POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {}\r\nContent-Type: application/json\r\nContent-Length: 2\r\nX-Cfetch-Attestation-Nonce: {}\r\n\r\n{{}}", token.to_string().repeat(64), "b".repeat(64)).into_bytes()
    }

    #[test]
    fn startup_requires_parent_bound_schema_and_unique_canonical_scopes() {
        StartupPermit::parse(&permit_bytes()).unwrap();
        assert!(
            StartupPermit::parse(format!("{{\"bearer\":\"{}\"}}\n", "a".repeat(64)).as_bytes())
                .is_err()
        );
        let good: serde_json::Value = serde_json::from_slice(&permit_bytes()).unwrap();
        for (key, value) in [
            ("schema_version", serde_json::json!(1)),
            ("bearer", serde_json::json!("A".repeat(64))),
            ("package_manifest_sha256", serde_json::json!("bad")),
            ("ordered_scope_ids", serde_json::json!(["cpu", "cpu"])),
            ("ordered_scope_ids", serde_json::json!(["../cpu"])),
            ("ordered_scope_ids", serde_json::json!([])),
            ("unknown", serde_json::json!(true)),
        ] {
            let mut changed = good.clone();
            changed[key] = value;
            let mut raw = serde_json::to_vec(&changed).unwrap();
            raw.push(b'\n');
            assert!(StartupPermit::parse(&raw).is_err());
        }
        let duplicate =
            String::from_utf8(permit_bytes())
                .unwrap()
                .replacen('{', "{\"schema_version\":2,", 1);
        assert!(StartupPermit::parse(duplicate.as_bytes()).is_err());
        assert!(StartupPermit::parse(&vec![b' '; MAX_STARTUP_BYTES + 1]).is_err());
    }

    #[test]
    fn request_auth_nonce_framing_and_exact_route_are_enforced() {
        let expected = format!("Bearer {}", "a".repeat(64));
        assert_eq!(
            request_headers(&request(), &expected).unwrap(),
            (2, [0xbb; 32])
        );
        for header in [
            "authorization",
            "content-length",
            "content-type",
            NONCE_HEADER,
        ] {
            let mut req = request();
            let value = req.headers()[header].clone();
            req.headers_mut().append(
                hyper::header::HeaderName::from_bytes(header.as_bytes()).unwrap(),
                value,
            );
            assert!(request_headers(&req, &expected).is_err());
        }
        let mut req = request();
        req.headers_mut()
            .insert("transfer-encoding", "chunked".parse().unwrap());
        assert!(request_headers(&req, &expected).is_err());
        let mut req = request();
        req.headers_mut()
            .insert("expect", "100-continue".parse().unwrap());
        assert_eq!(request_headers(&req, &expected).unwrap_err().0, 417);
        let mut req = request();
        *req.uri_mut() = "/v1/embeddings?extra=1".parse().unwrap();
        assert_eq!(request_headers(&req, &expected).unwrap_err().0, 404);
        let mut req = request();
        req.headers_mut().insert(
            "content-length",
            (MAX_BODY + 1).to_string().parse().unwrap(),
        );
        assert_eq!(request_headers(&req, &expected).unwrap_err().0, 413);
        let mut req = request();
        req.headers_mut()
            .insert(NONCE_HEADER, "B".repeat(64).parse().unwrap());
        assert!(request_headers(&req, &expected).is_err());
    }

    #[test]
    fn raw_header_guard_rejects_lengths_hyper_would_normalize() {
        assert!(raw_headers(&wire(1, 'a')).unwrap());
        let duplicate = String::from_utf8(wire(1, 'a')).unwrap().replace(
            "Content-Length: 2",
            "Content-Length: 2\r\ncontent-length: 2",
        );
        assert!(raw_headers(duplicate.as_bytes()).is_err());
        let mut oversized = b"POST /v1/embeddings HTTP/1.1\r\nX-Large: ".to_vec();
        oversized.extend(vec![b'x'; MAX_HEADER_BYTES]);
        assert!(raw_headers(&oversized).is_err());
        assert!(!raw_headers(b"POST /v1/embeddings HTTP/1.1\r\n").unwrap());
    }

    #[test]
    fn body_length_and_absolute_read_deadline_are_bounded() {
        struct PendingBody;
        impl hyper::body::Body for PendingBody {
            type Data = Bytes;
            type Error = Infallible;
            fn poll_frame(
                self: Pin<&mut Self>,
                _: &mut Context<'_>,
            ) -> Poll<Option<Result<hyper::body::Frame<Bytes>, Infallible>>> {
                Poll::Pending
            }
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let body = || Full::new(Bytes::from_static(b"{}"));
            assert_eq!(
                bounded_body(body(), 2, Duration::from_secs(1))
                    .await
                    .unwrap(),
                b"{}"
            );
            assert_eq!(
                bounded_body(body(), 1, Duration::from_secs(1))
                    .await
                    .unwrap_err()
                    .0,
                400
            );
            assert_eq!(
                bounded_body(body(), 3, Duration::from_secs(1))
                    .await
                    .unwrap_err()
                    .0,
                400
            );
            let start = std::time::Instant::now();
            assert_eq!(
                bounded_body(PendingBody, 2, Duration::from_millis(20))
                    .await
                    .unwrap_err()
                    .0,
                408
            );
            assert!(start.elapsed() < Duration::from_secs(1));
        });
    }

    struct FixtureService {
        calls: Arc<AtomicUsize>,
        stops: Arc<AtomicUsize>,
        blocked: bool,
        short_budget: bool,
    }
    impl Service for FixtureService {
        fn scope_ids(&self) -> Vec<String> {
            vec!["cpu".into()]
        }
        fn request_budget(&self) -> Duration {
            if self.short_budget {
                Duration::from_millis(100)
            } else {
                Duration::from_secs(30)
            }
        }
        fn handle(&mut self, _: &[u8], _: &[u8; 32]) -> Result<SignedReply, Failure> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.blocked {
                let directory = std::env::var_os("CFETCH_HTTP_FIXTURE_DIRECTORY").unwrap();
                let directory = std::path::PathBuf::from(directory);
                let mut pending = std::fs::File::create(directory.join("intent")).unwrap();
                pending.write_all(b"pending native fixture").unwrap();
                pending.sync_all().unwrap();
                let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "native_http::tests::parent_eof_and_execution_deadline_exit_even_with_blocked_owner", "--nocapture"])
                    .env("CFETCH_HTTP_FIXTURE_ROLE", "native")
                    .env("CFETCH_HTTP_FIXTURE_PARENT", std::process::id().to_string())
                    .stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null())
                    .spawn().unwrap();
                // The adapter cannot return from this simulated vendor wait.
                // Its process-level deadline must bound shutdown anyway.
                let _ = child.wait();
                panic!("native fixture unexpectedly exited before its owner");
            }
            Ok(SignedReply {
                body: br#"{"data":[]}"#.to_vec(),
                signature: "c".repeat(128),
            })
        }
        fn stop(&mut self) -> anyhow::Result<()> {
            self.stops.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    fn fixture(blocked: bool, short_budget: bool) -> FixtureService {
        FixtureService {
            calls: Arc::new(AtomicUsize::new(0)),
            stops: Arc::new(AtomicUsize::new(0)),
            blocked,
            short_budget,
        }
    }

    #[test]
    fn loopback_auth_precedes_core_and_success_keeps_signed_wire_bytes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
            let service = fixture(false, false);
            let calls = Arc::clone(&service.calls);
            let stops = Arc::clone(&service.stops);
            let (eof, eof_rx) = watch::channel(false);
            let (ready, receiver) = oneshot::channel();
            let server = tokio::spawn(serve(
                service,
                StartupPermit::parse(&permit_bytes()).unwrap(),
                eof_rx,
                move |bytes| {
                    let _ = ready.send(bytes.to_vec());
                    Ok(())
                },
            ));
            let document: serde_json::Value =
                serde_json::from_slice(&receiver.await.unwrap()).unwrap();
            assert_eq!(document["schema_version"], 1);
            assert_eq!(document["scope_ids"], serde_json::json!(["cpu"]));
            let port: u16 = document["url"]
                .as_str()
                .unwrap()
                .strip_prefix("http://127.0.0.1:")
                .unwrap()
                .strip_suffix("/v1")
                .unwrap()
                .parse()
                .unwrap();
            for (token, status, expect) in [('b', 401, false), ('a', 417, true), ('a', 200, false)]
            {
                let mut stream = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
                    .await
                    .unwrap();
                let mut request = wire(port, token);
                if expect {
                    request = String::from_utf8(request)
                        .unwrap()
                        .replace(
                            "Content-Length: 2",
                            "Content-Length: 2\r\nExpect: 100-continue",
                        )
                        .into_bytes();
                    // A conforming client waits for 100 before sending its body.
                    // The adapter must reject immediately without polling it.
                    request.truncate(request.len() - 2);
                }
                stream.write_all(&request).await.unwrap();
                let mut bytes = Vec::new();
                tokio::time::timeout(Duration::from_secs(2), stream.read_to_end(&mut bytes))
                    .await
                    .unwrap()
                    .unwrap();
                let mut headers = [httparse::EMPTY_HEADER; 32];
                let mut reply = httparse::Response::new(&mut headers);
                let end = reply.parse(&bytes).unwrap().unwrap();
                assert_eq!(reply.code, Some(status));
                assert!(
                    !bytes
                        .windows(b"100 Continue".len())
                        .any(|part| part == b"100 Continue")
                );
                if status != 200 {
                    assert_eq!(calls.load(Ordering::Relaxed), 0);
                } else {
                    assert_eq!(&bytes[end..], br#"{"data":[]}"#);
                    assert!(
                        reply
                            .headers
                            .iter()
                            .any(|h| h.name.eq_ignore_ascii_case(SIGNATURE_HEADER)
                                && h.value == "c".repeat(128).as_bytes())
                    );
                }
            }
            eof.send_replace(true);
            tokio::time::timeout(Duration::from_secs(3), server)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            assert_eq!(stops.load(Ordering::Relaxed), 1);
        });
    }

    #[test]
    fn concurrent_request_is_rejected_before_entering_core() {
        struct BusyService {
            calls: Arc<AtomicUsize>,
            started: Option<oneshot::Sender<()>>,
            release: mpsc::Receiver<()>,
        }
        impl Service for BusyService {
            fn scope_ids(&self) -> Vec<String> {
                vec!["cpu".into()]
            }
            fn request_budget(&self) -> Duration {
                Duration::from_secs(10)
            }
            fn handle(&mut self, _: &[u8], _: &[u8; 32]) -> Result<SignedReply, Failure> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.started.take().unwrap().send(()).unwrap();
                self.release.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(SignedReply {
                    body: br#"{"data":[]}"#.to_vec(),
                    signature: "c".repeat(128),
                })
            }
            fn stop(&mut self) -> anyhow::Result<()> {
                Ok(())
            }
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
            let calls = Arc::new(AtomicUsize::new(0));
            let (started, entered) = oneshot::channel();
            let (release, wait) = mpsc::channel();
            let service = BusyService {
                calls: Arc::clone(&calls),
                started: Some(started),
                release: wait,
            };
            let (eof, eof_rx) = watch::channel(false);
            let (ready, receiver) = oneshot::channel();
            let server = tokio::spawn(serve(
                service,
                StartupPermit::parse(&permit_bytes()).unwrap(),
                eof_rx,
                move |bytes| {
                    let _ = ready.send(bytes.to_vec());
                    Ok(())
                },
            ));
            let document: serde_json::Value =
                serde_json::from_slice(&receiver.await.unwrap()).unwrap();
            let port: u16 = document["url"]
                .as_str()
                .unwrap()
                .strip_prefix("http://127.0.0.1:")
                .unwrap()
                .strip_suffix("/v1")
                .unwrap()
                .parse()
                .unwrap();
            let mut first = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
                .await
                .unwrap();
            first.write_all(&wire(port, 'a')).await.unwrap();
            tokio::time::timeout(Duration::from_secs(2), entered)
                .await
                .unwrap()
                .unwrap();
            let mut second = TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port))
                .await
                .unwrap();
            second.write_all(&wire(port, 'a')).await.unwrap();
            let mut rejected = Vec::new();
            tokio::time::timeout(Duration::from_secs(2), second.read_to_end(&mut rejected))
                .await
                .unwrap()
                .unwrap();
            assert!(rejected.starts_with(b"HTTP/1.1 429 "));
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            release.send(()).unwrap();
            let mut accepted = Vec::new();
            tokio::time::timeout(Duration::from_secs(2), first.read_to_end(&mut accepted))
                .await
                .unwrap()
                .unwrap();
            assert!(accepted.starts_with(b"HTTP/1.1 200 "));
            assert_eq!(calls.load(Ordering::Relaxed), 1);
            eof.send_replace(true);
            tokio::time::timeout(Duration::from_secs(3), server)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
        });
    }

    #[test]
    fn parent_eof_and_execution_deadline_exit_even_with_blocked_owner() {
        match std::env::var("CFETCH_HTTP_FIXTURE_ROLE").as_deref() {
            Ok("native") => {
                let expected: u32 = std::env::var("CFETCH_HTTP_FIXTURE_PARENT")
                    .unwrap()
                    .parse()
                    .unwrap();
                rustix::process::set_parent_process_death_signal(Some(
                    rustix::process::Signal::KILL,
                ))
                .unwrap();
                assert_eq!(
                    rustix::process::getppid().unwrap().as_raw_nonzero().get() as u32,
                    expected
                );
                let directory = std::path::PathBuf::from(
                    std::env::var_os("CFETCH_HTTP_FIXTURE_DIRECTORY").unwrap(),
                );
                std::fs::write(directory.join("native-pid"), std::process::id().to_string())
                    .unwrap();
                loop {
                    std::thread::park_timeout(Duration::from_secs(60));
                }
            }
            Ok("adapter") => run_stdio_with(|_| {
                Ok(fixture(
                    true,
                    std::env::var("CFETCH_HTTP_FIXTURE_DEADLINE").unwrap() == "true",
                ))
            }),
            _ => (),
        }
        for deadline in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let mut adapter = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "native_http::tests::parent_eof_and_execution_deadline_exit_even_with_blocked_owner", "--nocapture"])
                .env("CFETCH_HTTP_FIXTURE_ROLE", "adapter").env("CFETCH_HTTP_FIXTURE_DIRECTORY", directory.path())
                .env("CFETCH_HTTP_FIXTURE_DEADLINE", deadline.to_string())
                .stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).spawn().unwrap();
            let mut input = adapter.stdin.take().unwrap();
            input.write_all(&permit_bytes()).unwrap();
            input.flush().unwrap();
            let stdout = adapter.stdout.take().unwrap();
            let (send, receive) = mpsc::sync_channel(1);
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(stdout)
                    .lines()
                    .map_while(Result::ok)
                {
                    // The test harness may prefix its test name; the actual
                    // product process writes only this readiness document.
                    if let Some(start) = line.find("{\"schema_version\":1") {
                        let _ = send.send(line[start..].to_string());
                        return;
                    }
                }
            });
            let line = match receive.recv_timeout(Duration::from_secs(3)) {
                Ok(line) => line,
                Err(error) => {
                    adapter.kill().unwrap();
                    adapter.wait().unwrap();
                    panic!("no fixture readiness: {error}");
                }
            };
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            let port: u16 = value["url"]
                .as_str()
                .unwrap()
                .strip_prefix("http://127.0.0.1:")
                .unwrap()
                .strip_suffix("/v1")
                .unwrap()
                .parse()
                .unwrap();
            let mut stream =
                std::net::TcpStream::connect((std::net::Ipv4Addr::LOCALHOST, port)).unwrap();
            stream.write_all(&wire(port, 'a')).unwrap();
            let bound = std::time::Instant::now() + Duration::from_secs(2);
            while !directory.path().join("native-pid").exists() && std::time::Instant::now() < bound
            {
                std::thread::sleep(Duration::from_millis(10));
            }
            if !deadline {
                drop(input);
            }
            let bound = std::time::Instant::now() + Duration::from_secs(5);
            while adapter.try_wait().unwrap().is_none() && std::time::Instant::now() < bound {
                std::thread::sleep(Duration::from_millis(10));
            }
            let exited = adapter.try_wait().unwrap().is_some();
            if !exited {
                adapter.kill().unwrap();
                adapter.wait().unwrap();
            }
            assert!(exited, "blocked owner must not hang adapter exit");
            assert_eq!(
                std::fs::read(directory.path().join("intent")).unwrap(),
                b"pending native fixture"
            );
            let pid = std::fs::read_to_string(directory.path().join("native-pid")).unwrap();
            let path = format!("/proc/{}/stat", pid.trim());
            let bound = std::time::Instant::now() + Duration::from_secs(2);
            let dead = loop {
                match std::fs::read_to_string(&path) {
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => break true,
                    Ok(row)
                        if row
                            .rsplit_once(')')
                            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z')) =>
                    {
                        break true;
                    }
                    _ if std::time::Instant::now() >= bound => break false,
                    _ => std::thread::sleep(Duration::from_millis(10)),
                }
            };
            assert!(dead, "owned native fixture survived adapter death");
        }
    }
}
