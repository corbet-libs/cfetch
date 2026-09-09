//! Candidate-only worker. Caller must enforce the owned-process deadline,
//! governor and duty cycle around this whole process, including initialization.
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use ort::ep::{ArbitrarilyConfigurableExecutionProvider, ExecutionProvider, OpenVINO};
use std::{collections::BTreeMap, error::Error, ffi::CStr, fs, io::{Read, Write}, path::{Path, PathBuf}, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn save_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    Ok(())
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut options = BTreeMap::new();
    while let Some(key) = args.next() {
        if key == "--help" {
            println!("--ort LIBRARY [--run-model AllMiniLML6V2|EmbeddingGemma300M --device CPU|NPU --cache DIR --output DIR [--cpu-fallback diagnostic] [--cpu-input JSON]]");
            return Ok(());
        }
        if !["--ort", "--run-model", "--device", "--cache", "--output", "--cpu-fallback", "--cpu-input"].contains(&key.as_str()) {
            return Err(format!("unknown argument: {key}").into());
        }
        let value = args.next().ok_or("argument needs a value")?;
        if options.insert(key, value).is_some() {
            return Err("duplicate argument".into());
        }
    }
    // Reject extending the physical canary before loading any native runtime.
    let cpu_input = if let Some(path) = options.get("--cpu-input") {
        if options.get("--device").map(String::as_str) != Some("CPU")
            || options.get("--run-model").map(String::as_str) != Some("EmbeddingGemma300M") {
            return Err("--cpu-input requires CPU and EmbeddingGemma300M".into());
        }
        if !fs::metadata(path)?.is_file() {
            return Err("CPU fixture must be a regular file".into());
        }
        let mut bytes = Vec::new();
        fs::File::open(path)?.take(131073).read_to_end(&mut bytes)?;
        if bytes.len() > 131072 {
            return Err("CPU fixture exceeds 128 KiB".into());
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        if value["schema_version"] != 1 || value["text"].as_str().is_none()
            || value["id"].as_str().is_none()
            || value["tokens"].as_u64().is_none_or(|n| n == 0 || n > 2048)
            || value["token_ids_sha256"].as_str().is_none_or(|hash| hash.len() != 64
                || !hash.bytes().all(|c| c.is_ascii_hexdigit())) {
            return Err("CPU fixture requires schema 1, id, text, 1..2048 tokens and token_ids_sha256".into());
        }
        Some(value)
    } else { None };
    let library = fs::canonicalize(options.get("--ort").ok_or("--ort is required")?)?;
    if ort::MINOR_VERSION != 24 {
        return Err("this worker requires exactly ORT API 24; check Cargo feature unification".into());
    }
    if !ort::init_from(&library)?.commit() {
        return Err("ORT was already initialized; explicit runtime selection did not take effect".into());
    }
    // Read the version from the same exact library selected above, not a filename.
    let version_handle = unsafe { libloading::Library::new(&library)? };
    let base = unsafe {
        version_handle.get::<unsafe extern "C" fn() -> *const ort::sys::OrtApiBase>(b"OrtGetApiBase\0")?()
    };
    if base.is_null() {
        return Err("ORT returned a null API base".into());
    }
    let version_ptr = unsafe { ((*base).GetVersionString)() };
    if version_ptr.is_null() {
        return Err("ORT returned a null version string".into());
    }
    let version = unsafe { CStr::from_ptr(version_ptr) }.to_str()?;
    let available = OpenVINO::default().is_available()?;
    println!("ort_library={library:?}\nort_version={version}\nort_api={}\nort_build={}\nopenvino_available={available}", ort::MINOR_VERSION, ort::info());
    if !available {
        return Err("selected ORT library does not provide OpenVINOExecutionProvider".into());
    }
    let Some(chosen) = options.get("--run-model") else {
        if options.len() != 1 {
            return Err("model arguments require explicit --run-model".into());
        }
        println!("mode=loader-only\nmodel=none\nmodel_executed=false");
        return Ok(());
    };
    let (model_name, dimensions, text) = match chosen.as_str() {
        "AllMiniLML6V2" => (EmbeddingModel::AllMiniLML6V2, 384, "Portable local semantic search."),
        "EmbeddingGemma300M" => (EmbeddingModel::EmbeddingGemma300M, 768, "task: search result | query: portable local semantic search"),
        _ => return Err("unsupported built-in model selection".into()),
    };
    let text = cpu_input.as_ref().and_then(|input| input["text"].as_str()).unwrap_or(text);
    let max_length = if cpu_input.is_some() { 2048 } else { 32 };
    let device = options.get("--device").ok_or("model mode requires --device")?;
    if device != "CPU" && device != "NPU" {
        return Err("device must be exactly CPU or NPU".into());
    }
    let diagnostic_fallback = match options.get("--cpu-fallback").map(String::as_str) {
        None => false,
        Some("diagnostic") if device == "CPU" => true,
        _ => return Err("--cpu-fallback diagnostic is restricted to CPU placement investigation".into()),
    };
    let cache = PathBuf::from(options.get("--cache").ok_or("model mode requires --cache")?);
    let output = PathBuf::from(options.get("--output").ok_or("model mode requires fresh --output")?);
    fs::create_dir(&output)?;
    let output = fs::canonicalize(output)?;
    let mut ep = OpenVINO::default().with_device_type(device).with_num_threads(1).with_num_streams(1);
    if device == "NPU" {
        // Do not request WORKLOAD_TYPE until its accepted value is verified on
        // the selected ORT/OpenVINO bundle. Unsupported properties must fail.
        ep = ep.with_arbitrary_config("load_config", r#"{"NPU":{"NPU_TILES":"1","NPU_TURBO":"NO","PERF_COUNT":"YES","LOG_LEVEL":"LOG_DEBUG"}}"#);
    }
    let options = TextInitOptions::new(model_name)
        .with_execution_providers(vec![ep.build().error_on_failure()])
        .with_intra_threads(1).with_max_length(max_length).with_cache_dir(cache);
    println!("mode=one-model-call\nmodel={chosen}\nrequested_device={device}\ncpu_fallback_disabled={}", !diagnostic_fallback);
    let load_started = Instant::now();
    let mut model = TextEmbedding::try_new_with_session_builder(options, |builder| {
        let builder = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)?
            .with_inter_threads(1)?
            .with_profiling(output.join("ort-profile"))?;
        if diagnostic_fallback { Ok(builder) } else { builder.with_disable_cpu_fallback() }
    })?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    // Refuse truncation or a changed token count. The supervisor compares the
    // emitted token IDs with the retained canonical SHA256 before success.
    let mut tokenizer = model.tokenizer.clone();
    tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
    let encoding = tokenizer.encode(text, true).map_err(|e| e.to_string())?;
    let tokens = encoding.len();
    if tokens > max_length {
        return Err("probe input exceeds its untruncated token limit".into());
    }
    if let Some(input) = &cpu_input {
        if input["tokens"].as_u64() != Some(tokens as u64) {
            return Err("probe token count differs from the retained fixture".into());
        }
    }
    let embed_started = Instant::now();
    let embedded = model.embed([text], Some(1)); // Exactly one model call; no retries.
    let embed_seconds = embed_started.elapsed().as_secs_f64();
    if let Err(error) = &embedded {
        eprintln!("embedding_error={error}");
    }
    let profile = fs::canonicalize(model.end_profiling()?)?;
    if profile.parent() != Some(output.as_path()) {
        return Err("ORT profile escaped the fresh output directory".into());
    }
    println!("profile={profile:?}");
    let events: serde_json::Value = serde_json::from_slice(&fs::read(&profile)?)?;
    fs::File::open(&profile)?.sync_all()?;
    let events = events.as_array().ok_or("ORT profile must be an event array")?;
    let providers: Vec<&str> = events
        .iter().filter_map(|event| event.get("args")?.get("provider")?.as_str()).collect();
    let mut placement = BTreeMap::<String, BTreeMap<String, usize>>::new();
    for event in events {
        if let Some(provider) = event.get("args").and_then(|args| args.get("provider")).and_then(|v| v.as_str()) {
            let op = event["args"]["op_name"].as_str().ok_or("profile provider event lacks op_name")?;
            *placement.entry(provider.to_string()).or_default().entry(op.to_string()).or_default() += 1;
        }
    }
    save_json(&output.join("placement.json"), &serde_json::to_value(&placement)?)?;
    if !providers.contains(&"OpenVINOExecutionProvider") || providers.iter().any(|provider| {
        *provider != "OpenVINOExecutionProvider" && !(diagnostic_fallback && *provider == "CPUExecutionProvider")
    }) {
        return Err("profile does not show exclusive OpenVINO execution".into());
    }
    let vectors = embedded?;
    if vectors.len() != 1 || vectors[0].len() != dimensions || vectors[0].iter().any(|v| !v.is_finite()) {
        return Err("unexpected vector count, dimensions, or nonfinite values".into());
    }
    if !vectors[0].iter().any(|v| *v != 0.0) {
        return Err("zero embedding refused".into());
    }
    let measured = serde_json::json!({
        "model": chosen, "requested_device": device, "cpu_fallback_disabled": !diagnostic_fallback,
        "tokens": tokens, "token_ids": encoding.get_ids(), "text": text, "vector": vectors[0],
        "input_id": cpu_input.as_ref().map(|input| &input["id"]),
        "model_initialization_seconds": load_seconds, "embedding_seconds": embed_seconds,
        "calls": 1, "admission": "not-evaluated"
    });
    save_json(&output.join("embedding.json"), &measured)?;
    fs::File::open(&output)?.sync_all()?;
    println!("tokens={tokens}\nvector_length={}\nfinite=true\nplacement={}\nadmission=not-evaluated", vectors[0].len(), serde_json::to_string(&placement)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("foundation probe failed: {error}");
        std::process::exit(1);
    }
}
