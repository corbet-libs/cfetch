//! MCP server: the one-protocol answer to supporting every MCP-capable agent.
//!
//! cfetch owns the tool behavior. Recall/find and maintenance packets are
//! read-only; the sole write surface can only place a typed proposal in
//! ring-5 quarantine. Applying trusted-memory changes remains CLI-only. The
//! official Rust MCP SDK owns JSON-RPC framing, lifecycle, version
//! negotiation, capability discovery, schema validation, and stdio concurrency.

use anyhow::Context as _;
use rmcp::{
    ErrorData as McpError, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
        JsonObject, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
        ToolAnnotations,
    },
    service::{RequestContext, RoleServer},
};
use serde_json::{Value, json};

use crate::{answer, code, config::Config, graph, index, maintenance, paths};

/// The tools we serve; a tools/call naming anything else is the caller's
/// protocol error (-32602), not a tool-execution failure.
const TOOL_NAMES: &[&str] = &[
    "cfetch_graph",
    "cfetch_graph_path",
    "cfetch_recall",
    "cfetch_expand",
    "cfetch_find",
    "cfetch_code_path",
    "cfetch_code_impact",
    "cfetch_code_context",
    "cfetch_code_symbol",
    "cfetch_runtime_status",
    "cfetch_maintenance_packet",
    "cfetch_maintenance_show",
    "cfetch_maintenance_propose",
    "cfetch_maintenance_review",
];

fn object_schema(value: Value) -> JsonObject {
    value
        .as_object()
        .cloned()
        .expect("tool schema is an object")
}

fn tool_defs() -> Vec<Tool> {
    let read_only = || {
        ToolAnnotations::new()
            .read_only(true)
            .destructive(false)
            .idempotent(true)
            .open_world(false)
    };
    let quarantine_write = || {
        ToolAnnotations::new()
            .read_only(false)
            .destructive(false)
            .idempotent(true)
            .open_world(false)
    };
    vec![
        Tool::new(
            "cfetch_graph",
            "Explore the current Obsidian link graph across available repositories. The answer is bounded by a node limit of 200 and an edge limit of four times the node limit. Missing and ambiguous notes are never invented.",
            object_schema(json!({"type":"object", "properties": {
                "focus":{"type":"string"}, "limit":{"type":"integer", "minimum":1, "maximum":200, "default":40}
            }})),
        ).with_annotations(read_only()),
        Tool::new(
            "cfetch_graph_path",
            "Find a shortest connection between two notes through explicit links and backlinks. The answer is bounded by a depth limit of 32 hops. Returns original edge directions and ambiguous or missing endpoints.",
            object_schema(json!({"type":"object", "properties": {
                "from":{"type":"string"}, "to":{"type":"string"},
                "depth":{"type":"integer", "minimum":1, "maximum":32, "default":6}
            }, "required":["from","to"]})),
        ).with_annotations(read_only()),
        Tool::new(
            "cfetch_recall",
            "Search the operator's knowledge brain (privilege rings 0-4, lexical or local hybrid search). Returns ring-prefixed citations (r<ring>-<hash>), file:line locations and snippets. Lower ring = higher trust; a ring-0/1 hit overrides contradicting outer-ring content. The answer is capped by a token budget, not only by limit: whatever the cap drops is named at the end, never omitted silently.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search terms (word-prefix matched)"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 8},
                    "mode": {"type": "string", "enum": ["lexical", "semantic", "hybrid"], "description": "Defaults to hybrid when local embeddings are enabled, otherwise lexical."}
                },
                "required": ["query"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_expand",
            "Expand a citation id from cfetch_recall to the full memory block. Long blocks are clipped to a token budget and the answer then names the file:line range holding the rest — read that range directly instead of expanding again.",
            object_schema(json!({
                "type": "object",
                "properties": {"cite": {"type": "string"}},
                "required": ["cite"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_find",
            "Locate a symbol or file in the indexed code roots, with exact line ranges — read one function instead of a whole file. Case- and separator-insensitive. Each hit carries the estimated cost of reading it; the answer itself is capped by a token budget and says what the cap dropped.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_code_path",
            "Explain one deterministic shortest path through the indexed source import graph. Every hop names the source file, typed `imports` relation, target file, and extraction evidence class. The traversal is read-only and bounded by depth.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "importing file path or unambiguous suffix"},
                    "to": {"type": "string", "description": "imported file path or unambiguous suffix"},
                    "depth": {"type": "integer", "default": graph::DEFAULT_PATH_DEPTH, "minimum": 1, "maximum": graph::MAX_DEPENDENCY_DEPTH}
                },
                "required": ["from", "to"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_code_impact",
            "Show the bounded blast radius of one indexed source file by walking import edges backwards. Every affected file names its depth and next explanatory hop toward the target; ambiguous suffixes fail instead of guessing.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "file path or unambiguous suffix"},
                    "depth": {"type": "integer", "default": graph::DEFAULT_IMPACT_DEPTH, "minimum": 1, "maximum": graph::MAX_DEPENDENCY_DEPTH},
                    "limit": {"type": "integer", "default": graph::DEFAULT_IMPACT_LIMIT, "minimum": 1, "maximum": graph::MAX_IMPACT_LIMIT}
                },
                "required": ["target"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_code_context",
            "Show bounded incoming and outgoing import context around one indexed source file. Every related file carries one deterministic shortest explanation edge with its real direction, typed relation, and extraction evidence; omitted files are counted.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "file path or unambiguous suffix"},
                    "depth": {"type": "integer", "default": graph::DEFAULT_CONTEXT_DEPTH, "minimum": 1, "maximum": graph::MAX_DEPENDENCY_DEPTH},
                    "limit": {"type": "integer", "default": graph::DEFAULT_CONTEXT_LIMIT, "minimum": 1, "maximum": graph::MAX_CONTEXT_LIMIT}
                },
                "required": ["target"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_code_symbol",
            "Show parser-proven calls and type references around an exact indexed symbol. A relationship resolves only through an explicit import binding to exactly one file-level definition; ambiguity produces no edge. Every edge carries an exact source range.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "exact symbol name"},
                    "limit": {"type": "integer", "default": graph::DEFAULT_SYMBOL_LIMIT, "minimum": 1, "maximum": graph::MAX_SYMBOL_LIMIT}
                },
                "required": ["query"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_runtime_status",
            "Read cfetch's cached RuntimeStatusV1: memory routing and freshness, retrieval coverage, configured/selected/last-used inference, maintenance counts, and stable failure actions. Call when runtime health affects the task; do not poll. This is read-only, performs no network request or inference, and is bounded to 2 KiB.",
            object_schema(json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_maintenance_packet",
            "Build a bounded, content-addressed evidence packet for one ring-5 staging candidate. It includes matching raw events, current candidate bytes, relevant cited statements, a target snapshot when available, and the exact proposal contract. This is lexical and deterministic: it does not call a model or spend an inference token budget.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string", "description": "id from `cfetch staging list`"}
                },
                "required": ["candidate_id"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_maintenance_show",
            "Read one quarantined maintenance proposal and its immutable semantic review, if present. Use this in a separate review pass to check evidence coverage, factual faithfulness, preservation, authority, target choice, and contradictions. The response is bounded by the proposal's 2 MiB content limit, not a model inference token budget.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"}
                },
                "required": ["proposal_id"]
            })),
        )
        .with_annotations(read_only()),
        Tool::new(
            "cfetch_maintenance_propose",
            "Submit one typed maintenance proposal into ring-5 quarantine for debugging or intervention. This cannot edit recalled or injected memory; it records an idempotent proposal on the same deterministic transaction substrate used by autonomous maintenance. Applying and finalizing remain unavailable over MCP.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "candidate_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "transition": {"type": "string", "enum": ["add", "fold", "supersede", "revalidate", "dismiss", "noop"]},
                    "target": {"type": ["string", "null"], "description": "brain-relative Markdown path for content transitions"},
                    "after": {"type": ["string", "null"], "description": "complete proposed Markdown bytes for content transitions"},
                    "authority": {"type": "string", "enum": ["authorized", "attested", "unendorsed"]},
                    "valid_until": {"type": ["integer", "null"], "description": "optional Unix timestamp"},
                    "rationale": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "related_citations": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["candidate_ids", "transition", "target", "after", "authority", "rationale", "evidence", "related_citations"]
            })),
        )
        .with_annotations(quarantine_write()),
        Tool::new(
            "cfetch_maintenance_review",
            "Record the first immutable semantic review of a quarantined proposal. A pass must explicitly assess evidence coverage, factual faithfulness, preservation, authority fit, target fit, and contradictions. This writes only ring-5 review metadata; it cannot apply or finalize memory and returns only a short receipt with no inference token budget.",
            object_schema(json!({
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["pass", "fail"]},
                    "method": {"type": "string", "enum": ["independent_agent", "human"]},
                    "evidence_coverage": {"type": "boolean"},
                    "factual_faithfulness": {"type": "boolean"},
                    "preservation": {"type": "boolean"},
                    "authority_fit": {"type": "boolean"},
                    "target_fit": {"type": "boolean"},
                    "contradiction_checked": {"type": "boolean"},
                    "notes": {"type": "string"}
                },
                "required": ["proposal_id", "verdict", "method", "evidence_coverage", "factual_faithfulness", "preservation", "authority_fit", "target_fit", "contradiction_checked", "notes"]
            })),
        )
        .with_annotations(quarantine_write()),
    ]
}

fn local_recall_entry(h: &index::Hit) -> String {
    answer::hit_entry(
        &h.cite,
        &h.path,
        h.ring,
        h.start_line,
        h.end_line,
        &h.snippet,
        &h.mirrors,
    )
}

fn local_expand_entry(b: &index::Block) -> answer::BlockIn {
    answer::BlockIn {
        cite: b.cite.clone(),
        path: b.path.clone(),
        ring: b.ring,
        start_line: b.start_line,
        end_line: b.end_line,
        text: b.text.clone(),
    }
}

fn run_tool(name: &str, args: &Value) -> anyhow::Result<String> {
    if name == "cfetch_runtime_status" {
        return crate::runtime_status::mcp_json();
    }
    let cfg = Config::load()?;
    let limit = args.get("limit").and_then(Value::as_u64).unwrap_or(0) as usize;
    match name {
        "cfetch_graph" | "cfetch_graph_path" => {
            let conn =
                index::ensure_fresh(&paths::state_dir(), &cfg.brain_root, None, &cfg.rings())?;
            if name == "cfetch_graph" {
                let focus = args.get("focus").and_then(Value::as_str);
                let graph = crate::knowledge_graph::build(
                    &conn,
                    focus,
                    if limit == 0 { 40 } else { limit },
                )?;
                return Ok(serde_json::to_string_pretty(&graph)?);
            }
            let from = args
                .get("from")
                .and_then(Value::as_str)
                .context("from must name a note")?;
            let to = args
                .get("to")
                .and_then(Value::as_str)
                .context("to must name a note")?;
            let depth = args.get("depth").and_then(Value::as_u64).unwrap_or(6);
            let depth =
                usize::try_from(depth).context("graph depth exceeds this platform's bounds")?;
            return Ok(serde_json::to_string_pretty(
                &crate::knowledge_graph::trace(&conn, from, to, depth)?,
            )?);
        }
        "cfetch_maintenance_packet" => {
            let candidate_id = args
                .get("candidate_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let packet = maintenance::packet(&cfg, candidate_id)?;
            return Ok(serde_json::to_string_pretty(&packet)?);
        }
        "cfetch_maintenance_show" => {
            let proposal_id = args
                .get("proposal_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let (state, proposal) = maintenance::get(&cfg, proposal_id)?;
            let review = maintenance::get_review(&cfg, proposal_id)?;
            return Ok(serde_json::to_string_pretty(&json!({
                "state": state,
                "proposal": proposal,
                "review": review,
            }))?);
        }
        "cfetch_maintenance_propose" => {
            let input: maintenance::ProposalInput = serde_json::from_value(args.clone())
                .map_err(|error| anyhow::anyhow!("invalid maintenance proposal: {error}"))?;
            let submitted = maintenance::submit(&cfg, input)?;
            let _ = crate::runtime_status::refresh_static();
            return Ok(format!(
                "{} {} in ring-5 quarantine; inspect with `cfetch maintain verify {}`. This MCP tool cannot apply or finalize it.",
                if submitted.created {
                    "submitted"
                } else {
                    "already recorded"
                },
                submitted.proposal.id,
                submitted.proposal.id,
            ));
        }
        "cfetch_maintenance_review" => {
            let proposal_id = args
                .get("proposal_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let mut review_args = args.clone();
            if let Value::Object(object) = &mut review_args {
                object.remove("proposal_id");
            }
            let input: maintenance::ReviewInput = serde_json::from_value(review_args)
                .map_err(|error| anyhow::anyhow!("invalid maintenance review: {error}"))?;
            let (review, created) = maintenance::submit_review(&cfg, proposal_id, input)?;
            return Ok(format!(
                "{} {} for {} with verdict {:?}; deterministic verification and application remain outside this MCP tool.",
                if created {
                    "recorded"
                } else {
                    "already recorded"
                },
                review.id,
                review.proposal_id,
                review.verdict,
            ));
        }
        _ => {}
    }

    match name {
        "cfetch_recall" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            let conn =
                index::ensure_fresh(&paths::state_dir(), &cfg.brain_root, None, &cfg.rings())?;
            let mode =
                args.get("mode")
                    .and_then(Value::as_str)
                    .unwrap_or(if cfg.embeddings.enabled {
                        "hybrid"
                    } else {
                        "lexical"
                    });
            anyhow::ensure!(
                ["lexical", "semantic", "hybrid"].contains(&mode),
                "unknown recall mode"
            );
            let ranked = crate::pipeline::ranked(
                &cfg,
                &conn,
                query,
                if limit == 0 { 8 } else { limit.min(100) },
                mode == "semantic",
                mode == "hybrid",
                None,
            )?;
            let mut response = if ranked.hits.is_empty() {
                format!("no hits for {query:?}")
            } else {
                answer::listing(
                    ranked.hits.iter().map(local_recall_entry).collect(),
                    answer::RECALL_BUDGET_TOKENS,
                    answer::MCP_RECOVERY,
                )
            };
            if let Some(note) = ranked.note {
                response.push_str(&format!("\n{note}"));
            }
            Ok(response)
        }
        "cfetch_expand" => {
            let cite = args.get("cite").and_then(Value::as_str).unwrap_or("");
            let conn =
                index::ensure_fresh(&paths::state_dir(), &cfg.brain_root, None, &cfg.rings())?;
            let blocks = index::expand(&conn, cite)?;
            if blocks.is_empty() {
                return Ok(format!("no block with citation {cite}"));
            }
            Ok(answer::blocks(
                &blocks.iter().map(local_expand_entry).collect::<Vec<_>>(),
                answer::RECALL_BUDGET_TOKENS,
            ))
        }
        "cfetch_find" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            // Snapshot only — an implicit code scan (minutes on NFS) must
            // never ride on an interactive tool call.
            let conn = index::open(&paths::state_dir())?;
            let hits = code::find(&conn, query, if limit == 0 { 10 } else { limit })?;
            if hits.is_empty() {
                return Ok(format!("no hits for \"{query}\""));
            }
            Ok(answer::listing(
                hits.iter()
                    .map(|h| {
                        answer::find_entry(
                            &h.path,
                            h.name.as_deref(),
                            h.kind.as_deref(),
                            h.start_line,
                            h.end_line,
                            h.token_estimate,
                        )
                    })
                    .collect(),
                answer::FIND_BUDGET_TOKENS,
                answer::MCP_RECOVERY,
            ))
        }
        "cfetch_code_path" => {
            let from = args.get("from").and_then(Value::as_str).unwrap_or("");
            let to = args.get("to").and_then(Value::as_str).unwrap_or("");
            let depth = args
                .get("depth")
                .and_then(Value::as_u64)
                .unwrap_or(graph::DEFAULT_PATH_DEPTH as u64) as usize;
            let conn = index::open(&paths::state_dir())?;
            let path = graph::dependency_path(&conn, &cfg.effective_code_roots(), from, to, depth)?;
            Ok(graph::render_dependency_path(&path))
        }
        "cfetch_code_impact" => {
            let target = args.get("target").and_then(Value::as_str).unwrap_or("");
            let depth = args
                .get("depth")
                .and_then(Value::as_u64)
                .unwrap_or(graph::DEFAULT_IMPACT_DEPTH as u64) as usize;
            let limit = if limit == 0 {
                graph::DEFAULT_IMPACT_LIMIT
            } else {
                limit
            };
            let conn = index::open(&paths::state_dir())?;
            let impact =
                graph::dependency_impact(&conn, &cfg.effective_code_roots(), target, depth, limit)?;
            Ok(graph::render_dependency_impact(&impact))
        }
        "cfetch_code_context" => {
            let target = args.get("target").and_then(Value::as_str).unwrap_or("");
            let depth = args
                .get("depth")
                .and_then(Value::as_u64)
                .unwrap_or(graph::DEFAULT_CONTEXT_DEPTH as u64) as usize;
            let limit = if limit == 0 {
                graph::DEFAULT_CONTEXT_LIMIT
            } else {
                limit
            };
            let conn = index::open(&paths::state_dir())?;
            let context = graph::dependency_context(
                &conn,
                &cfg.effective_code_roots(),
                target,
                depth,
                limit,
            )?;
            Ok(graph::render_dependency_context(&context))
        }
        "cfetch_code_symbol" => {
            let query = args.get("query").and_then(Value::as_str).unwrap_or("");
            let limit = if limit == 0 {
                graph::DEFAULT_SYMBOL_LIMIT
            } else {
                limit
            };
            let conn = index::open(&paths::state_dir())?;
            let context = graph::symbol_context(&conn, &cfg.effective_code_roots(), query, limit)?;
            Ok(graph::render_symbol_context(&context))
        }
        other => anyhow::bail!("unknown tool: {other}"),
    }
}

fn call_tool(name: &str, args: &Value) -> Result<CallToolResult, McpError> {
    if !TOOL_NAMES.contains(&name) {
        return Err(McpError::invalid_params(
            format!("unknown tool: {name}"),
            None,
        ));
    }
    Ok(match run_tool(name, args) {
        Ok(text) => CallToolResult::success(vec![ContentBlock::text(text)]),
        Err(error) => CallToolResult::error(vec![ContentBlock::text(format!("error: {error}"))]),
    })
}

struct CfetchMcp;

impl ServerHandler for CfetchMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("cfetch", env!("CARGO_PKG_VERSION")))
            // One doctrine, two renderers: this same source function feeds
            // the AGENTS.md/GEMINI.md marker block.
            .with_instructions(crate::markers::doctrine(crate::markers::Surface::Mcp))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult::with_all_items(tool_defs()))
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        tool_defs().into_iter().find(|tool| tool.name == name)
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        let name = request.name.into_owned();
        if !TOOL_NAMES.contains(&name.as_str()) {
            return Err(McpError::invalid_params(
                format!("unknown tool: {name}"),
                None,
            ));
        }
        let args = Value::Object(request.arguments.unwrap_or_default());
        let result = tokio::task::spawn_blocking(move || call_tool(&name, &args))
            .await
            .map_err(|error| {
                McpError::internal_error(format!("tool worker failed: {error}"), None)
            })??;
        Ok(result.into())
    }
}

/// Serve over the SDK's stdio transport until the client closes the session.
pub fn serve() -> anyhow::Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("cfetch-mcp")
        .build()?;
    runtime.block_on(async {
        let service = CfetchMcp.serve(rmcp::transport::stdio()).await?;
        service.waiting().await?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_info_and_tools_are_one_coherent_contract() {
        let info = CfetchMcp.get_info();
        assert_eq!(info.server_info.name, "cfetch");
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
        assert!(info.capabilities.tools.is_some());
        let tools = tool_defs();
        let names: Vec<&str> = tools.iter().map(|tool| tool.name.as_ref()).collect();
        assert_eq!(
            names, TOOL_NAMES,
            "tools/list and the -32602 gate must agree"
        );
        for tool in tools.iter().filter(|tool| {
            !matches!(
                tool.name.as_ref(),
                "cfetch_maintenance_propose" | "cfetch_maintenance_review"
            )
        }) {
            assert_eq!(
                tool.annotations
                    .as_ref()
                    .and_then(|annotations| annotations.read_only_hint),
                Some(true),
                "{} must stay read-only",
                tool.name
            );
        }
        let proposal = tools
            .iter()
            .find(|tool| tool.name == "cfetch_maintenance_propose")
            .unwrap();
        assert_eq!(proposal.name, "cfetch_maintenance_propose");
        assert_eq!(
            proposal
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.read_only_hint),
            Some(false),
            "the quarantine write must be declared honestly"
        );
        assert_eq!(
            proposal
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.destructive_hint),
            Some(false),
            "a proposal cannot edit trusted memory"
        );
        let review = tools
            .iter()
            .find(|tool| tool.name == "cfetch_maintenance_review")
            .unwrap();
        assert_eq!(review.name, "cfetch_maintenance_review");
        assert_eq!(
            review
                .annotations
                .as_ref()
                .and_then(|annotations| annotations.destructive_hint),
            Some(false),
            "a review cannot edit trusted memory"
        );
        let instructions = info.instructions.as_deref().unwrap();
        // Same source function as the AGENTS.md/GEMINI.md marker block.
        assert_eq!(
            instructions,
            crate::markers::doctrine(crate::markers::Surface::Mcp)
        );
        assert!(
            instructions.contains("cfetch_recall"),
            "MCP surface names the MCP tools"
        );
        assert!(instructions.contains("Before searching files or reading code wholesale"));
    }

    #[test]
    fn runtime_status_tool_is_cached_read_only_and_bounded() {
        let tool = tool_defs()
            .into_iter()
            .find(|tool| tool.name == "cfetch_runtime_status")
            .unwrap();
        assert_eq!(
            tool.annotations
                .and_then(|annotations| annotations.read_only_hint),
            Some(true)
        );
        let text = run_tool("cfetch_runtime_status", &json!({})).unwrap();
        assert!(text.len() <= crate::runtime_status::MCP_MAX_BYTES);
        let status: crate::runtime_status::RuntimeStatusV1 = serde_json::from_str(&text).unwrap();
        assert_eq!(status.schema_version, crate::runtime_status::SCHEMA_VERSION);
        assert!(!text.contains("://"));
        assert!(!text.contains("token_file"));
    }

    #[test]
    fn every_tool_tells_the_model_its_answer_is_capped() {
        // A cap the caller cannot see reads as a broken index: the model
        // re-asks the same question instead of following the file:line
        // pointer the truncated answer already handed it.
        for tool in tool_defs().into_iter().filter(|tool| {
            matches!(
                tool.name.as_ref(),
                "cfetch_recall" | "cfetch_expand" | "cfetch_find"
            )
        }) {
            let description = tool.description.as_deref().unwrap_or_default();
            assert!(
                description.contains("token budget"),
                "{} hides its answer budget: {description}",
                tool.name
            );
        }
    }

    #[test]
    fn unknown_tool_is_a_jsonrpc_invalid_params_error() {
        let error = call_tool("cfetch_delete_everything", &json!({})).unwrap_err();
        assert_eq!(
            error.code.0, -32602,
            "protocol error, not an isError result"
        );
    }
}
