//! cfetch's installation facade.
//!
//! cfetch owns the behavior it promises to agent harnesses. `agent-config`
//! owns the volatile file locations, schemas, atomic writes, backups, and
//! ownership ledgers for each harness. Claude's explicit `--settings` path is
//! deliberately kept here because it is a public cfetch feature that cannot be
//! represented by agent-config's global/project scopes.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use agent_config::{
    Event, HookCommand, HookSpec, InstallPlan, InstallReport, InstallStatus, InstructionPlacement,
    InstructionSpec, Matcher, McpSpec, PlanStatus, PlannedChange, Scope, ScopeKind, UninstallPlan,
};
use serde_json::{Map, Value, json};

use crate::paths;

const OWNER: &str = "cfetch";
const INSTRUCTION_NAME: &str = "CFETCH";
const LEGACY_MANAGED_BY: &str = "cfetch";

/// cfetch lifecycle events. Tags are intentionally distinct: agent-config's
/// hook tag is an ownership key, and one cfetch installation owns several
/// independent commands.
#[derive(Clone, Copy)]
struct HookRegistration {
    tag: &'static str,
    subcommand: &'static str,
    event: CfetchEvent,
}

#[derive(Clone, Copy)]
enum CfetchEvent {
    SessionStart,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    Stop,
    PreCompact,
}

impl CfetchEvent {
    fn agent_config(self) -> Event {
        match self {
            Self::SessionStart => Event::SessionStart,
            Self::UserPromptSubmit => Event::UserPromptSubmit,
            Self::PreToolUse => Event::PreToolUse,
            Self::PostToolUse => Event::PostToolUse,
            Self::Stop => Event::Stop,
            Self::PreCompact => Event::PreCompact,
        }
    }
}

const FULL_HOOKS: &[HookRegistration] = &[
    HookRegistration {
        tag: "cfetch-session-start",
        subcommand: "session-start",
        event: CfetchEvent::SessionStart,
    },
    HookRegistration {
        tag: "cfetch-user-prompt",
        subcommand: "user-prompt",
        event: CfetchEvent::UserPromptSubmit,
    },
    HookRegistration {
        tag: "cfetch-pre-tool",
        subcommand: "pre-tool",
        event: CfetchEvent::PreToolUse,
    },
    HookRegistration {
        tag: "cfetch-post-tool",
        subcommand: "post-tool",
        event: CfetchEvent::PostToolUse,
    },
    HookRegistration {
        tag: "cfetch-stop",
        subcommand: "stop",
        event: CfetchEvent::Stop,
    },
    HookRegistration {
        tag: "cfetch-precompact",
        subcommand: "precompact",
        event: CfetchEvent::PreCompact,
    },
];

const TOOL_HOOKS: &[HookRegistration] = &[
    HookRegistration {
        tag: "cfetch-pre-tool",
        subcommand: "pre-tool",
        event: CfetchEvent::PreToolUse,
    },
    HookRegistration {
        tag: "cfetch-post-tool",
        subcommand: "post-tool",
        event: CfetchEvent::PostToolUse,
    },
];

/// Native hooks are a stricter capability than "agent-config can write this
/// harness's files": cfetch also has to understand the hook's JSON contract.
/// Prompt-only agents and unverified payload shapes still get MCP and/or
/// instructions, but never a hook registration that only looks supported.
fn native_hooks(agent: &str, scope: &Scope) -> &'static [HookRegistration] {
    match agent {
        // The historical global Claude `--settings` contract is retained by
        // the cfetch-owned merger. Project-local Claude hooks use the adapter.
        "claude" if matches!(scope, Scope::Local(_)) => FULL_HOOKS,
        "codex" | "codebuddy" => FULL_HOOKS,
        "gemini" | "iflow" | "tabnine" => TOOL_HOOKS,
        // Claude supports the full lifecycle, but its public `--settings`
        // path is managed by the cfetch-owned merger below.
        _ => &[],
    }
}

pub fn default_settings_path() -> PathBuf {
    paths::home().join(".claude/settings.json")
}

fn posix_quote(word: &str) -> String {
    format!("'{}'", word.replace('\'', r"'\''"))
}

fn windows_quote(word: &str) -> String {
    format!("\"{word}\"")
}

fn shell_quote(word: &str) -> String {
    if cfg!(windows) {
        windows_quote(word)
    } else {
        posix_quote(word)
    }
}

fn hook_command_for(exe: &str, subcommand: &str) -> String {
    format!("{} hook {subcommand}", shell_quote(exe))
}

fn adapter_hook_command_for(exe: &str, subcommand: &str, agent: &str) -> String {
    format!(
        "{} hook {subcommand} --agent {}",
        shell_quote(exe),
        shell_quote(agent)
    )
}

fn status_line_command_for(exe: &str) -> String {
    format!("{} status --line", shell_quote(exe))
}

fn direct_cfetch_command(command: &str, suffix: &str) -> bool {
    let command = command.trim();
    let Some(program) = command.strip_suffix(suffix).map(str::trim) else {
        return false;
    };
    let program = match (program.as_bytes().first(), program.as_bytes().last()) {
        (Some(b'\''), Some(b'\'')) | (Some(b'"'), Some(b'"')) if program.len() >= 2 => {
            &program[1..program.len() - 1]
        }
        _ if !program.chars().any(char::is_whitespace) => program,
        _ => return false,
    };
    matches!(
        program.rsplit(['/', '\\']).next(),
        Some(name) if name.eq_ignore_ascii_case("cfetch")
            || name.eq_ignore_ascii_case("cfetch.exe")
    )
}

fn legacy_managed_entry(exe: &str, subcommand: &str) -> Value {
    json!({
        "hooks": [{
            "type": "command",
            "command": hook_command_for(exe, subcommand),
            "timeout": 10,
            "_managedBy": LEGACY_MANAGED_BY,
        }]
    })
}

/// Exact historical cfetch hook invocations are safe to adopt during an
/// explicit cfetch install. Older settings writers sometimes retained the
/// command while dropping cfetch's private ownership key; treating those as
/// foreign creates one more invocation on every upgrade. Shell wrappers stay
/// foreign because cfetch cannot prove what else they do.
fn exact_cfetch_hook(command: &str, expected_subcommand: Option<&str>) -> bool {
    let matches_subcommand = |subcommand: &str| {
        let suffix = format!(" hook {subcommand}");
        direct_cfetch_command(command, &suffix)
    };
    expected_subcommand.map_or_else(
        || {
            FULL_HOOKS
                .iter()
                .any(|registration| matches_subcommand(registration.subcommand))
        },
        matches_subcommand,
    )
}

fn owned_claude_status_line(exe: &str) -> Value {
    json!({
        "type": "command",
        "command": status_line_command_for(exe),
        "refreshInterval": 5,
    })
}

fn exact_owned_status_line(value: &Value) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.len() == 3
        && object.get("type").and_then(Value::as_str) == Some("command")
        && object.get("refreshInterval").and_then(Value::as_u64) == Some(5)
        && object
            .get("command")
            .and_then(Value::as_str)
            .is_some_and(|command| direct_cfetch_command(command, " status --line"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeStatusLineResult {
    Installed,
    Updated,
    Replaced,
    PreservedForeign,
    Removed,
    Absent,
}

fn strip_legacy_managed(entry: &mut Value, expected_subcommand: Option<&str>) -> bool {
    match entry.get_mut("hooks").and_then(Value::as_array_mut) {
        Some(hooks) => {
            hooks.retain(|hook| {
                let tagged =
                    hook.get("_managedBy").and_then(Value::as_str) == Some(LEGACY_MANAGED_BY);
                let orphaned = hook
                    .get("command")
                    .and_then(Value::as_str)
                    .is_some_and(|command| exact_cfetch_hook(command, expected_subcommand));
                !tagged && !orphaned
            });
            !hooks.is_empty()
        }
        None => true,
    }
}

/// Claude's explicit-settings merger. Only the handler object is tagged, so a
/// user hook co-located in the same group survives both repair and removal.
fn merge_claude_for_exe_with_status(
    settings: Value,
    exe: &str,
    replace_status_line: bool,
) -> anyhow::Result<(Value, ClaudeStatusLineResult)> {
    let mut root = match settings {
        Value::Object(map) => map,
        Value::Null => Map::new(),
        other => anyhow::bail!("settings.json is not a JSON object (found {other})"),
    };
    let hooks = root
        .entry("hooks")
        .or_insert_with(|| Value::Object(Map::new()))
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("settings.json 'hooks' is not an object"))?;

    for registration in FULL_HOOKS {
        let event = registration.event.agent_config();
        let event = event.as_str();
        let entries = hooks
            .entry(event)
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .ok_or_else(|| anyhow::anyhow!("settings.json hooks.{event} is not an array"))?;
        entries.retain_mut(|entry| strip_legacy_managed(entry, Some(registration.subcommand)));
        entries.push(legacy_managed_entry(exe, registration.subcommand));
    }
    let desired = owned_claude_status_line(exe);
    let status_line = match root.get("statusLine") {
        None => {
            root.insert("statusLine".to_string(), desired);
            ClaudeStatusLineResult::Installed
        }
        Some(current) if exact_owned_status_line(current) => {
            let updated = current != &desired;
            root.insert("statusLine".to_string(), desired);
            if updated {
                ClaudeStatusLineResult::Updated
            } else {
                ClaudeStatusLineResult::Installed
            }
        }
        Some(_) if replace_status_line => {
            root.insert("statusLine".to_string(), desired);
            ClaudeStatusLineResult::Replaced
        }
        Some(_) => ClaudeStatusLineResult::PreservedForeign,
    };
    Ok((Value::Object(root), status_line))
}

#[cfg(test)]
fn merge_claude_for_exe(settings: Value, exe: &str) -> anyhow::Result<Value> {
    merge_claude_for_exe_with_status(settings, exe, false).map(|(settings, _)| settings)
}

fn unmerge_legacy_with_status(settings: Value) -> anyhow::Result<(Value, ClaudeStatusLineResult)> {
    let mut root = match settings {
        Value::Object(map) => map,
        other => return Ok((other, ClaudeStatusLineResult::Absent)),
    };
    if let Some(hooks) = root.get_mut("hooks").and_then(Value::as_object_mut) {
        for entries in hooks.values_mut().filter_map(Value::as_array_mut) {
            entries.retain_mut(|entry| strip_legacy_managed(entry, None));
        }
    }
    let status_line = if root.get("statusLine").is_some_and(exact_owned_status_line) {
        root.remove("statusLine");
        ClaudeStatusLineResult::Removed
    } else {
        ClaudeStatusLineResult::Absent
    };
    Ok((Value::Object(root), status_line))
}

fn unmerge_legacy(settings: Value) -> anyhow::Result<Value> {
    unmerge_legacy_with_status(settings).map(|(settings, _)| settings)
}

fn current_exe_str() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.to_str().map(String::from))
        .unwrap_or_else(|| "cfetch".to_string())
}

fn read_or_empty(path: &Path) -> anyhow::Result<String> {
    match std::fs::read_to_string(path) {
        Ok(content) => Ok(content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(anyhow::anyhow!("read {}: {error}", path.display())),
    }
}

fn json_file(path: &Path) -> anyhow::Result<Value> {
    let raw = read_or_empty(path)?;
    if raw.trim().is_empty() {
        return Ok(Value::Object(Map::new()));
    }
    serde_json::from_str(&raw).map_err(|error| {
        anyhow::anyhow!("refusing to touch unparseable {}: {error}", path.display())
    })
}

fn apply_claude(
    settings_path: &Path,
    remove: bool,
    exe: &str,
    replace_status_line: bool,
) -> anyhow::Result<()> {
    let current = match std::fs::read_to_string(settings_path) {
        Ok(content) => serde_json::from_str(&content).map_err(|error| {
            anyhow::anyhow!(
                "refusing to touch unparseable {}: {error}",
                settings_path.display()
            )
        })?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && remove => return Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Object(Map::new()),
        Err(error) => {
            return Err(anyhow::anyhow!("read {}: {error}", settings_path.display()));
        }
    };
    let (next, status_line) = if remove {
        unmerge_legacy_with_status(current)?
    } else {
        merge_claude_for_exe_with_status(current, exe, replace_status_line)?
    };
    crate::fsutil::atomic_write(settings_path, serde_json::to_string_pretty(&next)?)?;
    println!(
        "claude: {} hooks in {}",
        if remove { "removed" } else { "registered" },
        settings_path.display()
    );
    match status_line {
        ClaudeStatusLineResult::Installed => {
            println!("claude: cfetch runtime status line registered (refresh 5s)")
        }
        ClaudeStatusLineResult::Updated => println!("claude: cfetch runtime status line updated"),
        ClaudeStatusLineResult::Replaced => {
            println!("claude: existing status line replaced with cfetch runtime status")
        }
        ClaudeStatusLineResult::PreservedForeign => println!(
            "claude: existing statusLine preserved; compose `cfetch status --line` into its command, or rerun with --replace-status-line"
        ),
        ClaudeStatusLineResult::Removed => println!("claude: cfetch runtime status line removed"),
        ClaudeStatusLineResult::Absent => {}
    }
    Ok(())
}

fn hook_spec(agent: &str, exe: &str, registration: HookRegistration) -> anyhow::Result<HookSpec> {
    let builder = HookSpec::builder(registration.tag)
        .matcher(Matcher::All)
        .event(registration.event.agent_config())
        .timeout_seconds(10);
    let builder = if cfg!(windows) {
        builder.command_shell_unchecked(adapter_hook_command_for(
            exe,
            registration.subcommand,
            agent,
        ))
    } else {
        builder.command_program(exe, ["hook", registration.subcommand, "--agent", agent])
    };
    Ok(builder
        .windows_command(HookCommand::shell_unchecked(adapter_hook_command_for(
            exe,
            registration.subcommand,
            agent,
        )))
        .try_build()?)
}

fn mcp_name(agent: &str) -> String {
    match agent {
        // Preserve the public names shipped by cfetch 0.9. Other adapters get
        // a per-harness key so agent-config ledgers shared by sibling config
        // files cannot overwrite one another's content hash.
        "claude" | "codex" | "gemini" => OWNER.to_string(),
        _ => format!("{OWNER}-{agent}"),
    }
}

fn mcp_spec(agent: &str, exe: &str) -> anyhow::Result<McpSpec> {
    Ok(McpSpec::builder(mcp_name(agent))
        .owner(OWNER)
        .stdio(exe, ["mcp"])
        // cfetch <=0.9 registered this same server name without a ledger.
        // Explicit installation is authority to convert that one entry.
        .adopt_unowned(true)
        .try_build()?)
}

fn instruction_placement(agent: &str) -> InstructionPlacement {
    match agent {
        "claude" => InstructionPlacement::ReferencedFile,
        "cline" | "roo" | "kilocode" | "windsurf" | "antigravity" => {
            InstructionPlacement::StandaloneFile
        }
        _ => InstructionPlacement::InlineBlock,
    }
}

fn instruction_spec(agent: &str) -> anyhow::Result<InstructionSpec> {
    let body = format!(
        "## cfetch — the operator's memory brain\n\n{}",
        crate::markers::doctrine(crate::markers::Surface::Cli)
    );
    Ok(InstructionSpec::builder(INSTRUCTION_NAME)
        .owner(OWNER)
        .placement(instruction_placement(agent))
        .body(body)
        .adopt_unowned(true)
        .try_build()?)
}

fn refused_install(agent: &str, surface: &str, plan: &InstallPlan) -> anyhow::Result<()> {
    if plan.status == PlanStatus::Refused {
        anyhow::bail!("{agent} {surface} installation refused: {:?}", plan.changes);
    }
    Ok(())
}

fn refused_uninstall(agent: &str, surface: &str, plan: &UninstallPlan) -> anyhow::Result<()> {
    if plan.status == PlanStatus::Refused {
        anyhow::bail!("{agent} {surface} removal refused: {:?}", plan.changes);
    }
    Ok(())
}

fn supports_scope(supported: &[ScopeKind], scope: &Scope) -> bool {
    supported.contains(&scope.kind())
}

#[derive(Clone, Copy, Default)]
struct SurfaceSelection {
    mcp: bool,
    instructions: bool,
    hooks: bool,
}

fn content_paths(plan: &InstallPlan) -> BTreeSet<PathBuf> {
    plan.changes
        .iter()
        .filter_map(|change| match change {
            PlannedChange::CreateFile { path }
            | PlannedChange::PatchFile { path }
            | PlannedChange::NoOp { path, .. } => Some(path.clone()),
            _ => None,
        })
        .collect()
}

fn claim_distinct(paths: BTreeSet<PathBuf>, claimed: &mut BTreeSet<PathBuf>) -> bool {
    if paths.is_empty() || paths.is_disjoint(claimed) {
        claimed.extend(paths);
        true
    } else {
        false
    }
}

fn select_surfaces(
    agents: &[String],
    scope: &Scope,
    exe: &str,
) -> anyhow::Result<Vec<SurfaceSelection>> {
    // Several project harnesses deliberately consume the same `.mcp.json` or
    // instruction file. Register that physical surface once: asking each
    // adapter to own the same `cfetch` entry can rewrite its schema and makes
    // the earlier adapter correctly report drift during removal.
    let mut mcp_paths = BTreeSet::new();
    let mut instruction_paths = BTreeSet::new();
    agents
        .iter()
        .map(|agent| {
            let mcp = match agent_config::mcp_by_id(agent) {
                Some(surface) if supports_scope(surface.supported_mcp_scopes(), scope) => {
                    let plan = surface.plan_install_mcp(scope, &mcp_spec(agent, exe)?)?;
                    claim_distinct(content_paths(&plan), &mut mcp_paths)
                }
                _ => false,
            };
            let instructions = match agent_config::instruction_by_id(agent) {
                Some(surface) if supports_scope(surface.supported_instruction_scopes(), scope) => {
                    let plan =
                        surface.plan_install_instruction(scope, &instruction_spec(agent)?)?;
                    claim_distinct(content_paths(&plan), &mut instruction_paths)
                }
                _ => false,
            };
            let hooks = agent_config::by_id(agent)
                .is_some_and(|integration| supports_scope(integration.supported_scopes(), scope));
            Ok(SurfaceSelection {
                mcp,
                instructions,
                hooks,
            })
        })
        .collect()
}

fn ownership_recovery_only(plan: &InstallPlan) -> bool {
    // Some integrations keep different global config files in the same
    // directory and therefore share agent-config's sidecar ledger. Removing
    // one cfetch registration can orphan the next one's identical entry. A
    // ledger-only plan proves the on-disk entry is still byte-for-byte our
    // current spec; never reclaim when the config itself would be changed.
    plan.status != PlanStatus::Refused
        && plan.changes.iter().all(|change| {
            matches!(
                change,
                PlannedChange::WriteLedger { .. } | PlannedChange::NoOp { .. }
            )
        })
}

fn preflight_agent(
    agent: &str,
    scope: &Scope,
    exe: &str,
    remove: bool,
    selected: SurfaceSelection,
) -> anyhow::Result<()> {
    if selected.mcp
        && let Some(surface) = agent_config::mcp_by_id(agent)
        && supports_scope(surface.supported_mcp_scopes(), scope)
    {
        if remove {
            let plan = surface.plan_uninstall_mcp(scope, &mcp_name(agent), OWNER)?;
            if plan.status == PlanStatus::Refused {
                let recovery = surface.plan_install_mcp(scope, &mcp_spec(agent, exe)?)?;
                if !ownership_recovery_only(&recovery) {
                    refused_uninstall(agent, "MCP", &plan)?;
                }
            } else {
                refused_uninstall(agent, "MCP", &plan)?;
            }
        } else {
            let plan = surface.plan_install_mcp(scope, &mcp_spec(agent, exe)?)?;
            refused_install(agent, "MCP", &plan)?;
        }
    }
    if selected.instructions
        && let Some(surface) = agent_config::instruction_by_id(agent)
        && supports_scope(surface.supported_instruction_scopes(), scope)
    {
        if remove {
            let plan = surface.plan_uninstall_instruction(scope, INSTRUCTION_NAME, OWNER)?;
            if plan.status == PlanStatus::Refused {
                let recovery =
                    surface.plan_install_instruction(scope, &instruction_spec(agent)?)?;
                if !ownership_recovery_only(&recovery) {
                    refused_uninstall(agent, "instructions", &plan)?;
                }
            } else {
                refused_uninstall(agent, "instructions", &plan)?;
            }
        } else {
            let plan = surface.plan_install_instruction(scope, &instruction_spec(agent)?)?;
            refused_install(agent, "instructions", &plan)?;
            // agent-config plans an upsert without inspecting the fence it is
            // about to replace, so a broken one is written over instead of
            // reported. These are the operator's own instruction files.
            for path in content_paths(&plan) {
                crate::markers::ensure_no_broken_block(&path, INSTRUCTION_NAME)?;
            }
        }
    }
    if selected.hooks
        && let Some(integration) = agent_config::by_id(agent)
        && supports_scope(integration.supported_scopes(), scope)
    {
        for registration in native_hooks(agent, scope) {
            if remove {
                let plan = integration.plan_uninstall(scope, registration.tag)?;
                refused_uninstall(agent, "hooks", &plan)?;
            } else {
                let plan =
                    integration.plan_install(scope, &hook_spec(agent, exe, *registration)?)?;
                refused_install(agent, "hooks", &plan)?;
            }
        }
    }
    Ok(())
}

#[derive(Default)]
struct InstalledSurfaces {
    mcp: bool,
    instructions: bool,
    hooks: usize,
}

#[derive(Default)]
struct InstallTracker {
    created: BTreeSet<PathBuf>,
}

impl InstallTracker {
    fn record(&mut self, report: InstallReport) -> anyhow::Result<()> {
        // Some harnesses share one config file across several cfetch surfaces.
        // agent-config correctly backs up an existing file on the second
        // write, but if the first write created that file in THIS invocation,
        // the backup is only a partial cfetch installation. It is not user
        // data and would otherwise survive --remove. Never apply this rule to
        // a path that existed before cfetch started.
        self.created.extend(report.created);
        for backup in report.backed_up {
            let Some(raw) = backup.to_str().and_then(|path| path.strip_suffix(".bak")) else {
                continue;
            };
            if self.created.contains(Path::new(raw)) {
                match std::fs::remove_file(&backup) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => {
                        return Err(anyhow::anyhow!(
                            "remove cfetch-only backup {}: {error}",
                            backup.display()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

impl InstalledSurfaces {
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.mcp {
            parts.push("MCP".to_string());
        }
        if self.instructions {
            parts.push("instructions".to_string());
        }
        if self.hooks > 0 {
            parts.push(format!("{} native hooks", self.hooks));
        }
        parts.join(", ")
    }
}

#[cfg(test)]
fn install_agent(agent: &str, scope: &Scope, exe: &str) -> anyhow::Result<InstalledSurfaces> {
    let selected = select_surfaces(&[agent.to_string()], scope, exe)?
        .into_iter()
        .next()
        .expect("one agent has one surface selection");
    install_agent_selected(agent, scope, exe, selected)
}

fn install_agent_selected(
    agent: &str,
    scope: &Scope,
    exe: &str,
    selected: SurfaceSelection,
) -> anyhow::Result<InstalledSurfaces> {
    preflight_agent(agent, scope, exe, false, selected)?;
    let mut installed = InstalledSurfaces::default();
    let mut tracker = InstallTracker::default();
    if selected.mcp
        && let Some(surface) = agent_config::mcp_by_id(agent)
        && supports_scope(surface.supported_mcp_scopes(), scope)
    {
        tracker.record(surface.install_mcp(scope, &mcp_spec(agent, exe)?)?)?;
        installed.mcp = true;
    }
    if selected.instructions
        && let Some(surface) = agent_config::instruction_by_id(agent)
        && supports_scope(surface.supported_instruction_scopes(), scope)
    {
        tracker.record(surface.install_instruction(scope, &instruction_spec(agent)?)?)?;
        installed.instructions = true;
    }
    if selected.hooks
        && let Some(integration) = agent_config::by_id(agent)
        && supports_scope(integration.supported_scopes(), scope)
    {
        for registration in native_hooks(agent, scope) {
            tracker.record(integration.install(scope, &hook_spec(agent, exe, *registration)?)?)?;
            installed.hooks += 1;
        }
    }
    Ok(installed)
}

#[cfg(test)]
fn uninstall_agent(agent: &str, scope: &Scope, exe: &str) -> anyhow::Result<InstalledSurfaces> {
    let selected = select_surfaces(&[agent.to_string()], scope, exe)?
        .into_iter()
        .next()
        .expect("one agent has one surface selection");
    uninstall_agent_selected(agent, scope, exe, selected)
}

fn uninstall_agent_selected(
    agent: &str,
    scope: &Scope,
    exe: &str,
    selected: SurfaceSelection,
) -> anyhow::Result<InstalledSurfaces> {
    preflight_agent(agent, scope, exe, true, selected)?;
    let mut installed = InstalledSurfaces::default();
    if selected.hooks
        && let Some(integration) = agent_config::by_id(agent)
        && supports_scope(integration.supported_scopes(), scope)
    {
        for registration in native_hooks(agent, scope) {
            let backups =
                FreshBackups::for_uninstall(&integration.plan_uninstall(scope, registration.tag)?);
            let _ = integration.uninstall(scope, registration.tag)?;
            backups.remove_after_success()?;
            installed.hooks += 1;
        }
    }
    if selected.instructions
        && let Some(surface) = agent_config::instruction_by_id(agent)
        && supports_scope(surface.supported_instruction_scopes(), scope)
    {
        let removal = surface.plan_uninstall_instruction(scope, INSTRUCTION_NAME, OWNER)?;
        if removal.status == PlanStatus::Refused {
            let recovery = surface.plan_install_instruction(scope, &instruction_spec(agent)?)?;
            if ownership_recovery_only(&recovery) {
                let _ = surface.install_instruction(scope, &instruction_spec(agent)?)?;
            }
        }
        let backups = FreshBackups::for_uninstall(&surface.plan_uninstall_instruction(
            scope,
            INSTRUCTION_NAME,
            OWNER,
        )?);
        let _ = surface.uninstall_instruction(scope, INSTRUCTION_NAME, OWNER)?;
        backups.remove_after_success()?;
        installed.instructions = true;
    }
    if selected.mcp
        && let Some(surface) = agent_config::mcp_by_id(agent)
        && supports_scope(surface.supported_mcp_scopes(), scope)
    {
        let name = mcp_name(agent);
        let removal = surface.plan_uninstall_mcp(scope, &name, OWNER)?;
        if removal.status == PlanStatus::Refused {
            let recovery = surface.plan_install_mcp(scope, &mcp_spec(agent, exe)?)?;
            if ownership_recovery_only(&recovery) {
                let _ = surface.install_mcp(scope, &mcp_spec(agent, exe)?)?;
            }
        }
        let backups =
            FreshBackups::for_uninstall(&surface.plan_uninstall_mcp(scope, &name, OWNER)?);
        let _ = surface.uninstall_mcp(scope, &name, OWNER)?;
        backups.remove_after_success()?;
        installed.mcp = true;
    }
    Ok(installed)
}

fn change_paths(change: &PlannedChange) -> Vec<&Path> {
    match change {
        PlannedChange::CreateFile { path }
        | PlannedChange::PatchFile { path }
        | PlannedChange::RemoveFile { path }
        | PlannedChange::CreateDir { path }
        | PlannedChange::RemoveDir { path }
        | PlannedChange::WriteLedger { path, .. }
        | PlannedChange::RemoveLedgerEntry { path, .. }
        | PlannedChange::SetPermissions { path, .. }
        | PlannedChange::NoOp { path, .. } => vec![path],
        PlannedChange::CreateBackup { backup, target }
        | PlannedChange::RestoreBackup { backup, target } => vec![backup, target],
        PlannedChange::Refuse { path, .. } => path.iter().map(PathBuf::as_path).collect(),
        _ => Vec::new(),
    }
}

fn backup_path(path: &Path) -> PathBuf {
    let mut backup = path.as_os_str().to_os_string();
    backup.push(".bak");
    PathBuf::from(backup)
}

struct FreshBackups {
    absent_before: BTreeSet<PathBuf>,
}

impl FreshBackups {
    fn for_uninstall(plan: &UninstallPlan) -> Self {
        let absent_before = plan
            .changes
            .iter()
            .flat_map(change_paths)
            .map(backup_path)
            .filter(|path| !path.exists())
            .collect();
        Self { absent_before }
    }

    fn remove_after_success(self) -> anyhow::Result<()> {
        // A successful uninstall leaves the user's non-cfetch content in the
        // live file. Backups created by that uninstall contain only the state
        // immediately before cfetch was removed and are not first-touch user
        // snapshots. Existing backups were excluded above and are preserved.
        for path in self.absent_before {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(anyhow::anyhow!(
                        "remove uninstall-only backup {}: {error}",
                        path.display()
                    ));
                }
            }
        }
        Ok(())
    }
}

fn planned_paths(agent: &str, exe: &str, scope: &Scope) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(surface) = agent_config::mcp_by_id(agent)
        && supports_scope(surface.supported_mcp_scopes(), scope)
        && let Ok(plan) =
            surface.plan_install_mcp(scope, &mcp_spec(agent, exe).expect("valid MCP spec"))
    {
        paths.extend(
            plan.changes
                .iter()
                .flat_map(change_paths)
                .map(Path::to_path_buf),
        );
    }
    if let Some(surface) = agent_config::instruction_by_id(agent)
        && supports_scope(surface.supported_instruction_scopes(), scope)
        && let Ok(spec) = instruction_spec(agent)
        && let Ok(plan) = surface.plan_install_instruction(scope, &spec)
    {
        paths.extend(
            plan.changes
                .iter()
                .flat_map(change_paths)
                .map(Path::to_path_buf),
        );
    }
    if let Some(integration) = agent_config::by_id(agent)
        && supports_scope(integration.supported_scopes(), scope)
    {
        for registration in native_hooks(agent, scope) {
            if let Ok(spec) = hook_spec(agent, exe, *registration)
                && let Ok(plan) = integration.plan_install(scope, &spec)
            {
                paths.extend(
                    plan.changes
                        .iter()
                        .flat_map(change_paths)
                        .map(Path::to_path_buf),
                );
            }
        }
    }
    paths
}

fn common_config_roots(scope: &Scope) -> Vec<PathBuf> {
    if let Some(root) = scope.local_root() {
        return vec![root.to_path_buf()];
    }
    let mut roots = vec![paths::home()];
    if let Ok(config) = agent_config::paths::config_dir() {
        roots.push(config.clone());
        let code = config.join("Code");
        roots.push(code.clone());
        let user = code.join("User");
        roots.push(user.clone());
        roots.push(user.join("globalStorage"));
    }
    roots
}

fn path_has_agent_footprint(path: &Path, common: &[PathBuf]) -> bool {
    if path.exists() {
        return true;
    }
    let mut ancestor = path.parent();
    while let Some(candidate) = ancestor {
        if common.iter().any(|root| candidate == root) {
            return false;
        }
        if candidate.exists() {
            return true;
        }
        ancestor = candidate.parent();
    }
    false
}

fn agent_detected(agent: &str, exe: &str, scope: &Scope) -> bool {
    let common = common_config_roots(scope);
    planned_paths(agent, exe, scope)
        .iter()
        .any(|path| path_has_agent_footprint(path, &common))
}

fn supported_agent_ids() -> Vec<&'static str> {
    agent_config::all()
        .into_iter()
        .map(|agent| agent.id())
        .collect()
}

fn resolve_agents(
    requested: &[String],
    all: bool,
    exe: &str,
    scope: &Scope,
) -> anyhow::Result<Vec<String>> {
    let supported = supported_agent_ids();
    if all {
        return Ok(supported.into_iter().map(String::from).collect());
    }
    if !requested.is_empty() {
        let requested: BTreeSet<&str> = requested.iter().map(String::as_str).collect();
        if let Some(unknown) = requested
            .iter()
            .find(|id| !supported.iter().any(|supported| supported == *id))
        {
            anyhow::bail!(
                "unknown agent {unknown:?}; supported agents: {}",
                supported.join(", ")
            );
        }
        return Ok(supported
            .into_iter()
            .filter(|id| requested.contains(id))
            .map(String::from)
            .collect());
    }
    Ok(supported
        .into_iter()
        .filter(|id| agent_detected(id, exe, scope))
        .map(String::from)
        .collect())
}

fn install_lock(agent: &str) -> anyhow::Result<crate::lockfile::Lock> {
    let dir = paths::state_dir().join("locks");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("install-{agent}.lock"));
    crate::lockfile::acquire(&path, 2_000, 0)
        .ok_or_else(|| anyhow::anyhow!("timed out waiting for {}", path.display()))
}

fn codex_toml_without_mcp(content: &str) -> anyhow::Result<Option<String>> {
    let mut document: toml_edit::DocumentMut = content
        .parse()
        .map_err(|error| anyhow::anyhow!("refusing to touch unparseable TOML: {error}"))?;
    let removed = document
        .get_mut("mcp_servers")
        .and_then(|servers| servers.as_table_like_mut())
        .and_then(|servers| servers.remove(OWNER))
        .is_some();
    Ok(removed.then(|| document.to_string()))
}

fn gemini_settings_without_mcp(settings: Value) -> Option<Value> {
    let Value::Object(mut root) = settings else {
        return None;
    };
    root.get_mut("mcpServers")?.as_object_mut()?.remove(OWNER)?;
    Some(Value::Object(root))
}

fn mcp_is_present_unowned(agent: &str) -> anyhow::Result<bool> {
    let Some(surface) = agent_config::mcp_by_id(agent) else {
        return Ok(false);
    };
    let report = surface.mcp_status(&Scope::Global, OWNER, OWNER)?;
    Ok(matches!(report.status, InstallStatus::PresentUnowned))
}

/// One-time conversion from cfetch 0.9's private marker formats. Conversion
/// happens before agent-config's first-touch backup so a later uninstall cannot
/// resurrect the old entries from that backup.
fn migrate_v090(agent: &str) -> anyhow::Result<()> {
    if agent == "codex" {
        let codex = paths::codex_home();
        let hooks_path = codex.join("hooks.json");
        if hooks_path.is_file() {
            let current = json_file(&hooks_path)?;
            let next = unmerge_legacy(current.clone())?;
            if next != current {
                crate::fsutil::atomic_write(&hooks_path, serde_json::to_string_pretty(&next)?)?;
            }
        }
        let agents_path = codex.join("AGENTS.md");
        crate::markers::remove_block_file(&agents_path)?;
        if mcp_is_present_unowned(agent)? {
            let config = codex.join("config.toml");
            let current = read_or_empty(&config)?;
            if let Some(next) = codex_toml_without_mcp(&current)? {
                crate::fsutil::atomic_write(&config, next)?;
            }
        }
    } else if agent == "gemini" {
        let gemini = paths::home().join(".gemini");
        crate::markers::remove_block_file(&gemini.join("GEMINI.md"))?;
        if mcp_is_present_unowned(agent)? {
            let settings = gemini.join("settings.json");
            let current = json_file(&settings)?;
            if let Some(next) = gemini_settings_without_mcp(current) {
                crate::fsutil::atomic_write(&settings, serde_json::to_string_pretty(&next)?)?;
            }
        }
    }
    Ok(())
}

/// Configure cfetch for selected harnesses. With no explicit selection, only
/// harnesses with an existing configuration footprint are touched. `--all` is
/// the explicit authority to create configuration for every supported agent.
pub fn configure(
    settings: Option<&Path>,
    requested: &[String],
    all: bool,
    remove: bool,
    project: Option<&Path>,
    replace_status_line: bool,
) -> anyhow::Result<()> {
    if settings.is_some() && project.is_some() {
        anyhow::bail!("--settings and --project target different scopes");
    }
    let scope = match project {
        Some(path) => {
            let root = path.canonicalize().map_err(|error| {
                anyhow::anyhow!("resolve project root {}: {error}", path.display())
            })?;
            if !root.is_dir() {
                anyhow::bail!("project root is not a directory: {}", root.display());
            }
            Scope::Local(root)
        }
        None => Scope::Global,
    };
    let exe = current_exe_str();
    let mut agents = resolve_agents(requested, all, &exe, &scope)?;
    if settings.is_some() && !agents.iter().any(|agent| agent == "claude") {
        agents.push("claude".to_string());
    }
    if agents.is_empty() {
        println!(
            "no supported agent configuration detected; use --agent <id> or --all (supported: {})",
            supported_agent_ids().join(", ")
        );
        return Ok(());
    }
    let mut selections = select_surfaces(&agents, &scope, &exe)?;
    // An explicit `--settings` file scopes Claude's hook surface to THAT
    // file: `apply_claude` below installs the hooks and status line there.
    // Without this, the same registration was also written to the DEFAULT
    // ~/.claude/settings.json through agent-config — a double install that
    // created files the operator pointed somewhere else precisely to avoid.
    if settings.is_some()
        && scope.local_root().is_none()
        && let Some(selection) = agents
            .iter()
            .position(|agent| agent == "claude")
            .and_then(|index| selections.get_mut(index))
    {
        selection.hooks = false;
    }

    // Refuse known schema/ownership conflicts before touching any harness.
    for (agent, selected) in agents.iter().zip(&selections) {
        if !remove {
            preflight_agent(agent, &scope, &exe, false, *selected)?;
        }
    }

    for (agent, selected) in agents.into_iter().zip(selections) {
        let _lock = install_lock(&agent)?;
        if scope.local_root().is_none() {
            migrate_v090(&agent)?;
        }
        let surfaces = if remove {
            uninstall_agent_selected(&agent, &scope, &exe, selected)
        } else {
            install_agent_selected(&agent, &scope, &exe, selected)
        }
        .map_err(|error| anyhow::anyhow!("{agent}: {error:#}"))?;
        if agent == "claude" && scope.local_root().is_none() {
            let path = settings
                .map(Path::to_path_buf)
                .unwrap_or_else(default_settings_path);
            apply_claude(&path, remove, &exe, replace_status_line)?;
        }
        let description = surfaces.describe();
        if !description.is_empty() {
            println!(
                "{agent}: {} {description}",
                if remove { "removed" } else { "registered" }
            );
        } else {
            println!(
                "{agent}: skipped (no confirmed {} surfaces)",
                if scope.local_root().is_none() {
                    "global"
                } else {
                    "project-local"
                }
            );
        }
    }
    if !remove {
        let _ = crate::runtime_status::refresh_static();
    }
    Ok(())
}

/// Reports drift in a detected Codex installation. Plans are the health check:
/// a correct registration is exactly a no-op for the current cfetch specs.
pub fn codex_registration_issues() -> Option<Vec<String>> {
    let codex = paths::codex_home();
    if !codex.is_dir() {
        return None;
    }
    let exe = current_exe_str();
    let mut issues = Vec::new();
    if let Some(surface) = agent_config::mcp_by_id("codex") {
        match surface.plan_install_mcp(
            &Scope::Global,
            &mcp_spec("codex", &exe).expect("valid MCP spec"),
        ) {
            Ok(plan) if plan.status == PlanStatus::NoOp => {}
            Ok(_) => issues.push("Codex MCP registration is absent or stale".to_string()),
            Err(error) => issues.push(format!("Codex MCP registration: {error}")),
        }
    }
    if let Some(surface) = agent_config::instruction_by_id("codex") {
        match instruction_spec("codex")
            .and_then(|spec| Ok(surface.plan_install_instruction(&Scope::Global, &spec)?))
        {
            Ok(plan) if plan.status == PlanStatus::NoOp => {}
            Ok(_) => issues.push("Codex instruction registration is absent or stale".to_string()),
            Err(error) => issues.push(format!("Codex instruction registration: {error}")),
        }
    }
    if let Some(integration) = agent_config::by_id("codex") {
        for registration in FULL_HOOKS {
            match hook_spec("codex", &exe, *registration)
                .and_then(|spec| Ok(integration.plan_install(&Scope::Global, &spec)?))
            {
                Ok(plan) if plan.status == PlanStatus::NoOp => {}
                Ok(_) => issues.push(format!(
                    "Codex native hook {} is absent or stale",
                    registration.event.agent_config().as_str()
                )),
                Err(error) => issues.push(format!("Codex native hooks: {error}")),
            }
        }
    }
    let config_path = codex.join("config.toml");
    match read_or_empty(&config_path).and_then(|content| {
        content
            .parse::<toml_edit::DocumentMut>()
            .map_err(|error| anyhow::anyhow!("parse {}: {error}", config_path.display()))
    }) {
        Ok(document)
            if document
                .get("features")
                .and_then(|features| features.as_table_like())
                .and_then(|features| features.get("hooks"))
                .and_then(|hooks| hooks.as_bool())
                == Some(false) =>
        {
            issues.push(format!(
                "{} disables native hooks with features.hooks=false",
                config_path.display()
            ));
        }
        Ok(_) => {}
        Err(error) => issues.push(error.to_string()),
    }
    Some(issues)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_merge_is_idempotent_and_preserves_foreign_entries() {
        let existing = json!({
            "permissions": {"allow": ["Bash(ls:*)"]},
            "hooks": {"SessionStart": [
                {"hooks": [{"type": "command", "command": "my-own-hook"}]}
            ]}
        });
        let once = merge_claude_for_exe(existing, "/opt/cfetch/bin/cfetch").unwrap();
        let twice = merge_claude_for_exe(once.clone(), "/opt/cfetch/bin/cfetch").unwrap();
        assert_eq!(once, twice);
        assert_eq!(
            twice["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            "my-own-hook"
        );
        assert_eq!(twice["permissions"]["allow"][0], "Bash(ls:*)");
        assert_eq!(twice["hooks"].as_object().unwrap().len(), FULL_HOOKS.len());
    }

    #[test]
    fn claude_foreign_status_line_is_preserved_exactly_without_opt_in() {
        let foreign = json!({
            "type": "command",
            "command": "my-status --json",
            "refreshInterval": 17,
            "padding": 1
        });
        let before = serde_json::to_vec(&foreign).unwrap();
        let (merged, result) = merge_claude_for_exe_with_status(
            json!({"statusLine": foreign}),
            "/usr/bin/cfetch",
            false,
        )
        .unwrap();
        assert_eq!(result, ClaudeStatusLineResult::PreservedForeign);
        assert_eq!(serde_json::to_vec(&merged["statusLine"]).unwrap(), before);

        let (replaced, result) =
            merge_claude_for_exe_with_status(merged, "/usr/bin/cfetch", true).unwrap();
        assert_eq!(result, ClaudeStatusLineResult::Replaced);
        assert_eq!(
            replaced["statusLine"],
            owned_claude_status_line("/usr/bin/cfetch")
        );
    }

    #[test]
    fn claude_removal_removes_only_the_exact_cfetch_status_line() {
        let (owned, _) =
            merge_claude_for_exe_with_status(Value::Null, "/usr/bin/cfetch", false).unwrap();
        let (removed, result) = unmerge_legacy_with_status(owned).unwrap();
        assert_eq!(result, ClaudeStatusLineResult::Removed);
        assert!(removed.get("statusLine").is_none());

        let foreign =
            json!({"type": "command", "command": "cfetch status --line", "refreshInterval": 9});
        let (preserved, result) =
            unmerge_legacy_with_status(json!({"statusLine": foreign.clone()})).unwrap();
        assert_eq!(result, ClaudeStatusLineResult::Absent);
        assert_eq!(preserved["statusLine"], foreign);
    }

    #[test]
    fn claude_merge_collapses_orphaned_cfetch_hook_duplicates() {
        let existing = json!({
            "hooks": {"Stop": [
                {"hooks": [{
                    "type": "command",
                    "command": "'/old/bin/cfetch' hook stop",
                    "timeout": 10
                }]},
                {"hooks": [
                    {
                        "type": "command",
                        "command": r#""C:\Program Files\cfetch.exe" hook stop"#,
                        "timeout": 10
                    },
                    {"type": "command", "command": "keep-me"}
                ]},
                {"hooks": [{
                    "type": "command",
                    "command": "bash -lc 'cfetch hook stop'"
                }]}
            ]}
        });

        let merged = merge_claude_for_exe(existing, "/usr/bin/cfetch").unwrap();
        let stop = merged["hooks"]["Stop"].as_array().unwrap();
        let commands: Vec<&str> = stop
            .iter()
            .flat_map(|entry| entry["hooks"].as_array().unwrap())
            .filter_map(|hook| hook["command"].as_str())
            .collect();
        assert_eq!(
            commands
                .iter()
                .filter(|command| exact_cfetch_hook(command, Some("stop")))
                .count(),
            1,
            "the event has exactly one direct cfetch invocation"
        );
        let desired = hook_command_for("/usr/bin/cfetch", "stop");
        assert!(commands.contains(&desired.as_str()));
        assert!(commands.contains(&"bash -lc 'cfetch hook stop'"));
        assert!(commands.contains(&"keep-me"));
        assert!(!commands.contains(&"'/old/bin/cfetch' hook stop"));
        assert!(!commands.contains(&r#""C:\Program Files\cfetch.exe" hook stop"#));
    }

    #[test]
    fn claude_removal_keeps_colocated_user_hooks() {
        let mut merged = merge_claude_for_exe(Value::Null, "/usr/bin/cfetch").unwrap();
        merged["hooks"]["Stop"][0]["hooks"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "command", "command": "keep-me"}));
        let clean = unmerge_legacy(merged).unwrap();
        assert_eq!(clean["hooks"]["Stop"][0]["hooks"][0]["command"], "keep-me");
        assert!(
            !serde_json::to_string(&clean)
                .unwrap()
                .contains("_managedBy")
        );
    }

    #[test]
    fn windows_and_posix_commands_quote_paths_with_spaces() {
        assert_eq!(posix_quote("/opt/a b/cfetch"), "'/opt/a b/cfetch'");
        assert_eq!(
            windows_quote(r"C:\Program Files\cfetch\cfetch.exe"),
            "\"C:\\Program Files\\cfetch\\cfetch.exe\""
        );
    }

    #[test]
    fn codex_agent_config_round_trip_owns_all_surfaces() {
        let root = tempfile::tempdir().unwrap();
        let scope = Scope::Local(root.path().to_path_buf());
        let installed = install_agent("codex", &scope, "/opt/cfetch/bin/cfetch").unwrap();
        assert!(installed.mcp);
        assert!(installed.instructions);
        assert_eq!(installed.hooks, FULL_HOOKS.len());

        let hooks = std::fs::read_to_string(root.path().join(".codex/hooks.json")).unwrap();
        assert_eq!(hooks.matches("_agent_config_tag").count(), FULL_HOOKS.len());
        assert_eq!(
            hooks.matches("--agent").count(),
            FULL_HOOKS.len(),
            "every managed hook carries an explicit adapter identity"
        );
        assert!(hooks.contains("codex"));
        assert!(
            !root.path().join(".codex/hooks.json.bak").exists(),
            "a file created by this invocation has no user state to back up"
        );
        let config = std::fs::read_to_string(root.path().join(".codex/config.toml")).unwrap();
        assert!(config.contains("[mcp_servers.cfetch]"));
        let instructions = std::fs::read_to_string(root.path().join("AGENTS.md")).unwrap();
        assert!(instructions.contains("BEGIN AGENT-CONFIG-INSTR:CFETCH"));

        uninstall_agent("codex", &scope, "/opt/cfetch/bin/cfetch").unwrap();
        let remaining = [
            root.path().join(".codex/hooks.json"),
            root.path().join(".codex/config.toml"),
            root.path().join("AGENTS.md"),
        ]
        .into_iter()
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .collect::<String>();
        assert!(!remaining.contains("cfetch"));
    }

    #[test]
    fn qwen_gets_core_surfaces_without_invented_hooks() {
        let root = tempfile::tempdir().unwrap();
        let scope = Scope::Local(root.path().to_path_buf());
        let installed = install_agent("qwen", &scope, "/usr/bin/cfetch").unwrap();
        assert!(installed.mcp);
        assert!(installed.instructions);
        assert_eq!(installed.hooks, 0);
        assert!(root.path().join(".qwen/settings.json").is_file());
        assert!(root.path().join("QWEN.md").is_file());
        assert!(!root.path().join("QWEN.md.bak").exists());
    }

    #[test]
    fn trae_project_scope_gets_its_confirmed_instruction_surface() {
        let root = tempfile::tempdir().unwrap();
        let scope = Scope::Local(root.path().to_path_buf());
        let installed = install_agent("trae", &scope, "/usr/bin/cfetch").unwrap();
        assert!(!installed.mcp);
        assert!(installed.instructions);
        assert_eq!(installed.hooks, 0);

        let rules = std::fs::read_to_string(root.path().join(".trae/project_rules.md")).unwrap();
        assert!(rules.contains("BEGIN AGENT-CONFIG-INSTR:CFETCH"));
        uninstall_agent("trae", &scope, "/usr/bin/cfetch").unwrap();
        assert!(!root.path().join(".trae/project_rules.md.bak").exists());
    }

    #[test]
    fn a_preexisting_user_file_keeps_its_first_touch_backup() {
        let root = tempfile::tempdir().unwrap();
        let codex = root.path().join(".codex");
        std::fs::create_dir(&codex).unwrap();
        let hooks = codex.join("hooks.json");
        std::fs::write(&hooks, r#"{"userSetting": true}"#).unwrap();

        install_agent(
            "codex",
            &Scope::Local(root.path().to_path_buf()),
            "/usr/bin/cfetch",
        )
        .unwrap();

        assert_eq!(
            std::fs::read_to_string(hooks.with_extension("json.bak")).unwrap(),
            r#"{"userSetting": true}"#
        );
    }

    #[test]
    fn a_broken_instruction_fence_stops_the_whole_install() {
        let root = tempfile::tempdir().unwrap();
        let agents = root.path().join("AGENTS.md");
        let block = "<!-- BEGIN AGENT-CONFIG-INSTR:CFETCH -->\nstale\n\
             <!-- END AGENT-CONFIG-INSTR:CFETCH -->\n";
        let broken = format!("# my notes\n\n{block}\n{block}");
        std::fs::write(&agents, &broken).unwrap();

        let error = match install_agent(
            "codex",
            &Scope::Local(root.path().to_path_buf()),
            "/usr/bin/cfetch",
        ) {
            Ok(_) => panic!("a broken instruction fence must refuse the install"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("AGENTS.md"), "{error}");
        assert!(error.contains("exactly one of each"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&agents).unwrap(),
            broken,
            "a refusal leaves the operator's file byte-identical"
        );
        assert!(
            !root.path().join(".codex/config.toml").exists(),
            "no surface is written once one of them is known to be unsafe"
        );
    }

    #[test]
    fn a_well_formed_block_still_reinstalls_in_place() {
        let root = tempfile::tempdir().unwrap();
        let scope = Scope::Local(root.path().to_path_buf());
        let agents = root.path().join("AGENTS.md");
        std::fs::write(&agents, "# my notes\n\nkeep me\n").unwrap();

        install_agent("codex", &scope, "/usr/bin/cfetch").unwrap();
        let once = std::fs::read_to_string(&agents).unwrap();
        install_agent("codex", &scope, "/usr/bin/cfetch").unwrap();

        assert_eq!(std::fs::read_to_string(&agents).unwrap(), once);
        assert_eq!(once.matches("BEGIN AGENT-CONFIG-INSTR:CFETCH").count(), 1);
        assert!(once.contains("keep me"));
    }

    #[test]
    fn shared_project_mcp_file_is_registered_once() {
        let root = tempfile::tempdir().unwrap();
        let agents = vec!["claude".to_string(), "copilot".to_string()];
        let selected = select_surfaces(
            &agents,
            &Scope::Local(root.path().to_path_buf()),
            "/usr/bin/cfetch",
        )
        .unwrap();
        assert!(selected[0].mcp);
        assert!(!selected[1].mcp);
    }

    #[test]
    fn sibling_configs_sharing_a_ledger_get_distinct_mcp_keys() {
        let root = tempfile::tempdir().unwrap();
        let scope = Scope::Local(root.path().to_path_buf());
        for agent in ["opencode", "crush"] {
            let surface = agent_config::mcp_by_id(agent).unwrap();
            let _ = surface
                .install_mcp(&scope, &mcp_spec(agent, "/usr/bin/cfetch").unwrap())
                .unwrap();
        }
        let ledger = std::fs::read_to_string(root.path().join(".agent-config-mcp.json")).unwrap();
        assert!(ledger.contains("cfetch-opencode"));
        assert!(ledger.contains("cfetch-crush"));

        for agent in ["opencode", "crush"] {
            let surface = agent_config::mcp_by_id(agent).unwrap();
            let _ = surface
                .uninstall_mcp(&scope, &mcp_name(agent), OWNER)
                .unwrap();
        }
    }

    #[test]
    fn unknown_agent_is_refused_with_the_live_registry() {
        let error = resolve_agents(
            &["made-up-agent".into()],
            false,
            "/usr/bin/cfetch",
            &Scope::Global,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown agent"));
        assert!(error.contains("qwen"));
        assert!(error.contains("codebuddy"));
    }

    #[test]
    fn arch_package_tracks_the_cargo_release_version() {
        let pkgbuild = include_str!("../packaging/arch/PKGBUILD");
        let declared = pkgbuild
            .lines()
            .find_map(|line| line.strip_prefix("pkgver="))
            .expect("PKGBUILD has pkgver");
        assert_eq!(declared, env!("CARGO_PKG_VERSION"));
    }
}
