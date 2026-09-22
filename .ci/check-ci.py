#!/usr/bin/env python3
"""Check the public CI configuration using existing tools, without compilation."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    found = shutil.which(name)
    if not found:
        candidates = sorted(Path("/nix/store").glob("*-" + name + "-*/bin/" + name))
        found = str(candidates[-1]) if candidates else None
    if not found:
        raise SystemExit(name + " must be provisioned; this checker installs nothing")
    return found


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    for path in sorted((ROOT / ".ci").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    for name in ("release.py", "test_release.py", "release_maintenance.py", "test_release_maintenance.py"):
        ast.parse((ROOT / "scripts" / name).read_text(), filename="scripts/" + name)
    config = tomllib.loads((ROOT / ".ci/ccid.toml").read_text())
    require(config["checks"]["ci-config"]["commands"] == [["bash", "scripts/ci-check.sh", "ci-config"]],
            "Configuration checks must use the common command")
    crow = (ROOT / ".crow/ccid.yaml").read_text()
    require(re.search(r"CCID_REVISION: '[0-9a-f]{40}'", crow) and "0d268389" not in crow,
            "Crow needs a verified source-path-aware executor")
    for name in ("CI_JOBS", "CI_TEST_THREADS"):
        require(name + ': {default: "1"}' in crow, "Crow must retain the one-thread default")
    paths = [".github/workflows/ci.yml", ".github/workflows/selected.yml", ".github/workflows/release.yml"]
    for path in paths:
        text = (ROOT / path).read_text()
        require('CARGO_BUILD_JOBS: "1"' in text and 'RUST_TEST_THREADS: "1"' in text,
                "Hosted checks must retain the one-thread budget")
    selected = (ROOT / paths[1]).read_text()
    require("secrets." not in selected and "github.event.repository.private == false" in selected,
            "Selected hosted checks require public input without publication secrets")
    require("toolchain: stable" in selected, "Hosted default compiler must be current stable")
    release = (ROOT / paths[2]).read_text()
    require("uses: ./.github/workflows/ci.yml" in release and "release_checks: true" in release,
            "Release must wait for the shared CI gates")
    ci = (ROOT / paths[0]).read_text()
    require('admitted=$(bash scripts/variant-matrix.sh --release)' in ci
            and "needs.catalog.outputs.has_variants == 'true'" in ci,
            "Release CI must retain unadmitted candidates without repeating admitted native builds")
    require("environment: crates-io" in release and "RELEASE_CARGO_AUTH: trusted" in release,
            "Cargo OIDC must retain its registered workflow/environment boundary")
    require("run: cargo publish" not in release and "python scripts/release.py publish cargo" in release,
            "Credentialed publication must reuse the inspected crate")
    release_crow = (ROOT / ".crow/release.yaml").read_text()
    require("CCID_REVISION: '33600b394910c643c5a6af6dff479fb16418428a'" in release_crow
            and release_crow.count("from_secret:") == 3, "Release requires its verified publisher and scoped secret leaves")
    public_step = release_crow.split("  - name: prepare-or-inspect\n", 1)[1].split("  - name: github-release-state\n", 1)[0]
    require("from_secret:" not in public_step and "RELEASE_OPERATION" in public_step,
            "Preparation/inspection must not request publication credentials")
    require('RELEASE_OPERATION == "publish" && RELEASE_SELECTION == "cargo"' in release_crow,
            "Cargo token resolution must be gated on the selected publication")
    require(config["checks"]["release"]["commands"] == [["python3", "scripts/release.py"]]
            and config["checks"]["release-guards"]["commands"] == [["bash", "scripts/ci-check.sh", "release-guards"]],
            "Release commands must share the repository driver")
    shellcheck, actionlint = tool("shellcheck"), tool("actionlint")
    subprocess.run([shellcheck, "scripts/ci-check.sh"], cwd=ROOT, check=True)
    paths += [".github/workflows/homebrew.yml", ".github/workflows/prepare-release.yml", ".github/workflows/maintenance.yml"]
    subprocess.run([actionlint, "-shellcheck", shellcheck, *paths], cwd=ROOT, check=True)
    print(json.dumps({"actionlint": subprocess.check_output([actionlint, "-version"], text=True).strip(),
                      "shellcheck": subprocess.check_output([shellcheck, "--version"], text=True).strip(),
                      "files": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}}, sort_keys=True))


if __name__ == "__main__":
    main()
