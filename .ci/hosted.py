#!/usr/bin/env python3
"""Select existing public checks and retain exact hosted input/result evidence."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CHECKS = {"ci-config", "catalog", "governor", "rust", "variants", "licenses", "profile"}
DEPENDENCIES = ("Cargo.toml", "Cargo.lock", "release/variants.json", "release/inference-backends.json",
                "experiments/embedding-profile/requirements-lock.txt")


def selected_checks(value, config):
    selected = [part.strip() for part in value.split(",")]
    if not selected or any(not part or part not in PUBLIC_CHECKS for part in selected) or len(set(selected)) != len(selected):
        raise ValueError("Select unique public checks; hardware/runtime bundle selectors require Crow")
    for check in selected:
        if config["checks"].get(check) != {"kind": "commands", "commands": [["bash", "scripts/ci-check.sh", check]]}:
            raise ValueError("Hosted command differs from the declared Crow check")
    return selected


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def captured(*command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def flags(selected):
    return {"rust": bool(set(selected) & {"rust", "variants", "licenses"}),
            "profile": "profile" in selected, "licenses": "licenses" in selected,
            "ci_config": "ci-config" in selected}


def receipt_path():
    return Path(os.environ["RUNNER_TEMP"]) / "cfetch-checks" / "result.json"


def save(receipt):
    path = receipt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")


def plan():
    config = tomllib.loads((ROOT / ".ci/ccid.toml").read_text())
    selected = selected_checks(os.environ["REQUESTED_CHECKS"], config)
    commit = captured("git", "rev-parse", "HEAD")
    if commit != os.environ["GITHUB_SHA"]:
        raise ValueError("Hosted checkout differs from the dispatched source")
    archive = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=ROOT)
    save({"schema": 1, "provider": "github-actions", "status": "prepared", "checks": selected,
          "commit": commit, "source_sha256": hashlib.sha256(archive).hexdigest(),
          "workflow_sha256": digest(ROOT / ".github/workflows/selected.yml"),
          "manifest_sha256": digest(ROOT / ".ci/ccid.toml"),
          "dependency_inputs": {path: digest(ROOT / path) for path in DEPENDENCIES},
          "budget": {"build_jobs": os.environ["CARGO_BUILD_JOBS"], "test_threads": os.environ["RUST_TEST_THREADS"]},
          "run_url": os.environ["GITHUB_SERVER_URL"] + "/" + os.environ["GITHUB_REPOSITORY"] + "/actions/runs/" + os.environ["GITHUB_RUN_ID"]})
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        for name, enabled in flags(selected).items():
            output.write(name + "=" + str(enabled).lower() + "\n")


def run():
    receipt = json.loads(receipt_path().read_text())
    selected = selected_checks(",".join(receipt["checks"]), tomllib.loads((ROOT / ".ci/ccid.toml").read_text()))
    if captured("git", "rev-parse", "HEAD") != receipt["commit"]:
        raise ValueError("Source changed after hosted planning")
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=True)
    if any(digest(ROOT / path) != expected for path, expected in receipt["dependency_inputs"].items()):
        raise ValueError("Dependencies changed after hosted planning")
    receipt["platform"] = platform.platform()
    receipt["machine"] = platform.machine()
    receipt["tools"] = {"python": sys.version}
    tools = ["rustc", "cargo"] if flags(selected)["rust"] else []
    if "licenses" in selected:
        tools += ["cargo-deny", "cargo-about"]
    for tool in tools:
        receipt["tools"][tool] = captured(tool, "--version")
    save(receipt)
    subprocess.run(["bash", "scripts/ci-check.sh", *selected], cwd=ROOT, check=True)


def finish():
    receipt = json.loads(receipt_path().read_text())
    receipt["status"] = os.environ["CHECK_OUTCOME"]
    save(receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    {"plan": plan, "run": run, "finish": finish}[sys.argv[1]]()
