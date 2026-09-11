#!/usr/bin/env python3
"""Shared release-maintenance preparation. This command never pushes or tags."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import release as r

PLATFORMS = (("mac", "aarch64"), ("mac", "x86_64"), ("linux", "aarch64"), ("linux", "x86_64"))
PATCH_FILES = {"Cargo.toml", "Cargo.lock", "CHANGELOG.md", "packaging/arch/PKGBUILD"}


def version(value):
    r.require(isinstance(value, str) and re.fullmatch(r"0\.[0-9]+\.[0-9]+", value),
              "Expected a pre-1.0 version without a v-prefix")
    return value


def immutable(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        r.require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload,
                  "Retained maintenance output differs; use its exact input identity")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def checksum_inventory(payload):
    result = {}
    for line in payload.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([A-Za-z0-9_.-]+)", line)
        r.require(match is not None and match[2] not in result, "Malformed or duplicate published checksum")
        result[match[2]] = match[1]
    r.require(result, "Published checksums are empty")
    return result


def formula(selected_version, catalog, checksums):
    version(selected_version)
    rows = catalog.get("variants")
    r.require(catalog.get("schema_version") == 1 and isinstance(rows, list), "Unsupported variant catalog")
    selected = []
    for operating_system, architecture in PLATFORMS:
        matches = [row for row in rows if row.get("os") == operating_system
                   and row.get("arch") == architecture and row.get("backend") == "endpoint"]
        r.require(len(matches) == 1, "Exactly one endpoint variant is required per Homebrew platform")
        row = matches[0]
        r.require(re.fullmatch(r"[a-z0-9_-]+", row.get("id", "")) and row.get("archive") == "tar.gz"
                  and row.get("binary") == "cfetch", "Unsupported Homebrew endpoint artifact")
        artifact = row["id"] + "-v" + selected_version + ".tar.gz"
        digest = checksums.get(artifact, "")
        r.require(re.fullmatch(r"[0-9a-f]{64}", digest), "Missing Homebrew artifact checksum: " + artifact)
        selected.append((artifact, digest))
    lines = ["class Cfetch < Formula",
             '  desc "Agent memory: ring-ordered recall over a shared brain, with native Claude Code and Codex hooks"',
             '  homepage "https://github.com/corbet-labs/cfetch"', '  license "FSL-1.1-ALv2"',
             f'  version "{selected_version}"']
    for offset, operating_system in ((0, "macos"), (2, "linux")):
        lines += ["", f"  on_{operating_system} do", "    if Hardware::CPU.arm?"]
        for index in (offset, offset + 1):
            if index != offset:
                lines.append("    else")
            artifact, digest = selected[index]
            lines += [f'      url "https://github.com/corbet-labs/cfetch/releases/download/v{selected_version}/{artifact}"',
                      f'      sha256 "{digest}"']
        lines += ["    end", "  end"]
    lines += ["", "  def install", '    bin.install "cfetch"', '    doc.install "LICENSE.md"',
              '    doc.install "THIRD-PARTY-LICENSES.txt"', "  end", "", "  test do",
              '    assert_match version.to_s, shell_output("#{bin}/cfetch --version")', "  end", "end", ""]
    return "\n".join(lines).encode(), {name: digest for name, digest in selected}


def public_metadata(directory, selected_version, expected_commit=""):
    """Legacy public releases have no retained bundle; do not invent gate proof."""
    version(selected_version)
    release = r.read_json(directory / "release.json")
    tag = r.read_json(directory / "tag.json")
    r.require(release.get("tagName") == "v" + selected_version and release.get("isDraft") is False
              and release.get("isPrerelease") is False, "Homebrew requires the selected public stable release")
    r.require(tag.get("type") == "commit" and re.fullmatch(r"[0-9a-f]{40}", tag.get("sha", "")),
              "Release tag must resolve to an exact commit")
    if expected_commit:
        r.require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) and tag["sha"] == expected_commit,
                  "Published tag differs from the successful producing commit")
    assets = release.get("assets")
    r.require(isinstance(assets, list) and all(isinstance(row, dict) and isinstance(row.get("name"), str) for row in assets),
              "Malformed public release assets")
    names = [row["name"] for row in assets]
    r.require(len(names) == len(set(names)), "Duplicate public release asset")
    checksums = checksum_inventory((directory / "checksums_sha256.txt").read_bytes())
    r.require(checksums.get("variants.json") == r.sha(directory / "variants.json"), "Published catalog checksum differs")
    payload, selected = formula(selected_version, r.read_json(directory / "variants.json"), checksums)
    r.require(set(checksums) <= set(names) and set(selected) <= set(names)
              and "checksums_sha256.txt" in names, "Public release asset inventory is incomplete")
    return payload, {"version": selected_version, "tag_commit": tag["sha"], "artifacts": selected,
                     "proof": "public-release-metadata; historical build gates are not re-established"}


def render_bundle(directory):
    r.no_credentials()
    with r.imported(r.publisher()) as (tree, inspected):
        source, _, _, _, _ = inspected
        metadata = tree / "artifacts"
        payload, selected = formula(source["version"], r.read_json(metadata / "variants.json"),
                                    checksum_inventory((metadata / "checksums_sha256.txt").read_bytes()))
        immutable(directory / "cfetch.rb", payload)
        receipt = {**source, "operation": "homebrew-render", "bundle_sha256": os.environ["RELEASE_BUNDLE_SHA256"],
                   "formula_sha256": hashlib.sha256(payload).hexdigest(), "artifacts": selected,
                   "gates": "complete", "publication": "not checked; verify the exact public release before tap writes"}
        r.durable(directory / "homebrew-plan.json", receipt)
        return receipt


def git(root, *args):
    return subprocess.check_output(["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
                                   stderr=subprocess.PIPE)


def patch_plan(directory):
    r.no_credentials()
    bundle = Path(os.environ.get("MAINTENANCE_HISTORY_BUNDLE", ""))
    digest = os.environ.get("MAINTENANCE_HISTORY_SHA256", "")
    r.require(bundle.is_file() and not bundle.is_symlink() and bundle.stat().st_size <= 128 * 1024**2
              and re.fullmatch(r"[0-9a-f]{64}", digest) and r.sha(bundle) == digest,
              "Stage a full Git history bundle and its independent MAINTENANCE_HISTORY_SHA256 before patch planning")
    source, files = r.current_source(directory)
    with tempfile.TemporaryDirectory(prefix="cfetch-patch-plan-") as temporary:
        checkout = Path(temporary) / "source"
        subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "clone", "--quiet", "--no-checkout",
                        "--no-hardlinks", "--", str(bundle), str(checkout)], check=True, capture_output=True)
        git(checkout, "checkout", "--quiet", "--detach", source["source_commit"])
        original_archive = Path(temporary) / "source.tar"
        original_archive.write_bytes(git(checkout, "archive", "--format=tar", source["source_commit"]))
        r.require(r.source_files(original_archive) == (source["source_commit"], files), "History bundle source differs")
        def modes(path):
            with tarfile.open(path) as archive:
                return {member.name: bool(member.mode & 0o111) for member in archive if member.isfile()}
        r.require(modes(original_archive) == modes(directory / "source.tar"), "History bundle source modes differ")
        tags = git(checkout, "tag", "--list", "v[0-9]*", "--sort=-v:refname").decode().splitlines()
        r.require(tags, "History bundle has no release tags")
        latest = tags[0]
        version(latest.removeprefix("v"))
        latest_commit = git(checkout, "rev-parse", latest + "^{commit}").decode().strip()
        git(checkout, "merge-base", "--is-ancestor", latest_commit, source["source_commit"])
        selected_version = version(subprocess.check_output(["bash", "scripts/prepare-patch-release.sh"], cwd=checkout,
                                                           stderr=subprocess.PIPE, text=True).strip())
        changed = set(git(checkout, "diff", "--name-only").decode().splitlines())
        r.require(changed <= PATCH_FILES, "Patch preparation changed an unexpected path")
        patch = git(checkout, "diff", "--binary", "--full-index")
        receipt = {**source, "operation": "patch-plan", "history_sha256": digest, "latest_tag": latest,
                   "latest_tag_commit": latest_commit, "proposed_version": selected_version,
                   "patch_sha256": hashlib.sha256(patch).hexdigest(),
                   "after": {name: r.sha(checkout / name) for name in sorted(PATCH_FILES)},
                   "pending": ["regenerate dependency licenses", "publish exactly one prepared commit",
                               "verify all required CI and native release evidence", "create the exact immutable tag"],
                   "publication": "not attempted"}
        immutable(directory / "patch.diff", patch)
        r.durable(directory / "patch-plan.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["guards", "version", "homebrew-metadata", "homebrew-render", "patch-plan"],
                        nargs="?", default=os.environ.get("MAINTENANCE_OPERATION", "guards"))
    parser.add_argument("--version", default=os.environ.get("MAINTENANCE_VERSION", ""))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit", default="")
    args = parser.parse_args()
    r.require(args.operation in {"guards", "version", "homebrew-metadata", "homebrew-render", "patch-plan"},
              "Unsupported maintenance operation")
    if args.operation == "guards":
        r.no_credentials()
        subprocess.run([sys.executable, "-m", "unittest", "-v", "scripts.test_release_maintenance"], cwd=r.ROOT, check=True)
    elif args.operation == "version":
        print(version(args.version))
    elif args.operation == "homebrew-metadata":
        r.require(args.metadata and args.output, "Supply metadata directory and formula output path")
        payload, receipt = public_metadata(args.metadata, args.version, args.expected_commit)
        immutable(args.output, payload)
        print(json.dumps(receipt, sort_keys=True))
    else:
        directory = r.output_directory() / "maintenance" / args.operation
        directory.mkdir(parents=True, exist_ok=True)
        result = render_bundle(directory) if args.operation == "homebrew-render" else patch_plan(directory)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (r.Failure, subprocess.CalledProcessError, OSError, ValueError, KeyError) as error:
        print("Release maintenance stopped: " + str(error), file=sys.stderr)
        sys.exit(1)
