#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SOURCE_DIRECTORY WORK_DIRECTORY OUTPUT_DIRECTORY" >&2
  exit 2
fi

readonly llama_repository="https://github.com/ggml-org/llama.cpp.git"
readonly llama_tag="b10516"
readonly llama_revision="b95502ba9aa0eb73a2f4fc8878d7fbe6a847a0b9"
readonly asset_url="https://github.com/ggml-org/llama.cpp/releases/download/b10516/llama-b10516-bin-ubuntu-vulkan-x64.tar.gz"
readonly asset_sha256="5ce186720f43c415465869b0cd93973b828b219cbf6fbcc22aa899531973c505"
readonly asset_bytes=33289144
readonly source_directory="$(realpath -m -- "$1")"
readonly work_directory="$(realpath -m -- "$2")"
readonly output_directory="$(realpath -m -- "$3")"
readonly helper_path="$(realpath -- "${BASH_SOURCE[0]}")"
export LC_ALL=C
# The version check and dependency trace must resolve the bundled libraries.
unset LD_LIBRARY_PATH LD_PRELOAD LD_AUDIT

# Refuse reuse rather than overwriting source, downloaded inputs, or a runtime.
python3 - "$source_directory" "$work_directory" "$output_directory" <<'PY'
import platform
import sys
from pathlib import Path

paths = [Path(value) for value in sys.argv[1:]]
for index, left in enumerate(paths):
    for right in paths[index + 1:]:
        if left == right or left in right.parents or right in left.parents:
            raise SystemExit("source, work, and output paths must not overlap")
for path in paths[1:]:
    if path.exists() or path.is_symlink():
        raise SystemExit(f"refusing existing work/output path: {path}")
release = platform.freedesktop_os_release()
if (platform.system(), platform.machine(), platform.libc_ver()) != (
    "Linux", "x86_64", ("glibc", "2.39")
) or (release.get("ID"), release.get("VERSION_ID")) != ("ubuntu", "24.04"):
    raise SystemExit("requires Ubuntu 24.04 x86_64 with glibc 2.39")
PY

if [[ ! -d "${source_directory}/.git" ]]; then
  if [[ -e "$source_directory" ]] && [[ -n "$(find "$source_directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "refusing to initialize non-empty source directory: $source_directory" >&2
    exit 1
  fi
  mkdir -p "$source_directory"
  git -C "$source_directory" init --quiet
  git -C "$source_directory" remote add origin "$llama_repository"
  git -C "$source_directory" fetch --depth 1 origin "refs/tags/${llama_tag}"
  git -C "$source_directory" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "$source_directory" rev-parse HEAD)" != "$llama_revision" ]] || \
   [[ "$(git -C "$source_directory" rev-parse --show-toplevel)" != "$source_directory" ]] || \
   [[ -n "$(git -C "$source_directory" status --porcelain --untracked-files=all --ignored)" ]]; then
  echo "requires the exact, clean llama.cpp b10516 source tree" >&2
  exit 1
fi

mkdir -p "$work_directory" "$output_directory/provenance"
readonly archive_path="$work_directory/llama-b10516-bin-ubuntu-vulkan-x64.tar.gz"
curl --fail --location --proto '=https' --tlsv1.2 \
  --output "$archive_path" "$asset_url"

# Validate every member before extraction. The destination is fresh; regular
# files use exclusive creation, and links are created only after their targets.
# Keep the complete official runtime, including dynamic CPU/Vulkan plugins.
python3 - "$archive_path" "$output_directory" "$asset_sha256" "$asset_bytes" <<'PY'
import hashlib
import json
import os
import posixpath
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive, output = map(Path, sys.argv[1:3])
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if digest != sys.argv[3] or archive.stat().st_size != int(sys.argv[4]):
    raise SystemExit("official runtime archive digest/size mismatch")
destination = output / "official-runtime"
destination.mkdir()
with tarfile.open(archive, "r:gz") as bundle:
    members = bundle.getmembers()
    names = {}
    link_targets = {}
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "llama-b10516":
            raise SystemExit(f"unsafe archive member: {member.name}")
        name = str(path)
        if name in names or not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
            raise SystemExit(f"duplicate or unsupported archive member: {name}")
        names[name] = member
        if member.issym() or member.islnk():
            target = posixpath.normpath(posixpath.join(str(path.parent), member.linkname) if member.issym() else member.linkname)
            if target.startswith("/") or PurePosixPath(target).parts[0] != "llama-b10516":
                raise SystemExit(f"archive link escapes runtime: {name}")
            link_targets[name] = target
    for name in names:
        if any(str(parent) in link_targets for parent in PurePosixPath(name).parents):
            raise SystemExit(f"archive member descends through a link: {name}")
    if any(target not in names or names[target].isdir() for target in link_targets.values()):
        raise SystemExit("archive link must target an existing non-directory member")
    for name, member in names.items():
        target = destination / name
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    pending = dict(link_targets)
    while pending:
        ready = [name for name, target in pending.items() if target not in pending]
        if not ready:
            raise SystemExit("archive contains a link cycle")
        for name in ready:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            original = destination / pending.pop(name)
            if names[name].issym():
                os.symlink(os.path.relpath(original, target.parent), target)
            else:
                os.link(original, target)
    (output / "provenance/official-archive-members.json").write_text(json.dumps([
        {"name": name, "type": member.type.decode("ascii"), "bytes": member.size,
         "mode": member.mode, "linkname": member.linkname}
        for name, member in names.items()
    ], sort_keys=True, indent=2) + "\n")
PY

readonly runtime_directory="$output_directory/official-runtime/llama-b10516"
cp /etc/os-release "$output_directory/provenance/os-release"
cp "$helper_path" "$output_directory/provenance/build-pinned-vulkan.sh"
cp "$source_directory/LICENSE" "$output_directory/LICENSE"
dpkg-query -W -f='${binary:Package}\t${Version}\n' > "$output_directory/provenance/dpkg-packages.tsv"
{
  uname -a
  getconf GNU_LIBC_VERSION
  /usr/bin/g++ --version
  /usr/bin/g++ -dumpmachine
  ld --version
} > "$output_directory/provenance/client-toolchain.txt"

# The official asset lacks llama-embedding. Compile only this exact upstream
# client against its matching prebuilt libraries; no backend/shader rebuild.
client_command=(
  /usr/bin/g++ -O2 -std=c++17 -pthread
  -I "$source_directory/include" -I "$source_directory/common" -I "$source_directory/ggml/include"
  "$source_directory/examples/embedding/embedding.cpp"
  -L "$runtime_directory" '-Wl,-rpath,$ORIGIN/official-runtime/llama-b10516'
  -lllama-common -lllama -lggml -lggml-base
  -o "$output_directory/llama-embedding"
)
printf '%q ' "${client_command[@]}" > "$output_directory/provenance/client-command.txt"
printf '\n' >> "$output_directory/provenance/client-command.txt"
"${client_command[@]}" 2>&1 | tee "$output_directory/provenance/client-build.log"

# This bounded information-only check was proved separately on the exact asset.
# It reports the PREBUILT library compiler, not the compiler of our new client.
timeout --signal=TERM --kill-after=2s 15s "$output_directory/llama-embedding" --version \
  > "$output_directory/provenance/official-library-version.txt" 2>&1

python3 - "$source_directory" "$output_directory" "$llama_revision" "$asset_url" "$asset_sha256" "$asset_bytes" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:3])
revision, asset_url, asset_sha256, asset_bytes = sys.argv[3:]

def run(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

def git_output(*args):
    # Git can warn about an unreadable user config while returning clean status.
    # Keep diagnostics as evidence, separate from identity/status stdout; a
    # nonzero exit still raises and prevents publication of the candidate.
    with (output / "provenance/git-diagnostics.log").open("a") as diagnostics:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, stderr=diagnostics
        )

def identity(path):
    with path.open("rb") as stream:
        return {"bytes": path.stat().st_size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}

if git_output("rev-parse", "HEAD").strip() != revision or git_output(
    "status", "--porcelain", "--untracked-files=all", "--ignored"
).strip():
    raise SystemExit("source changed during client build")
version = (output / "provenance/official-library-version.txt").read_text()
if "b95502ba9" not in version:
    raise SystemExit("prebuilt library version does not report the pinned commit")
elf = []
for path in sorted(output.rglob("*")):
    if path.is_symlink() or not path.is_file():
        continue
    with path.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            continue
    elf.append({"path": str(path.relative_to(output)), **identity(path),
                "metadata": run("readelf", "--file-header", "--dynamic", "--version-info", str(path))})
(output / "provenance/elf.json").write_text(json.dumps(elf, sort_keys=True, indent=2) + "\n")
trace = run("ldd", str(output / "llama-embedding"))
if "not found" in trace:
    raise SystemExit("client has unresolved runtime dependencies")
(output / "provenance/client-ldd.txt").write_text(trace)
document = {
    "schema_version": 1, "purpose": "candidate-client-and-official-runtime-not-admission",
    "repository_commit": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    "runner_image": os.environ.get("ImageOS"), "runner_image_version": os.environ.get("ImageVersion"),
    "source": {"repository": "https://github.com/ggml-org/llama.cpp.git", "tag": "b10516", "commit": revision,
               "tree": git_output("rev-parse", "HEAD^{tree}").strip(),
               "client": identity(source / "examples/embedding/embedding.cpp"), "clean_before_and_after_build": True},
    "official_asset": {"url": asset_url, "bytes": int(asset_bytes), "sha256": asset_sha256,
                       "member_manifest": "provenance/official-archive-members.json",
                       "reported_library_version_and_compiler": version},
    "client": {**identity(output / "llama-embedding"), "toolchain": "provenance/client-toolchain.txt",
               "command": "provenance/client-command.txt", "library_compiler_is_not_client_compiler": True},
    "runtime_contract": {"os": "Linux", "architecture": "x86_64", "minimum_glibc": "2.39",
                         "client_build_distribution": "Ubuntu 24.04", "system_vulkan_loader": "libvulkan.so.1",
                         "target_requirements": "Compatible system C/C++ runtime, Vulkan loader, and target-installed Vulkan ICD/driver"},
    "only_client_compiled": True, "information_only_check": "--version, 15 seconds plus 2-second termination grace",
    "gpu_executed": False, "model_inference_executed": False,
    "files": {str(path.relative_to(output)): ({"symlink": os.readlink(path)} if path.is_symlink() else identity(path))
              for path in sorted(output.rglob("*")) if path.is_symlink() or path.is_file()},
}
(output / "BUILD-PROVENANCE.json").write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
(output / "SHA256SUMS").write_text("".join(
    f"{identity(path)['sha256']}  {path.relative_to(output)}\n"
    for path in sorted(output.rglob("*")) if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
))
PY

printf 'Candidate client and official runtime prepared at %s; no model inference or admission.\n' "$output_directory"
