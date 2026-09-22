{ lib, stdenv, rustPlatform, fetchurl, runCommand, xz, src }:
let
  system = stdenv.hostPlatform.system;
  # Same pinned CPU archives as ort-sys 2.0.0-rc.13. Cargo cannot download
  # native code inside the Nix sandbox; this fetch has its own fixed hash.
  runtimes = {
    x86_64-linux = {
      target = "x86_64-unknown-linux-gnu";
      sha256 = "e454f710f8a49f53aa5b4ff51e3454ae1835777e431c6c35c5255ce6f205fd68";
    };
    aarch64-linux = {
      target = "aarch64-unknown-linux-gnu";
      sha256 = "06a050ab9137ccb32421d0cb49e9ccf72d9e18ab0aeb8f8d038d1b5cc844b35a";
    };
    aarch64-darwin = {
      target = "aarch64-apple-darwin+coreml";
      sha256 = "6934874e2e953576d9c1db47ff1af39c62c4f4220dbe6f988e131f72879674c7";
    };
  };
  embedded = builtins.hasAttr system runtimes;
  runtime = runtimes.${system};
  onnxRuntime = runCommand "cfetch-onnxruntime-1.28.0-${system}" {
    nativeBuildInputs = [ xz ];
    archive = fetchurl {
      url = "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/${runtime.target}.tar.lzma2";
      inherit (runtime) sha256;
    };
  } ''
    mkdir -p "$out"
    xz --decompress --stdout --format=raw --lzma2=dict=64MiB "$archive" | tar -xf - -C "$out"
    test -f "$out/libonnxruntime.a"
  '';
  os = if stdenv.hostPlatform.isDarwin then "mac" else "linux";
  arch = if stdenv.hostPlatform.isAarch64 then "arm64" else "x86_64";
  backend = if embedded then "cpu" else "cli";
in
rustPlatform.buildRustPackage ({
  pname = "cfetch";
  version = (builtins.fromTOML (builtins.readFile (src + "/Cargo.toml"))).package.version;
  inherit src;
  cargoLock.lockFile = src + "/Cargo.lock";
  cargoBuildFeatures = lib.optional embedded "embedded-embeddings";
  cargoCheckFeatures = lib.optional embedded "embedded-embeddings";
  CFETCH_VARIANT = "${os}-cfetch-${backend}-${arch}";
  ORT_SKIP_DOWNLOAD = "1";
  CARGO_BUILD_JOBS = "2";
  RUST_TEST_THREADS = "1";
  postInstall = ''
    install -Dm644 LICENSE.md "$out/share/licenses/cfetch/LICENSE.md"
    install -Dm644 THIRD-PARTY-LICENSES.txt "$out/share/licenses/cfetch/THIRD-PARTY-LICENSES.txt"
    install -Dm644 release/onnxruntime-LICENSE.txt "$out/share/licenses/cfetch/onnxruntime-LICENSE.txt"
    install -Dm644 release/onnxruntime-NOTICES.txt "$out/share/licenses/cfetch/onnxruntime-NOTICES.txt"
  '';
  meta = {
    description = "Cited memory, local retrieval and Markdown graphs for AI agents";
    homepage = "https://github.com/corbet-libs/cfetch";
    license = {
      fullName = "Functional Source License, Version 1.1, ALv2 Future License";
      url = "https://fsl.software/FSL-1.1-ALv2.template.md";
      free = false;
      redistributable = true;
    };
    platforms = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    mainProgram = "cfetch";
  };
} // lib.optionalAttrs embedded { ORT_LIB_PATH = "${onnxRuntime}"; })
