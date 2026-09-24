{
  description = "cfetch — local, cited memory over Markdown and Git";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.crane.url = "github:ipetkov/crane";
  outputs = { self, nixpkgs, crane }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg: nixpkgs.lib.getName pkg == "cfetch";
        }));
    in {
      packages = forAllSystems (pkgs: rec {
        cfetch = pkgs.callPackage ./nix/package.nix {
          src = self;
          craneLib = (crane.mkLib pkgs).overrideScope (_final: _prev:
            pkgs.lib.optionalAttrs (!(pkgs.cargo-auditable.meta.broken or false)) {
              # Crane's spliced scope keeps the complete overridden toolchain.
              inherit (pkgs) rustc clippy rustfmt;
              # Preserve the dependency metadata emitted by buildRustPackage.
              cargo = pkgs.buildPackages.cargo-auditable-cargo-wrapper.override {
                cargo = pkgs.buildPackages.cargo;
                cargo-auditable = pkgs.buildPackages.cargo-auditable;
              };
            });
        };
        default = cfetch;
      });
      checks = forAllSystems (pkgs: {
        cfetch = self.packages.${pkgs.stdenv.hostPlatform.system}.cfetch;
      });
    };
}
