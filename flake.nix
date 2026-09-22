{
  description = "cfetch — local, cited memory over Markdown and Git";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg: nixpkgs.lib.getName pkg == "cfetch";
        }));
    in {
      packages = forAllSystems (pkgs: rec {
        cfetch = pkgs.callPackage ./nix/package.nix { src = self; };
        default = cfetch;
      });
      checks = forAllSystems (pkgs: {
        cfetch = self.packages.${pkgs.stdenv.hostPlatform.system}.cfetch;
      });
    };
}
