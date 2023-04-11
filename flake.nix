{
  inputs = {
    nixpkgs.url = "nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, flake-utils, fenix, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ fenix.overlays.default ];
        };
        toolchain = pkgs.fenix.toolchainOf {
          channel = "1.41";
          sha256 = "sha256-CtlU5P+v0ZJDzYlP4ZLA9Kg5kMEbSHbeYugnhCx0q0Q=";
        };
      in
      with pkgs; {
        devShells.default = mkShell {
          nativeBuildInputs = [
            (with toolchain; pkgs.fenix.combine [
              defaultToolchain
              rust-src
            ])
          ];
        };
      }
    )
  ;
}
