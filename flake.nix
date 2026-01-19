{
  description = "Simple Rust development environment template";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    rust-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default];
      };

      # Native dependencies if needed
      buildInputs = with pkgs; [
        duckdb
        pkg-config
      ];
    in {

      devShells.default = pkgs.mkShell {

        packages = with pkgs; [
          (rust-bin.stable.latest.default.override {
            extensions = ["rust-src" "rust-analyzer"];
          })
          # Python with ML packages for training
          (python3.withPackages (ps: with ps; [
            numpy
            pandas
            scikit-learn
            xgboost
            onnx
            onnxmltools
            plotly
            matplotlib
          ]))
        ];

        inherit buildInputs;

        # Basic environment setup
        RUST_BACKTRACE = 1;

        shellHook = ''
          echo "🦀 Rust development environment loaded"
          echo "Rust version: $(rustc --version)"
          echo "Cargo version: $(cargo --version)"
        '';
      };

      # Build package
      packages.default = pkgs.rustPlatform.buildRustPackage {
        pname = "my-rust-project";
        version = "0.1.0";
        src = ./.;

        cargoLock = {
          lockFile = ./Cargo.lock;
        };
      };
    });
}
