let
    nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
    pkgs = import nixpkgs { 
        config = {}; 
        overlays = []; 
    };
    cstdenv = pkgs.llvmPackages_18.stdenv;
in
{
    vk-test-app = pkgs.callPackage ./vk-test-app.nix {
        stdenv = cstdenv;
        clang-tools = pkgs.clang-tools_18.override {stdenv = cstdenv;};
        ninja = pkgs.ninja;
    };
}
