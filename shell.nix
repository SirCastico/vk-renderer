let
    nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
    pkgs = import nixpkgs { 
        config = {}; 
        overlays = []; 
    };
    cstdenv = pkgs.llvmPackages_18.stdenv;
in

pkgs.mkShell.override {stdenv=cstdenv;} {
    
    nativeBuildInputs = with pkgs; [ 
        cmake 
        vulkan-tools-lunarg 
        vulkan-tools 
        clang-tools_18
        ninja 
        neovim
        wayland-utils
    ];

    buildInputs = with pkgs; [
        mesa
        vulkan-headers
        vulkan-loader
        SDL2
        xorg.libXt 
        xorg.libXrandr 
        xorg.libX11 
        xorg.libXinerama
        wayland
        wayland-protocols
    ];

}
