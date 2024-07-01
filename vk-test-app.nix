{
    stdenv,
    ninja,
    cmake,
    vulkan-headers,
    vulkan-loader,
    vulkan-tools-lunarg,
    clang-tools,
    #vk-bootstrap,
    glfw,
    xorg,
    wayland,
    wayland-protocols,
    neovim,
}:

stdenv.mkDerivation {
    pname = "vk-app-test";
    version = "0.1";
    src = ./.;

    # i don't know how to separate dev tools
    nativeBuildInputs = [ cmake vulkan-tools-lunarg clang-tools ninja neovim];

    buildInputs = [
        vulkan-headers
        vulkan-loader
        #vk-bootstrap
        glfw
        xorg.libXt 
        xorg.libX11 
        xorg.libXinerama
        wayland
        wayland-protocols
    ];
}

#pkgs.mkShell{
#    
#    packages = with pkgs; [
#        pkg-config
#        clang-tools
#        cmake
#        vulkan-headers
#        vulkan-loader
#        vulkan-tools-lunarg
#        #vk-bootstrap
#        glfw
#    ];
#}
