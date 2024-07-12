;; What follows is a "manifest" equivalent to the command line you gave.
;; You can store it in a file that you may then pass to any 'guix' command
;; that accepts a '--manifest' (or '-m') option.

(specifications->manifest
  (list 
        "mesa"
        "vulkan-loader"
        "vulkan-headers"
        "vulkan-tools"
        "vulkan-validationlayers"
        "neovim"
        "cmake"
        "ninja"
        "clang-toolchain"
        "sdl2"
        "libxt"
        "libxrandr"
        "libx11"
        "libxinerama"
        "wayland"
        "wayland-protocols"))
