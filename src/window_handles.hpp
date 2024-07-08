#pragma once

#include <cstdint>
#include <vulkan/vulkan_core.h>

enum WindowPlatform : uint8_t{
    WH_WINDOWS,
    WH_WAYLAND,
    WH_XLIB,
    WH_XCB,
};

struct Win32Handles{
    void *hwnd;
    void *hinstance;
};

struct XCBHandles{
    void *connection;
    uint32_t window;
};

struct XlibHandles{
    void *display;
    unsigned long window;
};

struct WaylandHandles{
    void *display, *surface;
};

struct WindowHandles{
    WindowPlatform platform;
    union{
        Win32Handles win32_h;
        XCBHandles xcb_h;
        XlibHandles xlib_h;
        WaylandHandles wayland_h;
    }data;
};

const char*const windowhandles_get_vk_extension(const WindowHandles *h);

// TODO: replace vkresult return
VkResult windowhandles_create_surface(VkInstance instance, const WindowHandles *handle, VkSurfaceKHR *out_surf);
