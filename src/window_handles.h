#ifndef ASDASDASD
#define ASDASDASD

#include <stdint.h>
#include <vulkan/vulkan_core.h>

typedef enum WindowPlatform : uint8_t{
    WH_WINDOWS,
    WH_WAYLAND,
    WH_XLIB,
    WH_XCB,
}WindowPlatform;

typedef struct Win32Handles{
    void *hwnd;
    void *hinstance;
}Win32Handles;

typedef struct XCBHandles{
    void *connection;
    uint32_t window;
}XCBHandles;

typedef struct XlibHandles{
    void *display;
    unsigned long window;
}XlibHandles;

typedef struct WaylandHandles{
    void *display, *surface;
}WaylandHandles;

typedef struct WindowHandles{
    WindowPlatform platform;
    union{
        Win32Handles win32_h;
        XCBHandles xcb_h;
        XlibHandles xlib_h;
        WaylandHandles wayland_h;
    }data;
}WindowHandles;


//VkResult windowhandles_create_surface(VkInstance instance, const WindowHandles *handle, VkSurfaceKHR *out_surf);

//const char*const windowhandles_get_vk_extension(const WindowHandles *h);

#endif
