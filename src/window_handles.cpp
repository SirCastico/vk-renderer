module;

#include <vulkan/vulkan_core.h>
#include "window_handles_header.h"

#define VKRETURNIF(res) if(res!=VK_SUCCESS) return res

export module window_handles;


export enum WindowPlatform : uint8_t{
    WH_WINDOWS,
    WH_WAYLAND,
    WH_XLIB,
    WH_XCB,
};

export struct Win32Handles{
    void *hwnd;
    void *hinstance;
};

export struct XCBHandles{
    void *connection;
    uint32_t window;
};

export struct XlibHandles{
    void *display;
    unsigned long window;
};

export struct WaylandHandles{
    void *display, *surface;
};

export struct WindowHandles{
    WindowPlatform platform;
    union{
        Win32Handles win32_h;
        XCBHandles xcb_h;
        XlibHandles xlib_h;
        WaylandHandles wayland_h;
    }data;
};

export const char*const windowhandles_get_vk_extension(const WindowHandles *h);

export VkResult windowhandles_create_surface(VkInstance instance, const WindowHandles *handle, VkSurfaceKHR *out_surf);

module :private;

const char*const windowhandles_get_vk_extension(const WindowHandles *h){
    switch (h->platform) {
        case WH_WINDOWS:
            return VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
        case WH_XCB:
            return VK_KHR_XCB_SURFACE_EXTENSION_NAME;
        case WH_XLIB:
            return VK_KHR_XLIB_SURFACE_EXTENSION_NAME;
        case WH_WAYLAND:
            return VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME;
        default:
            return NULL;
    }
}

VkResult windowhandles_create_surface(VkInstance instance, const WindowHandles *handle, VkSurfaceKHR *out_surf){
    VkSurfaceKHR vk_surface;
    switch (handle->platform) {
        case WH_WINDOWS:{
            VkWin32SurfaceCreateInfoKHR sinfo = {
                .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                .pNext = NULL,
                .flags = 0,
                .hinstance = (HINSTANCE)handle->data.win32_h.hinstance,
                .hwnd = (HWND)handle->data.win32_h.hwnd,
            };
            PFN_vkCreateWin32SurfaceKHR p = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateWin32SurfaceKHR");
            if(!p){
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            VkResult r = p(instance, &sinfo, NULL, &vk_surface);
            VKRETURNIF(r);
            break;
        }

        case WH_XCB: {
            VkXcbSurfaceCreateInfoKHR sinfo = {
                .sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                .pNext = NULL,
                .flags = 0,
                .connection = (xcb_connection_t *)handle->data.xcb_h.connection,
                .window = (xcb_window_t)handle->data.xcb_h.window
            };
            PFN_vkCreateXcbSurfaceKHR p = (PFN_vkCreateXcbSurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateXcbSurfaceKHR");
            if(!p){
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            VkResult r = p(instance, &sinfo, NULL, &vk_surface);
            VKRETURNIF(r);
            break;
        }
        case WH_XLIB: {
            VkXlibSurfaceCreateInfoKHR sinfo = {
                .sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                .pNext = NULL,
                .flags = 0,
                .dpy = (Display *)handle->data.xlib_h.display,
                .window = (Window)handle->data.xlib_h.window,
            };
            PFN_vkCreateXlibSurfaceKHR p = (PFN_vkCreateXlibSurfaceKHR)vkGetInstanceProcAddr(instance, "vkCreateXlibSurfaceKHR");
            if(!p){
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            VkResult r = p(instance, &sinfo, NULL, &vk_surface);
            VKRETURNIF(r);
            break;
        }
        case WH_WAYLAND: {
            VkWaylandSurfaceCreateInfoKHR sinfo = {
                .sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR,
                .pNext = NULL,
                .flags = 0,
                .display = (struct wl_display*)handle->data.wayland_h.display,
                .surface = (struct wl_surface*)handle->data.wayland_h.surface
            };
            PFN_vkCreateWaylandSurfaceKHR p = (PFN_vkCreateWaylandSurfaceKHR)vkGetInstanceProcAddr(instance, "PFN_vkCreateWaylandSurfaceKHR");
            if(!p){
                return VK_ERROR_EXTENSION_NOT_PRESENT;
            }
            VkResult r = p(instance, &sinfo, NULL, &vk_surface);
            VKRETURNIF(r);
            break;
        }
        default:
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    
    *out_surf = vk_surface;
    return VK_SUCCESS;
}
