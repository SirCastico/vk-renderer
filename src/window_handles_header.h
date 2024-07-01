#ifndef _WHHEADERASSDASDSD_
#define _WHHEADERASSDASDSD_

#include <wchar.h>

typedef unsigned long DWORD;
typedef const wchar_t* LPCWSTR;
typedef void* HANDLE;
typedef struct HINSTANCE__* HINSTANCE;
typedef struct HWND__* HWND;
typedef struct HMONITOR__* HMONITOR;
typedef struct _SECURITY_ATTRIBUTES SECURITY_ATTRIBUTES;
#include <vulkan/vulkan_win32.h>

typedef uint32_t xcb_visualid_t;
typedef struct xcb_connection_t xcb_connection_t;
typedef uint32_t xcb_window_t;
#include <vulkan/vulkan_xcb.h>

typedef struct _XDisplay Display;

//NOTE: x11 header has some ifdefs for different unsigned long size, defining like this may be wrong
typedef unsigned long Window; 
typedef unsigned long VisualID;
#include <vulkan/vulkan_xlib.h>
#include <vulkan/vulkan_wayland.h>

#endif
