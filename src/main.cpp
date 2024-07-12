
#include "renderer.h"
#include "window_handles.hpp"

#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>

#include <stdexcept>
#include <chrono>
#include <thread>

#include <cstdint>
#include <cstdio>


WindowHandles get_sdl2_window_handles(SDL_Window *window){

    SDL_SysWMinfo wm_info;
    SDL_VERSION(&wm_info.version);
    if(!SDL_GetWindowWMInfo(window, &wm_info)){
        throw std::runtime_error{"failed to get window info"};
    }
    const char*const video_driver = SDL_GetCurrentVideoDriver();

    if(strcmp(video_driver, "x11")==0){
        return WindowHandles{
            .platform = WH_XLIB,
            .data = {
                .xlib_h = {
                    .display = wm_info.info.x11.display,
                    .window = wm_info.info.x11.window,
                }
            }
        };
    } else if(strcmp(video_driver, "wayland")==0){
        return WindowHandles{
            .platform = WH_WAYLAND,
            .data = {
                .wayland_h = {
                    .display = wm_info.info.wl.display,
                    .surface = wm_info.info.wl.surface,
                }
            }
        };
        
    } else {
        throw std::runtime_error{"no window handle"};
    }
}

int main(){
    SDL_Init(SDL_INIT_VIDEO);

    uint32_t width=800, height=600;

    SDL_Window *window = SDL_CreateWindow(
            "cool",
            width,
            height,
            width,
            height,
            SDL_WINDOW_VULKAN
    );

    if(!window){
        std::fprintf(stderr, "failed to create window: %s\n", SDL_GetError());
        return 1;
    }

    WindowHandles wh = get_sdl2_window_handles(window);

    auto *ctx = renderer_new(&wh,width,height,true);

    bool quit=false,stop_rendering=false;
    while(!quit){
        SDL_Event event;
        while(SDL_PollEvent(&event)){
            switch (event.type) {
            case SDL_QUIT:
                quit=true;
                break;
            case SDL_WINDOWEVENT:
                switch (event.window.event){
                    case SDL_WINDOWEVENT_MINIMIZED:
                        stop_rendering=true;
                        break;
                    case SDL_WINDOWEVENT_RESTORED:
                        stop_rendering=false;
                        break;
                }
                break;
            }
        }
        renderer_draw(ctx);
        if(stop_rendering){
            std::puts("sleeping");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    renderer_destroy(ctx);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
