#pragma once

#include "window_handles.hpp"
#include <stdint.h>

#ifdef __cplusplus
extern "C"{
#endif

struct RenderContext;

RenderContext* renderer_new(const WindowHandles *wh, uint32_t width, uint32_t height, bool validation_layers);

void renderer_draw(RenderContext*);

void renderer_resize(RenderContext *ctx, uint32_t width, uint32_t height);

bool renderer_is_resize_req(RenderContext *ctx);

void renderer_destroy(RenderContext*);


#ifdef __cplusplus
}
#endif
