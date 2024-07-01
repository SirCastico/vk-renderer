//#include <VkBootstrap.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdio>

import renderer;

int main(){
    glfwSetErrorCallback([](int error, const char* description){
        std::fprintf(stderr, "Error: %s\n", description);
    });


    if(!glfwInit()){
        std::fprintf(stderr, "glfw failed to init\n");
        return 1;
    }

    uint32_t width=800, height=600;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWwindow *window = glfwCreateWindow(width, height, "cool window", nullptr, nullptr);
    if(!window){
        std::fprintf(stderr, "glfw failed to create window\n");
        return 1;
    }

    renderer::RenderContext ctx{window,width,height,true};

    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
