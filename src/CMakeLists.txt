cmake_minimum_required (VERSION 3.25)

find_package(Vulkan REQUIRED)
#find_package(vk-bootstrap REQUIRED)
find_package(SDL2 REQUIRED)
#find_package(VulkanMemoryAllocator REQUIRED)

add_executable(test main.cpp window_handles.cpp renderer.cpp)

#target_sources(test 
#    PUBLIC
#    FILE_SET CXX_MODULES FILES
#            renderer.cpp
#            window_handles.cpp)

target_link_libraries(
    test PRIVATE 
        Vulkan::Vulkan VkBootstrap SDL2::SDL2 cgltf HandmadeMath)
