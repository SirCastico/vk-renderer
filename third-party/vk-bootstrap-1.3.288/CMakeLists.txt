cmake_minimum_required (VERSION 3.25)

find_package(Vulkan REQUIRED)

add_library(VkBootstrap src/VkBootstrap.cpp src/VkBootstrap.h src/VkBootstrapDispatch.h)

target_include_directories(VkBootstrap PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src/)

target_link_libraries(VkBootstrap PRIVATE ${CMAKE_DL_LIBS} Vulkan::Vulkan)

