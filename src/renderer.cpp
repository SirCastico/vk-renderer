module;

#include <cstdio>
#include <vulkan/vulkan_core.h>
//#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <utility>

export module renderer;

import window_handles;

namespace renderer {

    export class RenderInstance{
        public:
            vkb::Instance vkb_instance;

        RenderInstance(bool validation_layers=false){
            vkb::InstanceBuilder builder;

            auto inst_ret = builder.set_app_name("cool vulkan app")
                .request_validation_layers(validation_layers)
                .use_default_debug_messenger()
                .require_api_version(1,3,0)
                .build();

            this->vkb_instance = inst_ret.value(); // may throw
        }

        ~RenderInstance(){
            if(this->vkb_instance.instance!=VK_NULL_HANDLE){
                vkb::destroy_debug_utils_messenger(this->vkb_instance.instance, this->vkb_instance.debug_messenger);
                vkDestroyInstance(this->vkb_instance.instance, nullptr);
            }
        }

        RenderInstance(RenderInstance&& other){
            this->vkb_instance = std::move(other.vkb_instance);
            other.vkb_instance.instance = VK_NULL_HANDLE;
        }

        RenderInstance(const RenderInstance&) = delete;

        RenderInstance& operator=(RenderInstance&& other){
            this->vkb_instance = std::move(other.vkb_instance);
            other.vkb_instance.instance = VK_NULL_HANDLE;
            return *this;
        }

        RenderInstance& operator=(const RenderInstance&) = delete;

        VkInstance instance() const {
            return this->vkb_instance.instance;
        }
    };

    // TODO: init surface with instance and windowhandle info

    vkb::Instance make_instance_wh(const WindowHandles &wh, bool validation_layers){
        vkb::InstanceBuilder builder;

        const char*const instance_extensions[] = {
            VK_KHR_SURFACE_EXTENSION_NAME,
            windowhandles_get_vk_extension(&wh)
        };

        auto inst_ret = builder.set_app_name("cool vulkan app")
            .enable_extensions(2,instance_extensions)
            .request_validation_layers(validation_layers)
            .use_default_debug_messenger()
            .require_api_version(1,3,0)
            .build();

        if (!inst_ret.has_value()){
            throw std::runtime_error{inst_ret.error().message()};
        }
        return inst_ret.value();
    }

    export class RenderContext{

        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debug_messenger;
        VkSurfaceKHR surface;
        VkPhysicalDevice phys_dev;
        VkDevice dev;
    
        VkSwapchainKHR swapchain;
        VkExtent2D swp_extent;
        VkSurfaceFormatKHR format;

        std::vector<VkImage> swp_images;
        std::vector<VkImageView> swp_image_views;

        RenderContext() = default;


        public:

        static RenderContext make(const WindowHandles &wh, uint32_t width, uint32_t height, bool validation_layers){ 
            vkb::Instance vkb_instance = make_instance_wh(wh, validation_layers); // may throw
                                                   
            VkSurfaceKHR surface;
            VkResult err = windowhandles_create_surface(vkb_instance.instance,&wh,&surface);
            if(err!=VK_SUCCESS){
                throw std::runtime_error("failed to create surface");
            }

            //vulkan 1.3 features
            VkPhysicalDeviceVulkan13Features features13{
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                .synchronization2 = true,
                .dynamicRendering = true,
            };

            //vulkan 1.2 features
            VkPhysicalDeviceVulkan12Features features12{ 
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                .drawIndirectCount = true,
                .descriptorIndexing = true,
                .bufferDeviceAddress = true,
            };

            VkPhysicalDeviceVulkan11Features features11 = {
                .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                .shaderDrawParameters = VK_TRUE,
            };

            VkPhysicalDeviceFeatures features10 = {
                .multiDrawIndirect = VK_TRUE
            };

            //use vkbootstrap to select a gpu. 
            vkb::PhysicalDeviceSelector selector{ vkb_instance };
            vkb::PhysicalDevice physicalDevice = selector
                .set_minimum_version(1, 3)
                .set_required_features_13(features13)
                .set_required_features_12(features12)
                .set_required_features_11(features11)
                .set_required_features(features10)
                .set_surface(surface)
                .select()
                .value(); // may throw

            //create the final vulkan device
            vkb::DeviceBuilder deviceBuilder{ physicalDevice };

            vkb::Device vkbDevice = deviceBuilder.build().value();

            // Create swapchain
            vkb::SwapchainBuilder swapchainBuilder{ physicalDevice,vkbDevice,surface };

            VkSurfaceFormatKHR swp_format = { .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

            vkb::Swapchain vkbSwapchain = swapchainBuilder
                .set_desired_format(swp_format)
                //use vsync present mode
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // vsync
                .set_desired_extent(width, height)
                .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                .build()
                .value();

            RenderContext ctx{};

            ctx.instance = vkb_instance.instance;
            ctx.debug_messenger = vkb_instance.debug_messenger,
            ctx.surface = surface;
            ctx.phys_dev = physicalDevice.physical_device;
            ctx.dev = vkbDevice.device;
            ctx.format = swp_format;

            ctx.swp_extent = vkbSwapchain.extent;
            ctx.swapchain = vkbSwapchain.swapchain;
            ctx.swp_images = vkbSwapchain.get_images().value();
            ctx.swp_image_views = vkbSwapchain.get_image_views().value();

            return ctx;
        }

        ~RenderContext(){
            if(this->instance != VK_NULL_HANDLE){
                vkDestroySwapchainKHR(this->dev, this->swapchain, nullptr);
                for(int i=0;i<this->swp_image_views.size();++i){
                    vkDestroyImageView(this->dev,this->swp_image_views[i],nullptr);
                }

                vkDestroySurfaceKHR(this->instance, this->surface, nullptr);
                vkDestroyDevice(this->dev,nullptr);

                //vkb::destroy_debug_utils_messenger(this->instance, this->debug_messenger);
                vkDestroyInstance(this->instance, nullptr);
            }
        }

        RenderContext(RenderContext&& other){
            this->instance = other.instance;
            this->debug_messenger = other.debug_messenger;
            this->surface = other.surface;
            this->phys_dev = other.phys_dev;
            this->dev = other.dev;
        
            this->swapchain = other.swapchain;
            this->swp_extent = other.swp_extent;
            this->format = other.format;

            this->swp_images = std::move(other.swp_images);
            this->swp_image_views = std::move(other.swp_image_views);

            other.instance = VK_NULL_HANDLE;
        }

        RenderContext& operator=(RenderContext&& other){
            this->instance = other.instance;
            this->debug_messenger = other.debug_messenger;
            this->surface = other.surface;
            this->phys_dev = other.phys_dev;
            this->dev = other.dev;
        
            this->swapchain = other.swapchain;
            this->swp_extent = other.swp_extent;
            this->format = other.format;

            this->swp_images = std::move(other.swp_images);
            this->swp_image_views = std::move(other.swp_image_views);

            other.instance = VK_NULL_HANDLE;
            return *this;
        }

        RenderContext(const RenderContext&) = delete;

        RenderContext& operator=(const RenderContext&) = delete;
    };
}
