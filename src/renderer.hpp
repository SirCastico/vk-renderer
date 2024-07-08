#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vulkan/vulkan_core.h>
//#include <vulkan/vulkan.h>
#include <VkBootstrap.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <utility>


#include "window_handles.hpp"

namespace renderer {

    class RenderInstance{
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

    
    class RenderContext{

        struct FrameData{
            VkCommandPool cmd_pool;
            VkSemaphore swp_semaphore, render_semaphore;
            VkFence render_fence;
        };

        VkInstance instance;
        VkDebugUtilsMessengerEXT debug_messenger;
        VkSurfaceKHR surface;
        VkPhysicalDevice phys_dev;
        VkDevice dev;
        VkQueue g_queue;
        uint32_t g_queue_fam;
    
        VkSwapchainKHR swapchain;
        VkExtent2D swp_extent;
        VkSurfaceFormatKHR format;

        std::vector<VkImage> swp_images;
        std::vector<VkImageView> swp_image_views;

        static constexpr uint32_t FRAME_OVERLAP = 2;
        using FrameDataArray = std::array<FrameData, FRAME_OVERLAP>;

        FrameDataArray frames;
        uint64_t frame_count;

        RenderContext() = default;

        FrameData& get_current_frame(){
            return this->frames[this->frame_count % FRAME_OVERLAP];
        }

        public:

        // TODO: init surface with instance and windowhandle info
    
        static RenderContext make(const WindowHandles &wh, uint32_t width, uint32_t height, bool validation_layers){ 
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
            vkb::Instance vkb_instance = inst_ret.value(); // may throw
                                                   
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

            vkb::Device vkb_device = deviceBuilder.build().value();

            // Create swapchain
            vkb::SwapchainBuilder swapchainBuilder{ physicalDevice,vkb_device,surface };

            VkSurfaceFormatKHR swp_format = { .format = VK_FORMAT_B8G8R8A8_UNORM, .colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };

            vkb::Swapchain vkbSwapchain = swapchainBuilder
                .set_desired_format(swp_format)
                //use vsync present mode
                .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR) // vsync
                .set_desired_extent(width, height)
                .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                .build()
                .value();


            VkQueue g_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
            uint32_t g_queue_fam = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

            // frame data
            FrameDataArray frames;

            //create a command pool for commands submitted to the graphics queue.
            VkCommandPoolCreateInfo commandPoolInfo = {};
            commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolInfo.pNext = nullptr;
            //commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            commandPoolInfo.queueFamilyIndex = g_queue_fam;
            
            VkFenceCreateInfo fence_info = {
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_FENCE_CREATE_SIGNALED_BIT,
            };
            VkSemaphoreCreateInfo sem_info = {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            };

            for (int i = 0; i < FRAME_OVERLAP; i++) {
                VkResult res = vkCreateCommandPool(vkb_device.device, &commandPoolInfo, nullptr, &frames[i].cmd_pool);
                if(res!=VK_SUCCESS) throw std::runtime_error{"failed to create command pool"};



                // sync structs
                res = vkCreateFence(vkb_device.device, &fence_info, nullptr, &frames[i].render_fence);
                if(res!=VK_SUCCESS) throw std::runtime_error{"failed to create frame fence"};

                res = vkCreateSemaphore(vkb_device.device, &sem_info, nullptr, &frames[i].swp_semaphore);
                if(res!=VK_SUCCESS) throw std::runtime_error{"failed to create frame sem"};

                res = vkCreateSemaphore(vkb_device.device, &sem_info, nullptr, &frames[i].render_semaphore);
                if(res!=VK_SUCCESS) throw std::runtime_error{"failed to create frame sem"};

            }

            RenderContext ctx{};

            ctx.instance = vkb_instance.instance;
            ctx.debug_messenger = vkb_instance.debug_messenger,
            ctx.surface = surface;
            ctx.phys_dev = physicalDevice.physical_device;
            ctx.dev = vkb_device.device;
            ctx.g_queue = g_queue;
            ctx.g_queue_fam = g_queue_fam;

            ctx.format = swp_format;
            ctx.swp_extent = vkbSwapchain.extent;
            ctx.swapchain = vkbSwapchain.swapchain;
            ctx.swp_images = vkbSwapchain.get_images().value();
            ctx.swp_image_views = vkbSwapchain.get_image_views().value();
            ctx.frames = frames;
            ctx.frame_count = 0;

            return ctx;
        }

        ~RenderContext(){
            if(this->instance != VK_NULL_HANDLE){
                for(auto& frame : this->frames){
                    vkDestroyCommandPool(this->dev, frame.cmd_pool, nullptr);
                    vkDestroyFence(this->dev, frame.render_fence, nullptr);
                    vkDestroySemaphore(this->dev, frame.render_semaphore, nullptr);
                    vkDestroySemaphore(this->dev, frame.swp_semaphore, nullptr);
                }
                vkDestroySwapchainKHR(this->dev, this->swapchain, nullptr);
                for(int i=0;i<this->swp_image_views.size();++i){
                    vkDestroyImageView(this->dev,this->swp_image_views[i],nullptr);
                }

                vkDestroyDevice(this->dev,nullptr);
                vkDestroySurfaceKHR(this->instance, this->surface, nullptr);

                //vkb::destroy_debug_utils_messenger(this->instance, this->debug_messenger);
                vkDestroyInstance(this->instance, nullptr);
            }
        }

        // TODO
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

        void draw(){
            FrameData &frame = this->get_current_frame();
            VkResult res = vkWaitForFences(this->dev, 1, &frame.render_fence, true, 1000000000);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed wait for fence"};
            res = vkResetFences(this->dev, 1, &frame.render_fence);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed fence reset"};

            uint32_t swp_img_ind;
            res = vkAcquireNextImageKHR(
                    this->dev, 
                    this->swapchain, 
                    1000000000, 
                    frame.swp_semaphore,
                    nullptr,
                    &swp_img_ind);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed img acquire"};


            res = vkResetCommandPool(this->dev, frame.cmd_pool, 0);

            // allocate the default command buffer that we will use for rendering
            VkCommandBufferAllocateInfo cmdAllocInfo = {};
            cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAllocInfo.pNext = nullptr;
            cmdAllocInfo.commandPool = frame.cmd_pool;
            cmdAllocInfo.commandBufferCount = 1;
            cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

            VkCommandBuffer cmd_buffer;
            res = vkAllocateCommandBuffers(this->dev, &cmdAllocInfo, &cmd_buffer);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed to create command buffer"};

            VkCommandBufferBeginInfo cmd_info = {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            // begin
            res = vkBeginCommandBuffer(cmd_buffer, &cmd_info);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed to begin command buffer"};

            transition_image(
                    cmd_buffer,
                    this->swp_images[swp_img_ind],
                    VK_IMAGE_LAYOUT_UNDEFINED,
                    VK_IMAGE_LAYOUT_GENERAL);

            //make a clear-color from frame number. This will flash with a 120 frame period.
            VkClearColorValue clearValue;
            float flash = std::abs(std::sin(this->frame_count / 120.f));
            clearValue = { { 0.0f, 0.0f, flash, 1.0f } };

            VkImageSubresourceRange clearRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            };

            //clear image
            vkCmdClearColorImage(cmd_buffer, this->swp_images[swp_img_ind], VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

            //make the swapchain image into presentable mode
            transition_image(cmd_buffer, this->swp_images[swp_img_ind],VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

            //finalize the command buffer (we can no longer add commands, but it can now be executed)
            res = vkEndCommandBuffer(cmd_buffer);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed to end cmd buffer"};

            //prepare the submission to the queue. 
            //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
            //we will signal the _renderSemaphore, to signal that rendering has finished
            VkCommandBufferSubmitInfo cmdinfo = command_buffer_submit_info(cmd_buffer);	
            
            VkSemaphoreSubmitInfo waitInfo = semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,frame.swp_semaphore);
            VkSemaphoreSubmitInfo signalInfo = semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, frame.render_semaphore);	
            
            VkSubmitInfo2 submit = submit_info(&cmdinfo,&signalInfo,&waitInfo);	

            //submit command buffer to the queue and execute it.
            // _renderFence will now block until the graphic commands finish execution
            res = vkQueueSubmit2(this->g_queue, 1, &submit, frame.render_fence);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed to submit cmds"};

            //prepare present
            // this will put the image we just rendered to into the visible window.
            // we want to wait on the _renderSemaphore for that, 
            // as its necessary that drawing commands have finished before the image is displayed to the user
            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            presentInfo.pNext = nullptr;
            presentInfo.pSwapchains = &this->swapchain;
            presentInfo.swapchainCount = 1;

            presentInfo.pWaitSemaphores = &frame.render_semaphore;
            presentInfo.waitSemaphoreCount = 1;

            presentInfo.pImageIndices = &swp_img_ind;

            res = vkQueuePresentKHR(this->g_queue, &presentInfo);
            if(res!=VK_SUCCESS) throw std::runtime_error{"failed to present"};

            //increase the number of frames drawn
            this->frame_count++;
        }

        private:
        static void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout cur_layout, VkImageLayout new_layout){

            VkImageMemoryBarrier2 imageBarrier {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};

            imageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            imageBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
            imageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            imageBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;

            imageBarrier.oldLayout = cur_layout;
            imageBarrier.newLayout = new_layout;

            VkImageAspectFlags aspectMask = 
                (new_layout == VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL) ?
                    VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

            imageBarrier.subresourceRange = {
                .aspectMask = aspectMask,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            };
            imageBarrier.image = image;

            VkDependencyInfo depInfo {};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.pNext = nullptr;

            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &imageBarrier;

            vkCmdPipelineBarrier2(cmd, &depInfo);
        }


        VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore)
        {
            VkSemaphoreSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            submitInfo.pNext = nullptr;
            submitInfo.semaphore = semaphore;
            submitInfo.stageMask = stageMask;
            submitInfo.deviceIndex = 0;
            submitInfo.value = 1;

            return submitInfo;
        }

        VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd)
        {
            VkCommandBufferSubmitInfo info{};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            info.pNext = nullptr;
            info.commandBuffer = cmd;
            info.deviceMask = 0;

            return info;
        }

        VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signalSemaphoreInfo,
            VkSemaphoreSubmitInfo* waitSemaphoreInfo)
        {
            VkSubmitInfo2 info = {};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            info.pNext = nullptr;

            info.waitSemaphoreInfoCount = waitSemaphoreInfo == nullptr ? 0 : 1;
            info.pWaitSemaphoreInfos = waitSemaphoreInfo;

            info.signalSemaphoreInfoCount = signalSemaphoreInfo == nullptr ? 0 : 1;
            info.pSignalSemaphoreInfos = signalSemaphoreInfo;

            info.commandBufferInfoCount = 1;
            info.pCommandBufferInfos = cmd;

            return info;
        }
    };
}
