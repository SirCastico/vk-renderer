#include "renderer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vulkan/vulkan_core.h>
#include <VkBootstrap.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <utility>

void abort_msg(const char*msg){
    std::puts(msg);
    std::abort();
}

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

namespace vkinit{
    VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent)
    {
        VkImageCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.pNext = nullptr;

        info.imageType = VK_IMAGE_TYPE_2D;

        info.format = format;
        info.extent = extent;

        info.mipLevels = 1;
        info.arrayLayers = 1;

        //for MSAA. we will not be using it by default, so default it to 1 sample per pixel.
        info.samples = VK_SAMPLE_COUNT_1_BIT;

        //optimal tiling, which means the image is stored on the best gpu format
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usageFlags;

        return info;
    }

    VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags)
    {
        // build a image-view for the depth image to use for rendering
        VkImageViewCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.pNext = nullptr;

        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.image = image;
        info.format = format;
        info.subresourceRange.baseMipLevel = 0;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount = 1;
        info.subresourceRange.aspectMask = aspectFlags;

        return info;
    }
}

namespace vkutils{
    enum class MemoryTypes : uint8_t{
        SHARED=0,
        DEVICE_LOCAL=1,
        HOST_COHERENT=2,
        HOST_CACHED=3,
        HOST_CACHED_COHERENT=4,
    };
    constexpr uint8_t MEMORY_TYPES_NUM = 5;

    struct DeviceMemory{
        VkPhysicalDeviceMemoryProperties properties;
        int16_t types[MEMORY_TYPES_NUM];

        DeviceMemory() = default;

        DeviceMemory(VkPhysicalDevice phdev){
            DeviceMemory dm;
            memset(dm.types, -1, sizeof(dm.types));
            vkGetPhysicalDeviceMemoryProperties(phdev, &dm.properties);

            for(int i=0;i<dm.properties.memoryTypeCount;++i){
                VkMemoryType type = dm.properties.memoryTypes[i];
                uint32_t shared_bits = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
                if((type.propertyFlags & shared_bits)==shared_bits){
                    if(dm.types[(uint8_t)MemoryTypes::SHARED]==-1){
                        dm.types[(uint8_t)MemoryTypes::SHARED] = i;
                        continue;
                    }
                }
                if(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT){
                    if(dm.types[(uint8_t)MemoryTypes::DEVICE_LOCAL]==-1){
                        dm.types[(uint8_t)MemoryTypes::DEVICE_LOCAL] = i;
                        continue;
                    }
                }
                uint32_t host_cached_coherent_bits =
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | 
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                if((type.propertyFlags & host_cached_coherent_bits)==host_cached_coherent_bits){
                    if(dm.types[(uint8_t)MemoryTypes::HOST_CACHED_COHERENT]==-1){
                        dm.types[(uint8_t)MemoryTypes::HOST_CACHED_COHERENT] = i;
                        continue;
                    }
                }
                uint32_t host_coherent_bits = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if((type.propertyFlags & host_coherent_bits)==host_coherent_bits){
                    if(dm.types[(uint8_t)MemoryTypes::HOST_COHERENT]==-1){
                        dm.types[(uint8_t)MemoryTypes::HOST_COHERENT] = i;
                        continue;
                    }
                }
                uint32_t host_cached_bits = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
                if((type.propertyFlags & host_cached_bits)==host_cached_bits){
                    if(dm.types[(uint8_t)MemoryTypes::HOST_CACHED]==-1){
                        dm.types[(uint8_t)MemoryTypes::HOST_CACHED] = i;
                        continue;
                    }
                }
            }
            *this = dm;
        }

        void print_device_memory(){
            printf("Device memory properties\n");
            for(int i=0;i<this->properties.memoryTypeCount;++i){
                VkMemoryType type = this->properties.memoryTypes[i];
                printf("type %d in heap %d:\n", i, type.heapIndex);
                if(type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                    printf("VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT\n");
                if(type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                    printf("VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT\n");
                if(type.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                    printf("VK_MEMORY_PROPERTY_HOST_COHERENT_BIT\n");
                if(type.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
                    printf("VK_MEMORY_PROPERTY_HOST_CACHED_BIT\n");
                if(type.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
                    printf("VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT\n");
                if(type.propertyFlags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
                    printf("VK_MEMORY_PROPERTY_PROTECTED_BIT\n");
                printf("\n");
            }
            for(int i=0;i<this->properties.memoryHeapCount;++i){
                printf("heap %d has size %ld\n", i, this->properties.memoryHeaps[i].size);
            }
            for(int i=0;i<MEMORY_TYPES_NUM;++i){
                printf("vulcn type: %d, vulkan type:%d\n", i, this->types[i]);
            }
        }
    };

    struct BoundImage{
        VkImage img;
        VkDeviceSize size,offset;

        void destroy(VkDevice dev){
            vkDestroyImage(dev, this->img, nullptr);
        }
    }; 

    struct BoundBuffer{
        VkImage img;
        VkDeviceSize size,offset;
    }; 

    struct BufferBAlloc{
        VkDeviceMemory dev_mem;
        VkDeviceSize cap, len;
        VkMemoryPropertyFlags properties;
    };

    struct ImageBAlloc{
        VkDeviceMemory dev_mem;
        VkDeviceSize cap, len;
        VkMemoryPropertyFlags properties;

        static ImageBAlloc make(VkDevice dev, DeviceMemory &mem, MemoryTypes type, VkDeviceSize size){
            VkMemoryAllocateInfo info = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = NULL,
                .allocationSize = size,
                .memoryTypeIndex = (uint32_t)mem.types[(uint8_t)type]
            };
            VkDeviceMemory dev_mem;
            VkResult res = vkAllocateMemory(dev, &info, NULL, &dev_mem);// TODO
            if(res!=VK_SUCCESS) abort_msg("failed to alloc");

            ImageBAlloc b{
                .dev_mem = dev_mem,
                .cap = size,
                .len = 0,
                .properties = mem.properties.memoryTypes->propertyFlags,
            };

            return b;
        }
       
        BoundImage bind_image(VkDevice dev, VkImage img){
            VkMemoryRequirements mem_req;
            vkGetImageMemoryRequirements(dev, img, &mem_req);

            if((mem_req.memoryTypeBits & this->properties)==0){
                abort_msg("wrong memory"); // TODO
            }

            VkDeviceSize offset = mem_req.alignment - this->len % mem_req.alignment;

            if(this->len + offset + mem_req.size > this->cap){
                abort_msg("not enough memory"); // TODO
            }

            vkBindImageMemory(dev, img, this->dev_mem, this->len+offset);

            BoundImage bimg = {
                .img = img,
                .size = mem_req.size,
                .offset = this->len+offset,
            };

            this->len += offset + mem_req.size;

            return bimg;
        }

        void destroy(VkDevice dev){
            vkFreeMemory(dev, this->dev_mem, nullptr);
        }
    };


    void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize)
    {
        VkImageBlit2 blitRegion{ .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2, .pNext = nullptr };

        blitRegion.srcOffsets[1].x = srcSize.width;
        blitRegion.srcOffsets[1].y = srcSize.height;
        blitRegion.srcOffsets[1].z = 1;

        blitRegion.dstOffsets[1].x = dstSize.width;
        blitRegion.dstOffsets[1].y = dstSize.height;
        blitRegion.dstOffsets[1].z = 1;

        blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.srcSubresource.baseArrayLayer = 0;
        blitRegion.srcSubresource.layerCount = 1;
        blitRegion.srcSubresource.mipLevel = 0;

        blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blitRegion.dstSubresource.baseArrayLayer = 0;
        blitRegion.dstSubresource.layerCount = 1;
        blitRegion.dstSubresource.mipLevel = 0;

        VkBlitImageInfo2 blitInfo{ .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2, .pNext = nullptr };
        blitInfo.dstImage = destination;
        blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blitInfo.srcImage = source;
        blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blitInfo.filter = VK_FILTER_LINEAR;
        blitInfo.regionCount = 1;
        blitInfo.pRegions = &blitRegion;

        vkCmdBlitImage2(cmd, &blitInfo);
    }
}

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

    vkutils::ImageBAlloc img_allocator;
    vkutils::BoundImage draw_img;
    VkExtent2D draw_img_extent;

    VkImage *swp_images;
    VkImageView *swp_image_views;
    uint32_t swp_img_num;

    static constexpr uint32_t FRAME_OVERLAP = 2;
    using FrameDataArray = std::array<FrameData, FRAME_OVERLAP>;

    FrameDataArray frames;
    uint64_t frame_count;


    FrameData& get_current_frame(){
        return this->frames[this->frame_count % FRAME_OVERLAP];
    }

    public:
    RenderContext() = default;

    // TODO: allow surface to be created externally

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
            abort_msg(inst_ret.error().message().c_str());
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

        uint32_t swp_image_count;
        err = vkGetSwapchainImagesKHR(vkb_device.device, vkbSwapchain.swapchain, &swp_image_count, NULL);
        if(err!=VK_SUCCESS) abort_msg("failed to count swapchain images");

        /// The swapchain images.
        //VkImage *swp_images = (VkImage*)malloc(image_count*sizeof(VkImage));
        VkImage *swp_images = new VkImage[swp_image_count];

        err = vkGetSwapchainImagesKHR(vkb_device.device, vkbSwapchain.swapchain, &swp_image_count, swp_images);
        if(err!=VK_SUCCESS) abort_msg("failed to read swapchain images");

        VkImageView *swp_image_views = new VkImageView[swp_image_count];

        for (size_t i = 0; i < swp_image_count; i++)
        {
            // Create an image view which we can render into.
            VkImageViewCreateInfo view_info={VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view_info.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format                      = vkbSwapchain.image_format;
            view_info.image                       = swp_images[i];
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.layerCount = 1;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.components.r                = VK_COMPONENT_SWIZZLE_R;
            view_info.components.g                = VK_COMPONENT_SWIZZLE_G;
            view_info.components.b                = VK_COMPONENT_SWIZZLE_B;
            view_info.components.a                = VK_COMPONENT_SWIZZLE_A;

            err = vkCreateImageView(vkb_device.device, &view_info, NULL, &swp_image_views[i]);
            if(err!=VK_SUCCESS) abort_msg("failed to create swapchain image view");
        }

        VkQueue g_queue = vkb_device.get_queue(vkb::QueueType::graphics).value();
        uint32_t g_queue_fam = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

        // mem info
        vkutils::DeviceMemory dev_mem{vkb_device.physical_device};
        dev_mem.print_device_memory();

        // draw image
        VkExtent3D draw_img_extent = {
            .width = width,
            .height = height,
            .depth = 1
        };

        VkImageCreateInfo img_cinfo = vkinit::image_create_info(
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | 
                    VK_IMAGE_USAGE_STORAGE_BIT  |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                draw_img_extent        
        );

        VkImage draw_img;
        vkCreateImage(vkb_device.device, &img_cinfo, nullptr, &draw_img);

        auto img_alloc = vkutils::ImageBAlloc::make(
                vkb_device.device, 
                dev_mem, 
                vkutils::MemoryTypes::DEVICE_LOCAL, 
                draw_img_extent.width*draw_img_extent.height*8*2);

        vkutils::BoundImage bound_img = img_alloc.bind_image(vkb_device.device, draw_img);

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
            if(res!=VK_SUCCESS) abort_msg("failed to create command pool");

            // sync structs
            res = vkCreateFence(vkb_device.device, &fence_info, nullptr, &frames[i].render_fence);
            if(res!=VK_SUCCESS) abort_msg("failed to create frame fence");

            res = vkCreateSemaphore(vkb_device.device, &sem_info, nullptr, &frames[i].swp_semaphore);
            if(res!=VK_SUCCESS) abort_msg("failed to create frame sem");

            res = vkCreateSemaphore(vkb_device.device, &sem_info, nullptr, &frames[i].render_semaphore);
            if(res!=VK_SUCCESS) abort_msg("failed to create frame sem");

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
        ctx.swp_images = swp_images;
        ctx.swp_image_views = swp_image_views;
        ctx.swp_img_num = swp_image_count;

        ctx.img_allocator = img_alloc;
        ctx.draw_img = bound_img;
        ctx.draw_img_extent = VkExtent2D{.width=width,.height=height};

        ctx.frames = frames;
        ctx.frame_count = 0;

        return ctx;
    }

    void destroy(){
        for(auto& frame : this->frames){
            VkResult res = vkWaitForFences(this->dev, 1, &frame.render_fence, true, 1000000000);
            vkDestroyCommandPool(this->dev, frame.cmd_pool, nullptr);
            vkDestroyFence(this->dev, frame.render_fence, nullptr);
            vkDestroySemaphore(this->dev, frame.render_semaphore, nullptr);
            vkDestroySemaphore(this->dev, frame.swp_semaphore, nullptr);
        }
        vkDestroySwapchainKHR(this->dev, this->swapchain, nullptr);
        for(int i=0;i<this->swp_img_num;++i){
            vkDestroyImageView(this->dev,this->swp_image_views[i],nullptr);
            //vkDestroyImage(this->dev, this->swp_images[i],nullptr);
        }

        delete[] this->swp_images;
        delete[] this->swp_image_views;

        this->draw_img.destroy(this->dev);
        this->img_allocator.destroy(this->dev);

        vkDestroyDevice(this->dev,nullptr);
        vkDestroySurfaceKHR(this->instance, this->surface, nullptr);

        vkb::destroy_debug_utils_messenger(this->instance, this->debug_messenger);
        vkDestroyInstance(this->instance, nullptr);
    }

    //~RenderContext(){
    //    if(this->instance != VK_NULL_HANDLE){
    //        for(auto& frame : this->frames){
    //            vkDestroyCommandPool(this->dev, frame.cmd_pool, nullptr);
    //            vkDestroyFence(this->dev, frame.render_fence, nullptr);
    //            vkDestroySemaphore(this->dev, frame.render_semaphore, nullptr);
    //            vkDestroySemaphore(this->dev, frame.swp_semaphore, nullptr);
    //        }
    //        vkDestroySwapchainKHR(this->dev, this->swapchain, nullptr);
    //        for(int i=0;i<this->swp_img_num;++i){
    //            vkDestroyImageView(this->dev,this->swp_image_views[i],nullptr);
    //            vkDestroyImage(this->dev, this->swp_images[i],nullptr);
    //        }

    //        vkDestroyDevice(this->dev,nullptr);
    //        vkDestroySurfaceKHR(this->instance, this->surface, nullptr);

    //        vkb::destroy_debug_utils_messenger(this->instance, this->debug_messenger);
    //        vkDestroyInstance(this->instance, nullptr);
    //    }
    //}

    //RenderContext(RenderContext&& other) :
    //        instance(other.instance),
    //        debug_messenger(other.debug_messenger),
    //        surface(other.surface),
    //        phys_dev(other.phys_dev),
    //        dev(other.dev),
    //        g_queue(other.g_queue),
    //        g_queue_fam(other.g_queue_fam),
    //        swapchain(other.swapchain),
    //        swp_extent(other.swp_extent),
    //        format(other.format),
    //        swp_images(std::move(other.swp_images)),
    //        swp_image_views(std::move(other.swp_image_views)),
    //        frames(other.frames),
    //        frame_count(other.frame_count)
    //{
    //    other.instance = VK_NULL_HANDLE;
    //}

    //RenderContext& operator=(RenderContext&& other){
    //    this->instance = other.instance;
    //    this->debug_messenger = other.debug_messenger;
    //    this->surface = other.surface;
    //    this->phys_dev = other.phys_dev;
    //    this->dev = other.dev;
    //    this->g_queue = other.g_queue;
    //    this->g_queue_fam = other.g_queue_fam;
    //
    //    this->swapchain = other.swapchain;
    //    this->swp_extent = other.swp_extent;
    //    this->format = other.format;

    //    this->swp_images = std::move(other.swp_images);
    //    this->swp_image_views = std::move(other.swp_image_views);

    //    this->frames = other.frames;
    //    this->frame_count = other.frame_count;

    //    other.instance = VK_NULL_HANDLE;
    //    return *this;
    //}

    //RenderContext(const RenderContext&) = delete;

    //RenderContext& operator=(const RenderContext&) = delete;

    void draw(){
        FrameData &frame = this->get_current_frame();
        VkResult res = vkWaitForFences(this->dev, 1, &frame.render_fence, true, 1000000000);
        if(res!=VK_SUCCESS) abort_msg("failed wait for fence");
        res = vkResetFences(this->dev, 1, &frame.render_fence);
        if(res!=VK_SUCCESS) abort_msg("failed fence reset");

        uint32_t swp_img_ind;
        res = vkAcquireNextImageKHR(
                this->dev, 
                this->swapchain, 
                1000000000, 
                frame.swp_semaphore,
                nullptr,
                &swp_img_ind);
        if(res!=VK_SUCCESS) abort_msg("failed img acquire");


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
        if(res!=VK_SUCCESS) abort_msg("failed to create command buffer");

        VkCommandBufferBeginInfo cmd_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        // begin
        res = vkBeginCommandBuffer(cmd_buffer, &cmd_info);
        if(res!=VK_SUCCESS) abort_msg("failed to begin command buffer");

        transition_image(
                cmd_buffer,
                this->draw_img.img,
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
        vkCmdClearColorImage(cmd_buffer, this->draw_img.img, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &clearRange);

        transition_image(cmd_buffer, this->draw_img.img,VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        transition_image(cmd_buffer, this->swp_images[swp_img_ind],VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // TODO: decouple draw img and swapchain extents
        vkutils::copy_image_to_image(cmd_buffer, this->draw_img.img, this->swp_images[swp_img_ind] , this->draw_img_extent, this->swp_extent);

        transition_image(cmd_buffer, this->swp_images[swp_img_ind],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        //finalize the command buffer (we can no longer add commands, but it can now be executed)
        res = vkEndCommandBuffer(cmd_buffer);
        if(res!=VK_SUCCESS) abort_msg("failed to end cmd buffer");

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
        if(res!=VK_SUCCESS) abort_msg("failed to submit cmds");

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
        if(res!=VK_SUCCESS) abort_msg("failed to present");

        //increase the number of frames drawn
        this->frame_count++;
    }
};


extern "C" RenderContext *renderer_new(const WindowHandles *wh, uint32_t width, uint32_t height, bool validation_layers){
    RenderContext *ctx = new RenderContext{};
    *ctx = RenderContext::make(*wh, width, height, validation_layers);
    return ctx;
}

extern "C" void renderer_draw(RenderContext *ctx){
    ctx->draw();
}

extern "C" void renderer_destroy(RenderContext* ctx){
    ctx->destroy();
    delete ctx;
}
