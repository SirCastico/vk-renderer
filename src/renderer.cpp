#include "renderer.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <span>
#include <vulkan/vulkan_core.h>
#include <VkBootstrap.h>
#include <stdexcept>
#include <cstring>

void abort_msg(const char*msg){
    std::puts(msg);
    std::abort();
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

    template<uint32_t LEN>
    struct DescriptorLayoutBuilder{
        std::array<VkDescriptorSetLayoutBinding, LEN> arr;
        DescriptorLayoutBuilder()=default;

        constexpr static DescriptorLayoutBuilder<1> make(VkDescriptorSetLayoutBinding b){
            DescriptorLayoutBuilder<1> builder{};
            builder.arr[0] = b;
            return builder;
        }
        constexpr DescriptorLayoutBuilder<LEN+1> add_binding(VkDescriptorSetLayoutBinding b){
            DescriptorLayoutBuilder<LEN+1> builder{};
            std::memcpy(builder.arr, this->arr, sizeof(b)*LEN);
            builder.arr[LEN] = b;
            return builder;
        }

        constexpr VkDescriptorSetLayout build(VkDevice dev, void *pnext, VkDescriptorSetLayoutCreateFlags flags){
            VkDescriptorSetLayoutCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.pNext = pnext;

            info.pBindings = this->arr.data();
            info.bindingCount = (uint32_t)this->arr.size();
            info.flags = flags;

            VkDescriptorSetLayout set;
            VkResult res = vkCreateDescriptorSetLayout(dev, &info, nullptr, &set);
            if(res!=VK_SUCCESS) abort_msg("failed to build descriptor set layout");
            return set;
        };
    };

    struct DescriptorAllocator{
        struct PoolSizeRatio{
            VkDescriptorType type;
            float ratio;
        };

        VkDescriptorPool pool;

        DescriptorAllocator(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios){
            
        }
        void clear_descriptors(VkDevice device);
        void destroy_pool(VkDevice device);

        VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
    };

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


    std::optional<VkShaderModule> load_shader_module(
            const char* filePath, VkDevice device){

        // open the file. With cursor at the end
        std::ifstream file(filePath, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            return std::nullopt;
        }

        // find what the size of the file is by looking up the location of the cursor
        // because the cursor is at the end, it gives the size directly in bytes
        size_t fileSize = (size_t)file.tellg();

        // spirv expects the buffer to be on uint32, so make sure to reserve a int
        // vector big enough for the entire file
        std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

        // put file cursor at beginning
        file.seekg(0);

        // load the entire file into the buffer
        file.read((char*)buffer.data(), fileSize);

        // now that the file is loaded into the buffer, we can close it
        file.close();

        // create a new shader module, using the buffer we loaded
        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pNext = nullptr;

        // codeSize has to be in bytes, so multply the ints in the buffer by size of
        // int to know the real size of the buffer
        createInfo.codeSize = buffer.size() * sizeof(uint32_t);
        createInfo.pCode = buffer.data();

        // check that the creation goes well.
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            return std::nullopt;
        }
        return shaderModule;
    }
}

struct RenderContext{

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
    VkImageView draw_img_view;
    static constexpr VkFormat draw_img_format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImage *swp_images;
    VkImageView *swp_image_views;
    uint32_t swp_img_num;

    VkDescriptorPool desc_pool;
    VkDescriptorSet desc_set;
    VkDescriptorSetLayout desc_set_layout;

    static constexpr uint32_t FRAME_OVERLAP = 2;
    using FrameDataArray = std::array<FrameData, FRAME_OVERLAP>;

    FrameDataArray frames;
    uint64_t frame_count;

    VkPipeline gradient_pipeline;
    VkPipelineLayout gradient_pipeline_layout;

    VkPipeline g_pipeline;
    VkPipelineLayout g_pipeline_layout;

    FrameData& get_current_frame(){
        return this->frames[this->frame_count % FRAME_OVERLAP];
    }

    RenderContext() = default;

    // TODO: allow surface to be created externally
    // TODO: better error handling

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
                RenderContext::draw_img_format,
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

        VkImageViewCreateInfo img_view_info = vkinit::imageview_create_info(
                RenderContext::draw_img_format, bound_img.img, VK_IMAGE_ASPECT_COLOR_BIT);

        VkImageView draw_img_view;
        err = vkCreateImageView(vkb_device, &img_view_info, nullptr, &draw_img_view);
        if(err!=VK_SUCCESS)abort_msg("failed to create image view");

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

        // descriptors

        VkDescriptorPool desc_pool;
        constexpr uint32_t pool_sizes_len = 1;
        VkDescriptorPoolSize pool_sizes[pool_sizes_len] = {
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 10,
            }
        };

        VkDescriptorPoolCreateInfo desc_pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = 32,
            .poolSizeCount = pool_sizes_len,
            .pPoolSizes = pool_sizes,
        };

        vkCreateDescriptorPool(
            vkb_device.device,
            &desc_pool_info, 
            nullptr,
            &desc_pool);

        
        VkDescriptorSetLayout desc_layout = vkutils::DescriptorLayoutBuilder<1>::make(VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        }).build(vkb_device.device, nullptr, 0);


        VkDescriptorSetAllocateInfo desc_alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = desc_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &desc_layout
        };
        VkDescriptorSet desc_set;
        err = vkAllocateDescriptorSets(vkb_device, &desc_alloc_info, &desc_set);
        if(err!=VK_SUCCESS) abort_msg("failed desc set alloc");

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgInfo.imageView = draw_img_view;
        
        VkWriteDescriptorSet drawImageWrite = {};
        drawImageWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        drawImageWrite.pNext = nullptr;
        
        drawImageWrite.dstBinding = 0;
        drawImageWrite.dstSet = desc_set;
        drawImageWrite.descriptorCount = 1;
        drawImageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        drawImageWrite.pImageInfo = &imgInfo;

        vkUpdateDescriptorSets(vkb_device, 1, &drawImageWrite, 0, nullptr);

        // create pipeline
        VkPipelineLayoutCreateInfo compute_layout{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .setLayoutCount = 1,
            .pSetLayouts = &desc_layout,
        };
        VkPipelineLayout pipeline_layout;
        err = vkCreatePipelineLayout(vkb_device, &compute_layout, nullptr, &pipeline_layout);
        if(err!=VK_SUCCESS) abort_msg("failed create pipeline layout");

        auto shader_opt = vkutils::load_shader_module("./shaders/gradient.comp.spv", vkb_device);
        if(!shader_opt) abort_msg("failed to read compute shader");

        VkShaderModule shader = shader_opt.value();

        VkPipelineShaderStageCreateInfo stageinfo{};
        stageinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageinfo.pNext = nullptr;
        stageinfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageinfo.module = shader;
        stageinfo.pName = "main";

        VkComputePipelineCreateInfo computePipelineCreateInfo{};
        computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        computePipelineCreateInfo.pNext = nullptr;
        computePipelineCreateInfo.layout = pipeline_layout;
        computePipelineCreateInfo.stage = stageinfo;

        VkPipeline c_pipeline;
        err = vkCreateComputePipelines(vkb_device,VK_NULL_HANDLE,1,&computePipelineCreateInfo, nullptr, &c_pipeline);

        vkDestroyShaderModule(vkb_device, shader, nullptr);

        // create graphics pipeline
        VkPipelineLayoutCreateInfo gpinfo{.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout g_pipeline_layout;
        err = vkCreatePipelineLayout(vkb_device, &gpinfo, nullptr, &g_pipeline_layout);
        if(err!=VK_SUCCESS) abort_msg("failed create graphics pipeline layout");
        
        VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

        // Specify we will use triangle lists to draw geometry.
        VkPipelineInputAssemblyStateCreateInfo input_assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        // Specify rasterization state.
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.cullMode  = VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
        raster.lineWidth = 1.0f;

        // Our attachment will write to all color channels, but no blending is enabled.
        VkPipelineColorBlendAttachmentState blend_attachment{};
        blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 1;
        blend.pAttachments    = &blend_attachment;

        // We will have one viewport and scissor box.
        VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount  = 1;

        // Disable all depth testing.
        VkPipelineDepthStencilStateCreateInfo depth_stencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

        // No multisampling.
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Specify that these states will be dynamic, i.e. not part of pipeline state object.
        std::array<VkDynamicState, 2> dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

        VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.pDynamicStates    = dynamics.data();
        dynamic.dynamicStateCount = (uint32_t)dynamics.size();

        // Load our SPIR-V shaders.
        std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{};

        // Vertex stage of the pipeline
        shader_stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        shader_stages[0].module = vkutils::load_shader_module("shaders/triangle.vert.spv", vkb_device).value(); // may throw
        shader_stages[0].pName  = "main";

        // Fragment stage of the pipeline
        shader_stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        shader_stages[1].module = vkutils::load_shader_module("shaders/triangle.frag.spv", vkb_device).value(); // may throw
        shader_stages[1].pName  = "main";

        // dynamic rendering
        VkPipelineRenderingCreateInfo g_pip_render_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &RenderContext::draw_img_format,
        };

        VkGraphicsPipelineCreateInfo g_pipe_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = (void*)&g_pip_render_info,
            .stageCount = (uint32_t)shader_stages.size(),
            .pStages = shader_stages.data(),
            .pVertexInputState   = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pViewportState      = &viewport,
            .pRasterizationState = &raster,
            .pMultisampleState   = &multisample,
            .pDepthStencilState  = &depth_stencil,
            .pColorBlendState    = &blend,
            .pDynamicState       = &dynamic,
            .layout = g_pipeline_layout,
        };

        VkPipeline g_pipeline;
        vkCreateGraphicsPipelines(vkb_device, VK_NULL_HANDLE, 1, &g_pipe_info, nullptr, &g_pipeline);

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
        ctx.draw_img_view = draw_img_view;

        ctx.desc_pool = desc_pool;
        ctx.desc_set = desc_set;
        ctx.desc_set_layout = desc_layout;

        ctx.frames = frames;
        ctx.frame_count = 0;

        ctx.gradient_pipeline = c_pipeline;
        ctx.gradient_pipeline_layout = pipeline_layout;

        ctx.g_pipeline = g_pipeline;
        ctx.g_pipeline_layout = g_pipeline_layout;

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

        vkDestroyImageView(this->dev, this->draw_img_view, nullptr);
        this->draw_img.destroy(this->dev);
        this->img_allocator.destroy(this->dev);

        vkDestroyPipeline(this->dev, this->gradient_pipeline, nullptr);
        vkDestroyPipelineLayout(this->dev, this->gradient_pipeline_layout, nullptr);

        vkDestroyPipeline(this->dev, this->g_pipeline, nullptr);
        vkDestroyPipelineLayout(this->dev, this->g_pipeline_layout, nullptr);

        vkDestroyDescriptorPool(this->dev, this->desc_pool, nullptr);
        vkDestroyDescriptorSetLayout(this->dev, this->desc_set_layout, nullptr);

        vkDestroySwapchainKHR(this->dev, this->swapchain, nullptr);
        for(int i=0;i<this->swp_img_num;++i){
            vkDestroyImageView(this->dev,this->swp_image_views[i],nullptr);
            //vkDestroyImage(this->dev, this->swp_images[i],nullptr);
        }

        delete[] this->swp_images;
        delete[] this->swp_image_views;


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
        if(res!=VK_SUCCESS) abort_msg("failed to reset command pool");

        // allocate the default command buffer that we will use for rendering
        VkCommandBufferAllocateInfo cmdAllocInfo = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = frame.cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

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

        vkutils::transition_image(
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

        // graphics
        
        vkutils::transition_image(
                cmd_buffer,
                this->draw_img.img,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                //VK_IMAGE_LAYOUT_GENERAL);

        VkRenderingAttachmentInfo color_att_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = this->draw_img_view,
            .imageLayout = VkImageLayout::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VkAttachmentLoadOp::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VkAttachmentStoreOp::VK_ATTACHMENT_STORE_OP_STORE,
        };
        VkRenderingInfo render_info = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = VkRect2D{
                .offset=VkOffset2D{0,0},
                .extent=this->draw_img_extent,
            },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_att_info
        };

        vkCmdBeginRendering(cmd_buffer, &render_info);
        
        vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, this->g_pipeline);
        VkViewport viewport = {
            .x = 0,
            .y = 0,
            .width = (float)this->draw_img_extent.width,
            .height = (float)this->draw_img_extent.height,
            .minDepth = 0.f,
            .maxDepth = 1.f,
        };

        vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);

        VkRect2D scissor = {};
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent.width = this->draw_img_extent.width;
        scissor.extent.height = this->draw_img_extent.height;

        vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

        //launch a draw command to draw 3 vertices
        vkCmdDraw(cmd_buffer, 3, 1, 0, 0);

        vkCmdEndRendering(cmd_buffer);

        // compute
        //vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, this->gradient_pipeline);
        //vkCmdBindDescriptorSets(
        //    cmd_buffer,
        //    VK_PIPELINE_BIND_POINT_COMPUTE,
        //    this->gradient_pipeline_layout,
        //    0,1,
        //    &this->desc_set,
        //    0, nullptr);

        //vkCmdDispatch(
        //    cmd_buffer, 
        //    std::ceil(this->draw_img_extent.width / 16.0), 
        //    std::ceil(this->draw_img_extent.height / 16.0),
        //    1);

        // write to swapchain image
        vkutils::transition_image(cmd_buffer, this->draw_img.img,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        vkutils::transition_image(cmd_buffer, this->swp_images[swp_img_ind],VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // TODO: decouple draw img and swapchain extents
        vkutils::copy_image_to_image(cmd_buffer, this->draw_img.img, this->swp_images[swp_img_ind] , this->draw_img_extent, this->swp_extent);

        vkutils::transition_image(cmd_buffer, this->swp_images[swp_img_ind],VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        //finalize the command buffer (we can no longer add commands, but it can now be executed)
        res = vkEndCommandBuffer(cmd_buffer);
        if(res!=VK_SUCCESS) abort_msg("failed to end cmd buffer");

        //prepare the submission to the queue. 
        //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
        //we will signal the _renderSemaphore, to signal that rendering has finished
        VkCommandBufferSubmitInfo cmdinfo = vkutils::command_buffer_submit_info(cmd_buffer);	
        
        VkSemaphoreSubmitInfo waitInfo = vkutils::semaphore_submit_info(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR,frame.swp_semaphore);
        VkSemaphoreSubmitInfo signalInfo = vkutils::semaphore_submit_info(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, frame.render_semaphore);	
        
        VkSubmitInfo2 submit = vkutils::submit_info(&cmdinfo,&signalInfo,&waitInfo);	

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
