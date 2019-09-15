#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan\vulkan.h>

#ifndef NDEBUG
#define VK_CHECK(x) do { VkResult res = x; assert(res == VK_SUCCESS); } while(false)
#else
#define VK_CHECK(x) x
#endif

// Find the appropriate memory allocation type based on flags passed by app
uint32_t FindMemoryType(const uint32_t deviceReq, VkMemoryPropertyFlags memoryFlags, const VkPhysicalDeviceMemoryProperties& memoryProps)
{
    for (unsigned i = 0; i < memoryProps.memoryTypeCount; i++)
    {
        if (deviceReq & (1u << i))
        {
            if ((memoryProps.memoryTypes[i].propertyFlags & memoryFlags) == memoryFlags)
            {
                return i;
            }
        }
    }

    assert(false);
    return ~0u;
}

// Runtime parameters
struct Args
{
    // Frame buffer width & height
    uint32_t    m_RenderWidth = 1024;
    uint32_t    m_RenderHeight = 720;

    // Compute local workgroup size
    uint32_t    m_LocalWorkgroupSize = 16;

    // Use dGPU or iGPU
    bool        m_UseDiscreteGPU = false;
};

struct ComputeContext
{
    VkInstance                  m_Instance;
    VkDebugUtilsMessengerEXT    m_DebugUtils;

    VkPhysicalDevice            m_PhysicalDevice;
    VkDevice                    m_Device;
    VkQueue                     m_ComputeQueue;
    uint32_t                    m_QueueFamilyIdx;

    VkCommandPool               m_CmdPool;
    VkCommandBuffer             m_CmdBuffer;

    VkPipeline                  m_Pipeline;
    VkPipelineLayout            m_Layout;
    VkDescriptorPool            m_DescriptorPool;
    VkDescriptorSetLayout       m_DescriptorLayout;
    VkDescriptorSet             m_DescriptorSet;
    VkShaderModule              m_ShaderModule;

    VkImage                     m_StorageImage;
    VkDeviceMemory              m_StorageImageMemory;
    VkImageView                 m_StorageImageView;

    VkBuffer                    m_ResultBuffer;
    VkDeviceMemory              m_ResultBufferMemory;
};

VKAPI_ATTR VkBool32 VKAPI_CALL debug_messenger_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    switch (messageSeverity)
    {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        printf("%s\n", pCallbackData->pMessage);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        printf("%s\n", pCallbackData->pMessage);
        assert(false);
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        break;
    default:
        break;
    }

    // Don't bail out, but keep going.
    return false;
}

int main(int argc, char* pArgv[]);

void ParseCommandsLine(int argc, char* pArgv[], Args* pArgs);

bool InitVulkanInstance(Args* pArgs, ComputeContext* pContext);

bool InitVulkanDevice(Args* pArgs, ComputeContext* pContext);

void InitCommandBuffer(ComputeContext* pContext);

void CreateStorageImage(Args* pArgs, ComputeContext* pContext);

void InitDescriptorSet(ComputeContext* pContext);

void CreatePipeline(ComputeContext* pContext);

void CleanUp(ComputeContext* pContext);

void DispatchGPGPU(Args* pArgs, ComputeContext* pComputeContext);

void FetchStorageImageContents(Args* pArgs, ComputeContext* computeContext);

void OutputFrame(Args* pArgs, ComputeContext* computeContext);

void ConvertImageLayoutToGeneral(ComputeContext* pContext);

int main(int argc, char* pArgv[])
{
    Args args = {};
    ParseCommandsLine(argc, pArgv, &args);

    ComputeContext computeContext = {};

    if (!InitVulkanInstance(&args, &computeContext))
    {
        assert(false && "Failed to initialize Vulkan instance!");
    }
    printf("Instance initialized\n");

    if (!InitVulkanDevice(&args, &computeContext))
    {
        assert(false && "Failed to initialize Vulkan device!");
    }
    printf("Device and queue initialized\n");

    InitCommandBuffer(&computeContext);
    printf("Command pool and buffer created\n");

    VkCommandBufferBeginInfo cmdBeginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cmdBeginInfo.pNext = nullptr;
    cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    cmdBeginInfo.pInheritanceInfo = nullptr;
    VK_CHECK(vkBeginCommandBuffer(computeContext.m_CmdBuffer, &cmdBeginInfo));

    CreateStorageImage(&args, &computeContext);
    printf("Created storage image resources\n");

    InitDescriptorSet(&computeContext);
    printf("Descriptor set and layout created\n");

    CreatePipeline(&computeContext);
    printf("Pipeline created\n");

    ConvertImageLayoutToGeneral(&computeContext);
    printf("Image converted to general layout\n");

    DispatchGPGPU(&args, &computeContext);
    printf("GPGPU op recorded\n");

    FetchStorageImageContents(&args, &computeContext);
    printf("Copy op to read results back recorded\n");

    VK_CHECK(vkEndCommandBuffer(computeContext.m_CmdBuffer));

    auto begin = std::chrono::high_resolution_clock::now();

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.pNext = nullptr;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeContext.m_CmdBuffer;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    VK_CHECK(vkQueueSubmit(computeContext.m_ComputeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    printf("Work submitted to the compute queue\n");
    vkDeviceWaitIdle(computeContext.m_Device);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::seconds> (end - begin).count();
    printf("Raytracing done in %d seconds\n", diff);

    OutputFrame(&args, &computeContext);
    printf("Render saved to file gi_basic_vk.ppm\n");

    CleanUp(&computeContext);
    printf("Clean-up done\n");

#ifdef _DEBUG
    system("pause");
#endif

    return 0;
}

void ParseCommandsLine(int argc, char* pArgv[], Args* pArgs)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp("-dGPU", pArgv[i]) == 0)
        {
            pArgs->m_UseDiscreteGPU = strcmp(pArgv[i + 1], "1") == 0;
        }
        else if ((strcmp("-w", pArgv[i]) == 0) || (strcmp("--width", pArgv[i]) == 0))
        {
            pArgs->m_RenderWidth = atoi(pArgv[i + 1]);
        }
        else if ((strcmp("-h", pArgv[i]) == 0) || (strcmp("--height", pArgv[i]) == 0))
        {
            pArgs->m_RenderHeight = atoi(pArgv[i + 1]);
        }
        else if (strcmp("-local_size", pArgv[i]) == 0)
        {
            pArgs->m_LocalWorkgroupSize = atoi(pArgv[i + 1]);
        }
    }
}

bool InitVulkanInstance(Args* pArgs, ComputeContext* pContext)
{
    VkApplicationInfo appInfo = {};
    appInfo.pNext = nullptr;
    appInfo.pEngineName = "GI_Basic";
    appInfo.pApplicationName = "GI_Basic";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_VERSION_1_1;

#ifdef _DEBUG
    VkDebugUtilsMessengerCreateInfoEXT debugInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
    debugInfo.pNext = nullptr;
    debugInfo.flags = 0;
    debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback = debug_messenger_callback;
    debugInfo.pUserData = pContext;
#endif

    const char* enabledExts = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
    const char* enabledLayers = { "VK_LAYER_KHRONOS_validation" };

    VkInstanceCreateInfo instanceInfo = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
#ifdef _DEBUG
    instanceInfo.pNext = &debugInfo;
    instanceInfo.enabledExtensionCount = 1;
    instanceInfo.ppEnabledExtensionNames = &enabledExts;
    instanceInfo.enabledLayerCount = 1;
    instanceInfo.ppEnabledLayerNames = &enabledLayers;
#else
    instanceInfo.enabledExtensionCount = 0;
    instanceInfo.ppEnabledExtensionNames = nullptr;
    instanceInfo.enabledLayerCount = 0;
    instanceInfo.ppEnabledLayerNames = nullptr;
#endif
    instanceInfo.flags = VkInstanceCreateFlags(0);
    instanceInfo.pApplicationInfo = &appInfo;

    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &pContext->m_Instance));
    assert(pContext->m_Instance != nullptr);

#ifdef _DEBUG
    PFN_vkCreateDebugUtilsMessengerEXT CreateDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(pContext->m_Instance, "vkCreateDebugUtilsMessengerEXT"));
    assert(CreateDebugUtilsMessengerEXT != nullptr);

    VK_CHECK(CreateDebugUtilsMessengerEXT(pContext->m_Instance, &debugInfo, nullptr, &pContext->m_DebugUtils));
#endif

    return pContext->m_Instance != nullptr;
}

bool InitVulkanDevice(Args* pArgs, ComputeContext* pContext)
{
    uint32_t numPhysicalDevices = 0u;
    VK_CHECK(vkEnumeratePhysicalDevices(pContext->m_Instance, &numPhysicalDevices, nullptr));

    assert(numPhysicalDevices > 0);

    std::vector<VkPhysicalDevice> physicalDevices(numPhysicalDevices);
    VK_CHECK(vkEnumeratePhysicalDevices(pContext->m_Instance, &numPhysicalDevices, physicalDevices.data()));

    VkPhysicalDevice physicalDevice = nullptr;

    for (size_t i = 0; i < numPhysicalDevices; i++)
    {
        VkPhysicalDevice physicalDeviceHandle = physicalDevices[i];
        assert(physicalDeviceHandle != nullptr);

        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(physicalDeviceHandle, &props);

        if ((pArgs->m_UseDiscreteGPU && (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)) ||
            (!pArgs->m_UseDiscreteGPU && (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)))
        {
            physicalDevice = physicalDeviceHandle;
            break;
        }
    }

    pContext->m_PhysicalDevice = physicalDevice;

    uint32_t numQueueFamilyProps = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilyProps, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilyProps(numQueueFamilyProps);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilyProps, queueFamilyProps.data());

    for (size_t i = 0; i < numQueueFamilyProps; i++)
    {
        auto props = queueFamilyProps[i];
        if ((props.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
        {
            // Use the first queue capable of compute work
            pContext->m_QueueFamilyIdx = i;
        }
    }

    float queuePriority = { 1.f };

    VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueInfo.pNext = nullptr;
    queueInfo.queueCount = 1;
    queueInfo.queueFamilyIndex = pContext->m_QueueFamilyIdx;
    queueInfo.pQueuePriorities = &queuePriority;
    queueInfo.flags = VkDeviceQueueCreateFlags(0);

    VkDeviceCreateInfo deviceInfo = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceInfo.pNext = nullptr;
    deviceInfo.enabledExtensionCount = 0;
    deviceInfo.enabledLayerCount = 0;
    deviceInfo.pEnabledFeatures = nullptr;
    deviceInfo.flags = VkDeviceCreateFlags(0);
    deviceInfo.ppEnabledExtensionNames = nullptr;
    deviceInfo.ppEnabledLayerNames = nullptr;
    deviceInfo.queueCreateInfoCount = 1; // Use single queue capable of compute work
    deviceInfo.pQueueCreateInfos = &queueInfo;
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &pContext->m_Device));

    vkGetDeviceQueue(pContext->m_Device, pContext->m_QueueFamilyIdx, 0, &pContext->m_ComputeQueue);
    assert(pContext->m_ComputeQueue != nullptr);

    return pContext->m_Device != nullptr;
}

void InitCommandBuffer(ComputeContext* pContext)
{
    VkCommandPoolCreateInfo commandPoolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    commandPoolInfo.pNext = nullptr;
    commandPoolInfo.queueFamilyIndex = pContext->m_QueueFamilyIdx;

    VK_CHECK(vkCreateCommandPool(pContext->m_Device, &commandPoolInfo, nullptr, &pContext->m_CmdPool));
    assert(pContext->m_CmdPool != nullptr);

    VkCommandBufferAllocateInfo commandBufferInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    commandBufferInfo.pNext = nullptr;
    commandBufferInfo.commandPool = pContext->m_CmdPool;
    commandBufferInfo.commandBufferCount = 1;
    commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    VK_CHECK(vkAllocateCommandBuffers(pContext->m_Device, &commandBufferInfo, &pContext->m_CmdBuffer));
    assert(pContext->m_CmdBuffer != nullptr);
}

void CreateStorageImage(Args* pArgs, ComputeContext* pContext)
{
    VkImageCreateInfo imageInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.pNext = nullptr;
    imageInfo.flags = 0;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // Used in CS to store ray-traced pixel values
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM; // Corresponds to rgba8 in CS
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL; // No tiling
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.arrayLayers = 1;
    imageInfo.extent.width = pArgs->m_RenderWidth;
    imageInfo.extent.height = pArgs->m_RenderHeight;
    imageInfo.extent.depth = 1;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.mipLevels = 1;
    imageInfo.pQueueFamilyIndices = &pContext->m_QueueFamilyIdx;
    imageInfo.queueFamilyIndexCount = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // Used in single queue
    VK_CHECK(vkCreateImage(pContext->m_Device, &imageInfo, nullptr, &pContext->m_StorageImage));

    VkMemoryRequirements memreq = {};
    vkGetImageMemoryRequirements(pContext->m_Device, pContext->m_StorageImage, &memreq);

    VkPhysicalDeviceMemoryProperties memprops = {};
    vkGetPhysicalDeviceMemoryProperties(pContext->m_PhysicalDevice, &memprops);

    uint32_t memType = FindMemoryType(memreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memprops);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.pNext = nullptr;
    allocInfo.allocationSize = memreq.size;
    allocInfo.memoryTypeIndex = memType;
    VK_CHECK(vkAllocateMemory(pContext->m_Device, &allocInfo, nullptr, &pContext->m_StorageImageMemory));

    VK_CHECK(vkBindImageMemory(pContext->m_Device, pContext->m_StorageImage, pContext->m_StorageImageMemory, 0));

    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseArrayLayer = 0;
    range.baseMipLevel = 0;
    range.layerCount = 1;
    range.levelCount = 1;

    VkImageViewCreateInfo imageviewInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    imageviewInfo.pNext = nullptr;
    imageviewInfo.flags = 0;
    imageviewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageviewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageviewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageviewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageviewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageviewInfo.image = pContext->m_StorageImage;
    imageviewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageviewInfo.subresourceRange = range;
    VK_CHECK(vkCreateImageView(pContext->m_Device, &imageviewInfo, nullptr, &pContext->m_StorageImageView));
}

void InitDescriptorSet(ComputeContext* pContext)
{
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorCount = 1;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.pImmutableSamplers = nullptr;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo setLayoutInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    setLayoutInfo.pNext = nullptr;
    setLayoutInfo.flags = 0;
    setLayoutInfo.bindingCount = 1;
    setLayoutInfo.pBindings = &binding;
    VK_CHECK(vkCreateDescriptorSetLayout(pContext->m_Device, &setLayoutInfo, nullptr, &pContext->m_DescriptorLayout));
    assert(pContext->m_DescriptorLayout != nullptr);

    VkDescriptorPoolSize poolSize = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        1
    };

    VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.pNext = nullptr;
    poolInfo.flags = 0;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    VK_CHECK(vkCreateDescriptorPool(pContext->m_Device, &poolInfo, nullptr, &pContext->m_DescriptorPool));
    assert(pContext->m_DescriptorPool != nullptr);

    VkPipelineLayoutCreateInfo layoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.pNext = nullptr;
    layoutInfo.flags = 0;
    layoutInfo.pPushConstantRanges = nullptr;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pushConstantRangeCount = 0;
    layoutInfo.pSetLayouts = &pContext->m_DescriptorLayout;
    VK_CHECK(vkCreatePipelineLayout(pContext->m_Device, &layoutInfo, nullptr, &pContext->m_Layout));

    VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.pNext = nullptr;
    allocInfo.pSetLayouts = &pContext->m_DescriptorLayout;
    allocInfo.descriptorPool = pContext->m_DescriptorPool;
    allocInfo.descriptorSetCount = 1;
    VK_CHECK(vkAllocateDescriptorSets(pContext->m_Device, &allocInfo, &pContext->m_DescriptorSet));

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.sampler = nullptr;
    imageInfo.imageView = pContext->m_StorageImageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writeDescSet = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writeDescSet.pNext = nullptr;
    writeDescSet.pBufferInfo = nullptr;
    writeDescSet.pImageInfo = &imageInfo;
    writeDescSet.descriptorCount = 1;
    writeDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDescSet.dstSet = pContext->m_DescriptorSet;
    writeDescSet.dstBinding = 0;
    vkUpdateDescriptorSets(pContext->m_Device, 1, &writeDescSet, 0, nullptr);
}

void CreatePipeline(ComputeContext* pContext)
{
    std::vector<char> source;
    {
        // Open a file at its end
        std::ifstream file("./gi-vk_comp.spv", std::ifstream::binary | std::ifstream::ate | std::ifstream::in);
        assert(file.is_open());

        // Determine content's length
        size_t length = (size_t)file.tellg();
        source.resize(length);

        // Go to the beginning and read whole file content
        file.seekg(0, file.beg);
        file.read(source.data(), length);
        file.close();
    }

    VkShaderModuleCreateInfo moduleInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    moduleInfo.pNext = nullptr;
    moduleInfo.flags = 0;
    moduleInfo.codeSize = source.size();
    moduleInfo.pCode = reinterpret_cast<const uint32_t*> (source.data());
    VK_CHECK(vkCreateShaderModule(pContext->m_Device, &moduleInfo, nullptr, &pContext->m_ShaderModule));

    VkPipelineShaderStageCreateInfo shaderInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    shaderInfo.pNext = nullptr;
    shaderInfo.flags = 0;
    shaderInfo.pSpecializationInfo = nullptr;
    shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderInfo.pName = "main";
    shaderInfo.module = pContext->m_ShaderModule;

    VkComputePipelineCreateInfo pipeInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeInfo.pNext = nullptr;
    pipeInfo.flags = 0;
    pipeInfo.stage = shaderInfo;
    pipeInfo.basePipelineIndex = ~0ul;
    pipeInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipeInfo.layout = pContext->m_Layout;
    VK_CHECK(vkCreateComputePipelines(pContext->m_Device, nullptr, 1, &pipeInfo, nullptr, &pContext->m_Pipeline));
    assert(pContext->m_Pipeline != nullptr);
}

void ConvertImageLayoutToGeneral(ComputeContext* pContext)
{
    VkImageMemoryBarrier imageBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    imageBarrier.pNext = nullptr;
    imageBarrier.srcQueueFamilyIndex = imageBarrier.dstQueueFamilyIndex = pContext->m_QueueFamilyIdx;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.image = pContext->m_StorageImage;
    imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.layerCount = imageBarrier.subresourceRange.levelCount = 1;

    vkCmdPipelineBarrier(
        pContext->m_CmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &imageBarrier);
}

void DispatchGPGPU(Args* pArgs, ComputeContext* pContext)
{
    // Dispatch GPGPU command

    vkCmdBindPipeline(pContext->m_CmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pContext->m_Pipeline);
    vkCmdBindDescriptorSets(pContext->m_CmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pContext->m_Layout, 0, 1, &pContext->m_DescriptorSet, 0, nullptr);

    vkCmdDispatch(pContext->m_CmdBuffer, pArgs->m_RenderWidth / pArgs->m_LocalWorkgroupSize, pArgs->m_RenderHeight / pArgs->m_LocalWorkgroupSize, 1);
}

void FetchStorageImageContents(Args* pArgs, ComputeContext* pContext)
{
    // Prepare buffer to copy the storage image to

    VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.pNext = nullptr;
    bufferInfo.flags = 0;
    bufferInfo.queueFamilyIndexCount = 1;
    bufferInfo.pQueueFamilyIndices = &pContext->m_QueueFamilyIdx;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.size = pArgs->m_RenderWidth * pArgs->m_RenderHeight * 4u;
    VK_CHECK(vkCreateBuffer(pContext->m_Device, &bufferInfo, nullptr, &pContext->m_ResultBuffer));

    VkMemoryRequirements memreq = {};
    vkGetBufferMemoryRequirements(pContext->m_Device, pContext->m_ResultBuffer, &memreq);

    VkPhysicalDeviceMemoryProperties memprops = {};
    vkGetPhysicalDeviceMemoryProperties(pContext->m_PhysicalDevice, &memprops);

    uint32_t memType = FindMemoryType(memreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memprops);

    VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.pNext = nullptr;
    allocInfo.allocationSize = memreq.size;
    allocInfo.memoryTypeIndex = memType;
    VK_CHECK(vkAllocateMemory(pContext->m_Device, &allocInfo, nullptr, &pContext->m_ResultBufferMemory));

    VK_CHECK(vkBindBufferMemory(pContext->m_Device, pContext->m_ResultBuffer, pContext->m_ResultBufferMemory, 0));

    // Copy storage image contents from the device-local memory to host-visible buffer

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent.width = pArgs->m_RenderWidth;
    copyRegion.imageExtent.height = pArgs->m_RenderHeight;
    copyRegion.imageExtent.depth = 1;
    copyRegion.imageOffset = {};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;;

    vkCmdCopyImageToBuffer(pContext->m_CmdBuffer, pContext->m_StorageImage, VK_IMAGE_LAYOUT_GENERAL, pContext->m_ResultBuffer, 1, &copyRegion);
}

void OutputFrame(Args* pArgs, ComputeContext* pContext)
{
    FILE* pFile = nullptr;
    fopen_s(&pFile, "gi_basic_vk.ppm", "w");
    fprintf_s(pFile, "P3\n%d %d\n%d\n ", pArgs->m_RenderWidth, pArgs->m_RenderHeight, 255);

    uint8_t* pBuffer = nullptr;
    VK_CHECK(vkMapMemory(pContext->m_Device, pContext->m_ResultBufferMemory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&pBuffer)));
    assert(pBuffer != nullptr);

    for (auto i = 0; i < pArgs->m_RenderWidth * pArgs->m_RenderHeight * 4; i += 4)
    {
        // Write out color values clamped to [0, 255] 
        uint8_t r = pBuffer[i];
        uint8_t g = pBuffer[i + 1];
        uint8_t b = pBuffer[i + 2];
        fprintf_s(pFile, "%d %d %d ", r, g, b);
    }
    fclose(pFile);
}

void CleanUp(ComputeContext* pContext)
{
    vkDeviceWaitIdle(pContext->m_Device);

    vkDestroyDescriptorSetLayout(pContext->m_Device, pContext->m_DescriptorLayout, nullptr);
    vkDestroyPipelineLayout(pContext->m_Device, pContext->m_Layout, nullptr);
    vkDestroyShaderModule(pContext->m_Device, pContext->m_ShaderModule, nullptr);
    vkDestroyPipeline(pContext->m_Device, pContext->m_Pipeline, nullptr);

    vkDestroyDescriptorPool(pContext->m_Device, pContext->m_DescriptorPool, nullptr);

    vkFreeCommandBuffers(pContext->m_Device, pContext->m_CmdPool, 1, &pContext->m_CmdBuffer);
    vkDestroyCommandPool(pContext->m_Device, pContext->m_CmdPool, nullptr);

    vkDestroyImage(pContext->m_Device, pContext->m_StorageImage, nullptr);
    vkDestroyImageView(pContext->m_Device, pContext->m_StorageImageView, nullptr);
    vkFreeMemory(pContext->m_Device, pContext->m_StorageImageMemory, nullptr);

    vkDestroyBuffer(pContext->m_Device, pContext->m_ResultBuffer, nullptr);
    vkFreeMemory(pContext->m_Device, pContext->m_ResultBufferMemory, nullptr);

    vkDestroyDevice(pContext->m_Device, nullptr);

    if (pContext->m_DebugUtils != nullptr)
    {
        PFN_vkDestroyDebugUtilsMessengerEXT DestroyDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(pContext->m_Instance, "vkDestroyDebugUtilsMessengerEXT"));
        assert(DestroyDebugUtilsMessengerEXT != nullptr);

        DestroyDebugUtilsMessengerEXT(pContext->m_Instance, pContext->m_DebugUtils, nullptr);
    }

    vkDestroyInstance(pContext->m_Instance, nullptr);
}