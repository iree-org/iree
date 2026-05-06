// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(IREE_HAL_VULKAN_INSTANCE_OPTIONAL_PFN)
#define IREE_HAL_VULKAN_INSTANCE_OPTIONAL_PFN IREE_HAL_VULKAN_INSTANCE_PFN
#define IREE_HAL_VULKAN_UNDEFINE_INSTANCE_OPTIONAL_PFN
#endif  // !IREE_HAL_VULKAN_INSTANCE_OPTIONAL_PFN

#if !defined(IREE_HAL_VULKAN_DEVICE_OPTIONAL_PFN)
#define IREE_HAL_VULKAN_DEVICE_OPTIONAL_PFN IREE_HAL_VULKAN_DEVICE_PFN
#define IREE_HAL_VULKAN_UNDEFINE_DEVICE_OPTIONAL_PFN
#endif  // !IREE_HAL_VULKAN_DEVICE_OPTIONAL_PFN

// Global loader-level entry points.

IREE_HAL_VULKAN_LOADER_PFN(VkResult, vkCreateInstance,
                           DECL(const VkInstanceCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkInstance* pInstance),
                           ARGS(pCreateInfo, pAllocator, pInstance))

IREE_HAL_VULKAN_LOADER_PFN(VkResult, vkEnumerateInstanceExtensionProperties,
                           DECL(const char* pLayerName,
                                uint32_t* pPropertyCount,
                                VkExtensionProperties* pProperties),
                           ARGS(pLayerName, pPropertyCount, pProperties))

IREE_HAL_VULKAN_LOADER_PFN(VkResult, vkEnumerateInstanceLayerProperties,
                           DECL(uint32_t* pPropertyCount,
                                VkLayerProperties* pProperties),
                           ARGS(pPropertyCount, pProperties))

IREE_HAL_VULKAN_LOADER_PFN(VkResult, vkEnumerateInstanceVersion,
                           DECL(uint32_t* pApiVersion), ARGS(pApiVersion))

// Instance-level entry points.

IREE_HAL_VULKAN_INSTANCE_PFN(void, vkDestroyInstance,
                             DECL(VkInstance instance,
                                  const VkAllocationCallbacks* pAllocator),
                             ARGS(instance, pAllocator))

IREE_HAL_VULKAN_INSTANCE_PFN(VkResult, vkEnumeratePhysicalDevices,
                             DECL(VkInstance instance,
                                  uint32_t* pPhysicalDeviceCount,
                                  VkPhysicalDevice* pPhysicalDevices),
                             ARGS(instance, pPhysicalDeviceCount,
                                  pPhysicalDevices))

IREE_HAL_VULKAN_INSTANCE_PFN(
    VkResult, vkCreateDevice,
    DECL(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo* pCreateInfo,
         const VkAllocationCallbacks* pAllocator, VkDevice* pDevice),
    ARGS(physicalDevice, pCreateInfo, pAllocator, pDevice))

IREE_HAL_VULKAN_INSTANCE_PRIVATE_PFN(PFN_vkVoidFunction, vkGetDeviceProcAddr,
                                     DECL(VkDevice device, const char* pName),
                                     ARGS(device, pName))

IREE_HAL_VULKAN_INSTANCE_PFN(void, vkGetPhysicalDeviceProperties2,
                             DECL(VkPhysicalDevice physicalDevice,
                                  VkPhysicalDeviceProperties2* pProperties),
                             ARGS(physicalDevice, pProperties))

IREE_HAL_VULKAN_INSTANCE_PFN(void, vkGetPhysicalDeviceFeatures2,
                             DECL(VkPhysicalDevice physicalDevice,
                                  VkPhysicalDeviceFeatures2* pFeatures),
                             ARGS(physicalDevice, pFeatures))

IREE_HAL_VULKAN_INSTANCE_PFN(
    void, vkGetPhysicalDeviceMemoryProperties2,
    DECL(VkPhysicalDevice physicalDevice,
         VkPhysicalDeviceMemoryProperties2* pMemoryProperties),
    ARGS(physicalDevice, pMemoryProperties))

IREE_HAL_VULKAN_INSTANCE_PFN(
    void, vkGetPhysicalDeviceQueueFamilyProperties2,
    DECL(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount,
         VkQueueFamilyProperties2* pQueueFamilyProperties),
    ARGS(physicalDevice, pQueueFamilyPropertyCount, pQueueFamilyProperties))

IREE_HAL_VULKAN_INSTANCE_PFN(
    VkResult, vkEnumerateDeviceExtensionProperties,
    DECL(VkPhysicalDevice physicalDevice, const char* pLayerName,
         uint32_t* pPropertyCount, VkExtensionProperties* pProperties),
    ARGS(physicalDevice, pLayerName, pPropertyCount, pProperties))

IREE_HAL_VULKAN_INSTANCE_OPTIONAL_PFN(
    VkResult, vkGetPhysicalDeviceCalibrateableTimeDomainsEXT,
    DECL(VkPhysicalDevice physicalDevice, uint32_t* pTimeDomainCount,
         VkTimeDomainEXT* pTimeDomains),
    ARGS(physicalDevice, pTimeDomainCount, pTimeDomains))

// Device-level entry points.

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyDevice,
                           DECL(VkDevice device,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkGetDeviceQueue2,
                           DECL(VkDevice device,
                                const VkDeviceQueueInfo2* pQueueInfo,
                                VkQueue* pQueue),
                           ARGS(device, pQueueInfo, pQueue))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateBuffer,
                           DECL(VkDevice device,
                                const VkBufferCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkBuffer* pBuffer),
                           ARGS(device, pCreateInfo, pAllocator, pBuffer))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyBuffer,
                           DECL(VkDevice device, VkBuffer buffer,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, buffer, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateFence,
                           DECL(VkDevice device,
                                const VkFenceCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkFence* pFence),
                           ARGS(device, pCreateInfo, pAllocator, pFence))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyFence,
                           DECL(VkDevice device, VkFence fence,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, fence, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkWaitForFences,
                           DECL(VkDevice device, uint32_t fenceCount,
                                const VkFence* pFences, VkBool32 waitAll,
                                uint64_t timeout),
                           ARGS(device, fenceCount, pFences, waitAll, timeout))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkGetBufferMemoryRequirements,
                           DECL(VkDevice device, VkBuffer buffer,
                                VkMemoryRequirements* pMemoryRequirements),
                           ARGS(device, buffer, pMemoryRequirements))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkAllocateMemory,
                           DECL(VkDevice device,
                                const VkMemoryAllocateInfo* pAllocateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkDeviceMemory* pMemory),
                           ARGS(device, pAllocateInfo, pAllocator, pMemory))

IREE_HAL_VULKAN_DEVICE_PFN(
    VkResult, vkGetMemoryHostPointerPropertiesEXT,
    DECL(VkDevice device, VkExternalMemoryHandleTypeFlagBits handleType,
         const void* pHostPointer,
         VkMemoryHostPointerPropertiesEXT* pMemoryHostPointerProperties),
    ARGS(device, handleType, pHostPointer, pMemoryHostPointerProperties))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkFreeMemory,
                           DECL(VkDevice device, VkDeviceMemory memory,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, memory, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkBindBufferMemory,
                           DECL(VkDevice device, VkBuffer buffer,
                                VkDeviceMemory memory,
                                VkDeviceSize memoryOffset),
                           ARGS(device, buffer, memory, memoryOffset))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkQueueBindSparse,
                           DECL(VkQueue queue, uint32_t bindInfoCount,
                                const VkBindSparseInfo* pBindInfo,
                                VkFence fence),
                           ARGS(queue, bindInfoCount, pBindInfo, fence))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkQueueSubmit2,
                           DECL(VkQueue queue, uint32_t submitCount,
                                const VkSubmitInfo2* pSubmits, VkFence fence),
                           ARGS(queue, submitCount, pSubmits, fence))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateCommandPool,
                           DECL(VkDevice device,
                                const VkCommandPoolCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkCommandPool* pCommandPool),
                           ARGS(device, pCreateInfo, pAllocator, pCommandPool))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyCommandPool,
                           DECL(VkDevice device, VkCommandPool commandPool,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, commandPool, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateQueryPool,
                           DECL(VkDevice device,
                                const VkQueryPoolCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkQueryPool* pQueryPool),
                           ARGS(device, pCreateInfo, pAllocator, pQueryPool))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyQueryPool,
                           DECL(VkDevice device, VkQueryPool queryPool,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, queryPool, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkGetQueryPoolResults,
                           DECL(VkDevice device, VkQueryPool queryPool,
                                uint32_t firstQuery, uint32_t queryCount,
                                size_t dataSize, void* pData,
                                VkDeviceSize stride, VkQueryResultFlags flags),
                           ARGS(device, queryPool, firstQuery, queryCount,
                                dataSize, pData, stride, flags))

IREE_HAL_VULKAN_DEVICE_PFN(
    VkResult, vkAllocateCommandBuffers,
    DECL(VkDevice device, const VkCommandBufferAllocateInfo* pAllocateInfo,
         VkCommandBuffer* pCommandBuffers),
    ARGS(device, pAllocateInfo, pCommandBuffers))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkFreeCommandBuffers,
                           DECL(VkDevice device, VkCommandPool commandPool,
                                uint32_t commandBufferCount,
                                const VkCommandBuffer* pCommandBuffers),
                           ARGS(device, commandPool, commandBufferCount,
                                pCommandBuffers))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkResetCommandBuffer,
                           DECL(VkCommandBuffer commandBuffer,
                                VkCommandBufferResetFlags flags),
                           ARGS(commandBuffer, flags))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkBeginCommandBuffer,
                           DECL(VkCommandBuffer commandBuffer,
                                const VkCommandBufferBeginInfo* pBeginInfo),
                           ARGS(commandBuffer, pBeginInfo))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkEndCommandBuffer,
                           DECL(VkCommandBuffer commandBuffer),
                           ARGS(commandBuffer))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdResetQueryPool,
                           DECL(VkCommandBuffer commandBuffer,
                                VkQueryPool queryPool, uint32_t firstQuery,
                                uint32_t queryCount),
                           ARGS(commandBuffer, queryPool, firstQuery,
                                queryCount))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdWriteTimestamp2,
                           DECL(VkCommandBuffer commandBuffer,
                                VkPipelineStageFlags2 stage,
                                VkQueryPool queryPool, uint32_t query),
                           ARGS(commandBuffer, stage, queryPool, query))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdFillBuffer,
                           DECL(VkCommandBuffer commandBuffer,
                                VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                VkDeviceSize size, uint32_t data),
                           ARGS(commandBuffer, dstBuffer, dstOffset, size,
                                data))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdUpdateBuffer,
                           DECL(VkCommandBuffer commandBuffer,
                                VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                VkDeviceSize dataSize, const void* pData),
                           ARGS(commandBuffer, dstBuffer, dstOffset, dataSize,
                                pData))

IREE_HAL_VULKAN_DEVICE_PFN(
    void, vkCmdCopyBuffer,
    DECL(VkCommandBuffer commandBuffer, VkBuffer srcBuffer, VkBuffer dstBuffer,
         uint32_t regionCount, const VkBufferCopy* pRegions),
    ARGS(commandBuffer, srcBuffer, dstBuffer, regionCount, pRegions))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdPipelineBarrier2,
                           DECL(VkCommandBuffer commandBuffer,
                                const VkDependencyInfo* pDependencyInfo),
                           ARGS(commandBuffer, pDependencyInfo))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkMapMemory,
                           DECL(VkDevice device, VkDeviceMemory memory,
                                VkDeviceSize offset, VkDeviceSize size,
                                VkMemoryMapFlags flags, void** ppData),
                           ARGS(device, memory, offset, size, flags, ppData))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkUnmapMemory,
                           DECL(VkDevice device, VkDeviceMemory memory),
                           ARGS(device, memory))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkInvalidateMappedMemoryRanges,
                           DECL(VkDevice device, uint32_t memoryRangeCount,
                                const VkMappedMemoryRange* pMemoryRanges),
                           ARGS(device, memoryRangeCount, pMemoryRanges))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkFlushMappedMemoryRanges,
                           DECL(VkDevice device, uint32_t memoryRangeCount,
                                const VkMappedMemoryRange* pMemoryRanges),
                           ARGS(device, memoryRangeCount, pMemoryRanges))

IREE_HAL_VULKAN_DEVICE_PFN(VkDeviceAddress, vkGetBufferDeviceAddress,
                           DECL(VkDevice device,
                                const VkBufferDeviceAddressInfo* pInfo),
                           ARGS(device, pInfo))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateSemaphore,
                           DECL(VkDevice device,
                                const VkSemaphoreCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkSemaphore* pSemaphore),
                           ARGS(device, pCreateInfo, pAllocator, pSemaphore))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroySemaphore,
                           DECL(VkDevice device, VkSemaphore semaphore,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, semaphore, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkGetSemaphoreCounterValue,
                           DECL(VkDevice device, VkSemaphore semaphore,
                                uint64_t* pValue),
                           ARGS(device, semaphore, pValue))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkSignalSemaphore,
                           DECL(VkDevice device,
                                const VkSemaphoreSignalInfo* pSignalInfo),
                           ARGS(device, pSignalInfo))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkWaitSemaphores,
                           DECL(VkDevice device,
                                const VkSemaphoreWaitInfo* pWaitInfo,
                                uint64_t timeout),
                           ARGS(device, pWaitInfo, timeout))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreatePipelineCache,
                           DECL(VkDevice device,
                                const VkPipelineCacheCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkPipelineCache* pPipelineCache),
                           ARGS(device, pCreateInfo, pAllocator,
                                pPipelineCache))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyPipelineCache,
                           DECL(VkDevice device, VkPipelineCache pipelineCache,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, pipelineCache, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(
    VkResult, vkCreateDescriptorSetLayout,
    DECL(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo,
         const VkAllocationCallbacks* pAllocator,
         VkDescriptorSetLayout* pSetLayout),
    ARGS(device, pCreateInfo, pAllocator, pSetLayout))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyDescriptorSetLayout,
                           DECL(VkDevice device,
                                VkDescriptorSetLayout descriptorSetLayout,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, descriptorSetLayout, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateDescriptorPool,
                           DECL(VkDevice device,
                                const VkDescriptorPoolCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkDescriptorPool* pDescriptorPool),
                           ARGS(device, pCreateInfo, pAllocator,
                                pDescriptorPool))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyDescriptorPool,
                           DECL(VkDevice device,
                                VkDescriptorPool descriptorPool,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, descriptorPool, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(
    VkResult, vkAllocateDescriptorSets,
    DECL(VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo,
         VkDescriptorSet* pDescriptorSets),
    ARGS(device, pAllocateInfo, pDescriptorSets))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkUpdateDescriptorSets,
                           DECL(VkDevice device, uint32_t descriptorWriteCount,
                                const VkWriteDescriptorSet* pDescriptorWrites,
                                uint32_t descriptorCopyCount,
                                const VkCopyDescriptorSet* pDescriptorCopies),
                           ARGS(device, descriptorWriteCount, pDescriptorWrites,
                                descriptorCopyCount, pDescriptorCopies))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreatePipelineLayout,
                           DECL(VkDevice device,
                                const VkPipelineLayoutCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkPipelineLayout* pPipelineLayout),
                           ARGS(device, pCreateInfo, pAllocator,
                                pPipelineLayout))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyPipelineLayout,
                           DECL(VkDevice device,
                                VkPipelineLayout pipelineLayout,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, pipelineLayout, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateShaderModule,
                           DECL(VkDevice device,
                                const VkShaderModuleCreateInfo* pCreateInfo,
                                const VkAllocationCallbacks* pAllocator,
                                VkShaderModule* pShaderModule),
                           ARGS(device, pCreateInfo, pAllocator, pShaderModule))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyShaderModule,
                           DECL(VkDevice device, VkShaderModule shaderModule,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, shaderModule, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(VkResult, vkCreateComputePipelines,
                           DECL(VkDevice device, VkPipelineCache pipelineCache,
                                uint32_t createInfoCount,
                                const VkComputePipelineCreateInfo* pCreateInfos,
                                const VkAllocationCallbacks* pAllocator,
                                VkPipeline* pPipelines),
                           ARGS(device, pipelineCache, createInfoCount,
                                pCreateInfos, pAllocator, pPipelines))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkDestroyPipeline,
                           DECL(VkDevice device, VkPipeline pipeline,
                                const VkAllocationCallbacks* pAllocator),
                           ARGS(device, pipeline, pAllocator))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdBindPipeline,
                           DECL(VkCommandBuffer commandBuffer,
                                VkPipelineBindPoint pipelineBindPoint,
                                VkPipeline pipeline),
                           ARGS(commandBuffer, pipelineBindPoint, pipeline))

IREE_HAL_VULKAN_DEVICE_PFN(
    void, vkCmdBindDescriptorSets,
    DECL(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
         VkPipelineLayout layout, uint32_t firstSet,
         uint32_t descriptorSetCount, const VkDescriptorSet* pDescriptorSets,
         uint32_t dynamicOffsetCount, const uint32_t* pDynamicOffsets),
    ARGS(commandBuffer, pipelineBindPoint, layout, firstSet, descriptorSetCount,
         pDescriptorSets, dynamicOffsetCount, pDynamicOffsets))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdPushConstants,
                           DECL(VkCommandBuffer commandBuffer,
                                VkPipelineLayout layout,
                                VkShaderStageFlags stageFlags, uint32_t offset,
                                uint32_t size, const void* pValues),
                           ARGS(commandBuffer, layout, stageFlags, offset, size,
                                pValues))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdDispatch,
                           DECL(VkCommandBuffer commandBuffer,
                                uint32_t groupCountX, uint32_t groupCountY,
                                uint32_t groupCountZ),
                           ARGS(commandBuffer, groupCountX, groupCountY,
                                groupCountZ))

IREE_HAL_VULKAN_DEVICE_PFN(void, vkCmdDispatchIndirect,
                           DECL(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                VkDeviceSize offset),
                           ARGS(commandBuffer, buffer, offset))

IREE_HAL_VULKAN_DEVICE_OPTIONAL_PFN(
    VkResult, vkGetCalibratedTimestampsEXT,
    DECL(VkDevice device, uint32_t timestampCount,
         const VkCalibratedTimestampInfoEXT* pTimestampInfos,
         uint64_t* pTimestamps, uint64_t* pMaxDeviation),
    ARGS(device, timestampCount, pTimestampInfos, pTimestamps, pMaxDeviation))

#if defined(IREE_HAL_VULKAN_UNDEFINE_DEVICE_OPTIONAL_PFN)
#undef IREE_HAL_VULKAN_DEVICE_OPTIONAL_PFN
#undef IREE_HAL_VULKAN_UNDEFINE_DEVICE_OPTIONAL_PFN
#endif  // IREE_HAL_VULKAN_UNDEFINE_DEVICE_OPTIONAL_PFN

#if defined(IREE_HAL_VULKAN_UNDEFINE_INSTANCE_OPTIONAL_PFN)
#undef IREE_HAL_VULKAN_INSTANCE_OPTIONAL_PFN
#undef IREE_HAL_VULKAN_UNDEFINE_INSTANCE_OPTIONAL_PFN
#endif  // IREE_HAL_VULKAN_UNDEFINE_INSTANCE_OPTIONAL_PFN
