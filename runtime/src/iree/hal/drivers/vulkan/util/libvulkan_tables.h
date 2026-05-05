// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
