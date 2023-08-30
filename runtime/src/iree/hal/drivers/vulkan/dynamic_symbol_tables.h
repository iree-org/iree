// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Use these tables whenever enumerating all functions in the Vulkan API is
// required. In most cases IREE_VULKAN_DYNAMIC_SYMBOL_TABLES is the right
// choice (includes both common and enabled platform-specific functions).
//
// Table macros are designed to take two macros: one for each instance-specific
// function and one for each device-specific function. These macros are also
// passed a requirement flag that enables compile-time exclusion of methods that
// are not used in the binary. If you find yourself getting compilation errors
// on missing methods you probably need to change it in the tables below from
// EXCLUDED to REQUIRED or OPTIONAL.
//
// Define to get instance-specific functions:
// #define INS_PFN(requirement, function_name)
//
// Define to get device-specific functions:
// #define DEV_PFN(requirement, function_name)
//
// requirement is one of REQUIRED, OPTIONAL, or EXCLUDED.

#ifndef IREE_HAL_DRIVERS_VULKAN_DYNAMIC_SYMBOL_TABLES_H_
#define IREE_HAL_DRIVERS_VULKAN_DYNAMIC_SYMBOL_TABLES_H_

namespace iree {
namespace hal {
namespace vulkan {

// Defines the list of symbols that can be queried from vkGetInstanceProcAddr
// before Vulkan instance creation.
#define IREE_VULKAN_DYNAMIC_SYMBOL_INSTANCELESS_TABLE(INS_PFN) \
  INS_PFN(REQUIRED, vkCreateInstance)                          \
  INS_PFN(REQUIRED, vkEnumerateInstanceExtensionProperties)    \
  INS_PFN(REQUIRED, vkEnumerateInstanceLayerProperties)        \
  INS_PFN(OPTIONAL, vkEnumerateInstanceVersion)

// Defines the list of instance/device symbols that are queried from
// vkGetInstanceProcAddr/vkGetDeviceProcAddr after Vulkan instance/device
// creation.
#define IREE_VULKAN_DYNAMIC_SYMBOL_COMMON_TABLE(INS_PFN, DEV_PFN)       \
  DEV_PFN(REQUIRED, vkBeginCommandBuffer)                               \
  DEV_PFN(EXCLUDED, vkCmdBeginConditionalRenderingEXT)                  \
  DEV_PFN(OPTIONAL, vkCmdBeginDebugUtilsLabelEXT)                       \
  DEV_PFN(EXCLUDED, vkCmdBeginQuery)                                    \
  DEV_PFN(EXCLUDED, vkCmdBeginQueryIndexedEXT)                          \
  DEV_PFN(EXCLUDED, vkCmdBeginRenderPass)                               \
  DEV_PFN(EXCLUDED, vkCmdBeginRenderPass2KHR)                           \
  DEV_PFN(EXCLUDED, vkCmdBeginTransformFeedbackEXT)                     \
  DEV_PFN(REQUIRED, vkCmdBindDescriptorSets)                            \
  DEV_PFN(EXCLUDED, vkCmdBindIndexBuffer)                               \
  DEV_PFN(REQUIRED, vkCmdBindPipeline)                                  \
  DEV_PFN(EXCLUDED, vkCmdBindShadingRateImageNV)                        \
  DEV_PFN(EXCLUDED, vkCmdBindTransformFeedbackBuffersEXT)               \
  DEV_PFN(EXCLUDED, vkCmdBindVertexBuffers)                             \
  DEV_PFN(EXCLUDED, vkCmdBlitImage)                                     \
  DEV_PFN(EXCLUDED, vkCmdBuildAccelerationStructureNV)                  \
  DEV_PFN(EXCLUDED, vkCmdClearAttachments)                              \
  DEV_PFN(EXCLUDED, vkCmdClearColorImage)                               \
  DEV_PFN(EXCLUDED, vkCmdClearDepthStencilImage)                        \
  DEV_PFN(EXCLUDED, vkCmdCopyAccelerationStructureNV)                   \
  DEV_PFN(REQUIRED, vkCmdCopyBuffer)                                    \
  DEV_PFN(EXCLUDED, vkCmdCopyBufferToImage)                             \
  DEV_PFN(EXCLUDED, vkCmdCopyImage)                                     \
  DEV_PFN(EXCLUDED, vkCmdCopyImageToBuffer)                             \
  DEV_PFN(EXCLUDED, vkCmdCopyQueryPoolResults)                          \
  DEV_PFN(EXCLUDED, vkCmdDebugMarkerBeginEXT)                           \
  DEV_PFN(EXCLUDED, vkCmdDebugMarkerEndEXT)                             \
  DEV_PFN(EXCLUDED, vkCmdDebugMarkerInsertEXT)                          \
  DEV_PFN(REQUIRED, vkCmdDispatch)                                      \
  DEV_PFN(EXCLUDED, vkCmdDispatchBase)                                  \
  DEV_PFN(EXCLUDED, vkCmdDispatchBaseKHR)                               \
  DEV_PFN(REQUIRED, vkCmdDispatchIndirect)                              \
  DEV_PFN(EXCLUDED, vkCmdDraw)                                          \
  DEV_PFN(EXCLUDED, vkCmdDrawIndexed)                                   \
  DEV_PFN(EXCLUDED, vkCmdDrawIndexedIndirect)                           \
  DEV_PFN(EXCLUDED, vkCmdDrawIndexedIndirectCountAMD)                   \
  DEV_PFN(EXCLUDED, vkCmdDrawIndexedIndirectCountKHR)                   \
  DEV_PFN(EXCLUDED, vkCmdDrawIndirect)                                  \
  DEV_PFN(EXCLUDED, vkCmdDrawIndirectByteCountEXT)                      \
  DEV_PFN(EXCLUDED, vkCmdDrawIndirectCountAMD)                          \
  DEV_PFN(EXCLUDED, vkCmdDrawIndirectCountKHR)                          \
  DEV_PFN(EXCLUDED, vkCmdDrawMeshTasksIndirectCountNV)                  \
  DEV_PFN(EXCLUDED, vkCmdDrawMeshTasksIndirectNV)                       \
  DEV_PFN(EXCLUDED, vkCmdDrawMeshTasksNV)                               \
  DEV_PFN(EXCLUDED, vkCmdEndConditionalRenderingEXT)                    \
  DEV_PFN(OPTIONAL, vkCmdEndDebugUtilsLabelEXT)                         \
  DEV_PFN(EXCLUDED, vkCmdEndQuery)                                      \
  DEV_PFN(EXCLUDED, vkCmdEndQueryIndexedEXT)                            \
  DEV_PFN(EXCLUDED, vkCmdEndRenderPass)                                 \
  DEV_PFN(EXCLUDED, vkCmdEndRenderPass2KHR)                             \
  DEV_PFN(EXCLUDED, vkCmdEndTransformFeedbackEXT)                       \
  DEV_PFN(REQUIRED, vkCmdExecuteCommands)                               \
  DEV_PFN(REQUIRED, vkCmdFillBuffer)                                    \
  DEV_PFN(OPTIONAL, vkCmdInsertDebugUtilsLabelEXT)                      \
  DEV_PFN(EXCLUDED, vkCmdNextSubpass)                                   \
  DEV_PFN(EXCLUDED, vkCmdNextSubpass2KHR)                               \
  DEV_PFN(REQUIRED, vkCmdPipelineBarrier)                               \
  DEV_PFN(EXCLUDED, vkCmdProcessCommandsNVX)                            \
  DEV_PFN(REQUIRED, vkCmdPushConstants)                                 \
  DEV_PFN(OPTIONAL, vkCmdPushDescriptorSetKHR)                          \
  DEV_PFN(EXCLUDED, vkCmdPushDescriptorSetWithTemplateKHR)              \
  DEV_PFN(EXCLUDED, vkCmdReserveSpaceForCommandsNVX)                    \
  DEV_PFN(REQUIRED, vkCmdResetEvent)                                    \
  DEV_PFN(REQUIRED, vkCmdResetQueryPool)                                \
  DEV_PFN(EXCLUDED, vkCmdResolveImage)                                  \
  DEV_PFN(EXCLUDED, vkCmdSetBlendConstants)                             \
  DEV_PFN(EXCLUDED, vkCmdSetCheckpointNV)                               \
  DEV_PFN(EXCLUDED, vkCmdSetCoarseSampleOrderNV)                        \
  DEV_PFN(EXCLUDED, vkCmdSetDepthBias)                                  \
  DEV_PFN(EXCLUDED, vkCmdSetDepthBounds)                                \
  DEV_PFN(EXCLUDED, vkCmdSetDeviceMask)                                 \
  DEV_PFN(EXCLUDED, vkCmdSetDeviceMaskKHR)                              \
  DEV_PFN(EXCLUDED, vkCmdSetDiscardRectangleEXT)                        \
  DEV_PFN(REQUIRED, vkCmdSetEvent)                                      \
  DEV_PFN(EXCLUDED, vkCmdSetExclusiveScissorNV)                         \
  DEV_PFN(EXCLUDED, vkCmdSetLineWidth)                                  \
  DEV_PFN(EXCLUDED, vkCmdSetSampleLocationsEXT)                         \
  DEV_PFN(EXCLUDED, vkCmdSetScissor)                                    \
  DEV_PFN(EXCLUDED, vkCmdSetStencilCompareMask)                         \
  DEV_PFN(EXCLUDED, vkCmdSetStencilReference)                           \
  DEV_PFN(EXCLUDED, vkCmdSetStencilWriteMask)                           \
  DEV_PFN(EXCLUDED, vkCmdSetViewport)                                   \
  DEV_PFN(EXCLUDED, vkCmdSetViewportShadingRatePaletteNV)               \
  DEV_PFN(EXCLUDED, vkCmdSetViewportWScalingNV)                         \
  DEV_PFN(EXCLUDED, vkCmdTraceRaysNV)                                   \
  DEV_PFN(REQUIRED, vkCmdUpdateBuffer)                                  \
  DEV_PFN(REQUIRED, vkCmdWaitEvents)                                    \
  DEV_PFN(EXCLUDED, vkCmdWriteAccelerationStructuresPropertiesNV)       \
  DEV_PFN(EXCLUDED, vkCmdWriteBufferMarkerAMD)                          \
  DEV_PFN(REQUIRED, vkCmdWriteTimestamp)                                \
  DEV_PFN(REQUIRED, vkEndCommandBuffer)                                 \
  DEV_PFN(EXCLUDED, vkResetCommandBuffer)                               \
  DEV_PFN(EXCLUDED, vkAcquireNextImage2KHR)                             \
  DEV_PFN(EXCLUDED, vkAcquireNextImageKHR)                              \
  DEV_PFN(REQUIRED, vkAllocateCommandBuffers)                           \
  DEV_PFN(REQUIRED, vkAllocateDescriptorSets)                           \
  DEV_PFN(REQUIRED, vkAllocateMemory)                                   \
  DEV_PFN(EXCLUDED, vkBindAccelerationStructureMemoryNV)                \
  DEV_PFN(REQUIRED, vkBindBufferMemory)                                 \
  DEV_PFN(EXCLUDED, vkBindBufferMemory2)                                \
  DEV_PFN(EXCLUDED, vkBindBufferMemory2KHR)                             \
  DEV_PFN(REQUIRED, vkBindImageMemory)                                  \
  DEV_PFN(EXCLUDED, vkBindImageMemory2)                                 \
  DEV_PFN(EXCLUDED, vkBindImageMemory2KHR)                              \
  DEV_PFN(EXCLUDED, vkCompileDeferredNV)                                \
  DEV_PFN(EXCLUDED, vkCreateAccelerationStructureNV)                    \
  DEV_PFN(REQUIRED, vkCreateBuffer)                                     \
  DEV_PFN(REQUIRED, vkCreateBufferView)                                 \
  DEV_PFN(REQUIRED, vkCreateCommandPool)                                \
  DEV_PFN(REQUIRED, vkCreateComputePipelines)                           \
  DEV_PFN(REQUIRED, vkCreateDescriptorPool)                             \
  DEV_PFN(REQUIRED, vkCreateDescriptorSetLayout)                        \
  DEV_PFN(EXCLUDED, vkCreateDescriptorUpdateTemplate)                   \
  DEV_PFN(EXCLUDED, vkCreateDescriptorUpdateTemplateKHR)                \
  DEV_PFN(REQUIRED, vkCreateEvent)                                      \
  DEV_PFN(REQUIRED, vkCreateFence)                                      \
  DEV_PFN(EXCLUDED, vkCreateFramebuffer)                                \
  DEV_PFN(EXCLUDED, vkCreateGraphicsPipelines)                          \
  DEV_PFN(REQUIRED, vkCreateImage)                                      \
  DEV_PFN(EXCLUDED, vkCreateImageView)                                  \
  DEV_PFN(EXCLUDED, vkCreateIndirectCommandsLayoutNVX)                  \
  DEV_PFN(EXCLUDED, vkCreateObjectTableNVX)                             \
  DEV_PFN(REQUIRED, vkCreatePipelineCache)                              \
  DEV_PFN(REQUIRED, vkCreatePipelineLayout)                             \
  DEV_PFN(REQUIRED, vkCreateQueryPool)                                  \
  DEV_PFN(EXCLUDED, vkCreateRayTracingPipelinesNV)                      \
  DEV_PFN(EXCLUDED, vkCreateRenderPass)                                 \
  DEV_PFN(EXCLUDED, vkCreateRenderPass2KHR)                             \
  DEV_PFN(EXCLUDED, vkCreateSampler)                                    \
  DEV_PFN(EXCLUDED, vkCreateSamplerYcbcrConversion)                     \
  DEV_PFN(EXCLUDED, vkCreateSamplerYcbcrConversionKHR)                  \
  DEV_PFN(REQUIRED, vkCreateSemaphore)                                  \
  DEV_PFN(REQUIRED, vkCreateShaderModule)                               \
  DEV_PFN(EXCLUDED, vkCreateSharedSwapchainsKHR)                        \
  DEV_PFN(EXCLUDED, vkCreateSwapchainKHR)                               \
  DEV_PFN(EXCLUDED, vkCreateValidationCacheEXT)                         \
  DEV_PFN(EXCLUDED, vkDebugMarkerSetObjectNameEXT)                      \
  DEV_PFN(EXCLUDED, vkDebugMarkerSetObjectTagEXT)                       \
  DEV_PFN(EXCLUDED, vkDestroyAccelerationStructureNV)                   \
  DEV_PFN(REQUIRED, vkDestroyBuffer)                                    \
  DEV_PFN(REQUIRED, vkDestroyBufferView)                                \
  DEV_PFN(REQUIRED, vkDestroyCommandPool)                               \
  DEV_PFN(REQUIRED, vkDestroyDescriptorPool)                            \
  DEV_PFN(REQUIRED, vkDestroyDescriptorSetLayout)                       \
  DEV_PFN(EXCLUDED, vkDestroyDescriptorUpdateTemplate)                  \
  DEV_PFN(EXCLUDED, vkDestroyDescriptorUpdateTemplateKHR)               \
  DEV_PFN(REQUIRED, vkDestroyDevice)                                    \
  DEV_PFN(REQUIRED, vkDestroyEvent)                                     \
  DEV_PFN(REQUIRED, vkDestroyFence)                                     \
  DEV_PFN(EXCLUDED, vkDestroyFramebuffer)                               \
  DEV_PFN(REQUIRED, vkDestroyImage)                                     \
  DEV_PFN(EXCLUDED, vkDestroyImageView)                                 \
  DEV_PFN(EXCLUDED, vkDestroyIndirectCommandsLayoutNVX)                 \
  DEV_PFN(EXCLUDED, vkDestroyObjectTableNVX)                            \
  DEV_PFN(REQUIRED, vkDestroyPipeline)                                  \
  DEV_PFN(REQUIRED, vkDestroyPipelineCache)                             \
  DEV_PFN(REQUIRED, vkDestroyPipelineLayout)                            \
  DEV_PFN(REQUIRED, vkDestroyQueryPool)                                 \
  DEV_PFN(EXCLUDED, vkDestroyRenderPass)                                \
  DEV_PFN(EXCLUDED, vkDestroySampler)                                   \
  DEV_PFN(EXCLUDED, vkDestroySamplerYcbcrConversion)                    \
  DEV_PFN(EXCLUDED, vkDestroySamplerYcbcrConversionKHR)                 \
  DEV_PFN(REQUIRED, vkDestroySemaphore)                                 \
  DEV_PFN(REQUIRED, vkDestroyShaderModule)                              \
  DEV_PFN(EXCLUDED, vkDestroySwapchainKHR)                              \
  DEV_PFN(EXCLUDED, vkDestroyValidationCacheEXT)                        \
  DEV_PFN(REQUIRED, vkDeviceWaitIdle)                                   \
  DEV_PFN(EXCLUDED, vkDisplayPowerControlEXT)                           \
  DEV_PFN(REQUIRED, vkFlushMappedMemoryRanges)                          \
  DEV_PFN(REQUIRED, vkFreeCommandBuffers)                               \
  DEV_PFN(REQUIRED, vkFreeDescriptorSets)                               \
  DEV_PFN(REQUIRED, vkFreeMemory)                                       \
  DEV_PFN(EXCLUDED, vkGetAccelerationStructureHandleNV)                 \
  DEV_PFN(EXCLUDED, vkGetAccelerationStructureMemoryRequirementsNV)     \
  DEV_PFN(EXCLUDED, vkGetBufferDeviceAddressEXT)                        \
  DEV_PFN(REQUIRED, vkGetBufferMemoryRequirements)                      \
  DEV_PFN(EXCLUDED, vkGetBufferMemoryRequirements2)                     \
  DEV_PFN(EXCLUDED, vkGetBufferMemoryRequirements2KHR)                  \
  DEV_PFN(OPTIONAL, vkGetCalibratedTimestampsEXT)                       \
  DEV_PFN(EXCLUDED, vkGetDescriptorSetLayoutSupport)                    \
  DEV_PFN(EXCLUDED, vkGetDescriptorSetLayoutSupportKHR)                 \
  DEV_PFN(EXCLUDED, vkGetDeviceGroupPeerMemoryFeatures)                 \
  DEV_PFN(EXCLUDED, vkGetDeviceGroupPeerMemoryFeaturesKHR)              \
  DEV_PFN(EXCLUDED, vkGetDeviceGroupPresentCapabilitiesKHR)             \
  DEV_PFN(EXCLUDED, vkGetDeviceGroupSurfacePresentModesKHR)             \
  DEV_PFN(EXCLUDED, vkGetDeviceMemoryCommitment)                        \
  DEV_PFN(REQUIRED, vkGetDeviceQueue)                                   \
  DEV_PFN(EXCLUDED, vkGetDeviceQueue2)                                  \
  DEV_PFN(REQUIRED, vkGetEventStatus)                                   \
  DEV_PFN(OPTIONAL, vkGetFenceFdKHR)                                    \
  DEV_PFN(REQUIRED, vkGetFenceStatus)                                   \
  DEV_PFN(EXCLUDED, vkGetImageDrmFormatModifierPropertiesEXT)           \
  DEV_PFN(REQUIRED, vkGetImageMemoryRequirements)                       \
  DEV_PFN(EXCLUDED, vkGetImageMemoryRequirements2)                      \
  DEV_PFN(EXCLUDED, vkGetImageMemoryRequirements2KHR)                   \
  DEV_PFN(EXCLUDED, vkGetImageSparseMemoryRequirements)                 \
  DEV_PFN(EXCLUDED, vkGetImageSparseMemoryRequirements2)                \
  DEV_PFN(EXCLUDED, vkGetImageSparseMemoryRequirements2KHR)             \
  DEV_PFN(EXCLUDED, vkGetImageSubresourceLayout)                        \
  DEV_PFN(EXCLUDED, vkGetImageViewHandleNVX)                            \
  DEV_PFN(EXCLUDED, vkGetMemoryFdKHR)                                   \
  DEV_PFN(EXCLUDED, vkGetMemoryFdPropertiesKHR)                         \
  DEV_PFN(OPTIONAL, vkGetMemoryHostPointerPropertiesEXT)                \
  DEV_PFN(EXCLUDED, vkGetPastPresentationTimingGOOGLE)                  \
  DEV_PFN(REQUIRED, vkGetPipelineCacheData)                             \
  DEV_PFN(REQUIRED, vkGetQueryPoolResults)                              \
  DEV_PFN(EXCLUDED, vkGetRayTracingShaderGroupHandlesNV)                \
  DEV_PFN(EXCLUDED, vkGetRefreshCycleDurationGOOGLE)                    \
  DEV_PFN(EXCLUDED, vkGetRenderAreaGranularity)                         \
  DEV_PFN(OPTIONAL, vkGetSemaphoreFdKHR)                                \
  DEV_PFN(EXCLUDED, vkGetShaderInfoAMD)                                 \
  DEV_PFN(EXCLUDED, vkGetSwapchainCounterEXT)                           \
  DEV_PFN(EXCLUDED, vkGetSwapchainImagesKHR)                            \
  DEV_PFN(EXCLUDED, vkGetSwapchainStatusKHR)                            \
  DEV_PFN(EXCLUDED, vkGetValidationCacheDataEXT)                        \
  DEV_PFN(OPTIONAL, vkImportFenceFdKHR)                                 \
  DEV_PFN(OPTIONAL, vkImportSemaphoreFdKHR)                             \
  DEV_PFN(REQUIRED, vkInvalidateMappedMemoryRanges)                     \
  DEV_PFN(REQUIRED, vkMapMemory)                                        \
  DEV_PFN(REQUIRED, vkMergePipelineCaches)                              \
  DEV_PFN(EXCLUDED, vkMergeValidationCachesEXT)                         \
  DEV_PFN(EXCLUDED, vkRegisterDeviceEventEXT)                           \
  DEV_PFN(EXCLUDED, vkRegisterDisplayEventEXT)                          \
  DEV_PFN(EXCLUDED, vkRegisterObjectsNVX)                               \
  DEV_PFN(EXCLUDED, vkResetCommandPool)                                 \
  DEV_PFN(REQUIRED, vkResetDescriptorPool)                              \
  DEV_PFN(REQUIRED, vkResetEvent)                                       \
  DEV_PFN(REQUIRED, vkResetFences)                                      \
  DEV_PFN(OPTIONAL, vkResetQueryPool)                                   \
  DEV_PFN(OPTIONAL, vkResetQueryPoolEXT)                                \
  DEV_PFN(OPTIONAL, vkSetDebugUtilsObjectNameEXT)                       \
  DEV_PFN(OPTIONAL, vkSetDebugUtilsObjectTagEXT)                        \
  DEV_PFN(REQUIRED, vkSetEvent)                                         \
  DEV_PFN(EXCLUDED, vkSetHdrMetadataEXT)                                \
  DEV_PFN(EXCLUDED, vkSetLocalDimmingAMD)                               \
  DEV_PFN(EXCLUDED, vkTrimCommandPool)                                  \
  DEV_PFN(EXCLUDED, vkTrimCommandPoolKHR)                               \
  DEV_PFN(REQUIRED, vkUnmapMemory)                                      \
  DEV_PFN(EXCLUDED, vkUnregisterObjectsNVX)                             \
  DEV_PFN(EXCLUDED, vkUpdateDescriptorSetWithTemplate)                  \
  DEV_PFN(EXCLUDED, vkUpdateDescriptorSetWithTemplateKHR)               \
  DEV_PFN(REQUIRED, vkUpdateDescriptorSets)                             \
  DEV_PFN(REQUIRED, vkWaitForFences)                                    \
                                                                        \
  DEV_PFN(OPTIONAL, vkGetSemaphoreCounterValue)                         \
  DEV_PFN(OPTIONAL, vkGetSemaphoreCounterValueKHR)                      \
  DEV_PFN(OPTIONAL, vkWaitSemaphores)                                   \
  DEV_PFN(OPTIONAL, vkWaitSemaphoresKHR)                                \
  DEV_PFN(OPTIONAL, vkSignalSemaphore)                                  \
  DEV_PFN(OPTIONAL, vkSignalSemaphoreKHR)                               \
                                                                        \
  DEV_PFN(OPTIONAL, vkGetBufferDeviceAddress)                           \
  DEV_PFN(OPTIONAL, vkGetBufferDeviceAddressKHR)                        \
                                                                        \
  INS_PFN(EXCLUDED, vkCreateDebugReportCallbackEXT)                     \
  INS_PFN(OPTIONAL, vkCreateDebugUtilsMessengerEXT)                     \
  INS_PFN(EXCLUDED, vkCreateDisplayPlaneSurfaceKHR)                     \
  INS_PFN(EXCLUDED, vkCreateHeadlessSurfaceEXT)                         \
  INS_PFN(EXCLUDED, vkDebugReportMessageEXT)                            \
  INS_PFN(EXCLUDED, vkDestroyDebugReportCallbackEXT)                    \
  INS_PFN(OPTIONAL, vkDestroyDebugUtilsMessengerEXT)                    \
  INS_PFN(REQUIRED, vkDestroyInstance)                                  \
  INS_PFN(EXCLUDED, vkDestroySurfaceKHR)                                \
  INS_PFN(EXCLUDED, vkEnumeratePhysicalDeviceGroups)                    \
  INS_PFN(EXCLUDED, vkEnumeratePhysicalDeviceGroupsKHR)                 \
  INS_PFN(REQUIRED, vkEnumeratePhysicalDevices)                         \
  INS_PFN(EXCLUDED, vkSubmitDebugUtilsMessageEXT)                       \
  INS_PFN(REQUIRED, vkCreateDevice)                                     \
  INS_PFN(EXCLUDED, vkCreateDisplayModeKHR)                             \
  INS_PFN(REQUIRED, vkEnumerateDeviceExtensionProperties)               \
  INS_PFN(REQUIRED, vkEnumerateDeviceLayerProperties)                   \
  INS_PFN(EXCLUDED, vkGetDisplayModeProperties2KHR)                     \
  INS_PFN(EXCLUDED, vkGetDisplayModePropertiesKHR)                      \
  INS_PFN(EXCLUDED, vkGetDisplayPlaneCapabilities2KHR)                  \
  INS_PFN(EXCLUDED, vkGetDisplayPlaneCapabilitiesKHR)                   \
  INS_PFN(EXCLUDED, vkGetDisplayPlaneSupportedDisplaysKHR)              \
  INS_PFN(OPTIONAL, vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)     \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)   \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceDisplayPlaneProperties2KHR)      \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceDisplayPlanePropertiesKHR)       \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceDisplayProperties2KHR)           \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceDisplayPropertiesKHR)            \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceExternalBufferPropertiesKHR)     \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceExternalBufferProperties)        \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceExternalFenceProperties)         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceExternalFencePropertiesKHR)      \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceExternalImageFormatPropertiesNV) \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceExternalSemaphoreProperties)     \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceExternalSemaphorePropertiesKHR)  \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceFeatures)                        \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceFeatures2)                       \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceFeatures2KHR)                    \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceFormatProperties)                \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceFormatProperties2)               \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceFormatProperties2KHR)            \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceGeneratedCommandsPropertiesNVX)  \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceImageFormatProperties)           \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceImageFormatProperties2)          \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceImageFormatProperties2KHR)       \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceMemoryProperties)                \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceMemoryProperties2)               \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceMemoryProperties2KHR)            \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceMultisamplePropertiesEXT)        \
  INS_PFN(EXCLUDED, vkGetPhysicalDevicePresentRectanglesKHR)            \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceProperties)                      \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceProperties2)                     \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceProperties2KHR)                  \
  INS_PFN(REQUIRED, vkGetPhysicalDeviceQueueFamilyProperties)           \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceQueueFamilyProperties2)          \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceQueueFamilyProperties2KHR)       \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSparseImageFormatProperties)     \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSparseImageFormatProperties2)    \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSparseImageFormatProperties2KHR) \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfaceCapabilities2EXT)         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfaceCapabilities2KHR)         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfaceCapabilitiesKHR)          \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfaceFormats2KHR)              \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfaceFormatsKHR)               \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfacePresentModesKHR)          \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfaceSupportKHR)               \
  INS_PFN(EXCLUDED, vkReleaseDisplayEXT)                                \
  DEV_PFN(EXCLUDED, vkGetQueueCheckpointDataNV)                         \
  DEV_PFN(OPTIONAL, vkQueueBeginDebugUtilsLabelEXT)                     \
  DEV_PFN(OPTIONAL, vkQueueBindSparse)                                  \
  DEV_PFN(OPTIONAL, vkQueueEndDebugUtilsLabelEXT)                       \
  DEV_PFN(OPTIONAL, vkQueueInsertDebugUtilsLabelEXT)                    \
  DEV_PFN(EXCLUDED, vkQueuePresentKHR)                                  \
  DEV_PFN(REQUIRED, vkQueueSubmit)                                      \
  DEV_PFN(REQUIRED, vkQueueWaitIdle)

#ifdef VK_USE_PLATFORM_ANDROID_KHR
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_ANDROID_KHR(INS_PFN, DEV_PFN) \
  DEV_PFN(OPTIONAL, vkGetAndroidHardwareBufferPropertiesANDROID)       \
  DEV_PFN(OPTIONAL, vkGetMemoryAndroidHardwareBufferANDROID)           \
  INS_PFN(EXCLUDED, vkCreateAndroidSurfaceKHR)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_ANDROID_KHR(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_ANDROID_KHR

#ifdef VK_USE_PLATFORM_GGP
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_GGP(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateStreamDescriptorSurfaceGGP)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_GGP(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_GGP

#ifdef VK_USE_PLATFORM_IOS_MVK
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_IOS_MVK(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateIOSSurfaceMVK)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_IOS_MVK(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_IOS_MVK

#ifdef VK_USE_PLATFORM_FUCHSIA
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_FUSCHIA(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateImagePipeSurfaceFUCHSIA)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_FUSCHIA(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_FUCHSIA

#ifdef VK_USE_PLATFORM_MACOS_MVK
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_MACOS_MVK(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateMacOSSurfaceMVK)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_MACOS_MVK(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_MACOS_MVK

#ifdef VK_USE_PLATFORM_METAL_EXT
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_METAL_EXT(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateMetalSurfaceEXT)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_METAL_EXT(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_METAL_EXT

#ifdef VK_USE_PLATFORM_VI_NN
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_VI_NN(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateViSurfaceNN)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_VI_NN(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_VI_NN

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_WAYLAND_KHR(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateWaylandSurfaceKHR)                         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceWaylandPresentationSupportKHR)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_WAYLAND_KHR(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_WAYLAND_KHR

#ifdef VK_USE_PLATFORM_WIN32_KHR
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_WIN32_KHR(INS_PFN, DEV_PFN) \
  DEV_PFN(EXCLUDED, vkAcquireFullScreenExclusiveModeEXT)             \
  DEV_PFN(EXCLUDED, vkGetDeviceGroupSurfacePresentModes2EXT)         \
  DEV_PFN(OPTIONAL, vkGetFenceWin32HandleKHR)                        \
  DEV_PFN(EXCLUDED, vkGetMemoryWin32HandleKHR)                       \
  DEV_PFN(EXCLUDED, vkGetMemoryWin32HandleNV)                        \
  DEV_PFN(EXCLUDED, vkGetMemoryWin32HandlePropertiesKHR)             \
  DEV_PFN(OPTIONAL, vkGetSemaphoreWin32HandleKHR)                    \
  DEV_PFN(OPTIONAL, vkImportFenceWin32HandleKHR)                     \
  DEV_PFN(OPTIONAL, vkImportSemaphoreWin32HandleKHR)                 \
  DEV_PFN(EXCLUDED, vkReleaseFullScreenExclusiveModeEXT)             \
  INS_PFN(EXCLUDED, vkCreateWin32SurfaceKHR)                         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceSurfacePresentModes2EXT)      \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceWin32PresentationSupportKHR)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_WIN32_KHR(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_WIN32_KHR

#ifdef VK_USE_PLATFORM_XCB_KHR
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XCB_KHR(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateXcbSurfaceKHR)                         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceXcbPresentationSupportKHR)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XCB_KHR(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_XCB_KHR

#ifdef VK_USE_PLATFORM_XLIB_KHR
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XLIB_KHR(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkCreateXlibSurfaceKHR)                         \
  INS_PFN(EXCLUDED, vkGetPhysicalDeviceXlibPresentationSupportKHR)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XLIB_KHR(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_XLIB_KHR

#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XLIB_XRANDR_EXT(INS_PFN, DEV_PFN) \
  INS_PFN(EXCLUDED, vkAcquireXlibDisplayEXT)                               \
  INS_PFN(EXCLUDED, vkGetRandROutputDisplayEXT)
#else
#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XLIB_XRANDR_EXT(INS_PFN, DEV_PFN)
#endif  // VK_USE_PLATFORM_XLIB_XRANDR_EXT

#define IREE_VULKAN_DYNAMIC_SYMBOL_PLATFORM_TABLES(INS_PFN, DEV_PFN) \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_ANDROID_KHR(INS_PFN, DEV_PFN)     \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_GGP(INS_PFN, DEV_PFN)             \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_IOS_MVK(INS_PFN, DEV_PFN)         \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_FUSCHIA(INS_PFN, DEV_PFN)         \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_MACOS_MVK(INS_PFN, DEV_PFN)       \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_METAL_EXT(INS_PFN, DEV_PFN)       \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_VI_NN(INS_PFN, DEV_PFN)           \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_WAYLAND_KHR(INS_PFN, DEV_PFN)     \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_WIN32_KHR(INS_PFN, DEV_PFN)       \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XCB_KHR(INS_PFN, DEV_PFN)         \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XLIB_KHR(INS_PFN, DEV_PFN)        \
  IREE_VULKAN_DYNAMIC_SYMBOL_TABLE_XLIB_XRANDR_EXT(INS_PFN, DEV_PFN)

#define IREE_VULKAN_DYNAMIC_SYMBOL_INSTANCE_DEVICE_TABLES(INS_PFN, DEV_PFN) \
  IREE_VULKAN_DYNAMIC_SYMBOL_COMMON_TABLE(INS_PFN, DEV_PFN)                 \
  IREE_VULKAN_DYNAMIC_SYMBOL_PLATFORM_TABLES(INS_PFN, DEV_PFN)

#define IREE_VULKAN_DYNAMIC_SYMBOL_TABLES(INS_PFN, DEV_PFN) \
  IREE_VULKAN_DYNAMIC_SYMBOL_INSTANCELESS_TABLE(INS_PFN)    \
  IREE_VULKAN_DYNAMIC_SYMBOL_COMMON_TABLE(INS_PFN, DEV_PFN) \
  IREE_VULKAN_DYNAMIC_SYMBOL_PLATFORM_TABLES(INS_PFN, DEV_PFN)

}  // namespace vulkan
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DRIVERS_VULKAN_DYNAMIC_SYMBOL_TABLES_H_
