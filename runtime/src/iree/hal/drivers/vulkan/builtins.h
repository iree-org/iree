// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_BUILTINS_H_
#define IREE_HAL_DRIVERS_VULKAN_BUILTINS_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_vulkan_physical_device_snapshot_t
    iree_hal_vulkan_physical_device_snapshot_t;

// Device-owned Vulkan built-in pipelines used for command polyfills.
typedef struct iree_hal_vulkan_builtins_t {
  // Device-level Vulkan dispatch table copied at initialization.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device owning all built-in handles.
  VkDevice logical_device;

  // Required descriptor offset alignment for storage buffer descriptors.
  VkDeviceSize min_storage_buffer_offset_alignment;

  // Descriptor set layout for built-in storage-buffer operands.
  VkDescriptorSetLayout storage_buffer_descriptor_set_layout;

  // Pipeline layout for built-in storage-buffer patch pipelines.
  VkPipelineLayout storage_buffer_pipeline_layout;

  // Compute pipeline patching partial dwords for unaligned fills.
  VkPipeline fill_pipeline;

  // Compute pipeline patching partial dwords for unaligned updates.
  VkPipeline update_pipeline;
} iree_hal_vulkan_builtins_t;

// Initializes built-in Vulkan pipelines for |logical_device|.
iree_status_t iree_hal_vulkan_builtins_initialize(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_physical_device_snapshot_t* physical_device,
    iree_hal_vulkan_builtins_t* out_builtins);

// Deinitializes built-in Vulkan pipelines and releases device handles.
void iree_hal_vulkan_builtins_deinitialize(
    iree_hal_vulkan_builtins_t* builtins);

// Records shader patches for the unaligned edges of a buffer fill.
//
// The aligned interior, if any, remains the caller's responsibility and should
// use vkCmdFillBuffer. |target_offset| is relative to |target_buffer|.
iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, VkBuffer target_buffer,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* pattern,
    iree_host_size_t pattern_length);

// Returns the descriptor set count required for an unaligned fill patch.
uint32_t iree_hal_vulkan_builtins_fill_unaligned_descriptor_set_count(
    VkDeviceSize target_offset, VkDeviceSize length);

// Records shader patches for the unaligned edges using pre-leased descriptors.
iree_status_t iree_hal_vulkan_builtins_record_fill_unaligned_descriptor_sets(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* pattern, iree_host_size_t pattern_length);

// Returns the descriptor set count required for an unaligned update patch.
uint32_t iree_hal_vulkan_builtins_update_unaligned_descriptor_set_count(
    VkDeviceSize target_offset, VkDeviceSize length);

// Records shader patches for the unaligned edges of a buffer update.
//
// The aligned interior, if any, remains the caller's responsibility and should
// use vkCmdUpdateBuffer in aligned chunks. |target_offset| is relative to
// |target_buffer|.
iree_status_t iree_hal_vulkan_builtins_record_update_unaligned(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    VkDescriptorPool descriptor_pool, VkBuffer target_buffer,
    VkDeviceSize target_offset, VkDeviceSize length, const uint8_t* source_data,
    iree_host_size_t source_data_length);

// Records shader patches for the unaligned edges of a buffer update.
iree_status_t iree_hal_vulkan_builtins_record_update_unaligned_descriptor_sets(
    const iree_hal_vulkan_builtins_t* builtins, VkCommandBuffer command_buffer,
    const VkDescriptorSet* descriptor_sets, uint32_t descriptor_set_count,
    VkBuffer target_buffer, VkDeviceSize target_offset, VkDeviceSize length,
    const uint8_t* source_data, iree_host_size_t source_data_length);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_BUILTINS_H_
