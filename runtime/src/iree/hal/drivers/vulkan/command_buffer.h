// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Vulkan HAL command buffer.
iree_status_t iree_hal_vulkan_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a Vulkan command buffer.
bool iree_hal_vulkan_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns true if |command_buffer| contains no recorded device commands.
bool iree_hal_vulkan_command_buffer_is_empty(
    iree_hal_command_buffer_t* command_buffer);

// Returns true if |command_buffer| contains host-replayed commands.
bool iree_hal_vulkan_command_buffer_has_host_commands(
    iree_hal_command_buffer_t* command_buffer);

// Returns true if |command_buffer| contains Vulkan-native commands.
bool iree_hal_vulkan_command_buffer_has_native_commands(
    iree_hal_command_buffer_t* command_buffer);

// Replays a recorded Vulkan command buffer using host-mediated operations.
iree_status_t iree_hal_vulkan_command_buffer_replay_host(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table);

// Records Vulkan-native commands into |native_command_buffer|.
//
// Descriptor sets are allocated from a transient descriptor pool returned in
// |out_descriptor_pool|. The caller must keep that pool alive until
// |native_command_buffer| is no longer executing, then destroy it.
iree_status_t iree_hal_vulkan_command_buffer_record_native(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_allocator_t host_allocator, VkDescriptorPool* out_descriptor_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_
