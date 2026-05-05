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
#include "iree/hal/local/profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a Vulkan HAL command buffer.
//
// |device_allocator| is retained and used for command-owned staging buffers
// needed when recorded transfer commands cannot be represented directly by
// Vulkan.
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

// Returns the number of dispatch commands recorded in |command_buffer|.
iree_host_size_t iree_hal_vulkan_command_buffer_dispatch_count(
    iree_hal_command_buffer_t* command_buffer);

// Emits executable/export metadata referenced by recorded dispatch commands.
iree_status_t iree_hal_vulkan_command_buffer_record_profile_metadata(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id);

// Replays a recorded Vulkan command buffer using host-mediated operations.
iree_status_t iree_hal_vulkan_command_buffer_replay_host(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table);

// Appends dispatch profile events from timestamp pairs recorded around each
// dispatch command.
//
// |dispatch_ticks| contains |dispatch_count| adjacent start/end tick pairs.
// |command_buffer_id| is 0 for direct queue dispatches and nonzero for reusable
// command-buffer dispatches.
iree_status_t iree_hal_vulkan_command_buffer_append_dispatch_profile_events(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t submission_id,
    uint64_t command_buffer_id, const uint64_t* dispatch_ticks,
    iree_host_size_t dispatch_count);

#define IREE_HAL_VULKAN_PROFILE_QUERY_ABSENT UINT32_MAX

// Optional timestamp marker injected around one dispatch command.
typedef struct iree_hal_vulkan_command_buffer_dispatch_profile_marker_t {
  // Query pool receiving start/end timestamps.
  VkQueryPool query_pool;

  // Query index written before the dispatch payload.
  uint32_t start_query;

  // Query index written after the dispatch payload.
  uint32_t end_query;
} iree_hal_vulkan_command_buffer_dispatch_profile_marker_t;

// Optional timestamp plan injected into native command-buffer payloads.
typedef struct iree_hal_vulkan_command_buffer_profile_marker_t {
  // Query pool receiving timestamp records, or VK_NULL_HANDLE when absent.
  VkQueryPool query_pool;

  // Number of query slots allocated in query_pool.
  uint32_t query_count;

  // Query index written before the command-buffer payload, or ABSENT.
  uint32_t queue_start_query;

  // Query index written after the command-buffer payload, or ABSENT.
  uint32_t queue_end_query;

  // First query slot for per-dispatch timestamp pairs, or ABSENT.
  uint32_t dispatch_base_query;

  // Number of dispatch commands with per-dispatch timestamp pairs.
  uint32_t dispatch_query_count;
} iree_hal_vulkan_command_buffer_profile_marker_t;

// Records Vulkan-native commands into |native_command_buffer|.
//
// |profile_marker| may be NULL. When present, the query pool is reset and
// timestamped inside |native_command_buffer| around the requested queue payload
// and dispatch commands.
//
// Descriptor sets are allocated from a transient descriptor pool returned in
// |out_descriptor_pool|. The caller must keep that pool alive until
// |native_command_buffer| is no longer executing, then destroy it.
iree_status_t iree_hal_vulkan_command_buffer_record_native(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    VkCommandBuffer native_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_buffer_profile_marker_t* profile_marker,
    iree_allocator_t host_allocator, VkDescriptorPool* out_descriptor_pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_
