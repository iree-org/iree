// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_
#define IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/vulkan/builtins.h"
#include "iree/hal/drivers/vulkan/debug_utils.h"
#include "iree/hal/drivers/vulkan/util/libvulkan.h"
#include "iree/hal/local/profile.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

// Creates a Vulkan HAL command buffer.
iree_status_t iree_hal_vulkan_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* command_buffer_block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns true if |command_buffer| is a Vulkan command buffer.
bool iree_hal_vulkan_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer);

// Returns true if |command_buffer| contains no recorded device commands.
bool iree_hal_vulkan_command_buffer_is_empty(
    iree_hal_command_buffer_t* command_buffer);

// Returns true if |command_buffer| contains Vulkan-native commands.
bool iree_hal_vulkan_command_buffer_has_native_commands(
    iree_hal_command_buffer_t* command_buffer);

// Returns the number of dispatch commands recorded in |command_buffer|.
iree_host_size_t iree_hal_vulkan_command_buffer_dispatch_count(
    iree_hal_command_buffer_t* command_buffer);

// Descriptor pool capacity required to replay a command buffer once into a
// native VkCommandBuffer.
typedef struct iree_hal_vulkan_command_buffer_descriptor_requirements_t {
  // Number of descriptor sets required.
  uint32_t set_count;

  // Number of sampler descriptors required.
  uint32_t sampler_count;

  // Number of uniform-buffer descriptors required.
  uint32_t uniform_buffer_count;

  // Number of storage-buffer descriptors required.
  uint32_t storage_buffer_count;
} iree_hal_vulkan_command_buffer_descriptor_requirements_t;

// Returns descriptor pool capacity required to replay |command_buffer| once
// into a native VkCommandBuffer.
iree_status_t
iree_hal_vulkan_command_buffer_native_descriptor_pool_requirements(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_vulkan_command_buffer_descriptor_requirements_t* out_requirements);

// Returns true if replaying |command_buffer| embeds binding-table-dependent
// descriptor state into the native VkCommandBuffer.
bool iree_hal_vulkan_command_buffer_has_descriptor_dispatches(
    iree_hal_command_buffer_t* command_buffer);

// Host-published BDA table storage used while replaying a command buffer once
// into a native VkCommandBuffer.
typedef struct iree_hal_vulkan_command_buffer_bda_publication_t {
  // Host-visible span reserved for all BDA dispatch tables in the replay.
  iree_byte_span_t host_span;

  // Device address corresponding to host_span.data.
  VkDeviceAddress device_address;
} iree_hal_vulkan_command_buffer_bda_publication_t;

// Cached BDA properties for one queue_execute binding table slot.
typedef struct iree_hal_vulkan_command_buffer_bda_binding_slot_t {
  // Device address of the bound buffer range at the binding-table offset.
  VkDeviceAddress device_address;

  // Byte length of the bound buffer range after the binding-table offset.
  iree_device_size_t length;
} iree_hal_vulkan_command_buffer_bda_binding_slot_t;

// Per-submit BDA binding-table cache used while publishing replay tables.
typedef struct iree_hal_vulkan_command_buffer_bda_binding_cache_t {
  // Mutable slot cache storage indexed by HAL binding-table slot ordinal.
  iree_hal_vulkan_command_buffer_bda_binding_slot_t* slots;

  // Number of entries in slots.
  iree_host_size_t slot_count;
} iree_hal_vulkan_command_buffer_bda_binding_cache_t;

// Resolves one HAL binding table slot to its BDA device address and length.
//
// |bda_binding_cache| may be NULL. When present and the slot is within cache
// bounds, the resolved address and length are cached for the caller.
iree_status_t iree_hal_vulkan_command_buffer_resolve_bda_binding_table_slot(
    iree_hal_buffer_binding_table_t binding_table, uint32_t buffer_slot,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    iree_hal_vulkan_command_buffer_bda_binding_slot_t* out_slot);

// Returns host-published BDA table bytes required to replay |command_buffer|
// once into a native VkCommandBuffer.
iree_status_t iree_hal_vulkan_command_buffer_native_bda_publication_length(
    iree_hal_command_buffer_t* command_buffer,
    iree_device_size_t* out_publication_length);

// Publishes BDA binding tables into an already-reserved publication range.
//
// The table layout matches iree_hal_vulkan_command_buffer_record_native, so a
// native command buffer recorded against |bda_publication| can be resubmitted
// after this updates the publication contents for a new binding table.
iree_status_t iree_hal_vulkan_command_buffer_publish_bda_binding_tables(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_buffer_bda_publication_t* bda_publication,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache);

// Emits executable/export metadata referenced by recorded dispatch commands.
iree_status_t iree_hal_vulkan_command_buffer_record_profile_metadata(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id);

// Counts dispatch commands that match the active profile capture filter.
iree_status_t iree_hal_vulkan_command_buffer_count_profiled_dispatches(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_local_profile_recorder_t* profile_recorder,
    iree_hal_local_profile_queue_scope_t scope, uint64_t command_buffer_id,
    uint32_t* out_dispatch_count);

// Appends dispatch profile events from timestamp pairs recorded around each
// profiled dispatch command.
//
// |dispatch_ticks| contains |dispatch_count| adjacent start/end tick pairs for
// dispatch commands that matched the active profile capture filter.
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

  // First query slot reset before timestamp writes.
  uint32_t first_query;

  // Number of query slots reset before timestamp writes.
  uint32_t query_count;

  // Query index written before the command-buffer payload, or ABSENT.
  uint32_t queue_start_query;

  // Query index written after the command-buffer payload, or ABSENT.
  uint32_t queue_end_query;

  // First query slot for per-dispatch timestamp pairs, or ABSENT.
  uint32_t dispatch_base_query;

  // Number of dispatch commands with per-dispatch timestamp pairs.
  uint32_t dispatch_query_count;

  // Profile recorder used to apply capture filters to dispatch timestamping.
  iree_hal_local_profile_recorder_t* recorder;

  // Queue metadata identity used to match profile capture filters.
  iree_hal_local_profile_queue_scope_t scope;

  // Session-local command-buffer id used to match profile capture filters.
  uint64_t command_buffer_id;
} iree_hal_vulkan_command_buffer_profile_marker_t;

// Records Vulkan-native commands into |native_command_buffer|.
//
// |debug_utils| controls optional VK_EXT_debug_utils labels emitted while
// replaying recorded debug groups.
//
// |profile_marker| may be NULL. When present, the query pool is reset and
// timestamped inside |native_command_buffer| around the requested queue payload
// and dispatch commands.
//
// Descriptor sets are allocated from |descriptor_pool|. The caller must keep
// that pool alive until |native_command_buffer| is no longer executing. When
// the command buffer has no descriptor requirements this may be VK_NULL_HANDLE.
//
// BDA binding tables are allocated from |bda_publication|. The caller must keep
// that publication alive until |native_command_buffer| is no longer executing.
// When the command buffer has no BDA dispatches this may be NULL.
//
// |bda_binding_cache| may be NULL. When present, it caches binding-table slot
// device addresses for the duration of this recording/publication pass.
iree_status_t iree_hal_vulkan_command_buffer_record_native(
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    const iree_hal_vulkan_debug_utils_t* debug_utils,
    const iree_hal_vulkan_builtins_t* builtins,
    VkCommandBuffer native_command_buffer,
    VkCommandBufferUsageFlags usage_flags, VkDescriptorPool descriptor_pool,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_vulkan_command_buffer_bda_publication_t* bda_publication,
    iree_hal_vulkan_command_buffer_bda_binding_cache_t* bda_binding_cache,
    const iree_hal_vulkan_command_buffer_profile_marker_t* profile_marker,
    iree_allocator_t host_allocator);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_VULKAN_COMMAND_BUFFER_H_
