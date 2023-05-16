// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/direct_command_buffer.h"

#include <cstddef>
#include <cstdint>

#include "iree/base/api.h"
#include "iree/base/internal/inline_array.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/vulkan/descriptor_set_arena.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_event.h"
#include "iree/hal/drivers/vulkan/native_executable.h"
#include "iree/hal/drivers/vulkan/native_pipeline_layout.h"
#include "iree/hal/drivers/vulkan/status_util.h"
#include "iree/hal/drivers/vulkan/util/ref_ptr.h"
#include "iree/hal/drivers/vulkan/vma_buffer.h"
#include "iree/hal/utils/resource_set.h"

using namespace iree::hal::vulkan;

// Command buffer implementation that directly maps to VkCommandBuffer.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct iree_hal_vulkan_direct_command_buffer_t {
  iree_hal_command_buffer_t base;
  VkDeviceHandle* logical_device;
  iree_hal_vulkan_tracing_context_t* tracing_context;
  iree_arena_block_pool_t* block_pool;

  VkCommandPoolHandle* command_pool;
  VkCommandBuffer handle;

  DynamicSymbols* syms;

  // Maintains a reference to all resources used within the command buffer.
  // Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // TODO(benvanik): may grow large - should try to reclaim or reuse.
  DescriptorSetArena descriptor_set_arena;

  // The current descriptor set group in use by the command buffer, if any.
  // This must remain valid until all in-flight submissions of the command
  // buffer complete.
  DescriptorSetGroup descriptor_set_group;

  BuiltinExecutables* builtin_executables;

  // Shadow copy of push constants used during normal operation, for restoring
  // after builtin_executables uses vkCmdPushConstants. Size must be greater
  // than or equal to the push constant memory used by builtin_executables.
  // TODO(scotttodd): use [maxPushConstantsSize - 16, maxPushConstantsSize]
  //                  instead of [0, 16] to reduce frequency of updates
  uint8_t push_constants_storage[IREE_HAL_VULKAN_BUILTIN_PUSH_CONSTANT_COUNT];
} iree_hal_vulkan_direct_command_buffer_t;

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_direct_command_buffer_vtable;
}  // namespace

static iree_hal_vulkan_direct_command_buffer_t*
iree_hal_vulkan_direct_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_vulkan_direct_command_buffer_vtable);
  return (iree_hal_vulkan_direct_command_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_direct_command_buffer_allocate(
    iree_hal_device_t* device,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::VkCommandPoolHandle* command_pool,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_vulkan_tracing_context_t* tracing_context,
    iree::hal::vulkan::DescriptorPoolCache* descriptor_pool_cache,
    iree::hal::vulkan::BuiltinExecutables* builtin_executables,
    iree_arena_block_pool_t* block_pool,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(command_pool);
  IREE_ASSERT_ARGUMENT(descriptor_pool_cache);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  VkCommandBufferAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.pNext = NULL;
  allocate_info.commandPool = *command_pool;
  allocate_info.commandBufferCount = 1;
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

  VkCommandBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_pool->Allocate(&allocate_info, &handle));

  iree_hal_vulkan_direct_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(logical_device->host_allocator(),
                            sizeof(*command_buffer), (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, queue_affinity, binding_capacity,
        &iree_hal_vulkan_direct_command_buffer_vtable, &command_buffer->base);
    command_buffer->logical_device = logical_device;
    command_buffer->tracing_context = tracing_context;
    command_buffer->block_pool = block_pool;
    command_buffer->command_pool = command_pool;
    command_buffer->handle = handle;
    command_buffer->syms = logical_device->syms().get();

    new (&command_buffer->descriptor_set_arena)
        DescriptorSetArena(descriptor_pool_cache);
    new (&command_buffer->descriptor_set_group) DescriptorSetGroup();

    command_buffer->builtin_executables = builtin_executables;
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    command_pool->Free(handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_vulkan_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_vulkan_direct_command_buffer_vtable);
}

static void iree_hal_vulkan_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  command_buffer->command_pool->Free(command_buffer->handle);

  IREE_IGNORE_ERROR(command_buffer->descriptor_set_group.Reset());
  command_buffer->descriptor_set_group.~DescriptorSetGroup();
  command_buffer->descriptor_set_arena.~DescriptorSetArena();

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

VkCommandBuffer iree_hal_vulkan_direct_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  if (!iree_hal_vulkan_direct_command_buffer_isa(base_command_buffer)) {
    return VK_NULL_HANDLE;
  }
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  return command_buffer->handle;
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = NULL;
  begin_info.flags = iree_all_bits_set(command_buffer->base.mode,
                                       IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)
                         ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                         : 0;
  begin_info.pInheritanceInfo = NULL;
  VK_RETURN_IF_ERROR(command_buffer->syms->vkBeginCommandBuffer(
                         command_buffer->handle, &begin_info),
                     "vkBeginCommandBuffer");

  IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->handle,
      /*file_name=*/NULL, 0,
      /*line=*/0, /*func_name=*/NULL, 0,
      "iree_hal_vulkan_direct_command_buffer",
      strlen("iree_hal_vulkan_direct_command_buffer"));

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  VK_RETURN_IF_ERROR(
      command_buffer->syms->vkEndCommandBuffer(command_buffer->handle),
      "vkEndCommandBuffer");

  // Flush all pending descriptor set writes (if any).
  command_buffer->descriptor_set_group =
      command_buffer->descriptor_set_arena.Flush();

  return iree_ok_status();
}

static void iree_hal_vulkan_direct_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->handle,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0, /*func_name=*/NULL, 0, label.data,
      label.size);
  if (command_buffer->syms->vkCmdBeginDebugUtilsLabelEXT) {
    char label_buffer[128];
    snprintf(label_buffer, sizeof(label_buffer), "%.*s", (int)label.size,
             label.data);
    VkDebugUtilsLabelEXT label_info = {
        /*.sType=*/VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        /*.pNext=*/NULL,
        /*.pLabelName=*/label_buffer,
        /*.color=*/
        {
            /*r=*/label_color.r / 255.0f,
            /*g=*/label_color.g / 255.0f,
            /*b=*/label_color.b / 255.0f,
            /*a=*/label_color.a / 255.0f,
        },
    };
    command_buffer->syms->vkCmdBeginDebugUtilsLabelEXT(command_buffer->handle,
                                                       &label_info);
  }
}

static void iree_hal_vulkan_direct_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  if (command_buffer->syms->vkCmdEndDebugUtilsLabelEXT) {
    command_buffer->syms->vkCmdEndDebugUtilsLabelEXT(command_buffer->handle);
  }
  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);
}

static VkPipelineStageFlags iree_hal_vulkan_convert_pipeline_stage_flags(
    iree_hal_execution_stage_t stage_mask) {
  VkPipelineStageFlags flags = 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE)
               ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
               : 0;
  flags |=
      iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS)
          ? VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
          : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_DISPATCH)
               ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
               : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_TRANSFER)
               ? VK_PIPELINE_STAGE_TRANSFER_BIT
               : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE)
               ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
               : 0;
  flags |= iree_any_bit_set(stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)
               ? VK_PIPELINE_STAGE_HOST_BIT
               : 0;
  return flags;
}

static VkAccessFlags iree_hal_vulkan_convert_access_mask(
    iree_hal_access_scope_t access_mask) {
  VkAccessFlags flags = 0;
  flags |=
      iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ)
          ? VK_ACCESS_INDIRECT_COMMAND_READ_BIT
          : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_CONSTANT_READ)
               ? VK_ACCESS_UNIFORM_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_DISPATCH_READ)
               ? VK_ACCESS_SHADER_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE)
               ? VK_ACCESS_SHADER_WRITE_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_TRANSFER_READ)
               ? VK_ACCESS_TRANSFER_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE)
               ? VK_ACCESS_TRANSFER_WRITE_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_HOST_READ)
               ? VK_ACCESS_HOST_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_HOST_WRITE)
               ? VK_ACCESS_HOST_WRITE_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_MEMORY_READ)
               ? VK_ACCESS_MEMORY_READ_BIT
               : 0;
  flags |= iree_any_bit_set(access_mask, IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE)
               ? VK_ACCESS_MEMORY_WRITE_BIT
               : 0;
  return flags;
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();

  iree_inline_array(VkMemoryBarrier, memory_barrier_infos, memory_barrier_count,
                    host_allocator);
  for (int i = 0; i < memory_barrier_count; ++i) {
    const auto& memory_barrier = memory_barriers[i];
    VkMemoryBarrier* info = iree_inline_array_at(memory_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.target_scope);
  }

  iree_inline_array(VkBufferMemoryBarrier, buffer_barrier_infos,
                    buffer_barrier_count, host_allocator);
  for (int i = 0; i < buffer_barrier_count; ++i) {
    const auto& buffer_barrier = buffer_barriers[i];
    VkBufferMemoryBarrier* info = iree_inline_array_at(buffer_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.target_scope);
    info->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->buffer = iree_hal_vulkan_vma_buffer_handle(
        iree_hal_buffer_allocated_buffer(buffer_barrier.buffer));
    info->offset = buffer_barrier.offset;
    info->size = buffer_barrier.length;
  }

  command_buffer->syms->vkCmdPipelineBarrier(
      command_buffer->handle,
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask),
      iree_hal_vulkan_convert_pipeline_stage_flags(target_stage_mask),
      /*dependencyFlags=*/0, (uint32_t)memory_barrier_count,
      iree_inline_array_data(memory_barrier_infos),
      (uint32_t)buffer_barrier_count,
      iree_inline_array_data(buffer_barrier_infos), 0, NULL);

  iree_inline_array_deinitialize(memory_barrier_infos);
  iree_inline_array_deinitialize(buffer_barrier_infos);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &event));

  command_buffer->syms->vkCmdSetEvent(
      command_buffer->handle, iree_hal_vulkan_native_event_handle(event),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask));

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &event));

  command_buffer->syms->vkCmdResetEvent(
      command_buffer->handle, iree_hal_vulkan_native_event_handle(event),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask));

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, event_count, events));

  iree_inline_array(VkEvent, event_handles, event_count, host_allocator);
  for (int i = 0; i < event_count; ++i) {
    *iree_inline_array_at(event_handles, i) =
        iree_hal_vulkan_native_event_handle(events[i]);
  }

  iree_inline_array(VkMemoryBarrier, memory_barrier_infos, memory_barrier_count,
                    host_allocator);
  for (int i = 0; i < memory_barrier_count; ++i) {
    const auto& memory_barrier = memory_barriers[i];
    VkMemoryBarrier* info = iree_inline_array_at(memory_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.target_scope);
  }

  iree_inline_array(VkBufferMemoryBarrier, buffer_barrier_infos,
                    buffer_barrier_count, host_allocator);
  for (int i = 0; i < buffer_barrier_count; ++i) {
    const auto& buffer_barrier = buffer_barriers[i];
    VkBufferMemoryBarrier* info = iree_inline_array_at(buffer_barrier_infos, i);
    info->sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info->pNext = NULL;
    info->srcAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.source_scope);
    info->dstAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.target_scope);
    info->srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info->buffer = iree_hal_vulkan_vma_buffer_handle(
        iree_hal_buffer_allocated_buffer(buffer_barrier.buffer));
    info->offset = buffer_barrier.offset;
    info->size = buffer_barrier.length;
  }

  command_buffer->syms->vkCmdWaitEvents(
      command_buffer->handle, (uint32_t)event_count,
      iree_inline_array_data(event_handles),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask),
      iree_hal_vulkan_convert_pipeline_stage_flags(target_stage_mask),
      (uint32_t)memory_barrier_count,
      iree_inline_array_data(memory_barrier_infos),
      (uint32_t)buffer_barrier_count,
      iree_inline_array_data(buffer_barrier_infos), 0, NULL);

  iree_inline_array_deinitialize(event_handles);
  iree_inline_array_deinitialize(memory_barrier_infos);
  iree_inline_array_deinitialize(buffer_barrier_infos);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // NOTE: we could use this to prevent queue family transitions.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t iree_hal_vulkan_splat_pattern(const void* pattern,
                                              size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *static_cast<const uint8_t*>(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *static_cast<const uint16_t*>(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *static_cast<const uint32_t*>(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  VkBuffer target_device_buffer = iree_hal_vulkan_vma_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_buffer));

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_buffer));

  // vkCmdFillBuffer requires a 4 byte alignment for the offset, pattern, and
  // length. We use a polyfill here that fills the unaligned start and end of
  // fill operations, if needed.

  if (target_offset % 4 != 0 || length % 4 != 0) {
    // TODO(scotttodd): only restore push constants that have been modified?
    //                  (this can pass uninitialized memory right now, which
    //                   *should* be safe but is wasteful)
    IREE_RETURN_IF_ERROR(
        command_buffer->builtin_executables->FillBufferUnaligned(
            command_buffer->handle, &(command_buffer->descriptor_set_arena),
            target_buffer, target_offset, length, pattern, pattern_length,
            command_buffer->push_constants_storage));

    // Continue using vkCmdFillBuffer below, but only for the inner aligned
    // portion of the fill operation.
    // For example:
    //   original offset 2, length 8
    //   aligned  offset 4, length 4
    // [0x00,0x00,0xAB,0xAB | 0xAB,0xAB,0xAB,0xAB | 0xAB,0xAB,0x00,0x00]
    //            <-------> <---------------------> <------->
    //            unaligned     vkCmdFillBuffer     unaligned
    iree_device_size_t aligned_target_offset =
        iree_device_align(target_offset, 4);
    iree_device_size_t target_end = target_offset + length;
    iree_device_size_t rounded_down_target_end = (target_end / 4) * 4;
    length -= (aligned_target_offset - target_offset) +
              (target_end - rounded_down_target_end);
    target_offset = aligned_target_offset;
  }

  if (length > 0) {
    // Note that vkCmdFillBuffer only accepts 4-byte aligned values so we need
    // to splat out our variable-length pattern.
    target_offset += iree_hal_buffer_byte_offset(target_buffer);
    uint32_t dword_pattern =
        iree_hal_vulkan_splat_pattern(pattern, pattern_length);
    command_buffer->syms->vkCmdFillBuffer(command_buffer->handle,
                                          target_device_buffer, target_offset,
                                          length, dword_pattern);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  VkBuffer target_device_buffer = iree_hal_vulkan_vma_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_buffer));

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &target_buffer));

  // Vulkan only allows updates of <= 65536 because you really, really, really
  // shouldn't do large updates like this (as it wastes command buffer space and
  // may be slower than just using write-through mapped memory). The
  // recommendation in the spec for larger updates is to split the single update
  // into multiple updates over the entire desired range.
  const auto* source_buffer_ptr =
      static_cast<const uint8_t*>(source_buffer) + source_offset;
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  while (length > 0) {
    iree_device_size_t chunk_length =
        iree_min((iree_device_size_t)65536u, length);
    command_buffer->syms->vkCmdUpdateBuffer(command_buffer->handle,
                                            target_device_buffer, target_offset,
                                            chunk_length, source_buffer_ptr);
    source_buffer_ptr += chunk_length;
    target_offset += chunk_length;
    length -= chunk_length;
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  VkBuffer source_device_buffer = iree_hal_vulkan_vma_buffer_handle(
      iree_hal_buffer_allocated_buffer(source_buffer));
  VkBuffer target_device_buffer = iree_hal_vulkan_vma_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_buffer));

  const iree_hal_buffer_t* buffers[2] = {source_buffer, target_buffer};
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  VkBufferCopy region;
  region.srcOffset = iree_hal_buffer_byte_offset(source_buffer) + source_offset;
  region.dstOffset = iree_hal_buffer_byte_offset(target_buffer) + target_offset;
  region.size = length;
  command_buffer->syms->vkCmdCopyBuffer(command_buffer->handle,
                                        source_device_buffer,
                                        target_device_buffer, 1, &region);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on Vulkan");
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  iree_host_size_t storage_size =
      IREE_ARRAYSIZE(command_buffer->push_constants_storage);
  if (offset < storage_size) {
    memcpy(command_buffer->push_constants_storage + offset, values,
           std::min(values_length, storage_size) - offset);
  }

  command_buffer->syms->vkCmdPushConstants(
      command_buffer->handle,
      iree_hal_vulkan_native_pipeline_layout_handle(pipeline_layout),
      VK_SHADER_STAGE_COMPUTE_BIT, (uint32_t)offset, (uint32_t)values_length,
      values);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  // TODO(benvanik): batch insert by getting the resources in their own list.
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (bindings[i].buffer) {
      IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
          command_buffer->resource_set, 1, &bindings[i].buffer));
    }
  }

  // Either allocate, update, and bind a descriptor set or use push descriptor
  // sets to use the command buffer pool when supported.
  return command_buffer->descriptor_set_arena.BindDescriptorSet(
      command_buffer->handle, pipeline_layout, set, binding_count, bindings);
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  IREE_TRACE({
    iree_hal_vulkan_source_location_t source_location;
    iree_hal_vulkan_native_executable_entry_point_source_location(
        executable, entry_point, &source_location);
    IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
        command_buffer->tracing_context, command_buffer->handle,
        source_location.file_name.data, source_location.file_name.size,
        source_location.line, /*func_name=*/NULL, 0,
        source_location.func_name.data, source_location.func_name.size);
  });

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &executable));

  // Get the compiled and linked pipeline for the specified entry point and
  // bind it to the command buffer.
  VkPipeline pipeline_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_native_executable_pipeline_for_entry_point(
          executable, entry_point, &pipeline_handle));
  command_buffer->syms->vkCmdBindPipeline(
      command_buffer->handle, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_handle);

  command_buffer->syms->vkCmdDispatch(command_buffer->handle, workgroup_x,
                                      workgroup_y, workgroup_z);

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  const void* resources[2] = {executable, workgroups_buffer};
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, IREE_ARRAYSIZE(resources), resources));

  iree_hal_vulkan_source_location_t source_location;
  iree_hal_vulkan_native_executable_entry_point_source_location(
      executable, entry_point, &source_location);
  IREE_VULKAN_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->handle,
      source_location.file_name.data, source_location.file_name.size,
      source_location.line, /*func_name=*/NULL, 0,
      source_location.func_name.data, source_location.func_name.size);

  // Get the compiled and linked pipeline for the specified entry point and
  // bind it to the command buffer.
  VkPipeline pipeline_handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(
      iree_hal_vulkan_native_executable_pipeline_for_entry_point(
          executable, entry_point, &pipeline_handle));
  command_buffer->syms->vkCmdBindPipeline(
      command_buffer->handle, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_handle);

  VkBuffer workgroups_device_buffer = iree_hal_vulkan_vma_buffer_handle(
      iree_hal_buffer_allocated_buffer(workgroups_buffer));
  workgroups_offset += iree_hal_buffer_byte_offset(workgroups_buffer);
  command_buffer->syms->vkCmdDispatchIndirect(
      command_buffer->handle, workgroups_device_buffer, workgroups_offset);

  IREE_VULKAN_TRACE_ZONE_END(command_buffer->tracing_context,
                             command_buffer->handle);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  if (binding_table.count > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    // Since Vulkan doesn't natively support this we'd need to emulate things
    // with an iree_hal_vulkan_indirect_command_buffer_t type that captured the
    // command buffer using deferred command buffer and allowed replay with a
    // binding table. If we wanted to actually reuse the command buffers we'd
    // need to use update-after-bind (where supported), device pointers (where
    // supported), or descriptor indexing and a big ringbuffer (make a 1024
    // element descriptor array and cycle through it with each submission).
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &base_commands));

  iree_hal_vulkan_direct_command_buffer_t* commands =
      iree_hal_vulkan_direct_command_buffer_cast(base_commands);

  command_buffer->syms->vkCmdExecuteCommands(command_buffer->handle, 1,
                                             &commands->handle);

  return iree_ok_status();
}

namespace {
const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_direct_command_buffer_vtable = {
        /*.destroy=*/iree_hal_vulkan_direct_command_buffer_destroy,
        /*.begin=*/iree_hal_vulkan_direct_command_buffer_begin,
        /*.end=*/iree_hal_vulkan_direct_command_buffer_end,
        /*.begin_debug_group=*/
        iree_hal_vulkan_direct_command_buffer_begin_debug_group,
        /*.end_debug_group=*/
        iree_hal_vulkan_direct_command_buffer_end_debug_group,
        /*.execution_barrier=*/
        iree_hal_vulkan_direct_command_buffer_execution_barrier,
        /*.signal_event=*/
        iree_hal_vulkan_direct_command_buffer_signal_event,
        /*.reset_event=*/iree_hal_vulkan_direct_command_buffer_reset_event,
        /*.wait_events=*/iree_hal_vulkan_direct_command_buffer_wait_events,
        /*.discard_buffer=*/
        iree_hal_vulkan_direct_command_buffer_discard_buffer,
        /*.fill_buffer=*/iree_hal_vulkan_direct_command_buffer_fill_buffer,
        /*.update_buffer=*/
        iree_hal_vulkan_direct_command_buffer_update_buffer,
        /*.copy_buffer=*/iree_hal_vulkan_direct_command_buffer_copy_buffer,
        /*.collective=*/
        iree_hal_vulkan_direct_command_buffer_collective,
        /*.push_constants=*/
        iree_hal_vulkan_direct_command_buffer_push_constants,
        /*.push_descriptor_set=*/
        iree_hal_vulkan_direct_command_buffer_push_descriptor_set,
        /*.dispatch=*/iree_hal_vulkan_direct_command_buffer_dispatch,
        /*.dispatch_indirect=*/
        iree_hal_vulkan_direct_command_buffer_dispatch_indirect,
        /*.execute_commands=*/
        iree_hal_vulkan_direct_command_buffer_execute_commands,
};
}  // namespace
