// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/vulkan/direct_command_buffer.h"

#include "absl/container/inlined_vector.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/descriptor_set_arena.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/native_descriptor_set.h"
#include "iree/hal/vulkan/native_event.h"
#include "iree/hal/vulkan/native_executable_layout.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/vma_buffer.h"

using namespace iree::hal::vulkan;

// Command buffer implementation that directly maps to VkCommandBuffer.
// This records the commands on the calling thread without additional threading
// indirection.
typedef struct {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;

  VkCommandPoolHandle* command_pool;
  VkCommandBuffer handle;

  DynamicSymbols* syms;

  // TODO(benvanik): may grow large - should try to reclaim or reuse.
  DescriptorSetArena descriptor_set_arena;

  // The current descriptor set group in use by the command buffer, if any.
  // This must remain valid until all in-flight submissions of the command
  // buffer complete.
  DescriptorSetGroup descriptor_set_group;
} iree_hal_vulkan_direct_command_buffer_t;

extern const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_direct_command_buffer_vtable;

static iree_hal_vulkan_direct_command_buffer_t*
iree_hal_vulkan_direct_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_vulkan_direct_command_buffer_vtable);
  return (iree_hal_vulkan_direct_command_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_direct_command_buffer_allocate(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    iree::hal::vulkan::VkCommandPoolHandle* command_pool,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree::hal::vulkan::DescriptorPoolCache* descriptor_pool_cache,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(command_pool);
  IREE_ASSERT_ARGUMENT(descriptor_pool_cache);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
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
    iree_hal_resource_initialize(&iree_hal_vulkan_direct_command_buffer_vtable,
                                 &command_buffer->resource);
    command_buffer->logical_device = logical_device;
    command_buffer->mode = mode;
    command_buffer->allowed_categories = command_categories;
    command_buffer->command_pool = command_pool;
    command_buffer->handle = handle;
    command_buffer->syms = logical_device->syms().get();

    new (&command_buffer->descriptor_set_arena)
        DescriptorSetArena(descriptor_pool_cache);
    new (&command_buffer->descriptor_set_group) DescriptorSetGroup();

    *out_command_buffer = (iree_hal_command_buffer_t*)command_buffer;
  } else {
    command_pool->Free(handle);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_direct_command_buffer_reset(
    iree_hal_vulkan_direct_command_buffer_t* command_buffer) {
  // NOTE: we require that command buffers not be recorded while they are
  // in-flight so this is safe.
  IREE_IGNORE_ERROR(command_buffer->descriptor_set_group.Reset());
}

static void iree_hal_vulkan_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator =
      command_buffer->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_direct_command_buffer_reset(command_buffer);
  command_buffer->command_pool->Free(command_buffer->handle);

  command_buffer->descriptor_set_group.~DescriptorSetGroup();
  command_buffer->descriptor_set_arena.~DescriptorSetArena();

  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

VkCommandBuffer iree_hal_vulkan_direct_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);
  return command_buffer->handle;
}

static iree_hal_command_category_t
iree_hal_vulkan_direct_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* base_command_buffer) {
  return ((const iree_hal_vulkan_direct_command_buffer_t*)base_command_buffer)
      ->allowed_categories;
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  iree_hal_vulkan_direct_command_buffer_reset(command_buffer);

  VkCommandBufferBeginInfo begin_info;
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.pNext = NULL;
  begin_info.flags = iree_all_bits_set(command_buffer->mode,
                                       IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)
                         ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                         : 0;
  begin_info.pInheritanceInfo = NULL;
  VK_RETURN_IF_ERROR(command_buffer->syms->vkBeginCommandBuffer(
                         command_buffer->handle, &begin_info),
                     "vkBeginCommandBuffer");

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  VK_RETURN_IF_ERROR(
      command_buffer->syms->vkEndCommandBuffer(command_buffer->handle),
      "vkEndCommandBuffer");

  // Flush all pending descriptor set writes (if any).
  IREE_ASSIGN_OR_RETURN(command_buffer->descriptor_set_group,
                        command_buffer->descriptor_set_arena.Flush());

  return iree_ok_status();
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
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  absl::InlinedVector<VkMemoryBarrier, 8> memory_barrier_infos(
      memory_barrier_count);
  for (int i = 0; i < memory_barrier_count; ++i) {
    const auto& memory_barrier = memory_barriers[i];
    auto& info = memory_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info.pNext = NULL;
    info.srcAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.source_scope);
    info.dstAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.target_scope);
  }

  absl::InlinedVector<VkBufferMemoryBarrier, 8> buffer_barrier_infos(
      buffer_barrier_count);
  for (int i = 0; i < buffer_barrier_count; ++i) {
    const auto& buffer_barrier = buffer_barriers[i];
    auto& info = buffer_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.pNext = NULL;
    info.srcAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.source_scope);
    info.dstAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.target_scope);
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.buffer = iree_hal_vulkan_vma_buffer_handle(
        iree_hal_buffer_allocated_buffer(buffer_barrier.buffer));
    info.offset = buffer_barrier.offset;
    info.size = buffer_barrier.length;
  }

  command_buffer->syms->vkCmdPipelineBarrier(
      command_buffer->handle,
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask),
      iree_hal_vulkan_convert_pipeline_stage_flags(target_stage_mask),
      /*dependencyFlags=*/0, static_cast<uint32_t>(memory_barrier_infos.size()),
      memory_barrier_infos.data(),
      static_cast<uint32_t>(buffer_barrier_infos.size()),
      buffer_barrier_infos.data(), 0, NULL);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

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

  absl::InlinedVector<VkEvent, 4> event_handles(event_count);
  for (int i = 0; i < event_count; ++i) {
    event_handles[i] = iree_hal_vulkan_native_event_handle(events[i]);
  }

  absl::InlinedVector<VkMemoryBarrier, 8> memory_barrier_infos(
      memory_barrier_count);
  for (int i = 0; i < memory_barrier_count; ++i) {
    const auto& memory_barrier = memory_barriers[i];
    auto& info = memory_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    info.pNext = NULL;
    info.srcAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.source_scope);
    info.dstAccessMask =
        iree_hal_vulkan_convert_access_mask(memory_barrier.target_scope);
  }

  absl::InlinedVector<VkBufferMemoryBarrier, 8> buffer_barrier_infos(
      buffer_barrier_count);
  for (int i = 0; i < buffer_barrier_count; ++i) {
    const auto& buffer_barrier = buffer_barriers[i];
    auto& info = buffer_barrier_infos[i];
    info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    info.pNext = NULL;
    info.srcAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.source_scope);
    info.dstAccessMask =
        iree_hal_vulkan_convert_access_mask(buffer_barrier.target_scope);
    info.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.buffer = iree_hal_vulkan_vma_buffer_handle(
        iree_hal_buffer_allocated_buffer(buffer_barrier.buffer));
    info.offset = buffer_barrier.offset;
    info.size = buffer_barrier.length;
  }

  command_buffer->syms->vkCmdWaitEvents(
      command_buffer->handle, (uint32_t)event_count, event_handles.data(),
      iree_hal_vulkan_convert_pipeline_stage_flags(source_stage_mask),
      iree_hal_vulkan_convert_pipeline_stage_flags(target_stage_mask),
      (uint32_t)memory_barrier_count, memory_barrier_infos.data(),
      (uint32_t)buffer_barrier_count, buffer_barrier_infos.data(), 0, NULL);

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

  // Note that fill only accepts 4-byte aligned values so we need to splat out
  // our variable-length pattern.
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  uint32_t dword_pattern =
      iree_hal_vulkan_splat_pattern(pattern, pattern_length);
  command_buffer->syms->vkCmdFillBuffer(command_buffer->handle,
                                        target_device_buffer, target_offset,
                                        length, dword_pattern);

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

  // Vulkan only allows updates of <= 65536 because you really, really, really
  // shouldn't do large updates like this (as it wastes command buffer space and
  // may be slower than just using write-through mapped memory). The
  // recommendation in the spec for larger updates is to split the single update
  // into multiple updates over the entire desired range.
  const auto* source_buffer_ptr = static_cast<const uint8_t*>(source_buffer);
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

  VkBufferCopy region;
  region.srcOffset = iree_hal_buffer_byte_offset(source_buffer) + source_offset;
  region.dstOffset = iree_hal_buffer_byte_offset(target_buffer) + target_offset;
  region.size = length;
  command_buffer->syms->vkCmdCopyBuffer(command_buffer->handle,
                                        source_device_buffer,
                                        target_device_buffer, 1, &region);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  command_buffer->syms->vkCmdPushConstants(
      command_buffer->handle,
      iree_hal_vulkan_native_executable_layout_handle(executable_layout),
      VK_SHADER_STAGE_COMPUTE_BIT, (uint32_t)offset, (uint32_t)values_length,
      values);

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  // Either allocate, update, and bind a descriptor set or use push descriptor
  // sets to use the command buffer pool when supported.
  return command_buffer->descriptor_set_arena.BindDescriptorSet(
      command_buffer->handle, executable_layout, set, binding_count, bindings);
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

  // Vulkan takes uint32_t as the size here, unlike everywhere else.
  absl::InlinedVector<uint32_t, 4> dynamic_offsets_i32(dynamic_offset_count);
  for (int i = 0; i < dynamic_offset_count; ++i) {
    dynamic_offsets_i32[i] = static_cast<uint32_t>(dynamic_offsets[i]);
  }

  VkDescriptorSet descriptor_sets[1] = {
      iree_hal_vulkan_native_descriptor_set_handle(descriptor_set),
  };
  command_buffer->syms->vkCmdBindDescriptorSets(
      command_buffer->handle, VK_PIPELINE_BIND_POINT_COMPUTE,
      iree_hal_vulkan_native_executable_layout_handle(executable_layout), set,
      (uint32_t)IREE_ARRAYSIZE(descriptor_sets), descriptor_sets,
      static_cast<uint32_t>(dynamic_offsets_i32.size()),
      dynamic_offsets_i32.data());

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

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

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_direct_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  iree_hal_vulkan_direct_command_buffer_t* command_buffer =
      iree_hal_vulkan_direct_command_buffer_cast(base_command_buffer);

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

  return iree_ok_status();
}

const iree_hal_command_buffer_vtable_t
    iree_hal_vulkan_direct_command_buffer_vtable = {
        /*.destroy=*/iree_hal_vulkan_direct_command_buffer_destroy,
        /*.allowed_categories=*/
        iree_hal_vulkan_direct_command_buffer_allowed_categories,
        /*.begin=*/iree_hal_vulkan_direct_command_buffer_begin,
        /*.end=*/iree_hal_vulkan_direct_command_buffer_end,
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
        /*.push_constants=*/
        iree_hal_vulkan_direct_command_buffer_push_constants,
        /*.push_descriptor_set=*/
        iree_hal_vulkan_direct_command_buffer_push_descriptor_set,
        /*.bind_descriptor_set=*/
        iree_hal_vulkan_direct_command_buffer_bind_descriptor_set,
        /*.dispatch=*/iree_hal_vulkan_direct_command_buffer_dispatch,
        /*.dispatch_indirect=*/
        iree_hal_vulkan_direct_command_buffer_dispatch_indirect,
};
