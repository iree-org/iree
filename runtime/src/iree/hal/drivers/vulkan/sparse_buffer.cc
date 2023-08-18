// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/sparse_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/base_buffer.h"
#include "iree/hal/drivers/vulkan/status_util.h"

typedef struct iree_hal_vulkan_sparse_buffer_t {
  iree_hal_vulkan_base_buffer_t base;
  iree::hal::vulkan::VkDeviceHandle* logical_device;
  iree_host_size_t physical_block_count;
  VkDeviceMemory physical_blocks[];
} iree_hal_vulkan_sparse_buffer_t;

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_vulkan_sparse_buffer_vtable;
}  // namespace

static iree_hal_vulkan_sparse_buffer_t* iree_hal_vulkan_sparse_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_sparse_buffer_vtable);
  return (iree_hal_vulkan_sparse_buffer_t*)base_value;
}

static iree_status_t iree_hal_vulkan_sparse_buffer_commit_sync(
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkQueue queue,
    VkBuffer handle, VkMemoryRequirements requirements,
    uint32_t memory_type_index, VkDeviceSize physical_block_size,
    iree_host_size_t physical_block_count,
    VkDeviceMemory out_physical_blocks[]) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)requirements.size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)requirements.alignment);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)physical_block_size);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)physical_block_count);

  // Allocate all physical blocks; note that the last block may be of partial
  // size and we'll just allocate whatever remains from the total requested
  // size.
  VkMemoryAllocateInfo allocate_info;
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.pNext = NULL;
  allocate_info.memoryTypeIndex = memory_type_index;
  VkSparseMemoryBind* binds = (VkSparseMemoryBind*)iree_alloca(
      sizeof(VkSparseMemoryBind) * physical_block_count);
  for (iree_host_size_t i = 0; i < physical_block_count; ++i) {
    if (i < physical_block_count - 1) {
      allocate_info.allocationSize = physical_block_size;
    } else {
      allocate_info.allocationSize =
          requirements.size - physical_block_size * (physical_block_count - 1);
    }
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "vkAllocateMemory");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z1, (int64_t)allocate_info.allocationSize);
    iree_status_t allocate_status = VK_RESULT_TO_STATUS(
        logical_device->syms()->vkAllocateMemory(
            *logical_device, &allocate_info, logical_device->allocator(),
            &out_physical_blocks[i]),
        "vkAllocateMemory");
    IREE_TRACE_ZONE_END(z1);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, allocate_status);

    binds[i].resourceOffset = i * physical_block_size;
    binds[i].size = allocate_info.allocationSize;
    binds[i].memory = out_physical_blocks[i];
    binds[i].memoryOffset = 0;
    binds[i].flags = 0;
  }

  // Temporary fence for enforcing host-synchronous execution.
  VkFenceCreateInfo fence_info;
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.pNext = NULL;
  fence_info.flags = 0;
  VkFence fence = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateFence(
                                  *logical_device, &fence_info,
                                  logical_device->allocator(), &fence),
                              "vkCreateFence"));

  IREE_TRACE_ZONE_BEGIN_NAMED(z1, "vkQueueBindSparse");

  // Enqueue sparse binding operation. This will complete asynchronously.
  VkSparseBufferMemoryBindInfo memory_bind_info;
  memory_bind_info.buffer = handle;
  memory_bind_info.bindCount = (uint32_t)physical_block_count;
  memory_bind_info.pBinds = binds;
  VkBindSparseInfo bind_info;
  memset(&bind_info, 0, sizeof(bind_info));
  bind_info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
  bind_info.pNext = NULL;
  bind_info.bufferBindCount = 1;
  bind_info.pBufferBinds = &memory_bind_info;
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkQueueBindSparse(queue, 1, &bind_info, fence),
      "vkQueueBindSparse");

  // If enqueuing succeeded then wait for the binding to finish.
  if (iree_status_is_ok(status)) {
    status = VK_RESULT_TO_STATUS(
        logical_device->syms()->vkWaitForFences(
            *logical_device, 1, &fence, /*waitAll=*/VK_TRUE, UINT64_MAX),
        "vkWaitForFences");
  }

  IREE_TRACE_ZONE_END(z1);

  logical_device->syms()->vkDestroyFence(*logical_device, fence,
                                         logical_device->allocator());

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_vulkan_sparse_buffer_create_bound_sync(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree::hal::vulkan::VkDeviceHandle* logical_device, VkQueue queue,
    VkBuffer handle, VkMemoryRequirements requirements,
    uint32_t memory_type_index, VkDeviceSize max_allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  // The maximum allocation size reported by Vulkan does not need to be a power
  // of two or aligned to anything in particular - sparse buffers do require
  // alignment though and must also be under the limit so here we adjust down to
  // the maximum aligned value.
  iree_device_size_t physical_block_size =
      iree_device_size_floor_div(max_allocation_size, requirements.alignment) *
      requirements.alignment;

  // ceil-div for the number of blocks as the last block may be partial.
  iree_host_size_t physical_block_count =
      (iree_host_size_t)iree_device_size_ceil_div(requirements.size,
                                                  physical_block_size);

  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(allocator);
  iree_hal_vulkan_sparse_buffer_t* buffer = NULL;
  iree_host_size_t total_size =
      iree_host_align(sizeof(*buffer), iree_max_align_t) +
      sizeof(buffer->physical_blocks[0]) * physical_block_count;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&buffer));
  iree_hal_buffer_initialize(
      host_allocator, allocator, &buffer->base.base, allocation_size,
      byte_offset, byte_length, memory_type, allowed_access, allowed_usage,
      &iree_hal_vulkan_sparse_buffer_vtable, &buffer->base.base);
  buffer->base.handle = handle;
  buffer->logical_device = logical_device;
  buffer->physical_block_count = physical_block_count;

  // Synchronously commit all physical blocks and bind them to the buffer.
  iree_status_t status = iree_hal_vulkan_sparse_buffer_commit_sync(
      logical_device, queue, handle, requirements, memory_type_index,
      physical_block_size, physical_block_count, buffer->physical_blocks);

  if (iree_status_is_ok(status)) {
    *out_buffer = &buffer->base.base;
  } else {
    iree_hal_buffer_destroy((iree_hal_buffer_t*)buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_sparse_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_sparse_buffer_t* buffer =
      iree_hal_vulkan_sparse_buffer_cast(base_buffer);
  iree::hal::vulkan::VkDeviceHandle* logical_device = buffer->logical_device;
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  // Destroy buffer prior to freeing physical blocks.
  if (buffer->base.handle != VK_NULL_HANDLE) {
    logical_device->syms()->vkDestroyBuffer(
        *logical_device, buffer->base.handle, logical_device->allocator());
  }
  for (iree_host_size_t i = 0; i < buffer->physical_block_count; ++i) {
    if (buffer->physical_blocks[i] != VK_NULL_HANDLE) {
      logical_device->syms()->vkFreeMemory(*logical_device,
                                           buffer->physical_blocks[i],
                                           logical_device->allocator());
    }
  }

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_sparse_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "sparse buffers do not support mapping");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "sparse buffers do not support mapping");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "sparse buffers do not support mapping");
}

static iree_status_t iree_hal_vulkan_sparse_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "sparse buffers do not support mapping");
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_vulkan_sparse_buffer_vtable = {
    /*.recycle=*/iree_hal_buffer_recycle,
    /*.destroy=*/iree_hal_vulkan_sparse_buffer_destroy,
    /*.map_range=*/iree_hal_vulkan_sparse_buffer_map_range,
    /*.unmap_range=*/iree_hal_vulkan_sparse_buffer_unmap_range,
    /*.invalidate_range=*/iree_hal_vulkan_sparse_buffer_invalidate_range,
    /*.flush_range=*/iree_hal_vulkan_sparse_buffer_flush_range,
};
}  // namespace
