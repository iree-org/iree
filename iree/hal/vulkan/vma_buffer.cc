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

#include "iree/hal/vulkan/vma_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

typedef struct iree_hal_vulkan_vma_buffer_s {
  iree_hal_buffer_t base;

  VmaAllocator vma;
  VkBuffer handle;
  VmaAllocation allocation;
  VmaAllocationInfo allocation_info;
} iree_hal_vulkan_vma_buffer_t;

extern const iree_hal_buffer_vtable_t iree_hal_vulkan_vma_buffer_vtable;

static iree_hal_vulkan_vma_buffer_t* iree_hal_vulkan_vma_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_vma_buffer_vtable);
  return (iree_hal_vulkan_vma_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_vma_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    VmaAllocator vma, VkBuffer handle, VmaAllocation allocation,
    VmaAllocationInfo allocation_info, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(vma);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_vulkan_vma_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(iree_hal_allocator_host_allocator(allocator),
                            sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_vulkan_vma_buffer_vtable,
                                 &buffer->base.resource);
    buffer->base.allocator = allocator;
    buffer->base.allocated_buffer = &buffer->base;
    buffer->base.allocation_size = allocation_size;
    buffer->base.byte_offset = byte_offset;
    buffer->base.byte_length = byte_length;
    buffer->base.memory_type = memory_type;
    buffer->base.allowed_access = allowed_access;
    buffer->base.allowed_usage = allowed_usage;
    buffer->vma = vma;
    buffer->handle = handle;
    buffer->allocation = allocation;
    buffer->allocation_info = allocation_info;

    // TODO(benvanik): set debug name instead and use the
    //     VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT flag.
    vmaSetAllocationUserData(buffer->vma, buffer->allocation, buffer);

    *out_buffer = &buffer->base;
  } else {
    vmaDestroyBuffer(vma, handle, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_vulkan_vma_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(iree_hal_buffer_allocator(base_buffer));
  IREE_TRACE_ZONE_BEGIN(z0);

  vmaDestroyBuffer(buffer->vma, buffer->handle, buffer->allocation);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

VkBuffer iree_hal_vulkan_vma_buffer_handle(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  return buffer->handle;
}

static iree_status_t iree_hal_vulkan_vma_buffer_fill(
    iree_hal_buffer_t* base_buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_buffer_mapping_t target_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      base_buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, byte_offset,
      byte_length, &target_mapping));
  iree_status_t status = iree_ok_status();
  void* data_ptr = target_mapping.contents.data;
  switch (pattern_length) {
    case 1: {
      uint8_t* data = (uint8_t*)data_ptr;
      uint8_t value_bits = *(const uint8_t*)(pattern);
      memset(data, value_bits, byte_length);
      break;
    }
    case 2: {
      uint16_t* data = (uint16_t*)data_ptr;
      uint16_t value_bits = *(const uint16_t*)(pattern);
      for (iree_device_size_t i = 0; i < byte_length / sizeof(uint16_t); ++i) {
        data[i] = value_bits;
      }
      break;
    }
    case 4: {
      uint32_t* data = (uint32_t*)data_ptr;
      uint32_t value_bits = *(const uint32_t*)(pattern);
      for (iree_device_size_t i = 0; i < byte_length / sizeof(uint32_t); ++i) {
        data[i] = value_bits;
      }
      break;
    }
    default:
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported fill pattern length: %zu",
                                pattern_length);
      break;
  }
  iree_hal_buffer_flush_range(&target_mapping, byte_offset, byte_length);
  iree_status_ignore(iree_hal_buffer_unmap_range(&target_mapping));
  return status;
}

static iree_status_t iree_hal_vulkan_vma_buffer_read_data(
    iree_hal_buffer_t* base_buffer, iree_device_size_t source_offset,
    void* target_buffer, iree_device_size_t data_length) {
  iree_hal_buffer_mapping_t source_mapping;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_range(base_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                source_offset, data_length, &source_mapping));
  memcpy(target_buffer, source_mapping.contents.data, data_length);
  return iree_hal_buffer_unmap_range(&source_mapping);
}

static iree_status_t iree_hal_vulkan_vma_buffer_write_data(
    iree_hal_buffer_t* base_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_device_size_t data_length) {
  iree_hal_buffer_mapping_t target_mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      base_buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, target_offset,
      data_length, &target_mapping));
  memcpy(target_mapping.contents.data, source_buffer, data_length);
  return iree_hal_buffer_unmap_range(&target_mapping);
}

static iree_status_t iree_hal_vulkan_vma_buffer_copy_data(
    iree_hal_buffer_t* base_source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* base_target_buffer, iree_device_size_t target_offset,
    iree_device_size_t data_length) {
  iree_hal_buffer_mapping_t source_mapping;
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_map_range(base_source_buffer, IREE_HAL_MEMORY_ACCESS_READ,
                                source_offset, data_length, &source_mapping));
  iree_hal_buffer_mapping_t target_mapping;
  iree_status_t status = iree_hal_buffer_map_range(
      base_target_buffer, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE, target_offset,
      data_length, &target_mapping);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    IREE_IGNORE_ERROR(iree_hal_buffer_unmap_range(&source_mapping));
    return status;
  }

  memcpy(target_mapping.contents.data, source_mapping.contents.data,
         data_length);

  IREE_IGNORE_ERROR(iree_hal_buffer_unmap_range(&source_mapping));
  IREE_IGNORE_ERROR(iree_hal_buffer_unmap_range(&target_mapping));
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    void** out_data_ptr) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);

  uint8_t* data_ptr = nullptr;
  VK_RETURN_IF_ERROR(
      vmaMapMemory(buffer->vma, buffer->allocation, (void**)&data_ptr),
      "vmaMapMemory");
  *out_data_ptr = data_ptr + local_byte_offset;

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr + local_byte_offset, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  return iree_ok_status();
}

static void iree_hal_vulkan_vma_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  vmaUnmapMemory(buffer->vma, buffer->allocation);
}

static iree_status_t iree_hal_vulkan_vma_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  VK_RETURN_IF_ERROR(
      vmaInvalidateAllocation(buffer->vma, buffer->allocation,
                              local_byte_offset, local_byte_length),
      "vmaInvalidateAllocation");
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  VK_RETURN_IF_ERROR(vmaFlushAllocation(buffer->vma, buffer->allocation,
                                        local_byte_offset, local_byte_length),
                     "vmaFlushAllocation");
  return iree_ok_status();
}

const iree_hal_buffer_vtable_t iree_hal_vulkan_vma_buffer_vtable = {
    /*.destroy=*/iree_hal_vulkan_vma_buffer_destroy,
    /*.fill=*/iree_hal_vulkan_vma_buffer_fill,
    /*.read_data=*/iree_hal_vulkan_vma_buffer_read_data,
    /*.write_data=*/iree_hal_vulkan_vma_buffer_write_data,
    /*.copy_data=*/iree_hal_vulkan_vma_buffer_copy_data,
    /*.map_range=*/iree_hal_vulkan_vma_buffer_map_range,
    /*.unmap_range=*/iree_hal_vulkan_vma_buffer_unmap_range,
    /*.invalidate_range=*/iree_hal_vulkan_vma_buffer_invalidate_range,
    /*.flush_range=*/iree_hal_vulkan_vma_buffer_flush_range,
};
