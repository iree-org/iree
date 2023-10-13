// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/native_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/base_buffer.h"
#include "iree/hal/drivers/vulkan/status_util.h"

typedef struct iree_hal_vulkan_native_buffer_t {
  iree_hal_vulkan_base_buffer_t base;
  iree::hal::vulkan::VkDeviceHandle* logical_device;
  iree_hal_vulkan_native_buffer_release_callback_t internal_release_callback;
  iree_hal_buffer_release_callback_t user_release_callback;
} iree_hal_vulkan_native_buffer_t;

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_vulkan_native_buffer_vtable;
}  // namespace

static iree_hal_vulkan_native_buffer_t* iree_hal_vulkan_native_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_buffer_vtable);
  return (iree_hal_vulkan_native_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkDeviceMemory device_memory, VkBuffer handle,
    iree_hal_vulkan_native_buffer_release_callback_t internal_release_callback,
    iree_hal_buffer_release_callback_t user_release_callback,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(allocator);
  iree_hal_vulkan_native_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(
        host_allocator, allocator, &buffer->base.base, allocation_size,
        byte_offset, byte_length, memory_type, allowed_access, allowed_usage,
        &iree_hal_vulkan_native_buffer_vtable, &buffer->base.base);
    buffer->base.device_memory = device_memory;
    buffer->base.handle = handle;
    buffer->logical_device = logical_device;
    buffer->internal_release_callback = internal_release_callback;
    buffer->user_release_callback = user_release_callback;

    *out_buffer = &buffer->base.base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_buffer_destroy(
    iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_native_buffer_t* buffer =
      iree_hal_vulkan_native_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  if (buffer->internal_release_callback.fn) {
    buffer->internal_release_callback.fn(
        buffer->internal_release_callback.user_data, buffer->logical_device,
        buffer->base.device_memory, buffer->base.handle);
  }
  if (buffer->user_release_callback.fn) {
    buffer->user_release_callback.fn(buffer->user_release_callback.user_data,
                                     &buffer->base.base);
  }

  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_vulkan_native_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_native_buffer_t* buffer =
      iree_hal_vulkan_native_buffer_cast(base_buffer);
  if (IREE_UNLIKELY(!buffer->base.device_memory)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  auto* logical_device = buffer->logical_device;

  // TODO(benvanik): add upload/download for unmapped buffers.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  // TODO(benvanik): map VK_WHOLE_SIZE and subset ourselves? may need to get
  // around some minimum mapping alignment rules.
  uint8_t* data_ptr = nullptr;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkMapMemory(
                         *logical_device, buffer->base.device_memory,
                         /*offset=*/local_byte_offset,
                         /*size=*/local_byte_length,
                         /*flags=*/0, (void**)&data_ptr),
                     "vkMapMemory");
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(mapping->contents.data, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_native_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_native_buffer_t* buffer =
      iree_hal_vulkan_native_buffer_cast(base_buffer);
  if (IREE_UNLIKELY(!buffer->base.device_memory)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  auto* logical_device = buffer->logical_device;
  logical_device->syms()->vkUnmapMemory(*logical_device,
                                        buffer->base.device_memory);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_native_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_native_buffer_t* buffer =
      iree_hal_vulkan_native_buffer_cast(base_buffer);
  if (IREE_UNLIKELY(!buffer->base.device_memory)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  auto* logical_device = buffer->logical_device;
  VkMappedMemoryRange range;
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.pNext = NULL;
  range.memory = buffer->base.device_memory;
  range.offset = local_byte_offset;
  range.size = local_byte_length;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkInvalidateMappedMemoryRanges(
                         *logical_device, 1, &range),
                     "vkInvalidateMappedMemoryRanges");
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_native_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_native_buffer_t* buffer =
      iree_hal_vulkan_native_buffer_cast(base_buffer);
  if (IREE_UNLIKELY(!buffer->base.device_memory)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "buffer does not have device memory attached and cannot be mapped");
  }
  auto* logical_device = buffer->logical_device;
  VkMappedMemoryRange range;
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.pNext = NULL;
  range.memory = buffer->base.device_memory;
  range.offset = local_byte_offset;
  range.size = local_byte_length;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkFlushMappedMemoryRanges(
                         *logical_device, 1, &range),
                     "vkFlushMappedMemoryRanges");
  return iree_ok_status();
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_vulkan_native_buffer_vtable = {
    /*.recycle=*/iree_hal_buffer_recycle,
    /*.destroy=*/iree_hal_vulkan_native_buffer_destroy,
    /*.map_range=*/iree_hal_vulkan_native_buffer_map_range,
    /*.unmap_range=*/iree_hal_vulkan_native_buffer_unmap_range,
    /*.invalidate_range=*/iree_hal_vulkan_native_buffer_invalidate_range,
    /*.flush_range=*/iree_hal_vulkan_native_buffer_flush_range,
};
}  // namespace
