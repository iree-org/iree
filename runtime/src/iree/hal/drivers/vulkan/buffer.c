// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/vulkan/buffer.h"

#include <string.h>

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_buffer_t
//===----------------------------------------------------------------------===//

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_ALLOCATOR_ID = "iree-hal-vulkan-unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_vulkan_buffer_t {
  // Base HAL buffer resource returned to callers.
  iree_hal_buffer_t base;

  // Host allocator used to free wrapper storage.
  iree_allocator_t host_allocator;

  // Device-level Vulkan dispatch table copied from the creating device.
  iree_hal_vulkan_device_syms_t syms;

  // Vulkan logical device that owns |handle| and |device_memory|.
  VkDevice logical_device;

  // Memory backing |handle|. May be VK_NULL_HANDLE for future sparse buffers.
  VkDeviceMemory device_memory;

  // Vulkan buffer handle.
  VkBuffer handle;

  // Device pointer returned by vkGetBufferDeviceAddress.
  VkDeviceAddress device_address;

  // Vulkan memory properties for map/flush policy.
  VkMemoryPropertyFlags memory_property_flags;

  // Physical device nonCoherentAtomSize used for mapped-memory ranges.
  VkDeviceSize non_coherent_atom_size;

  // Optional callback issued after Vulkan resources are released.
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_vulkan_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_vulkan_buffer_vtable;

static iree_hal_vulkan_buffer_t* iree_hal_vulkan_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_buffer_vtable);
  return (iree_hal_vulkan_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_buffer_create(
    const iree_hal_vulkan_device_syms_t* syms, VkDevice logical_device,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_length, VkMemoryPropertyFlags memory_property_flags,
    VkDeviceSize non_coherent_atom_size, VkDeviceMemory device_memory,
    VkBuffer handle, VkDeviceAddress device_address,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(syms);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);

  iree_hal_vulkan_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer));
  memset(buffer, 0, sizeof(*buffer));
  iree_hal_buffer_initialize(placement, &buffer->base, allocation_size,
                             /*byte_offset=*/0, byte_length, memory_type,
                             allowed_access, allowed_usage,
                             &iree_hal_vulkan_buffer_vtable, &buffer->base);
  buffer->host_allocator = host_allocator;
  buffer->syms = *syms;
  buffer->logical_device = logical_device;
  buffer->device_memory = device_memory;
  buffer->handle = handle;
  buffer->device_address = device_address;
  buffer->memory_property_flags = memory_property_flags;
  buffer->non_coherent_atom_size =
      non_coherent_atom_size ? non_coherent_atom_size : 1;
  buffer->release_callback = release_callback;

  *out_buffer = &buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_vulkan_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(
      z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  if (buffer->handle) {
    IREE_TRACE_FREE_NAMED(IREE_HAL_VULKAN_ALLOCATOR_ID, (void*)buffer->handle);
    iree_vkDestroyBuffer(IREE_VULKAN_DEVICE(&buffer->syms),
                         buffer->logical_device, buffer->handle,
                         /*pAllocator=*/NULL);
  }
  if (buffer->device_memory) {
    iree_vkFreeMemory(IREE_VULKAN_DEVICE(&buffer->syms), buffer->logical_device,
                      buffer->device_memory,
                      /*pAllocator=*/NULL);
  }
  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_vulkan_buffer_isa(iree_hal_buffer_t* buffer) {
  return iree_hal_resource_is((const iree_hal_resource_t*)buffer,
                              &iree_hal_vulkan_buffer_vtable);
}

iree_status_t iree_hal_vulkan_buffer_handle(iree_hal_buffer_t* buffer,
                                            VkDeviceMemory* out_memory,
                                            VkBuffer* out_handle) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_memory);
  IREE_ASSERT_ARGUMENT(out_handle);
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_vulkan_buffer_isa(allocated_buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer is not backed by the Vulkan HAL rewrite");
  }
  iree_hal_vulkan_buffer_t* vulkan_buffer =
      iree_hal_vulkan_buffer_cast(allocated_buffer);
  *out_memory = vulkan_buffer->device_memory;
  *out_handle = vulkan_buffer->handle;
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_buffer_device_address(
    iree_hal_buffer_t* buffer, VkDeviceAddress* out_device_address) {
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_device_address);
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_vulkan_buffer_isa(allocated_buffer)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "buffer is not backed by the Vulkan HAL rewrite");
  }
  iree_hal_vulkan_buffer_t* vulkan_buffer =
      iree_hal_vulkan_buffer_cast(allocated_buffer);
  *out_device_address =
      vulkan_buffer->device_address + iree_hal_buffer_byte_offset(buffer);
  return iree_ok_status();
}

static bool iree_hal_vulkan_buffer_is_host_coherent(
    const iree_hal_vulkan_buffer_t* buffer) {
  return iree_all_bits_set(buffer->memory_property_flags,
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

static VkMappedMemoryRange iree_hal_vulkan_buffer_make_mapped_memory_range(
    const iree_hal_vulkan_buffer_t* buffer,
    iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  const VkDeviceSize atom_size = buffer->non_coherent_atom_size;
  const VkDeviceSize atom_mask = atom_size - 1;
  const VkDeviceSize allocation_size =
      (VkDeviceSize)iree_hal_buffer_allocation_size(&buffer->base);
  VkDeviceSize range_offset = (VkDeviceSize)local_byte_offset & ~atom_mask;
  VkDeviceSize range_end =
      (VkDeviceSize)local_byte_offset + (VkDeviceSize)local_byte_length;
  if (range_end < allocation_size) {
    range_end = (range_end + atom_mask) & ~atom_mask;
  }
  if (range_end > allocation_size) range_end = allocation_size;
  return (VkMappedMemoryRange){
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = buffer->device_memory,
      .offset = range_offset,
      .size = range_end == allocation_size ? VK_WHOLE_SIZE
                                           : range_end - range_offset,
  };
}

static iree_status_t iree_hal_vulkan_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  (void)memory_access;

  if (!buffer->device_memory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "Vulkan buffer has no dense device memory to map");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer),
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer),
      mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
          ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
          : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));
  if (local_byte_length == 0) {
    mapping->contents = iree_make_byte_span(NULL, 0);
    return iree_ok_status();
  }

  void* data = NULL;
  IREE_RETURN_IF_ERROR(iree_vkMapMemory(
      IREE_VULKAN_DEVICE(&buffer->syms), buffer->logical_device,
      buffer->device_memory, (VkDeviceSize)local_byte_offset,
      (VkDeviceSize)local_byte_length, /*flags=*/0, &data));
  mapping->contents = iree_make_byte_span(data, local_byte_length);

#if !defined(NDEBUG)
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(mapping->contents.data, 0xCD, local_byte_length);
  }
#endif  // !defined(NDEBUG)

  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  (void)local_byte_offset;
  (void)local_byte_length;
  (void)mapping;
  iree_vkUnmapMemory(IREE_VULKAN_DEVICE(&buffer->syms), buffer->logical_device,
                     buffer->device_memory);
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  if (local_byte_length == 0 ||
      iree_hal_vulkan_buffer_is_host_coherent(buffer)) {
    return iree_ok_status();
  }
  VkMappedMemoryRange range = iree_hal_vulkan_buffer_make_mapped_memory_range(
      buffer, local_byte_offset, local_byte_length);
  return iree_vkInvalidateMappedMemoryRanges(IREE_VULKAN_DEVICE(&buffer->syms),
                                             buffer->logical_device,
                                             /*memoryRangeCount=*/1, &range);
}

static iree_status_t iree_hal_vulkan_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_buffer_t* buffer = iree_hal_vulkan_buffer_cast(base_buffer);
  if (local_byte_length == 0 ||
      iree_hal_vulkan_buffer_is_host_coherent(buffer)) {
    return iree_ok_status();
  }
  VkMappedMemoryRange range = iree_hal_vulkan_buffer_make_mapped_memory_range(
      buffer, local_byte_offset, local_byte_length);
  return iree_vkFlushMappedMemoryRanges(IREE_VULKAN_DEVICE(&buffer->syms),
                                        buffer->logical_device,
                                        /*memoryRangeCount=*/1, &range);
}

static const iree_hal_buffer_vtable_t iree_hal_vulkan_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_vulkan_buffer_destroy,
    .map_range = iree_hal_vulkan_buffer_map_range,
    .unmap_range = iree_hal_vulkan_buffer_unmap_range,
    .invalidate_range = iree_hal_vulkan_buffer_invalidate_range,
    .flush_range = iree_hal_vulkan_buffer_flush_range,
};
