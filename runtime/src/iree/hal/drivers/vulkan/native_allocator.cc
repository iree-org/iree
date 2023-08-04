// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/vulkan/base_buffer.h"
#include "iree/hal/drivers/vulkan/dynamic_symbols.h"
#include "iree/hal/drivers/vulkan/native_buffer.h"
#include "iree/hal/drivers/vulkan/status_util.h"

using namespace iree::hal::vulkan;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_NATIVE_ALLOCATOR_ID = "Vulkan/Native";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_vulkan_native_allocator_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  iree_hal_device_t* device;  // unretained to avoid cycles
  iree_allocator_t host_allocator;

  // Cached from the API to avoid additional queries in hot paths.
  VkPhysicalDeviceProperties device_props;
  VkPhysicalDeviceMemoryProperties memory_props;

  // Used to quickly look up the memory type index used for a particular usage.
  iree_hal_vulkan_memory_types_t memory_types;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_vulkan_native_allocator_t;

namespace {
extern const iree_hal_allocator_vtable_t
    iree_hal_vulkan_native_allocator_vtable;
}  // namespace

static iree_hal_vulkan_native_allocator_t*
iree_hal_vulkan_native_allocator_cast(iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_allocator_vtable);
  return (iree_hal_vulkan_native_allocator_t*)base_value;
}

static void iree_hal_vulkan_native_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator);

extern "C" iree_status_t iree_hal_vulkan_native_allocator_create(
    const iree_hal_vulkan_device_options_t* options, VkInstance instance,
    VkPhysicalDevice physical_device, VkDeviceHandle* logical_device,
    iree_hal_device_t* device, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = logical_device->host_allocator();
  iree_hal_vulkan_native_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_vulkan_native_allocator_vtable,
                               &allocator->resource);
  allocator->logical_device = logical_device;
  allocator->device = device;
  allocator->host_allocator = host_allocator;

  const auto& syms = logical_device->syms();
  syms->vkGetPhysicalDeviceProperties(physical_device,
                                      &allocator->device_props);
  syms->vkGetPhysicalDeviceMemoryProperties(physical_device,
                                            &allocator->memory_props);
  iree_status_t status = iree_hal_vulkan_populate_memory_types(
      &allocator->device_props, &allocator->memory_props,
      &allocator->memory_types);

  if (iree_status_is_ok(status)) {
    *out_allocator = (iree_hal_allocator_t*)allocator;
  } else {
    iree_hal_vulkan_native_allocator_destroy((iree_hal_allocator_t*)allocator);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_vulkan_native_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_vulkan_native_allocator_t* allocator =
      (iree_hal_vulkan_native_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_vulkan_native_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_vulkan_native_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_vulkan_native_allocator_t* allocator =
        iree_hal_vulkan_native_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_status_t iree_hal_vulkan_native_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);
  return iree_hal_vulkan_query_memory_heaps(
      &allocator->device_props, &allocator->memory_props,
      &allocator->memory_types, capacity, heaps, out_count);
}

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_native_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  // Align allocation sizes to 4 bytes so shaders operating on 32 bit types can
  // act safely even on buffer ranges that are not naturally aligned.
  *allocation_size = iree_host_align(*allocation_size, 4);

  return compatibility;
}

static void iree_hal_vulkan_native_allocator_native_buffer_release(
    void* user_data, iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkDeviceMemory device_memory, VkBuffer handle) {
  IREE_TRACE_FREE_NAMED(IREE_HAL_VULKAN_NATIVE_ALLOCATOR_ID, (void*)handle);
  logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                          logical_device->allocator());
  logical_device->syms()->vkFreeMemory(*logical_device, device_memory,
                                       logical_device->allocator());
}

static iree_status_t iree_hal_vulkan_native_allocator_allocate_internal(
    iree_hal_vulkan_native_allocator_t* IREE_RESTRICT allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  VkDeviceHandle* logical_device = allocator->logical_device;

  // TODO(benvanik): if on a unified memory system and initial data is present
  // we could set the mapping bit and ensure a much more efficient upload.

  // Allocate the device memory we'll attach the buffer to.
  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.pNext = NULL;
  allocate_info.memoryTypeIndex = 0;
  allocate_info.allocationSize = allocation_size;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_find_memory_type(
      &allocator->device_props, &allocator->memory_props, params,
      &allocate_info.memoryTypeIndex));
  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkAllocateMemory(
                         *logical_device, &allocate_info,
                         logical_device->allocator(), &device_memory),
                     "vkAllocateMemory");

  // Create an initially unbound buffer handle.
  VkBufferCreateInfo buffer_create_info = {};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = NULL;
  buffer_create_info.flags = 0;
  buffer_create_info.size = allocation_size;
  buffer_create_info.usage = 0;
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (iree_all_bits_set(params->usage,
                        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = NULL;
  VkBuffer handle = VK_NULL_HANDLE;
  iree_status_t status =
      VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateBuffer(
                              *logical_device, &buffer_create_info,
                              logical_device->allocator(), &handle),
                          "vkCreateBuffer");

  iree_hal_vulkan_native_buffer_release_callback_t release_callback = {0};
  release_callback.fn = iree_hal_vulkan_native_allocator_native_buffer_release;
  release_callback.user_data = NULL;
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_native_buffer_wrap(
        (iree_hal_allocator_t*)allocator, params->type, params->access,
        params->usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, logical_device, device_memory, handle,
        release_callback, &buffer);
  }
  if (!iree_status_is_ok(status)) {
    // Early exit after cleaning up the buffer and allocation.
    // After this point releasing the wrapping buffer will take care of this.
    if (handle) {
      logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                              logical_device->allocator());
    }
    if (device_memory) {
      logical_device->syms()->vkFreeMemory(*logical_device, device_memory,
                                           logical_device->allocator());
    }
    return status;
  }

  IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_NATIVE_ALLOCATOR_ID, (void*)handle,
                         allocation_size);

  // Bind the memory to the buffer.
  if (iree_status_is_ok(status)) {
    status = VK_RESULT_TO_STATUS(
        logical_device->syms()->vkBindBufferMemory(
            *logical_device, handle, device_memory, /*memoryOffset=*/0),
        "vkBindBufferMemory");
  }

  // Copy the initial contents into the buffer. This may require staging.
  if (iree_status_is_ok(status) &&
      !iree_const_byte_span_is_empty(initial_data)) {
    status = iree_hal_device_transfer_range(
        allocator->device,
        iree_hal_make_host_transfer_buffer_span((void*)initial_data.data,
                                                initial_data.data_length),
        0, iree_hal_make_device_transfer_buffer(buffer), 0,
        initial_data.data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, params->type, buffer->allocation_size);
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_native_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, iree_const_byte_span_t initial_data,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  if (!iree_all_bits_set(
          iree_hal_vulkan_native_allocator_query_buffer_compatibility(
              base_allocator, &compat_params, &allocation_size),
          IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  return iree_hal_vulkan_native_allocator_allocate_internal(
      allocator, &compat_params, allocation_size, initial_data, out_buffer);
}

static void iree_hal_vulkan_native_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);
  (void)allocator;
  iree_hal_allocator_statistics_record_free(&allocator->statistics,
                                            base_buffer->memory_type,
                                            base_buffer->allocation_size);
  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_vulkan_native_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO(#7242): use VK_EXT_external_memory_host to import memory.
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "importing from external buffers not supported");
}

static iree_status_t iree_hal_vulkan_native_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "exporting to external buffers not supported");
}

namespace {
const iree_hal_allocator_vtable_t iree_hal_vulkan_native_allocator_vtable = {
    /*.destroy=*/iree_hal_vulkan_native_allocator_destroy,
    /*.host_allocator=*/iree_hal_vulkan_native_allocator_host_allocator,
    /*.trim=*/iree_hal_vulkan_native_allocator_trim,
    /*.query_statistics=*/iree_hal_vulkan_native_allocator_query_statistics,
    /*.query_memory_heaps=*/iree_hal_vulkan_native_allocator_query_memory_heaps,
    /*.query_buffer_compatibility=*/
    iree_hal_vulkan_native_allocator_query_buffer_compatibility,
    /*.allocate_buffer=*/iree_hal_vulkan_native_allocator_allocate_buffer,
    /*.deallocate_buffer=*/iree_hal_vulkan_native_allocator_deallocate_buffer,
    /*.import_buffer=*/iree_hal_vulkan_native_allocator_import_buffer,
    /*.export_buffer=*/iree_hal_vulkan_native_allocator_export_buffer,
};
}  // namespace
