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
#include "iree/hal/drivers/vulkan/sparse_buffer.h"
#include "iree/hal/drivers/vulkan/status_util.h"

#if defined(IREE_PLATFORM_LINUX)
#include <sys/mman.h>
#endif  // IREE_PLATFORM_LINUX

using namespace iree::hal::vulkan;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_VULKAN_NATIVE_ALLOCATOR_ID = "Vulkan/Native";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_vulkan_native_allocator_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  iree_allocator_t host_allocator;

  // Cached from the API to avoid additional queries in hot paths.
  VkPhysicalDeviceProperties device_props;
  VkPhysicalDeviceVulkan11Properties device_props_11;
  VkPhysicalDeviceMemoryProperties memory_props;
  VkDeviceSize min_imported_host_pointer_alignment;

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
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(instance);
  IREE_ASSERT_ARGUMENT(physical_device);
  IREE_ASSERT_ARGUMENT(logical_device);
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
  allocator->host_allocator = host_allocator;

  const auto& syms = logical_device->syms();

  VkPhysicalDeviceExternalMemoryHostPropertiesEXT external_memory_props;
  external_memory_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
  external_memory_props.pNext = NULL;
  allocator->device_props_11.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
  allocator->device_props_11.pNext = &external_memory_props;
  VkPhysicalDeviceProperties2 device_props_2;
  device_props_2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  device_props_2.pNext = &allocator->device_props_11;
  syms->vkGetPhysicalDeviceProperties2(physical_device, &device_props_2);
  allocator->device_props = device_props_2.properties;
  syms->vkGetPhysicalDeviceMemoryProperties(physical_device,
                                            &allocator->memory_props);
  allocator->min_imported_host_pointer_alignment =
      external_memory_props.minImportedHostPointerAlignment;

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

// Returns true if a buffer with the given parameters and size should use
// sparse binding to attach segmented device memory to a single buffer.
static bool iree_hal_vulkan_buffer_needs_sparse_binding(
    iree_hal_vulkan_native_allocator_t* allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size) {
  if (allocation_size <= allocator->device_props_11.maxMemoryAllocationSize) {
    return false;  // fits under the normal allocation limit
  }
  return true;
}

static iree_hal_buffer_compatibility_t
iree_hal_vulkan_native_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);

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

  // Sparse allocations are used only when required and supported.
  const bool use_sparse_allocation =
      iree_hal_vulkan_buffer_needs_sparse_binding(allocator, params,
                                                  *allocation_size);
  if (use_sparse_allocation &&
      iree_all_bits_set(allocator->logical_device->enabled_features(),
                        IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING)) {
    // For now we don't allow import/export of sparsely bound buffers. This is
    // not a strict Vulkan requirement but it does complicate things as we
    // cannot get a single VkDeviceMemory handle to use in managing the external
    // buffer.
    compatibility &= ~IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE |
                     IREE_HAL_BUFFER_COMPATIBILITY_EXPORTABLE;
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                             IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT)) {
      if (iree_all_bits_set(params->usage,
                            IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL)) {
        // Mapping was optionally requested and sparse buffers can't be mapped
        // so we strip the request flags.
        params->usage &=
            ~(IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
              IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT |
              IREE_HAL_BUFFER_USAGE_MAPPING_OPTIONAL |
              IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_RANDOM |
              IREE_HAL_BUFFER_USAGE_MAPPING_ACCESS_SEQUENTIAL_WRITE);
      } else {
        // Mapping required but cannot be serviced with sparse bindings.
        compatibility = IREE_HAL_BUFFER_COMPATIBILITY_NONE;
      }
    }
  } else if (*allocation_size >=
             allocator->device_props_11.maxMemoryAllocationSize) {
    // Cannot allocate buffers larger than the max allowed without sparse
    // binding.
    compatibility = IREE_HAL_BUFFER_COMPATIBILITY_NONE;
  }

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

static iree_status_t iree_hal_vulkan_native_allocator_commit_and_wrap(
    iree_hal_vulkan_native_allocator_t* IREE_RESTRICT allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, bool use_sparse_allocation,
    VkBuffer handle, iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  VkDeviceHandle* logical_device = allocator->logical_device;

  // TODO(benvanik): map queue affinity.
  VkQueue queue = VK_NULL_HANDLE;
  logical_device->syms()->vkGetDeviceQueue(*logical_device, 0, 0, &queue);

  // Ask Vulkan what the implementation requires of the allocation(s) for the
  // buffer. We should in most cases always get the same kind of values but
  // alignment and valid memory types will differ for dense and sparse buffers.
  VkMemoryRequirements requirements = {0};
  logical_device->syms()->vkGetBufferMemoryRequirements(*logical_device, handle,
                                                        &requirements);
  uint32_t memory_type_index = 0;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_find_memory_type(
      &allocator->device_props, &allocator->memory_props, params,
      /*allowed_type_indices=*/requirements.memoryTypeBits,
      &memory_type_index));

  if (use_sparse_allocation) {
    // Use sparse allocation for this buffer in order to exceed the maximum
    // allocation size of the implementation. This is not a very efficient way
    // to allocate such buffers (synchronously from raw allocations) but this
    // path is primarily used by large persistent variables and constants.
    return iree_hal_vulkan_sparse_buffer_create_bound_sync(
        (iree_hal_allocator_t*)allocator, params->type, params->access,
        params->usage, allocation_size, /*byte_offset=*/0,
        /*byte_length=*/allocation_size, logical_device, queue, handle,
        requirements, memory_type_index,
        allocator->device_props_11.maxMemoryAllocationSize, out_buffer);
  }

  // Allocate the device memory we'll attach the buffer to.
  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.pNext = NULL;
  allocate_info.allocationSize = requirements.size;
  allocate_info.memoryTypeIndex = memory_type_index;
  VkMemoryAllocateFlagsInfo allocate_flags_info = {};
  allocate_flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  allocate_flags_info.pNext = NULL;
  allocate_flags_info.flags = 0;
  if (iree_all_bits_set(
          logical_device->enabled_features(),
          IREE_HAL_VULKAN_FEATURE_ENABLE_BUFFER_DEVICE_ADDRESSES)) {
    allocate_flags_info.flags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  }
  allocate_flags_info.deviceMask = 0;
  allocate_info.pNext = &allocate_flags_info;
  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkAllocateMemory(
                         *logical_device, &allocate_info,
                         logical_device->allocator(), &device_memory),
                     "vkAllocateMemory");

  // Wrap the device memory allocation and buffer handle in our own buffer type.
  iree_hal_vulkan_native_buffer_release_callback_t internal_release_callback = {
      0};
  internal_release_callback.fn =
      iree_hal_vulkan_native_allocator_native_buffer_release;
  internal_release_callback.user_data = NULL;
  iree_status_t status = iree_hal_vulkan_native_buffer_wrap(
      (iree_hal_allocator_t*)allocator, params->type, params->access,
      params->usage, allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/allocation_size, logical_device, device_memory, handle,
      internal_release_callback, iree_hal_buffer_release_callback_null(),
      out_buffer);
  if (!iree_status_is_ok(status)) {
    logical_device->syms()->vkFreeMemory(*logical_device, device_memory,
                                         logical_device->allocator());
    return status;
  }

  // Bind the memory to the buffer.
  if (iree_status_is_ok(status)) {
    status = VK_RESULT_TO_STATUS(
        logical_device->syms()->vkBindBufferMemory(
            *logical_device, handle, device_memory, /*memoryOffset=*/0),
        "vkBindBufferMemory");
  }

  return status;
}

static iree_status_t iree_hal_vulkan_native_allocator_create_buffer(
    VkDeviceHandle* logical_device,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size, bool use_sparse_allocation,
    bool bind_host_memory, VkBuffer* out_handle) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(params);
  *out_handle = VK_NULL_HANDLE;

  // Create an initially unbound buffer handle. The buffer is the logical view
  // into the physical allocation(s) that are bound to it below.
  VkBufferCreateInfo buffer_create_info = {};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = NULL;
  buffer_create_info.flags = 0;
  buffer_create_info.size = allocation_size;
  buffer_create_info.usage = 0;
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  }
  if (iree_all_bits_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  }
  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
    buffer_create_info.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    buffer_create_info.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  }
  if (use_sparse_allocation) {
    buffer_create_info.flags |= VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
                                VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;
  }
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = NULL;

  // If trying to bind to external memory we need to verify we can create a
  // buffer that can be bound.
  if (bind_host_memory) {
    VkPhysicalDeviceExternalBufferInfo external_info;
    memset(&external_info, 0, sizeof(external_info));
    external_info.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    external_info.pNext = NULL;
    external_info.flags = buffer_create_info.flags;
    external_info.usage = buffer_create_info.usage;
    external_info.handleType =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkExternalBufferProperties external_props;
    memset(&external_props, 0, sizeof(external_props));
    external_props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    logical_device->syms()->vkGetPhysicalDeviceExternalBufferProperties(
        logical_device->physical_device(), &external_info, &external_props);
    if (!iree_all_bits_set(
            external_props.externalMemoryProperties.externalMemoryFeatures,
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
#if IREE_STATUS_MODE
      iree_bitfield_string_temp_t temp0;
      iree_string_view_t usage_str =
          iree_hal_buffer_usage_format(params->usage, &temp0);
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "implementation does not support binding "
                              "imported host memory to buffers for usage=%.*s",
                              (int)usage_str.size, usage_str.data);
#else
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
#endif  // IREE_STATUS_MODE
    }
    if (!iree_all_bits_set(
            external_props.externalMemoryProperties.compatibleHandleTypes,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT)) {
#if IREE_STATUS_MODE
      iree_bitfield_string_temp_t temp0;
      iree_string_view_t usage_str =
          iree_hal_buffer_usage_format(params->usage, &temp0);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "implementation does not support binding external host allocations "
          "to buffers for usage=%.*s",
          (int)usage_str.size, usage_str.data);
#else
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
#endif  // IREE_STATUS_MODE
    }
  }

  VkExternalMemoryBufferCreateInfo external_create_info = {};
  if (bind_host_memory) {
    external_create_info.sType =
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_create_info.pNext = NULL;
    external_create_info.handleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    buffer_create_info.pNext = &external_create_info;
  }

  VkBuffer handle = VK_NULL_HANDLE;
  VK_RETURN_IF_ERROR(logical_device->syms()->vkCreateBuffer(
                         *logical_device, &buffer_create_info,
                         logical_device->allocator(), &handle),
                     "vkCreateBuffer");

  *out_handle = handle;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_native_allocator_allocate_internal(
    iree_hal_vulkan_native_allocator_t* IREE_RESTRICT allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  VkDeviceHandle* logical_device = allocator->logical_device;

  // TODO(benvanik): if on a unified memory system and initial data is present
  // we could set the mapping bit and ensure a much more efficient upload.

  // When required and available we allocate buffers using sparse binding.
  const bool use_sparse_allocation =
      iree_hal_vulkan_buffer_needs_sparse_binding(allocator, params,
                                                  allocation_size);
  if (use_sparse_allocation &&
      !iree_all_bits_set(allocator->logical_device->enabled_features(),
                         IREE_HAL_VULKAN_FEATURE_ENABLE_SPARSE_BINDING)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "sparse binding support is required for buffers larger than %" PRId64
        " but is not present or enabled on this device",
        (int64_t)allocator->device_props_11.maxMemoryAllocationSize);
  }

  // Create an initially unbound buffer handle. The buffer is the logical view
  // into the physical allocation(s) that are bound to it below.
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_native_allocator_create_buffer(
      logical_device, params, allocation_size, use_sparse_allocation,
      /*bind_host_memory=*/false, &handle));

  // Commit the backing memory for the buffer and wrap it in a HAL buffer type.
  // If this fails the buffer may still be set and need to be released below.
  // If the buffer is not created the handle needs to be cleaned up.
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_vulkan_native_allocator_commit_and_wrap(
      allocator, params, allocation_size, use_sparse_allocation, handle,
      &buffer);
  if (!iree_status_is_ok(status)) {
    // Early exit and make sure to destroy the buffer if we didn't get the
    // chance to wrap it.
    if (buffer) {
      iree_hal_buffer_release(buffer);
    } else if (handle != VK_NULL_HANDLE) {
      logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                              logical_device->allocator());
    }
    return status;
  }

  IREE_TRACE_ALLOC_NAMED(IREE_HAL_VULKAN_NATIVE_ALLOCATOR_ID, (void*)handle,
                         allocation_size);

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
    iree_device_size_t allocation_size,
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
      allocator, &compat_params, allocation_size, out_buffer);
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

static void iree_hal_vulkan_native_allocator_external_host_buffer_release(
    void* user_data, iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkDeviceMemory device_memory, VkBuffer handle) {
  if (handle) {
    logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                            logical_device->allocator());
  }
  if (device_memory) {
    logical_device->syms()->vkFreeMemory(*logical_device, device_memory,
                                         logical_device->allocator());
  }
}

// Aligns |host_ptr| down to the nearest minimum alignment that satisfies the
// device and buffer requirements. Adjusts |size| up if required to cover at
// least the originally defined range. Returns the offset into the new
// |host_ptr| where the original pointer started.
static VkDeviceSize iree_hal_vulkan_native_allocator_align_external_ptr(
    iree_hal_vulkan_native_allocator_t* allocator,
    const VkMemoryRequirements* requirements, void** host_ptr,
    VkDeviceSize* size) {
  VkDeviceSize desired_alignment = (VkDeviceSize)iree_device_size_lcm(
      (iree_device_size_t)requirements->alignment,
      (iree_device_size_t)allocator->min_imported_host_pointer_alignment);
  VkDeviceSize unaligned_addr = *((VkDeviceSize*)host_ptr);
  VkDeviceSize aligned_addr =
      (unaligned_addr / desired_alignment) * desired_alignment;
  IREE_ASSERT(unaligned_addr >= aligned_addr);
  VkDeviceSize memory_offset = unaligned_addr - aligned_addr;
  *host_ptr = (void*)aligned_addr;

  VkDeviceSize unaligned_size = *size;
  VkDeviceSize unaligned_end = unaligned_addr + unaligned_size;
  IREE_ASSERT(unaligned_end >= aligned_addr);
  VkDeviceSize aligned_end =
      ((unaligned_end + desired_alignment - 1) / desired_alignment) *
      desired_alignment;
  IREE_ASSERT(aligned_end >= unaligned_end);
  VkDeviceSize aligned_size = aligned_end - aligned_addr;
  IREE_ASSERT(aligned_size >= unaligned_size);
  IREE_ASSERT(aligned_size % desired_alignment == 0);
  *size = aligned_size;

  return memory_offset;
}

static iree_status_t iree_hal_vulkan_native_allocator_import_host_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);
  VkDeviceHandle* logical_device = allocator->logical_device;

  // Extension must be present, though note that the presence of the extension
  // does not imply that the particular pointer passed can actually be used.
  if (!logical_device->enabled_extensions().external_memory_host) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "external host memory import is not supported on this device");
  }

#if defined(IREE_PLATFORM_LINUX)
  // First check if the memory is importable.
  // Some drivers incorrectly succeed when attempting to import already-mapped
  // memory: https://gitlab.freedesktop.org/mesa/mesa/-/issues/9251.
  //
  // Attempt to synchronize the file with its memory map.
  // If the memory is not mapped from a file, attempting to synchronize it with
  // its memory map should fail fast and we can import the buffer. If the memory
  // *is* mapped, import may fail on some drivers (this may also be slow).

  // TODO(scotttodd): Further restrict this slow path to buggy drivers only?
  //                  We'd need to plumb some driver information through to here
  errno = 0;
  (void)msync(external_buffer->handle.host_allocation.ptr,
              external_buffer->size, MS_SYNC);
  if (errno != ENOMEM) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot import mapped memory");
  }
#endif  // IREE_PLATFORM_LINUX

  // Query the properties of the pointer to see what memory types it can be
  // imported with. This can be very expensive as on some platforms it does
  // a linear scan of the virtual address range to ensure all pages have the
  // same properties.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_a, "vkGetMemoryHostPointerPropertiesEXT");
  VkMemoryHostPointerPropertiesEXT props = {};
  props.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
  props.pNext = NULL;
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkGetMemoryHostPointerPropertiesEXT(
          *logical_device,
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
          external_buffer->handle.host_allocation.ptr, &props),
      "vkGetMemoryHostPointerPropertiesEXT");
  IREE_TRACE_ZONE_END(z_a);
  IREE_RETURN_IF_ERROR(status);

  // TODO(benvanik): snoop and adjust parameters: if the returned host ptr
  // properties memory types contains allocator->memory_types.dispatch_idx then
  // we can import as device-local! Otherwise we should only allow host-local.
  // For now we just trust the user to have passed the right thing and otherwise
  // they'll get validation errors on any misuse.

  // Create the unbound buffer first as we need it to query the requirements the
  // imported buffer must satisfy.
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_native_allocator_create_buffer(
      logical_device, params, external_buffer->size,
      /*use_sparse_allocation=*/false,
      /*bind_host_memory=*/true, &handle));

  // Ask Vulkan what the implementation requires of the allocation(s) for the
  // buffer. We should in most cases always get the same kind of values but
  // alignment and valid memory types will differ for dense and sparse buffers.
  // We also can't trust the memory passed in is even usable.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_b, "vkGetBufferMemoryRequirements");
  VkMemoryRequirements requirements = {0};
  logical_device->syms()->vkGetBufferMemoryRequirements(*logical_device, handle,
                                                        &requirements);
  IREE_TRACE_ZONE_END(z_b);
  uint32_t memory_type_index = 0;
  status = iree_hal_vulkan_find_memory_type(
      &allocator->device_props, &allocator->memory_props, params,
      /*allowed_type_indices=*/
      (props.memoryTypeBits & requirements.memoryTypeBits), &memory_type_index);
  if (!iree_status_is_ok(status)) {
    logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                            logical_device->allocator());
    return status;
  }

  // Align the pointer and its size to the requirements of the memory type and
  // allocator. This may extend the base pointer down to a page boundary and the
  // size up to a page boundary but we'll subrange so that the buffer still
  // appears to have the same logical range.
  void* host_ptr = external_buffer->handle.host_allocation.ptr;
  VkDeviceSize allocation_size = (VkDeviceSize)external_buffer->size;
  VkDeviceSize memory_offset =
      iree_hal_vulkan_native_allocator_align_external_ptr(
          allocator, &requirements, &host_ptr, &allocation_size);

  // Allocate the device memory we'll attach the buffer to.
  VkMemoryAllocateInfo allocate_info = {};
  allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocate_info.pNext = NULL;
  allocate_info.allocationSize = allocation_size;
  allocate_info.memoryTypeIndex = memory_type_index;
  VkImportMemoryHostPointerInfoEXT import_host_ptr_info = {};
  import_host_ptr_info.sType =
      VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
  import_host_ptr_info.pNext = NULL;
  import_host_ptr_info.pHostPointer = host_ptr;
  import_host_ptr_info.handleType =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
  allocate_info.pNext = &import_host_ptr_info;
  VkDeviceMemory device_memory = VK_NULL_HANDLE;
  IREE_TRACE_ZONE_BEGIN_NAMED(z_c, "vkAllocateMemory");
  status = VK_RESULT_TO_STATUS(logical_device->syms()->vkAllocateMemory(
                                   *logical_device, &allocate_info,
                                   logical_device->allocator(), &device_memory),
                               "vkAllocateMemory");
  IREE_TRACE_ZONE_END(z_c);
  if (!iree_status_is_ok(status)) {
    logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                            logical_device->allocator());
    return status;
  }

  // Wrap the device memory allocation and buffer handle in our own buffer type.
  iree_hal_vulkan_native_buffer_release_callback_t internal_release_callback = {
      0};
  internal_release_callback.fn =
      iree_hal_vulkan_native_allocator_external_host_buffer_release;
  internal_release_callback.user_data = NULL;
  iree_hal_buffer_t* buffer = NULL;
  status = iree_hal_vulkan_native_buffer_wrap(
      (iree_hal_allocator_t*)allocator, params->type, params->access,
      params->usage, (iree_device_size_t)allocation_size,
      /*byte_offset=*/0,
      /*byte_length=*/external_buffer->size, logical_device, device_memory,
      handle, internal_release_callback, release_callback, &buffer);
  if (!iree_status_is_ok(status)) {
    logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                            logical_device->allocator());
    logical_device->syms()->vkFreeMemory(*logical_device, device_memory,
                                         logical_device->allocator());
    return status;
  }

  // Bind the memory to the buffer at a possibly non-zero offset if we had to
  // align the host pointer down to a page boundary.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_d, "vkBindBufferMemory");
  status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkBindBufferMemory(*logical_device, handle,
                                                 device_memory, memory_offset),
      "vkBindBufferMemory");
  IREE_TRACE_ZONE_END(z_d);

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static void iree_hal_vulkan_native_allocator_external_device_buffer_release(
    void* user_data, iree::hal::vulkan::VkDeviceHandle* logical_device,
    VkDeviceMemory device_memory, VkBuffer handle) {
  // NOTE: device memory is unowned but the buffer handle is ours to clean up.
  if (handle) {
    logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                            logical_device->allocator());
  }
}

static iree_status_t iree_hal_vulkan_native_allocator_import_device_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_vulkan_native_allocator_t* allocator =
      iree_hal_vulkan_native_allocator_cast(base_allocator);
  VkDeviceHandle* logical_device = allocator->logical_device;

  // A 'device allocation' is a VkDeviceMemory. We'll need to wrap a logical
  // VkBuffer around it for using within the HAL.
  VkDeviceMemory device_memory =
      (VkDeviceMemory)external_buffer->handle.device_allocation.ptr;
  if (IREE_UNLIKELY(!device_memory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no device memory handle provided");
  }

  // Create the logical buffer we can attach the memory to.
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_native_allocator_create_buffer(
      logical_device, params, external_buffer->size,
      /*use_sparse_allocation=*/false,
      /*bind_host_memory=*/false, &handle));

  // Bind the memory to the buffer.
  IREE_TRACE_ZONE_BEGIN_NAMED(z_a, "vkBindBufferMemory");
  iree_status_t status = VK_RESULT_TO_STATUS(
      logical_device->syms()->vkBindBufferMemory(
          *logical_device, handle, device_memory, /*memoryOffset=*/0),
      "vkBindBufferMemory");
  IREE_TRACE_ZONE_END(z_a);
  if (!iree_status_is_ok(status)) {
    logical_device->syms()->vkDestroyBuffer(*logical_device, handle,
                                            logical_device->allocator());
    return status;
  }

  // Wrap the device memory allocation and buffer handle in our own buffer type.
  iree_hal_vulkan_native_buffer_release_callback_t internal_release_callback = {
      0};
  internal_release_callback.fn =
      iree_hal_vulkan_native_allocator_external_device_buffer_release;
  internal_release_callback.user_data = NULL;
  return iree_hal_vulkan_native_buffer_wrap(
      (iree_hal_allocator_t*)allocator, params->type, params->access,
      params->usage, (iree_device_size_t)external_buffer->size,
      /*byte_offset=*/0,
      /*byte_length=*/external_buffer->size, logical_device, device_memory,
      handle, internal_release_callback, release_callback, out_buffer);
}

static iree_status_t iree_hal_vulkan_native_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION:
      return iree_hal_vulkan_native_allocator_import_host_buffer(
          base_allocator, params, external_buffer, release_callback,
          out_buffer);
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
      return iree_hal_vulkan_native_allocator_import_device_buffer(
          base_allocator, params, external_buffer, release_callback,
          out_buffer);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external buffer type import not implemented");
  }
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
