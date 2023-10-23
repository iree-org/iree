// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/direct_allocator.h"

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/metal_buffer.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_METAL_ALLOCATOR_ID = "Metal";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_metal_allocator_t {
  // Abstract resource used for injecting reference counting and vtable; must be at offset 0.
  iree_hal_resource_t resource;

  // The device that this allocator is attached to.
  id<MTLDevice> device;
  // The command queue that we can use to issue commands to make buffer contents visible to CPU.
#if defined(IREE_PLATFORM_MACOS)
  id<MTLCommandQueue> queue;
#endif  // IREE_PLATFORM_MACOS

  bool is_unified_memory;
  iree_hal_metal_resource_hazard_tracking_mode_t resource_tracking_mode;

  iree_allocator_t host_allocator;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_metal_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_metal_allocator_vtable;

static iree_hal_metal_allocator_t* iree_hal_metal_allocator_cast(iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_allocator_vtable);
  return (iree_hal_metal_allocator_t*)base_value;
}

static const iree_hal_metal_allocator_t* iree_hal_metal_allocator_const_cast(
    const iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_allocator_vtable);
  return (const iree_hal_metal_allocator_t*)base_value;
}

iree_status_t iree_hal_metal_allocator_create(
    id<MTLDevice> device,
#if defined(IREE_PLATFORM_MACOS)
    id<MTLCommandQueue> queue,
#endif  // IREE_PLATFORM_MACOS
    iree_hal_metal_resource_hazard_tracking_mode_t resource_tracking_mode,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_metal_allocator_t* allocator = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*allocator), (void**)&allocator);

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_metal_allocator_vtable, &allocator->resource);
    allocator->device = [device retain];  // +1
    allocator->is_unified_memory = [device hasUnifiedMemory];
    allocator->resource_tracking_mode = resource_tracking_mode;
    allocator->host_allocator = host_allocator;

    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_allocator_destroy(iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  [allocator->device release];  // -1
  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_metal_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_metal_allocator_t* allocator = (iree_hal_metal_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

#if defined(IREE_PLATFORM_MACOS)
id<MTLCommandQueue> iree_hal_metal_allocator_command_queue(
    const iree_hal_allocator_t* base_allocator) {
  const iree_hal_metal_allocator_t* allocator = (const iree_hal_metal_allocator_t*)base_allocator;
  return allocator->queue;
}
#endif  // IREE_PLATFORM_MACOS

static iree_status_t iree_hal_metal_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_metal_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t iree_hal_metal_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility = IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  if (iree_all_bits_set(params->type,
                        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    // On iOS, we don't have device local + host visible memory. But given the unified memory
    // architecture, it's fine to just request host local + device visible memory.
    // On macOS, for unified memory architecture, it's similar to iOS. Otherwise, we can have
    // device local + host visible memory backed by Managed storage mode.
    iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);
    if (allocator->is_unified_memory) {
      params->type &= ~(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
      params->type |= IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
      // We have suggested other configurations than the original request. Mark accordingly.
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
    }
  }

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0. The application is unlikely
  // to do anything when requesting a 0-byte buffer; but it can happen in real world use cases. So
  // we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  // Align allocation sizes to 4 bytes so shaders operating on 32 bit types can act safely even on
  // buffer ranges that are not naturally aligned.
  *allocation_size = iree_host_align(*allocation_size, 4);

  return compatibility;
}

// Returns the corresponding Metal resource options controlling storage modes, CPU caching modes,
// and hazard tracking modes for the given IREE HAL memory |type|.
static MTLResourceOptions iree_hal_metal_select_resource_options(
    iree_hal_memory_type_t type, bool is_unified_memory,
    iree_hal_metal_resource_hazard_tracking_mode_t resource_tracking_mode) {
  MTLResourceOptions options;

  // Select a storage mode. There are four MTLStorageMode:
  // * Shared: The resource is stored in system memory and is accessible to both the CPU and
  //   the GPU.
  // * Managed: The CPU and GPU may maintain separate copies of the resource, and any changes
  //   must be explicitly synchronized.
  // * Private: The resource can be accessed only by the GPU.
  // * Memoryless: The resource's contents can be accessed only by the GPU and only exist
  //   temporarily during a render pass.
  // macOS has all of the above; MTLStorageModeManaged is not available on iOS.
  //
  // The IREE HAL is modeled after explicit APIs like Vulkan. For buffers visible to both the host
  // and the device, we would like to opt in with the explicit version (MTLStorageManaged) when
  // possible because it should be more performant.
  if (iree_all_bits_set(type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    if (iree_all_bits_set(type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      // Device local + host visible.
      // iree_hal_metal_allocator_query_buffer_compatibility guarantees that we only fall into this
      // case for macOS devices with non-uniform memory.
#if defined(IREE_PLATFORM_MACOS)
      IREE_ASSERT(!is_unified_memory);
      options = MTLResourceStorageModeManaged;
#else
      options = MTLResourceStorageModeShared;
#endif  // IREE_PLATFORM_MACOS
    } else {
      // Device local + host invisible.
      options = MTLResourceStorageModePrivate;
    }
  } else {
    if (iree_all_bits_set(type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
      // Host local + device visible.
      options = MTLResourceStorageModeShared;
    } else {
      // Host local + device invisible.
      options = MTLResourceStorageModeShared;
    }
  }

  // Select a CPU cache mode.
  if (iree_all_bits_set(type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
    // The default CPU cache mode for the resource, which guarantees that read and write operations
    // are executed in the expected order.
    options |= MTLResourceCPUCacheModeDefaultCache;
  } else {
    // A write-combined CPU cache mode that is optimized for resources that the CPU writes into, but
    // never reads.
    options |= MTLResourceCPUCacheModeWriteCombined;
  }

  options |= resource_tracking_mode == IREE_HAL_METAL_RESOURCE_HAZARD_TRACKING_MODE_TRACKED
                 ? MTLResourceHazardTrackingModeTracked
                 : MTLResourceHazardTrackingModeUntracked;
  return options;
}

static iree_status_t iree_hal_metal_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params, iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_size);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_metal_allocator_query_buffer_compatibility(base_allocator, &compat_params,
                                                          &allocation_size);
  if (!iree_all_bits_set(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str = iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str = iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator cannot allocate a buffer with the given parameters; "
                            "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
                            (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
                            usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  MTLResourceOptions options = iree_hal_metal_select_resource_options(
      compat_params.type, allocator->is_unified_memory, allocator->resource_tracking_mode);

  id<MTLBuffer> metal_buffer = [allocator->device newBufferWithLength:allocation_size
                                                              options:options];  // +1

  if (!metal_buffer) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "unable to allocate buffer");
  }
  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_metal_buffer_wrap(
#if defined(IREE_PLATFORM_MACOS)
      allocator->queue,
#endif  // IREE_PLATFORM_MACOS
      metal_buffer, base_allocator, compat_params.type, compat_params.access, compat_params.usage,
      allocation_size, /*byte_offset=*/0, /*byte_length=*/allocation_size,
      iree_hal_buffer_release_callback_null(), &buffer);  // +1

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_METAL_ALLOCATOR_ID, (void*)iree_hal_metal_buffer_handle(buffer),
                           allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (buffer) iree_hal_buffer_release(buffer);
  }

  [metal_buffer release];  // -1

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);

  IREE_TRACE_FREE_NAMED(IREE_HAL_METAL_ALLOCATOR_ID,
                        (void*)iree_hal_metal_buffer_handle(base_buffer));
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);  // -1
}

static iree_status_t iree_hal_metal_allocator_import_host_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);

  // MTLResourceOptions options = iree_hal_metal_select_resource_options(
  //     params->type, allocator->is_unified_memory, allocator->resource_tracking_mode);
  MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache | MTLResourceStorageModeShared |
                               MTLResourceHazardTrackingModeUntracked;

  id<MTLBuffer> metal_buffer =
      [allocator->device newBufferWithBytesNoCopy:external_buffer->handle.host_allocation.ptr
                                           length:(NSUInteger)external_buffer->size
                                          options:options
                                      deallocator:nil];  // +1
  if (!metal_buffer) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "unable to allocate buffer");
  }

  return iree_hal_metal_buffer_wrap(
#if defined(IREE_PLATFORM_MACOS)
      allocator->queue,
#endif  // IREE_PLATFORM_MACOS
      metal_buffer, base_allocator, params->type, params->access, params->usage,
      external_buffer->size, /*byte_offset=*/0, /*byte_length=*/external_buffer->size,
      release_callback, out_buffer);  // +1
}

static iree_status_t iree_hal_metal_allocator_import_device_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_metal_allocator_t* allocator = iree_hal_metal_allocator_cast(base_allocator);

  // Device allocation is an unowned MTLBuffer; we need to retain it to keep it live.
  id<MTLBuffer> metal_buffer =
      (__bridge id<MTLBuffer>)external_buffer->handle.device_allocation.ptr;
  if (!metal_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no handle provided");
  }

  // Wrap the externally-provided buffer in a HAL buffer handle that will retain the MTLBuffer until
  // it has been released.
  return iree_hal_metal_buffer_wrap(
#if defined(IREE_PLATFORM_MACOS)
      allocator->queue,
#endif  // IREE_PLATFORM_MACOS
      metal_buffer, base_allocator, params->type, params->access, params->usage,
      external_buffer->size, /*byte_offset=*/0, /*byte_length=*/external_buffer->size,
      release_callback, out_buffer);  // +1
}

static iree_status_t iree_hal_metal_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION:
      return iree_hal_metal_allocator_import_host_buffer(base_allocator, params, external_buffer,
                                                         release_callback, out_buffer);
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
      return iree_hal_metal_allocator_import_device_buffer(base_allocator, params, external_buffer,
                                                           release_callback, out_buffer);
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external buffer type import not implemented");
  }
}

static iree_status_t iree_hal_metal_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator, iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE, "unsupported exporting to external buffer");
}

static const iree_hal_allocator_vtable_t iree_hal_metal_allocator_vtable = {
    .destroy = iree_hal_metal_allocator_destroy,
    .host_allocator = iree_hal_metal_allocator_host_allocator,
    .trim = iree_hal_metal_allocator_trim,
    .query_statistics = iree_hal_metal_allocator_query_statistics,
    .query_buffer_compatibility = iree_hal_metal_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_metal_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_metal_allocator_deallocate_buffer,
    .import_buffer = iree_hal_metal_allocator_import_buffer,
    .export_buffer = iree_hal_metal_allocator_export_buffer,
};
