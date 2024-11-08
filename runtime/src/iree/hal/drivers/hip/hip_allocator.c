// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_allocator.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/status_util.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_HIP_ALLOCATOR_ID = "HIP unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_hip_allocator_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The device that this allocator allocates memory from.
  hipDevice_t device;

  // The HIP stream that allocations should be used in.
  hipStream_t stream;

  // NOTE: optional depending on device support.
  iree_hal_hip_memory_pools_t* pools;

  const iree_hal_hip_dynamic_symbols_t* symbols;

  iree_allocator_t host_allocator;

  // Whether the GPU and CPU can concurrently access HIP managed data in a
  // coherent way. We would need to explicitly perform flushing and invalidation
  // between GPU and CPU if not.
  bool supports_concurrent_managed_access;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_hip_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_hip_allocator_vtable;

static iree_hal_hip_allocator_t* iree_hal_hip_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_allocator_vtable);
  return (iree_hal_hip_allocator_t*)base_value;
}

iree_status_t iree_hal_hip_allocator_create(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols, hipDevice_t device,
    hipStream_t stream, iree_hal_hip_memory_pools_t* pools,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(hip_symbols);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  // To support device-local + host-visible memory we need concurrent managed
  // access indicating that the host and devices can concurrently access the
  // device memory. If we don't have this feature then we fall back to forcing
  // all device-local + host-visible memory into host-local + device-visible
  // page-locked memory. The compiler tries to avoid this for high-traffic
  // buffers except for readback staging buffers.
  int supports_concurrent_managed_access = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HIP_RESULT_TO_STATUS(
              hip_symbols,
              hipDeviceGetAttribute(&supports_concurrent_managed_access,
                                    hipDeviceAttributeConcurrentManagedAccess,
                                    device),
              "hipDeviceGetAttribute"));

  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, supports_concurrent_managed_access
              ? "has CONCURRENT_MANAGED_ACCESS"
              : "no CONCURRENT_MANAGED_ACCESS (expect slow accesses on "
                "device-local + host-visible memory)");

  iree_hal_hip_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_hip_allocator_vtable,
                               &allocator->resource);
  allocator->device = device;
  allocator->stream = stream;
  allocator->pools = pools;
  allocator->symbols = hip_symbols;
  allocator->host_allocator = host_allocator;
  allocator->supports_concurrent_managed_access =
      supports_concurrent_managed_access != 0;
  *out_allocator = (iree_hal_allocator_t*)allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_hip_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hip_allocator_t* allocator =
      (iree_hal_hip_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_hip_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_hip_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_hip_allocator_t* allocator =
        iree_hal_hip_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
    if (allocator->pools) {
      iree_hal_hip_memory_pools_merge_statistics(allocator->pools,
                                                 out_statistics);
    }
  });
}

static iree_status_t iree_hal_hip_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);

  iree_host_size_t count = 3;
  if (allocator->supports_concurrent_managed_access) {
    ++count;  // device-local | host-visible
  }
  if (out_count) *out_count = count;
  if (capacity < count) {
    // NOTE: lightweight as this is hit in normal pre-sizing usage.
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  // Don't think there's a query for these.
  // Max allocation size may be much smaller in certain memory types such as
  // page-locked memory and it'd be good to enforce that.
  const iree_device_size_t max_allocation_size = ~(iree_device_size_t)0;
  const iree_device_size_t min_alignment = 64;

  int i = 0;

  // Device-local memory (dispatch resources):
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  if (allocator->supports_concurrent_managed_access) {
    // Device-local managed memory with host mapping support:
    heaps[i++] = (iree_hal_allocator_memory_heap_t){
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
                IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
        .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                         IREE_HAL_BUFFER_USAGE_DISPATCH |
                         IREE_HAL_BUFFER_USAGE_MAPPING,
        .max_allocation_size = max_allocation_size,
        .min_alignment = min_alignment,
    };
  }

  // Write-combined page-locked host-local memory (upload):
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_DISPATCH |
                       IREE_HAL_BUFFER_USAGE_MAPPING,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  // Cached page-locked host-local memory (download):
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE |
              IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
              IREE_HAL_MEMORY_TYPE_HOST_COHERENT |
              IREE_HAL_MEMORY_TYPE_HOST_CACHED,
      .allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                       IREE_HAL_BUFFER_USAGE_DISPATCH |
                       IREE_HAL_BUFFER_USAGE_MAPPING,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  IREE_ASSERT(i == count);
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
iree_hal_hip_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers are importable in HIP under most cases, though performance may
  // vary wildly. We don't fully verify that the buffer parameters are
  // self-consistent and just look at whether we can get a device pointer.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE;
  }

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_any_bit_set(params->usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_any_bit_set(params->usage,
                         IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    // Device local and host visible in general is much more slower than device
    // only for discrete GPUs. So mark as so accordingly.
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
    // If concurrent managed access is not supported then make device-local +
    // host-visible allocations fall back to host-local + device-visible
    // page-locked memory. This will be significantly slower for the device to
    // access but the compiler only uses this type for readback staging buffers
    // and it's better to function than function fast.
    if (!allocator->supports_concurrent_managed_access) {
      params->type &= ~(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
      params->type |=
          IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    }
  }

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (*allocation_size == 0) *allocation_size = 4;

  return compatibility;
}

static void iree_hal_hip_buffer_free(
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    iree_hal_hip_buffer_type_t buffer_type, hipDeviceptr_t device_ptr,
    void* host_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  switch (buffer_type) {
    case IREE_HAL_HIP_BUFFER_TYPE_DEVICE: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hipFree");
      IREE_HIP_IGNORE_ERROR(hip_symbols, hipFree(device_ptr));
      break;
    }
    case IREE_HAL_HIP_BUFFER_TYPE_HOST: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hipHostFree");
      IREE_HIP_IGNORE_ERROR(hip_symbols, hipHostFree(host_ptr));
      break;
    }
    case IREE_HAL_HIP_BUFFER_TYPE_HOST_REGISTERED: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hipHostUnregister");
      IREE_HIP_IGNORE_ERROR(hip_symbols, hipHostUnregister(host_ptr));
      break;
    }
    case IREE_HAL_HIP_BUFFER_TYPE_ASYNC: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(ignored; async)");
      break;
    }
    case IREE_HAL_HIP_BUFFER_TYPE_EXTERNAL: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(ignored; external)");
      break;
    }
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hip_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hip_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  iree_status_t status = iree_ok_status();
  iree_hal_hip_buffer_type_t buffer_type = IREE_HAL_HIP_BUFFER_TYPE_DEVICE;
  void* host_ptr = NULL;
  hipDeviceptr_t device_ptr = NULL;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_hip_buffer_allocate");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_size);
  if (iree_all_bits_set(compat_params.type,
                        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local case.
    if (iree_all_bits_set(compat_params.type,
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      // Device local and host visible.
      buffer_type = IREE_HAL_HIP_BUFFER_TYPE_DEVICE;
      status = IREE_HIP_RESULT_TO_STATUS(
          allocator->symbols,
          hipMallocManaged(&device_ptr, allocation_size, hipMemAttachGlobal));
      if (iree_status_is_ok(status) &&
          allocator->supports_concurrent_managed_access) {
        // Prefetch the buffer on the GPU device.
        status = IREE_HIP_RESULT_TO_STATUS(
            allocator->symbols,
            hipMemPrefetchAsync(device_ptr, allocation_size, allocator->device,
                                allocator->stream));
      }
      host_ptr = (void*)device_ptr;
    } else {
      // Device only.
      buffer_type = IREE_HAL_HIP_BUFFER_TYPE_DEVICE;
      status = IREE_HIP_RESULT_TO_STATUS(
          allocator->symbols, hipMalloc(&device_ptr, allocation_size));
    }
  } else {
    // Host local case.
    buffer_type = IREE_HAL_HIP_BUFFER_TYPE_HOST;
    unsigned int flags = hipHostMallocMapped;
    if (!iree_all_bits_set(compat_params.type,
                           IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
      flags |= hipHostMallocWriteCombined;
    }
    status = IREE_HIP_RESULT_TO_STATUS(
        allocator->symbols, hipHostMalloc(&host_ptr, allocation_size, flags));
    if (iree_status_is_ok(status)) {
      status = IREE_HIP_RESULT_TO_STATUS(
          allocator->symbols,
          hipHostGetDevicePointer(&device_ptr, host_ptr, /*flags=*/0));
    }
  }
  IREE_TRACE_ZONE_END(z0);

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_buffer_wrap(
        base_allocator, compat_params.type, compat_params.access,
        compat_params.usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, buffer_type, device_ptr, host_ptr,
        iree_hal_buffer_release_callback_null(),
        iree_hal_allocator_host_allocator(base_allocator), &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_HIP_ALLOCATOR_ID,
                           (void*)iree_hal_hip_buffer_device_pointer(buffer),
                           allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (!buffer && (device_ptr || host_ptr)) {
      iree_hal_hip_buffer_free(allocator->symbols, buffer_type, device_ptr,
                               host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static void iree_hal_hip_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);

  const iree_hal_hip_buffer_type_t buffer_type =
      iree_hal_hip_buffer_type(base_buffer);

  iree_hal_hip_buffer_free(allocator->symbols, buffer_type,
                           iree_hal_hip_buffer_device_pointer(base_buffer),
                           iree_hal_hip_buffer_host_pointer(base_buffer));

  switch (buffer_type) {
    case IREE_HAL_HIP_BUFFER_TYPE_DEVICE:
    case IREE_HAL_HIP_BUFFER_TYPE_HOST: {
      IREE_TRACE_FREE_NAMED(
          IREE_HAL_HIP_ALLOCATOR_ID,
          (void*)iree_hal_hip_buffer_device_pointer(base_buffer));
      IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
          &allocator->statistics, iree_hal_buffer_memory_type(base_buffer),
          iree_hal_buffer_allocation_size(base_buffer)));
      break;
    }
    default:
      // Buffer type not tracked.
      break;
  }

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t iree_hal_hip_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_device_size_t allocation_size = external_buffer->size;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hip_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_IMPORTABLE)) {
#if IREE_STATUS_MODE
    iree_bitfield_string_temp_t temp0, temp1, temp2;
    iree_string_view_t memory_type_str =
        iree_hal_memory_type_format(params->type, &temp0);
    iree_string_view_t usage_str =
        iree_hal_buffer_usage_format(params->usage, &temp1);
    iree_string_view_t compatibility_str =
        iree_hal_buffer_compatibility_format(compatibility, &temp2);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters; "
        "memory_type=%.*s, usage=%.*s, compatibility=%.*s",
        (int)memory_type_str.size, memory_type_str.data, (int)usage_str.size,
        usage_str.data, (int)compatibility_str.size, compatibility_str.data);
#else
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters");
#endif  // IREE_STATUS_MODE
  }

  iree_status_t status = iree_ok_status();
  iree_hal_hip_buffer_type_t buffer_type = IREE_HAL_HIP_BUFFER_TYPE_DEVICE;
  void* host_ptr = NULL;
  hipDeviceptr_t device_ptr = NULL;

  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION: {
      if (iree_all_bits_set(compat_params.type,
                            IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "unable to register host allocations as device-local memory");
      }
      buffer_type = IREE_HAL_HIP_BUFFER_TYPE_HOST_REGISTERED;
      host_ptr = external_buffer->handle.host_allocation.ptr;
      uint32_t register_flags = hipHostRegisterMapped;
      status = IREE_HIP_RESULT_TO_STATUS(
          allocator->symbols,
          hipHostRegister(host_ptr, external_buffer->size, register_flags),
          "hipHostRegister");
      if (iree_status_is_ok(status)) {
        status = IREE_HIP_RESULT_TO_STATUS(
            allocator->symbols,
            hipHostGetDevicePointer(&device_ptr, host_ptr, 0),
            "hipHostGetDevicePointer");
      }
      break;
    }
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION: {
      buffer_type = IREE_HAL_HIP_BUFFER_TYPE_EXTERNAL;
      device_ptr =
          (hipDeviceptr_t)external_buffer->handle.device_allocation.ptr;
      break;
    }
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_FD:
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_OPAQUE_WIN32:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "handle-based imports not yet implemented");
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external buffer type not supported");
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_buffer_wrap(
        base_allocator, compat_params.type, compat_params.access,
        compat_params.usage, external_buffer->size,
        /*byte_offset=*/0,
        /*byte_length=*/external_buffer->size, buffer_type, device_ptr,
        host_ptr, release_callback,
        iree_hal_allocator_host_allocator(base_allocator), &buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    if (!buffer && (device_ptr || host_ptr)) {
      iree_hal_hip_buffer_free(allocator->symbols, buffer_type, device_ptr,
                               host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static iree_status_t iree_hal_hip_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  iree_hal_hip_buffer_type_t buffer_type = iree_hal_hip_buffer_type(buffer);

  switch (requested_type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
      switch (buffer_type) {
        case IREE_HAL_HIP_BUFFER_TYPE_DEVICE:
        case IREE_HAL_HIP_BUFFER_TYPE_EXTERNAL:
          out_external_buffer->flags = requested_flags;
          out_external_buffer->type = requested_type;
          out_external_buffer->handle.device_allocation.ptr =
              ((uint64_t)(uintptr_t)iree_hal_hip_buffer_device_pointer(buffer));
          out_external_buffer->size = iree_hal_buffer_allocation_size(buffer);
          return iree_ok_status();

        default:
          return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "HIP buffer type is not supported for "
                                  "export as an external device allocation");
      }

    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "external buffer type not supported");
  }
}

static const iree_hal_allocator_vtable_t iree_hal_hip_allocator_vtable = {
    .destroy = iree_hal_hip_allocator_destroy,
    .host_allocator = iree_hal_hip_allocator_host_allocator,
    .trim = iree_hal_hip_allocator_trim,
    .query_statistics = iree_hal_hip_allocator_query_statistics,
    .query_memory_heaps = iree_hal_hip_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_hip_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_hip_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_hip_allocator_deallocate_buffer,
    .import_buffer = iree_hal_hip_allocator_import_buffer,
    .export_buffer = iree_hal_hip_allocator_export_buffer,
};
