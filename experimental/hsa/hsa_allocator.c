// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/hsa_allocator.h"

#include <stddef.h>

#include "experimental/hsa/dynamic_symbols.h"
#include "experimental/hsa/hsa_buffer.h"
#include "experimental/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_hsa_allocator_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  hsa_agent_t hsa_agent;

  hsa_agent_t cpu_agent;
  hsa_amd_memory_pool_t cpu_pool;

  // One memory pool and region for now
  hsa_amd_memory_pool_t buffers_pool;
  hsa_region_t kernel_argument_pool;

  const iree_hal_hsa_dynamic_symbols_t* symbols;

  iree_allocator_t host_allocator;

  // Whether the GPU and CPU can concurrently access HSA managed data in a
  // coherent way. We would need to explicitly perform flushing and invalidation
  // between GPU and CPU if not.
  bool supports_concurrent_managed_access;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_hsa_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_hsa_allocator_vtable;

static iree_hal_hsa_allocator_t* iree_hal_hsa_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_allocator_vtable);
  return (iree_hal_hsa_allocator_t*)base_value;
}

static hsa_status_t get_kernarg_memory_region(hsa_region_t region,
                                              void* allocator_untyped) {
  iree_hal_hsa_allocator_t* allocator =
      (iree_hal_hsa_allocator_t*)(allocator_untyped);

  hsa_region_segment_t segment;
  allocator->symbols->hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT,
                                          &segment);
  if (HSA_REGION_SEGMENT_GLOBAL != segment) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  allocator->symbols->hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS,
                                          &flags);
  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    hsa_region_t* ret = (hsa_region_t*)(&(allocator->kernel_argument_pool));
    *ret = region;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

static hsa_status_t get_fine_grained_memory_pool(hsa_amd_memory_pool_t pool,
                                                 void* allocator_untyped) {
  iree_hal_hsa_allocator_t* allocator =
      (iree_hal_hsa_allocator_t*)(allocator_untyped);

  hsa_amd_segment_t segment;
  hsa_status_t status = allocator->symbols->hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t flags;
  status = allocator->symbols->hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  bool is_fine_grained =
      (flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED));
  bool is_kernel_arg_region =
      (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);

  if (is_fine_grained && !is_kernel_arg_region) {
    allocator->buffers_pool = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t iterate_find_cpu_agent_callback(hsa_agent_t agent,
                                                    void* base_allocator) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  hsa_device_type_t type;
  hsa_status_t status = allocator->symbols->hsa_agent_get_info(
      agent, HSA_AGENT_INFO_DEVICE, &type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (type == HSA_DEVICE_TYPE_CPU) {
    allocator->cpu_agent = agent;
  }
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t iterate_find_cpu_agent_pool_callback(
    hsa_amd_memory_pool_t pool, void* base_allocator) {
  iree_hal_hsa_allocator_t* allocator =
      (iree_hal_hsa_allocator_t*)(base_allocator);

  hsa_amd_segment_t segment;
  hsa_status_t status = allocator->symbols->hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  uint32_t flags;
  status = allocator->symbols->hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  bool is_fine_grained =
      (flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED |
                HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED));
  bool is_kernel_arg_region =
      (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);

  if (is_fine_grained && !is_kernel_arg_region) {
    allocator->cpu_pool = pool;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

iree_status_t iree_hal_hsa_allocator_create(
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols, hsa_agent_t agent,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  // To support device-local + host-visible memory we need concurrent managed
  // access indicating that the host and devices can concurrently access the
  // device memory. If we don't have this feature then we fall back to forcing
  // all device-local + host-visible memory into host-local + device-visible
  // page-locked memory. The compiler tries to avoid this for high-traffic
  // buffers except for readback staging buffers.
  int supports_concurrent_managed_access = 1;

  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, supports_concurrent_managed_access
              ? "has CONCURRENT_MANAGED_ACCESS"
              : "no CONCURRENT_MANAGED_ACCESS (expect slow accesses on "
                "device-local + host-visible memory)");

  iree_hal_hsa_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_hsa_allocator_vtable,
                               &allocator->resource);
  allocator->hsa_agent = agent;
  allocator->symbols = hsa_symbols;
  allocator->host_allocator = host_allocator;
  allocator->supports_concurrent_managed_access =
      supports_concurrent_managed_access != 0;

  hsa_symbols->hsa_agent_iterate_regions(agent, get_kernarg_memory_region,
                                         allocator);
  hsa_symbols->hsa_amd_agent_iterate_memory_pools(
      agent, get_fine_grained_memory_pool, allocator);

  hsa_symbols->hsa_iterate_agents(&iterate_find_cpu_agent_callback,
                                  (void*)allocator);
  hsa_symbols->hsa_amd_agent_iterate_memory_pools(
      allocator->cpu_agent, &iterate_find_cpu_agent_pool_callback,
      (void*)allocator);

  *out_allocator = (iree_hal_allocator_t*)allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hsa_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_hsa_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hsa_allocator_t* allocator =
      (iree_hal_hsa_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_hsa_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  return iree_ok_status();
}

static void iree_hal_hsa_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_hsa_allocator_t* allocator =
        iree_hal_hsa_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_status_t iree_hal_hsa_allocator_query_memory_heaps(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_host_size_t capacity,
    iree_hal_allocator_memory_heap_t* IREE_RESTRICT heaps,
    iree_host_size_t* IREE_RESTRICT out_count) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

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
iree_hal_hsa_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t* IREE_RESTRICT allocation_size) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers are importable in HSA under most cases, though performance may
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

static void iree_hal_hsa_buffer_free(
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_hsa_buffer_type_t buffer_type, hsa_device_pointer_t device_ptr,
    void* host_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  switch (buffer_type) {
    case IREE_HAL_HSA_BUFFER_TYPE_DEVICE: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hsa_amd_memory_pool_free");
      IREE_HSA_IGNORE_ERROR(hsa_symbols, hsa_amd_memory_pool_free(device_ptr));
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_HOST: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hsa_amd_memory_pool_free");
      IREE_HSA_IGNORE_ERROR(hsa_symbols, hsa_amd_memory_pool_free(device_ptr));
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_HOST_REGISTERED: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "host unregister");
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_ASYNC: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(ignored; async)");
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(ignored; external)");
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_KERNEL_ARG: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(ignored; external)");
      break;
    }
  }
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hsa_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hsa_allocator_query_buffer_compatibility(
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
  iree_hal_hsa_buffer_type_t buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
  void* host_ptr = NULL;
  hsa_device_pointer_t device_ptr = NULL;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_hsa_buffer_allocate");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_size);

  // TODO(muhaawad): Not sure if this is the right way to do kernel arguments
  // allocations
  if (iree_all_bits_set(compat_params.usage,
                        IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE |
                            IREE_HAL_BUFFER_USAGE_TRANSFER) &&
      iree_all_bits_set(
          compat_params.access,
          IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE) &&
      iree_all_bits_set(compat_params.type,
                        IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                            IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    // Kernel arguments
    IREE_HSA_RETURN_IF_ERROR(
        allocator->symbols,
        hsa_memory_allocate(allocator->kernel_argument_pool, allocation_size,
                            &host_ptr),
        "hsa_memory_allocate");
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_KERNEL_ARG;
    device_ptr = host_ptr;
  } else if (iree_all_bits_set(compat_params.type,
                               IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local case.
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
    if (iree_all_bits_set(compat_params.type,
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      status = IREE_HSA_RESULT_TO_STATUS(
          allocator->symbols,
          hsa_amd_memory_pool_allocate(allocator->buffers_pool, allocation_size,
                                       /*flags=*/0, &device_ptr));
      host_ptr = (void*)device_ptr;

    } else {
      // Device only.
      buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;

      status = IREE_HSA_RESULT_TO_STATUS(
          allocator->symbols,
          hsa_amd_memory_pool_allocate(allocator->buffers_pool, allocation_size,
                                       /*flags=*/0, &device_ptr));
    }
  } else {
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_HOST;
    status = IREE_HSA_RESULT_TO_STATUS(
        allocator->symbols,
        hsa_amd_memory_pool_allocate(allocator->buffers_pool, allocation_size,
                                     /*flags=*/0, &host_ptr));
    device_ptr = host_ptr;
  }
  IREE_TRACE_ZONE_END(z0);

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_buffer_wrap(
        base_allocator, compat_params.type, compat_params.access,
        compat_params.usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, buffer_type, device_ptr, host_ptr,
        iree_hal_buffer_release_callback_null(),
        iree_hal_allocator_host_allocator(base_allocator), &buffer);
  }

  if (iree_status_is_ok(status)) {
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_HSA_ALLOCATOR_ID,
                           (void*)iree_hal_hsa_buffer_device_pointer(buffer),
                           allocation_size);
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, compat_params.type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (!buffer && (device_ptr || host_ptr)) {
      iree_hal_hsa_buffer_free(allocator->symbols, buffer_type, device_ptr,
                               host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static void iree_hal_hsa_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  const iree_hal_hsa_buffer_type_t buffer_type =
      iree_hal_hsa_buffer_type(base_buffer);

  iree_hal_hsa_buffer_free(allocator->symbols, buffer_type,
                           iree_hal_hsa_buffer_device_pointer(base_buffer),
                           iree_hal_hsa_buffer_host_pointer(base_buffer));

  switch (buffer_type) {
    case IREE_HAL_HSA_BUFFER_TYPE_DEVICE:
    case IREE_HAL_HSA_BUFFER_TYPE_HOST: {
      IREE_TRACE_FREE_NAMED(
          IREE_HAL_HSA_ALLOCATOR_ID,
          (void*)iree_hal_hsa_buffer_device_pointer(base_buffer));
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

static iree_status_t iree_hal_hsa_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);
  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_device_size_t allocation_size = external_buffer->size;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hsa_allocator_query_buffer_compatibility(
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
  iree_hal_hsa_buffer_type_t buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
  void* host_ptr = NULL;
  hsa_device_pointer_t device_ptr = NULL;

  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION: {
      uint32_t flags = 0;
      int num_agents = 1;
      status = IREE_HSA_RESULT_TO_STATUS(
          allocator->symbols,
          hsa_amd_memory_lock_to_pool(host_ptr, external_buffer->size,
                                      &allocator->cpu_agent, num_agents,
                                      allocator->cpu_pool, flags, device_ptr),
          "hsa_amd_memory_lock_to_pool");

      break;
    }
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION: {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "not yet implemented");
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
    status = iree_hal_hsa_buffer_wrap(
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
      iree_hal_hsa_buffer_free(allocator->symbols, buffer_type, device_ptr,
                               host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static iree_status_t iree_hal_hsa_allocator_export_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT buffer,
    iree_hal_external_buffer_type_t requested_type,
    iree_hal_external_buffer_flags_t requested_flags,
    iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer) {
  iree_hal_hsa_buffer_type_t buffer_type = iree_hal_hsa_buffer_type(buffer);

  switch (requested_type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION:
      switch (buffer_type) {
        case IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL:
          out_external_buffer->flags = requested_flags;
          out_external_buffer->type = requested_type;
          out_external_buffer->handle.device_allocation.ptr =
              ((uint64_t)(uintptr_t)iree_hal_hsa_buffer_device_pointer(buffer));
          out_external_buffer->size = iree_hal_buffer_allocation_size(buffer);
          return iree_ok_status();

        default:
          return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                  "HSA buffer type is not supported for "
                                  "export as an external device allocation");
      }

    default:
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "external buffer type not supported");
  }
}

static const iree_hal_allocator_vtable_t iree_hal_hsa_allocator_vtable = {
    .destroy = iree_hal_hsa_allocator_destroy,
    .host_allocator = iree_hal_hsa_allocator_host_allocator,
    .trim = iree_hal_hsa_allocator_trim,
    .query_statistics = iree_hal_hsa_allocator_query_statistics,
    .query_memory_heaps = iree_hal_hsa_allocator_query_memory_heaps,
    .query_buffer_compatibility =
        iree_hal_hsa_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_hsa_allocator_allocate_buffer,
    .deallocate_buffer = iree_hal_hsa_allocator_deallocate_buffer,
    .import_buffer = iree_hal_hsa_allocator_import_buffer,
    .export_buffer = iree_hal_hsa_allocator_export_buffer,
};
