// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hsa/hsa_allocator.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hsa/dynamic_symbols.h"
#include "iree/hal/drivers/hsa/hsa_buffer.h"
#include "iree/hal/drivers/hsa/per_device_information.h"
#include "iree/hal/drivers/hsa/status_util.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_HSA_ALLOCATOR_ID = "HSA";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

typedef struct iree_hal_hsa_allocator_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // Parent device that this allocator is associated with. Unowned.
  iree_hal_device_t* parent_device;

  iree_hal_hsa_device_topology_t topology;

  const iree_hal_hsa_dynamic_symbols_t* symbols;

  iree_allocator_t host_allocator;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_hsa_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_hsa_allocator_vtable;

static iree_hal_hsa_allocator_t* iree_hal_hsa_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_allocator_vtable);
  return (iree_hal_hsa_allocator_t*)base_value;
}

iree_status_t iree_hal_hsa_allocator_create(
    iree_hal_device_t* parent_device,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_hsa_device_topology_t topology,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(parent_device);
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;

  if (topology.count < 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one device must be specified");
  }

  iree_hal_hsa_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*allocator),
                                (void**)&allocator));

  iree_hal_resource_initialize(&iree_hal_hsa_allocator_vtable,
                               &allocator->resource);
  allocator->parent_device = parent_device;
  allocator->symbols = hsa_symbols;
  allocator->host_allocator = host_allocator;
  allocator->topology = topology;

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

bool iree_hal_hsa_allocator_isa(iree_hal_allocator_t* base_value) {
  return iree_hal_resource_is(base_value, &iree_hal_hsa_allocator_vtable);
}

static iree_allocator_t iree_hal_hsa_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hsa_allocator_t* allocator =
      (iree_hal_hsa_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_hsa_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  // Nothing to trim in our basic implementation.
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
  iree_host_size_t count = 3;  // device-local, host-visible, host-local
  if (out_count) *out_count = count;
  if (capacity < count) {
    return iree_status_from_code(IREE_STATUS_OUT_OF_RANGE);
  }

  const iree_device_size_t max_allocation_size = ~(iree_device_size_t)0;
  const iree_device_size_t min_alignment = 64;

  int i = 0;

  // Device-local memory:
  heaps[i++] = (iree_hal_allocator_memory_heap_t){
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .allowed_usage =
          IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH,
      .max_allocation_size = max_allocation_size,
      .min_alignment = min_alignment,
  };

  // Host-visible device memory:
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

  // Host-local cached memory:
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
  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers are importable.
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

  // HSA doesn't have managed memory like HIP/CUDA, so device-local + host-visible
  // allocations need to fall back to host-local + device-visible page-locked
  // memory. This will be significantly slower for the device to access but the
  // compiler only uses this type for readback staging buffers.
  if (iree_all_bits_set(params->type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_LOW_PERFORMANCE;
    params->type &= ~(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
    params->type |=
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }

  // We are now optimal.
  params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;

  // Guard against the corner case where the requested buffer size is 0.
  if (*allocation_size == 0) *allocation_size = 4;

  return compatibility;
}

static void iree_hal_hsa_buffer_free(
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_hsa_buffer_type_t buffer_type, void* device_ptr, void* host_ptr) {
  IREE_TRACE_ZONE_BEGIN(z0);
  switch (buffer_type) {
    case IREE_HAL_HSA_BUFFER_TYPE_DEVICE: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hsa_amd_memory_pool_free");
      IREE_HSA_IGNORE_ERROR(hsa_symbols, hsa_amd_memory_pool_free(device_ptr));
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_HOST: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hsa_amd_memory_pool_free (host)");
      IREE_HSA_IGNORE_ERROR(hsa_symbols, hsa_amd_memory_pool_free(host_ptr));
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_HOST_REGISTERED: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "hsa_amd_memory_unlock");
      IREE_HSA_IGNORE_ERROR(hsa_symbols, hsa_amd_memory_unlock(host_ptr));
      break;
    }
    case IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL: {
      IREE_TRACE_ZONE_APPEND_TEXT(z0, "(ignored; external)");
      break;
    }
  }
  IREE_TRACE_ZONE_END(z0);
}

static void iree_hal_hsa_buffer_release_callback(void* user_data,
                                                 iree_hal_buffer_t* buffer);

static iree_status_t iree_hal_hsa_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  iree_hal_hsa_allocator_t* allocator =
      iree_hal_hsa_allocator_cast(base_allocator);

  // Coerce options into those required by the current device.
  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
      iree_hal_hsa_allocator_query_buffer_compatibility(
          base_allocator, &compat_params, &allocation_size);
  if (!iree_all_bits_set(compatibility,
                         IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot allocate a buffer with the given parameters");
  }

  iree_status_t status = iree_ok_status();
  iree_hal_hsa_buffer_type_t buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
  void* host_ptr = NULL;
  void* device_ptr = NULL;
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_hsa_buffer_allocate");
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, allocation_size);

  int device_ordinal = 0;
  if (params->queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(params->queue_affinity);
  }

  iree_hal_hsa_per_device_info_t* device_info =
      &allocator->topology.devices[device_ordinal];

  if (iree_all_bits_set(compat_params.type,
                        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local allocation.
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
    if (device_info->device_local_memory_pool_valid) {
      status = IREE_HSA_CALL_TO_STATUS(
          allocator->symbols,
          hsa_amd_memory_pool_allocate(device_info->device_local_memory_pool,
                                       allocation_size, 0, &device_ptr),
          "hsa_amd_memory_pool_allocate");
      // Allow access from all agents.
      if (iree_status_is_ok(status) && device_info->cpu_agent.handle) {
        hsa_agent_t agents[2] = {device_info->agent, device_info->cpu_agent};
        IREE_HSA_IGNORE_ERROR(allocator->symbols,
                              hsa_amd_agents_allow_access(2, agents, NULL,
                                                          device_ptr));
      }
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "no device local memory pool available");
    }
  } else {
    // Host local allocation.
    buffer_type = IREE_HAL_HSA_BUFFER_TYPE_HOST;
    if (device_info->host_visible_memory_pool_valid) {
      status = IREE_HSA_CALL_TO_STATUS(
          allocator->symbols,
          hsa_amd_memory_pool_allocate(device_info->host_visible_memory_pool,
                                       allocation_size, 0, &host_ptr),
          "hsa_amd_memory_pool_allocate (host)");
      device_ptr = host_ptr;
      // Allow access from GPU.
      if (iree_status_is_ok(status)) {
        hsa_agent_t agents[1] = {device_info->agent};
        IREE_HSA_IGNORE_ERROR(allocator->symbols,
                              hsa_amd_agents_allow_access(1, agents, NULL,
                                                          host_ptr));
      }
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "no host visible memory pool available");
    }
  }
  IREE_TRACE_ZONE_END(z0);

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_affinity = params->queue_affinity ? params->queue_affinity
                                                 : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    iree_hal_buffer_release_callback_t callback = {
        .fn = iree_hal_hsa_buffer_release_callback,
        .user_data = (void*)base_allocator};
    status = iree_hal_hsa_buffer_wrap(
        placement, compat_params.type, compat_params.access,
        compat_params.usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, buffer_type, device_ptr, host_ptr,
        callback, iree_hal_allocator_host_allocator(base_allocator), &buffer);
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
  iree_hal_buffer_destroy(base_buffer);
}

static void iree_hal_hsa_buffer_release_callback(void* user_data,
                                                 iree_hal_buffer_t* buffer) {
  iree_hal_hsa_allocator_t* allocator = (iree_hal_hsa_allocator_t*)user_data;

  iree_hal_hsa_buffer_free(allocator->symbols, iree_hal_hsa_buffer_type(buffer),
                           iree_hal_hsa_buffer_device_pointer(buffer),
                           iree_hal_hsa_buffer_host_pointer(buffer));

  IREE_TRACE_FREE_NAMED(IREE_HAL_HSA_ALLOCATOR_ID,
                        (void*)iree_hal_hsa_buffer_device_pointer(buffer));
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(buffer),
      iree_hal_buffer_allocation_size(buffer)));
}

static iree_status_t iree_hal_hsa_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
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
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocator cannot import a buffer with the given parameters");
  }

  int device_ordinal = 0;
  if (params->queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(params->queue_affinity);
  }

  iree_hal_hsa_per_device_info_t* device_info =
      &allocator->topology.devices[device_ordinal];

  iree_status_t status = iree_ok_status();
  iree_hal_hsa_buffer_type_t buffer_type = IREE_HAL_HSA_BUFFER_TYPE_DEVICE;
  void* host_ptr = NULL;
  void* device_ptr = NULL;

  switch (external_buffer->type) {
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION: {
      buffer_type = IREE_HAL_HSA_BUFFER_TYPE_HOST_REGISTERED;
      host_ptr = external_buffer->handle.host_allocation.ptr;
      // Lock host memory and get device pointer.
      status = IREE_HSA_CALL_TO_STATUS(
          allocator->symbols,
          hsa_amd_memory_lock(host_ptr, external_buffer->size,
                              &device_info->agent, 1, &device_ptr),
          "hsa_amd_memory_lock");
      break;
    }
    case IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION: {
      buffer_type = IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL;
      device_ptr = (void*)external_buffer->handle.device_allocation.ptr;
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external buffer type not supported");
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_affinity = params->queue_affinity ? params->queue_affinity
                                                 : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_hsa_buffer_wrap(
        placement, compat_params.type, compat_params.access,
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
        case IREE_HAL_HSA_BUFFER_TYPE_DEVICE:
        case IREE_HAL_HSA_BUFFER_TYPE_EXTERNAL:
          out_external_buffer->flags = requested_flags;
          out_external_buffer->type = requested_type;
          out_external_buffer->handle.device_allocation.ptr =
              (uint64_t)(uintptr_t)iree_hal_hsa_buffer_device_pointer(buffer);
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

