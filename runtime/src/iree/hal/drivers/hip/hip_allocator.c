// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_allocator.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/hip_device.h"
#include "iree/hal/drivers/hip/per_device_information.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/util/queue.h"
#include "iree/hal/drivers/hip/util/tree.h"

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_ALLOCATION_TRACKING
static const char* IREE_HAL_HIP_ALLOCATOR_ID = "HIP unpooled";
#endif  // IREE_TRACING_FEATURE_ALLOCATION_TRACKING

#define IREE_HAL_HIP_ASYNC_ALLOCATION_DEFAULT_QUEUE_SIZE 8

typedef struct iree_hal_hip_async_allocation_t {
  iree_device_size_t allocation_size;
  hipDeviceptr_t pointer;
} iree_hal_hip_async_allocation_t;

IREE_HAL_HIP_UTIL_TYPED_QUEUE_WRAPPER(
    iree_hal_hip_async_allocation_queue, iree_hal_hip_async_allocation_t,
    IREE_HAL_HIP_ASYNC_ALLOCATION_DEFAULT_QUEUE_SIZE);

typedef struct iree_hal_hip_async_allocation_map_item_t {
  iree_hal_hip_async_allocation_queue_t queue;
} iree_hal_hip_async_allocation_map_item_t;

typedef struct iree_hal_hip_allocator_async_allocation_map {
  iree_hal_hip_util_tree_t tree;
  // Inline storage for this tree. We expect the normal number of
  // nodes in use for a single semaphore to be relatively small.
  uint8_t inline_storage[sizeof(iree_hal_hip_async_allocation_map_item_t) * 16];
} iree_hal_hip_allocator_async_allocation_map_t;

typedef struct iree_hal_hip_allocator_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // Parent device that this allocator is associated with. Unowned.
  iree_hal_device_t* parent_device;

  iree_hal_hip_device_topology_t topology;

  bool supports_memory_pools;

  const iree_hal_hip_dynamic_symbols_t* symbols;

  iree_allocator_t host_allocator;

  // Whether the GPU and CPU can concurrently access HIP managed data in a
  // coherent way. We would need to explicitly perform flushing and invalidation
  // between GPU and CPU if not.
  bool supports_concurrent_managed_access;

  iree_slim_mutex_t async_allocation_mutex;

  // One allocation map per device in the topology.
  iree_hal_hip_allocator_async_allocation_map_t* async_allocation_maps;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_hip_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_hip_allocator_vtable;

static iree_hal_hip_allocator_t* iree_hal_hip_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_allocator_vtable);
  return (iree_hal_hip_allocator_t*)base_value;
}

iree_status_t iree_hal_hip_allocator_create(
    iree_hal_device_t* parent_device,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    iree_hal_hip_device_topology_t topology, bool supports_memory_pools,
    iree_allocator_t host_allocator, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(parent_device);
  IREE_ASSERT_ARGUMENT(hip_symbols);
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_allocator = NULL;
  if (topology.count < 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one device must be specified");
  }

  // To support device-local + host-visible memory we need concurrent managed
  // access indicating that the host and devices can concurrently access the
  // device memory. If we don't have this feature then we fall back to forcing
  // all device-local + host-visible memory into host-local + device-visible
  // page-locked memory. The compiler tries to avoid this for high-traffic
  // buffers except for readback staging buffers.
  int supports_concurrent_managed_access = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HIP_CALL_TO_STATUS(
              hip_symbols,
              hipDeviceGetAttribute(&supports_concurrent_managed_access,
                                    hipDeviceAttributeConcurrentManagedAccess,
                                    topology.devices[0].hip_device),
              "hipDeviceGetAttribute"));

  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, supports_concurrent_managed_access
              ? "has CONCURRENT_MANAGED_ACCESS"
              : "no CONCURRENT_MANAGED_ACCESS (expect slow accesses on "
                "device-local + host-visible memory)");

  iree_hal_hip_allocator_t* allocator = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              host_allocator,
              sizeof(*allocator) +
                  topology.count * sizeof(*allocator->async_allocation_maps),
              (void**)&allocator));
  iree_hal_resource_initialize(&iree_hal_hip_allocator_vtable,
                               &allocator->resource);
  allocator->parent_device = parent_device;
  allocator->supports_memory_pools = supports_memory_pools;
  allocator->symbols = hip_symbols;
  allocator->host_allocator = host_allocator;
  allocator->supports_concurrent_managed_access =
      supports_concurrent_managed_access != 0;
  allocator->topology = topology;
  allocator->async_allocation_maps =
      (iree_hal_hip_allocator_async_allocation_map_t*)((uint8_t*)allocator +
                                                       sizeof(*allocator));
  for (iree_host_size_t i = 0; i < topology.count; ++i) {
    iree_hal_hip_util_tree_initialize(
        host_allocator, sizeof(iree_hal_hip_async_allocation_map_item_t),
        allocator->async_allocation_maps[i].inline_storage,
        sizeof(allocator->async_allocation_maps[i].inline_storage),
        &allocator->async_allocation_maps[i].tree);
  }
  iree_slim_mutex_initialize(&allocator->async_allocation_mutex);

  *out_allocator = (iree_hal_allocator_t*)allocator;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_allocator_destroy(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < allocator->topology.count; ++i) {
    for (iree_hal_hip_util_tree_node_t* j = iree_hal_hip_util_tree_first(
             &allocator->async_allocation_maps[i].tree);
         j != NULL; j = iree_hal_hip_util_tree_node_next(j)) {
      iree_hal_hip_async_allocation_map_item_t* queue_item =
          (iree_hal_hip_async_allocation_map_item_t*)
              iree_hal_hip_util_tree_node_get_value(j);
      while (!iree_hal_hip_async_allocation_queue_empty(&queue_item->queue)) {
        iree_hal_hip_async_allocation_t allocation =
            iree_hal_hip_async_allocation_queue_at(&queue_item->queue, 0);

        iree_status_ignore(IREE_HIP_CALL_TO_STATUS(
            allocator->symbols, hipFree(allocation.pointer), "hipFree"));
        iree_hal_hip_async_allocation_queue_pop_front(&queue_item->queue, 1);
      }
      iree_hal_hip_async_allocation_queue_deinitialize(&queue_item->queue);
    }
    iree_hal_hip_util_tree_deinitialize(
        &allocator->async_allocation_maps[i].tree);
  }
  iree_slim_mutex_deinitialize(&allocator->async_allocation_mutex);
  iree_allocator_free(allocator->host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_hip_allocator_isa(iree_hal_allocator_t* base_value) {
  return iree_hal_resource_is(base_value, &iree_hal_hip_allocator_vtable);
}

static iree_allocator_t iree_hal_hip_allocator_host_allocator(
    const iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hip_allocator_t* allocator =
      (iree_hal_hip_allocator_t*)base_allocator;
  return allocator->host_allocator;
}

static iree_status_t iree_hal_hip_allocator_trim(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);

  iree_slim_mutex_lock(&allocator->async_allocation_mutex);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < allocator->topology.count; ++i) {
    status = IREE_HIP_CALL_TO_STATUS(
        allocator->symbols,
        hipStreamSynchronize(
            allocator->topology.devices[i].hip_dispatch_stream),
        "hipStreamSynchronize");
    for (iree_hal_hip_util_tree_node_t* j = iree_hal_hip_util_tree_first(
             &allocator->async_allocation_maps[i].tree);
         iree_status_is_ok(status) && j != NULL;
         j = iree_hal_hip_util_tree_node_next(j)) {
      iree_hal_hip_async_allocation_map_item_t* queue_item =
          (iree_hal_hip_async_allocation_map_item_t*)
              iree_hal_hip_util_tree_node_get_value(j);
      while (!iree_hal_hip_async_allocation_queue_empty(&queue_item->queue)) {
        iree_hal_hip_async_allocation_t allocation =
            iree_hal_hip_async_allocation_queue_at(&queue_item->queue, 0);

        status = IREE_HIP_CALL_TO_STATUS(
            allocator->symbols, hipFree(allocation.pointer), "hipFree");
        iree_hal_hip_async_allocation_queue_pop_front(&queue_item->queue, 1);
      }
    }
  }
  iree_slim_mutex_unlock(&allocator->async_allocation_mutex);

  return status;
}

static void iree_hal_hip_allocator_query_statistics(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_allocator_statistics_t* IREE_RESTRICT out_statistics) {
  IREE_STATISTICS({
    iree_hal_hip_allocator_t* allocator =
        iree_hal_hip_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));

    if (allocator->supports_memory_pools) {
      for (iree_host_size_t i = 0; i < allocator->topology.count; ++i) {
        iree_hal_hip_memory_pools_merge_statistics(
            &allocator->topology.devices[i].memory_pools, out_statistics);
      }
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

static void iree_hal_hip_buffer_release_callback(void* user_data,
                                                 iree_hal_buffer_t* buffer);

static iree_status_t iree_hal_hip_allocator_allocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
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

  int device_ordinal = 0;
  if (params->queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(params->queue_affinity);
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_HIP_CALL_TO_STATUS(
              allocator->symbols,
              hipCtxPushCurrent(
                  allocator->topology.devices[device_ordinal].hip_context)));

  if (iree_all_bits_set(compat_params.type,
                        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local case.
    if (iree_all_bits_set(compat_params.type,
                          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      // Device local and host visible.
      buffer_type = IREE_HAL_HIP_BUFFER_TYPE_DEVICE;
      status = IREE_HIP_CALL_TO_STATUS(
          allocator->symbols,
          hipMallocManaged(&device_ptr, allocation_size, hipMemAttachGlobal));
      if (iree_status_is_ok(status) &&
          allocator->supports_concurrent_managed_access) {
        // Prefetch the buffer on the GPU device.
        status = IREE_HIP_CALL_TO_STATUS(
            allocator->symbols,
            hipMemPrefetchAsync(
                device_ptr, allocation_size,
                allocator->topology.devices[device_ordinal].hip_device,
                allocator->topology.devices[device_ordinal]
                    .hip_dispatch_stream));
      }
      host_ptr = (void*)device_ptr;
    } else {
      // Device only.
      buffer_type = IREE_HAL_HIP_BUFFER_TYPE_DEVICE;
      status = IREE_HIP_CALL_TO_STATUS(allocator->symbols,
                                       hipMalloc(&device_ptr, allocation_size));
    }
  } else {
    // Host local case.
    buffer_type = IREE_HAL_HIP_BUFFER_TYPE_HOST;
    unsigned int flags = hipHostMallocMapped;
    if (!iree_all_bits_set(compat_params.type,
                           IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
      flags |= hipHostMallocWriteCombined;
    }
    status = IREE_HIP_CALL_TO_STATUS(
        allocator->symbols, hipHostMalloc(&host_ptr, allocation_size, flags));
    if (iree_status_is_ok(status)) {
      status = IREE_HIP_CALL_TO_STATUS(
          allocator->symbols,
          hipHostGetDevicePointer(&device_ptr, host_ptr, /*flags=*/0));
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
        .fn = iree_hal_hip_buffer_release_callback,
        .user_data = (void*)base_allocator};
    status = iree_hal_hip_buffer_wrap(
        placement, compat_params.type, compat_params.access,
        compat_params.usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, buffer_type, device_ptr, host_ptr,
        callback, iree_hal_allocator_host_allocator(base_allocator), &buffer);
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

  return iree_status_join(
      status,
      IREE_HIP_CALL_TO_STATUS(allocator->symbols, hipCtxPopCurrent(NULL)));
}

static void iree_hal_hip_allocator_deallocate_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    iree_hal_buffer_t* IREE_RESTRICT base_buffer) {
  iree_hal_buffer_destroy(base_buffer);
}

typedef struct iree_hal_hip_release_async_data_t {
  iree_hal_hip_allocator_t* allocator;
  iree_hal_hip_buffer_type_t buffer_type;
  hipDeviceptr_t device_pointer;
  void* host_pointer;
  IREE_STATISTICS(iree_hal_memory_type_t memory_type;
                  iree_device_size_t allocation_size;)
} iree_hal_hip_release_async_data_t;

static iree_status_t iree_hal_hip_buffer_release_callback_async(
    void* user_data, iree_hal_hip_event_t* event, iree_status_t status) {
  iree_hal_hip_release_async_data_t* async_data =
      (iree_hal_hip_release_async_data_t*)user_data;

  iree_hal_hip_buffer_free(async_data->allocator->symbols,
                           async_data->buffer_type, async_data->device_pointer,
                           async_data->host_pointer);

  switch (async_data->buffer_type) {
    case IREE_HAL_HIP_BUFFER_TYPE_DEVICE:
    case IREE_HAL_HIP_BUFFER_TYPE_HOST: {
      IREE_TRACE_FREE_NAMED(IREE_HAL_HIP_ALLOCATOR_ID,
                            (void*)async_data->device_pointer);
      IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
          &async_data->allocator->statistics, async_data->memory_type,
          async_data->allocation_size));
      break;
    }
    default:
      // Buffer type not tracked.
      break;
  }
  iree_allocator_free(async_data->allocator->host_allocator, async_data);
  return status;
}

static void iree_hal_hip_buffer_release_callback(void* user_data,
                                                 iree_hal_buffer_t* buffer) {
  iree_hal_hip_allocator_t* allocator = (iree_hal_hip_allocator_t*)user_data;

  iree_hal_hip_release_async_data_t* release_async_data = NULL;

  iree_status_t status = iree_allocator_malloc(allocator->host_allocator,
                                               sizeof(*release_async_data),
                                               (void**)&release_async_data);
  if (iree_status_is_ok(status)) {
    release_async_data->allocator = allocator;
    release_async_data->device_pointer =
        iree_hal_hip_buffer_device_pointer(buffer);
    release_async_data->host_pointer = iree_hal_hip_buffer_host_pointer(buffer);
    release_async_data->buffer_type = iree_hal_hip_buffer_type(buffer);
    IREE_STATISTICS({
      release_async_data->memory_type = iree_hal_buffer_memory_type(buffer);
      release_async_data->allocation_size =
          iree_hal_buffer_allocation_size(buffer);
    })
    status = iree_hal_hip_device_add_asynchronous_cleanup(
        allocator->parent_device, &iree_hal_hip_buffer_release_callback_async,
        (void*)release_async_data);
  }
  iree_status_ignore(status);
}

static iree_status_t iree_hal_hip_allocator_import_buffer(
    iree_hal_allocator_t* IREE_RESTRICT base_allocator,
    const iree_hal_buffer_params_t* IREE_RESTRICT params,
    iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
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

  int device_ordinal = 0;
  if (params->queue_affinity) {
    device_ordinal = iree_math_count_trailing_zeros_u64(params->queue_affinity);
  }

  IREE_RETURN_IF_ERROR(IREE_HIP_CALL_TO_STATUS(
      allocator->symbols,
      hipCtxPushCurrent(
          allocator->topology.devices[device_ordinal].hip_context)));

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
      status = IREE_HIP_CALL_TO_STATUS(
          allocator->symbols,
          hipHostRegister(host_ptr, external_buffer->size, register_flags),
          "hipHostRegister");
      if (iree_status_is_ok(status)) {
        status = IREE_HIP_CALL_TO_STATUS(
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
    const iree_hal_buffer_placement_t placement = {
        .device = allocator->parent_device,
        .queue_affinity = params->queue_affinity ? params->queue_affinity
                                                 : IREE_HAL_QUEUE_AFFINITY_ANY,
        .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE,
    };
    status = iree_hal_hip_buffer_wrap(
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
      iree_hal_hip_buffer_free(allocator->symbols, buffer_type, device_ptr,
                               host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }

  return iree_status_join(
      status,
      IREE_HIP_CALL_TO_STATUS(allocator->symbols, hipCtxPopCurrent(NULL)));
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
        case IREE_HAL_HIP_BUFFER_TYPE_ASYNC:
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

iree_status_t iree_hal_hip_allocator_alloc_async(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);
  if (iree_hal_buffer_allocation_size(buffer) > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer allocation of %" PRIdsz
                            " too large to map into host address space with a "
                            "max value of %" PRIhsz,
                            iree_hal_buffer_allocation_size(buffer),
                            IREE_HOST_SIZE_MAX);
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  int device_ordinal = 0;
  device_ordinal =
      iree_math_count_trailing_zeros_u64(buffer->placement.queue_affinity);

  iree_host_size_t allocation_size_request =
      (iree_host_size_t)iree_hal_buffer_allocation_size(buffer);

  hipDeviceptr_t ptr = NULL;
  iree_slim_mutex_lock(&allocator->async_allocation_mutex);
  iree_hal_hip_util_tree_node_t* sized_allocations =
      iree_hal_hip_util_tree_lower_bound(
          &allocator->async_allocation_maps[device_ordinal].tree,
          allocation_size_request);
  if (sized_allocations != NULL) {
    iree_host_size_t actual_allocation_size =
        iree_hal_hip_util_tree_node_get_key(sized_allocations);
    if (actual_allocation_size == allocation_size_request) {
      iree_hal_hip_async_allocation_map_item_t* queue_item =
          (iree_hal_hip_async_allocation_map_item_t*)
              iree_hal_hip_util_tree_node_get_value(sized_allocations);
      if (iree_hal_hip_async_allocation_queue_count(&queue_item->queue) != 0) {
        iree_hal_hip_async_allocation_t allocation =
            iree_hal_hip_async_allocation_queue_at(&queue_item->queue, 0);
        ptr = allocation.pointer;
        iree_hal_hip_async_allocation_queue_pop_front(&queue_item->queue, 1);
        if (iree_hal_hip_async_allocation_queue_empty(&queue_item->queue)) {
          iree_hal_hip_async_allocation_queue_deinitialize(&queue_item->queue);
          iree_hal_hip_util_tree_erase(
              &allocator->async_allocation_maps[device_ordinal].tree,
              sized_allocations);
        }
      }
    }
  }
  iree_slim_mutex_unlock(&allocator->async_allocation_mutex);

  iree_status_t status = iree_ok_status();
  if (ptr == NULL) {
    status = IREE_HIP_CALL_TO_STATUS(
        allocator->symbols,
        hipCtxPushCurrent(
            allocator->topology.devices[device_ordinal].hip_context));
    if (iree_status_is_ok(status)) {
      // In an ideal world we would use hipMallocAsync/hipFreeAsync,
      // however the caching inside can cause lots of slack
      // to the point of unusability depending on the memory allocation
      // patterns of the host program, so instead we simply hipMalloc/hipFree.
      status = IREE_HIP_CALL_TO_STATUS(
          allocator->symbols,
          hipMalloc(&ptr, (size_t)iree_hal_buffer_allocation_size(buffer)),
          "hipMalloc");

      status = iree_status_join(
          status,
          IREE_HIP_CALL_TO_STATUS(allocator->symbols, hipCtxPopCurrent(NULL),
                                  "hipCtxPopCurrent"));
    }
  }

  if (iree_status_is_ok(status)) {
    iree_hal_hip_buffer_set_device_pointer(buffer, ptr);
    IREE_TRACE_ALLOC_NAMED(IREE_HAL_HIP_ALLOCATOR_ID, (void*)ptr,
                           iree_hal_buffer_allocation_size(buffer));
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, iree_hal_buffer_memory_type(buffer),
        iree_hal_buffer_allocation_size(buffer)));
  } else {
    iree_hal_hip_buffer_set_allocation_empty(buffer);
  }

  IREE_TRACE_ZONE_END(z0);

  return status;
}

iree_status_t iree_hal_hip_allocator_free_async(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);
  hipDeviceptr_t device_ptr = iree_hal_hip_buffer_device_pointer(buffer);
  if (!device_ptr) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_FREE_NAMED(IREE_HAL_HIP_ALLOCATOR_ID, (void*)device_ptr);
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(buffer),
      iree_hal_buffer_allocation_size(buffer)));

  int device_ordinal = 0;
  device_ordinal =
      iree_math_count_trailing_zeros_u64(buffer->placement.queue_affinity);

  iree_status_t status = iree_ok_status();
  iree_slim_mutex_lock(&allocator->async_allocation_mutex);
  iree_hal_hip_util_tree_node_t* bucket = iree_hal_hip_util_tree_get(
      &allocator->async_allocation_maps[device_ordinal].tree,
      (iree_host_size_t)iree_hal_buffer_allocation_size(buffer));
  if (bucket == NULL) {
    status = iree_hal_hip_util_tree_insert(
        &allocator->async_allocation_maps[device_ordinal].tree,
        (iree_host_size_t)iree_hal_buffer_allocation_size(buffer), &bucket);
    iree_hal_hip_async_allocation_map_item_t* queue_item =
        (iree_hal_hip_async_allocation_map_item_t*)
            iree_hal_hip_util_tree_node_get_value(bucket);
    iree_hal_hip_async_allocation_queue_initialize(allocator->host_allocator,
                                                   &queue_item->queue);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_hip_async_allocation_t allocation = {
        .allocation_size = iree_hal_buffer_allocation_size(buffer),
        .pointer = device_ptr};
    status = iree_hal_hip_async_allocation_queue_push_back(
        &((iree_hal_hip_async_allocation_map_item_t*)
              iree_hal_hip_util_tree_node_get_value(bucket))
             ->queue,
        allocation);
  }
  iree_slim_mutex_unlock(&allocator->async_allocation_mutex);

  if (iree_status_is_ok(status)) {
    iree_hal_hip_buffer_set_allocation_empty(buffer);
  }
  IREE_TRACE_ZONE_END(z0);

  return status;
}

iree_status_t iree_hal_hip_allocator_free_sync(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* buffer) {
  iree_hal_hip_allocator_t* allocator =
      iree_hal_hip_allocator_cast(base_allocator);
  hipDeviceptr_t device_ptr = iree_hal_hip_buffer_device_pointer(buffer);
  if (!device_ptr) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_FREE_NAMED(IREE_HAL_HIP_ALLOCATOR_ID, (void*)device_ptr);
  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, iree_hal_buffer_memory_type(buffer),
      iree_hal_buffer_allocation_size(buffer)));

  iree_status_t status = IREE_HIP_CALL_TO_STATUS(
      allocator->symbols, hipFree(device_ptr), "hipFree");

  if (iree_status_is_ok(status)) {
    iree_hal_hip_buffer_set_allocation_empty(buffer);
  }
  IREE_TRACE_ZONE_END(z0);

  return status;
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
