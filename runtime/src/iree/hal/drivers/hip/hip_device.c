// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/hip/cleanup_thread.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/event_pool.h"
#include "iree/hal/drivers/hip/event_semaphore.h"
#include "iree/hal/drivers/hip/graph_command_buffer.h"
#include "iree/hal/drivers/hip/hip_allocator.h"
#include "iree/hal/drivers/hip/hip_buffer.h"
#include "iree/hal/drivers/hip/hip_multi_queue_command_buffer.h"
#include "iree/hal/drivers/hip/memory_pools.h"
#include "iree/hal/drivers/hip/nop_executable_cache.h"
#include "iree/hal/drivers/hip/per_device_information.h"
#include "iree/hal/drivers/hip/rccl_channel.h"
#include "iree/hal/drivers/hip/rccl_dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/drivers/hip/stream_command_buffer.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/hal/utils/stream_tracing.h"

//===----------------------------------------------------------------------===//
// iree_hal_hip_device_t
//===----------------------------------------------------------------------===//

typedef enum iree_hip_device_commandbuffer_type_e {
  IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_STREAM,
  IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_GRAPH,
} iree_hip_device_commandbuffer_type_t;

typedef struct iree_hal_hip_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the HIP symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols;

  // Parameters used to control device behavior.
  iree_hal_hip_device_params_t params;

  iree_allocator_t host_allocator;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;

  // Device memory pools and allocators.
  bool supports_memory_pools;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  iree_hal_allocator_t* device_allocator;

  iree_hal_hip_memory_pools_t memory_pools;

  iree_hal_hip_cleanup_thread_t* cleanup_thread;

  iree_hal_hip_device_topology_t topology;
} iree_hal_hip_device_t;

static iree_hal_hip_device_t* iree_hal_hip_device_cast(
    iree_hal_device_t* base_value);

static const iree_hal_device_vtable_t iree_hal_hip_device_vtable;

static iree_status_t iree_hal_hip_device_create_command_buffer_internal(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hip_device_commandbuffer_type_t type,
    iree_hal_command_buffer_t** out_command_buffer);

typedef struct iree_hal_hip_tracing_device_interface_t {
  iree_hal_stream_tracing_device_interface_t base;
  iree_hal_hip_per_device_info_t* device_context;
  iree_allocator_t host_allocator;
  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
} iree_hal_hip_tracing_device_interface_t;
static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_hip_tracing_device_interface_vtable_t;

static void iree_hal_hip_tracing_device_interface_destroy(
    iree_hal_stream_tracing_device_interface_t* base_device_interface) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  iree_allocator_free(device_interface->host_allocator, device_interface);
}

static iree_status_t
iree_hal_hip_tracing_device_interface_synchronize_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_CALL_TO_STATUS(device_interface->hip_symbols,
                                 hipEventSynchronize((hipEvent_t)base_event));
}

static iree_status_t iree_hal_hip_tracing_device_interface_create_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t* base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_CALL_TO_STATUS(
      device_interface->hip_symbols,
      hipEventCreateWithFlags((hipEvent_t*)base_event, hipEventDefault));
}

static iree_status_t iree_hal_hip_tracing_device_interface_query_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_CALL_TO_STATUS(device_interface->hip_symbols,
                                 hipEventQuery((hipEvent_t)base_event));
}

static void iree_hal_hip_tracing_device_interface_event_elapsed_time(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    float* relative_millis, iree_hal_stream_tracing_native_event_t start_event,
    iree_hal_stream_tracing_native_event_t end_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  IREE_HIP_IGNORE_ERROR(
      device_interface->hip_symbols,
      hipEventElapsedTime(relative_millis, (hipEvent_t)start_event,
                          (hipEvent_t)end_event));
}

static void iree_hal_hip_tracing_device_interface_destroy_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  IREE_HIP_IGNORE_ERROR(device_interface->hip_symbols,
                        hipEventDestroy((hipEvent_t)base_event));
}

static iree_status_t iree_hal_hip_tracing_device_interface_record_native_event(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_event_t base_event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_CALL_TO_STATUS(
      device_interface->hip_symbols,
      hipEventRecord(
          (hipEvent_t)base_event,
          (hipStream_t)device_interface->device_context->hip_dispatch_stream));
}

static iree_status_t
iree_hal_hip_tracing_device_interface_add_graph_event_record_node(
    iree_hal_stream_tracing_device_interface_t* base_device_interface,
    iree_hal_stream_tracing_native_graph_node_t* out_node,
    iree_hal_stream_tracing_native_graph_t graph,
    iree_hal_stream_tracing_native_graph_node_t* dependency_nodes,
    size_t dependency_nodes_count,
    iree_hal_stream_tracing_native_event_t event) {
  iree_hal_hip_tracing_device_interface_t* device_interface =
      (iree_hal_hip_tracing_device_interface_t*)base_device_interface;

  return IREE_HIP_CALL_TO_STATUS(
      device_interface->hip_symbols,
      hipGraphAddEventRecordNode((hipGraphNode_t*)out_node, (hipGraph_t)graph,
                                 (hipGraphNode_t*)dependency_nodes,
                                 dependency_nodes_count, (hipEvent_t)event));
}

static iree_hal_hip_device_t* iree_hal_hip_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hip_device_vtable);
  return (iree_hal_hip_device_t*)base_value;
}

static iree_hal_hip_device_t* iree_hal_hip_device_cast_unsafe(
    iree_hal_device_t* base_value) {
  return (iree_hal_hip_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_hip_device_params_initialize(
    iree_hal_hip_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->command_buffer_mode = IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM;
  out_params->stream_tracing = 0;
  out_params->async_allocations = true;
  out_params->allow_inline_execution = false;
}

static iree_status_t iree_hal_hip_device_check_params(
    const iree_hal_hip_device_params_t* params) {
  if (params->arena_block_size < 4096) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "arena block size too small (< 4096 bytes)");
  }
  if (params->queue_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one queue is required");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_initialize_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params, iree_hal_hip_device_t* device,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_allocator_t host_allocator) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t identifier_offset =
      sizeof(*device) +
      sizeof(iree_hal_hip_per_device_info_t) * device->topology.count;

  iree_hal_resource_initialize(&iree_hal_hip_device_vtable, &device->resource);
  iree_string_view_append_to_buffer(identifier, &device->identifier,
                                    (char*)device + identifier_offset);
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->hip_symbols = symbols;
  device->nccl_symbols = nccl_symbols;
  device->params = *params;

  device->host_allocator = host_allocator;
  iree_status_t status = iree_ok_status();
  // Enable tracing for each of the streams - no-op if disabled.
  if (device->params.stream_tracing) {
    if (device->params.stream_tracing >=
            IREE_HAL_STREAM_TRACING_VERBOSITY_MAX ||
        device->params.stream_tracing < IREE_HAL_STREAM_TRACING_VERBOSITY_OFF) {
      iree_hal_device_release((iree_hal_device_t*)device);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid stream_tracing argument: expected to be between %d and %d",
          IREE_HAL_STREAM_TRACING_VERBOSITY_OFF,
          IREE_HAL_STREAM_TRACING_VERBOSITY_MAX);
    }

    for (iree_host_size_t i = 0; i < device->topology.count; ++i) {
      iree_hal_hip_tracing_device_interface_t* tracing_device_interface = NULL;
      status = iree_allocator_malloc(host_allocator,
                                     sizeof(*tracing_device_interface),
                                     (void**)&tracing_device_interface);

      if (!iree_status_is_ok(status)) {
        break;
      }

      tracing_device_interface->base.vtable =
          &iree_hal_hip_tracing_device_interface_vtable_t;
      tracing_device_interface->device_context = &device->topology.devices[i];
      tracing_device_interface->host_allocator = host_allocator;
      tracing_device_interface->hip_symbols = symbols;

      status = IREE_HIP_CALL_TO_STATUS(
          symbols, hipCtxPushCurrent(device->topology.devices[i].hip_context));
      if (!iree_status_is_ok(status)) {
        break;
      }
      status = iree_hal_stream_tracing_context_allocate(
          (iree_hal_stream_tracing_device_interface_t*)tracing_device_interface,
          device->identifier, device->params.stream_tracing,
          &device->block_pool, host_allocator,
          &device->topology.devices[i].tracing_context);
      status = IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL));
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
  }

  // Memory pool support is conditional.
  if (iree_status_is_ok(status) && params->async_allocations) {
    device->supports_memory_pools = true;
    for (iree_host_size_t i = 0; i < device->topology.count; ++i) {
      int supports_memory_pools = 0;
      status = IREE_HIP_CALL_TO_STATUS(
          symbols,
          hipDeviceGetAttribute(&supports_memory_pools,
                                hipDeviceAttributeMemoryPoolsSupported,
                                device->topology.devices[i].hip_device),
          "hipDeviceGetAttribute");
      device->supports_memory_pools &= (supports_memory_pools != 0);
    }
  }

  // Create memory pools first so that we can share them with the allocator.
  if (iree_status_is_ok(status) && device->supports_memory_pools) {
    device->supports_memory_pools = false;
    // TODO(awoloszyn): Figure out how to set up memory pools in a device group
    // status = iree_hal_hip_memory_pools_initialize(
    //     symbols, hip_devices[i], &params->memory_pools, host_allocator,
    //     &device->topology.devices[i].memory_pools);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_allocator_create(
        (iree_hal_device_t*)device, symbols, &device->topology,
        device->supports_memory_pools ? &device->memory_pools : NULL,
        host_allocator, &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_cleanup_thread_initialize(symbols, host_allocator,
                                                    &device->cleanup_thread);
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_hip_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_hip_device_params_t* params,
    const iree_hal_hip_dynamic_symbols_t* symbols,
    const iree_hal_hip_nccl_dynamic_symbols_t* nccl_symbols,
    iree_host_size_t device_count, hipDevice_t* devices,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_device_t* device = NULL;
  const iree_host_size_t total_device_size =
      sizeof(*device) + sizeof(iree_hal_hip_per_device_info_t) * device_count +
      identifier.size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_device_size,
                                (void**)&device));
  device->topology.count = device_count;

  iree_status_t status = iree_hal_hip_device_check_params(params);

  // Get the main context for the device.
  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    device->topology.devices[i].hip_device = devices[i];
    status = IREE_HIP_CALL_TO_STATUS(
        symbols, hipDevicePrimaryCtxRetain(
                     &device->topology.devices[i].hip_context, devices[i]));
    if (iree_status_is_ok(status)) {
      status = IREE_HIP_CALL_TO_STATUS(
          symbols, hipCtxSetCurrent(device->topology.devices[i].hip_context));
    }

    // Create the default dispatch stream for the device.
    if (iree_status_is_ok(status)) {
      status = IREE_HIP_CALL_TO_STATUS(
          symbols, hipStreamCreateWithFlags(
                       &device->topology.devices[i].hip_dispatch_stream,
                       hipStreamNonBlocking));
    }

    if (iree_status_is_ok(status)) {
      for (iree_host_size_t j = 0;
           j < device_count && iree_status_is_ok(status); ++j) {
        if (i == j) {
          continue;
        }
        status = IREE_HIP_CALL_TO_STATUS(
            symbols, hipDeviceEnablePeerAccess(devices[j], 0));
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_initialize_internal(
        driver, identifier, params, device, symbols, nccl_symbols,
        host_allocator);
  } else {
    for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
         ++i) {
      if (device->topology.devices[i].hip_dispatch_stream)
        symbols->hipStreamDestroy(
            device->topology.devices[i].hip_dispatch_stream);
      // NOTE: This function return hipSuccess though doesn't release the
      // primaryCtx by design on HIP/HCC path.
      if (device->topology.devices[i].hip_context)
        symbols->hipDevicePrimaryCtxRelease(
            device->topology.devices[i].hip_device);
    }
  }

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  for (iree_host_size_t i = 0; i < device_count && iree_status_is_ok(status);
       ++i) {
    if (iree_status_is_ok(status)) {
      status = iree_hal_hip_event_pool_allocate(
          symbols, params->event_pool_capacity, host_allocator,
          device->topology.devices[i].hip_context,
          &device->topology.devices[i].device_event_pool);
    }
  }

  if (iree_status_is_ok(status)) {
    device->host_event_pool = host_event_pool;
    *out_device = (iree_hal_device_t*)device;
  } else {
    // Release resources we have accquired after HAL device creation.
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      if (device->topology.devices[i].device_event_pool)
        iree_hal_hip_event_pool_release(
            device->topology.devices[i].device_event_pool);
    }
    if (host_event_pool) iree_event_pool_free(host_event_pool);
    // Release other resources via the HAL device.
    iree_hal_device_release((iree_hal_device_t*)device);
    device = NULL;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

const iree_hal_hip_dynamic_symbols_t* iree_hal_hip_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast_unsafe(base_device);
  return device->hip_symbols;
}

static void iree_hal_hip_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  const iree_hal_hip_dynamic_symbols_t* symbols = device->hip_symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_cleanup_thread_deinitialize(device->cleanup_thread);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // Destroy memory pools that hold on to reserved memory.
  iree_hal_hip_memory_pools_deinitialize(&device->memory_pools);
  for (iree_host_size_t i = 0; i < device->topology.count; ++i) {
    iree_hal_stream_tracing_context_free(
        device->topology.devices[i].tracing_context);
  }

  for (iree_host_size_t i = 0; i < device->topology.count; ++i) {
    iree_hal_hip_event_pool_release(
        device->topology.devices[i].device_event_pool);
  }
  if (device->host_event_pool) iree_event_pool_free(device->host_event_pool);

  for (iree_host_size_t i = 0; i < device->topology.count; ++i) {
    IREE_HIP_IGNORE_ERROR(
        symbols,
        hipStreamDestroy(device->topology.devices[i].hip_dispatch_stream));
    // NOTE: This function return hipSuccess though doesn't release the
    // primaryCtx by design on HIP/HCC path.
    IREE_HIP_IGNORE_ERROR(symbols, hipDevicePrimaryCtxRelease(
                                       device->topology.devices[i].hip_device));
  }

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_hip_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_hip_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_hip_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_hip_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_hip_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_hip_device_trim(iree_hal_device_t* base_device) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  if (device->supports_memory_pools) {
    IREE_RETURN_IF_ERROR(iree_hal_hip_memory_pools_trim(
        &device->memory_pools, &device->params.memory_pools));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_query_attribute(
    iree_hal_hip_device_t* device, hipDeviceAttribute_t attribute,
    int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = 0;
  int value = 0;
  IREE_HIP_RETURN_IF_ERROR(
      device->hip_symbols,
      hipDeviceGetAttribute(&value, attribute,
                            device->topology.devices[0].hip_device),
      "hipDeviceGetAttribute");
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("rocm-hsaco-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      *out_value = device->topology.count;
      return iree_ok_status();
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_hip_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  if (!device->nccl_symbols || !device->nccl_symbols->dylib) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "RCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (rccl.dll/librccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  // Today we only allow a single logical device per channel.
  // We could multiplex channels but it'd be better to surface that to the
  // compiler so that it can emit the right rank math.
  int requested_count = iree_math_count_ones_u64(queue_affinity);
  // TODO(#12206): properly assign affinity in the compiler.
  if (requested_count != 64 && requested_count != 1) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exactly one participant is allowed in a "
                            "channel but %d were specified",
                            requested_count);
  }

  // Ask the channel provider (if configured) for the default rank and count
  // if the user did not set them.
  if (device->channel_provider &&
      (params.rank == IREE_HAL_CHANNEL_RANK_DEFAULT ||
       params.count == IREE_HAL_CHANNEL_COUNT_DEFAULT)) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_channel_provider_query_default_rank_and_count(
            device->channel_provider, &params.rank, &params.count),
        "querying default collective group rank and count");
  }

  // An ID is required to initialize NCCL. On the root it'll be the local ID and
  // on all other participants it'll be the root ID.
  iree_hal_hip_nccl_id_t id;
  memset(&id, 0, sizeof(id));
  if (iree_const_byte_span_is_empty(params.id)) {
    // User wants the default ID.
    if (!device->channel_provider) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "default collective channel ID requested but no channel provider has "
          "been set on the device to provide it");
    }
    if (params.rank == 0) {
      // Bootstrap NCCL to get the root ID.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_hip_nccl_get_unique_id(device->nccl_symbols, &id),
          "bootstrapping NCCL root");
    }
    // Exchange NCCL ID with all participants.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_channel_provider_exchange_default_id(
            device->channel_provider,
            iree_make_byte_span((void*)&id, sizeof(id))),
        "exchanging NCCL ID with other participants");
  } else if (params.id.data_length != IREE_ARRAYSIZE(id.data)) {
    IREE_TRACE_ZONE_END(z0);
    // User provided something but it's not what we expect.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NCCL ID must be %zu bytes matching the "
                            "ncclUniqueId struct but caller provided %zu bytes",
                            IREE_ARRAYSIZE(id.data), sizeof(id));
  } else {
    // User provided the ID - we treat it as opaque here and let NCCL validate.
    memcpy(id.data, params.id.data, IREE_ARRAYSIZE(id.data));
  }

  if (iree_hal_hip_nccl_id_is_empty(&id)) {
    IREE_TRACE_ZONE_END(z0);
    // TODO: maybe this is ok? a localhost alias or something?
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no default NCCL ID specified (all zeros)");
  }

  // TODO: when we support multiple logical devices we'll want to pass in the
  // context of the device mapped to the queue_affinity. For now since this
  // implementation only supports one device we pass in the only one we have.
  iree_status_t status = iree_hal_hip_nccl_channel_create(
      device->hip_symbols, device->nccl_symbols, &id, params.rank, params.count,
      device->host_allocator, out_channel);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_create_command_buffer_internal(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hip_device_commandbuffer_type_t type,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);

  iree_hal_command_buffer_t* buffers[IREE_HAL_MAX_QUEUES];
  memset(buffers, 0x00, sizeof(buffers[0]) * IREE_HAL_MAX_QUEUES);
  if (queue_affinity == 0) {
    queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
  }
  queue_affinity =
      queue_affinity & ~(IREE_HAL_QUEUE_AFFINITY_ANY << device->topology.count);

  iree_status_t status = iree_ok_status();
  iree_host_size_t device_ordinal = 0;
  iree_host_size_t command_buffer_ordinal = 0;
  iree_hal_queue_affinity_t current_affinity = queue_affinity;
  while (current_affinity) {
    int next_device_ordinal_offset =
        iree_math_count_trailing_zeros_u64(current_affinity);
    device_ordinal += next_device_ordinal_offset;
    current_affinity >>= next_device_ordinal_offset + 1;
    status = IREE_HIP_CALL_TO_STATUS(
        device->hip_symbols,
        hipCtxPushCurrent(
            device->topology.devices[device_ordinal].hip_context));
    if (!iree_status_is_ok(status)) {
      break;
    }
    switch (type) {
      case IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_STREAM:
        status = iree_hal_hip_stream_command_buffer_create(
            iree_hal_device_allocator(base_device), device->hip_symbols,
            device->nccl_symbols,
            device->topology.devices[device_ordinal].tracing_context, mode,
            command_categories, (iree_hal_queue_affinity_t)1 << device_ordinal,
            binding_capacity,
            device->topology.devices[device_ordinal].hip_dispatch_stream,
            &device->block_pool, device->host_allocator,
            &buffers[command_buffer_ordinal]);
        break;
      case IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_GRAPH:
        status = iree_hal_hip_graph_command_buffer_create(
            iree_hal_device_allocator(base_device), device->hip_symbols,
            device->topology.devices[device_ordinal].tracing_context,
            device->topology.devices[device_ordinal].hip_context, mode,
            command_categories, (iree_hal_queue_affinity_t)1 << device_ordinal,
            binding_capacity, &device->block_pool, device->host_allocator,
            &buffers[command_buffer_ordinal]);
        break;
    }

    status = iree_status_join(
        status,
        IREE_HIP_CALL_TO_STATUS(device->hip_symbols, hipCtxPopCurrent(NULL)));
    ++device_ordinal;
    ++command_buffer_ordinal;
    if (!iree_status_is_ok(status)) {
      break;
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_multi_queue_command_buffer_create(
        command_buffer_ordinal, &buffers[0], device->device_allocator, mode,
        command_categories, queue_affinity, device->hip_symbols,
        &device->topology, binding_capacity, device->host_allocator,
        out_command_buffer);
  }

  // If |iree_hal_hip_multi_queue_command_buffer_create| was successful, it will
  // have retained the command buffers, if not, then it will have not.
  // So we release here either way.
  for (iree_host_size_t i = 0; i < IREE_HAL_MAX_QUEUES; ++i) {
    iree_hal_resource_release(buffers[i]);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_create_stream_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  return iree_hal_hip_device_create_command_buffer_internal(
      base_device, mode, command_categories, queue_affinity, binding_capacity,
      IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_STREAM, out_command_buffer);
}

static iree_status_t iree_hal_hip_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  if (device->params.allow_inline_execution &&
      iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    // The caller has indicated the command buffer can be executed as it is
    // recorded, implying that the command buffer cannot be reused and doesn't
    // need to be persisted. This lets us lower the execution delay as we can
    // directly route commands to a HIP stream and let it eagerly flush.
    return iree_hal_hip_device_create_command_buffer_internal(
        base_device, mode, command_categories, queue_affinity, binding_capacity,
        IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_STREAM, out_command_buffer);
  }
  switch (device->params.command_buffer_mode) {
    case IREE_HAL_HIP_COMMAND_BUFFER_MODE_GRAPH:
      // TODO(indirect-cmd): when we can record indirect graphs we won't need
      // to use deferred command buffers - this is here to emulate indirect
      // command buffers.
      if (binding_capacity > 0) {
        return iree_hal_deferred_command_buffer_create(
            iree_hal_device_allocator(base_device), mode, command_categories,
            queue_affinity, binding_capacity, &device->block_pool,
            iree_hal_device_host_allocator(base_device), out_command_buffer);
      } else {
        return iree_hal_hip_device_create_command_buffer_internal(
            base_device, mode, command_categories, queue_affinity,
            binding_capacity, IREE_HAL_HIP_DEVICE_COMMAND_BUFFER_TYPE_GRAPH,
            out_command_buffer);
      }
    case IREE_HAL_HIP_COMMAND_BUFFER_MODE_STREAM:
      return iree_hal_deferred_command_buffer_create(
          iree_hal_device_allocator(base_device), mode, command_categories,
          queue_affinity, binding_capacity, &device->block_pool,
          iree_hal_device_host_allocator(base_device), out_command_buffer);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid command buffer mode");
  }
}

static iree_status_t iree_hal_hip_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implmeneted");
}

static iree_status_t iree_hal_hip_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  *out_file = NULL;
  if (iree_io_file_handle_type(handle) !=
      IREE_IO_FILE_HANDLE_TYPE_HOST_ALLOCATION) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "implementation does not support the external file type");
  }
  return iree_hal_memory_file_wrap(
      queue_affinity, access, handle, iree_hal_device_allocator(base_device),
      iree_hal_device_host_allocator(base_device), out_file);
}

static iree_status_t iree_hal_hip_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  hipDevice_t devices[IREE_HAL_MAX_QUEUES];
  hipCtx_t contexts[IREE_HAL_MAX_QUEUES];
  for (iree_host_size_t i = 0; i < device->topology.count; ++i) {
    devices[i] = device->topology.devices[i].hip_device;
    contexts[i] = device->topology.devices[i].hip_context;
  }
  return iree_hal_hip_nop_executable_cache_create(
      identifier, device->hip_symbols, &device->topology,
      device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_hip_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_flags_t flags, iree_hal_semaphore_t** out_semaphore) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return iree_hal_hip_event_semaphore_create(initial_value, device->hip_symbols,
                                             device->host_allocator,
                                             out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_hip_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement HIP semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_hip_device_pepare_async_alloc(
    iree_hal_hip_device_t* device, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)allocation_size);
  *out_buffer = NULL;
  iree_hal_buffer_params_canonicalize(&params);

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_hip_buffer_wrap(
      device->device_allocator, params.type, params.access, params.usage,
      allocation_size, /*byte_offset=*/0,
      /*byte_length=*/allocation_size, IREE_HAL_HIP_BUFFER_TYPE_ASYNC,
      /*device_ptr=*/NULL, /*host_ptr=*/NULL,
      iree_hal_buffer_release_callback_null(), device->host_allocator, &buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else if (buffer) {
    iree_hal_hip_buffer_set_allocation_empty(buffer);
    iree_hal_buffer_release(buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
typedef enum iree_hal_hip_device_semaphore_buffer_operation_type_e {
  IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_ALLOC,
  IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_DEALLOC,
  IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_MAX =
      IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_DEALLOC,
} iree_hal_hip_device_semaphore_buffer_operation_type_t;
typedef struct iree_hal_hip_device_semaphore_buffer_operation_callback_data_t {
  iree_atomic_ref_count_t wait_semaphore_count;
  iree_hal_hip_device_t* device;
  iree_hal_queue_affinity_t queue_affinity;
  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_hal_semaphore_list_t signal_semaphore_list;
  iree_hal_buffer_t* buffer;
  iree_hal_hip_device_semaphore_buffer_operation_type_t type;
  iree_slim_mutex_t status_mutex;
  iree_status_t status;
} iree_hal_hip_device_semaphore_buffer_operation_callback_data_t;

static iree_status_t iree_hal_hip_device_make_buffer_callback_data(
    iree_hal_hip_device_t* device, iree_allocator_t host_allocator,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer,
    iree_hal_hip_device_semaphore_buffer_operation_type_t type,
    iree_hal_hip_device_semaphore_buffer_operation_callback_data_t** out_data) {
  *out_data = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  // Embed captured tables in the action allocation.
  iree_hal_hip_device_semaphore_buffer_operation_callback_data_t*
      callback_data = NULL;

  const iree_host_size_t wait_semaphore_list_size =
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores) +
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values);
  const iree_host_size_t signal_semaphore_list_size =
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores) +
      signal_semaphore_list.count *
          sizeof(*signal_semaphore_list.payload_values);

  const iree_host_size_t total_callback_size = sizeof(*callback_data) +
                                               wait_semaphore_list_size +
                                               signal_semaphore_list_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_callback_size,
                                (void**)&callback_data));
  uint8_t* callback_ptr = (uint8_t*)callback_data + sizeof(*callback_data);

  iree_atomic_ref_count_init_value(&callback_data->wait_semaphore_count,
                                   wait_semaphore_list.count);

  callback_data->device = device;
  callback_data->queue_affinity = queue_affinity;

  // Copy wait list for later access.
  callback_data->wait_semaphore_list.count = wait_semaphore_list.count;
  callback_data->wait_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)callback_ptr;
  memcpy(callback_data->wait_semaphore_list.semaphores,
         wait_semaphore_list.semaphores,
         wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores));
  callback_data->wait_semaphore_list.payload_values =
      (uint64_t*)(callback_ptr + wait_semaphore_list.count *
                                     sizeof(*wait_semaphore_list.semaphores));
  memcpy(
      callback_data->wait_semaphore_list.payload_values,
      wait_semaphore_list.payload_values,
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values));
  callback_ptr += wait_semaphore_list_size;

  // Copy signal list for later access.
  callback_data->signal_semaphore_list.count = signal_semaphore_list.count;
  callback_data->signal_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)callback_ptr;
  memcpy(
      callback_data->signal_semaphore_list.semaphores,
      signal_semaphore_list.semaphores,
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores));
  callback_data->signal_semaphore_list.payload_values =
      (uint64_t*)(callback_ptr + signal_semaphore_list.count *
                                     sizeof(*signal_semaphore_list.semaphores));
  memcpy(callback_data->signal_semaphore_list.payload_values,
         signal_semaphore_list.payload_values,
         signal_semaphore_list.count *
             sizeof(*signal_semaphore_list.payload_values));
  callback_ptr += signal_semaphore_list_size;

  callback_data->buffer = buffer;
  iree_hal_buffer_retain(buffer);
  callback_data->type = type;

  iree_slim_mutex_initialize(&callback_data->status_mutex);
  callback_data->status = iree_ok_status();
  *out_data = callback_data;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t
iree_hal_hip_device_stream_signal_semaphores_and_add_cleanup(
    iree_hal_hip_device_t* device,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t device_ordinal, iree_hal_hip_cleanup_callback_t callback,
    void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_hip_event_t* event = NULL;
    iree_status_t status = iree_hal_hip_semaphore_get_hip_event(
        signal_semaphore_list.semaphores[i],
        signal_semaphore_list.payload_values[i],
        device->topology.devices[device_ordinal].device_event_pool, true,
        &event);

    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!event) {
      status =
          iree_make_status(IREE_STATUS_ABORTED, "the hip event is missing");
      break;
    }
    status = IREE_HIP_CALL_TO_STATUS(
        device->hip_symbols,
        hipEventRecord(
            iree_hal_hip_event_handle(event),
            device->topology.devices[device_ordinal].hip_dispatch_stream));
    iree_hal_hip_event_release(event);
    if (!iree_status_is_ok(status)) {
      break;
    }
  }

  for (iree_host_size_t i = 0;
       i < signal_semaphore_list.count && iree_status_is_ok(status); ++i) {
    status = iree_hal_hip_semaphore_notify_forward_progress_to(
        signal_semaphore_list.semaphores[i],
        signal_semaphore_list.payload_values[i]);
  }

  iree_hal_hip_event_t* event = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_event_pool_acquire(
        device->topology.devices[device_ordinal].device_event_pool, 1, &event);
  }

  if (iree_status_is_ok(status)) {
    status = IREE_HIP_CALL_TO_STATUS(
        device->hip_symbols,
        hipEventRecord(
            iree_hal_hip_event_handle(event),
            device->topology.devices[device_ordinal].hip_dispatch_stream));
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_cleanup_thread_add_cleanup(
        device->cleanup_thread, event, callback, user_data);
  }
  IREE_TRACE_ZONE_END(z0);

  return status;
}

static iree_status_t iree_hal_hip_device_complete_buffer_operation(
    void* user_data, iree_hal_hip_event_t* event, iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_device_semaphore_buffer_operation_callback_data_t* data =
      (iree_hal_hip_device_semaphore_buffer_operation_callback_data_t*)
          user_data;
  iree_hal_hip_device_t* device = data->device;

  // Free the event we specifically created.
  iree_hal_hip_event_release(event);

  // Notify all of the signal semaphores that they have been incremented.
  for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
    uint64_t unused_return_value = 0;
    // We use query to force the semaphore to update.
    iree_status_ignore(iree_hal_semaphore_query(
        data->signal_semaphore_list.semaphores[i], &unused_return_value));
  }

  for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
    iree_hal_resource_release(data->signal_semaphore_list.semaphores[i]);
  }

  // Free the iree_hal_hip_device_semaphore_buffer_operation_callback_data_t
  // and the buffer attached.
  iree_hal_buffer_release(data->buffer);
  iree_slim_mutex_deinitialize(&data->status_mutex);
  iree_allocator_free(device->host_allocator, data);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_stream_wait_for_semaphores(
    iree_hal_hip_device_t* device,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_host_size_t device_ordinal) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_ok_status();
  // TODO(awoloszyn): Because of how hip works, if we only have a single
  // physical device in the hip_device we could avoid waiting on any of these
  // semaphores, we are guaranteed to have waits, but if we want this
  // to work across multiple device/streams, we need these waits.
  for (iree_host_size_t i = 0;
       i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
    iree_hal_hip_event_t* event = NULL;
    status = iree_hal_hip_semaphore_get_hip_event(
        wait_semaphore_list.semaphores[i],
        wait_semaphore_list.payload_values[i],
        device->topology.devices[device_ordinal].device_event_pool, false,
        &event);
    if (!iree_status_is_ok(status)) {
      break;
    }
    // If we don't have an event, then we don't have to wait for it since it
    // has already been signaled on the host.
    if (!event) {
      continue;
    }

    status = IREE_HIP_CALL_TO_STATUS(
        device->hip_symbols,
        hipStreamWaitEvent(
            device->topology.devices[device_ordinal].hip_dispatch_stream,
            iree_hal_hip_event_handle(event), 0));
    iree_hal_hip_event_release(event);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_perform_buffer_operation_now(
    iree_hal_hip_device_semaphore_buffer_operation_callback_data_t* data) {
  IREE_ASSERT_LE(data->type, IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_MAX);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_device_t* device = data->device;
  iree_status_t status = iree_ok_status();

  // If we had a semaphore failure then we should propagate it
  // but not run anything.
  if (!iree_status_is_ok(data->status)) {
    status = data->status;
  }

  int device_ordinal = iree_math_count_trailing_zeros_u64(data->queue_affinity);

  if (iree_status_is_ok(status)) {
    status = IREE_HIP_CALL_TO_STATUS(
        data->device->hip_symbols,
        hipCtxPushCurrent(
            data->device->topology.devices[device_ordinal].hip_context));
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, device_ordinal);

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_stream_wait_for_semaphores(
        data->device, data->wait_semaphore_list, device_ordinal);
  }

  // We have satisfied all of the waits.
  IREE_TRACE_ZONE_BEGIN_NAMED(
      z3, "iree_hal_hip_device_perform_buffer_operation_now_launch_operation");
  if (iree_status_is_ok(status)) {
    switch (data->type) {
      case IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_ALLOC:
        if (device->supports_memory_pools) {
          status = iree_hal_hip_memory_pools_allocate_pointer(
              &device->memory_pools, data->buffer,
              device->topology.devices[device_ordinal].hip_dispatch_stream,
              iree_hal_buffer_allocation_size(data->buffer));
          break;
        }
        status = iree_hal_hip_allocator_alloc_async(
            iree_hal_device_allocator((iree_hal_device_t*)data->device),
            device->topology.devices[device_ordinal].hip_dispatch_stream,
            data->buffer);
        break;
      case IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_DEALLOC:
        if (device->supports_memory_pools) {
          status = iree_hal_hip_memory_pools_deallocate(
              &device->memory_pools,
              device->topology.devices[device_ordinal].hip_dispatch_stream,
              data->buffer);
          break;
        }
        status = iree_hal_hip_allocator_free_async(
            iree_hal_device_allocator((iree_hal_device_t*)data->device),
            device->topology.devices[device_ordinal].hip_dispatch_stream,
            data->buffer);
        break;
    }
  }
  IREE_TRACE_ZONE_END(z3);
  const iree_hal_hip_dynamic_symbols_t* symbols = data->device->hip_symbols;
  if (iree_status_is_ok(status)) {
    // Retain the semaphores for the cleanup thread.
    for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
      iree_hal_resource_retain(data->signal_semaphore_list.semaphores[i]);
    }
    // Data may get deleted any time after adding it to the cleanup,
    // so retain the symbols here.
    status = iree_hal_hip_device_stream_signal_semaphores_and_add_cleanup(
        data->device, data->signal_semaphore_list, device_ordinal,
        &iree_hal_hip_device_complete_buffer_operation, data);
  } else {
    for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
      iree_hal_semaphore_fail(data->signal_semaphore_list.semaphores[i],
                              iree_status_clone(data->status));
    }
    iree_hal_buffer_release(data->buffer);
    iree_slim_mutex_deinitialize(&data->status_mutex);
    iree_allocator_free(device->host_allocator, data);
  }
  status = iree_status_join(
      status, IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_semaphore_buffer_operation_callback(
    void* user_context, iree_hal_semaphore_t* semaphore, iree_status_t status) {
  iree_hal_hip_device_semaphore_buffer_operation_callback_data_t* data =
      (iree_hal_hip_device_semaphore_buffer_operation_callback_data_t*)
          user_context;
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&data->status_mutex);
    data->status = iree_status_join(data->status, status);
    iree_slim_mutex_unlock(&data->status_mutex);
  }
  if (iree_atomic_ref_count_dec(&data->wait_semaphore_count) != 1) {
    return iree_ok_status();
  }

  // Now the actual buffer_operation happens, as all semaphore have been
  // satisfied (by satisfied here, we specifically mean that the semaphore has
  // been scheduled, not necessarily completed)
  return iree_hal_hip_device_perform_buffer_operation_now(data);
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in HIP to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_hip_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  uint64_t queue_affinity_mask =
      ((iree_hal_queue_affinity_t)1 << device->topology.count);
  queue_affinity_mask = queue_affinity_mask | (queue_affinity_mask - 1);
  queue_affinity &= queue_affinity_mask;

  int device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  queue_affinity = (uint64_t)1 << device_ordinal;

  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
      (device->supports_memory_pools ||
       iree_hal_hip_allocator_isa(iree_hal_device_allocator(base_device)))) {
    iree_hal_buffer_t* buffer = NULL;

    status = iree_hal_hip_device_pepare_async_alloc(device, params,
                                                    allocation_size, &buffer);

    iree_hal_hip_device_semaphore_buffer_operation_callback_data_t*
        callback_data = NULL;
    if (iree_status_is_ok(status)) {
      status = iree_hal_hip_device_make_buffer_callback_data(
          device, device->host_allocator, queue_affinity, wait_semaphore_list,
          signal_semaphore_list, buffer,
          IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_ALLOC, &callback_data);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_hip_device_perform_buffer_operation_now(callback_data);
      *out_buffer = buffer;
      IREE_TRACE_ZONE_END(z0);
      return status;
    }

    for (iree_host_size_t i = 0;
         i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
      status = iree_hal_hip_semaphore_notify_work(
          wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i],
          device->topology.devices[device_ordinal].device_event_pool,
          &iree_hal_hip_device_semaphore_buffer_operation_callback,
          callback_data);
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      if (callback_data) {
        iree_allocator_free(device->host_allocator, callback_data);
      }
      if (buffer) {
        iree_hal_hip_buffer_set_allocation_empty(buffer);
        iree_hal_resource_release(&buffer->resource);
      }
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HIP HAL is not currently
  // asynchronous.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_list_wait(wait_semaphore_list,
                                       iree_infinite_timeout()));

  status =
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer);

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in HIP to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_hip_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  uint64_t queue_affinity_mask =
      ((iree_hal_queue_affinity_t)1 << device->topology.count);
  queue_affinity_mask = queue_affinity_mask | (queue_affinity_mask - 1);
  queue_affinity &= queue_affinity_mask;

  int device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);

  if (device_ordinal > device->topology.count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "device affinity out of range, maximum device is %" PRIhsz,
        device->topology.count);
  }

  queue_affinity = (uint64_t)1 << device_ordinal;

  iree_status_t status = iree_ok_status();
  if (iree_hal_hip_allocator_isa(iree_hal_device_allocator(base_device))) {
    iree_hal_hip_device_semaphore_buffer_operation_callback_data_t*
        callback_data;
    status = iree_hal_hip_device_make_buffer_callback_data(
        device, device->host_allocator, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, buffer,
        IREE_HAL_HIP_DEVICE_SEMAPHORE_OPERATION_ASYNC_DEALLOC, &callback_data);

    if (iree_status_is_ok(status)) {
      if (wait_semaphore_list.count == 0) {
        iree_status_t status =
            iree_hal_hip_device_perform_buffer_operation_now(callback_data);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }

    for (iree_host_size_t i = 0;
         i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
      status = iree_hal_hip_semaphore_notify_work(
          wait_semaphore_list.semaphores[i],
          wait_semaphore_list.payload_values[i],
          device->topology.devices[device_ordinal].device_event_pool,
          &iree_hal_hip_device_semaphore_buffer_operation_callback,
          callback_data);
    }
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(device->host_allocator, callback_data);
    }
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The HIP HAL is not currently
  // asynchronous.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_semaphore_list_wait(wait_semaphore_list,
                                       iree_infinite_timeout()));

  // Schedule the buffer deallocation if we got it from a pool and otherwise
  // drop it on the floor and let it be freed when the buffer is released.
  if (device->supports_memory_pools) {
    status = iree_hal_hip_memory_pools_deallocate(
        &device->memory_pools,
        device->topology.devices[device_ordinal].hip_dispatch_stream, buffer);
  }

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_read_streaming(
              base_device, queue_affinity, wait_semaphore_list,
              signal_semaphore_list, source_file, source_offset, target_buffer,
              target_offset, length, flags, options));
  IREE_TRACE_ZONE_END(z0);

  return loop_status;
}

static iree_status_t iree_hal_hip_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_device_queue_write_streaming(
              base_device, queue_affinity, wait_semaphore_list,
              signal_semaphore_list, source_buffer, source_offset, target_file,
              target_offset, length, flags, options));
  IREE_TRACE_ZONE_END(z0);
  return loop_status;
}

typedef struct iree_hal_hip_device_semaphore_submit_callback_data_t {
  iree_atomic_ref_count_t wait_semaphore_count;
  iree_hal_hip_device_t* device;
  iree_hal_queue_affinity_t queue_affinity;
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t binding_table;
  iree_hal_semaphore_list_t wait_semaphore_list;
  iree_hal_semaphore_list_t signal_semaphore_list;
  iree_hal_resource_set_t* resource_set;
  iree_slim_mutex_t status_mutex;
  iree_status_t status;
} iree_hal_hip_device_semaphore_submit_callback_data_t;

static iree_status_t iree_hal_hip_device_complete_submission(
    void* user_data, iree_hal_hip_event_t* event, iree_status_t status) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hip_device_semaphore_submit_callback_data_t* data =
      (iree_hal_hip_device_semaphore_submit_callback_data_t*)user_data;
  iree_hal_hip_device_t* device = data->device;

  // Get the device_context from the queue_affinity.
  int device_ordinal = iree_math_count_trailing_zeros_u64(data->queue_affinity);

  // Read any tracing events that were submitted.

  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_t* command_buffer = data->command_buffer;
    if (iree_hal_hip_multi_queue_command_buffer_isa(command_buffer)) {
      status = iree_hal_hip_multi_queue_command_buffer_get(
          command_buffer, data->queue_affinity, &command_buffer);
    }

    if (iree_status_is_ok(status)) {
      if (iree_hal_hip_stream_command_buffer_isa(command_buffer)) {
        status = iree_hal_stream_tracing_context_collect_list(
            // Get the tracing context from the device/stream/queue affinity.
            device->topology.devices[device_ordinal].tracing_context,
            // Get the tracing event list from the command buffer.
            iree_hal_hip_stream_command_buffer_tracing_events(command_buffer)
                .head);
      } else if (iree_hal_hip_graph_command_buffer_isa(command_buffer)) {
        status = iree_hal_stream_tracing_context_collect_list(
            // Get the tracing context from the device/stream/queue affinity.
            device->topology.devices[device_ordinal].tracing_context,
            // Get the tracing event list from the command buffer.
            iree_hal_hip_graph_command_buffer_tracing_events(command_buffer)
                .head);
      }
    }
  }

  // Free the event we specifically created.
  iree_hal_hip_event_release(event);

  // Notify all of the signal semaphores that they have been incremented.
  for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
    uint64_t unused_return_value = 0;
    // We use query to force the semaphore to update.
    iree_status_ignore(iree_hal_semaphore_query(
        data->signal_semaphore_list.semaphores[i], &unused_return_value));
  }

  for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
    iree_hal_resource_release(data->signal_semaphore_list.semaphores[i]);
  }

  // Free the iree_hal_hip_device_semaphore_submit_callback_data_t and
  // the resource set attached.
  iree_hal_resource_set_free(data->resource_set);
  iree_slim_mutex_deinitialize(&data->status_mutex);
  iree_allocator_free(device->host_allocator, data);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_execute_now(
    iree_hal_hip_device_semaphore_submit_callback_data_t* data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_EQ(iree_math_count_ones_u64(data->queue_affinity), 1,
                 "Cannot execute a command buffer on more than one queue");
  iree_hal_hip_device_t* device = data->device;

  iree_status_t status = iree_ok_status();
  // If we had a semaphore failure then we should propagate it
  // but not run anything.
  if (!iree_status_is_ok(data->status)) {
    status = data->status;
  }

  int device_ordinal = iree_math_count_trailing_zeros_u64(data->queue_affinity);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, device_ordinal);

  if (iree_status_is_ok(status)) {
    status = IREE_HIP_CALL_TO_STATUS(
        data->device->hip_symbols,
        hipCtxPushCurrent(
            data->device->topology.devices[device_ordinal].hip_context));
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_hip_device_stream_wait_for_semaphores(
        data->device, data->wait_semaphore_list, device_ordinal);
  }

  // We have satisfied all of the waits.

  IREE_TRACE_ZONE_BEGIN_NAMED(z1, "iree_hal_hip_device_execute_now_launch");
  iree_hal_command_buffer_t* command_buffer = data->command_buffer;
  if (iree_status_is_ok(status)) {
    if (iree_hal_hip_multi_queue_command_buffer_isa(command_buffer)) {
      status = iree_hal_hip_multi_queue_command_buffer_get(
          command_buffer, data->queue_affinity, &command_buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t binding_table = data->binding_table;
    if (iree_hal_deferred_command_buffer_isa(command_buffer)) {
      iree_hal_command_buffer_t* stream_command_buffer = NULL;
      iree_hal_command_buffer_mode_t mode =
          iree_hal_command_buffer_mode(command_buffer) |
          IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
          // NOTE: we need to validate if a binding table is provided as the
          // bindings were not known when it was originally recorded.
          (iree_hal_buffer_binding_table_is_empty(binding_table)
               ? IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED
               : 0);
      status = iree_hal_hip_device_create_stream_command_buffer(
          (iree_hal_device_t*)data->device, mode,
          command_buffer->allowed_categories, data->queue_affinity, 0,
          &stream_command_buffer);
      if (iree_status_is_ok(status)) {
        status = iree_hal_resource_set_insert(data->resource_set, 1,
                                              &stream_command_buffer);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_deferred_command_buffer_apply(
            command_buffer, stream_command_buffer, binding_table);
      }
      data->command_buffer = stream_command_buffer;
      iree_hal_resource_release(stream_command_buffer);
    } else if (iree_hal_hip_stream_command_buffer_isa(command_buffer)) {
      status =
          iree_hal_resource_set_insert(data->resource_set, 1, &command_buffer);
    } else if (iree_hal_hip_graph_command_buffer_isa(command_buffer)) {
      status =
          iree_hal_resource_set_insert(data->resource_set, 1, &command_buffer);
      if (iree_status_is_ok(status)) {
        IREE_TRACE_ZONE_BEGIN_NAMED(
            z2, "iree_hal_hip_device_execute_now_hip_graph_launch");
        hipGraphExec_t exec =
            iree_hal_hip_graph_command_buffer_handle(command_buffer);
        status = IREE_HIP_CALL_TO_STATUS(
            data->device->hip_symbols,
            hipGraphLaunch(
                exec,
                device->topology.devices[device_ordinal].hip_dispatch_stream));
        IREE_TRACE_ZONE_END(z2);
      }
    } else if (command_buffer) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "unsupported command buffer type");
    }
  }

  IREE_TRACE_ZONE_END(z1);

  // Store symbols, because the cleanup may trigger off-thread
  // before it returns.
  const iree_hal_hip_dynamic_symbols_t* symbols = data->device->hip_symbols;

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
      iree_hal_resource_retain(data->signal_semaphore_list.semaphores[i]);
    }
    status = iree_hal_hip_device_stream_signal_semaphores_and_add_cleanup(
        data->device, data->signal_semaphore_list, device_ordinal,
        iree_hal_hip_device_complete_submission, data);
  }

  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < data->signal_semaphore_list.count; ++i) {
      iree_hal_semaphore_fail(data->signal_semaphore_list.semaphores[i],
                              iree_status_clone(data->status));
    }
    iree_hal_resource_set_free(data->resource_set);
    iree_slim_mutex_deinitialize(&data->status_mutex);
    iree_allocator_free(device->host_allocator, data);
  }

  status = iree_status_join(
      status, IREE_HIP_CALL_TO_STATUS(symbols, hipCtxPopCurrent(NULL)));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_semaphore_submit_callback(
    void* user_context, iree_hal_semaphore_t* semaphore, iree_status_t status) {
  iree_hal_hip_device_semaphore_submit_callback_data_t* data =
      (iree_hal_hip_device_semaphore_submit_callback_data_t*)user_context;
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&data->status_mutex);
    data->status = iree_status_join(data->status, status);
    iree_slim_mutex_unlock(&data->status_mutex);
  }
  if (iree_atomic_ref_count_dec(&data->wait_semaphore_count) != 1) {
    return iree_ok_status();
  }

  // Now the actual submit happens, as all semaphore have been satisfied
  // (by satisfied here, we specifically mean that the semaphore has been
  // scheduled, not necessarily completed)
  return iree_hal_hip_device_execute_now(data);
}

static iree_status_t iree_hal_hip_device_make_callback_data(
    iree_hal_hip_device_t* device, iree_allocator_t host_allocator,
    iree_arena_block_pool_t* block_pool,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_hip_device_semaphore_submit_callback_data_t** out_data) {
  *out_data = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  // Embed captured tables in the action allocation.
  iree_hal_hip_device_semaphore_submit_callback_data_t* callback_data = NULL;

  const iree_host_size_t wait_semaphore_list_size =
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores) +
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values);
  const iree_host_size_t signal_semaphore_list_size =
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores) +
      signal_semaphore_list.count *
          sizeof(*signal_semaphore_list.payload_values);

  const iree_host_size_t payload_size =
      binding_table.count * sizeof(*binding_table.bindings);

  const iree_host_size_t total_callback_size =
      sizeof(*callback_data) + wait_semaphore_list_size +
      signal_semaphore_list_size + payload_size;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_callback_size,
                                (void**)&callback_data));
  uint8_t* callback_ptr = (uint8_t*)callback_data + sizeof(*callback_data);

  callback_data->device = device;

  iree_atomic_ref_count_init_value(&callback_data->wait_semaphore_count,
                                   wait_semaphore_list.count);
  // Copy wait list for later access.
  callback_data->wait_semaphore_list.count = wait_semaphore_list.count;
  callback_data->wait_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)callback_ptr;
  memcpy(callback_data->wait_semaphore_list.semaphores,
         wait_semaphore_list.semaphores,
         wait_semaphore_list.count * sizeof(*wait_semaphore_list.semaphores));
  callback_data->wait_semaphore_list.payload_values =
      (uint64_t*)(callback_ptr + wait_semaphore_list.count *
                                     sizeof(*wait_semaphore_list.semaphores));
  memcpy(
      callback_data->wait_semaphore_list.payload_values,
      wait_semaphore_list.payload_values,
      wait_semaphore_list.count * sizeof(*wait_semaphore_list.payload_values));
  callback_ptr += wait_semaphore_list_size;

  // Copy signal list for later access.
  callback_data->signal_semaphore_list.count = signal_semaphore_list.count;
  callback_data->signal_semaphore_list.semaphores =
      (iree_hal_semaphore_t**)callback_ptr;
  memcpy(
      callback_data->signal_semaphore_list.semaphores,
      signal_semaphore_list.semaphores,
      signal_semaphore_list.count * sizeof(*signal_semaphore_list.semaphores));
  callback_data->signal_semaphore_list.payload_values =
      (uint64_t*)(callback_ptr + signal_semaphore_list.count *
                                     sizeof(*signal_semaphore_list.semaphores));
  memcpy(callback_data->signal_semaphore_list.payload_values,
         signal_semaphore_list.payload_values,
         signal_semaphore_list.count *
             sizeof(*signal_semaphore_list.payload_values));
  callback_ptr += signal_semaphore_list_size;

  // Copy the execution resources for later access.
  callback_data->queue_affinity = queue_affinity;
  callback_data->command_buffer = command_buffer;

  // Retain all command buffers and semaphores.
  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &callback_data->resource_set);
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(callback_data->resource_set,
                                          wait_semaphore_list.count,
                                          wait_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(callback_data->resource_set,
                                          signal_semaphore_list.count,
                                          signal_semaphore_list.semaphores);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_resource_set_insert(callback_data->resource_set, 1,
                                          &command_buffer);
  }

  callback_data->binding_table = binding_table;
  iree_hal_buffer_binding_t* binding_element_ptr =
      (iree_hal_buffer_binding_t*)callback_ptr;
  callback_data->binding_table.bindings = binding_element_ptr;
  memcpy(binding_element_ptr, binding_table.bindings,
         sizeof(*binding_element_ptr) * binding_table.count);
  status = iree_hal_resource_set_insert_strided(
      callback_data->resource_set, binding_table.count,
      callback_data->binding_table.bindings,
      offsetof(iree_hal_buffer_binding_t, buffer),
      sizeof(iree_hal_buffer_binding_t));

  callback_data->status = iree_ok_status();
  iree_slim_mutex_initialize(&callback_data->status_mutex);
  *out_data = callback_data;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (queue_affinity == IREE_HAL_QUEUE_AFFINITY_ANY) {
    queue_affinity = 0x1;
  }

  uint64_t queue_affinity_mask =
      ((iree_hal_queue_affinity_t)1 << device->topology.count);
  queue_affinity_mask = queue_affinity_mask | (queue_affinity_mask - 1);
  queue_affinity &= queue_affinity_mask;

  int device_ordinal = iree_math_count_trailing_zeros_u64(queue_affinity);
  queue_affinity = (uint64_t)1 << device_ordinal;

  iree_hal_hip_device_semaphore_submit_callback_data_t* callback_data = NULL;
  iree_status_t status = iree_ok_status();
  status = iree_hal_hip_device_make_callback_data(
      device, device->host_allocator, &device->block_pool, queue_affinity,
      wait_semaphore_list, signal_semaphore_list, command_buffer, binding_table,
      &callback_data);

  if (iree_status_is_ok(status)) {
    if (wait_semaphore_list.count == 0) {
      status = iree_hal_hip_device_execute_now(callback_data);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
  }

  for (iree_host_size_t i = 0;
       i < wait_semaphore_list.count && iree_status_is_ok(status); ++i) {
    status = iree_hal_hip_semaphore_notify_work(
        wait_semaphore_list.semaphores[i],
        wait_semaphore_list.payload_values[i],
        device->topology.devices[device_ordinal].device_event_pool,
        &iree_hal_hip_device_semaphore_submit_callback, callback_data);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(device->host_allocator, callback_data);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hip_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  iree_hal_hip_device_t* device = iree_hal_hip_device_cast(base_device);
  return iree_hal_hip_semaphore_multi_wait(semaphore_list, wait_mode, timeout,
                                           device->host_allocator);
}

static iree_status_t iree_hal_hip_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_hip_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_hip_device_vtable = {
    .destroy = iree_hal_hip_device_destroy,
    .id = iree_hal_hip_device_id,
    .host_allocator = iree_hal_hip_device_host_allocator,
    .device_allocator = iree_hal_hip_device_allocator,
    .replace_device_allocator = iree_hal_hip_replace_device_allocator,
    .replace_channel_provider = iree_hal_hip_replace_channel_provider,
    .trim = iree_hal_hip_device_trim,
    .query_i64 = iree_hal_hip_device_query_i64,
    .create_channel = iree_hal_hip_device_create_channel,
    .create_command_buffer = iree_hal_hip_device_create_command_buffer,
    .create_event = iree_hal_hip_device_create_event,
    .create_executable_cache = iree_hal_hip_device_create_executable_cache,
    .import_file = iree_hal_hip_device_import_file,
    .create_semaphore = iree_hal_hip_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_hip_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_hip_device_queue_alloca,
    .queue_dealloca = iree_hal_hip_device_queue_dealloca,
    .queue_fill = iree_hal_device_queue_emulated_fill,
    .queue_update = iree_hal_device_queue_emulated_update,
    .queue_copy = iree_hal_device_queue_emulated_copy,
    .queue_read = iree_hal_hip_device_queue_read,
    .queue_write = iree_hal_hip_device_queue_write,
    .queue_execute = iree_hal_hip_device_queue_execute,
    .queue_flush = iree_hal_hip_device_queue_flush,
    .wait_semaphores = iree_hal_hip_device_wait_semaphores,
    .profiling_begin = iree_hal_hip_device_profiling_begin,
    .profiling_flush = iree_hal_hip_device_profiling_flush,
    .profiling_end = iree_hal_hip_device_profiling_end,
};

static const iree_hal_stream_tracing_device_interface_vtable_t
    iree_hal_hip_tracing_device_interface_vtable_t = {
        .destroy = iree_hal_hip_tracing_device_interface_destroy,
        .synchronize_native_event =
            iree_hal_hip_tracing_device_interface_synchronize_native_event,
        .create_native_event =
            iree_hal_hip_tracing_device_interface_create_native_event,
        .query_native_event =
            iree_hal_hip_tracing_device_interface_query_native_event,
        .event_elapsed_time =
            iree_hal_hip_tracing_device_interface_event_elapsed_time,
        .destroy_native_event =
            iree_hal_hip_tracing_device_interface_destroy_native_event,
        .record_native_event =
            iree_hal_hip_tracing_device_interface_record_native_event,
        .add_graph_event_record_node =
            iree_hal_hip_tracing_device_interface_add_graph_event_record_node,
};
