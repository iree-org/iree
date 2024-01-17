// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda2/cuda_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/cuda2/cuda_allocator.h"
#include "iree/hal/drivers/cuda2/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda2/cuda_status_util.h"
#include "iree/hal/drivers/cuda2/graph_command_buffer.h"
#include "iree/hal/drivers/cuda2/memory_pools.h"
#include "iree/hal/drivers/cuda2/nccl_channel.h"
#include "iree/hal/drivers/cuda2/nccl_dynamic_symbols.h"
#include "iree/hal/drivers/cuda2/nop_executable_cache.h"
#include "iree/hal/drivers/cuda2/pipeline_layout.h"
#include "iree/hal/drivers/cuda2/stream_command_buffer.h"
#include "iree/hal/drivers/cuda2/tracing.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/event_pool.h"
#include "iree/hal/utils/event_semaphore.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"
#include "iree/hal/utils/pending_queue_actions.h"
#include "iree/hal/utils/timepoint_pool.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda2_device_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the CUDA symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols;
  const iree_hal_cuda2_nccl_dynamic_symbols_t* nccl_symbols;

  // Parameters used to control device behavior.
  iree_hal_cuda2_device_params_t params;

  CUcontext cu_context;
  CUdevice cu_device;
  // TODO: Support multiple device streams.
  // The CUstream used to issue device kernels and allocations.
  CUstream dispatch_cu_stream;
  // The CUstream used to issue host callback functions.
  CUstream callback_cu_stream;

  iree_hal_cuda2_tracing_context_t* tracing_context;

  iree_allocator_t host_allocator;

  // Symbol tables for CUevent-implemented semaphore and pending action queue.
  iree_hal_event_impl_symtable_t event_impl_symtable;
  iree_hal_stream_impl_symtable_t stream_impl_symtable;
  iree_hal_command_buffer_impl_symtable_t command_buffer_impl_symtable;

  // Host/device event pools, used for backing semaphore timepoints.
  iree_event_pool_t* host_event_pool;
  iree_hal_cuda2_event_pool_t* device_event_pool;
  // Timepoint pools, shared by various semaphores.
  iree_hal_cuda2_timepoint_pool_t* timepoint_pool;

  // A queue to order device workloads and relase to the GPU when constraints
  // are met. It buffers submissions and allocations internally before they
  // are ready. This queue couples with HAL semaphores backed by iree_event_t
  // and CUevent objects.
  iree_hal_cuda2_pending_queue_actions_t* pending_queue_actions;

  // Device memory pools and allocators.
  bool supports_memory_pools;
  iree_hal_cuda2_memory_pools_t memory_pools;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;
} iree_hal_cuda2_device_t;

static const iree_hal_device_vtable_t iree_hal_cuda2_device_vtable;

static iree_hal_cuda2_device_t* iree_hal_cuda2_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda2_device_vtable);
  return (iree_hal_cuda2_device_t*)base_value;
}

static iree_hal_cuda2_device_t* iree_hal_cuda2_device_cast_unsafe(
    iree_hal_device_t* base_value) {
  return (iree_hal_cuda2_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_cuda2_device_params_initialize(
    iree_hal_cuda2_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->event_pool_capacity = 32;
  out_params->queue_count = 1;
  out_params->command_buffer_mode = IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH;
  out_params->stream_tracing = false;
  out_params->async_allocations = true;
}

static iree_status_t iree_hal_cuda2_device_check_params(
    const iree_hal_cuda2_device_params_t* params) {
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

static iree_status_t iree_hal_cuda2_event_impl_create(
    void* user_data, iree_hal_event_impl_t* out_event) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  return IREE_CURESULT_TO_STATUS(
      symbols, cuEventCreate((CUevent*)out_event, CU_EVENT_DISABLE_TIMING),
      "cuEventCreate");
}

static void iree_hal_cuda2_event_impl_destroy(void* user_data,
                                              iree_hal_event_impl_t event) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  IREE_CUDA_IGNORE_ERROR(symbols, cuEventDestroy((CUevent)event));
}

static iree_status_t iree_hal_cuda2_event_impl_synchronize(
    void* user_data, iree_hal_event_impl_t event) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  return IREE_CURESULT_TO_STATUS(symbols, cuEventSynchronize((CUevent)event),
                                 "cuEventSynchronize");
}

static iree_status_t iree_hal_cuda2_stream_impl_record_event(
    void* user_data, iree_hal_stream_impl_t stream,
    iree_hal_event_impl_t event) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  return IREE_CURESULT_TO_STATUS(
      symbols, cuEventRecord((CUevent)event, (CUstream)stream),
      "cuEventRecord");
}

static iree_status_t iree_hal_cuda2_stream_impl_wait_event(
    void* user_data, iree_hal_stream_impl_t stream,
    iree_hal_event_impl_t event) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  return IREE_CURESULT_TO_STATUS(
      symbols,
      cuStreamWaitEvent((CUstream)stream, (CUevent)event,
                        CU_EVENT_WAIT_DEFAULT),
      "cuStreamWaitEvent");
}

static iree_status_t iree_hal_cuda2_stream_impl_launch_graph(
    void* user_data, iree_hal_stream_impl_t stream,
    iree_hal_graph_executable_impl_t graph) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  return IREE_CURESULT_TO_STATUS(
      symbols, cuGraphLaunch((CUgraphExec)graph, (CUstream)stream),
      "cuGraphLaunch");
}

static iree_status_t iree_hal_cuda2_stream_impl_launch_host_fn(
    void* user_data, iree_hal_stream_impl_t stream,
    iree_hal_host_fn_impl_t host_fn, void* host_fn_user_data) {
  const iree_hal_cuda2_dynamic_symbols_t* symbols =
      (const iree_hal_cuda2_dynamic_symbols_t*)user_data;
  return IREE_CURESULT_TO_STATUS(
      symbols, cuLaunchHostFunc((CUstream)stream, host_fn, host_fn_user_data),
      "cuLaunchHostFunc");
}

static iree_hal_graph_executable_impl_t
iree_hal_cuda2_command_buffer_impl_graph_command_buffer_handle(
    iree_hal_command_buffer_t* command_buffer) {
  return (iree_hal_graph_executable_impl_t)
      iree_hal_cuda2_graph_command_buffer_handle(command_buffer);
}

static iree_status_t iree_hal_cuda2_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda2_device_params_t* params, CUdevice cu_device,
    CUstream dispatch_stream, CUstream callback_stream, CUcontext context,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda2_nccl_dynamic_symbols_t* nccl_symbols,
    iree_allocator_t host_allocator, iree_hal_cuda2_device_t** out_device) {
  iree_hal_cuda2_device_t* device = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));

  iree_hal_resource_initialize(&iree_hal_cuda2_device_vtable,
                               &device->resource);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + iree_sizeof_struct(*device));
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  device->cuda_symbols = cuda_symbols;
  device->nccl_symbols = nccl_symbols;
  device->params = *params;
  device->cu_context = context;
  device->cu_device = cu_device;
  device->dispatch_cu_stream = dispatch_stream;
  device->callback_cu_stream = callback_stream;
  device->host_allocator = host_allocator;

  iree_status_t status = iree_hal_cuda2_pending_queue_actions_create(
      &device->stream_impl_symtable, &device->command_buffer_impl_symtable,
      (void*)cuda_symbols, &device->block_pool, host_allocator,
      &device->pending_queue_actions);

  // Enable tracing for the (currently only) stream - no-op if disabled.
  if (iree_status_is_ok(status) && device->params.stream_tracing) {
    status = iree_hal_cuda2_tracing_context_allocate(
        device->cuda_symbols, device->identifier, dispatch_stream,
        &device->block_pool, host_allocator, &device->tracing_context);
  }

  // Memory pool support is conditional.
  if (iree_status_is_ok(status) && params->async_allocations) {
    int supports_memory_pools = 0;
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols,
        cuDeviceGetAttribute(&supports_memory_pools,
                             CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                             cu_device),
        "cuDeviceGetAttribute");
    device->supports_memory_pools = supports_memory_pools != 0;
  }

  // Create memory pools first so that we can share them with the allocator.
  if (iree_status_is_ok(status) && device->supports_memory_pools) {
    status = iree_hal_cuda2_memory_pools_initialize(
        cuda_symbols, cu_device, &params->memory_pools, host_allocator,
        &device->memory_pools);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_allocator_create(
        cuda_symbols, cu_device, dispatch_stream,
        device->supports_memory_pools ? &device->memory_pools : NULL,
        host_allocator, &device->device_allocator);
  }

  if (iree_status_is_ok(status)) {
    *out_device = device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_cuda2_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda2_device_params_t* params,
    const iree_hal_cuda2_dynamic_symbols_t* cuda_symbols,
    const iree_hal_cuda2_nccl_dynamic_symbols_t* nccl_symbols,
    CUdevice cuda_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(driver);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(cuda_symbols);
  IREE_ASSERT_ARGUMENT(out_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_cuda2_device_check_params(params);

  // Get the main context for the device.
  CUcontext context = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols, cuDevicePrimaryCtxRetain(&context, cuda_device));
  }
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(cuda_symbols, cuCtxSetCurrent(context));
  }

  // Create the default dispatch stream for the device.
  CUstream dispatch_stream = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols, cuStreamCreate(&dispatch_stream, CU_STREAM_NON_BLOCKING));
  }
  // Create the default callback stream for the device.
  CUstream callback_stream = NULL;
  if (iree_status_is_ok(status)) {
    status = IREE_CURESULT_TO_STATUS(
        cuda_symbols, cuStreamCreate(&callback_stream, CU_STREAM_NON_BLOCKING));
  }

  iree_hal_cuda2_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_device_create_internal(
        driver, identifier, params, cuda_device, dispatch_stream,
        callback_stream, context, cuda_symbols, nccl_symbols, host_allocator,
        &device);
  } else {
    // Release resources we have accquired thus far.
    if (callback_stream) cuda_symbols->cuStreamDestroy(callback_stream);
    if (dispatch_stream) cuda_symbols->cuStreamDestroy(dispatch_stream);
    if (context) cuda_symbols->cuDevicePrimaryCtxRelease(cuda_device);
  }

  iree_event_pool_t* host_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_event_pool_allocate(params->event_pool_capacity,
                                      host_allocator, &host_event_pool);
  }

  iree_hal_cuda2_event_pool_t* device_event_pool = NULL;
  if (iree_status_is_ok(status)) {
    device->event_impl_symtable.create = iree_hal_cuda2_event_impl_create;
    device->event_impl_symtable.destroy = iree_hal_cuda2_event_impl_destroy;
    device->event_impl_symtable.synchronize =
        iree_hal_cuda2_event_impl_synchronize;

    device->stream_impl_symtable.record_event =
        iree_hal_cuda2_stream_impl_record_event;
    device->stream_impl_symtable.wait_event =
        iree_hal_cuda2_stream_impl_wait_event;
    device->stream_impl_symtable.launch_graph =
        iree_hal_cuda2_stream_impl_launch_graph;
    device->stream_impl_symtable.launch_host_fn =
        iree_hal_cuda2_stream_impl_launch_host_fn;

    device->command_buffer_impl_symtable.is_graph_command_buffer =
        iree_hal_cuda2_graph_command_buffer_isa;
    device->command_buffer_impl_symtable.graph_command_buffer_handle =
        iree_hal_cuda2_command_buffer_impl_graph_command_buffer_handle;
    device->command_buffer_impl_symtable.create_stream_command_buffer =
        iree_hal_cuda2_device_create_stream_command_buffer;

    status = iree_hal_cuda2_event_pool_allocate(
        &device->event_impl_symtable, (void*)cuda_symbols,
        params->event_pool_capacity, host_allocator, &device_event_pool);
  }

  iree_hal_cuda2_timepoint_pool_t* timepoint_pool = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda2_timepoint_pool_allocate(
        host_event_pool, device_event_pool, params->event_pool_capacity,
        host_allocator, &timepoint_pool);
  }

  if (iree_status_is_ok(status)) {
    device->host_event_pool = host_event_pool;
    device->device_event_pool = device_event_pool;
    device->timepoint_pool = timepoint_pool;

    *out_device = (iree_hal_device_t*)device;
  } else {
    // Release resources we have accquired after HAL device creation.
    if (timepoint_pool) iree_hal_cuda2_timepoint_pool_free(timepoint_pool);
    if (device_event_pool) iree_hal_cuda2_event_pool_release(device_event_pool);
    if (host_event_pool) iree_event_pool_free(host_event_pool);

    // Release other resources via the HAL device.
    iree_hal_device_release((iree_hal_device_t*)device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

CUcontext iree_hal_cuda2_device_context(iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device =
      iree_hal_cuda2_device_cast_unsafe(base_device);
  return device->cu_context;
}

const iree_hal_cuda2_dynamic_symbols_t* iree_hal_cuda2_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device =
      iree_hal_cuda2_device_cast_unsafe(base_device);
  return device->cuda_symbols;
}

static void iree_hal_cuda2_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  const iree_hal_cuda2_dynamic_symbols_t* symbols = device->cuda_symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Destroy the pending workload queue.
  iree_hal_cuda2_pending_queue_actions_destroy(
      (iree_hal_resource_t*)device->pending_queue_actions);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // Destroy memory pools that hold on to reserved memory.
  iree_hal_cuda2_memory_pools_deinitialize(&device->memory_pools);

  iree_hal_cuda2_tracing_context_free(device->tracing_context);

  // Destroy various pools for synchronization.
  if (device->timepoint_pool) {
    iree_hal_cuda2_timepoint_pool_free(device->timepoint_pool);
  }
  if (device->device_event_pool) {
    iree_hal_cuda2_event_pool_release(device->device_event_pool);
  }
  if (device->host_event_pool) iree_event_pool_free(device->host_event_pool);

  IREE_CUDA_IGNORE_ERROR(symbols, cuStreamDestroy(device->dispatch_cu_stream));
  IREE_CUDA_IGNORE_ERROR(symbols, cuStreamDestroy(device->callback_cu_stream));

  IREE_CUDA_IGNORE_ERROR(symbols, cuDevicePrimaryCtxRelease(device->cu_device));

  iree_arena_block_pool_deinitialize(&device->block_pool);

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_cuda2_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_cuda2_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t* iree_hal_cuda2_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_cuda2_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_cuda2_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_cuda2_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  if (device->supports_memory_pools) {
    IREE_RETURN_IF_ERROR(iree_hal_cuda2_memory_pools_trim(
        &device->memory_pools, &device->params.memory_pools));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_device_query_attribute(
    iree_hal_cuda2_device_t* device, CUdevice_attribute attribute,
    int64_t* out_value) {
  int value = 0;
  IREE_CUDA_RETURN_IF_ERROR(
      device->cuda_symbols,
      cuDeviceGetAttribute(&value, attribute, device->cu_device),
      "cuDeviceGetAttribute");
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    *out_value = iree_string_view_equal(key, IREE_SV("cuda-nvptx-fb")) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category, IREE_SV("cuda.device"))) {
    if (iree_string_view_equal(key, IREE_SV("compute_capability_major"))) {
      return iree_hal_cuda2_device_query_attribute(
          device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, out_value);
    } else if (iree_string_view_equal(key,
                                      IREE_SV("compute_capability_minor"))) {
      return iree_hal_cuda2_device_query_attribute(
          device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, out_value);
    }
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_cuda2_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  if (!device->nccl_symbols || !device->nccl_symbols->dylib) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL runtime library version %d.%d and greater not available; "
        "ensure installed and the shared library (nccl.dll/libnccl.so) "
        "is on your PATH/LD_LIBRARY_PATH.",
        NCCL_MAJOR, NCCL_MINOR);
  }

  // Today we only allow a single logical device per channel.
  // We could multiplex channels but it'd be better to surface that to the
  // compiler so that it can emit the right rank math.
  int requested_count = iree_math_count_ones_u64(queue_affinity);
  // TODO(#12206): properly assign affinity in the compiler.
  if (requested_count != 64 && requested_count != 1) {
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
    IREE_RETURN_IF_ERROR(
        iree_hal_channel_provider_query_default_rank_and_count(
            device->channel_provider, &params.rank, &params.count),
        "querying default collective group rank and count");
  }

  // An ID is required to initialize NCCL. On the root it'll be the local ID and
  // on all other participants it'll be the root ID.
  iree_hal_cuda2_nccl_id_t id;
  memset(&id, 0, sizeof(id));
  if (iree_const_byte_span_is_empty(params.id)) {
    // User wants the default ID.
    if (!device->channel_provider) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "default collective channel ID requested but no channel provider has "
          "been set on the device to provide it");
    }
    if (params.rank == 0) {
      // Bootstrap NCCL to get the root ID.
      IREE_RETURN_IF_ERROR(
          iree_hal_cuda2_nccl_get_unique_id(device->nccl_symbols, &id),
          "bootstrapping NCCL root");
    }
    // Exchange NCCL ID with all participants.
    IREE_RETURN_IF_ERROR(iree_hal_channel_provider_exchange_default_id(
                             device->channel_provider,
                             iree_make_byte_span((void*)&id, sizeof(id))),
                         "exchanging NCCL ID with other participants");
  } else if (params.id.data_length != IREE_ARRAYSIZE(id.data)) {
    // User provided something but it's not what we expect.
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "NCCL ID must be %zu bytes matching the "
                            "ncclUniqueId struct but caller provided %zu bytes",
                            IREE_ARRAYSIZE(id.data), sizeof(id));
  } else {
    // User provided the ID - we treat it as opaque here and let NCCL validate.
    memcpy(id.data, params.id.data, IREE_ARRAYSIZE(id.data));
  }

  if (iree_hal_cuda2_nccl_id_is_empty(&id)) {
    // TODO: maybe this is ok? a localhost alias or something?
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no default NCCL ID specified (all zeros)");
  }

  // TODO: when we support multiple logical devices we'll want to pass in the
  // context of the device mapped to the queue_affinity. For now since this
  // implementation only supports one device we pass in the only one we have.
  return iree_hal_cuda2_nccl_channel_create(
      device->cuda_symbols, device->nccl_symbols, &id, params.rank,
      params.count, device->host_allocator, out_channel);
}

iree_status_t iree_hal_cuda2_device_create_stream_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return iree_hal_cuda2_stream_command_buffer_create(
      base_device, device->cuda_symbols, device->nccl_symbols,
      device->tracing_context, mode, command_categories, binding_capacity,
      device->dispatch_cu_stream, &device->block_pool, device->host_allocator,
      out_command_buffer);
}

static iree_status_t iree_hal_cuda2_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);

  switch (device->params.command_buffer_mode) {
    case IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH:
      return iree_hal_cuda2_graph_command_buffer_create(
          base_device, device->cuda_symbols, device->cu_context, mode,
          command_categories, queue_affinity, binding_capacity,
          &device->block_pool, device->host_allocator, out_command_buffer);
    case IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM:
      return iree_hal_deferred_command_buffer_create(
          base_device, mode, command_categories, binding_capacity,
          &device->block_pool, iree_hal_device_host_allocator(base_device),
          out_command_buffer);
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid command buffer mode");
  }
}

static iree_status_t iree_hal_cuda2_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return iree_hal_cuda2_descriptor_set_layout_create(
      flags, binding_count, bindings, device->host_allocator,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_cuda2_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "event not yet implmeneted");
}

static iree_status_t iree_hal_cuda2_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return iree_hal_cuda2_nop_executable_cache_create(
      identifier, device->cuda_symbols, device->cu_device,
      device->host_allocator, out_executable_cache);
}

static iree_status_t iree_hal_cuda2_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
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

static iree_status_t iree_hal_cuda2_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return iree_hal_cuda2_pipeline_layout_create(
      set_layout_count, set_layouts, push_constants, device->host_allocator,
      out_pipeline_layout);
}

static iree_status_t iree_hal_cuda2_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  return iree_hal_cuda2_event_semaphore_create(
      &device->event_impl_symtable, (void*)device->cuda_symbols, initial_value,
      device->timepoint_pool, device->pending_queue_actions,
      device->host_allocator, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_cuda2_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement CUDA semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in CUDA to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_cuda2_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The CUDA HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Allocate from the pool; likely to fail in cases of virtual memory
  // exhaustion but the error may be deferred until a later synchronization.
  // If pools are not supported we allocate a buffer as normal from whatever
  // allocator is set on the device.
  iree_status_t status = iree_ok_status();
  if (device->supports_memory_pools &&
      !iree_all_bits_set(params.type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    status = iree_hal_cuda2_memory_pools_alloca(
        &device->memory_pools, device->dispatch_cu_stream, pool, params,
        allocation_size, out_buffer);
  } else {
    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(base_device), params, allocation_size,
        out_buffer);
  }

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in CUDA to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_cuda2_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The CUDA HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Schedule the buffer deallocation if we got it from a pool and otherwise
  // drop it on the floor and let it be freed when the buffer is released.
  iree_status_t status = iree_ok_status();
  if (device->supports_memory_pools) {
    status = iree_hal_cuda2_memory_pools_dealloca(
        &device->memory_pools, device->dispatch_cu_stream, buffer);
  }

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list);
  }
  return status;
}

static iree_status_t iree_hal_cuda2_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_read_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_file, source_offset, target_buffer, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_cuda2_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, uint32_t flags) {
  // TODO: expose streaming chunk count/size options.
  iree_status_t loop_status = iree_ok_status();
  iree_hal_file_transfer_options_t options = {
      .loop = iree_loop_inline(&loop_status),
      .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
      .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_write_streaming(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
      source_buffer, source_offset, target_file, target_offset, length, flags,
      options));
  return loop_status;
}

static iree_status_t iree_hal_cuda2_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_cuda2_pending_queue_actions_enqueue_execution(
      base_device, (iree_hal_stream_impl_t)device->dispatch_cu_stream,
      (iree_hal_stream_impl_t)device->callback_cu_stream,
      device->pending_queue_actions, wait_semaphore_list, signal_semaphore_list,
      command_buffer_count, command_buffers);
  if (iree_status_is_ok(status)) {
    // Try to advance the pending workload queue.
    status = iree_hal_cuda2_pending_queue_actions_issue(
        device->pending_queue_actions);
  }

  iree_hal_cuda2_tracing_context_collect(device->tracing_context);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda2_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_cuda2_device_t* device = iree_hal_cuda2_device_cast(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);
  // Try to advance the pending workload queue.
  iree_status_t status =
      iree_hal_cuda2_pending_queue_actions_issue(device->pending_queue_actions);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_cuda2_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "waiting multiple semaphores not yet implemented");
}

static iree_status_t iree_hal_cuda2_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  // We could hook in to CUPTI here or use the much simpler cuProfilerStart API.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda2_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_cuda2_device_vtable = {
    .destroy = iree_hal_cuda2_device_destroy,
    .id = iree_hal_cuda2_device_id,
    .host_allocator = iree_hal_cuda2_device_host_allocator,
    .device_allocator = iree_hal_cuda2_device_allocator,
    .replace_device_allocator = iree_hal_cuda2_replace_device_allocator,
    .replace_channel_provider = iree_hal_cuda2_replace_channel_provider,
    .trim = iree_hal_cuda2_device_trim,
    .query_i64 = iree_hal_cuda2_device_query_i64,
    .create_channel = iree_hal_cuda2_device_create_channel,
    .create_command_buffer = iree_hal_cuda2_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_cuda2_device_create_descriptor_set_layout,
    .create_event = iree_hal_cuda2_device_create_event,
    .create_executable_cache = iree_hal_cuda2_device_create_executable_cache,
    .import_file = iree_hal_cuda2_device_import_file,
    .create_pipeline_layout = iree_hal_cuda2_device_create_pipeline_layout,
    .create_semaphore = iree_hal_cuda2_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_cuda2_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_cuda2_device_queue_alloca,
    .queue_dealloca = iree_hal_cuda2_device_queue_dealloca,
    .queue_read = iree_hal_cuda2_device_queue_read,
    .queue_write = iree_hal_cuda2_device_queue_write,
    .queue_execute = iree_hal_cuda2_device_queue_execute,
    .queue_flush = iree_hal_cuda2_device_queue_flush,
    .wait_semaphores = iree_hal_cuda2_device_wait_semaphores,
    .profiling_begin = iree_hal_cuda2_device_profiling_begin,
    .profiling_flush = iree_hal_cuda2_device_profiling_flush,
    .profiling_end = iree_hal_cuda2_device_profiling_end,
};
