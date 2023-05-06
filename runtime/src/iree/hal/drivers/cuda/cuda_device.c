// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/cuda_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/base/tracing.h"
#include "iree/hal/drivers/cuda/context_wrapper.h"
#include "iree/hal/drivers/cuda/cuda_allocator.h"
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#include "iree/hal/drivers/cuda/cuda_event.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"
#include "iree/hal/drivers/cuda/event_semaphore.h"
#include "iree/hal/drivers/cuda/graph_command_buffer.h"
#include "iree/hal/drivers/cuda/memory_pools.h"
#include "iree/hal/drivers/cuda/nccl_channel.h"
#include "iree/hal/drivers/cuda/nop_executable_cache.h"
#include "iree/hal/drivers/cuda/pipeline_layout.h"
#include "iree/hal/drivers/cuda/status_util.h"
#include "iree/hal/drivers/cuda/stream_command_buffer.h"
#include "iree/hal/drivers/cuda/tracing.h"
#include "iree/hal/utils/buffer_transfer.h"
#include "iree/hal/utils/deferred_command_buffer.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cuda_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the CUDA symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  // Parameters used to control device behavior.
  iree_hal_cuda_device_params_t params;

  CUdevice device;

  // TODO: support multiple streams.
  CUstream stream;
  iree_hal_cuda_context_wrapper_t context_wrapper;
  iree_hal_cuda_tracing_context_t* tracing_context;

  iree_hal_cuda_memory_pools_t memory_pools;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Cache of the direct stream command buffer initialized when in stream mode.
  // TODO: have one cached per stream once there are multiple streams.
  iree_hal_command_buffer_t* stream_command_buffer;
} iree_hal_cuda_device_t;

static const iree_hal_device_vtable_t iree_hal_cuda_device_vtable;

static iree_hal_cuda_device_t* iree_hal_cuda_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_device_vtable);
  return (iree_hal_cuda_device_t*)base_value;
}

static iree_hal_cuda_device_t* iree_hal_cuda_device_cast_unsafe(
    iree_hal_device_t* base_value) {
  return (iree_hal_cuda_device_t*)base_value;
}

IREE_API_EXPORT void iree_hal_cuda_device_params_initialize(
    iree_hal_cuda_device_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->arena_block_size = 32 * 1024;
  out_params->queue_count = 1;
  out_params->command_buffer_mode = IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH;
  out_params->allow_inline_execution = false;
  out_params->stream_tracing = false;
}

static iree_status_t iree_hal_cuda_device_check_params(
    const iree_hal_cuda_device_params_t* params) {
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

static iree_status_t iree_hal_cuda_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* params, CUdevice cu_device,
    CUstream stream, CUcontext context, iree_hal_cuda_dynamic_symbols_t* syms,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_cuda_device_t* device = NULL;
  iree_host_size_t total_size = iree_sizeof_struct(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_cuda_device_vtable, &device->resource);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char*)device + iree_sizeof_struct(*device));
  device->params = *params;
  device->device = cu_device;
  device->stream = stream;
  device->context_wrapper.cu_device = cu_device;
  device->context_wrapper.cu_context = context;
  device->context_wrapper.host_allocator = host_allocator;
  iree_arena_block_pool_initialize(params->arena_block_size, host_allocator,
                                   &device->block_pool);
  device->context_wrapper.syms = syms;

  // Enable tracing for the (currently only) stream - no-op if disabled.
  iree_status_t status = iree_ok_status();
  if (device->params.stream_tracing) {
    status = iree_hal_cuda_tracing_context_allocate(
        &device->context_wrapper, device->identifier, stream,
        &device->block_pool, host_allocator, &device->tracing_context);
  }

  // Create memory pools first so that we can share them with the allocator.
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_memory_pools_initialize(
        &device->context_wrapper, &params->memory_pools, &device->memory_pools);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_allocator_create(
        (iree_hal_device_t*)device, &device->context_wrapper, cu_device, stream,
        &device->memory_pools, &device->device_allocator);
  }

  if (iree_status_is_ok(status) &&
      params->command_buffer_mode == IREE_HAL_CUDA_COMMAND_BUFFER_MODE_STREAM) {
    status = iree_hal_cuda_stream_command_buffer_create(
        (iree_hal_device_t*)device, &device->context_wrapper,
        device->tracing_context,
        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
            IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
        IREE_HAL_COMMAND_CATEGORY_ANY, /*binding_capacity=*/0, device->stream,
        &device->block_pool, &device->stream_command_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_cuda_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* params,
    iree_hal_cuda_dynamic_symbols_t* syms, CUdevice device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_hal_cuda_device_check_params(params));
  CUcontext context;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      CU_RESULT_TO_STATUS(syms, cuDevicePrimaryCtxRetain(&context, device)));
  iree_status_t status = CU_RESULT_TO_STATUS(syms, cuCtxSetCurrent(context));
  CUstream stream;
  if (iree_status_is_ok(status)) {
    status = CU_RESULT_TO_STATUS(
        syms, cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_device_create_internal(driver, identifier, params,
                                                  device, stream, context, syms,
                                                  host_allocator, out_device);
  }
  if (!iree_status_is_ok(status)) {
    if (stream) {
      syms->cuStreamDestroy(stream);
    }
    syms->cuDevicePrimaryCtxRelease(device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

CUcontext iree_hal_cuda_device_context(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device =
      iree_hal_cuda_device_cast_unsafe(base_device);
  return device->context_wrapper.cu_context;
}

iree_hal_cuda_dynamic_symbols_t* iree_hal_cuda_device_dynamic_symbols(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device =
      iree_hal_cuda_device_cast_unsafe(base_device);
  return device->context_wrapper.syms;
}

static void iree_hal_cuda_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_command_buffer_release(device->stream_command_buffer);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  // Destroy memory pools that hold on to reserved memory.
  iree_hal_cuda_memory_pools_deinitialize(&device->memory_pools);

  // TODO: support multiple streams.
  iree_hal_cuda_tracing_context_free(device->tracing_context);
  CUDA_IGNORE_ERROR(device->context_wrapper.syms,
                    cuStreamDestroy(device->stream));

  iree_arena_block_pool_deinitialize(&device->block_pool);

  CUDA_IGNORE_ERROR(device->context_wrapper.syms,
                    cuDevicePrimaryCtxRelease(device->device));

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_cuda_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_cuda_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->context_wrapper.host_allocator;
}

static iree_hal_allocator_t* iree_hal_cuda_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_cuda_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_cuda_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_cuda_device_trim(iree_hal_device_t* base_device) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  IREE_RETURN_IF_ERROR(iree_hal_allocator_trim(device->device_allocator));
  // TODO: move to memory pool manager.
  CUDA_RETURN_IF_ERROR(
      device->context_wrapper.syms,
      cuMemPoolTrimTo(
          device->memory_pools.device_local,
          device->params.memory_pools.device_local.minimum_capacity),
      "cuMemPoolTrimTo");
  CUDA_RETURN_IF_ERROR(
      device->context_wrapper.syms,
      cuMemPoolTrimTo(device->memory_pools.other,
                      device->params.memory_pools.other.minimum_capacity),
      "cuMemPoolTrimTo");
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  // iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, iree_make_cstring_view("cuda-nvptx-fb"))
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_cuda_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  if (!device->context_wrapper.syms->nccl_library) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "NCCL runtime library (%d.%d.%d) not available; ensure installed and "
        "the shared library is on your PATH/LD_LIBRARY_PATH "
        "(nccl.dll/libnccl.so)",
        NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
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
  iree_hal_cuda_nccl_id_t id;
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
      IREE_RETURN_IF_ERROR(iree_hal_cuda_nccl_get_unique_id_from_context(
                               &device->context_wrapper, &id),
                           "bootstrapping NCCL root");
    }
    // Exchange NCCL ID with all participants.
    IREE_RETURN_IF_ERROR(iree_hal_channel_provider_exchange_default_id(
                             device->channel_provider,
                             iree_make_byte_span((void*)&id, sizeof(id))),
                         "exchanging NCCL ID with other participants");
  } else if (params.id.data_length != IREE_ARRAYSIZE(id.data)) {
    // User provided something but it's not what we expect.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "NCCL ID must be %" PRIhsz
        " bytes matching the ncclUniqueId struct but caller provided %" PRIhsz
        " bytes",
        IREE_ARRAYSIZE(id.data), sizeof(id));
  } else {
    // User provided the ID - we treat it as opaque here and let NCCL validate.
    memcpy(id.data, params.id.data, IREE_ARRAYSIZE(id.data));
  }

  if (iree_hal_cuda_nccl_id_is_empty(&id)) {
    // TODO: maybe this is ok? a localhost alias or something?
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no default NCCL ID specified (all zeros)");
  }

  // TODO: when we support multiple logical devices we'll want to pass in the
  // context of the device mapped to the queue_affinity. For now since this
  // implementation only supports one device we pass in the only one we have.
  return iree_hal_cuda_nccl_channel_create(
      &device->context_wrapper, &id, params.rank, params.count, out_channel);
}

static iree_status_t iree_hal_cuda_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  if (device->params.allow_inline_execution &&
      iree_all_bits_set(mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    // The caller has indicated the command buffer can be executed as it is
    // recorded, implying that the command buffer cannot be reused and doesn't
    // need to be persisted. This lets us lower the execution delay as we can
    // directly route commands to a CUDA stream and let it eagerly flush.
    return iree_hal_cuda_stream_command_buffer_create(
        base_device, &device->context_wrapper, device->tracing_context, mode,
        command_categories, binding_capacity, device->stream,
        &device->block_pool, out_command_buffer);
  }
  switch (device->params.command_buffer_mode) {
    case IREE_HAL_CUDA_COMMAND_BUFFER_MODE_GRAPH:
      return iree_hal_cuda_graph_command_buffer_create(
          base_device, &device->context_wrapper, mode, command_categories,
          queue_affinity, binding_capacity, &device->block_pool,
          out_command_buffer);
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

static iree_status_t iree_hal_cuda_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_descriptor_set_layout_create(
      &device->context_wrapper, flags, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_cuda_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_event_create(&device->context_wrapper, out_event);
}

static iree_status_t iree_hal_cuda_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_nop_executable_cache_create(
      &device->context_wrapper, identifier, out_executable_cache);
}

static iree_status_t iree_hal_cuda_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_pipeline_layout_create(
      &device->context_wrapper, set_layout_count, set_layouts, push_constants,
      out_pipeline_layout);
}

static iree_status_t iree_hal_cuda_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);
  return iree_hal_cuda_semaphore_create(&device->context_wrapper, initial_value,
                                        out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_cuda_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement CUDA semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in CUDA to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_cuda_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The CUDA HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Allocate from the pool; likely to fail in cases of virtual memory
  // exhaustion but the error may be deferred until a later synchronization.
  iree_status_t status = iree_hal_cuda_memory_pools_alloca(
      &device->memory_pools, device->stream, pool, params, allocation_size,
      out_buffer);

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  }
  return status;
}

// TODO: implement multiple streams; today we only have one and queue_affinity
//       is ignored.
// TODO: implement proper semaphores in CUDA to ensure ordering and avoid
//       the barrier here.
static iree_status_t iree_hal_cuda_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);

  // NOTE: block on the semaphores here; we could avoid this by properly
  // sequencing device work with semaphores. The CUDA HAL is not currently
  // asynchronous.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));

  // Schedule the buffer deallocation.
  iree_status_t status = iree_hal_cuda_memory_pools_dealloca(
      &device->memory_pools, device->stream, buffer);

  // Only signal if not returning a synchronous error - synchronous failure
  // indicates that the stream is unchanged (it's not really since we waited
  // above, but we at least won't deadlock like this).
  if (iree_status_is_ok(status)) {
    IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  }
  return status;
}

static iree_status_t iree_hal_cuda_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_cuda_device_t* device = iree_hal_cuda_device_cast(base_device);

  // TODO(benvanik): trace around the entire submission.

  for (iree_host_size_t i = 0; i < command_buffer_count; i++) {
    iree_hal_command_buffer_t* command_buffer = command_buffers[i];
    if (iree_hal_cuda_stream_command_buffer_isa(command_buffer)) {
      // Nothing to do for an inline command buffer; all the work has already
      // been submitted. When we support semaphores we'll still need to signal
      // their completion but do not have to worry about any waits: if there
      // were waits we wouldn't have been able to execute inline!
    } else if (iree_hal_cuda_graph_command_buffer_isa(command_buffer)) {
      CUgraphExec exec =
          iree_hal_cuda_graph_command_buffer_handle(command_buffers[i]);
      CUDA_RETURN_IF_ERROR(device->context_wrapper.syms,
                           cuGraphLaunch(exec, device->stream),
                           "cuGraphLaunch");
    } else {
      IREE_RETURN_IF_ERROR(iree_hal_deferred_command_buffer_apply(
          command_buffers[i], device->stream_command_buffer,
          iree_hal_buffer_binding_table_empty()));
    }
  }

  // TODO(thomasraoux): implement semaphores - for now this conservatively
  // synchronizes after every submit.
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "cuStreamSynchronize");
  CUDA_RETURN_IF_ERROR(device->context_wrapper.syms,
                       cuStreamSynchronize(device->stream),
                       "cuStreamSynchronize");
  iree_hal_cuda_tracing_context_collect(device->tracing_context);
  IREE_TRACE_ZONE_END(z0);

  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");
}

static iree_status_t iree_hal_cuda_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  // We could hook in to CUPTI here or use the much simpler cuProfilerStart API.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_cuda_device_vtable = {
    .destroy = iree_hal_cuda_device_destroy,
    .id = iree_hal_cuda_device_id,
    .host_allocator = iree_hal_cuda_device_host_allocator,
    .device_allocator = iree_hal_cuda_device_allocator,
    .replace_device_allocator = iree_hal_cuda_replace_device_allocator,
    .replace_channel_provider = iree_hal_cuda_replace_channel_provider,
    .trim = iree_hal_cuda_device_trim,
    .query_i64 = iree_hal_cuda_device_query_i64,
    .create_channel = iree_hal_cuda_device_create_channel,
    .create_command_buffer = iree_hal_cuda_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_cuda_device_create_descriptor_set_layout,
    .create_event = iree_hal_cuda_device_create_event,
    .create_executable_cache = iree_hal_cuda_device_create_executable_cache,
    .create_pipeline_layout = iree_hal_cuda_device_create_pipeline_layout,
    .create_semaphore = iree_hal_cuda_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_cuda_device_query_semaphore_compatibility,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_alloca = iree_hal_cuda_device_queue_alloca,
    .queue_dealloca = iree_hal_cuda_device_queue_dealloca,
    .queue_execute = iree_hal_cuda_device_queue_execute,
    .queue_flush = iree_hal_cuda_device_queue_flush,
    .wait_semaphores = iree_hal_cuda_device_wait_semaphores,
    .profiling_begin = iree_hal_cuda_device_profiling_begin,
    .profiling_end = iree_hal_cuda_device_profiling_end,
};
