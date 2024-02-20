// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/rocm_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/rocm/context_wrapper.h"
#include "experimental/rocm/direct_command_buffer.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/event_semaphore.h"
#include "experimental/rocm/nop_executable_cache.h"
#include "experimental/rocm/pipeline_layout.h"
#include "experimental/rocm/rocm_allocator.h"
#include "experimental/rocm/rocm_event.h"
#include "experimental/rocm/status_util.h"
#include "experimental/rocm/tracing.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

//===----------------------------------------------------------------------===//
// iree_hal_rocm_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_rocm_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the ROCM symbols. We retain it for our lifetime
  // to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  hipDevice_t device;

  // TODO: support multiple streams.
  hipStream_t stream;
  iree_hal_rocm_tracing_context_t* tracing_context;
  iree_hal_rocm_context_wrapper_t context_wrapper;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;
} iree_hal_rocm_device_t;

static const iree_hal_device_vtable_t iree_hal_rocm_device_vtable;

static iree_hal_rocm_device_t* iree_hal_rocm_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_device_vtable);
  return (iree_hal_rocm_device_t*)base_value;
}

static void iree_hal_rocm_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);

  // Buffers may have been retaining collective resources.
  iree_hal_channel_provider_release(device->channel_provider);

  iree_hal_rocm_tracing_context_free(device->tracing_context);
  ROCM_IGNORE_ERROR(device->context_wrapper.syms,
                    hipStreamDestroy(device->stream));

  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_rocm_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    hipDevice_t rocm_device, hipStream_t stream, hipCtx_t context,
    iree_hal_rocm_dynamic_symbols_t* syms, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  iree_hal_rocm_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_rocm_device_vtable, &device->resource);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);
  device->device = rocm_device;
  device->stream = stream;
  device->context_wrapper.rocm_context = context;
  device->context_wrapper.rocm_device = rocm_device;
  device->context_wrapper.host_allocator = host_allocator;
  device->context_wrapper.syms = syms;
  // Enable tracing for the (currently only) stream - no-op if disabled.
  iree_status_t status = iree_hal_rocm_tracing_context_allocate(
      &device->context_wrapper, device->identifier, stream, &device->block_pool,
      host_allocator, &device->tracing_context);
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_allocator_create(&device->context_wrapper,
                                            &device->device_allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_rocm_device_create(iree_hal_driver_t* driver,
                                          iree_string_view_t identifier,
                                          iree_hal_rocm_dynamic_symbols_t* syms,
                                          hipDevice_t device,
                                          iree_allocator_t host_allocator,
                                          iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  hipCtx_t context;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      ROCM_RESULT_TO_STATUS(syms, hipDevicePrimaryCtxRetain(&context, device)));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(syms, hipCtxSetCurrent(context)));

  hipStream_t stream;
  iree_status_t status = ROCM_RESULT_TO_STATUS(
      syms, hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_device_create_internal(driver, identifier, device,
                                                  stream, context, syms,
                                                  host_allocator, out_device);
  }
  if (!iree_status_is_ok(status)) {
    if (stream) {
      syms->hipStreamDestroy(stream);
    }
    syms->hipDevicePrimaryCtxRelease(device);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_string_view_t iree_hal_rocm_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_rocm_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->context_wrapper.host_allocator;
}

static iree_hal_allocator_t* iree_hal_rocm_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return device->device_allocator;
}

static void iree_hal_rocm_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void iree_hal_rocm_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t iree_hal_rocm_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    *out_value =
        iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    return iree_ok_status();
  }

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, iree_make_cstring_view("rocm-hsaco-fb"))
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_rocm_device_trim(iree_hal_device_t* base_device) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_rocm_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not implemented");
}

static iree_status_t iree_hal_rocm_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_direct_command_buffer_create(
      base_device, &device->context_wrapper, device->tracing_context, mode,
      command_categories, queue_affinity, binding_capacity, &device->block_pool,
      out_command_buffer);
}

static iree_status_t iree_hal_rocm_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_descriptor_set_layout_create(
      &device->context_wrapper, flags, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_rocm_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_event_create(&device->context_wrapper, out_event);
}

static iree_status_t iree_hal_rocm_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_nop_executable_cache_create(
      &device->context_wrapper, identifier, out_executable_cache);
}

static iree_status_t iree_hal_rocm_device_import_file(
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

static iree_status_t iree_hal_rocm_device_create_pipeline_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_hal_pipeline_layout_t** out_pipeline_layout) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_pipeline_layout_create(
      &device->context_wrapper, set_layout_count, set_layouts, push_constants,
      out_pipeline_layout);
}

static iree_status_t iree_hal_rocm_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  return iree_hal_rocm_semaphore_create(&device->context_wrapper, initial_value,
                                        out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_rocm_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement ROCM semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_rocm_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_allocator_pool_t pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  // TODO: queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_wait(wait_semaphore_list,
                                                    iree_infinite_timeout()));
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_allocate_buffer(iree_hal_device_allocator(base_device),
                                         params, allocation_size, out_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer) {
  // TODO: queue-ordered allocations.
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_barrier(
      base_device, queue_affinity, wait_semaphore_list, signal_semaphore_list));
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_queue_read(
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

static iree_status_t iree_hal_rocm_device_queue_write(
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

static iree_status_t iree_hal_rocm_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_rocm_device_t* device = iree_hal_rocm_device_cast(base_device);
  // TODO(raikonenfnu): Once semaphore is implemented wait for semaphores
  // TODO(thomasraoux): implement semaphores - for now this conservatively
  // synchronizes after every submit.
  // TODO(raikonenfnu): currently run on default/null stream, when cmd buffer
  // stream work with device->stream, we'll change
  IREE_TRACE_ZONE_BEGIN_NAMED(z0, "hipStreamSynchronize");
  ROCM_RETURN_IF_ERROR(device->context_wrapper.syms, hipStreamSynchronize(0),
                       "hipStreamSynchronize");
  iree_hal_rocm_tracing_context_collect(device->tracing_context);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  // Currently unused; we flush as submissions are made.
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");
}

static iree_status_t iree_hal_rocm_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_profiling_flush(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static iree_status_t iree_hal_rocm_device_profiling_end(
    iree_hal_device_t* base_device) {
  // Unimplemented (and that's ok).
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_rocm_device_vtable = {
    .destroy = iree_hal_rocm_device_destroy,
    .id = iree_hal_rocm_device_id,
    .host_allocator = iree_hal_rocm_device_host_allocator,
    .device_allocator = iree_hal_rocm_device_allocator,
    .replace_device_allocator = iree_hal_rocm_replace_device_allocator,
    .replace_channel_provider = iree_hal_rocm_replace_channel_provider,
    .trim = iree_hal_rocm_device_trim,
    .query_i64 = iree_hal_rocm_device_query_i64,
    .create_channel = iree_hal_rocm_device_create_channel,
    .create_command_buffer = iree_hal_rocm_device_create_command_buffer,
    .create_descriptor_set_layout =
        iree_hal_rocm_device_create_descriptor_set_layout,
    .create_event = iree_hal_rocm_device_create_event,
    .create_executable_cache = iree_hal_rocm_device_create_executable_cache,
    .import_file = iree_hal_rocm_device_import_file,
    .create_pipeline_layout = iree_hal_rocm_device_create_pipeline_layout,
    .create_semaphore = iree_hal_rocm_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_rocm_device_query_semaphore_compatibility,
    .queue_alloca = iree_hal_rocm_device_queue_alloca,
    .queue_dealloca = iree_hal_rocm_device_queue_dealloca,
    .queue_read = iree_hal_rocm_device_queue_read,
    .queue_write = iree_hal_rocm_device_queue_write,
    .queue_execute = iree_hal_rocm_device_queue_execute,
    .queue_flush = iree_hal_rocm_device_queue_flush,
    .wait_semaphores = iree_hal_rocm_device_wait_semaphores,
    .profiling_begin = iree_hal_rocm_device_profiling_begin,
    .profiling_flush = iree_hal_rocm_device_profiling_flush,
    .profiling_end = iree_hal_rocm_device_profiling_end,
};
