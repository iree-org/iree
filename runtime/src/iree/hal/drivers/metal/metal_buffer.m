// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/metal/metal_buffer.h"

#import <Metal/Metal.h>

#include "iree/base/api.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/metal/direct_allocator.h"
#include "iree/hal/drivers/metal/metal_device.h"

typedef struct iree_hal_metal_buffer_t {
  iree_hal_buffer_t base;
  id<MTLBuffer> buffer;
  // The command queue that we can use to issue commands to make buffer contents visible to CPU.
#if defined(IREE_PLATFORM_MACOS)
  id<MTLCommandQueue> queue;
#endif  // IREE_PLATFORM_MACOS
  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_metal_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_metal_buffer_vtable;

static iree_hal_metal_buffer_t* iree_hal_metal_buffer_cast(iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_buffer_vtable);
  return (iree_hal_metal_buffer_t*)base_value;
}

static const iree_hal_metal_buffer_t* iree_hal_metal_buffer_const_cast(
    const iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_metal_buffer_vtable);
  return (const iree_hal_metal_buffer_t*)base_value;
}

iree_status_t iree_hal_metal_buffer_wrap(
#if defined(IREE_PLATFORM_MACOS)
    id<MTLCommandQueue> queue,
#endif  // IREE_PLATFORM_MACOS
    id<MTLBuffer> metal_buffer, iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access, iree_hal_buffer_usage_t allowed_usage,
    iree_device_size_t allocation_size, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator = iree_hal_allocator_host_allocator(allocator);
  iree_hal_metal_buffer_t* buffer = NULL;
  iree_status_t status = iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(host_allocator, allocator, &buffer->base, allocation_size,
                               byte_offset, byte_length, memory_type, allowed_access, allowed_usage,
                               &iree_hal_metal_buffer_vtable, &buffer->base);
    buffer->buffer = [metal_buffer retain];  // +1
#if defined(IREE_PLATFORM_MACOS)
    buffer->queue = queue;
#endif  // IREE_PLATFORM_MACOS
    buffer->release_callback = release_callback;
    *out_buffer = &buffer->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_metal_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)iree_hal_buffer_allocation_size(base_buffer));

  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data, base_buffer);
  }
  [buffer->buffer release];  // -1
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_metal_buffer_is_external(const iree_hal_buffer_t* base_buffer) {
  const iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_const_cast(base_buffer);
  return buffer->release_callback.fn != NULL;
}

id<MTLBuffer> iree_hal_metal_buffer_handle(const iree_hal_buffer_t* base_buffer) {
  const iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_const_cast(base_buffer);
  return buffer->buffer;
}

static iree_status_t iree_hal_metal_buffer_invalidate_range(iree_hal_buffer_t* base_buffer,
                                                            iree_device_size_t local_byte_offset,
                                                            iree_device_size_t local_byte_length) {
  IREE_TRACE_ZONE_BEGIN(z0);
#if defined(IREE_PLATFORM_MACOS)
  // Special treatment for the MTLStorageManaged storage mode on macOS.
  // In order to synchronize the GPU modifications back to CPU, we need to record a command buffer
  // and commit to the queue.
  iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_cast(base_buffer);
  if (buffer->buffer.storageMode == MTLStorageModeManaged) {
    id<MTLCommandBuffer> command_buffer = [buffer->queue commandBuffer];

    id<MTLBlitCommandEncoder> blitCommandEncoder = [command_buffer blitCommandEncoder];
    [blitCommandEncoder synchronizeResource:buffer->buffer];
    [blitCommandEncoder endEncoding];

    __block dispatch_semaphore_t work_done = dispatch_semaphore_create(0);
    [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      dispatch_semaphore_signal(work_done);
    }];

    [command_buffer commit];

    intptr_t timed_out = dispatch_semaphore_wait(work_done, DISPATCH_TIME_FOREVER);
    (void)timed_out;
    dispatch_release(work_done);
  }
#endif  // IREE_PLATFORM_MACOS
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_metal_buffer_flush_range(iree_hal_buffer_t* base_buffer,
                                                       iree_device_size_t local_byte_offset,
                                                       iree_device_size_t local_byte_length) {
#if defined(IREE_PLATFORM_MACOS)
  // Special treatment for the MTLStorageManaged storage mode on macOS.
  iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_cast(base_buffer);
  if (buffer->buffer.storageMode == MTLStorageModeManaged) {
    [buffer->buffer didModifyRange:NSMakeRange(local_byte_offset, local_byte_length)];
  }
#endif  // IREE_PLATFORM_MACOS
  return iree_ok_status();
}

#if defined(IREE_PLATFORM_MACOS)
// Returns true if the given buffer should require "automatic" synchronization when mapping or
// unmapping ranges.
static inline bool iree_hal_metal_require_autosync_cpu_gpu(iree_hal_buffer_t* base_buffer,
                                                           id<MTLBuffer> metal_buffer) {
  return iree_any_bit_set(iree_hal_buffer_memory_type(base_buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_COHERENT) &&
         metal_buffer.storageMode == MTLStorageModeManaged;
}
#endif  // IREE_PLATFORM_MACOS

static iree_status_t iree_hal_metal_buffer_map_range(iree_hal_buffer_t* base_buffer,
                                                     iree_hal_mapping_mode_t mapping_mode,
                                                     iree_hal_memory_access_t memory_access,
                                                     iree_device_size_t local_byte_offset,
                                                     iree_device_size_t local_byte_length,
                                                     iree_hal_buffer_mapping_t* mapping) {
  iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_cast(base_buffer);

  // TODO(benvanik): add upload/download for unmapped buffers.
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer), mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
                                                      ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
                                                      : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED));

  void* host_ptr = buffer->buffer.contents;
  IREE_ASSERT(host_ptr != NULL);  // Should be guaranteed by previous checks.
  uint8_t* data_ptr = (uint8_t*)host_ptr + local_byte_offset;
  // If we mapped for discard scribble over the bytes. This is not a mandated behavior but it will
  // make debugging issues easier. Alternatively for heap buffers we could reallocate them such that
  // ASAN yells, but that would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  iree_status_t status = iree_ok_status();
#if defined(IREE_PLATFORM_MACOS)
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_READ) &&
      iree_hal_metal_require_autosync_cpu_gpu(base_buffer, buffer->buffer)) {
    status =
        iree_hal_metal_buffer_invalidate_range(base_buffer, local_byte_offset, local_byte_length);
  }
#endif  // IREE_PLATFORM_MACOS
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
  return status;
}

static iree_status_t iree_hal_metal_buffer_unmap_range(iree_hal_buffer_t* base_buffer,
                                                       iree_device_size_t local_byte_offset,
                                                       iree_device_size_t local_byte_length,
                                                       iree_hal_buffer_mapping_t* mapping) {
#if defined(IREE_PLATFORM_MACOS)
  iree_hal_metal_buffer_t* buffer = iree_hal_metal_buffer_cast(base_buffer);
  if (iree_hal_metal_require_autosync_cpu_gpu(base_buffer, buffer->buffer)) {
    return iree_hal_metal_buffer_flush_range(base_buffer, local_byte_offset, local_byte_length);
  }
#endif  // IREE_PLATFORM_MACOS
  return iree_ok_status();
}

static const iree_hal_buffer_vtable_t iree_hal_metal_buffer_vtable = {
    .recycle = iree_hal_buffer_recycle,
    .destroy = iree_hal_metal_buffer_destroy,
    .map_range = iree_hal_metal_buffer_map_range,
    .unmap_range = iree_hal_metal_buffer_unmap_range,
    .invalidate_range = iree_hal_metal_buffer_invalidate_range,
    .flush_range = iree_hal_metal_buffer_flush_range,
};
