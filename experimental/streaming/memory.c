// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/streaming/internal.h"
#include "experimental/streaming/util/buffer_table.h"

//===----------------------------------------------------------------------===//
// Memory management
//===----------------------------------------------------------------------===//

// Wraps a HAL buffer in a stream buffer and caches information.
static iree_status_t iree_hal_streaming_buffer_wrap(
    iree_hal_streaming_context_t* context, iree_hal_buffer_t* buffer,
    int memory_type, iree_hal_streaming_buffer_t** out_wrapper) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_wrapper);
  *out_wrapper = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(context->host_allocator, sizeof(*wrapper),
                                (void**)&wrapper));

  // Initialize wrapper.
  wrapper->buffer = buffer;
  iree_hal_buffer_retain(buffer);
  wrapper->context = context;
  iree_hal_streaming_context_retain(context);
  wrapper->memory_type = memory_type;
  wrapper->host_register_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;
  wrapper->ipc_handle = NULL;
  wrapper->size = iree_hal_buffer_byte_length(buffer);

  // Initialize unified memory attributes.
  wrapper->read_mostly_hint = false;
  wrapper->preferred_location = -2;      // Unspecified initially.
  wrapper->last_prefetch_location = -2;  // Never prefetched.

  iree_hal_external_buffer_t device_ptr;
  iree_status_t status = iree_hal_allocator_export_buffer(
      context->device_allocator, buffer,
      IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &device_ptr);
  if (iree_status_is_ok(status)) {
    wrapper->device_ptr =
        (iree_hal_streaming_deviceptr_t)device_ptr.handle.device_allocation.ptr;
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_export_buffer(
        context->device_allocator, buffer,
        IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE, &device_ptr);
  }
  if (iree_status_is_ok(status)) {
    wrapper->host_ptr = (void*)device_ptr.handle.host_allocation.ptr;
  }

  if (iree_status_is_ok(status)) {
    // Register buffer in context's mapping table.
    status =
        iree_hal_streaming_buffer_table_insert(context->buffer_table, wrapper);
  }

  if (iree_status_is_ok(status)) {
    *out_wrapper = wrapper;
  } else {
    // Clean up on failure.
    iree_hal_buffer_release(wrapper->buffer);
    iree_hal_streaming_context_release(wrapper->context);
    iree_allocator_free(context->host_allocator, wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Frees a buffer wrapper and releases the underlying buffer.
static void iree_hal_streaming_buffer_free(
    iree_hal_streaming_buffer_t* buffer) {
  if (!buffer) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  const iree_allocator_t host_allocator = buffer->context->host_allocator;
  iree_hal_buffer_release(buffer->buffer);
  iree_hal_streaming_context_release(buffer->context);
  iree_allocator_free(host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

iree_hal_streaming_deviceptr_t iree_hal_streaming_buffer_device_pointer(
    iree_hal_streaming_buffer_t* buffer) {
  return buffer ? buffer->device_ptr : 0;
}

iree_status_t iree_hal_streaming_memory_lookup(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_buffer_table_lookup(
      context->buffer_table, device_ptr, &out_ref->buffer));
  // TODO(benvanik): make the buffer table return a ref so we can hide this?
  if (IREE_LIKELY(out_ref->buffer->device_ptr <= device_ptr &&
                  device_ptr <
                      out_ref->buffer->device_ptr + out_ref->buffer->size)) {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->device_ptr;
  } else {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->host_ptr;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_lookup_range(
    iree_hal_streaming_context_t* context,
    iree_hal_streaming_deviceptr_t device_ptr, iree_device_size_t size,
    iree_hal_streaming_buffer_ref_t* out_ref) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_ref);
  memset(out_ref, 0, sizeof(*out_ref));
  IREE_RETURN_IF_ERROR(iree_hal_streaming_buffer_table_lookup_range(
      context->buffer_table, device_ptr, size, &out_ref->buffer));
  // TODO(benvanik): make the buffer table return a ref so we can hide this?
  if (IREE_LIKELY(out_ref->buffer->device_ptr <= device_ptr &&
                  device_ptr <
                      out_ref->buffer->device_ptr + out_ref->buffer->size)) {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->device_ptr;
  } else {
    out_ref->offset = (iree_device_size_t)device_ptr -
                      (iree_device_size_t)out_ref->buffer->host_ptr;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_device(
    iree_hal_streaming_context_t* context, iree_device_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 64,
  };

  // Allocate HAL buffer.
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(context->device_allocator, params,
                                             size, &buffer));

  // Wrap in stream buffer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)memory_type, &wrapper);

  // Release our reference (wrapper holds its own).
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_allocate_device_pitched(
    iree_hal_streaming_context_t* context, iree_device_size_t width_bytes,
    iree_device_size_t height, iree_device_size_t element_size_bytes,
    iree_device_size_t* out_pitch, iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_pitch);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_pitch = 0;
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Calculate pitch with 128-byte alignment for optimal memory access.
  // This is typical for both CUDA and HIP.
  const iree_device_size_t alignment = 128;
  iree_device_size_t pitch =
      (width_bytes + alignment - 1) / alignment * alignment;

  // For CUDA, element_size_bytes should be 4, 8, or 16 for coalesced access.
  // We don't enforce this but could warn if needed.

  // Calculate total size.
  iree_device_size_t total_size = pitch * height;

  // Allocate the buffer with the calculated total size.
  iree_hal_streaming_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_streaming_memory_allocate_device(
      context, total_size, 0, &buffer);

  if (iree_status_is_ok(status)) {
    *out_pitch = pitch;
    *out_buffer = buffer;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_free_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize context to ensure all operations using this memory complete.
  // This matches CUDA/HIP behavior where free operations are blocking.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_synchronize(context));

  // Look up buffer from device pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_buffer_table_lookup(context->buffer_table, ptr,
                                                 &wrapper));

  // Remove from mapping table.
  if (wrapper->context && wrapper->context->buffer_table) {
    iree_hal_streaming_buffer_table_remove(wrapper->context->buffer_table,
                                           wrapper->device_ptr);
  }

  // Update free memory tracking.
  if (wrapper->context && wrapper->context->device_entry) {
    wrapper->context->device_entry->free_memory += wrapper->size;
  }

  // Free wrapper.
  iree_hal_streaming_buffer_free(wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_allocate_host(
    iree_hal_streaming_context_t* context, iree_host_size_t size,
    iree_hal_streaming_memory_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_buffer = NULL;

  // Determine memory type based on flags.
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                  IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                                  IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  iree_hal_memory_type_t memory_type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;

  // Check for pinned memory flag (commonly used in CUDA/HIP).
  if (flags & 0x01) {  // hipHostMallocDefault or equivalent
    memory_type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  }

  iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 64,
  };

  // Allocate HAL buffer.
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_allocate_buffer(context->device_allocator, params,
                                             size, &buffer));

  // Wrap in stream buffer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, (int)memory_type, &wrapper);

  // Release our reference (wrapper holds its own).
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_free_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize context to ensure all operations using this memory complete.
  // This matches CUDA/HIP behavior where free operations are blocking.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_synchronize(context));

  // For host memory, we need to find the buffer by host pointer.
  // Since we store host pointers as device pointers for host allocations,
  // we can look it up directly.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_buffer_table_lookup(
              context->buffer_table, (iree_hal_streaming_deviceptr_t)ptr,
              &wrapper));

  // Remove from mapping table.
  if (wrapper->context && wrapper->context->buffer_table) {
    iree_hal_streaming_buffer_table_remove(wrapper->context->buffer_table,
                                           wrapper->device_ptr);
  }

  // Update free memory tracking.
  if (wrapper->context && wrapper->context->device_entry) {
    wrapper->context->device_entry->free_memory += wrapper->size;
  }

  // Free wrapper.
  iree_hal_streaming_buffer_free(wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_register_host(
    iree_hal_streaming_context_t* context, void* ptr, iree_host_size_t size,
    iree_hal_streaming_host_register_flags_t flags,
    iree_hal_streaming_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Import the existing host memory as a HAL buffer.
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
      .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
      .size = size,
      .handle =
          {
              .host_allocation =
                  {
                      .ptr = ptr,
                  },
          },
  };

  // Import with appropriate usage flags.
  iree_hal_buffer_usage_t usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                  IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                                  IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  iree_hal_memory_type_t memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  iree_hal_memory_access_t access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  iree_hal_buffer_params_t params = {
      .usage = usage,
      .access = access,
      .type = memory_type,
      .queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY,
      .min_alignment = 64,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_allocator_import_buffer(
              context->device_allocator, params, &external_buffer,
              /*release_callback=*/iree_hal_buffer_release_callback_null(),
              &buffer));

  // Wrap in stream buffer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_wrap(
      context, buffer, 2 /* host-registered */, &wrapper);

  // Release our reference (wrapper holds its own).
  iree_hal_buffer_release(buffer);

  if (iree_status_is_ok(status)) {
    // Store the host pointer explicitly.
    wrapper->host_ptr = ptr;
    // TODO(benvanik): that device_ptr == host_ptr may not be true! We probably
    // need to re-query the imported buffer to get the device pointer.
    wrapper->device_ptr = (iree_hal_streaming_deviceptr_t)ptr;
    wrapper->host_register_flags = flags;

    // Register in buffer table.
    status =
        iree_hal_streaming_buffer_table_insert(context->buffer_table, wrapper);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = wrapper;
  } else {
    iree_hal_streaming_buffer_free(wrapper);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_streaming_memory_unregister_host(
    iree_hal_streaming_context_t* context, void* ptr) {
  if (!ptr) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Synchronize context to ensure all operations using this memory complete.
  // This matches CUDA/HIP behavior where unregistration is blocking.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_context_synchronize(context));

  // Look up buffer from host pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_buffer_table_lookup(
              context->buffer_table, (iree_hal_streaming_deviceptr_t)ptr,
              &wrapper));

  // Remove from buffer table.
  if (wrapper->context && wrapper->context->buffer_table) {
    iree_hal_streaming_buffer_table_remove(wrapper->context->buffer_table,
                                           (iree_hal_streaming_deviceptr_t)ptr);
  }

  // Free wrapper (this will release the HAL buffer and context references).
  iree_hal_streaming_buffer_free(wrapper);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_address_range(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t ptr,
    iree_hal_streaming_deviceptr_t* out_base, iree_device_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_base);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_base = 0;
  *out_size = 0;

  // Look up buffer from pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_table_lookup(
      context->buffer_table, ptr, &wrapper);
  if (!iree_status_is_ok(status)) {
    return status;
  }

  // For registered host memory, the base is the registered pointer.
  // For device memory, the base is the allocated device pointer.
  if (wrapper->host_ptr) {
    *out_base = (iree_hal_streaming_deviceptr_t)wrapper->host_ptr;
  } else {
    *out_base = wrapper->device_ptr;
  }
  *out_size = wrapper->size;

  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_host_flags(
    iree_hal_streaming_context_t* context, void* ptr,
    iree_hal_streaming_host_register_flags_t* out_flags) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(ptr);
  IREE_ASSERT_ARGUMENT(out_flags);
  *out_flags = IREE_HAL_STREAMING_HOST_REGISTER_FLAG_DEFAULT;

  // Look up buffer from host pointer.
  iree_hal_streaming_buffer_t* wrapper = NULL;
  iree_status_t status = iree_hal_streaming_buffer_table_lookup(
      context->buffer_table, (iree_hal_streaming_deviceptr_t)ptr, &wrapper);
  if (iree_status_is_ok(status)) {
    *out_flags = wrapper->host_register_flags;
  }

  return status;
}

iree_status_t iree_hal_streaming_memory_memset(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(pattern);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);

  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Record fill command.
  iree_hal_buffer_ref_t target_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_fill_buffer(
              stream->command_buffer, target_ref, pattern, pattern_length,
              IREE_HAL_FILL_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memory_memcpy(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Record copy command.
  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                              src_buffer_ref, dst_buffer_ref,
                                              IREE_HAL_COPY_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_peer(
    iree_hal_streaming_context_t* dst_context,
    iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_context_t* src_context,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(dst_context);
  IREE_ASSERT_ARGUMENT(src_context);
  IREE_ASSERT_ARGUMENT(stream);
  IREE_TRACE_ZONE_BEGIN(z0);

  bool can_access = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_device_can_access_peer(src_context->device_ordinal,
                                                    dst_context->device_ordinal,
                                                    &can_access));
  if (!can_access) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_PERMISSION_DENIED,
                             "P2P access not supported between devices %" PRIhsz
                             " and %" PRIhsz,
                             src_context->device_ordinal,
                             dst_context->device_ordinal));
  }

  // Look up buffers from device pointers.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(dst_context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(src_context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  // Ensure command buffer is available.
  if (!stream->command_buffer) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                      iree_hal_streaming_stream_begin(stream));
  }

  // Record copy command.
  iree_hal_buffer_ref_t src_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
  iree_hal_buffer_ref_t dst_buffer_ref =
      iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_command_buffer_copy_buffer(stream->command_buffer,
                                              src_buffer_ref, dst_buffer_ref,
                                              IREE_HAL_COPY_FLAG_NONE));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Memory copy helper functions
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_streaming_memcpy_host_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    const void* src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up destination buffer from device pointer.
  iree_hal_streaming_buffer_ref_t dst_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
      "resolving `dst` buffer ref %p", (void*)dst);

  // For host-to-device, we can use the HAL update command if stream is NULL,
  // or copy command if stream is provided.
  if (!stream) {
    // Transfer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_device_transfer_h2d(
            context->device, src, dst_ref.buffer->buffer, dst_ref.offset, size,
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  } else {
    // Async: create a host buffer view and copy via command buffer.
    if (!stream->command_buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_stream_begin(stream));
    }

    // Import host memory as external buffer.
    iree_hal_external_buffer_t external_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = size,
        .handle =
            {
                .host_allocation =
                    {
                        .ptr = (void*)src,
                    },
            },
    };

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .access = IREE_HAL_MEMORY_ACCESS_READ,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        .queue_affinity = stream->queue_affinity,
        .min_alignment = 1,
    };

    iree_hal_buffer_t* src_buffer = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_allocator_import_buffer(
                context->device_allocator, params, &external_buffer,
                iree_hal_buffer_release_callback_null(), &src_buffer));

    // Record copy command.
    iree_hal_buffer_ref_t src_buffer_ref =
        iree_hal_make_buffer_ref(src_buffer, 0, size);
    iree_hal_buffer_ref_t dst_buffer_ref =
        iree_hal_streaming_convert_range_buffer_ref(dst_ref, size);
    iree_status_t status = iree_hal_command_buffer_copy_buffer(
        stream->command_buffer, src_buffer_ref, dst_buffer_ref,
        IREE_HAL_COPY_FLAG_NONE);

    iree_hal_buffer_release(src_buffer);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_host(
    iree_hal_streaming_context_t* context, void* dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up source buffer from device pointer.
  iree_hal_streaming_buffer_ref_t src_ref;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
      "resolving `src` buffer ref %p", (void*)src);

  // For device-to-host, we can use the HAL transfer command if stream is
  // NULL, or copy command if stream is provided.
  if (!stream) {
    // Transfer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_device_transfer_d2h(
            context->device, src_ref.buffer->buffer, src_ref.offset, dst, size,
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  } else {
    // Async: create a host buffer view and copy via command buffer.
    if (!stream->command_buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_streaming_stream_begin(stream));
    }

    // Import host memory as external buffer.
    iree_hal_external_buffer_t external_buffer = {
        .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION,
        .flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE,
        .size = size,
        .handle =
            {
                .host_allocation =
                    {
                        .ptr = dst,
                    },
            },
    };

    iree_hal_buffer_params_t params = {
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .access = IREE_HAL_MEMORY_ACCESS_WRITE,
        .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
        .queue_affinity = stream->queue_affinity,
        .min_alignment = 1,
    };

    iree_hal_buffer_t* dst_buffer = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_allocator_import_buffer(
                context->device_allocator, params, &external_buffer,
                iree_hal_buffer_release_callback_null(), &dst_buffer));

    // Record copy command.
    iree_hal_buffer_ref_t src_buffer_ref =
        iree_hal_streaming_convert_range_buffer_ref(src_ref, size);
    iree_hal_buffer_ref_t dst_buffer_ref =
        iree_hal_make_buffer_ref(dst_buffer, 0, size);
    iree_status_t status = iree_hal_command_buffer_copy_buffer(
        stream->command_buffer, src_buffer_ref, dst_buffer_ref,
        IREE_HAL_COPY_FLAG_NONE);

    iree_hal_buffer_release(dst_buffer);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_streaming_memcpy_device_to_device(
    iree_hal_streaming_context_t* context, iree_hal_streaming_deviceptr_t dst,
    iree_hal_streaming_deviceptr_t src, iree_device_size_t size,
    iree_hal_streaming_stream_t* stream) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(dst);
  IREE_ASSERT_ARGUMENT(src);
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!stream) {
    // Look up buffers from device pointers.
    iree_hal_streaming_buffer_ref_t dst_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, dst, &dst_ref),
        "resolving `dst` buffer ref %p", (void*)dst);
    iree_hal_streaming_buffer_ref_t src_ref;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_lookup(context, src, &src_ref),
        "resolving `src` buffer ref %p", (void*)src);

    // Transfer.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_device_transfer_d2d(
            context->device, src_ref.buffer->buffer, src_ref.offset,
            dst_ref.buffer->buffer, dst_ref.offset, size,
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  } else {
    // Device-to-device copy is the same as memcpy with offset 0.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_streaming_memory_memcpy(context, dst, src, size, stream));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
