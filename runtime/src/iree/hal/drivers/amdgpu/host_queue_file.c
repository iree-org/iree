// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_file.h"

#include <string.h>

#include "iree/async/operations/file.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_staging.h"
#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

typedef enum iree_hal_amdgpu_file_action_kind_e {
  IREE_HAL_AMDGPU_FILE_ACTION_READ,
  IREE_HAL_AMDGPU_FILE_ACTION_WRITE,
} iree_hal_amdgpu_file_action_kind_t;

static iree_hal_profile_queue_event_type_t
iree_hal_amdgpu_file_action_profile_event_type(
    iree_hal_amdgpu_file_action_kind_t kind) {
  return kind == IREE_HAL_AMDGPU_FILE_ACTION_READ
             ? IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ
             : IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE;
}

typedef struct iree_hal_amdgpu_file_action_state_t {
  // Resource header retained by the queue reclaim entry and async completion.
  iree_hal_resource_t resource;

  // Host allocator used for this state and cloned semaphore-list storage.
  iree_allocator_t host_allocator;

  // Proactor used for async file I/O. Borrowed from the logical device.
  iree_async_proactor_t* proactor;

  // Queue used to publish final signal semaphores after async file I/O.
  iree_hal_amdgpu_host_queue_t* queue;

  // Logical device retained while async file I/O is pending so |queue| storage
  // and its physical-device resources remain live.
  iree_hal_device_t* logical_device;

  // File being read or written. Retained while the action is pending.
  iree_hal_file_t* file;

  // Async file handle borrowed from |file|.
  iree_async_file_t* async_file;

  // Buffer being read into or written from. Retained while the action is
  // pending.
  iree_hal_buffer_t* buffer;

  // File byte offset for the next transfer.
  uint64_t file_offset;

  // Buffer byte offset for the mapped range.
  iree_device_size_t buffer_offset;

  // Total requested transfer length.
  iree_host_size_t requested_length;

  // Number of wait semaphores supplied to the queue_read/write operation.
  uint32_t profile_wait_count;

  // Total bytes transferred by completed async file operations.
  iree_host_size_t completed_length;

  // Direction of the file action.
  iree_hal_amdgpu_file_action_kind_t kind;

  // Scoped mapping of |buffer| used by async file operations.
  iree_hal_buffer_mapping_t mapping;

  // Cloned signal list published by a final queue barrier after file I/O.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Completion-thread retry queued when the final signal barrier is blocked by
  // temporary queue capacity pressure.
  iree_hal_amdgpu_host_queue_post_drain_action_t signal_capacity_retry;

  // Async read operation reused across partial completions.
  iree_async_file_read_operation_t read_op;

  // Async write operation reused across partial completions.
  iree_async_file_write_operation_t write_op;
} iree_hal_amdgpu_file_action_state_t;

static void iree_hal_amdgpu_file_action_state_destroy(
    iree_hal_resource_t* resource) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)resource;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    iree_hal_semaphore_list_free(state->signal_semaphore_list,
                                 state->host_allocator);
  }
  iree_hal_buffer_release(state->buffer);
  iree_hal_file_release(state->file);
  iree_hal_device_release(state->logical_device);
  iree_allocator_free(state->host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

static const iree_hal_resource_vtable_t
    iree_hal_amdgpu_file_action_state_vtable = {
        .destroy = iree_hal_amdgpu_file_action_state_destroy,
};

static iree_status_t iree_hal_amdgpu_host_queue_file_barrier(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  return queue->vtable->execute(
      queue, wait_semaphore_list, signal_semaphore_list,
      /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
      IREE_HAL_EXECUTE_FLAG_NONE);
}

static iree_status_t iree_hal_amdgpu_host_queue_cast_file_offset(
    uint64_t file_offset, iree_device_size_t length,
    iree_device_size_t* out_device_offset) {
  *out_device_offset = 0;
  if (IREE_UNLIKELY(file_offset > IREE_DEVICE_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file offset %" PRIu64
                            " exceeds device size max %" PRIdsz,
                            file_offset, IREE_DEVICE_SIZE_MAX);
  }
  iree_device_size_t device_offset = (iree_device_size_t)file_offset;
  iree_device_size_t device_end = 0;
  if (IREE_UNLIKELY(
          !iree_device_size_checked_add(device_offset, length, &device_end))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "file range overflows device size (offset=%" PRIdsz
                            ", length=%" PRIdsz ")",
                            device_offset, length);
  }
  *out_device_offset = device_offset;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_file_range(
    iree_hal_file_t* file, const char* operation_name, uint64_t file_offset,
    iree_device_size_t length, iree_device_size_t* out_device_offset) {
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_cast_file_offset(
      file_offset, length, out_device_offset));
  const uint64_t file_length = iree_hal_file_length(file);
  if (IREE_UNLIKELY(file_offset > file_length ||
                    (uint64_t)length > file_length - file_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s range [%" PRIu64 ", %" PRIu64
                            ") exceeds file length %" PRIu64,
                            operation_name, file_offset,
                            file_offset + (uint64_t)length, file_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_direct_file_buffer(
    iree_hal_buffer_t* buffer, const char* operation_name,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(buffer, buffer_offset, length));
  if (IREE_UNLIKELY(length > (iree_device_size_t)IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s length %" PRIdsz
                            " exceeds host addressable size %" PRIhsz,
                            operation_name, length, IREE_HOST_SIZE_MAX);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                                       IREE_HAL_MEMORY_TYPE_HOST_VISIBLE))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_%s for non-host-visible buffers requires bounded "
        "chunked staging",
        operation_name);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                                       IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_%s for non-mappable buffers requires bounded chunked "
        "staging",
        operation_name);
  }
  return iree_ok_status();
}

static bool iree_hal_amdgpu_host_queue_file_buffer_supports_direct_io(
    iree_hal_buffer_t* buffer) {
  return iree_all_bits_set(iree_hal_buffer_memory_type(buffer),
                           IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         iree_all_bits_set(iree_hal_buffer_allowed_usage(buffer),
                           IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED);
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_direct_file_handle(
    iree_hal_file_t* file, const char* operation_name) {
  if (IREE_UNLIKELY(!iree_hal_file_async_handle(file))) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AMDGPU queue_%s for non-memory files requires a proactor-backed "
        "async file handle",
        operation_name);
  }
  return iree_ok_status();
}

static void iree_hal_amdgpu_file_action_fail_with_borrowed_status(
    iree_hal_amdgpu_file_action_state_t* state, iree_status_t status) {
  if (iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    return;
  }
  iree_hal_semaphore_list_fail(state->signal_semaphore_list,
                               iree_status_clone(status));
}

static iree_status_t iree_hal_amdgpu_file_action_clone_queue_error(
    iree_hal_amdgpu_file_action_state_t* state) {
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &state->queue->error_status, iree_memory_order_acquire);
  return iree_status_is_ok(error) ? iree_ok_status() : iree_status_clone(error);
}

static void iree_hal_amdgpu_file_action_signal_capacity_post_drain(
    void* user_data);

static iree_status_t iree_hal_amdgpu_file_action_submit_signal_barrier(
    iree_hal_amdgpu_file_action_state_t* state) {
  if (iree_hal_semaphore_list_is_empty(state->signal_semaphore_list)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_file_action_clone_queue_error(state));

  iree_hal_amdgpu_wait_resolution_t resolution;
  memset(&resolution, 0, sizeof(resolution));
  resolution.inline_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;
  resolution.barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_SYSTEM;

  iree_slim_mutex_lock(&state->queue->submission_mutex);
  bool ready = false;
  uint64_t submission_id = 0;
  iree_status_t status = iree_hal_amdgpu_host_queue_try_submit_barrier(
      state->queue, &resolution, state->signal_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){0},
      /*operation_resources=*/NULL, /*operation_resource_count=*/0,
      iree_hal_amdgpu_host_queue_post_commit_callback_null(),
      /*resource_set=*/NULL,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES, &ready,
      &submission_id);
  if (iree_status_is_ok(status) && ready) {
    iree_hal_amdgpu_wait_resolution_t profile_resolution = resolution;
    profile_resolution.wait_count = state->profile_wait_count;
    iree_hal_amdgpu_host_queue_record_profile_queue_event(
        state->queue, &profile_resolution, state->signal_semaphore_list,
        &(iree_hal_amdgpu_host_queue_profile_event_info_t){
            .type = iree_hal_amdgpu_file_action_profile_event_type(state->kind),
            .submission_id = submission_id,
            .payload_length = state->requested_length,
            .operation_count = 1,
        });
  }
  if (iree_status_is_ok(status) && !ready) {
    iree_hal_resource_retain(&state->resource);
    iree_hal_amdgpu_host_queue_enqueue_post_drain_action(
        state->queue, &state->signal_capacity_retry,
        iree_hal_amdgpu_file_action_signal_capacity_post_drain, state);
  }
  iree_slim_mutex_unlock(&state->queue->submission_mutex);
  return status;
}

static void iree_hal_amdgpu_file_action_signal_capacity_post_drain(
    void* user_data) {
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;
  iree_status_t status =
      iree_hal_amdgpu_file_action_submit_signal_barrier(state);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }
  iree_hal_resource_release(&state->resource);
}

static void iree_hal_amdgpu_file_action_complete(
    iree_hal_amdgpu_file_action_state_t* state, iree_status_t status) {
  if (iree_status_is_ok(status) &&
      state->kind == IREE_HAL_AMDGPU_FILE_ACTION_READ &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(state->mapping.buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_status_join(
        status, iree_hal_buffer_mapping_flush_range(&state->mapping, 0,
                                                    state->requested_length));
  }
  if (state->mapping.buffer) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&state->mapping));
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_file_action_submit_signal_barrier(state);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
  }
  iree_hal_resource_release(&state->resource);
}

static iree_status_t iree_hal_amdgpu_file_action_submit_next_read(
    iree_hal_amdgpu_file_action_state_t* state);

static iree_status_t iree_hal_amdgpu_file_action_submit_next_write(
    iree_hal_amdgpu_file_action_state_t* state);

static void iree_hal_amdgpu_file_action_read_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;

  bool should_complete = true;
  if (iree_status_is_ok(status) && state->read_op.bytes_read > 0) {
    state->completed_length += state->read_op.bytes_read;
    if (state->completed_length < state->requested_length) {
      status = iree_hal_amdgpu_file_action_submit_next_read(state);
      should_complete = !iree_status_is_ok(status);
    }
  } else if (iree_status_is_ok(status) &&
             state->completed_length < state->requested_length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short read: requested %" PRIhsz
                              " bytes, got %" PRIhsz,
                              state->requested_length, state->completed_length);
  }

  if (should_complete) {
    iree_hal_amdgpu_file_action_complete(state, status);
  }
}

static void iree_hal_amdgpu_file_action_write_complete(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;

  bool should_complete = true;
  if (iree_status_is_ok(status) && state->write_op.bytes_written > 0) {
    state->completed_length += state->write_op.bytes_written;
    if (state->completed_length < state->requested_length) {
      status = iree_hal_amdgpu_file_action_submit_next_write(state);
      should_complete = !iree_status_is_ok(status);
    }
  } else if (iree_status_is_ok(status) &&
             state->completed_length < state->requested_length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "short write: requested %" PRIhsz
                              " bytes, wrote %" PRIhsz,
                              state->requested_length, state->completed_length);
  }

  if (should_complete) {
    iree_hal_amdgpu_file_action_complete(state, status);
  }
}

static iree_status_t iree_hal_amdgpu_file_action_submit_next_read(
    iree_hal_amdgpu_file_action_state_t* state) {
  const iree_host_size_t remaining_length =
      state->requested_length - state->completed_length;
  iree_async_operation_zero(&state->read_op.base, sizeof(state->read_op));
  iree_async_operation_initialize(
      &state->read_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_READ,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_hal_amdgpu_file_action_read_complete,
      state);
  state->read_op.file = state->async_file;
  state->read_op.offset = state->file_offset + state->completed_length;
  state->read_op.buffer = iree_async_span_from_ptr(
      state->mapping.contents.data + state->completed_length, remaining_length);
  return iree_async_proactor_submit_one(state->proactor, &state->read_op.base);
}

static iree_status_t iree_hal_amdgpu_file_action_submit_next_write(
    iree_hal_amdgpu_file_action_state_t* state) {
  const iree_host_size_t remaining_length =
      state->requested_length - state->completed_length;
  iree_async_operation_zero(&state->write_op.base, sizeof(state->write_op));
  iree_async_operation_initialize(
      &state->write_op.base, IREE_ASYNC_OPERATION_TYPE_FILE_WRITE,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_amdgpu_file_action_write_complete, state);
  state->write_op.file = state->async_file;
  state->write_op.offset = state->file_offset + state->completed_length;
  state->write_op.buffer = iree_async_span_from_ptr(
      state->mapping.contents.data + state->completed_length, remaining_length);
  return iree_async_proactor_submit_one(state->proactor, &state->write_op.base);
}

static iree_status_t iree_hal_amdgpu_file_action_start_async(
    iree_hal_amdgpu_file_action_state_t* state) {
  iree_hal_memory_access_t mapping_access = IREE_HAL_MEMORY_ACCESS_READ;
  if (state->kind == IREE_HAL_AMDGPU_FILE_ACTION_READ) {
    mapping_access = IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE;
  }
  iree_status_t status = iree_hal_buffer_map_range(
      state->buffer, IREE_HAL_MAPPING_MODE_SCOPED, mapping_access,
      state->buffer_offset, state->requested_length, &state->mapping);

  if (iree_status_is_ok(status) &&
      state->kind == IREE_HAL_AMDGPU_FILE_ACTION_WRITE &&
      !iree_all_bits_set(iree_hal_buffer_memory_type(state->mapping.buffer),
                         IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
    status = iree_hal_buffer_mapping_invalidate_range(&state->mapping, 0,
                                                      state->requested_length);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_retain(&state->resource);
    if (state->kind == IREE_HAL_AMDGPU_FILE_ACTION_READ) {
      status = iree_hal_amdgpu_file_action_submit_next_read(state);
    } else {
      status = iree_hal_amdgpu_file_action_submit_next_write(state);
    }
    if (iree_status_is_ok(status)) {
      return iree_ok_status();
    }
    iree_hal_resource_release(&state->resource);
  }

  if (state->mapping.buffer) {
    status =
        iree_status_join(status, iree_hal_buffer_unmap_range(&state->mapping));
  }
  return status;
}

static void iree_hal_amdgpu_file_action_execute(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  iree_hal_amdgpu_file_action_state_t* state =
      (iree_hal_amdgpu_file_action_state_t*)user_data;

  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_file_action_start_async(state);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_list_fail(state->signal_semaphore_list, status);
    }
  } else {
    iree_hal_amdgpu_file_action_fail_with_borrowed_status(state, status);
  }
}

static iree_status_t iree_hal_amdgpu_file_action_state_create(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_file_action_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    uint32_t profile_wait_count,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_file_action_state_t** out_state) {
  *out_state = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, length);

  iree_hal_amdgpu_file_action_state_t* state = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(queue->host_allocator, sizeof(*state),
                                (void**)&state));
  memset(state, 0, sizeof(*state));
  iree_hal_resource_initialize(&iree_hal_amdgpu_file_action_state_vtable,
                               &state->resource);
  state->host_allocator = queue->host_allocator;
  state->proactor = queue->proactor;
  state->queue = queue;
  state->logical_device = queue->logical_device;
  iree_hal_device_retain(state->logical_device);
  state->file = file;
  iree_hal_file_retain(state->file);
  state->async_file = iree_hal_file_async_handle(file);
  state->buffer = buffer;
  iree_hal_buffer_retain(state->buffer);
  state->file_offset = file_offset;
  state->buffer_offset = buffer_offset;
  state->requested_length = (iree_host_size_t)length;
  state->profile_wait_count = profile_wait_count;
  state->kind = kind;

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, state->host_allocator,
      &state->signal_semaphore_list);
  if (iree_status_is_ok(status)) {
    *out_state = state;
  } else {
    iree_hal_resource_release(&state->resource);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_direct_file_action(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_file_action_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  iree_hal_amdgpu_file_action_state_t* state = NULL;
  const uint32_t profile_wait_count =
      iree_hal_amdgpu_host_queue_profile_semaphore_count(wait_semaphore_list);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_file_action_state_create(
      queue, kind, file, file_offset, buffer, buffer_offset, length,
      profile_wait_count, signal_semaphore_list, &state));

  iree_hal_resource_t* resources[1] = {&state->resource};
  iree_status_t status = iree_hal_amdgpu_host_queue_enqueue_host_action(
      queue, wait_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_file_action_execute,
          .user_data = state,
      },
      resources, IREE_ARRAYSIZE(resources));
  iree_hal_resource_release(&state->resource);
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_read_file(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(source_file, IREE_HAL_MEMORY_ACCESS_READ));
  if (IREE_UNLIKELY(flags != IREE_HAL_READ_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported read flags: 0x%" PRIx64, flags);
  }
  if (length == 0) {
    return iree_hal_amdgpu_host_queue_file_barrier(queue, wait_semaphore_list,
                                                   signal_semaphore_list);
  }

  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(source_file);
  iree_device_size_t source_device_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_file_range(
      source_file, "read", source_offset, length, &source_device_offset));
  if (!storage_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_direct_file_handle(
        source_file, "read"));
    if (iree_hal_amdgpu_host_queue_file_buffer_supports_direct_io(
            target_buffer)) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_host_queue_validate_direct_file_buffer(
              target_buffer, "read", target_offset, length));
      return iree_hal_amdgpu_host_queue_submit_direct_file_action(
          (iree_hal_amdgpu_host_queue_t*)queue, wait_semaphore_list,
          signal_semaphore_list, IREE_HAL_AMDGPU_FILE_ACTION_READ, source_file,
          source_offset, target_buffer, target_offset, length);
    }
    return iree_hal_amdgpu_host_queue_submit_staged_read(
        (iree_hal_amdgpu_host_queue_t*)queue, wait_semaphore_list,
        signal_semaphore_list, source_file, source_offset, target_buffer,
        target_offset, length);
  }
  return iree_hal_amdgpu_host_queue_copy_buffer(
      (iree_hal_amdgpu_host_queue_t*)queue, wait_semaphore_list,
      signal_semaphore_list, storage_buffer, source_device_offset,
      target_buffer, target_offset, length, IREE_HAL_COPY_FLAG_NONE,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_READ);
}

iree_status_t iree_hal_amdgpu_host_queue_write_file(
    iree_hal_amdgpu_virtual_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  IREE_RETURN_IF_ERROR(
      iree_hal_file_validate_access(target_file, IREE_HAL_MEMORY_ACCESS_WRITE));
  if (IREE_UNLIKELY(flags != IREE_HAL_WRITE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported write flags: 0x%" PRIx64, flags);
  }
  if (length == 0) {
    return iree_hal_amdgpu_host_queue_file_barrier(queue, wait_semaphore_list,
                                                   signal_semaphore_list);
  }

  iree_hal_buffer_t* storage_buffer = iree_hal_file_storage_buffer(target_file);
  iree_device_size_t target_device_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_file_range(
      target_file, "write", target_offset, length, &target_device_offset));
  if (!storage_buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_validate_direct_file_handle(
        target_file, "write"));
    if (iree_hal_amdgpu_host_queue_file_buffer_supports_direct_io(
            source_buffer)) {
      IREE_RETURN_IF_ERROR(
          iree_hal_amdgpu_host_queue_validate_direct_file_buffer(
              source_buffer, "write", source_offset, length));
      return iree_hal_amdgpu_host_queue_submit_direct_file_action(
          (iree_hal_amdgpu_host_queue_t*)queue, wait_semaphore_list,
          signal_semaphore_list, IREE_HAL_AMDGPU_FILE_ACTION_WRITE, target_file,
          target_offset, source_buffer, source_offset, length);
    }
    return iree_hal_amdgpu_host_queue_submit_staged_write(
        (iree_hal_amdgpu_host_queue_t*)queue, wait_semaphore_list,
        signal_semaphore_list, source_buffer, source_offset, target_file,
        target_offset, length);
  }
  return iree_hal_amdgpu_host_queue_copy_buffer(
      (iree_hal_amdgpu_host_queue_t*)queue, wait_semaphore_list,
      signal_semaphore_list, source_buffer, source_offset, storage_buffer,
      target_device_offset, length, IREE_HAL_COPY_FLAG_NONE,
      IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_WRITE);
}
