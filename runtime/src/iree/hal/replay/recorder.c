// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder.h"

#include <stddef.h>
#include <string.h>

#if defined(IREE_PLATFORM_ANDROID)
#include <unistd.h>
#elif defined(IREE_PLATFORM_APPLE)
#include <pthread.h>
#elif defined(IREE_PLATFORM_LINUX)
#include <sys/syscall.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_*

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/hal/replay/recorder_allocator.h"
#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_command_buffer.h"
#include "iree/hal/replay/recorder_event.h"
#include "iree/hal/replay/recorder_executable.h"
#include "iree/hal/replay/recorder_file.h"
#include "iree/hal/replay/recorder_record.h"

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_semaphore_entry_t {
  // Base HAL semaphore retained by the recorder registry.
  iree_hal_semaphore_t* semaphore;
  // Session-local object id assigned to the semaphore.
  iree_hal_replay_object_id_t semaphore_id;
  // Next retained semaphore registry entry.
  struct iree_hal_replay_recorder_semaphore_entry_t* next;
} iree_hal_replay_recorder_semaphore_entry_t;

struct iree_hal_replay_recorder_t {
  // Reference count used to manage shared recorder lifetime.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for recorder lifetime.
  iree_allocator_t host_allocator;
  // Mutex serializing writer access and assigning capture-order ordinals.
  iree_slim_mutex_t mutex;
  // Append-only replay file writer owned by the recorder.
  iree_hal_replay_file_writer_t* writer;
  // Next sequence ordinal assigned to a replay record.
  uint64_t next_sequence_ordinal;
  // Next session-local object id assigned to a captured HAL object.
  iree_hal_replay_object_id_t next_object_id;
  // Terminal recorder failure code, or OK while recording may continue.
  iree_status_code_t terminal_status_code;
  // Retained raw semaphores assigned replay object ids.
  iree_hal_replay_recorder_semaphore_entry_t* semaphore_list;
  // True once the writer has been closed.
  bool closed;
};

static uint64_t iree_hal_replay_current_thread_id(void) {
#if defined(IREE_SYNCHRONIZATION_DISABLE_UNSAFE)
  return 0;
#elif defined(IREE_PLATFORM_ANDROID)
  return (uint64_t)gettid();
#elif defined(IREE_PLATFORM_APPLE)
  return (uint64_t)pthread_mach_thread_np(pthread_self());
#elif defined(IREE_PLATFORM_LINUX)
  return (uint64_t)syscall(__NR_gettid);
#elif defined(IREE_PLATFORM_WINDOWS)
  return (uint64_t)GetCurrentThreadId();
#else
  return 0;
#endif  // IREE_PLATFORM_*
}

static iree_status_t iree_hal_replay_recorder_make_terminal_status(
    iree_hal_replay_recorder_t* recorder) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT(recorder->terminal_status_code != IREE_STATUS_OK);
  return iree_make_status(
      recorder->terminal_status_code,
      "HAL replay recorder is already in a failed terminal state");
}

static iree_status_t iree_hal_replay_recorder_check_open_locked(
    iree_hal_replay_recorder_t* recorder) {
  if (IREE_UNLIKELY(recorder->terminal_status_code != IREE_STATUS_OK)) {
    return iree_hal_replay_recorder_make_terminal_status(recorder);
  }
  if (IREE_UNLIKELY(recorder->closed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL replay recorder is already closed");
  }
  return iree_ok_status();
}

static void iree_hal_replay_recorder_fail_locked(
    iree_hal_replay_recorder_t* recorder, iree_status_t status) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (IREE_UNLIKELY(!iree_status_is_ok(status) &&
                    recorder->terminal_status_code == IREE_STATUS_OK)) {
    recorder->terminal_status_code = iree_status_code(status);
  }
}

void iree_hal_replay_recorder_fail(iree_hal_replay_recorder_t* recorder,
                                   iree_status_t status) {
  IREE_ASSERT_ARGUMENT(recorder);
  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_replay_recorder_fail_locked(recorder, status);
  iree_slim_mutex_unlock(&recorder->mutex);
}

static iree_status_t iree_hal_replay_recorder_append_record_locked(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_file_record_metadata_t metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_hal_replay_file_range_t* out_payload_range) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_check_open_locked(recorder));
  metadata.sequence_ordinal = recorder->next_sequence_ordinal++;
  metadata.thread_id = iree_hal_replay_current_thread_id();
  iree_status_t status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &metadata, iovec_count, iovecs, out_payload_range);
  iree_hal_replay_recorder_fail_locked(recorder, status);
  return status;
}

static iree_status_t iree_hal_replay_recorder_record_session(
    iree_hal_replay_recorder_t* recorder) {
  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_replay_file_record_metadata_t metadata = {
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION,
  };
  iree_status_t status = iree_hal_replay_recorder_append_record_locked(
      recorder, metadata, 0, NULL, NULL);
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

iree_status_t iree_hal_replay_recorder_reserve_object_id(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_object_id_t* out_object_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(out_object_id);
  *out_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  iree_hal_replay_object_id_t object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    object_id = recorder->next_object_id++;
  }
  iree_slim_mutex_unlock(&recorder->mutex);

  if (iree_status_is_ok(status)) *out_object_id = object_id;
  return status;
}

static iree_status_t iree_hal_replay_recorder_append_object_locked(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_hal_replay_file_record_metadata_t metadata = {
      .device_id = device_id,
      .object_id = object_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .payload_type = payload_type,
      .object_type = object_type,
  };
  return iree_hal_replay_recorder_append_record_locked(
      recorder, metadata, iovec_count, iovecs, NULL);
}

iree_status_t iree_hal_replay_recorder_record_object(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs,
    iree_hal_replay_object_id_t* out_object_id) {
  IREE_ASSERT_ARGUMENT(out_object_id);
  *out_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  iree_hal_replay_object_id_t object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    object_id = recorder->next_object_id++;
    status = iree_hal_replay_recorder_append_object_locked(
        recorder, device_id, object_id, object_type, payload_type, iovec_count,
        iovecs);
  }
  iree_slim_mutex_unlock(&recorder->mutex);

  if (iree_status_is_ok(status)) *out_object_id = object_id;
  return status;
}

static iree_status_t iree_hal_replay_recorder_register_semaphore_locked(
    iree_hal_replay_recorder_t* recorder, iree_hal_semaphore_t* semaphore,
    iree_hal_replay_object_id_t semaphore_id) {
  iree_hal_replay_recorder_semaphore_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(recorder->host_allocator,
                                             sizeof(*entry), (void**)&entry));
  entry->semaphore = semaphore;
  iree_hal_semaphore_retain(entry->semaphore);
  entry->semaphore_id = semaphore_id;
  entry->next = recorder->semaphore_list;
  recorder->semaphore_list = entry;
  return iree_ok_status();
}

static iree_hal_replay_object_id_t iree_hal_replay_recorder_lookup_semaphore_id(
    iree_hal_replay_recorder_t* recorder, iree_hal_semaphore_t* semaphore) {
  if (!semaphore) return IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_replay_object_id_t semaphore_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  for (iree_hal_replay_recorder_semaphore_entry_t* entry =
           recorder->semaphore_list;
       entry; entry = entry->next) {
    if (entry->semaphore == semaphore) {
      semaphore_id = entry->semaphore_id;
      break;
    }
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return semaphore_id;
}

static iree_status_t iree_hal_replay_recorder_encode_semaphore_list(
    iree_hal_replay_recorder_t* recorder,
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_replay_semaphore_timepoint_payload_t* out_payloads) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!semaphore_list.semaphores ||
                    !semaphore_list.payload_values || !out_payloads)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "semaphore list storage is required");
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_replay_object_id_t semaphore_id =
        iree_hal_replay_recorder_lookup_semaphore_id(
            recorder, semaphore_list.semaphores[i]);
    if (IREE_UNLIKELY(semaphore_id == IREE_HAL_REPLAY_OBJECT_ID_NONE)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "semaphore list contains a semaphore not created by the replay "
          "recorder");
    }
    out_payloads[i] = (iree_hal_replay_semaphore_timepoint_payload_t){
        .semaphore_id = semaphore_id,
        .value = semaphore_list.payload_values[i],
    };
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_allocate_semaphore_payloads(
    iree_hal_replay_recorder_t* recorder,
    const iree_hal_semaphore_list_t semaphore_list,
    iree_allocator_t host_allocator,
    iree_hal_replay_semaphore_timepoint_payload_t** out_payloads,
    iree_host_size_t* out_payloads_size) {
  IREE_ASSERT_ARGUMENT(out_payloads);
  IREE_ASSERT_ARGUMENT(out_payloads_size);
  *out_payloads = NULL;
  *out_payloads_size = 0;
  if (semaphore_list.count == 0) return iree_ok_status();

  iree_host_size_t payloads_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          semaphore_list.count, sizeof(**out_payloads), &payloads_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay semaphore list count overflow");
  }
  iree_hal_replay_semaphore_timepoint_payload_t* payloads = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, payloads_size, (void**)&payloads));
  iree_status_t status = iree_hal_replay_recorder_encode_semaphore_list(
      recorder, semaphore_list, payloads);
  if (iree_status_is_ok(status)) {
    *out_payloads = payloads;
    *out_payloads_size = payloads_size;
  } else {
    iree_allocator_free(host_allocator, payloads);
  }
  return status;
}

iree_status_t iree_hal_replay_recorder_begin_operation(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  IREE_ASSERT_ARGUMENT(out_pending_record);
  memset(out_pending_record, 0, sizeof(*out_pending_record));

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&recorder->mutex);
    return status;
  }

  out_pending_record->recorder = recorder;
  out_pending_record->metadata = (iree_hal_replay_file_record_metadata_t){
      .sequence_ordinal = recorder->next_sequence_ordinal++,
      .thread_id = iree_hal_replay_current_thread_id(),
      .device_id = device_id,
      .object_id = object_id,
      .related_object_id = related_object_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = payload_type,
      .object_type = object_type,
      .operation_code = operation_code,
  };
  return iree_ok_status();
}

iree_status_t iree_hal_replay_recorder_end_operation_with_payload(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_hal_replay_recorder_t* recorder = pending_record->recorder;
  pending_record->metadata.status_code =
      (uint32_t)iree_status_code(operation_status);
  iree_status_t record_status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &pending_record->metadata, iovec_count, iovecs, NULL);
  iree_hal_replay_recorder_fail_locked(recorder, record_status);
  iree_slim_mutex_unlock(&recorder->mutex);
  if (!iree_status_is_ok(record_status)) {
    iree_status_ignore(operation_status);
    return record_status;
  }
  return operation_status;
}

iree_status_t iree_hal_replay_recorder_end_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status) {
  return iree_hal_replay_recorder_end_operation_with_payload(
      pending_record, operation_status, 0, NULL);
}

iree_status_t iree_hal_replay_recorder_end_creation_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t operation_iovec_count,
    const iree_const_byte_span_t* operation_iovecs,
    iree_hal_replay_object_type_t created_object_type,
    iree_hal_replay_object_id_t created_object_id,
    iree_hal_replay_payload_type_t object_payload_type,
    iree_host_size_t object_iovec_count,
    const iree_const_byte_span_t* object_iovecs) {
  iree_hal_replay_recorder_t* recorder = pending_record->recorder;
  pending_record->metadata.status_code =
      (uint32_t)iree_status_code(operation_status);
  iree_status_t record_status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &pending_record->metadata, operation_iovec_count,
      operation_iovecs, NULL);
  if (iree_status_is_ok(record_status) && iree_status_is_ok(operation_status)) {
    record_status = iree_hal_replay_recorder_append_object_locked(
        recorder, pending_record->metadata.device_id, created_object_id,
        created_object_type, object_payload_type, object_iovec_count,
        object_iovecs);
  }
  iree_hal_replay_recorder_fail_locked(recorder, record_status);
  iree_slim_mutex_unlock(&recorder->mutex);
  if (!iree_status_is_ok(record_status)) {
    iree_status_ignore(operation_status);
    return record_status;
  }
  return operation_status;
}

IREE_API_EXPORT iree_status_t iree_hal_replay_recorder_create(
    iree_io_file_handle_t* file_handle,
    const iree_hal_replay_recorder_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_t** out_recorder) {
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(out_recorder);
  *out_recorder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_replay_recorder_options_t default_options =
      iree_hal_replay_recorder_options_default();
  if (!options) options = &default_options;
  if (IREE_UNLIKELY(options->flags != IREE_HAL_REPLAY_RECORDER_FLAG_NONE ||
                    options->reserved0 != 0)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay recorder reserved options must be zero");
  }

  iree_hal_replay_file_writer_t* writer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_replay_file_writer_create(file_handle, host_allocator, &writer));

  iree_hal_replay_recorder_t* recorder = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*recorder), (void**)&recorder);
  if (iree_status_is_ok(status)) {
    memset(recorder, 0, sizeof(*recorder));
    iree_atomic_ref_count_init(&recorder->ref_count);
    recorder->host_allocator = host_allocator;
    iree_slim_mutex_initialize(&recorder->mutex);
    recorder->writer = writer;
    recorder->next_object_id = 1;
    recorder->terminal_status_code = IREE_STATUS_OK;
    writer = NULL;
    status = iree_hal_replay_recorder_record_session(recorder);
  }

  if (iree_status_is_ok(status)) {
    *out_recorder = recorder;
  } else {
    if (recorder) {
      iree_hal_replay_recorder_release(recorder);
    }
  }
  iree_hal_replay_file_writer_free(writer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_replay_recorder_retain(
    iree_hal_replay_recorder_t* recorder) {
  if (IREE_LIKELY(recorder)) {
    iree_atomic_ref_count_inc(&recorder->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_replay_recorder_release(
    iree_hal_replay_recorder_t* recorder) {
  if (IREE_LIKELY(recorder) &&
      iree_atomic_ref_count_dec(&recorder->ref_count) == 1) {
    iree_allocator_t host_allocator = recorder->host_allocator;
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_hal_replay_recorder_semaphore_entry_t* semaphore_entry =
        recorder->semaphore_list;
    while (semaphore_entry) {
      iree_hal_replay_recorder_semaphore_entry_t* next_entry =
          semaphore_entry->next;
      iree_hal_semaphore_release(semaphore_entry->semaphore);
      iree_allocator_free(host_allocator, semaphore_entry);
      semaphore_entry = next_entry;
    }
    iree_hal_replay_file_writer_free(recorder->writer);
    iree_slim_mutex_deinitialize(&recorder->mutex);
    iree_allocator_free(host_allocator, recorder);
    IREE_TRACE_ZONE_END(z0);
  }
}

IREE_API_EXPORT iree_status_t
iree_hal_replay_recorder_close(iree_hal_replay_recorder_t* recorder) {
  IREE_ASSERT_ARGUMENT(recorder);

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(recorder->terminal_status_code != IREE_STATUS_OK)) {
    status = iree_hal_replay_recorder_make_terminal_status(recorder);
  } else if (!recorder->closed) {
    status = iree_hal_replay_file_writer_close(recorder->writer);
    if (iree_status_is_ok(status)) {
      recorder->closed = true;
    } else {
      iree_hal_replay_recorder_fail_locked(recorder, status);
    }
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_replay_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_device_t {
  // HAL resource header for the recording wrapper device.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Source group retaining the underlying device and original topology.
  iree_hal_device_group_t* base_group;
  // Underlying device receiving forwarded HAL calls.
  iree_hal_device_t* base_device;
  // Recording allocator returned from iree_hal_device_allocator.
  iree_hal_allocator_t* allocator;
  // Session-local object id assigned to this wrapper.
  iree_hal_replay_object_id_t device_id;
  // Topology information assigned to this wrapper during group creation.
  iree_hal_device_topology_info_t topology_info;
} iree_hal_replay_device_t;

static const iree_hal_device_vtable_t iree_hal_replay_device_vtable;

static bool iree_hal_replay_device_isa(iree_hal_device_t* base_device) {
  return iree_hal_resource_is(base_device, &iree_hal_replay_device_vtable);
}

static iree_hal_replay_device_t* iree_hal_replay_device_cast(
    iree_hal_device_t* base_device) {
  IREE_HAL_ASSERT_TYPE(base_device, &iree_hal_replay_device_vtable);
  return (iree_hal_replay_device_t*)base_device;
}

static iree_hal_device_t* iree_hal_replay_device_base_or_self(
    iree_hal_device_t* device) {
  return iree_hal_replay_device_isa(device)
             ? iree_hal_replay_device_cast(device)->base_device
             : device;
}

static iree_status_t iree_hal_replay_device_begin_operation(
    iree_hal_replay_device_t* device,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      operation_code, IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, out_pending_record);
}

static iree_status_t iree_hal_replay_device_complete_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status) {
  return iree_hal_replay_recorder_end_operation(pending_record,
                                                operation_status);
}

static void iree_hal_replay_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_allocator_t host_allocator = device->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_allocator_release(device->allocator);
  iree_hal_device_release(device->base_device);
  iree_hal_device_group_release(device->base_group);
  iree_hal_replay_recorder_release(device->recorder);
  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_string_view_t iree_hal_replay_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_id(device->base_device);
}

static iree_allocator_t iree_hal_replay_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_host_allocator(device->base_device);
}

static iree_hal_allocator_t* iree_hal_replay_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return device->allocator;
}

static void iree_hal_replay_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  if (!new_allocator) {
    iree_hal_device_replace_allocator(device->base_device, new_allocator);
    iree_hal_allocator_release(device->allocator);
    device->allocator = NULL;
    return;
  }
  iree_hal_allocator_t* new_replay_allocator = NULL;
  iree_status_t status = iree_hal_replay_recorder_wrap_allocator(
      device->recorder, device->device_id, base_device, new_allocator,
      device->host_allocator, &new_replay_allocator);
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_recorder_fail(device->recorder, status);
    iree_status_ignore(status);
    return;
  }
  iree_hal_device_replace_allocator(device->base_device, new_allocator);
  iree_hal_allocator_release(device->allocator);
  device->allocator = new_replay_allocator;
}

static void iree_hal_replay_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_device_replace_channel_provider(device->base_device, new_provider);
}

static iree_status_t iree_hal_replay_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record = {0};
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM, &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_trim(device->base_device));
}

static iree_status_t iree_hal_replay_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record = {0};
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_I64,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_query_i64(device->base_device, category, key, out_value));
}

static iree_status_t iree_hal_replay_device_query_capabilities(
    iree_hal_device_t* base_device,
    iree_hal_device_capabilities_t* out_capabilities) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_CAPABILITIES,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_query_capabilities(device->base_device,
                                                          out_capabilities));
}

static const iree_hal_device_topology_info_t*
iree_hal_replay_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t iree_hal_replay_device_refine_topology_edge(
    iree_hal_device_t* src_device, iree_hal_device_t* dst_device,
    iree_hal_topology_edge_t* edge) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(src_device);
  iree_hal_device_t* base_dst_device =
      iree_hal_replay_device_base_or_self(dst_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_REFINE_TOPOLOGY_EDGE,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_refine_topology_edge(
                           device->base_device, base_dst_device, edge));
}

static iree_status_t iree_hal_replay_device_assign_topology_info(
    iree_hal_device_t* base_device,
    const iree_hal_device_topology_info_t* topology_info) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO,
      &pending_record));
  iree_status_t status = iree_ok_status();
  if (topology_info) {
    device->topology_info = *topology_info;
  } else {
    memset(&device->topology_info, 0, sizeof(device->topology_info));
  }
  return iree_hal_replay_device_complete_operation(&pending_record, status);
}

static iree_status_t iree_hal_replay_device_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_CHANNEL,
      &pending_record));
  iree_status_t status = iree_hal_channel_create(
      device->base_device, queue_affinity, params, out_channel);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_channel && *out_channel) {
    iree_hal_channel_release(*out_channel);
    *out_channel = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_command_buffer = NULL;

  iree_hal_replay_object_id_t command_buffer_id =
      IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      device->recorder, &command_buffer_id));

  iree_hal_replay_command_buffer_object_payload_t payload;
  iree_hal_replay_recorder_command_buffer_make_object_payload(
      mode, command_categories, queue_affinity, binding_capacity, &payload);
  iree_const_byte_span_t payload_iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, command_buffer_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT, &pending_record));

  iree_hal_command_buffer_t* base_command_buffer = NULL;
  iree_hal_command_buffer_t* replay_command_buffer = NULL;
  iree_status_t status = iree_hal_command_buffer_create(
      device->base_device, mode, command_categories, queue_affinity,
      binding_capacity, &base_command_buffer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_create_proxy(
        device->recorder, device->device_id, command_buffer_id,
        iree_hal_device_allocator(device->base_device), base_command_buffer,
        device->host_allocator, &replay_command_buffer);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 1, &payload_iovec,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, command_buffer_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT, 1, &payload_iovec);

  if (iree_status_is_ok(status)) {
    *out_command_buffer = replay_command_buffer;
  } else {
    iree_hal_command_buffer_release(replay_command_buffer);
  }
  iree_hal_command_buffer_release(base_command_buffer);
  return status;
}

static iree_status_t iree_hal_replay_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_event = NULL;

  iree_hal_replay_object_id_t event_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_reserve_object_id(device->recorder, &event_id));

  iree_hal_replay_event_object_payload_t payload;
  iree_hal_replay_recorder_event_make_object_payload(queue_affinity, flags,
                                                     &payload);
  iree_const_byte_span_t payload_iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, event_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EVENT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_EVENT_OBJECT, &pending_record));

  iree_hal_event_t* base_event = NULL;
  iree_hal_event_t* replay_event = NULL;
  iree_status_t status = iree_hal_event_create(
      device->base_device, queue_affinity, flags, &base_event);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_event_create_proxy(
        device->recorder, device->device_id, event_id, base_event,
        device->host_allocator, &replay_event);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 1, &payload_iovec,
      IREE_HAL_REPLAY_OBJECT_TYPE_EVENT, event_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_EVENT_OBJECT, 1, &payload_iovec);

  if (iree_status_is_ok(status)) {
    *out_event = replay_event;
  } else {
    iree_hal_event_release(replay_event);
  }
  iree_hal_event_release(base_event);
  return status;
}

static iree_status_t iree_hal_replay_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_executable_cache = NULL;

  iree_hal_replay_object_id_t executable_cache_id =
      IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      device->recorder, &executable_cache_id));

  iree_hal_replay_executable_cache_object_payload_t payload = {
      .identifier_length = identifier.size,
  };
  iree_const_byte_span_t payload_iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(identifier.data, identifier.size),
  };

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id,
      executable_cache_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EXECUTABLE_CACHE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_CACHE_OBJECT, &pending_record));

  iree_hal_executable_cache_t* base_executable_cache = NULL;
  iree_hal_executable_cache_t* replay_executable_cache = NULL;
  iree_status_t status = iree_hal_executable_cache_create(
      device->base_device, identifier, &base_executable_cache);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_executable_cache_create_proxy(
        device->recorder, device->device_id, executable_cache_id,
        base_executable_cache, device->host_allocator,
        &replay_executable_cache);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, IREE_ARRAYSIZE(payload_iovecs), payload_iovecs,
      IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE_CACHE, executable_cache_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_CACHE_OBJECT,
      IREE_ARRAYSIZE(payload_iovecs), payload_iovecs);

  if (iree_status_is_ok(status)) {
    *out_executable_cache = replay_executable_cache;
  } else {
    iree_hal_executable_cache_release(replay_executable_cache);
  }
  iree_hal_executable_cache_release(base_executable_cache);
  return status;
}

static iree_status_t iree_hal_replay_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_file = NULL;

  iree_hal_replay_object_id_t file_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_reserve_object_id(device->recorder, &file_id));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, file_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT, &pending_record));

  iree_hal_file_t* base_file = NULL;
  iree_hal_file_t* replay_file = NULL;
  iree_status_t status = iree_hal_file_import(
      device->base_device, queue_affinity, access, handle, flags, &base_file);

  char reference_storage[IREE_MAX_PATH];
  iree_hal_replay_file_object_payload_t payload;
  iree_string_view_t reference = iree_string_view_empty();
  iree_hal_replay_recorder_file_make_object_payload(
      handle, queue_affinity, access, flags, base_file,
      iree_make_byte_span((uint8_t*)reference_storage,
                          sizeof(reference_storage)),
      &payload, &reference);
  iree_const_byte_span_t payload_iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(reference.data, reference.size),
  };

  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_file_create_proxy(
        device->recorder, device->device_id, file_id, base_file,
        device->host_allocator, &replay_file);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, IREE_ARRAYSIZE(payload_iovecs), payload_iovecs,
      IREE_HAL_REPLAY_OBJECT_TYPE_FILE, file_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT, IREE_ARRAYSIZE(payload_iovecs),
      payload_iovecs);

  if (iree_status_is_ok(status)) {
    *out_file = replay_file;
  } else {
    iree_hal_file_release(replay_file);
  }
  iree_hal_file_release(base_file);
  return status;
}

static iree_status_t iree_hal_replay_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_semaphore = NULL;

  iree_hal_replay_object_id_t semaphore_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_reserve_object_id(
      device->recorder, &semaphore_id));

  iree_hal_replay_semaphore_object_payload_t payload = {
      .queue_affinity = queue_affinity,
      .initial_value = initial_value,
      .flags = flags,
  };
  iree_const_byte_span_t payload_iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      device->recorder, device->device_id, device->device_id, semaphore_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT, &pending_record));

  iree_hal_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_hal_semaphore_create(
      device->base_device, queue_affinity, initial_value, flags, &semaphore);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_register_semaphore_locked(
        device->recorder, semaphore, semaphore_id);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, 1, &payload_iovec,
      IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE, semaphore_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT, 1, &payload_iovec);

  if (iree_status_is_ok(status)) {
    *out_semaphore = semaphore;
  } else {
    iree_hal_semaphore_release(semaphore);
  }
  return status;
}

static iree_hal_semaphore_compatibility_t
iree_hal_replay_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  return iree_hal_device_query_semaphore_compatibility(device->base_device,
                                                       semaphore);
}

static iree_status_t iree_hal_replay_device_query_queue_pool_backend(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_queue_pool_backend_t* out_backend) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_QUEUE_POOL_BACKEND,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_query_queue_pool_backend(
                           device->base_device, queue_affinity, out_backend));
}

static iree_status_t iree_hal_replay_device_queue_alloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  *out_buffer = NULL;

  iree_hal_replay_object_id_t buffer_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_reserve_object_id(device->recorder, &buffer_id));

  iree_hal_buffer_params_t canonical_params = params;
  iree_hal_buffer_params_canonicalize(&canonical_params);
  iree_hal_replay_device_queue_alloca_payload_t operation_payload;
  memset(&operation_payload, 0, sizeof(operation_payload));
  iree_hal_replay_recorder_allocator_make_allocate_buffer_payload(
      &canonical_params, allocation_size, &operation_payload.allocation);
  operation_payload.queue_affinity = queue_affinity;
  operation_payload.flags = flags;
  operation_payload.wait_semaphore_count = wait_semaphore_list.count;
  operation_payload.signal_semaphore_count = signal_semaphore_list.count;
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t operation_iovecs[3] = {
      iree_make_const_byte_span(&operation_payload, sizeof(operation_payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id, buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA, &pending_record);
  }

  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* replay_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_alloca(device->base_device, queue_affinity,
                                          wait_semaphore_list,
                                          signal_semaphore_list, pool, params,
                                          allocation_size, flags, &base_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_create_proxy(
        device->recorder, device->device_id, buffer_id, base_device,
        base_buffer, device->host_allocator, &replay_buffer);
  }

  iree_hal_replay_buffer_object_payload_t object_payload;
  if (iree_status_is_ok(status)) {
    iree_hal_replay_recorder_buffer_make_object_payload(base_buffer,
                                                        &object_payload);
  } else {
    memset(&object_payload, 0, sizeof(object_payload));
  }
  iree_const_byte_span_t object_iovec = iree_make_const_byte_span(
      (const uint8_t*)&object_payload, sizeof(object_payload));
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_creation_operation(
        &pending_record, status, IREE_ARRAYSIZE(operation_iovecs),
        operation_iovecs, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, buffer_id,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT, 1, &object_iovec);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = replay_buffer;
  } else {
    iree_hal_buffer_release(replay_buffer);
  }
  iree_hal_buffer_release(base_buffer);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_dealloca_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(buffer, 0, iree_hal_buffer_byte_length(buffer)),
      &payload.buffer_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.buffer_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA, &pending_record);
  }
  iree_hal_buffer_t* base_buffer = NULL;
  iree_hal_buffer_t* temporary_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        buffer, device->host_allocator, &base_buffer, &temporary_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_buffer, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_fill_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.pattern_length = pattern_length;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(pattern, pattern_length),
  };

  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_fill(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_target_buffer, target_offset, length,
        pattern, pattern_length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue update length exceeds host size");
  }
  if (IREE_UNLIKELY(length != 0 && !source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay queue update source buffer is required");
  }

  iree_hal_replay_device_queue_update_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.source_offset = source_offset;
  payload.data_length = length;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  const uint8_t* source_data =
      length ? (const uint8_t*)source_buffer + source_offset : NULL;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(source_data, (iree_host_size_t)length),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, source_buffer, source_offset, base_target_buffer,
        target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_copy_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(source_buffer, source_offset, length),
      &payload.source_ref);
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY, &pending_record);
  }
  iree_hal_buffer_t* base_source_buffer = NULL;
  iree_hal_buffer_t* temporary_source_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        source_buffer, device->host_allocator, &base_source_buffer,
        &temporary_source_buffer);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_source_buffer, source_offset,
        base_target_buffer, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  iree_hal_replay_recorder_buffer_release_temporary(temporary_source_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_read_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.source_file_id =
      iree_hal_replay_recorder_file_id_or_none(source_file);
  if (IREE_UNLIKELY(payload.source_file_id == IREE_HAL_REPLAY_OBJECT_ID_NONE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot record replay queue_read with an unwrapped file");
  }
  payload.source_offset = source_offset;
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(target_buffer, target_offset, length),
      &payload.target_ref);
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.target_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ, &pending_record);
  }
  iree_hal_buffer_t* base_target_buffer = NULL;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_buffer, device->host_allocator, &base_target_buffer,
        &temporary_target_buffer);
  }
  iree_hal_file_t* base_source_file = NULL;
  if (iree_status_is_ok(status)) {
    base_source_file = iree_hal_replay_recorder_file_base_or_self(source_file);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_source_file, source_offset,
        base_target_buffer, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_device_queue_write_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      iree_hal_make_buffer_ref(source_buffer, source_offset, length),
      &payload.source_ref);
  payload.target_file_id =
      iree_hal_replay_recorder_file_id_or_none(target_file);
  if (IREE_UNLIKELY(payload.target_file_id == IREE_HAL_REPLAY_OBJECT_ID_NONE)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot record replay queue_write with an unwrapped file");
  }
  payload.target_offset = target_offset;
  payload.queue_affinity = queue_affinity;
  payload.flags = flags;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
  };

  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.source_ref.buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE, &pending_record);
  }
  iree_hal_buffer_t* base_source_buffer = NULL;
  iree_hal_buffer_t* temporary_source_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        source_buffer, device->host_allocator, &base_source_buffer,
        &temporary_source_buffer);
  }
  iree_hal_file_t* base_target_file = NULL;
  if (iree_status_is_ok(status)) {
    base_target_file = iree_hal_replay_recorder_file_base_or_self(target_file);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device->base_device, queue_affinity, wait_semaphore_list,
        signal_semaphore_list, base_source_buffer, source_offset,
        base_target_file, target_offset, length, flags);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_source_buffer);
  if (pending_record.recorder) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record, status, IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_host_call(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_HOST_CALL,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_host_call(
          device->base_device, queue_affinity, wait_semaphore_list,
          signal_semaphore_list, call, args, flags));
}

static iree_status_t iree_hal_replay_device_queue_dispatch(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);

  iree_hal_replay_dispatch_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.executable_id =
      iree_hal_replay_recorder_executable_id_or_none(executable);
  payload.queue_affinity = queue_affinity;
  payload.export_ordinal = export_ordinal;
  payload.flags = flags;
  memcpy(payload.workgroup_size, config.workgroup_size,
         sizeof(payload.workgroup_size));
  memcpy(payload.workgroup_count, config.workgroup_count,
         sizeof(payload.workgroup_count));
  iree_hal_replay_recorder_buffer_ref_make_payload(
      config.workgroup_count_ref, &payload.workgroup_count_ref);
  payload.dynamic_workgroup_local_memory =
      config.dynamic_workgroup_local_memory;
  payload.wait_semaphore_count = wait_semaphore_list.count;
  payload.signal_semaphore_count = signal_semaphore_list.count;
  payload.constants_length = constants.data_length;
  payload.binding_count = bindings.count;

  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_hal_replay_buffer_ref_payload_t* binding_payloads = NULL;
  iree_host_size_t binding_payloads_size = 0;
  iree_hal_buffer_ref_list_t base_bindings = bindings;
  iree_hal_buffer_ref_t* binding_storage = NULL;
  iree_hal_buffer_t** temporary_buffers = NULL;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  if (bindings.count) {
    iree_host_size_t binding_storage_size = 0;
    iree_host_size_t temporary_buffers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*binding_payloads),
                                                  &binding_payloads_size) ||
                      !iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*binding_storage),
                                                  &binding_storage_size) ||
                      !iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*temporary_buffers),
                                                  &temporary_buffers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay queue dispatch binding count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_payloads_size,
                                (void**)&binding_payloads);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_storage_size,
                                (void**)&binding_storage);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, temporary_buffers_size,
                                (void**)&temporary_buffers);
    }
    if (iree_status_is_ok(status)) {
      memset(temporary_buffers, 0, temporary_buffers_size);
      memcpy(binding_storage, bindings.values, binding_storage_size);
      for (iree_host_size_t i = 0;
           i < bindings.count && iree_status_is_ok(status); ++i) {
        iree_hal_replay_recorder_buffer_ref_make_payload(bindings.values[i],
                                                         &binding_payloads[i]);
        if (binding_storage[i].buffer) {
          status = iree_hal_replay_recorder_buffer_unwrap_for_call(
              binding_storage[i].buffer, device->host_allocator,
              &binding_storage[i].buffer, &temporary_buffers[i]);
        }
      }
      base_bindings.values = binding_storage;
    }
  }

  iree_const_byte_span_t iovecs[5] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      constants,
      iree_make_const_byte_span(binding_payloads, binding_payloads_size),
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.executable_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH, &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_device_queue_dispatch(
            device->base_device, queue_affinity, wait_semaphore_list,
            signal_semaphore_list,
            iree_hal_replay_recorder_executable_base_or_self(executable),
            export_ordinal, config, constants, base_bindings, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (temporary_buffers) {
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(temporary_buffers[i]);
    }
  }
  iree_allocator_free(device->host_allocator, temporary_buffers);
  iree_allocator_free(device->host_allocator, binding_storage);
  iree_allocator_free(device->host_allocator, binding_payloads);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);

  iree_hal_replay_device_queue_execute_payload_t payload = {
      .command_buffer_id =
          iree_hal_replay_recorder_command_buffer_id_or_none(command_buffer),
      .queue_affinity = queue_affinity,
      .flags = flags,
      .wait_semaphore_count = wait_semaphore_list.count,
      .signal_semaphore_count = signal_semaphore_list.count,
      .binding_count = binding_table.count,
  };
  iree_hal_replay_semaphore_timepoint_payload_t* wait_payloads = NULL;
  iree_host_size_t wait_payloads_size = 0;
  iree_hal_replay_semaphore_timepoint_payload_t* signal_payloads = NULL;
  iree_host_size_t signal_payloads_size = 0;
  iree_hal_replay_buffer_ref_payload_t* binding_payloads = NULL;
  iree_host_size_t binding_payloads_size = 0;
  iree_hal_buffer_binding_table_t base_binding_table = binding_table;
  iree_hal_buffer_binding_t* binding_storage = NULL;
  iree_hal_buffer_t** temporary_buffers = NULL;
  iree_status_t status = iree_hal_replay_recorder_allocate_semaphore_payloads(
      device->recorder, wait_semaphore_list, device->host_allocator,
      &wait_payloads, &wait_payloads_size);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_allocate_semaphore_payloads(
        device->recorder, signal_semaphore_list, device->host_allocator,
        &signal_payloads, &signal_payloads_size);
  }
  if (binding_table.count) {
    iree_host_size_t binding_storage_size = 0;
    iree_host_size_t temporary_buffers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(binding_table.count,
                                                  sizeof(*binding_payloads),
                                                  &binding_payloads_size) ||
                      !iree_host_size_checked_mul(binding_table.count,
                                                  sizeof(*binding_storage),
                                                  &binding_storage_size) ||
                      !iree_host_size_checked_mul(binding_table.count,
                                                  sizeof(*temporary_buffers),
                                                  &temporary_buffers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay queue execute binding count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_payloads_size,
                                (void**)&binding_payloads);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, binding_storage_size,
                                (void**)&binding_storage);
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(device->host_allocator, temporary_buffers_size,
                                (void**)&temporary_buffers);
    }
    if (iree_status_is_ok(status)) {
      memset(temporary_buffers, 0, temporary_buffers_size);
      memcpy(binding_storage, binding_table.bindings, binding_storage_size);
      for (iree_host_size_t i = 0;
           i < binding_table.count && iree_status_is_ok(status); ++i) {
        iree_hal_buffer_ref_t binding_ref = iree_hal_make_buffer_ref(
            binding_table.bindings[i].buffer, binding_table.bindings[i].offset,
            binding_table.bindings[i].length);
        iree_hal_replay_recorder_buffer_ref_make_payload(binding_ref,
                                                         &binding_payloads[i]);
        if (binding_storage[i].buffer) {
          status = iree_hal_replay_recorder_buffer_unwrap_for_call(
              binding_storage[i].buffer, device->host_allocator,
              &binding_storage[i].buffer, &temporary_buffers[i]);
        }
      }
      base_binding_table.bindings = binding_storage;
    }
  }

  iree_const_byte_span_t iovecs[4] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(wait_payloads, wait_payloads_size),
      iree_make_const_byte_span(signal_payloads, signal_payloads_size),
      iree_make_const_byte_span(binding_payloads, binding_payloads_size),
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_begin_operation(
        device->recorder, device->device_id, device->device_id,
        payload.command_buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
        IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE, &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_device_queue_execute(
            device->base_device, queue_affinity, wait_semaphore_list,
            signal_semaphore_list,
            iree_hal_replay_recorder_command_buffer_base_or_self(
                command_buffer),
            base_binding_table, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (temporary_buffers) {
    for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(temporary_buffers[i]);
    }
  }
  iree_allocator_free(device->host_allocator, temporary_buffers);
  iree_allocator_free(device->host_allocator, binding_storage);
  iree_allocator_free(device->host_allocator, binding_payloads);
  iree_allocator_free(device->host_allocator, signal_payloads);
  iree_allocator_free(device->host_allocator, wait_payloads);
  return status;
}

static iree_status_t iree_hal_replay_device_queue_flush(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FLUSH,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_flush(device->base_device, queue_affinity));
}

static iree_status_t iree_hal_replay_device_profiling_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_profiling_options_t* options) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_BEGIN,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_profiling_begin(device->base_device, options));
}

static iree_status_t iree_hal_replay_device_profiling_flush(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_FLUSH,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_profiling_flush(device->base_device));
}

static iree_status_t iree_hal_replay_device_profiling_end(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_END,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_profiling_end(device->base_device));
}

static iree_status_t iree_hal_replay_device_external_capture_begin(
    iree_hal_device_t* base_device,
    const iree_hal_device_external_capture_options_t* options) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_BEGIN,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_external_capture_begin(device->base_device, options));
}

static iree_status_t iree_hal_replay_device_external_capture_end(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_END,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_external_capture_end(device->base_device));
}

static iree_status_t iree_hal_replay_wrap_device(
    iree_hal_replay_recorder_t* recorder, iree_hal_device_group_t* base_group,
    iree_hal_device_t* base_device, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_group);
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_replay_device_t* device = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*device), (void**)&device));
  memset(device, 0, sizeof(*device));
  iree_hal_resource_initialize(&iree_hal_replay_device_vtable,
                               &device->resource);
  device->host_allocator = host_allocator;
  device->recorder = recorder;
  iree_hal_replay_recorder_retain(device->recorder);
  device->base_group = base_group;
  iree_hal_device_group_retain(device->base_group);
  device->base_device = base_device;
  iree_hal_device_retain(device->base_device);

  iree_status_t status = iree_hal_replay_recorder_record_object(
      recorder, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE, IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, 0,
      NULL, &device->device_id);
  iree_hal_allocator_t* base_allocator =
      iree_status_is_ok(status) ? iree_hal_device_allocator(device->base_device)
                                : NULL;
  if (iree_status_is_ok(status) && base_allocator) {
    status = iree_hal_replay_recorder_wrap_allocator(
        recorder, device->device_id, (iree_hal_device_t*)device, base_allocator,
        host_allocator, &device->allocator);
  }
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_allocator_release(device->allocator);
    iree_hal_device_release(device->base_device);
    iree_hal_device_group_release(device->base_group);
    iree_hal_replay_recorder_release(device->recorder);
    iree_allocator_free(host_allocator, device);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

typedef struct iree_hal_replay_wrap_device_group_context_t {
  // Shared recorder receiving all wrapper records.
  iree_hal_replay_recorder_t* recorder;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
} iree_hal_replay_wrap_device_group_context_t;

static iree_status_t iree_hal_replay_wrap_device_group_device(
    iree_hal_device_group_t* source_group, iree_host_size_t device_index,
    iree_hal_device_t* source_device, void* user_data,
    iree_hal_device_t** out_replacement_device) {
  (void)device_index;
  iree_hal_replay_wrap_device_group_context_t* context =
      (iree_hal_replay_wrap_device_group_context_t*)user_data;
  return iree_hal_replay_wrap_device(context->recorder, source_group,
                                     source_device, context->host_allocator,
                                     out_replacement_device);
}

IREE_API_EXPORT iree_status_t iree_hal_replay_wrap_device_group(
    iree_hal_replay_recorder_t* recorder, iree_hal_device_group_t* base_group,
    iree_allocator_t host_allocator, iree_hal_device_group_t** out_group) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_group);
  IREE_ASSERT_ARGUMENT(out_group);
  *out_group = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_replay_wrap_device_group_context_t context = {
      .recorder = recorder,
      .host_allocator = host_allocator,
  };
  iree_hal_device_group_replacement_callback_t replacement_callback = {
      .fn = iree_hal_replay_wrap_device_group_device,
      .user_data = &context,
  };
  iree_status_t status = iree_hal_device_group_create_with_replacements(
      base_group, replacement_callback, host_allocator, out_group);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_device_vtable_t iree_hal_replay_device_vtable = {
    .destroy = iree_hal_replay_device_destroy,
    .id = iree_hal_replay_device_id,
    .host_allocator = iree_hal_replay_device_host_allocator,
    .device_allocator = iree_hal_replay_device_allocator,
    .replace_device_allocator = iree_hal_replay_replace_device_allocator,
    .replace_channel_provider = iree_hal_replay_replace_channel_provider,
    .trim = iree_hal_replay_device_trim,
    .query_i64 = iree_hal_replay_device_query_i64,
    .query_capabilities = iree_hal_replay_device_query_capabilities,
    .topology_info = iree_hal_replay_device_topology_info,
    .refine_topology_edge = iree_hal_replay_device_refine_topology_edge,
    .assign_topology_info = iree_hal_replay_device_assign_topology_info,
    .create_channel = iree_hal_replay_device_create_channel,
    .create_command_buffer = iree_hal_replay_device_create_command_buffer,
    .create_event = iree_hal_replay_device_create_event,
    .create_executable_cache = iree_hal_replay_device_create_executable_cache,
    .import_file = iree_hal_replay_device_import_file,
    .create_semaphore = iree_hal_replay_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_replay_device_query_semaphore_compatibility,
    .query_queue_pool_backend = iree_hal_replay_device_query_queue_pool_backend,
    .queue_alloca = iree_hal_replay_device_queue_alloca,
    .queue_dealloca = iree_hal_replay_device_queue_dealloca,
    .queue_fill = iree_hal_replay_device_queue_fill,
    .queue_update = iree_hal_replay_device_queue_update,
    .queue_copy = iree_hal_replay_device_queue_copy,
    .queue_read = iree_hal_replay_device_queue_read,
    .queue_write = iree_hal_replay_device_queue_write,
    .queue_host_call = iree_hal_replay_device_queue_host_call,
    .queue_dispatch = iree_hal_replay_device_queue_dispatch,
    .queue_execute = iree_hal_replay_device_queue_execute,
    .queue_flush = iree_hal_replay_device_queue_flush,
    .profiling_begin = iree_hal_replay_device_profiling_begin,
    .profiling_flush = iree_hal_replay_device_profiling_flush,
    .profiling_end = iree_hal_replay_device_profiling_end,
    .external_capture_begin = iree_hal_replay_device_external_capture_begin,
    .external_capture_end = iree_hal_replay_device_external_capture_end,
};
