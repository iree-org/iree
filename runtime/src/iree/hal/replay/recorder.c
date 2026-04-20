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
#include "iree/hal/replay/file.h"

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_t
//===----------------------------------------------------------------------===//

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
  // True once the writer has been closed.
  bool closed;
};

typedef struct iree_hal_replay_pending_record_t {
  // Recorder whose mutex is held until the pending record is completed.
  iree_hal_replay_recorder_t* recorder;
  // Metadata to write once the intercepted operation has completed.
  iree_hal_replay_file_record_metadata_t metadata;
} iree_hal_replay_pending_record_t;

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

static iree_status_t iree_hal_replay_recorder_record_device_object(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_object_id_t* out_device_id) {
  IREE_ASSERT_ARGUMENT(out_device_id);
  *out_device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  iree_hal_replay_object_id_t device_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    device_id = recorder->next_object_id++;
    iree_hal_replay_file_record_metadata_t metadata = {
        .device_id = device_id,
        .object_id = device_id,
        .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
        .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
    };
    status = iree_hal_replay_recorder_append_record_locked(recorder, metadata,
                                                           0, NULL, NULL);
  }
  iree_slim_mutex_unlock(&recorder->mutex);

  if (iree_status_is_ok(status)) *out_device_id = device_id;
  return status;
}

static iree_status_t iree_hal_replay_recorder_begin_operation(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_operation_code_t operation_code,
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
      .object_type = IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      .operation_code = operation_code,
  };
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_end_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status) {
  iree_hal_replay_recorder_t* recorder = pending_record->recorder;
  pending_record->metadata.status_code =
      (uint32_t)iree_status_code(operation_status);
  iree_status_t record_status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &pending_record->metadata, 0, NULL, NULL);
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
      IREE_HAL_REPLAY_OBJECT_ID_NONE, operation_code, out_pending_record);
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
  return iree_hal_device_allocator(device->base_device);
}

static void iree_hal_replay_replace_device_allocator(
    iree_hal_device_t* base_device, iree_hal_allocator_t* new_allocator) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_device_replace_allocator(device->base_device, new_allocator);
}

static void iree_hal_replay_replace_channel_provider(
    iree_hal_device_t* base_device, iree_hal_channel_provider_t* new_provider) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_device_replace_channel_provider(device->base_device, new_provider);
}

static iree_status_t iree_hal_replay_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM, &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record, iree_hal_device_trim(device->base_device));
}

static iree_status_t iree_hal_replay_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
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
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER,
      &pending_record));
  iree_status_t status = iree_hal_command_buffer_create(
      device->base_device, mode, command_categories, queue_affinity,
      binding_capacity, out_command_buffer);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_command_buffer && *out_command_buffer) {
    iree_hal_command_buffer_release(*out_command_buffer);
    *out_command_buffer = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_create_event(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EVENT,
      &pending_record));
  iree_status_t status = iree_hal_event_create(
      device->base_device, queue_affinity, flags, out_event);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_event && *out_event) {
    iree_hal_event_release(*out_event);
    *out_event = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EXECUTABLE_CACHE,
      &pending_record));
  iree_status_t status = iree_hal_executable_cache_create(
      device->base_device, identifier, out_executable_cache);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_executable_cache &&
      *out_executable_cache) {
    iree_hal_executable_cache_release(*out_executable_cache);
    *out_executable_cache = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_import_file(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_memory_access_t access, iree_io_file_handle_t* handle,
    iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE,
      &pending_record));
  iree_status_t status = iree_hal_file_import(
      device->base_device, queue_affinity, access, handle, flags, out_file);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_file && *out_file) {
    iree_hal_file_release(*out_file);
    *out_file = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_create_semaphore(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    uint64_t initial_value, iree_hal_semaphore_flags_t flags,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE,
      &pending_record));
  iree_status_t status = iree_hal_semaphore_create(
      device->base_device, queue_affinity, initial_value, flags, out_semaphore);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_semaphore && *out_semaphore) {
    iree_hal_semaphore_release(*out_semaphore);
    *out_semaphore = NULL;
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
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA,
      &pending_record));
  iree_status_t status = iree_hal_device_queue_alloca(
      device->base_device, queue_affinity, wait_semaphore_list,
      signal_semaphore_list, pool, params, allocation_size, flags, out_buffer);
  status = iree_hal_replay_device_complete_operation(&pending_record, status);
  if (!iree_status_is_ok(status) && out_buffer && *out_buffer) {
    iree_hal_buffer_release(*out_buffer);
    *out_buffer = NULL;
  }
  return status;
}

static iree_status_t iree_hal_replay_device_queue_dealloca(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_dealloca(device->base_device, queue_affinity,
                                     wait_semaphore_list, signal_semaphore_list,
                                     buffer, flags));
}

static iree_status_t iree_hal_replay_device_queue_fill(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_fill(device->base_device, queue_affinity,
                                 wait_semaphore_list, signal_semaphore_list,
                                 target_buffer, target_offset, length, pattern,
                                 pattern_length, flags));
}

static iree_status_t iree_hal_replay_device_queue_update(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_update(device->base_device, queue_affinity,
                                   wait_semaphore_list, signal_semaphore_list,
                                   source_buffer, source_offset, target_buffer,
                                   target_offset, length, flags));
}

static iree_status_t iree_hal_replay_device_queue_copy(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_copy(device->base_device, queue_affinity,
                                 wait_semaphore_list, signal_semaphore_list,
                                 source_buffer, source_offset, target_buffer,
                                 target_offset, length, flags));
}

static iree_status_t iree_hal_replay_device_queue_read(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_read(device->base_device, queue_affinity,
                                 wait_semaphore_list, signal_semaphore_list,
                                 source_file, source_offset, target_buffer,
                                 target_offset, length, flags));
}

static iree_status_t iree_hal_replay_device_queue_write(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_write(device->base_device, queue_affinity,
                                  wait_semaphore_list, signal_semaphore_list,
                                  source_buffer, source_offset, target_file,
                                  target_offset, length, flags));
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
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_dispatch(device->base_device, queue_affinity,
                                     wait_semaphore_list, signal_semaphore_list,
                                     executable, export_ordinal, config,
                                     constants, bindings, flags));
}

static iree_status_t iree_hal_replay_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_replay_device_t* device = iree_hal_replay_device_cast(base_device);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_device_begin_operation(
      device, IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE,
      &pending_record));
  return iree_hal_replay_device_complete_operation(
      &pending_record,
      iree_hal_device_queue_execute(device->base_device, queue_affinity,
                                    wait_semaphore_list, signal_semaphore_list,
                                    command_buffer, binding_table, flags));
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

  iree_status_t status = iree_hal_replay_recorder_record_device_object(
      recorder, &device->device_id);
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
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
