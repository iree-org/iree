// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/profile.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/threading/mutex.h"
#include "iree/hal/local/local_executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_local_profile_recorder_t
//===----------------------------------------------------------------------===//

// Default number of records retained per enabled event stream between flushes.
#define IREE_HAL_LOCAL_PROFILE_DEFAULT_EVENT_CAPACITY 4096

typedef struct iree_hal_local_profile_event_ring_t {
  // Storage for fixed-size event records, or NULL when the stream is disabled.
  void* records;

  // Size in bytes of each event record in |records|.
  iree_host_size_t record_size;

  // Power-of-two number of event records in |records|.
  iree_host_size_t capacity;

  // Bit mask used to wrap absolute positions into |records|.
  iree_host_size_t mask;

  // Absolute position of the first unflushed event record.
  uint64_t read_position;

  // Absolute position one past the last appended event record.
  uint64_t write_position;

  // Next nonzero event id assigned to a captured record.
  uint64_t next_event_id;

  // Records dropped since the last successful truncated flush.
  uint64_t dropped_record_count;
} iree_hal_local_profile_event_ring_t;

typedef struct iree_hal_local_profile_event_ring_snapshot_t {
  // Absolute read position captured for this flush attempt.
  uint64_t read_position;

  // Number of event records captured for this flush attempt.
  iree_host_size_t record_count;

  // Dropped records captured for this flush attempt.
  uint64_t dropped_record_count;

  // First contiguous record span in the ring.
  const void* first_records;

  // Number of records in |first_records|.
  iree_host_size_t first_record_count;

  // Second contiguous record span after ring wraparound, or NULL.
  const void* second_records;

  // Number of records in |second_records|.
  iree_host_size_t second_record_count;
} iree_hal_local_profile_event_ring_snapshot_t;

typedef struct iree_hal_local_profile_id_set_t {
  // Open-addressed nonzero ids whose metadata was emitted.
  uint64_t* ids;

  // Power-of-two slot count in |ids|.
  iree_host_size_t capacity;

  // Number of occupied id slots.
  iree_host_size_t count;
} iree_hal_local_profile_id_set_t;

struct iree_hal_local_profile_recorder_t {
  // Host allocator used for recorder-owned storage.
  iree_allocator_t host_allocator;

  // Guards event rings while producers append and flush snapshots advance.
  iree_slim_mutex_t mutex;

  // Serializes flush operations while sink callbacks are being invoked.
  iree_slim_mutex_t flush_mutex;

  // Copied human-readable producer name used on emitted chunks.
  iree_string_view_t name;

  // Allocated storage backing |name| when the caller provided one.
  char* name_storage;

  // Owned profiling options for the active session.
  iree_hal_device_profiling_options_t options;

  // Opaque storage backing borrowed pointers in |options|, or NULL.
  iree_hal_device_profiling_options_storage_t* options_storage;

  // Process-local profiling session identifier.
  uint64_t session_id;

  // True while the sink session has begun and has not been ended.
  bool active;

  // Single allocation backing all enabled event rings.
  void* event_storage;

  // Ring of host queue event records.
  iree_hal_local_profile_event_ring_t queue_event_ring;

  // Ring of host execution span records.
  iree_hal_local_profile_event_ring_t host_execution_event_ring;

  // Ring of memory lifecycle event records.
  iree_hal_local_profile_event_ring_t memory_event_ring;

  // Ring of command-buffer region event records.
  iree_hal_local_profile_event_ring_t command_region_event_ring;

  // Metadata ids already emitted to the session sink.
  struct {
    // Executable ids whose metadata was emitted.
    iree_hal_local_profile_id_set_t executables;

    // Command-buffer ids whose metadata was emitted.
    iree_hal_local_profile_id_set_t command_buffers;
  } emitted;
};

static bool iree_hal_local_profile_recorder_requests_data(
    const iree_hal_local_profile_recorder_t* recorder,
    iree_hal_device_profiling_data_families_t data_families) {
  return iree_hal_device_profiling_options_requests_data(&recorder->options,
                                                         data_families);
}

static iree_hal_device_profiling_data_families_t
iree_hal_local_profile_recorder_lightweight_statistics_data_families(void) {
  return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
}

static iree_hal_device_profiling_options_t
iree_hal_local_profile_recorder_resolve_profiling_options(
    const iree_hal_device_profiling_options_t* profiling_options) {
  iree_hal_device_profiling_options_t resolved_options = *profiling_options;
  if (resolved_options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      iree_hal_device_profiling_options_requests_lightweight_statistics(
          profiling_options)) {
    resolved_options.data_families =
        iree_hal_local_profile_recorder_lightweight_statistics_data_families();
  }
  if (iree_hal_device_profiling_options_requests_data(
          &resolved_options,
          IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    resolved_options.data_families |=
        IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  }
  if (iree_hal_device_profiling_options_requests_data(
          &resolved_options,
          IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS)) {
    resolved_options.data_families |=
        IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA;
  }
  resolved_options.flags &=
      ~IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  return resolved_options;
}

static iree_status_t iree_hal_local_profile_recorder_validate_records(
    const iree_hal_local_profile_recorder_options_t* recorder_options) {
  if (recorder_options->device_record_count == 0 ||
      !recorder_options->device_records) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling requires at least one device metadata record");
  }
  if (recorder_options->queue_record_count == 0 ||
      !recorder_options->queue_records) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling requires at least one queue metadata record");
  }
  for (iree_host_size_t i = 0; i < recorder_options->device_record_count; ++i) {
    const iree_hal_profile_device_record_t* record =
        &recorder_options->device_records[i];
    if (record->record_length != sizeof(*record)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling device metadata record %" PRIhsz
                              " has unsupported length %" PRIu32,
                              i, record->record_length);
    }
    if (record->physical_device_ordinal == UINT32_MAX ||
        record->queue_count == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling device metadata record %" PRIhsz
                              " has invalid queue identity",
                              i);
    }
  }
  for (iree_host_size_t i = 0; i < recorder_options->queue_record_count; ++i) {
    const iree_hal_profile_queue_record_t* record =
        &recorder_options->queue_records[i];
    if (record->record_length != sizeof(*record)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling queue metadata record %" PRIhsz
                              " has unsupported length %" PRIu32,
                              i, record->record_length);
    }
    if (record->physical_device_ordinal == UINT32_MAX ||
        record->queue_ordinal == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "local profiling queue metadata record %" PRIhsz
                              " has invalid queue identity",
                              i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_validate_profiling_options(
    const iree_hal_device_profiling_options_t* profiling_options) {
  if (profiling_options->flags != IREE_HAL_DEVICE_PROFILING_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported local profiling flags 0x%x",
                            profiling_options->flags);
  }
  const iree_hal_device_profiling_data_families_t supported_data_families =
      iree_hal_local_profile_recorder_supported_data_families();
  const iree_hal_device_profiling_data_families_t unsupported_data_families =
      profiling_options->data_families & ~supported_data_families;
  if (unsupported_data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported local profiling data families 0x%" PRIx64,
        unsupported_data_families);
  }
  if (profiling_options->data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE &&
      !profiling_options->sink) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling with requested data families requires a profile sink");
  }
  if (profiling_options->capture_filter.reserved0 != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling capture filter reserved fields must be zero");
  }
  if (!iree_hal_profile_capture_filter_is_default(
          &profiling_options->capture_filter)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "local profiling does not support capture filters yet");
  }
  if (profiling_options->counter_set_count != 0 ||
      profiling_options->counter_sets) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "local profiling does not support counter set selections");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_normalize_capacity(
    iree_host_size_t requested_capacity, iree_host_size_t* out_capacity) {
  iree_host_size_t capacity =
      requested_capacity != 0 ? requested_capacity
                              : IREE_HAL_LOCAL_PROFILE_DEFAULT_EVENT_CAPACITY;
  capacity = iree_host_size_next_power_of_two(capacity);
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(capacity))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local profiling event capacity is too large");
  }
  *out_capacity = capacity;
  return iree_ok_status();
}

static void iree_hal_local_profile_event_ring_initialize(
    void* records, iree_host_size_t record_size, iree_host_size_t capacity,
    iree_hal_local_profile_event_ring_t* out_ring) {
  memset(out_ring, 0, sizeof(*out_ring));
  out_ring->records = records;
  out_ring->record_size = record_size;
  out_ring->capacity = capacity;
  out_ring->mask = capacity - 1;
  out_ring->next_event_id = 1;
}

static iree_status_t iree_hal_local_profile_recorder_allocate_events(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_recorder_options_t* recorder_options) {
  iree_host_size_t queue_event_capacity = 0;
  iree_host_size_t host_execution_event_capacity = 0;
  iree_host_size_t memory_event_capacity = 0;
  iree_host_size_t command_region_event_capacity = 0;
  if (iree_hal_local_profile_recorder_requests_data(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_normalize_capacity(
        recorder_options->queue_event_capacity, &queue_event_capacity));
  }
  if (iree_hal_local_profile_recorder_requests_data(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_normalize_capacity(
        recorder_options->host_execution_event_capacity,
        &host_execution_event_capacity));
  }
  if (iree_hal_local_profile_recorder_requests_data(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_normalize_capacity(
        recorder_options->memory_event_capacity, &memory_event_capacity));
  }
  if (iree_hal_local_profile_recorder_requests_data(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS)) {
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_normalize_capacity(
        recorder_options->command_region_event_capacity,
        &command_region_event_capacity));
  }

  iree_host_size_t queue_events_offset = 0;
  iree_host_size_t host_execution_events_offset = 0;
  iree_host_size_t memory_events_offset = 0;
  iree_host_size_t command_region_events_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          queue_event_capacity, iree_hal_profile_queue_event_t,
          iree_alignof(iree_hal_profile_queue_event_t), &queue_events_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          host_execution_event_capacity,
          iree_hal_profile_host_execution_event_t,
          iree_alignof(iree_hal_profile_host_execution_event_t),
          &host_execution_events_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          memory_event_capacity, iree_hal_profile_memory_event_t,
          iree_alignof(iree_hal_profile_memory_event_t), &memory_events_offset),
      IREE_STRUCT_FIELD_ALIGNED(
          command_region_event_capacity,
          iree_hal_profile_command_region_event_t,
          iree_alignof(iree_hal_profile_command_region_event_t),
          &command_region_events_offset)));
  if (total_size == 0) return iree_ok_status();

  void* event_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(recorder->host_allocator,
                                             total_size, &event_storage));
  memset(event_storage, 0, total_size);
  recorder->event_storage = event_storage;

  if (queue_event_capacity != 0) {
    iree_hal_local_profile_event_ring_initialize(
        (uint8_t*)event_storage + queue_events_offset,
        sizeof(iree_hal_profile_queue_event_t), queue_event_capacity,
        &recorder->queue_event_ring);
  }
  if (host_execution_event_capacity != 0) {
    iree_hal_local_profile_event_ring_initialize(
        (uint8_t*)event_storage + host_execution_events_offset,
        sizeof(iree_hal_profile_host_execution_event_t),
        host_execution_event_capacity, &recorder->host_execution_event_ring);
  }
  if (memory_event_capacity != 0) {
    iree_hal_local_profile_event_ring_initialize(
        (uint8_t*)event_storage + memory_events_offset,
        sizeof(iree_hal_profile_memory_event_t), memory_event_capacity,
        &recorder->memory_event_ring);
  }
  if (command_region_event_capacity != 0) {
    iree_hal_local_profile_event_ring_initialize(
        (uint8_t*)event_storage + command_region_events_offset,
        sizeof(iree_hal_profile_command_region_event_t),
        command_region_event_capacity, &recorder->command_region_event_ring);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_copy_name(
    iree_hal_local_profile_recorder_t* recorder, iree_string_view_t name) {
  if (iree_string_view_is_empty(name)) {
    recorder->name = IREE_SV("local");
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      recorder->host_allocator, iree_make_const_byte_span(name.data, name.size),
      (void**)&recorder->name_storage));
  recorder->name = iree_make_string_view(recorder->name_storage, name.size);
  return iree_ok_status();
}

static iree_hal_profile_chunk_metadata_t
iree_hal_local_profile_recorder_metadata(
    const iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type) {
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_profile_chunk_metadata_default();
  metadata.content_type = content_type;
  metadata.name = recorder->name;
  metadata.session_id = recorder->session_id;
  return metadata;
}

static iree_status_t iree_hal_local_profile_recorder_write_records(
    iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type, const void* records,
    iree_host_size_t record_count, iree_host_size_t record_size) {
  if (record_count == 0) return iree_ok_status();
  iree_host_size_t byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(record_count, record_size,
                                                &byte_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local profiling chunk size overflow for %" PRIhsz
                            " records",
                            record_count);
  }
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_metadata(recorder, content_type);
  iree_const_byte_span_t iovec =
      iree_make_const_byte_span(records, byte_length);
  return iree_hal_profile_sink_write(recorder->options.sink, &metadata, 1,
                                     &iovec);
}

static iree_status_t iree_hal_local_profile_recorder_write_metadata(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_recorder_options_t* recorder_options) {
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_DEVICES,
      recorder_options->device_records, recorder_options->device_record_count,
      sizeof(*recorder_options->device_records)));
  return iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUES,
      recorder_options->queue_records, recorder_options->queue_record_count,
      sizeof(*recorder_options->queue_records));
}

iree_status_t iree_hal_local_profile_recorder_create(
    const iree_hal_local_profile_recorder_options_t* recorder_options,
    const iree_hal_device_profiling_options_t* profiling_options,
    iree_allocator_t host_allocator,
    iree_hal_local_profile_recorder_t** out_recorder) {
  IREE_ASSERT_ARGUMENT(recorder_options);
  IREE_ASSERT_ARGUMENT(profiling_options);
  IREE_ASSERT_ARGUMENT(out_recorder);
  *out_recorder = NULL;

  const iree_hal_device_profiling_flags_t supported_flags =
      IREE_HAL_DEVICE_PROFILING_FLAG_LIGHTWEIGHT_STATISTICS;
  if (iree_any_bit_set(profiling_options->flags, ~supported_flags)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported local profiling flags 0x%x",
                            profiling_options->flags & ~supported_flags);
  }
  iree_hal_device_profiling_options_t resolved_options =
      iree_hal_local_profile_recorder_resolve_profiling_options(
          profiling_options);
  if (resolved_options.data_families == IREE_HAL_DEVICE_PROFILING_DATA_NONE) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_validate_profiling_options(
          &resolved_options));
  IREE_RETURN_IF_ERROR(
      iree_hal_local_profile_recorder_validate_records(recorder_options));

  iree_hal_local_profile_recorder_t* recorder = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*recorder),
                                             (void**)&recorder));
  memset(recorder, 0, sizeof(*recorder));
  recorder->host_allocator = host_allocator;
  recorder->session_id = recorder_options->session_id;
  iree_slim_mutex_initialize(&recorder->mutex);
  iree_slim_mutex_initialize(&recorder->flush_mutex);

  bool sink_session_begun = false;
  iree_status_t status = iree_hal_local_profile_recorder_copy_name(
      recorder, recorder_options->name);
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_profiling_options_clone(
        &resolved_options, host_allocator, &recorder->options,
        &recorder->options_storage);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_allocate_events(recorder,
                                                             recorder_options);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_local_profile_recorder_metadata(
            recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION);
    status =
        iree_hal_profile_sink_begin_session(recorder->options.sink, &metadata);
    sink_session_begun = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_metadata(recorder,
                                                            recorder_options);
  }

  if (iree_status_is_ok(status)) {
    recorder->active = true;
    *out_recorder = recorder;
  } else {
    if (sink_session_begun) {
      iree_hal_profile_chunk_metadata_t metadata =
          iree_hal_local_profile_recorder_metadata(
              recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION);
      status = iree_status_join(status, iree_hal_profile_sink_end_session(
                                            recorder->options.sink, &metadata,
                                            iree_status_code(status)));
    }
    iree_hal_local_profile_recorder_destroy(recorder);
  }
  return status;
}

void iree_hal_local_profile_recorder_destroy(
    iree_hal_local_profile_recorder_t* recorder) {
  if (!recorder) return;
  IREE_ASSERT(!recorder->active,
              "active local profile recorders must be ended before destroy");
  iree_allocator_t host_allocator = recorder->host_allocator;
  iree_allocator_free(host_allocator, recorder->event_storage);
  iree_allocator_free(host_allocator, recorder->emitted.executables.ids);
  iree_allocator_free(host_allocator, recorder->emitted.command_buffers.ids);
  iree_hal_device_profiling_options_storage_free(recorder->options_storage,
                                                 host_allocator);
  iree_allocator_free(host_allocator, recorder->name_storage);
  iree_slim_mutex_deinitialize(&recorder->flush_mutex);
  iree_slim_mutex_deinitialize(&recorder->mutex);
  iree_allocator_free(host_allocator, recorder);
}

bool iree_hal_local_profile_recorder_is_enabled(
    const iree_hal_local_profile_recorder_t* recorder,
    iree_hal_device_profiling_data_families_t data_families) {
  return recorder && recorder->active &&
         iree_hal_device_profiling_options_requests_data(&recorder->options,
                                                         data_families);
}

static iree_host_size_t iree_hal_local_profile_hash_id(
    uint64_t id, iree_host_size_t capacity) {
  id ^= id >> 33;
  id *= 0xff51afd7ed558ccdull;
  id ^= id >> 33;
  id *= 0xc4ceb9fe1a85ec53ull;
  id ^= id >> 33;
  return (iree_host_size_t)id & (capacity - 1);
}

static bool iree_hal_local_profile_id_set_find_slot(
    const iree_hal_local_profile_id_set_t* set, uint64_t id,
    iree_host_size_t* out_slot) {
  IREE_ASSERT(set->capacity != 0);
  iree_host_size_t slot = iree_hal_local_profile_hash_id(id, set->capacity);
  while (set->ids[slot] != 0) {
    if (set->ids[slot] == id) {
      *out_slot = slot;
      return true;
    }
    slot = (slot + 1) & (set->capacity - 1);
  }
  *out_slot = slot;
  return false;
}

static iree_status_t iree_hal_local_profile_id_set_reserve(
    iree_allocator_t host_allocator, iree_hal_local_profile_id_set_t* set,
    iree_host_size_t minimum_capacity, const char* label) {
  if (IREE_UNLIKELY(minimum_capacity > IREE_HOST_SIZE_MAX / 2)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local profiling %s metadata set is too large",
                            label);
  }
  const iree_host_size_t required_capacity = minimum_capacity * 2;
  iree_host_size_t capacity = set->capacity;
  if (required_capacity <= capacity) return iree_ok_status();
  capacity = capacity != 0 ? capacity : 16;
  while (required_capacity > capacity) {
    if (IREE_UNLIKELY(capacity > IREE_HOST_SIZE_MAX / 2)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "local profiling %s metadata set is too large",
                              label);
    }
    capacity *= 2;
  }

  iree_host_size_t byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(capacity, sizeof(*set->ids),
                                                &byte_length))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local profiling %s metadata set size overflow",
                            label);
  }
  uint64_t* ids = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, byte_length, (void**)&ids));
  memset(ids, 0, byte_length);
  iree_hal_local_profile_id_set_t new_set = {
      .ids = ids,
      .capacity = capacity,
  };
  for (iree_host_size_t i = 0; i < set->capacity; ++i) {
    const uint64_t id = set->ids[i];
    if (id == 0) continue;
    iree_host_size_t slot = 0;
    iree_hal_local_profile_id_set_find_slot(&new_set, id, &slot);
    ids[slot] = id;
  }
  new_set.count = set->count;

  iree_allocator_free(host_allocator, set->ids);
  *set = new_set;
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_mark_id_emitted(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_local_profile_id_set_t* set, uint64_t id, const char* label,
    bool* out_should_emit) {
  *out_should_emit = false;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_local_profile_id_set_reserve(
      recorder->host_allocator, set, set->count + 1, label);
  if (iree_status_is_ok(status)) {
    iree_host_size_t slot = 0;
    const bool already_emitted =
        iree_hal_local_profile_id_set_find_slot(set, id, &slot);
    if (!already_emitted) {
      set->ids[slot] = id;
      ++set->count;
      *out_should_emit = true;
    }
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

static bool iree_hal_local_profile_recorder_has_emitted_id(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_id_set_t* set, uint64_t id) {
  iree_slim_mutex_lock(&recorder->mutex);
  bool has_emitted = false;
  if (set->capacity != 0) {
    iree_host_size_t slot = 0;
    has_emitted = iree_hal_local_profile_id_set_find_slot(set, id, &slot);
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return has_emitted;
}

static iree_status_t iree_hal_local_profile_executable_export_record_length(
    iree_string_view_t name, iree_host_size_t* out_record_length) {
  *out_record_length = 0;
  if (IREE_UNLIKELY(name.size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile executable export name is too long");
  }
  iree_host_size_t record_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_profile_executable_export_record_t), &record_length,
      IREE_STRUCT_FIELD(name.size, uint8_t, NULL)));
  if (IREE_UNLIKELY(record_length > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile executable export record length exceeds uint32_t");
  }
  *out_record_length = record_length;
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_executable_export_data_length(
    iree_hal_executable_t* executable, iree_host_size_t export_count,
    iree_host_size_t* out_data_length) {
  *out_data_length = 0;
  iree_host_size_t data_length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
       ++i) {
    iree_hal_executable_export_info_t export_info = {0};
    status = iree_hal_executable_export_info(
        executable, (iree_hal_executable_export_ordinal_t)i, &export_info);
    iree_host_size_t record_length = 0;
    if (iree_status_is_ok(status)) {
      status = iree_hal_local_profile_executable_export_record_length(
          export_info.name, &record_length);
    }
    if (iree_status_is_ok(status) &&
        IREE_UNLIKELY(!iree_host_size_checked_add(data_length, record_length,
                                                  &data_length))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile executable export metadata length overflow");
    }
  }
  if (iree_status_is_ok(status)) *out_data_length = data_length;
  return status;
}

static iree_status_t iree_hal_local_profile_append_executable_export_records(
    uint64_t executable_id, iree_hal_executable_t* executable,
    iree_host_size_t export_count, uint8_t* target_data) {
  uint8_t* cursor = target_data;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    iree_hal_executable_export_info_t export_info = {0};
    IREE_RETURN_IF_ERROR(iree_hal_executable_export_info(
        executable, (iree_hal_executable_export_ordinal_t)i, &export_info));

    iree_host_size_t record_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_local_profile_executable_export_record_length(
        export_info.name, &record_length));

    iree_hal_profile_executable_export_record_t record =
        iree_hal_profile_executable_export_record_default();
    record.record_length = (uint32_t)record_length;
    record.executable_id = executable_id;
    record.export_ordinal = (uint32_t)i;
    record.constant_count = export_info.constant_count;
    record.binding_count = export_info.binding_count;
    record.parameter_count = export_info.parameter_count;
    memcpy(record.workgroup_size, export_info.workgroup_size,
           sizeof(record.workgroup_size));
    record.name_length = (uint32_t)export_info.name.size;

    memcpy(cursor, &record, sizeof(record));
    if (export_info.name.size > 0) {
      memcpy(cursor + sizeof(record), export_info.name.data,
             export_info.name.size);
    }
    cursor += record_length;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_write_span(
    iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type, iree_const_byte_span_t span) {
  if (span.data_length == 0) return iree_ok_status();
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_metadata(recorder, content_type);
  return iree_hal_profile_sink_write(recorder->options.sink, &metadata, 1,
                                     &span);
}

iree_status_t iree_hal_local_profile_recorder_record_executable(
    iree_hal_local_profile_recorder_t* recorder,
    iree_hal_executable_t* executable) {
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)) {
    return iree_ok_status();
  }

  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);
  const uint64_t executable_id =
      iree_hal_local_executable_profile_id(local_executable);
  if (iree_hal_local_profile_recorder_has_emitted_id(
          recorder, &recorder->emitted.executables, executable_id)) {
    return iree_ok_status();
  }

  const iree_host_size_t export_count =
      iree_hal_executable_export_count(executable);
  if (IREE_UNLIKELY(export_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile executable export count exceeds uint32_t");
  }

  iree_hal_profile_executable_record_t executable_record =
      iree_hal_profile_executable_record_default();
  executable_record.executable_id = executable_id;
  executable_record.export_count = (uint32_t)export_count;

  iree_host_size_t export_data_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_executable_export_data_length(
      executable, export_count, &export_data_length));
  uint8_t* export_data = NULL;
  iree_status_t status = iree_ok_status();
  if (export_data_length != 0) {
    status = iree_allocator_malloc(recorder->host_allocator, export_data_length,
                                   (void**)&export_data);
  }
  if (iree_status_is_ok(status) && export_data_length != 0) {
    status = iree_hal_local_profile_append_executable_export_records(
        executable_id, executable, export_count, export_data);
  }

  bool should_emit = false;
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_mark_id_emitted(
        recorder, &recorder->emitted.executables, executable_id, "executable",
        &should_emit);
  }
  if (iree_status_is_ok(status) && should_emit) {
    status = iree_hal_local_profile_recorder_write_records(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES, &executable_record,
        /*record_count=*/1, sizeof(executable_record));
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_span(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS,
        should_emit ? iree_make_const_byte_span(export_data, export_data_length)
                    : iree_const_byte_span_empty());
  }
  iree_allocator_free(recorder->host_allocator, export_data);
  return status;
}

iree_status_t iree_hal_local_profile_recorder_record_command_buffer(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    iree_host_size_t operation_count,
    const iree_hal_profile_command_operation_record_t* operations) {
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA)) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!command_buffer ||
                    command_buffer->command_buffer_id == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling command-buffer metadata requires a nonzero id");
  }
  if (IREE_UNLIKELY(operation_count != 0 && !operations)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local profiling command-buffer operation records are required");
  }

  bool should_emit = false;
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_mark_id_emitted(
      recorder, &recorder->emitted.command_buffers,
      command_buffer->command_buffer_id, "command-buffer", &should_emit));
  if (!should_emit) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS, command_buffer,
      /*record_count=*/1, sizeof(*command_buffer)));
  return iree_hal_local_profile_recorder_write_records(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS, operations,
      operation_count, sizeof(*operations));
}

static bool iree_hal_local_profile_queue_scope_is_valid(
    const iree_hal_local_profile_queue_scope_t* scope) {
  return scope->physical_device_ordinal != UINT32_MAX &&
         scope->queue_ordinal != UINT32_MAX;
}

static iree_hal_profile_queue_event_t* iree_hal_local_profile_queue_event_at(
    const iree_hal_local_profile_event_ring_t* ring, uint64_t position) {
  iree_hal_profile_queue_event_t* records =
      (iree_hal_profile_queue_event_t*)ring->records;
  return &records[position & ring->mask];
}

static iree_hal_profile_host_execution_event_t*
iree_hal_local_profile_host_execution_event_at(
    const iree_hal_local_profile_event_ring_t* ring, uint64_t position) {
  iree_hal_profile_host_execution_event_t* records =
      (iree_hal_profile_host_execution_event_t*)ring->records;
  return &records[position & ring->mask];
}

static iree_hal_profile_memory_event_t* iree_hal_local_profile_memory_event_at(
    const iree_hal_local_profile_event_ring_t* ring, uint64_t position) {
  iree_hal_profile_memory_event_t* records =
      (iree_hal_profile_memory_event_t*)ring->records;
  return &records[position & ring->mask];
}

static iree_hal_profile_command_region_event_t*
iree_hal_local_profile_command_region_event_at(
    const iree_hal_local_profile_event_ring_t* ring, uint64_t position) {
  iree_hal_profile_command_region_event_t* records =
      (iree_hal_profile_command_region_event_t*)ring->records;
  return &records[position & ring->mask];
}

void iree_hal_local_profile_recorder_append_queue_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_queue_event_info_t* event_info,
    uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(event_info);
  if (out_event_id) *out_event_id = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS)) {
    return;
  }
  const bool is_valid =
      iree_hal_local_profile_queue_scope_is_valid(&event_info->scope) &&
      event_info->type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
  IREE_ASSERT(is_valid);
  if (IREE_UNLIKELY(!is_valid)) return;

  iree_hal_local_profile_event_ring_t* ring = &recorder->queue_event_ring;
  iree_slim_mutex_lock(&recorder->mutex);
  if (ring->write_position - ring->read_position >= ring->capacity) {
    ++ring->dropped_record_count;
    iree_slim_mutex_unlock(&recorder->mutex);
    return;
  }

  const uint64_t event_id = ring->next_event_id++;
  iree_hal_profile_queue_event_t* event =
      iree_hal_local_profile_queue_event_at(ring, ring->write_position);
  *event = iree_hal_profile_queue_event_default();
  event->type = event_info->type;
  event->flags = event_info->flags;
  event->dependency_strategy = event_info->dependency_strategy;
  event->event_id = event_id;
  event->host_time_ns = event_info->host_time_ns != 0 ? event_info->host_time_ns
                                                      : iree_time_now();
  event->ready_host_time_ns = event_info->ready_host_time_ns;
  event->submission_id = event_info->submission_id;
  event->command_buffer_id = event_info->command_buffer_id;
  event->allocation_id = event_info->allocation_id;
  event->stream_id = event_info->scope.stream_id;
  event->physical_device_ordinal = event_info->scope.physical_device_ordinal;
  event->queue_ordinal = event_info->scope.queue_ordinal;
  event->wait_count = event_info->wait_count;
  event->signal_count = event_info->signal_count;
  event->barrier_count = event_info->barrier_count;
  event->operation_count = event_info->operation_count;
  event->payload_length = event_info->payload_length;
  ++ring->write_position;
  if (out_event_id) *out_event_id = event_id;
  iree_slim_mutex_unlock(&recorder->mutex);
}

void iree_hal_local_profile_recorder_append_host_execution_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_host_execution_event_info_t* event_info,
    uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(event_info);
  if (out_event_id) *out_event_id = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_HOST_EXECUTION_EVENTS)) {
    return;
  }
  const bool is_valid =
      iree_hal_local_profile_queue_scope_is_valid(&event_info->scope) &&
      event_info->type != IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_NONE;
  IREE_ASSERT(is_valid);
  if (IREE_UNLIKELY(!is_valid)) return;

  iree_time_t start_time_ns = event_info->start_host_time_ns;
  iree_time_t end_time_ns = event_info->end_host_time_ns;
  if (start_time_ns == 0) start_time_ns = iree_time_now();
  if (end_time_ns == 0) end_time_ns = iree_time_now();
  const bool has_valid_range = end_time_ns >= start_time_ns;
  IREE_ASSERT(has_valid_range);
  if (IREE_UNLIKELY(!has_valid_range)) end_time_ns = start_time_ns;

  iree_hal_local_profile_event_ring_t* ring =
      &recorder->host_execution_event_ring;
  iree_slim_mutex_lock(&recorder->mutex);
  if (ring->write_position - ring->read_position >= ring->capacity) {
    ++ring->dropped_record_count;
    iree_slim_mutex_unlock(&recorder->mutex);
    return;
  }

  const uint64_t event_id = ring->next_event_id++;
  iree_hal_profile_host_execution_event_t* event =
      iree_hal_local_profile_host_execution_event_at(ring,
                                                     ring->write_position);
  *event = iree_hal_profile_host_execution_event_default();
  event->type = event_info->type;
  event->flags = event_info->flags;
  event->status_code = event_info->status_code;
  event->event_id = event_id;
  event->submission_id = event_info->submission_id;
  event->command_buffer_id = event_info->command_buffer_id;
  event->executable_id = event_info->executable_id;
  event->allocation_id = event_info->allocation_id;
  event->stream_id = event_info->scope.stream_id;
  event->physical_device_ordinal = event_info->scope.physical_device_ordinal;
  event->queue_ordinal = event_info->scope.queue_ordinal;
  event->command_index = event_info->command_index;
  event->export_ordinal = event_info->export_ordinal;
  memcpy(event->workgroup_count, event_info->workgroup_count,
         sizeof(event->workgroup_count));
  memcpy(event->workgroup_size, event_info->workgroup_size,
         sizeof(event->workgroup_size));
  event->start_host_time_ns = start_time_ns;
  event->end_host_time_ns = end_time_ns;
  event->payload_length = event_info->payload_length;
  event->tile_count = event_info->tile_count;
  event->tile_duration_sum_ns = event_info->tile_duration_sum_ns;
  event->operation_count = event_info->operation_count;
  ++ring->write_position;
  if (out_event_id) *out_event_id = event_id;
  iree_slim_mutex_unlock(&recorder->mutex);
}

void iree_hal_local_profile_recorder_append_command_region_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_local_profile_command_region_event_info_t* event_info,
    uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(event_info);
  if (out_event_id) *out_event_id = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_COMMAND_REGION_EVENTS)) {
    return;
  }
  const bool is_valid =
      iree_hal_local_profile_queue_scope_is_valid(&event_info->scope) &&
      event_info->command_buffer_id != 0 &&
      event_info->command_region.index >= 0;
  IREE_ASSERT(is_valid);
  if (IREE_UNLIKELY(!is_valid)) return;

  iree_time_t start_time_ns = event_info->command_region.start_host_time_ns;
  iree_time_t end_time_ns = event_info->command_region.end_host_time_ns;
  if (start_time_ns == 0) start_time_ns = iree_time_now();
  if (end_time_ns == 0) end_time_ns = iree_time_now();
  const bool has_valid_range = end_time_ns >= start_time_ns;
  IREE_ASSERT(has_valid_range);
  if (IREE_UNLIKELY(!has_valid_range)) end_time_ns = start_time_ns;

  iree_hal_local_profile_event_ring_t* ring =
      &recorder->command_region_event_ring;
  iree_slim_mutex_lock(&recorder->mutex);
  if (ring->write_position - ring->read_position >= ring->capacity) {
    ++ring->dropped_record_count;
    iree_slim_mutex_unlock(&recorder->mutex);
    return;
  }

  const uint64_t event_id = ring->next_event_id++;
  iree_hal_profile_command_region_event_t* event =
      iree_hal_local_profile_command_region_event_at(ring,
                                                     ring->write_position);
  *event = iree_hal_profile_command_region_event_default();
  event->flags = event_info->flags;
  event->event_id = event_id;
  event->submission_id = event_info->submission_id;
  event->command_buffer_id = event_info->command_buffer_id;
  event->stream_id = event_info->scope.stream_id;
  event->queue.physical_device_ordinal =
      event_info->scope.physical_device_ordinal;
  event->queue.queue_ordinal = event_info->scope.queue_ordinal;
  event->command_region.block_sequence =
      event_info->command_region.block_sequence;
  event->command_region.epoch = event_info->command_region.epoch;
  event->command_region.index = event_info->command_region.index;
  event->command_region.dispatch_count =
      event_info->command_region.dispatch_count;
  event->command_region.tile_count = event_info->command_region.tile_count;
  event->command_region.width_bucket = event_info->command_region.width_bucket;
  event->command_region.lookahead_width_bucket =
      event_info->command_region.lookahead_width_bucket;
  event->command_region.useful_drain_count =
      event_info->command_region.useful_drain_count;
  event->command_region.no_work_drain_count =
      event_info->command_region.no_work_drain_count;
  event->command_region.tail_no_work.count =
      event_info->command_region.tail_no_work.count;
  event->command_region.tail_no_work.remaining_tiles.min =
      event_info->command_region.tail_no_work.remaining_tiles.min;
  event->command_region.tail_no_work.remaining_tiles.max =
      event_info->command_region.tail_no_work.remaining_tiles.max;
  memcpy(
      event->command_region.tail_no_work.remaining_tiles.bucket_counts,
      event_info->command_region.tail_no_work.remaining_tiles.bucket_counts,
      sizeof(event->command_region.tail_no_work.remaining_tiles.bucket_counts));
  event->command_region.first_useful_drain_start_host_time_ns =
      event_info->command_region.first_useful_drain_start_host_time_ns;
  event->command_region.last_useful_drain_end_host_time_ns =
      event_info->command_region.last_useful_drain_end_host_time_ns;
  event->command_region.tail_no_work.first_start_host_time_ns =
      event_info->command_region.tail_no_work.first_start_host_time_ns;
  event->command_region.tail_no_work.last_end_host_time_ns =
      event_info->command_region.tail_no_work.last_end_host_time_ns;
  event->command_region.tail_no_work.time_sums.start_offset_ns =
      event_info->command_region.tail_no_work.time_sums.start_offset_ns;
  event->command_region.tail_no_work.time_sums.drain_duration_ns =
      event_info->command_region.tail_no_work.time_sums.drain_duration_ns;
  event->command_region.start_host_time_ns = start_time_ns;
  event->command_region.end_host_time_ns = end_time_ns;
  event->next_command_region.index = event_info->next_command_region.index;
  event->next_command_region.tile_count =
      event_info->next_command_region.tile_count;
  event->next_command_region.width_bucket =
      event_info->next_command_region.width_bucket;
  event->next_command_region.lookahead_width_bucket =
      event_info->next_command_region.lookahead_width_bucket;
  event->scheduler.worker_count = event_info->scheduler.worker_count;
  event->scheduler.old_wake_budget = event_info->scheduler.old_wake_budget;
  event->scheduler.new_wake_budget = event_info->scheduler.new_wake_budget;
  event->scheduler.wake_delta = event_info->scheduler.wake_delta;
  event->retention.keep_active_count = event_info->retention.keep_active_count;
  event->retention.publish_keep_active_count =
      event_info->retention.publish_keep_active_count;
  event->retention.keep_warm_count = event_info->retention.keep_warm_count;
  ++ring->write_position;
  if (out_event_id) *out_event_id = event_id;
  iree_slim_mutex_unlock(&recorder->mutex);
}

static bool iree_hal_local_profile_memory_event_is_valid(
    const iree_hal_profile_memory_event_t* event) {
  if (event->type == IREE_HAL_PROFILE_MEMORY_EVENT_TYPE_NONE) return false;
  if (event->physical_device_ordinal == UINT32_MAX) return false;
  if (iree_all_bits_set(event->flags,
                        IREE_HAL_PROFILE_MEMORY_EVENT_FLAG_QUEUE_OPERATION) &&
      event->queue_ordinal == UINT32_MAX) {
    return false;
  }
  return true;
}

void iree_hal_local_profile_recorder_append_memory_event(
    iree_hal_local_profile_recorder_t* recorder,
    const iree_hal_profile_memory_event_t* event, uint64_t* out_event_id) {
  IREE_ASSERT_ARGUMENT(event);
  if (out_event_id) *out_event_id = 0;
  if (!iree_hal_local_profile_recorder_is_enabled(
          recorder, IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS)) {
    return;
  }
  const bool is_valid = iree_hal_local_profile_memory_event_is_valid(event);
  IREE_ASSERT(is_valid);
  if (IREE_UNLIKELY(!is_valid)) return;

  iree_hal_local_profile_event_ring_t* ring = &recorder->memory_event_ring;
  iree_slim_mutex_lock(&recorder->mutex);
  if (ring->write_position - ring->read_position >= ring->capacity) {
    ++ring->dropped_record_count;
    iree_slim_mutex_unlock(&recorder->mutex);
    return;
  }

  const uint64_t event_id = ring->next_event_id++;
  iree_hal_profile_memory_event_t* record =
      iree_hal_local_profile_memory_event_at(ring, ring->write_position);
  *record = *event;
  record->record_length = sizeof(*record);
  record->event_id = event_id;
  if (record->host_time_ns == 0) {
    record->host_time_ns = iree_time_now();
  }
  ++ring->write_position;
  if (out_event_id) *out_event_id = event_id;
  iree_slim_mutex_unlock(&recorder->mutex);
}

static void iree_hal_local_profile_event_ring_snapshot(
    const iree_hal_local_profile_event_ring_t* ring,
    iree_hal_local_profile_event_ring_snapshot_t* out_snapshot) {
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  if (!ring->records) return;

  out_snapshot->read_position = ring->read_position;
  out_snapshot->record_count =
      (iree_host_size_t)(ring->write_position - ring->read_position);
  out_snapshot->dropped_record_count = ring->dropped_record_count;
  IREE_ASSERT_LE(out_snapshot->record_count, ring->capacity);
  if (out_snapshot->record_count == 0) return;

  const iree_host_size_t first_record_index =
      (iree_host_size_t)(ring->read_position & ring->mask);
  out_snapshot->first_record_count =
      iree_min(out_snapshot->record_count, ring->capacity - first_record_index);
  out_snapshot->first_records =
      (const uint8_t*)ring->records + first_record_index * ring->record_size;
  out_snapshot->second_record_count =
      out_snapshot->record_count - out_snapshot->first_record_count;
  if (out_snapshot->second_record_count != 0) {
    out_snapshot->second_records = ring->records;
  }
}

static iree_status_t iree_hal_local_profile_append_iovec(
    const void* records, iree_host_size_t record_count,
    iree_host_size_t record_size, iree_host_size_t* iovec_count,
    iree_const_byte_span_t iovecs[2]) {
  if (record_count == 0) return iree_ok_status();
  iree_host_size_t byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(record_count, record_size,
                                                &byte_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "local profiling event chunk size overflow for %" PRIhsz " records",
        record_count);
  }
  iovecs[(*iovec_count)++] = iree_make_const_byte_span(records, byte_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_write_event_ring(
    iree_hal_local_profile_recorder_t* recorder,
    iree_string_view_t content_type,
    iree_hal_local_profile_event_ring_t* ring) {
  iree_hal_local_profile_event_ring_snapshot_t snapshot;
  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_local_profile_event_ring_snapshot(ring, &snapshot);
  iree_slim_mutex_unlock(&recorder->mutex);

  if (snapshot.record_count == 0 && snapshot.dropped_record_count == 0) {
    return iree_ok_status();
  }

  iree_const_byte_span_t iovecs[2];
  iree_host_size_t iovec_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_append_iovec(
      snapshot.first_records, snapshot.first_record_count, ring->record_size,
      &iovec_count, iovecs));
  IREE_RETURN_IF_ERROR(iree_hal_local_profile_append_iovec(
      snapshot.second_records, snapshot.second_record_count, ring->record_size,
      &iovec_count, iovecs));

  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_metadata(recorder, content_type);
  if (snapshot.dropped_record_count != 0) {
    metadata.flags |= IREE_HAL_PROFILE_CHUNK_FLAG_TRUNCATED;
    metadata.dropped_record_count = snapshot.dropped_record_count;
  }
  IREE_RETURN_IF_ERROR(iree_hal_profile_sink_write(
      recorder->options.sink, &metadata, iovec_count, iovecs));

  iree_slim_mutex_lock(&recorder->mutex);
  ring->read_position = snapshot.read_position + snapshot.record_count;
  if (ring->dropped_record_count >= snapshot.dropped_record_count) {
    ring->dropped_record_count -= snapshot.dropped_record_count;
  } else {
    ring->dropped_record_count = 0;
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_local_profile_recorder_flush_records(
    iree_hal_local_profile_recorder_t* recorder) {
  iree_slim_mutex_lock(&recorder->flush_mutex);
  iree_status_t status = iree_hal_local_profile_recorder_write_event_ring(
      recorder, IREE_HAL_PROFILE_CONTENT_TYPE_QUEUE_EVENTS,
      &recorder->queue_event_ring);
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_event_ring(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_HOST_EXECUTION_EVENTS,
        &recorder->host_execution_event_ring);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_event_ring(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_REGION_EVENTS,
        &recorder->command_region_event_ring);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_local_profile_recorder_write_event_ring(
        recorder, IREE_HAL_PROFILE_CONTENT_TYPE_MEMORY_EVENTS,
        &recorder->memory_event_ring);
  }
  iree_slim_mutex_unlock(&recorder->flush_mutex);
  return status;
}

iree_status_t iree_hal_local_profile_recorder_flush(
    iree_hal_local_profile_recorder_t* recorder) {
  if (!recorder || !recorder->active) return iree_ok_status();
  return iree_hal_local_profile_recorder_flush_records(recorder);
}

iree_status_t iree_hal_local_profile_recorder_end(
    iree_hal_local_profile_recorder_t* recorder) {
  if (!recorder || !recorder->active) return iree_ok_status();

  iree_status_t status =
      iree_hal_local_profile_recorder_flush_records(recorder);
  iree_hal_profile_chunk_metadata_t metadata =
      iree_hal_local_profile_recorder_metadata(
          recorder, IREE_HAL_PROFILE_CONTENT_TYPE_SESSION);
  status = iree_status_join(
      status, iree_hal_profile_sink_end_session(
                  recorder->options.sink, &metadata, iree_status_code(status)));
  recorder->active = false;
  return status;
}
