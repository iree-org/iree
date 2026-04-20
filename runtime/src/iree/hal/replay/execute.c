// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute.h"

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
#include <sys/stat.h>
#include <unistd.h>
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

#include "iree/hal/replay/digest.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/format.h"
#include "iree/io/file_handle.h"

#define IREE_HAL_REPLAY_EXECUTABLE_FORMAT_CAPACITY 128

typedef struct iree_hal_replay_object_entry_t {
  // Captured object type stored in this entry.
  iree_hal_replay_object_type_t type;
  // Retained HAL resource pointer for the captured object.
  union {
    iree_hal_device_t* device;
    iree_hal_allocator_t* allocator;
    iree_hal_buffer_t* buffer;
    iree_hal_command_buffer_t* command_buffer;
    iree_hal_executable_cache_t* executable_cache;
    iree_hal_executable_t* executable;
    iree_hal_semaphore_t* semaphore;
    iree_hal_event_t* event;
    iree_hal_file_t* file;
  } value;
} iree_hal_replay_object_entry_t;

typedef struct iree_hal_replay_executor_t {
  // Original replay file bytes.
  iree_const_byte_span_t file_contents;
  // Retained topology supplied by the caller.
  iree_hal_device_group_t* device_group;
  // Host allocator used for temporary replay state.
  iree_allocator_t host_allocator;
  // Execution options supplied by the caller.
  const iree_hal_replay_execute_options_t* options;
  // Dense session-local object table indexed by replay object id.
  iree_hal_replay_object_entry_t* objects;
  // Number of entries in |objects|.
  iree_host_size_t object_capacity;
  // Next caller-provided device consumed by a device object record.
  iree_host_size_t next_device_index;
} iree_hal_replay_executor_t;

static void iree_hal_replay_executor_release_entry(
    iree_hal_replay_object_entry_t* entry) {
  switch (entry->type) {
    case IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE:
      iree_hal_device_release(entry->value.device);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR:
      iree_hal_allocator_release(entry->value.allocator);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER:
      iree_hal_buffer_release(entry->value.buffer);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER:
      iree_hal_command_buffer_release(entry->value.command_buffer);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE_CACHE:
      iree_hal_executable_cache_release(entry->value.executable_cache);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE:
      iree_hal_executable_release(entry->value.executable);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE:
      iree_hal_semaphore_release(entry->value.semaphore);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_EVENT:
      iree_hal_event_release(entry->value.event);
      break;
    case IREE_HAL_REPLAY_OBJECT_TYPE_FILE:
      iree_hal_file_release(entry->value.file);
      break;
    default:
      break;
  }
  memset(entry, 0, sizeof(*entry));
}

static void iree_hal_replay_executor_deinitialize(
    iree_hal_replay_executor_t* executor) {
  if (!executor->objects) return;
  for (iree_host_size_t i = executor->object_capacity; i > 0; --i) {
    iree_hal_replay_executor_release_entry(&executor->objects[i - 1]);
  }
  iree_allocator_free(executor->host_allocator, executor->objects);
  executor->objects = NULL;
  executor->object_capacity = 0;
}

static iree_status_t iree_hal_replay_executor_lookup(
    iree_hal_replay_executor_t* executor, iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t expected_type,
    iree_hal_replay_object_entry_t** out_entry) {
  IREE_ASSERT_ARGUMENT(out_entry);
  *out_entry = NULL;
  if (IREE_UNLIKELY(object_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ||
                    object_id >= executor->object_capacity)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "replay object id %" PRIu64 " is not defined",
                            object_id);
  }
  iree_hal_replay_object_entry_t* entry = &executor->objects[object_id];
  if (IREE_UNLIKELY(entry->type != expected_type)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay object id %" PRIu64 " has type %s; expected %s", object_id,
        iree_hal_replay_object_type_string(entry->type),
        iree_hal_replay_object_type_string(expected_type));
  }
  *out_entry = entry;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_store(
    iree_hal_replay_executor_t* executor, iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_object_entry_t entry) {
  if (IREE_UNLIKELY(object_id == IREE_HAL_REPLAY_OBJECT_ID_NONE ||
                    object_id >= executor->object_capacity)) {
    iree_hal_replay_executor_release_entry(&entry);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay object id %" PRIu64
                            " exceeds object table capacity",
                            object_id);
  }
  iree_hal_replay_object_entry_t* existing = &executor->objects[object_id];
  if (existing->type != IREE_HAL_REPLAY_OBJECT_TYPE_NONE) {
    iree_hal_replay_executor_release_entry(&entry);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "replay object id %" PRIu64 " is already assigned",
                            object_id);
  }
  entry.type = object_type;
  *existing = entry;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_scan_object_capacity(
    iree_const_byte_span_t file_contents, iree_host_size_t* out_capacity) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_file_parse_header(file_contents, &file_header, &offset));
  iree_host_size_t valid_length =
      file_header.file_length ? (iree_host_size_t)file_header.file_length
                              : file_contents.data_length;

  iree_hal_replay_object_id_t max_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  while (offset < valid_length) {
    iree_hal_replay_file_record_t record;
    IREE_RETURN_IF_ERROR(iree_hal_replay_file_parse_record(
        file_contents, offset, &record, &offset));
    if (record.header.object_id > max_object_id) {
      max_object_id = record.header.object_id;
    }
    if (record.header.related_object_id > max_object_id) {
      max_object_id = record.header.related_object_id;
    }
  }
  if (IREE_UNLIKELY(max_object_id >= IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay object id exceeds host size");
  }
  *out_capacity = (iree_host_size_t)max_object_id + 1;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_initialize(
    iree_hal_replay_executor_t* executor, iree_const_byte_span_t file_contents,
    iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator) {
  memset(executor, 0, sizeof(*executor));
  executor->file_contents = file_contents;
  executor->device_group = device_group;
  executor->host_allocator = host_allocator;
  executor->options = options;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_scan_object_capacity(
      file_contents, &executor->object_capacity));
  if (executor->object_capacity == 0) return iree_ok_status();
  iree_host_size_t object_table_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(executor->object_capacity,
                                                sizeof(*executor->objects),
                                                &object_table_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay object table size overflow");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, object_table_size,
                                             (void**)&executor->objects));
  memset(executor->objects, 0, object_table_size);
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_require_payload(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type,
    iree_host_size_t minimum_length) {
  if (IREE_UNLIKELY(record->header.payload_type != payload_type)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED, "replay operation %s requires payload %s",
        iree_hal_replay_operation_code_string(record->header.operation_code),
        iree_hal_replay_payload_type_string(payload_type));
  }
  if (IREE_UNLIKELY(record->payload.data_length < minimum_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay payload %s is too short",
                            iree_hal_replay_payload_type_string(payload_type));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_make_buffer_params(
    const iree_hal_replay_allocator_allocate_buffer_payload_t* payload,
    iree_hal_buffer_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));
  out_params->usage = payload->usage;
  out_params->type = payload->type;
  out_params->access = payload->access;
  out_params->queue_affinity = payload->queue_affinity;
  out_params->min_alignment = payload->min_alignment;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_make_buffer_ref(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_buffer_ref_payload_t* payload,
    iree_hal_buffer_ref_t* out_ref) {
  if (payload->buffer_id == IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    *out_ref = iree_hal_make_indirect_buffer_ref(
        payload->buffer_slot, payload->offset, payload->length);
    return iree_ok_status();
  }
  iree_hal_replay_object_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, payload->buffer_id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      &entry));
  *out_ref = iree_hal_make_buffer_ref(entry->value.buffer, payload->offset,
                                      payload->length);
  return iree_ok_status();
}

typedef struct iree_hal_replay_semaphore_list_storage_t {
  // HAL semaphore list referencing arrays below.
  iree_hal_semaphore_list_t list;
  // Semaphore pointer array owned by this storage.
  iree_hal_semaphore_t** semaphores;
  // Semaphore payload value array owned by this storage.
  uint64_t* payload_values;
} iree_hal_replay_semaphore_list_storage_t;

static void iree_hal_replay_semaphore_list_storage_deinitialize(
    iree_hal_replay_semaphore_list_storage_t* storage,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, storage->payload_values);
  iree_allocator_free(host_allocator, storage->semaphores);
  memset(storage, 0, sizeof(*storage));
}

static iree_status_t iree_hal_replay_executor_make_semaphore_list(
    iree_hal_replay_executor_t* executor, iree_const_byte_span_t payloads,
    iree_host_size_t count,
    iree_hal_replay_semaphore_list_storage_t* out_storage) {
  memset(out_storage, 0, sizeof(*out_storage));
  if (count == 0) return iree_ok_status();
  iree_host_size_t expected_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &expected_length) ||
                    payloads.data_length != expected_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay semaphore list payload length mismatch");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executor->host_allocator, count * sizeof(*out_storage->semaphores),
      (void**)&out_storage->semaphores));
  iree_status_t status = iree_allocator_malloc(
      executor->host_allocator, count * sizeof(*out_storage->payload_values),
      (void**)&out_storage->payload_values);
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_semaphore_list_storage_deinitialize(
        out_storage, executor->host_allocator);
    return status;
  }
  const iree_hal_replay_semaphore_timepoint_payload_t* timepoints =
      (const iree_hal_replay_semaphore_timepoint_payload_t*)payloads.data;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_hal_replay_object_entry_t* entry = NULL;
    status = iree_hal_replay_executor_lookup(
        executor, timepoints[i].semaphore_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE, &entry);
    if (!iree_status_is_ok(status)) break;
    out_storage->semaphores[i] = entry->value.semaphore;
    out_storage->payload_values[i] = timepoints[i].value;
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_semaphore_list_storage_deinitialize(
        out_storage, executor->host_allocator);
    return status;
  }
  out_storage->list.count = count;
  out_storage->list.semaphores = out_storage->semaphores;
  out_storage->list.payload_values = out_storage->payload_values;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_make_queue_semaphore_lists(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record, iree_host_size_t header_length,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count,
    uint64_t trailing_payload_length,
    iree_hal_replay_semaphore_list_storage_t* out_wait_storage,
    iree_hal_replay_semaphore_list_storage_t* out_signal_storage,
    iree_const_byte_span_t* out_trailing_payload) {
  memset(out_wait_storage, 0, sizeof(*out_wait_storage));
  memset(out_signal_storage, 0, sizeof(*out_signal_storage));
  *out_trailing_payload = iree_make_const_byte_span(NULL, 0);
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(
          wait_semaphore_count > IREE_HOST_SIZE_MAX ||
          signal_semaphore_count > IREE_HOST_SIZE_MAX ||
          trailing_payload_length > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)wait_semaphore_count,
              sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
              &wait_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)signal_semaphore_count,
              sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
              &signal_size) ||
          !iree_host_size_checked_add(header_length, wait_size, &total_size) ||
          !iree_host_size_checked_add(total_size, signal_size, &total_size) ||
          !iree_host_size_checked_add(total_size,
                                      (iree_host_size_t)trailing_payload_length,
                                      &total_size) ||
          total_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue payload length mismatch");
  }

  const uint8_t* cursor = record->payload.data + header_length;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, wait_size),
      (iree_host_size_t)wait_semaphore_count, out_wait_storage));
  cursor += wait_size;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, signal_size),
      (iree_host_size_t)signal_semaphore_count, out_signal_storage);
  cursor += signal_size;
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_semaphore_list_storage_deinitialize(
        out_wait_storage, executor->host_allocator);
    return status;
  }
  *out_trailing_payload = iree_make_const_byte_span(
      cursor, (iree_host_size_t)trailing_payload_length);
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_flush_and_wait(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_list) {
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_flush(device, queue_affinity));
  if (signal_list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(signal_list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t iree_hal_replay_executor_store_device(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  if (executor->next_device_index >=
      iree_hal_device_group_device_count(executor->device_group)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay references more devices than provided");
  }
  iree_hal_device_t* device = iree_hal_device_group_device_at(
      executor->device_group, executor->next_device_index++);
  iree_hal_device_retain(device);
  iree_hal_replay_object_entry_t entry = {.value.device = device};
  return iree_hal_replay_executor_store(executor, record->header.object_id,
                                        IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                        entry);
}

static iree_status_t iree_hal_replay_executor_store_allocator(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.device_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(device_entry->value.device);
  iree_hal_allocator_retain(allocator);
  iree_hal_replay_object_entry_t entry = {.value.allocator = allocator};
  return iree_hal_replay_executor_store(executor, record->header.object_id,
                                        IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
                                        entry);
}

static iree_status_t iree_hal_replay_executor_replay_object(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  switch (record->header.object_type) {
    case IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE:
      return iree_hal_replay_executor_store_device(executor, record);
    case IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR:
      return iree_hal_replay_executor_store_allocator(executor, record);
    default:
      return iree_ok_status();
  }
}

static iree_status_t
iree_hal_replay_executor_compare_executable_export_parameters(
    iree_hal_replay_executor_t* executor,
    iree_hal_replay_object_id_t executable_id,
    iree_hal_executable_t* captured_executable,
    iree_hal_executable_t* replacement_executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    iree_host_size_t parameter_count) {
  if (parameter_count == 0) return iree_ok_status();
  iree_host_size_t allocation_count = 0;
  iree_host_size_t allocation_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(parameter_count, 2, &allocation_count) ||
          !iree_host_size_checked_mul(
              allocation_count, sizeof(iree_hal_executable_export_parameter_t),
              &allocation_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable parameter metadata is too large");
  }

  iree_hal_executable_export_parameter_t* captured_parameters = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executor->host_allocator, allocation_size, (void**)&captured_parameters));
  iree_hal_executable_export_parameter_t* replacement_parameters =
      captured_parameters + parameter_count;

  iree_status_t status = iree_hal_executable_export_parameters(
      captured_executable, export_ordinal, parameter_count,
      captured_parameters);
  if (iree_status_is_ok(status)) {
    status = iree_hal_executable_export_parameters(
        replacement_executable, export_ordinal, parameter_count,
        replacement_parameters);
  }
  for (iree_host_size_t i = 0; i < parameter_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_executable_export_parameter_t* captured =
        &captured_parameters[i];
    const iree_hal_executable_export_parameter_t* replacement =
        &replacement_parameters[i];
    if (captured->type != replacement->type ||
        captured->size != replacement->size ||
        captured->flags != replacement->flags ||
        captured->offset != replacement->offset) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "substituted executable %" PRIu64 " export %" PRIu32
          " parameter %" PRIhsz
          " ABI mismatch: captured=(type=%u size=%u flags=0x%04x "
          "offset=%u) replacement=(type=%u size=%u flags=0x%04x offset=%u)",
          executable_id, export_ordinal, i, (uint32_t)captured->type,
          (uint32_t)captured->size, (uint32_t)captured->flags,
          (uint32_t)captured->offset, (uint32_t)replacement->type,
          (uint32_t)replacement->size, (uint32_t)replacement->flags,
          (uint32_t)replacement->offset);
    }
  }
  iree_allocator_free(executor->host_allocator, captured_parameters);
  return status;
}

static iree_status_t iree_hal_replay_executor_validate_executable_substitution(
    iree_hal_replay_executor_t* executor,
    iree_hal_replay_object_id_t executable_id,
    iree_hal_executable_t* captured_executable,
    iree_hal_executable_t* replacement_executable) {
  iree_host_size_t captured_export_count =
      iree_hal_executable_export_count(captured_executable);
  iree_host_size_t replacement_export_count =
      iree_hal_executable_export_count(replacement_executable);
  if (IREE_UNLIKELY(captured_export_count != replacement_export_count)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "substituted executable %" PRIu64
        " export count mismatch: captured=%" PRIhsz " replacement=%" PRIhsz,
        executable_id, captured_export_count, replacement_export_count);
  }
  if (IREE_UNLIKELY(captured_export_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable export count exceeds HAL ordinal range");
  }

  for (iree_host_size_t i = 0; i < captured_export_count; ++i) {
    const iree_hal_executable_export_ordinal_t export_ordinal =
        (iree_hal_executable_export_ordinal_t)i;
    iree_hal_executable_export_info_t captured_info;
    IREE_RETURN_IF_ERROR(iree_hal_executable_export_info(
        captured_executable, export_ordinal, &captured_info));
    iree_hal_executable_export_info_t replacement_info;
    IREE_RETURN_IF_ERROR(iree_hal_executable_export_info(
        replacement_executable, export_ordinal, &replacement_info));

    if (captured_info.flags != replacement_info.flags ||
        captured_info.constant_count != replacement_info.constant_count ||
        captured_info.binding_count != replacement_info.binding_count ||
        captured_info.parameter_count != replacement_info.parameter_count ||
        captured_info.workgroup_size[0] != replacement_info.workgroup_size[0] ||
        captured_info.workgroup_size[1] != replacement_info.workgroup_size[1] ||
        captured_info.workgroup_size[2] != replacement_info.workgroup_size[2]) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "substituted executable %" PRIu64 " export %" PRIu32
          " ABI mismatch: captured=(flags=0x%016" PRIx64
          " constants=%u bindings=%u parameters=%u workgroup_size=[%u,%u,%u]) "
          "replacement=(flags=0x%016" PRIx64
          " constants=%u bindings=%u parameters=%u workgroup_size=[%u,%u,%u])",
          executable_id, export_ordinal, captured_info.flags,
          (uint32_t)captured_info.constant_count,
          (uint32_t)captured_info.binding_count,
          (uint32_t)captured_info.parameter_count,
          captured_info.workgroup_size[0], captured_info.workgroup_size[1],
          captured_info.workgroup_size[2], replacement_info.flags,
          (uint32_t)replacement_info.constant_count,
          (uint32_t)replacement_info.binding_count,
          (uint32_t)replacement_info.parameter_count,
          replacement_info.workgroup_size[0],
          replacement_info.workgroup_size[1],
          replacement_info.workgroup_size[2]);
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_replay_executor_compare_executable_export_parameters(
            executor, executable_id, captured_executable,
            replacement_executable, export_ordinal,
            captured_info.parameter_count));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_prepare_captured_executable(
    iree_hal_executable_cache_t* executable_cache,
    const iree_hal_executable_params_t* captured_params,
    iree_hal_executable_t** out_executable) {
  return iree_hal_executable_cache_prepare_executable(
      executable_cache, captured_params, out_executable);
}

static iree_status_t iree_hal_replay_executor_prepare_substitute_executable(
    iree_hal_executable_cache_t* executable_cache,
    const iree_hal_executable_params_t* captured_params,
    const iree_hal_replay_executable_substitution_t* substitution,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  if (IREE_UNLIKELY(substitution->executable_data.data_length != 0 &&
                    !substitution->executable_data.data)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "substituted executable data has non-zero length but no data pointer");
  }
  if (IREE_UNLIKELY(substitution->executable_data.data_length == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "substituted executable data is empty");
  }

  iree_hal_executable_params_t replacement_params = *captured_params;
  replacement_params.caching_mode &=
      ~IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
  replacement_params.executable_data = substitution->executable_data;

  char inferred_format[IREE_HAL_REPLAY_EXECUTABLE_FORMAT_CAPACITY] = {0};
  if (iree_string_view_is_empty(substitution->executable_format)) {
    iree_host_size_t inferred_size = 0;
    IREE_RETURN_IF_ERROR(iree_status_annotate(
        iree_hal_executable_cache_infer_format(
            executable_cache, replacement_params.caching_mode,
            replacement_params.executable_data, sizeof(inferred_format),
            inferred_format, &inferred_size),
        iree_make_cstring_view(
            "inferring substituted executable format; provide an explicit "
            "format if the target cache cannot infer one")));
    replacement_params.executable_format =
        iree_make_cstring_view(inferred_format);
  } else {
    replacement_params.executable_format = substitution->executable_format;
  }
  return iree_hal_executable_cache_prepare_executable(
      executable_cache, &replacement_params, out_executable);
}

static iree_status_t iree_hal_replay_executor_prepare_executable(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_PREPARE,
      sizeof(iree_hal_replay_executable_prepare_payload_t)));
  iree_hal_replay_executable_prepare_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_host_size_t constant_bytes = 0;
  iree_host_size_t data_offset = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t expected_length = 0;
  if (IREE_UNLIKELY(
          payload.executable_data_length > IREE_HOST_SIZE_MAX ||
          payload.constant_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.constant_count,
                                      sizeof(uint32_t), &constant_bytes) ||
          !iree_host_size_checked_add(sizeof(payload),
                                      payload.executable_format_length,
                                      &data_offset) ||
          !iree_host_size_checked_add(
              data_offset, (iree_host_size_t)payload.executable_data_length,
              &constants_offset) ||
          !iree_host_size_checked_add(constants_offset, constant_bytes,
                                      &expected_length) ||
          expected_length != record->payload.data_length)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay executable prepare payload length mismatch");
  }

  iree_hal_replay_object_entry_t* cache_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE_CACHE, &cache_entry));
  iree_hal_executable_params_t params;
  memset(&params, 0, sizeof(params));
  params.queue_affinity = payload.queue_affinity;
  params.executable_format =
      iree_make_string_view((const char*)record->payload.data + sizeof(payload),
                            payload.executable_format_length);
  params.executable_data = iree_make_const_byte_span(
      record->payload.data + data_offset,
      (iree_host_size_t)payload.executable_data_length);
  params.constant_count = (iree_host_size_t)payload.constant_count;
  params.constants = (const uint32_t*)(record->payload.data + constants_offset);
  params.caching_mode = payload.caching_mode;

  iree_hal_executable_t* executable = NULL;
  iree_status_t status = iree_ok_status();
  if (executor->options->executable_substitution_callback.fn) {
    iree_hal_replay_executable_substitution_request_t request = {
        .sequence_ordinal = record->header.sequence_ordinal,
        .device_id = record->header.device_id,
        .executable_cache_id = record->header.object_id,
        .executable_id = record->header.related_object_id,
        .captured_params = &params,
    };
    iree_hal_replay_executable_substitution_t substitution;
    memset(&substitution, 0, sizeof(substitution));
    status = executor->options->executable_substitution_callback.fn(
        executor->options->executable_substitution_callback.user_data, &request,
        &substitution);
    if (iree_status_is_ok(status) && substitution.substitute) {
      iree_hal_executable_t* captured_executable = NULL;
      status = iree_status_annotate_f(
          iree_hal_replay_executor_prepare_captured_executable(
              cache_entry->value.executable_cache, &params,
              &captured_executable),
          "preparing captured executable %" PRIu64
          " for substitution validation",
          record->header.related_object_id);
      if (iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            iree_hal_replay_executor_prepare_substitute_executable(
                cache_entry->value.executable_cache, &params, &substitution,
                &executable),
            "preparing substitute for captured executable %" PRIu64
            " from '%.*s'",
            record->header.related_object_id, (int)substitution.source.size,
            substitution.source.data);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_replay_executor_validate_executable_substitution(
            executor, record->header.related_object_id, captured_executable,
            executable);
      }
      iree_hal_executable_release(captured_executable);
      if (!iree_status_is_ok(status)) {
        iree_hal_executable_release(executable);
        executable = NULL;
      }
    }
  }
  if (iree_status_is_ok(status) && !executable) {
    status = iree_hal_replay_executor_prepare_captured_executable(
        cache_entry->value.executable_cache, &params, &executable);
  }
  IREE_RETURN_IF_ERROR(status);
  iree_hal_replay_object_entry_t entry = {.value.executable = executable};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE, entry);
}

static iree_status_t iree_hal_replay_executor_create_executable_cache(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_CACHE_OBJECT,
      sizeof(iree_hal_replay_executable_cache_object_payload_t)));
  iree_hal_replay_executable_cache_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.identifier_length >
                    record->payload.data_length - sizeof(payload))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable cache identifier overflow");
  }
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_string_view_t identifier =
      iree_make_string_view((const char*)record->payload.data + sizeof(payload),
                            (iree_host_size_t)payload.identifier_length);
  iree_hal_executable_cache_t* executable_cache = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_executable_cache_create(
      device_entry->value.device, identifier, &executable_cache));
  iree_hal_replay_object_entry_t entry = {.value.executable_cache =
                                              executable_cache};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE_CACHE, entry);
}

static iree_status_t iree_hal_replay_executor_create_command_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT,
      sizeof(iree_hal_replay_command_buffer_object_payload_t)));
  iree_hal_replay_command_buffer_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device_entry->value.device, payload.mode, payload.command_categories,
      payload.queue_affinity, (iree_host_size_t)payload.binding_capacity,
      &command_buffer));
  iree_hal_replay_object_entry_t entry = {.value.command_buffer =
                                              command_buffer};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, entry);
}

static iree_status_t iree_hal_replay_executor_create_semaphore(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT,
      sizeof(iree_hal_replay_semaphore_object_payload_t)));
  iree_hal_replay_semaphore_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device_entry->value.device, payload.queue_affinity, payload.initial_value,
      payload.flags, &semaphore));
  iree_hal_replay_object_entry_t entry = {.value.semaphore = semaphore};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE, entry);
}

static iree_status_t iree_hal_replay_executor_create_event(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_EVENT_OBJECT,
      sizeof(iree_hal_replay_event_object_payload_t)));
  iree_hal_replay_event_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_event_t* event = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_event_create(device_entry->value.device,
                                             payload.queue_affinity,
                                             payload.flags, &event));
  iree_hal_replay_object_entry_t entry = {.value.event = event};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_EVENT, entry);
}

static iree_status_t iree_hal_replay_executor_validate_file_reference(
    const iree_hal_replay_file_object_payload_t* payload,
    iree_string_view_t captured_path, iree_string_view_t resolved_path,
    iree_io_file_handle_t* handle, iree_hal_file_t* file) {
  const uint64_t file_length = iree_hal_file_length(file);
  if (IREE_UNLIKELY(payload->file_length != file_length)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external replay file length mismatch for '%.*s' captured as '%.*s': "
        "captured=%" PRIu64 " current=%" PRIu64
        "; restore the matching file or fix --replay_file_remap",
        (int)resolved_path.size, resolved_path.data, (int)captured_path.size,
        captured_path.data, payload->file_length, file_length);
  }

  switch (payload->validation_type) {
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_NONE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_IDENTITY:
      break;
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST:
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external replay file validation type %" PRIu32
                              " is not executable",
                              payload->validation_type);
  }

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  if (IREE_UNLIKELY(primitive.type != IREE_IO_FILE_HANDLE_TYPE_FD)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external replay file validation requires an fd-backed file");
  }
  if (payload->validation_type ==
      IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST) {
    if (IREE_UNLIKELY(payload->digest_type !=
                      IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external replay file digest type %" PRIu32
                              " is not executable",
                              (uint32_t)payload->digest_type);
    }
    uint64_t state = iree_hal_replay_digest_fnv1a64_initialize();
    uint64_t offset = 0;
    uint8_t buffer[64 * 1024];
    while (offset < payload->file_length) {
      uint64_t chunk_length = payload->file_length - offset;
      if (chunk_length > sizeof(buffer)) chunk_length = sizeof(buffer);
      ssize_t read_length = pread(primitive.value.fd, buffer,
                                  (size_t)chunk_length, (off_t)offset);
      if (read_length < 0 && errno == EINTR) continue;
      if (read_length <= 0) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "unable to read external replay file '%.*s' for digest validation",
            (int)resolved_path.size, resolved_path.data);
      }
      state = iree_hal_replay_digest_fnv1a64_update(
          state,
          iree_make_const_byte_span(buffer, (iree_host_size_t)read_length));
      offset += (uint64_t)read_length;
    }
    const uint64_t expected_digest =
        iree_hal_replay_digest_load_fnv1a64(payload->digest);
    if (IREE_UNLIKELY(state != expected_digest)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "external replay file digest mismatch for '%.*s' captured as "
          "'%.*s': expected=0x%016" PRIx64 " actual=0x%016" PRIx64
          "; restore the matching file or fix --replay_file_remap",
          (int)resolved_path.size, resolved_path.data, (int)captured_path.size,
          captured_path.data, expected_digest, state);
    }
    return iree_ok_status();
  }

  if (payload->file_device == 0 && payload->file_inode == 0 &&
      payload->file_mtime_ns == 0) {
    return iree_ok_status();
  }
  struct stat file_stat;
  if (IREE_UNLIKELY(fstat(primitive.value.fd, &file_stat) != 0)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "unable to stat external replay file '%.*s'",
                            (int)resolved_path.size, resolved_path.data);
  }
  const uint64_t file_device = (uint64_t)file_stat.st_dev;
  const uint64_t file_inode = (uint64_t)file_stat.st_ino;
  const uint64_t file_mtime_ns =
      ((uint64_t)file_stat.st_mtim.tv_sec * 1000000000ull) +
      (uint64_t)file_stat.st_mtim.tv_nsec;
  if (IREE_UNLIKELY(payload->file_device != file_device ||
                    payload->file_inode != file_inode ||
                    payload->file_mtime_ns != file_mtime_ns)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external replay file identity mismatch for '%.*s' captured as '%.*s': "
        "captured=(dev=%" PRIu64 ", inode=%" PRIu64 ", mtime_ns=%" PRIu64
        ") current=(dev=%" PRIu64 ", inode=%" PRIu64 ", mtime_ns=%" PRIu64
        "); identity validation is the default and does not read file "
        "contents; restore the original file identity, fix "
        "--replay_file_remap, or recapture with "
        "--device_replay_file_validation=digest when copied/staged files are "
        "intentional",
        (int)resolved_path.size, resolved_path.data, (int)captured_path.size,
        captured_path.data, payload->file_device, payload->file_inode,
        payload->file_mtime_ns, file_device, file_inode, file_mtime_ns);
  }
#else
  (void)handle;
  (void)captured_path;
  (void)resolved_path;
  if (payload->validation_type ==
      IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "external replay file content-digest validation requires POSIX file "
        "IO");
  }
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_resolve_file_path(
    iree_hal_replay_executor_t* executor, iree_string_view_t captured_path,
    iree_string_view_t* out_resolved_path, char** out_resolved_path_storage) {
  IREE_ASSERT_ARGUMENT(out_resolved_path);
  IREE_ASSERT_ARGUMENT(out_resolved_path_storage);
  *out_resolved_path = captured_path;
  *out_resolved_path_storage = NULL;

  const iree_hal_replay_file_path_remap_t* selected_remap = NULL;
  for (iree_host_size_t i = 0; i < executor->options->file_path_remap_count;
       ++i) {
    const iree_hal_replay_file_path_remap_t* remap =
        &executor->options->file_path_remaps[i];
    if (iree_string_view_is_empty(remap->captured_prefix)) continue;
    if (!iree_string_view_starts_with(captured_path, remap->captured_prefix)) {
      continue;
    }
    if (!selected_remap ||
        remap->captured_prefix.size > selected_remap->captured_prefix.size) {
      selected_remap = remap;
    }
  }
  if (!selected_remap) return iree_ok_status();

  iree_string_view_t captured_suffix = iree_string_view_remove_prefix(
      captured_path, selected_remap->captured_prefix.size);
  iree_host_size_t resolved_length = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(selected_remap->replay_prefix.size,
                                      captured_suffix.size, &resolved_length) ||
          resolved_length == IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remapped replay file path is too long");
  }
  char* resolved_path_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executor->host_allocator,
                                             resolved_length + 1,
                                             (void**)&resolved_path_storage));
  memcpy(resolved_path_storage, selected_remap->replay_prefix.data,
         selected_remap->replay_prefix.size);
  memcpy(resolved_path_storage + selected_remap->replay_prefix.size,
         captured_suffix.data, captured_suffix.size);
  resolved_path_storage[resolved_length] = 0;
  *out_resolved_path =
      iree_make_string_view(resolved_path_storage, resolved_length);
  *out_resolved_path_storage = resolved_path_storage;
  return iree_ok_status();
}

typedef struct iree_hal_replay_executor_inline_file_release_t {
  // Host allocator used for the inline file copy and this release record.
  iree_allocator_t host_allocator;
} iree_hal_replay_executor_inline_file_release_t;

static void iree_hal_replay_executor_inline_file_release(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_hal_replay_executor_inline_file_release_t* release =
      (iree_hal_replay_executor_inline_file_release_t*)user_data;
  iree_allocator_t host_allocator = release->host_allocator;
  iree_allocator_free(host_allocator,
                      handle_primitive.value.host_allocation.data);
  iree_allocator_free(host_allocator, release);
}

static iree_io_file_access_t iree_hal_replay_executor_make_file_access(
    iree_hal_memory_access_t access) {
  iree_io_file_access_t file_access = 0;
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
    file_access |= IREE_IO_FILE_ACCESS_READ;
  }
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    file_access |= IREE_IO_FILE_ACCESS_WRITE;
  }
  return file_access ? file_access : IREE_IO_FILE_ACCESS_READ;
}

static iree_status_t iree_hal_replay_executor_wrap_inline_file(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_object_payload_t* payload,
    iree_const_byte_span_t reference, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;

  if (IREE_UNLIKELY(payload->file_length != reference.data_length)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "inline replay file length mismatch: file_length=%" PRIu64
        " reference_length=%" PRIhsz,
        payload->file_length, reference.data_length);
  }

  iree_hal_replay_executor_inline_file_release_t* release = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executor->host_allocator, sizeof(*release), (void**)&release));
  release->host_allocator = executor->host_allocator;

  uint8_t* file_bytes = NULL;
  iree_status_t status = iree_ok_status();
  if (reference.data_length != 0) {
    status = iree_allocator_malloc(executor->host_allocator,
                                   reference.data_length, (void**)&file_bytes);
    if (iree_status_is_ok(status)) {
      memcpy(file_bytes, reference.data, reference.data_length);
    }
  }
  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_hal_replay_executor_inline_file_release,
      .user_data = release,
  };
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_wrap_host_allocation(
        iree_hal_replay_executor_make_file_access(payload->access),
        iree_make_byte_span(file_bytes, reference.data_length),
        release_callback, executor->host_allocator, out_handle);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executor->host_allocator, file_bytes);
    iree_allocator_free(executor->host_allocator, release);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_import_file(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT,
      sizeof(iree_hal_replay_file_object_payload_t)));
  iree_hal_replay_file_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.reference_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) +
                            (iree_host_size_t)payload.reference_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file object payload length mismatch");
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));

  iree_const_byte_span_t reference =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.reference_length);
  iree_io_file_handle_t* handle = NULL;
  char* resolved_path_storage = NULL;
  iree_string_view_t captured_external_path = iree_string_view_empty();
  iree_string_view_t resolved_external_path = iree_string_view_empty();
  iree_status_t status = iree_ok_status();
  switch (payload.reference_type) {
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH: {
      iree_string_view_t path = iree_make_string_view(
          (const char*)reference.data, reference.data_length);
      captured_external_path = path;
      status = iree_hal_replay_executor_resolve_file_path(
          executor, path, &path, &resolved_path_storage);
      resolved_external_path = path;
      iree_io_file_mode_t mode = 0;
      if (iree_any_bit_set(payload.access, IREE_HAL_MEMORY_ACCESS_READ)) {
        mode |= IREE_IO_FILE_MODE_READ;
      }
      if (iree_any_bit_set(payload.access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        mode |= IREE_IO_FILE_MODE_WRITE;
      }
      if (mode == 0) mode = IREE_IO_FILE_MODE_READ;
      if (iree_status_is_ok(status)) {
        status = iree_io_file_handle_open(mode, path, executor->host_allocator,
                                          &handle);
        if (!iree_status_is_ok(status)) {
          status = iree_status_annotate_f(
              status,
              "opening external replay file '%.*s' captured as '%.*s'; use "
              "--replay_file_remap=CAPTURED_PREFIX=REPLAY_PREFIX if the "
              "parameter root moved",
              (int)path.size, path.data, (int)captured_external_path.size,
              captured_external_path.data);
        }
      }
      break;
    }
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES:
      status = iree_hal_replay_executor_wrap_inline_file(executor, &payload,
                                                         reference, &handle);
      break;
    default:
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "replay file reference type %" PRIu32
                                " is not executable",
                                payload.reference_type);
      break;
  }

  iree_hal_file_t* file = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_file_import(device_entry->value.device, payload.queue_affinity,
                             payload.access, handle, payload.flags, &file);
  }
  if (iree_status_is_ok(status) &&
      payload.reference_type ==
          IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH) {
    status = iree_hal_replay_executor_validate_file_reference(
        &payload, captured_external_path, resolved_external_path, handle, file);
  }
  iree_io_file_handle_release(handle);
  iree_allocator_free(executor->host_allocator, resolved_path_storage);

  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.file = file};
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_FILE, entry);
  } else {
    iree_hal_file_release(file);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_allocate_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER,
      sizeof(iree_hal_replay_allocator_allocate_buffer_payload_t)));
  iree_hal_replay_allocator_allocate_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  iree_hal_buffer_params_t params;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_executor_make_buffer_params(&payload, &params));
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator_entry->value.allocator, params, payload.allocation_size,
      &buffer));
  iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
}

static iree_status_t iree_hal_replay_executor_replay_buffer_range_data(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA,
      sizeof(iree_hal_replay_buffer_range_data_payload_t)));
  iree_hal_replay_buffer_range_data_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.data_length >
                    record->payload.data_length - sizeof(payload))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay buffer range data overflows payload");
  }
  iree_hal_replay_object_entry_t* buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      &buffer_entry));
  iree_hal_buffer_mapping_t mapping;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      buffer_entry->value.buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      payload.memory_access, payload.byte_offset, payload.byte_length,
      &mapping));
  iree_status_t status = iree_ok_status();
  iree_byte_span_t target_span;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_buffer_mapping_subspan(&mapping, IREE_HAL_MEMORY_ACCESS_WRITE,
                                        0, payload.data_length, &target_span);
  }
  if (iree_status_is_ok(status)) {
    memcpy(target_span.data, record->payload.data + sizeof(payload),
           (iree_host_size_t)payload.data_length);
    status =
        iree_hal_buffer_mapping_flush_range(&mapping, 0, payload.data_length);
  }
  iree_status_t unmap_status = iree_hal_buffer_unmap_range(&mapping);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(unmap_status);
    return status;
  }
  return unmap_status;
}

static iree_status_t iree_hal_replay_executor_queue_alloca(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA,
      sizeof(iree_hal_replay_device_queue_alloca_payload_t)));
  iree_hal_replay_device_queue_alloca_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t params;
    status = iree_hal_replay_executor_make_buffer_params(&payload.allocation,
                                                         &params);
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_alloca(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, /*pool=*/NULL, params,
          payload.allocation.allocation_size, payload.flags, &buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
    buffer = NULL;
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
  }
  iree_hal_buffer_release(buffer);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_dealloca(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA,
      sizeof(iree_hal_replay_device_queue_dealloca_payload_t)));
  iree_hal_replay_device_queue_dealloca_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t buffer_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.buffer_ref, &buffer_ref);
  }
  if (iree_status_is_ok(status) && !buffer_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue dealloca requires a direct buffer reference");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, buffer_ref.buffer, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_fill(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL,
      sizeof(iree_hal_replay_device_queue_fill_payload_t)));
  iree_hal_replay_device_queue_fill_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t pattern;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, payload.pattern_length, &wait_storage,
      &signal_storage, &pattern);
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && !target_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue fill requires a direct target buffer reference");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_fill(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, target_ref.buffer, target_ref.offset,
        target_ref.length, pattern.data, pattern.data_length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_update(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE,
      sizeof(iree_hal_replay_device_queue_update_payload_t)));
  iree_hal_replay_device_queue_update_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t data;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, payload.data_length, &wait_storage,
      &signal_storage, &data);
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && !target_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue update requires a direct target buffer reference");
  }
  if (iree_status_is_ok(status) && data.data_length != target_ref.length) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue update data length does not match target length");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, data.data, /*source_offset=*/0, target_ref.buffer,
        target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_copy(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY,
      sizeof(iree_hal_replay_device_queue_copy_payload_t)));
  iree_hal_replay_device_queue_copy_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t source_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.source_ref, &source_ref);
  }
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && (!source_ref.buffer || !target_ref.buffer)) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "replay queue copy requires direct source and "
                              "target buffer references");
  }
  if (iree_status_is_ok(status) && source_ref.length != target_ref.length) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue copy source and target lengths do not match");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_copy(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, source_ref.buffer, source_ref.offset,
        target_ref.buffer, target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_read(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ,
      sizeof(iree_hal_replay_device_queue_read_payload_t)));
  iree_hal_replay_device_queue_read_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_replay_object_entry_t* file_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, payload.source_file_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_FILE,
                                             &file_entry);
  }
  iree_hal_buffer_ref_t target_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.target_ref, &target_ref);
  }
  if (iree_status_is_ok(status) && !target_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue read requires a direct target buffer reference");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_read(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, file_entry->value.file, payload.source_offset,
        target_ref.buffer, target_ref.offset, target_ref.length, payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_write(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE,
      sizeof(iree_hal_replay_device_queue_write_payload_t)));
  iree_hal_replay_device_queue_write_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_const_byte_span_t trailing_payload;
  iree_status_t status = iree_hal_replay_executor_make_queue_semaphore_lists(
      executor, record, sizeof(payload), payload.wait_semaphore_count,
      payload.signal_semaphore_count, /*trailing_payload_length=*/0,
      &wait_storage, &signal_storage, &trailing_payload);
  iree_hal_buffer_ref_t source_ref;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.source_ref, &source_ref);
  }
  iree_hal_replay_object_entry_t* file_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, payload.target_file_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_FILE,
                                             &file_entry);
  }
  if (iree_status_is_ok(status) && !source_ref.buffer) {
    status = iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay queue write requires a direct source buffer reference");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_write(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, source_ref.buffer, source_ref.offset,
        file_entry->value.file, payload.target_offset, source_ref.length,
        payload.flags);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_execution_barrier(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER,
      sizeof(iree_hal_replay_command_buffer_execution_barrier_payload_t)));
  iree_hal_replay_command_buffer_execution_barrier_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t memory_payloads_size = 0;
  iree_host_size_t buffer_payloads_size = 0;
  iree_host_size_t total_payload_size = 0;
  if (IREE_UNLIKELY(
          payload.memory_barrier_count > IREE_HOST_SIZE_MAX ||
          payload.buffer_barrier_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.memory_barrier_count,
              sizeof(iree_hal_replay_memory_barrier_payload_t),
              &memory_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.buffer_barrier_count,
              sizeof(iree_hal_replay_buffer_barrier_payload_t),
              &buffer_payloads_size) ||
          !iree_host_size_checked_add(sizeof(payload), memory_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, buffer_payloads_size,
                                      &total_payload_size) ||
          total_payload_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay execution barrier payload length mismatch");
  }

  iree_hal_memory_barrier_t* memory_barriers = NULL;
  iree_host_size_t memory_barriers_size = 0;
  if (payload.memory_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.memory_barrier_count,
            sizeof(*memory_barriers), &memory_barriers_size))) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay execution barrier memory barrier count overflow");
    }
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(executor->host_allocator,
                                               memory_barriers_size,
                                               (void**)&memory_barriers));
  }
  const iree_hal_replay_memory_barrier_payload_t* memory_payloads =
      (const iree_hal_replay_memory_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload));
  for (iree_host_size_t i = 0; i < payload.memory_barrier_count; ++i) {
    memory_barriers[i].source_scope = memory_payloads[i].source_scope;
    memory_barriers[i].target_scope = memory_payloads[i].target_scope;
  }

  iree_hal_buffer_barrier_t* buffer_barriers = NULL;
  iree_status_t status = iree_ok_status();
  iree_host_size_t buffer_barriers_size = 0;
  if (payload.buffer_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.buffer_barrier_count,
            sizeof(*buffer_barriers), &buffer_barriers_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay execution barrier buffer barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, buffer_barriers_size,
                                (void**)&buffer_barriers);
    }
  }
  const iree_hal_replay_buffer_barrier_payload_t* buffer_payloads =
      (const iree_hal_replay_buffer_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        memory_payloads_size);
  for (iree_host_size_t i = 0;
       i < payload.buffer_barrier_count && iree_status_is_ok(status); ++i) {
    buffer_barriers[i].source_scope = buffer_payloads[i].source_scope;
    buffer_barriers[i].target_scope = buffer_payloads[i].target_scope;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &buffer_payloads[i].buffer_ref,
        &buffer_barriers[i].buffer_ref);
  }

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, record->header.object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_execution_barrier(
        command_buffer_entry->value.command_buffer, payload.source_stage_mask,
        payload.target_stage_mask, payload.flags,
        (iree_host_size_t)payload.memory_barrier_count, memory_barriers,
        (iree_host_size_t)payload.buffer_barrier_count, buffer_barriers);
  }
  iree_allocator_free(executor->host_allocator, buffer_barriers);
  iree_allocator_free(executor->host_allocator, memory_barriers);
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_event(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EVENT,
      sizeof(iree_hal_replay_command_buffer_event_payload_t)));
  iree_hal_replay_command_buffer_event_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_replay_object_entry_t* event_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, payload.event_id, IREE_HAL_REPLAY_OBJECT_TYPE_EVENT,
      &event_entry));

  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_SIGNAL_EVENT:
      return iree_hal_command_buffer_signal_event(
          command_buffer_entry->value.command_buffer, event_entry->value.event,
          payload.source_stage_mask);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_RESET_EVENT:
      return iree_hal_command_buffer_reset_event(
          command_buffer_entry->value.command_buffer, event_entry->value.event,
          payload.source_stage_mask);
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED, "replay operation %s is not an event op",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}

static iree_status_t iree_hal_replay_executor_command_buffer_wait_events(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_WAIT_EVENTS,
      sizeof(iree_hal_replay_command_buffer_wait_events_payload_t)));
  iree_hal_replay_command_buffer_wait_events_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_host_size_t event_payloads_size = 0;
  iree_host_size_t memory_payloads_size = 0;
  iree_host_size_t buffer_payloads_size = 0;
  iree_host_size_t total_payload_size = 0;
  if (IREE_UNLIKELY(
          payload.event_count > IREE_HOST_SIZE_MAX ||
          payload.memory_barrier_count > IREE_HOST_SIZE_MAX ||
          payload.buffer_barrier_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.event_count,
                                      sizeof(iree_hal_replay_object_id_t),
                                      &event_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.memory_barrier_count,
              sizeof(iree_hal_replay_memory_barrier_payload_t),
              &memory_payloads_size) ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.buffer_barrier_count,
              sizeof(iree_hal_replay_buffer_barrier_payload_t),
              &buffer_payloads_size) ||
          !iree_host_size_checked_add(sizeof(payload), event_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, memory_payloads_size,
                                      &total_payload_size) ||
          !iree_host_size_checked_add(total_payload_size, buffer_payloads_size,
                                      &total_payload_size) ||
          total_payload_size != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay wait events payload length mismatch");
  }

  const iree_hal_replay_object_id_t* event_payloads =
      (const iree_hal_replay_object_id_t*)(record->payload.data +
                                           sizeof(payload));
  const iree_hal_replay_memory_barrier_payload_t* memory_payloads =
      (const iree_hal_replay_memory_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        event_payloads_size);
  const iree_hal_replay_buffer_barrier_payload_t* buffer_payloads =
      (const iree_hal_replay_buffer_barrier_payload_t*)(record->payload.data +
                                                        sizeof(payload) +
                                                        event_payloads_size +
                                                        memory_payloads_size);

  const iree_hal_event_t** events = NULL;
  iree_status_t status = iree_ok_status();
  iree_host_size_t events_size = 0;
  if (payload.event_count) {
    if (IREE_UNLIKELY(
            !iree_host_size_checked_mul((iree_host_size_t)payload.event_count,
                                        sizeof(*events), &events_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay wait event count overflow");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(executor->host_allocator, events_size,
                                     (void**)&events);
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.event_count && iree_status_is_ok(status); ++i) {
    iree_hal_replay_object_entry_t* event_entry = NULL;
    status = iree_hal_replay_executor_lookup(executor, event_payloads[i],
                                             IREE_HAL_REPLAY_OBJECT_TYPE_EVENT,
                                             &event_entry);
    if (iree_status_is_ok(status)) {
      events[i] = event_entry->value.event;
    }
  }

  iree_hal_memory_barrier_t* memory_barriers = NULL;
  iree_host_size_t memory_barriers_size = 0;
  if (iree_status_is_ok(status) && payload.memory_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.memory_barrier_count,
            sizeof(*memory_barriers), &memory_barriers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay wait memory barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, memory_barriers_size,
                                (void**)&memory_barriers);
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.memory_barrier_count && iree_status_is_ok(status); ++i) {
    memory_barriers[i].source_scope = memory_payloads[i].source_scope;
    memory_barriers[i].target_scope = memory_payloads[i].target_scope;
  }

  iree_hal_buffer_barrier_t* buffer_barriers = NULL;
  iree_host_size_t buffer_barriers_size = 0;
  if (iree_status_is_ok(status) && payload.buffer_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            (iree_host_size_t)payload.buffer_barrier_count,
            sizeof(*buffer_barriers), &buffer_barriers_size))) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "replay wait buffer barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(executor->host_allocator, buffer_barriers_size,
                                (void**)&buffer_barriers);
    }
  }
  for (iree_host_size_t i = 0;
       i < payload.buffer_barrier_count && iree_status_is_ok(status); ++i) {
    buffer_barriers[i].source_scope = buffer_payloads[i].source_scope;
    buffer_barriers[i].target_scope = buffer_payloads[i].target_scope;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &buffer_payloads[i].buffer_ref,
        &buffer_barriers[i].buffer_ref);
  }

  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, record->header.object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_wait_events(
        command_buffer_entry->value.command_buffer,
        (iree_host_size_t)payload.event_count, events,
        payload.source_stage_mask, payload.target_stage_mask,
        (iree_host_size_t)payload.memory_barrier_count, memory_barriers,
        (iree_host_size_t)payload.buffer_barrier_count, buffer_barriers);
  }

  iree_allocator_free(executor->host_allocator, buffer_barriers);
  iree_allocator_free(executor->host_allocator, memory_barriers);
  iree_allocator_free(executor->host_allocator, events);
  return status;
}

static iree_status_t iree_hal_replay_executor_dispatch_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dispatch_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size) {
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_size = 0;
  iree_host_size_t binding_size = 0;
  if (IREE_UNLIKELY(payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload->constants_length > IREE_HOST_SIZE_MAX ||
                    payload->binding_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->wait_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &wait_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->signal_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &signal_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->binding_count,
                        sizeof(iree_hal_replay_buffer_ref_payload_t),
                        &binding_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload count overflow");
  }
  constants_size = (iree_host_size_t)payload->constants_length;
  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_size;
  if (!iree_host_size_checked_add(offset, wait_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_size;
  if (!iree_host_size_checked_add(offset, signal_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_constants_offset = offset;
  if (!iree_host_size_checked_add(offset, constants_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_binding_payloads_offset = offset;
  *out_binding_payloads_size = binding_size;
  if (!iree_host_size_checked_add(offset, binding_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay dispatch payload length mismatch");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_command_buffer_dispatch(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      sizeof(iree_hal_replay_dispatch_payload_t)));
  iree_hal_replay_dispatch_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (payload.wait_semaphore_count != 0 ||
      payload.signal_semaphore_count != 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "command buffer dispatch semaphore lists are "
                            "reserved for immediate dispatch");
  }
  iree_host_size_t wait_offset = 0;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_offset = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t binding_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_dispatch_layout(
      record, &payload, &wait_offset, &wait_size, &signal_offset, &signal_size,
      &constants_offset, &binding_offset, &binding_size));
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(record->payload.data + constants_offset,
                                (iree_host_size_t)payload.constants_length);
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                    binding_offset);
  iree_hal_buffer_ref_t* bindings = NULL;
  if (payload.binding_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        executor->host_allocator, binding_size, (void**)&bindings));
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < payload.binding_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &binding_payloads[i], &bindings[i]);
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, record->header.object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  iree_hal_replay_object_entry_t* executable_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.executable_id, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE,
        &executable_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_dispatch_config_t config;
    memset(&config, 0, sizeof(config));
    memcpy(config.workgroup_size, payload.workgroup_size,
           sizeof(config.workgroup_size));
    memcpy(config.workgroup_count, payload.workgroup_count,
           sizeof(config.workgroup_count));
    config.dynamic_workgroup_local_memory =
        payload.dynamic_workgroup_local_memory;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.workgroup_count_ref, &config.workgroup_count_ref);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_ref_list_t binding_list = {
          (iree_host_size_t)payload.binding_count,
          bindings,
      };
      status = iree_hal_command_buffer_dispatch(
          command_buffer_entry->value.command_buffer,
          executable_entry->value.executable, payload.export_ordinal, config,
          constants, binding_list, payload.flags);
    }
  }
  iree_allocator_free(executor->host_allocator, bindings);
  return status;
}

static iree_status_t iree_hal_replay_executor_queue_dispatch(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH,
      sizeof(iree_hal_replay_dispatch_payload_t)));
  iree_hal_replay_dispatch_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t wait_offset = 0;
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_offset = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t binding_offset = 0;
  iree_host_size_t binding_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_dispatch_layout(
      record, &payload, &wait_offset, &wait_size, &signal_offset, &signal_size,
      &constants_offset, &binding_offset, &binding_size));
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(record->payload.data + constants_offset,
                                (iree_host_size_t)payload.constants_length);
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                    binding_offset);

  iree_hal_replay_semaphore_list_storage_t wait_storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(record->payload.data + wait_offset, wait_size),
      (iree_host_size_t)payload.wait_semaphore_count, &wait_storage));
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(record->payload.data + signal_offset,
                                signal_size),
      (iree_host_size_t)payload.signal_semaphore_count, &signal_storage);

  iree_hal_buffer_ref_t* bindings = NULL;
  if (iree_status_is_ok(status) && payload.binding_count) {
    status = iree_allocator_malloc(executor->host_allocator, binding_size,
                                   (void**)&bindings);
  }
  for (iree_host_size_t i = 0;
       i < payload.binding_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &binding_payloads[i], &bindings[i]);
  }
  iree_hal_replay_object_entry_t* device_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, record->header.object_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                             &device_entry);
  }
  iree_hal_replay_object_entry_t* executable_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.executable_id, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE,
        &executable_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_dispatch_config_t config;
    memset(&config, 0, sizeof(config));
    memcpy(config.workgroup_size, payload.workgroup_size,
           sizeof(config.workgroup_size));
    memcpy(config.workgroup_count, payload.workgroup_count,
           sizeof(config.workgroup_count));
    config.dynamic_workgroup_local_memory =
        payload.dynamic_workgroup_local_memory;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &payload.workgroup_count_ref, &config.workgroup_count_ref);
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_ref_list_t binding_list = {
          (iree_host_size_t)payload.binding_count,
          bindings,
      };
      status = iree_hal_device_queue_dispatch(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, executable_entry->value.executable,
          payload.export_ordinal, config, constants, binding_list,
          payload.flags);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_flush_and_wait(device_entry->value.device,
                                                     payload.queue_affinity,
                                                     signal_storage.list);
  }

  iree_allocator_free(executor->host_allocator, bindings);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_command_buffer_fill_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER,
      sizeof(iree_hal_replay_command_buffer_fill_buffer_payload_t)));
  iree_hal_replay_command_buffer_fill_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.pattern_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) +
                            (iree_host_size_t)payload.pattern_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay command buffer fill payload length "
                            "mismatch");
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.target_ref, &target_ref));
  iree_const_byte_span_t pattern =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.pattern_length);
  return iree_hal_command_buffer_fill_buffer(
      command_buffer_entry->value.command_buffer, target_ref, pattern.data,
      pattern.data_length, payload.flags);
}

static iree_status_t iree_hal_replay_executor_command_buffer_update_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER,
      sizeof(iree_hal_replay_command_buffer_update_buffer_payload_t)));
  iree_hal_replay_command_buffer_update_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.data_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) + (iree_host_size_t)payload.data_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay command buffer update payload length "
                            "mismatch");
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.target_ref, &target_ref));
  if (payload.data_length != target_ref.length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay command buffer update data length does not "
                            "match target length");
  }
  iree_const_byte_span_t data =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.data_length);
  return iree_hal_command_buffer_update_buffer(
      command_buffer_entry->value.command_buffer, data.data,
      /*source_offset=*/0, target_ref, payload.flags);
}

static iree_status_t iree_hal_replay_executor_command_buffer_copy_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER,
      sizeof(iree_hal_replay_command_buffer_copy_buffer_payload_t)));
  iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry));
  iree_hal_buffer_ref_t source_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.source_ref, &source_ref));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_ref(
      executor, &payload.target_ref, &target_ref));
  return iree_hal_command_buffer_copy_buffer(
      command_buffer_entry->value.command_buffer, source_ref, target_ref,
      payload.flags);
}

static iree_status_t iree_hal_replay_executor_queue_execute(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE,
      sizeof(iree_hal_replay_device_queue_execute_payload_t)));
  iree_hal_replay_device_queue_execute_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  iree_host_size_t binding_size = 0;
  if (IREE_UNLIKELY(payload.wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.binding_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.wait_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &wait_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.signal_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &signal_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.binding_count,
                        sizeof(iree_hal_replay_buffer_ref_payload_t),
                        &binding_size) ||
                    sizeof(payload) + wait_size + signal_size + binding_size !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue execute payload length mismatch");
  }
  const uint8_t* cursor = record->payload.data + sizeof(payload);
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, wait_size),
      (iree_host_size_t)payload.wait_semaphore_count, &wait_storage));
  cursor += wait_size;
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor, iree_make_const_byte_span(cursor, signal_size),
      (iree_host_size_t)payload.signal_semaphore_count, &signal_storage);
  cursor += signal_size;

  iree_hal_buffer_binding_t* bindings = NULL;
  if (iree_status_is_ok(status) && payload.binding_count) {
    status = iree_allocator_malloc(
        executor->host_allocator,
        (iree_host_size_t)payload.binding_count * sizeof(*bindings),
        (void**)&bindings);
  }
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)cursor;
  for (iree_host_size_t i = 0;
       i < payload.binding_count && iree_status_is_ok(status); ++i) {
    iree_hal_buffer_ref_t ref;
    status = iree_hal_replay_executor_make_buffer_ref(
        executor, &binding_payloads[i], &ref);
    if (iree_status_is_ok(status)) {
      bindings[i] = (iree_hal_buffer_binding_t){
          .buffer = ref.buffer,
          .offset = ref.offset,
          .length = ref.length,
      };
    }
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(executor, record->header.object_id,
                                             IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                             &device_entry);
  }
  iree_hal_replay_object_entry_t* command_buffer_entry = NULL;
  if (iree_status_is_ok(status) &&
      payload.command_buffer_id != IREE_HAL_REPLAY_OBJECT_ID_NONE) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t binding_table = {
        (iree_host_size_t)payload.binding_count,
        bindings,
    };
    if (command_buffer_entry) {
      status = iree_hal_device_queue_execute(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, command_buffer_entry->value.command_buffer,
          binding_table, payload.flags);
    } else if (payload.binding_count == 0) {
      status = iree_hal_device_queue_barrier(
          device_entry->value.device, payload.queue_affinity, wait_storage.list,
          signal_storage.list, payload.flags);
    } else {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "replay queue barrier payload unexpectedly has bindings");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_flush(device_entry->value.device,
                                         payload.queue_affinity);
  }
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_allocator_free(executor->host_allocator, bindings);
  iree_hal_replay_semaphore_list_storage_deinitialize(&signal_storage,
                                                      executor->host_allocator);
  iree_hal_replay_semaphore_list_storage_deinitialize(&wait_storage,
                                                      executor->host_allocator);
  return status;
}

static iree_status_t iree_hal_replay_executor_replay_operation(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  if (IREE_UNLIKELY(record->header.status_code != IREE_STATUS_OK)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "replaying captured failing HAL operations is not implemented");
  }
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_I64:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_CAPABILITIES:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_QUEUE_POOL_BACKEND:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_BEGIN:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_FLUSH:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_END:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_BEGIN:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_EXTERNAL_CAPTURE_END:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_TRIM:
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_QUERY_MEMORY_HEAPS:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN_DEBUG_GROUP:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END_DEBUG_GROUP:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ADVISE_BUFFER:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_COUNT:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_INFO:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_PARAMETERS:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_LOOKUP_EXPORT_BY_NAME:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EXECUTABLE_CACHE:
      return iree_hal_replay_executor_create_executable_cache(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_CACHE_PREPARE_EXECUTABLE:
      return iree_hal_replay_executor_prepare_executable(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER:
      return iree_hal_replay_executor_create_command_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EVENT:
      return iree_hal_replay_executor_create_event(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE:
      return iree_hal_replay_executor_import_file(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE:
      return iree_hal_replay_executor_create_semaphore(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER:
      return iree_hal_replay_executor_allocate_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA:
      return iree_hal_replay_executor_queue_alloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA:
      return iree_hal_replay_executor_queue_dealloca(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL:
      return iree_hal_replay_executor_queue_fill(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE:
      return iree_hal_replay_executor_queue_update(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY:
      return iree_hal_replay_executor_queue_copy(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ:
      return iree_hal_replay_executor_queue_read(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE:
      return iree_hal_replay_executor_queue_write(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE:
      if (record->header.payload_type ==
          IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA) {
        return iree_hal_replay_executor_replay_buffer_range_data(executor,
                                                                 record);
      }
      return iree_ok_status();
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN: {
      iree_hal_replay_object_entry_t* entry = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
          executor, record->header.object_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &entry));
      return iree_hal_command_buffer_begin(entry->value.command_buffer);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END: {
      iree_hal_replay_object_entry_t* entry = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
          executor, record->header.object_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &entry));
      return iree_hal_command_buffer_end(entry->value.command_buffer);
    }
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER:
      return iree_hal_replay_executor_command_buffer_execution_barrier(executor,
                                                                       record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_SIGNAL_EVENT:
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_RESET_EVENT:
      return iree_hal_replay_executor_command_buffer_event(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_WAIT_EVENTS:
      return iree_hal_replay_executor_command_buffer_wait_events(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH:
      return iree_hal_replay_executor_command_buffer_dispatch(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER:
      return iree_hal_replay_executor_command_buffer_fill_buffer(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER:
      return iree_hal_replay_executor_command_buffer_update_buffer(executor,
                                                                   record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COPY_BUFFER:
      return iree_hal_replay_executor_command_buffer_copy_buffer(executor,
                                                                 record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH:
      return iree_hal_replay_executor_queue_dispatch(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE:
      return iree_hal_replay_executor_queue_execute(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FLUSH: {
      iree_hal_replay_object_entry_t* entry = NULL;
      IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
          executor, record->header.object_id,
          IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE, &entry));
      return iree_hal_device_queue_flush(entry->value.device,
                                         IREE_HAL_QUEUE_AFFINITY_ANY);
    }
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED, "replay operation %s is not implemented",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}

IREE_API_EXPORT iree_status_t iree_hal_replay_execute_file(
    iree_const_byte_span_t file_contents, iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_ASSERT_ARGUMENT(device_group);
  iree_hal_replay_execute_options_t default_options =
      iree_hal_replay_execute_options_default();
  if (!options) options = &default_options;
  if (IREE_UNLIKELY(options->flags != IREE_HAL_REPLAY_EXECUTE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay execute flags are unsupported");
  }
  if (IREE_UNLIKELY(options->file_path_remap_count != 0 &&
                    !options->file_path_remaps)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replay execute file path remaps require a remap list");
  }

  iree_hal_replay_file_header_t file_header;
  iree_host_size_t offset = 0;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_file_parse_header(file_contents, &file_header, &offset));
  iree_const_byte_span_t valid_contents = file_contents;
  if (file_header.file_length != 0) {
    valid_contents.data_length = (iree_host_size_t)file_header.file_length;
  }

  iree_hal_replay_executor_t executor;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_initialize(
      &executor, valid_contents, device_group, options, host_allocator));

  iree_status_t status = iree_ok_status();
  uint64_t expected_sequence_ordinal = 0;
  while (iree_status_is_ok(status) && offset < valid_contents.data_length) {
    iree_hal_replay_file_record_t record;
    status = iree_hal_replay_file_parse_record(valid_contents, offset, &record,
                                               &offset);
    if (!iree_status_is_ok(status)) break;
    if (record.header.sequence_ordinal != expected_sequence_ordinal++) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay record sequence ordinal mismatch");
      break;
    }
    switch (record.header.record_type) {
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
        status = iree_hal_replay_executor_replay_object(&executor, &record);
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
        status = iree_hal_replay_executor_replay_operation(&executor, &record);
        break;
      case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "replay contains unsupported captured operation %s",
            iree_hal_replay_operation_code_string(
                record.header.operation_code));
        break;
      default:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "replay record type %s is not executable",
            iree_hal_replay_file_record_type_string(record.header.record_type));
        break;
    }
  }

  iree_hal_replay_executor_deinitialize(&executor);
  return status;
}
