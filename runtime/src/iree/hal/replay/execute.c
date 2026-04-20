// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute.h"

#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/format.h"

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
  } value;
} iree_hal_replay_object_entry_t;

typedef struct iree_hal_replay_executor_t {
  // Original replay file bytes.
  iree_const_byte_span_t file_contents;
  // Retained topology supplied by the caller.
  iree_hal_device_group_t* device_group;
  // Host allocator used for temporary replay state.
  iree_allocator_t host_allocator;
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
    iree_hal_device_group_t* device_group, iree_allocator_t host_allocator) {
  memset(executor, 0, sizeof(*executor));
  executor->file_contents = file_contents;
  executor->device_group = device_group;
  executor->host_allocator = host_allocator;
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
  IREE_RETURN_IF_ERROR(iree_hal_executable_cache_prepare_executable(
      cache_entry->value.executable_cache, &params, &executable));
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
  iree_host_size_t wait_size = 0;
  iree_host_size_t signal_size = 0;
  if (IREE_UNLIKELY(payload.wait_semaphore_count > IREE_HOST_SIZE_MAX ||
                    payload.signal_semaphore_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.wait_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &wait_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload.signal_semaphore_count,
                        sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
                        &signal_size) ||
                    sizeof(payload) + wait_size + signal_size !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue alloca payload length mismatch");
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_replay_semaphore_list_storage_t wait_storage;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                wait_size),
      (iree_host_size_t)payload.wait_semaphore_count, &wait_storage));
  iree_hal_replay_semaphore_list_storage_t signal_storage;
  iree_status_t status = iree_hal_replay_executor_make_semaphore_list(
      executor,
      iree_make_const_byte_span(
          record->payload.data + sizeof(payload) + wait_size, signal_size),
      (iree_host_size_t)payload.signal_semaphore_count, &signal_storage);
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
  if (iree_status_is_ok(status) && signal_storage.list.count != 0) {
    status = iree_hal_semaphore_list_wait(signal_storage.list,
                                          iree_infinite_timeout(),
                                          IREE_ASYNC_WAIT_FLAG_NONE);
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
  iree_host_size_t binding_size = 0;
  if (IREE_UNLIKELY(
          payload.constants_length > IREE_HOST_SIZE_MAX ||
          payload.binding_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul(
              (iree_host_size_t)payload.binding_count,
              sizeof(iree_hal_replay_buffer_ref_payload_t), &binding_size) ||
          sizeof(payload) + (iree_host_size_t)payload.constants_length +
                  binding_size !=
              record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay dispatch payload length mismatch");
  }
  iree_const_byte_span_t constants =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.constants_length);
  const iree_hal_replay_buffer_ref_payload_t* binding_payloads =
      (const iree_hal_replay_buffer_ref_payload_t*)(record->payload.data +
                                                    sizeof(payload) +
                                                    constants.data_length);
  iree_hal_buffer_ref_t* bindings = NULL;
  if (payload.binding_count) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        executor->host_allocator,
        (iree_host_size_t)payload.binding_count * sizeof(*bindings),
        (void**)&bindings));
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
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_lookup(
        executor, payload.command_buffer_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, &command_buffer_entry);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_binding_table_t binding_table = {
        (iree_host_size_t)payload.binding_count,
        bindings,
    };
    status = iree_hal_device_queue_execute(
        device_entry->value.device, payload.queue_affinity, wait_storage.list,
        signal_storage.list, command_buffer_entry->value.command_buffer,
        binding_table, payload.flags);
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
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_I64:
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_CAPABILITIES:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_COUNT:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_INFO:
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_EXPORT_PARAMETERS:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EXECUTABLE_CACHE:
      return iree_hal_replay_executor_create_executable_cache(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_CACHE_PREPARE_EXECUTABLE:
      return iree_hal_replay_executor_prepare_executable(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER:
      return iree_hal_replay_executor_create_command_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE:
      return iree_hal_replay_executor_create_semaphore(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER:
      return iree_hal_replay_executor_allocate_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA:
      return iree_hal_replay_executor_queue_alloca(executor, record);
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
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH:
      return iree_hal_replay_executor_command_buffer_dispatch(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COPY_BUFFER:
      return iree_hal_replay_executor_command_buffer_copy_buffer(executor,
                                                                 record);
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
  if (IREE_UNLIKELY(options->flags != IREE_HAL_REPLAY_EXECUTE_FLAG_NONE ||
                    options->reserved0 != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay execute reserved options must be zero");
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
      &executor, valid_contents, device_group, host_allocator));

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
