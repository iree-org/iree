// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/profile_metadata.h"

#include <string.h>

#include "iree/base/alignment.h"

typedef struct iree_hal_amdgpu_profile_metadata_snapshot_t {
  // Host allocator used for snapshot copies.
  iree_allocator_t host_allocator;
  // Executable records copied from the registry.
  iree_hal_profile_executable_record_t* executable_records;
  // Number of executable records in |executable_records|.
  iree_host_size_t executable_record_count;
  // Packed executable export records copied from the registry.
  uint8_t* executable_export_record_data;
  // Byte length of |executable_export_record_data|.
  iree_host_size_t executable_export_record_data_length;
  // Command-buffer records copied from the registry.
  iree_hal_profile_command_buffer_record_t* command_buffer_records;
  // Number of command-buffer records in |command_buffer_records|.
  iree_host_size_t command_buffer_record_count;
  // Command-operation records copied from the registry.
  iree_hal_profile_command_operation_record_t* command_operation_records;
  // Number of command-operation records in |command_operation_records|.
  iree_host_size_t command_operation_record_count;
  // Cursor position after this snapshot's copied records.
  iree_hal_amdgpu_profile_metadata_cursor_t end_cursor;
} iree_hal_amdgpu_profile_metadata_snapshot_t;

typedef struct iree_hal_amdgpu_profile_hash64_state_t {
  // First SipHash state word.
  uint64_t v0;
  // Second SipHash state word.
  uint64_t v1;
  // Third SipHash state word.
  uint64_t v2;
  // Fourth SipHash state word.
  uint64_t v3;
  // Total number of bytes absorbed by the hash.
  uint64_t total_length;
  // Partial input bytes waiting to form the next 64-bit word.
  uint8_t tail[8];
  // Number of valid bytes in |tail|.
  uint8_t tail_length;
} iree_hal_amdgpu_profile_hash64_state_t;

static uint64_t iree_hal_amdgpu_profile_hash64_rotate_left(uint64_t value,
                                                           int bit_count) {
  return (value << bit_count) | (value >> (64 - bit_count));
}

static void iree_hal_amdgpu_profile_hash64_round(
    iree_hal_amdgpu_profile_hash64_state_t* state) {
  state->v0 += state->v1;
  state->v1 = iree_hal_amdgpu_profile_hash64_rotate_left(state->v1, 13);
  state->v1 ^= state->v0;
  state->v0 = iree_hal_amdgpu_profile_hash64_rotate_left(state->v0, 32);
  state->v2 += state->v3;
  state->v3 = iree_hal_amdgpu_profile_hash64_rotate_left(state->v3, 16);
  state->v3 ^= state->v2;
  state->v0 += state->v3;
  state->v3 = iree_hal_amdgpu_profile_hash64_rotate_left(state->v3, 21);
  state->v3 ^= state->v0;
  state->v2 += state->v1;
  state->v1 = iree_hal_amdgpu_profile_hash64_rotate_left(state->v1, 17);
  state->v1 ^= state->v2;
  state->v2 = iree_hal_amdgpu_profile_hash64_rotate_left(state->v2, 32);
}

static void iree_hal_amdgpu_profile_hash64_initialize(
    uint64_t key0, uint64_t key1,
    iree_hal_amdgpu_profile_hash64_state_t* out_state) {
  memset(out_state, 0, sizeof(*out_state));
  out_state->v0 = UINT64_C(0x736f6d6570736575) ^ key0;
  out_state->v1 = UINT64_C(0x646f72616e646f6d) ^ key1;
  out_state->v2 = UINT64_C(0x6c7967656e657261) ^ key0;
  out_state->v3 = UINT64_C(0x7465646279746573) ^ key1;
}

static void iree_hal_amdgpu_profile_hash64_compress(
    iree_hal_amdgpu_profile_hash64_state_t* state, uint64_t word) {
  state->v3 ^= word;
  for (int i = 0; i < 2; ++i) {
    iree_hal_amdgpu_profile_hash64_round(state);
  }
  state->v0 ^= word;
}

static void iree_hal_amdgpu_profile_hash64_append(
    iree_hal_amdgpu_profile_hash64_state_t* state, const void* data,
    iree_host_size_t data_length) {
  if (data_length == 0) return;

  const uint8_t* cursor = (const uint8_t*)data;
  const uint8_t* const end = cursor + data_length;
  state->total_length += data_length;

  if (state->tail_length != 0) {
    while (cursor < end && state->tail_length < sizeof(state->tail)) {
      state->tail[state->tail_length++] = *cursor++;
    }
    if (state->tail_length == sizeof(state->tail)) {
      iree_hal_amdgpu_profile_hash64_compress(
          state, iree_unaligned_load_le_u64((const uint64_t*)state->tail));
      state->tail_length = 0;
    }
  }

  const iree_host_size_t remaining_length = (iree_host_size_t)(end - cursor);
  const uint8_t* const word_end =
      cursor + remaining_length - (remaining_length % sizeof(uint64_t));
  for (; cursor < word_end; cursor += sizeof(uint64_t)) {
    iree_hal_amdgpu_profile_hash64_compress(
        state, iree_unaligned_load_le_u64((const uint64_t*)cursor));
  }

  while (cursor < end) {
    state->tail[state->tail_length++] = *cursor++;
  }
}

static uint64_t iree_hal_amdgpu_profile_hash64_finalize(
    iree_hal_amdgpu_profile_hash64_state_t* state) {
  uint64_t final_word = (state->total_length & 0xFFu) << 56;
  for (uint8_t i = 0; i < state->tail_length; ++i) {
    final_word |= ((uint64_t)state->tail[i]) << (i * 8);
  }

  state->v3 ^= final_word;
  for (int i = 0; i < 2; ++i) {
    iree_hal_amdgpu_profile_hash64_round(state);
  }
  state->v0 ^= final_word;
  state->v2 ^= 0xFFu;
  for (int i = 0; i < 4; ++i) {
    iree_hal_amdgpu_profile_hash64_round(state);
  }
  return state->v0 ^ state->v1 ^ state->v2 ^ state->v3;
}

static void iree_hal_amdgpu_profile_hash128_initialize(
    iree_hal_amdgpu_profile_hash64_state_t out_states[2]) {
  iree_hal_amdgpu_profile_hash64_initialize(UINT64_C(0x0706050403020100),
                                            UINT64_C(0x0f0e0d0c0b0a0908),
                                            &out_states[0]);
  iree_hal_amdgpu_profile_hash64_initialize(UINT64_C(0x1716151413121110),
                                            UINT64_C(0x1f1e1d1c1b1a1918),
                                            &out_states[1]);
}

static void iree_hal_amdgpu_profile_hash128_append(
    iree_hal_amdgpu_profile_hash64_state_t states[2], const void* data,
    iree_host_size_t data_length) {
  iree_hal_amdgpu_profile_hash64_append(&states[0], data, data_length);
  iree_hal_amdgpu_profile_hash64_append(&states[1], data, data_length);
}

static void iree_hal_amdgpu_profile_hash128_finalize(
    iree_hal_amdgpu_profile_hash64_state_t states[2], uint64_t out_hash[2]) {
  out_hash[0] = iree_hal_amdgpu_profile_hash64_finalize(&states[0]);
  out_hash[1] = iree_hal_amdgpu_profile_hash64_finalize(&states[1]);
}

void iree_hal_amdgpu_profile_metadata_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_profile_metadata_registry_t* out_registry) {
  memset(out_registry, 0, sizeof(*out_registry));
  out_registry->host_allocator = host_allocator;
  out_registry->next_executable_id = 1;
  out_registry->next_command_buffer_id = 1;
  iree_slim_mutex_initialize(&out_registry->mutex);
}

void iree_hal_amdgpu_profile_metadata_deinitialize(
    iree_hal_amdgpu_profile_metadata_registry_t* registry) {
  iree_allocator_t host_allocator = registry->host_allocator;
  iree_allocator_free(host_allocator, registry->command_operation_records);
  iree_allocator_free(host_allocator, registry->command_buffer_records);
  iree_allocator_free(host_allocator, registry->executable_export_record_data);
  iree_allocator_free(host_allocator, registry->executable_records);
  iree_slim_mutex_deinitialize(&registry->mutex);
  memset(registry, 0, sizeof(*registry));
}

void iree_hal_amdgpu_profile_metadata_hash_code_object(
    iree_const_byte_span_t code_object_data, uint64_t out_hash[2]) {
  iree_hal_amdgpu_profile_hash64_state_t states[2];
  iree_hal_amdgpu_profile_hash128_initialize(states);
  iree_hal_amdgpu_profile_hash128_append(states, code_object_data.data,
                                         code_object_data.data_length);
  iree_hal_amdgpu_profile_hash128_finalize(states, out_hash);
}

static void iree_hal_amdgpu_profile_metadata_hash_pipeline(
    const uint64_t code_object_hash[2], uint32_t export_ordinal,
    const iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    iree_string_view_t export_name, uint64_t out_hash[2]) {
  iree_hal_amdgpu_profile_hash64_state_t states[2];
  iree_hal_amdgpu_profile_hash128_initialize(states);
  iree_hal_amdgpu_profile_hash128_append(states, code_object_hash,
                                         2 * sizeof(code_object_hash[0]));
  iree_hal_amdgpu_profile_hash128_append(states, &export_ordinal,
                                         sizeof(export_ordinal));
  iree_hal_amdgpu_profile_hash128_append(
      states, &host_kernel_args->constant_count,
      sizeof(host_kernel_args->constant_count));
  iree_hal_amdgpu_profile_hash128_append(
      states, &host_kernel_args->binding_count,
      sizeof(host_kernel_args->binding_count));
  iree_hal_amdgpu_profile_hash128_append(
      states, host_kernel_args->workgroup_size,
      sizeof(host_kernel_args->workgroup_size));
  const uint64_t export_name_length = export_name.size;
  iree_hal_amdgpu_profile_hash128_append(states, &export_name_length,
                                         sizeof(export_name_length));
  iree_hal_amdgpu_profile_hash128_append(states, export_name.data,
                                         export_name.size);
  iree_hal_amdgpu_profile_hash128_finalize(states, out_hash);
}

static iree_status_t iree_hal_amdgpu_profile_metadata_export_record_length(
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

static iree_status_t iree_hal_amdgpu_profile_metadata_export_data_length(
    iree_host_size_t export_count,
    const iree_hal_executable_export_info_t* export_infos,
    const iree_host_size_t* export_parameter_offsets,
    iree_host_size_t* out_data_length) {
  *out_data_length = 0;
  iree_host_size_t data_length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < export_count && iree_status_is_ok(status);
       ++i) {
    iree_host_size_t record_length = 0;
    status = iree_hal_amdgpu_profile_metadata_export_record_length(
        export_infos[i].name, &record_length);
    if (iree_status_is_ok(status) &&
        IREE_UNLIKELY(!iree_host_size_checked_add(data_length, record_length,
                                                  &data_length))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile executable export metadata length overflow");
    }
    const iree_host_size_t parameter_count =
        export_parameter_offsets[i + 1] - export_parameter_offsets[i];
    if (iree_status_is_ok(status) &&
        IREE_UNLIKELY(parameter_count > UINT32_MAX)) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "profile executable export parameter count exceeds uint32_t");
    }
  }
  if (iree_status_is_ok(status)) {
    *out_data_length = data_length;
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_profile_metadata_append_export_records(
    uint64_t executable_id, iree_host_size_t export_count,
    const iree_hal_executable_export_info_t* export_infos,
    const iree_host_size_t* export_parameter_offsets,
    const uint64_t code_object_hash[2],
    const iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    uint8_t* target_data) {
  uint8_t* cursor = target_data;
  for (iree_host_size_t i = 0; i < export_count; ++i) {
    const iree_string_view_t name = export_infos[i].name;
    iree_host_size_t record_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_export_record_length(
        name, &record_length));

    iree_hal_profile_executable_export_record_t record =
        iree_hal_profile_executable_export_record_default();
    record.record_length = (uint32_t)record_length;
    record.executable_id = executable_id;
    record.export_ordinal = (uint32_t)i;
    record.constant_count = host_kernel_args[i].constant_count;
    record.binding_count = host_kernel_args[i].binding_count;
    record.parameter_count = (uint32_t)(export_parameter_offsets[i + 1] -
                                        export_parameter_offsets[i]);
    record.workgroup_size[0] = host_kernel_args[i].workgroup_size[0];
    record.workgroup_size[1] = host_kernel_args[i].workgroup_size[1];
    record.workgroup_size[2] = host_kernel_args[i].workgroup_size[2];
    if (code_object_hash) {
      record.flags |= IREE_HAL_PROFILE_EXECUTABLE_EXPORT_FLAG_PIPELINE_HASH;
      iree_hal_amdgpu_profile_metadata_hash_pipeline(
          code_object_hash, record.export_ordinal, &host_kernel_args[i], name,
          record.pipeline_hash);
    }
    record.name_length = (uint32_t)name.size;

    memcpy(cursor, &record, sizeof(record));
    if (name.size > 0) {
      memcpy(cursor + sizeof(record), name.data, name.size);
    }
    cursor += record_length;
  }
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_profile_metadata_register_executable(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_host_size_t export_count,
    const iree_hal_executable_export_info_t* export_infos,
    const iree_host_size_t* export_parameter_offsets,
    const uint64_t code_object_hash[2],
    const iree_hal_amdgpu_device_kernel_args_t* host_kernel_args,
    uint64_t* out_executable_id) {
  IREE_ASSERT_ARGUMENT(out_executable_id);
  *out_executable_id = 0;
  if (IREE_UNLIKELY(
          export_count > 0 &&
          (!export_infos || !export_parameter_offsets || !host_kernel_args))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile executable metadata is incomplete");
  }
  if (IREE_UNLIKELY(export_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile executable export count exceeds uint32_t");
  }

  iree_host_size_t export_data_length = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_profile_metadata_export_data_length(
      export_count, export_infos, export_parameter_offsets,
      &export_data_length));

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, export_count);

  iree_slim_mutex_lock(&registry->mutex);

  const uint64_t executable_id = registry->next_executable_id;
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(executable_id > UINT32_MAX)) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile executable id exceeds command buffer storage");
  }

  if (iree_status_is_ok(status) && registry->executable_record_count + 1 >
                                       registry->executable_record_capacity) {
    status = iree_allocator_grow_array(
        registry->host_allocator,
        iree_max((iree_host_size_t)16, registry->executable_record_count + 1),
        sizeof(registry->executable_records[0]),
        &registry->executable_record_capacity,
        (void**)&registry->executable_records);
  }

  iree_host_size_t new_export_data_length =
      registry->executable_export_record_data_length;
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(!iree_host_size_checked_add(new_export_data_length,
                                                export_data_length,
                                                &new_export_data_length))) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "profile executable export metadata length overflow");
  }
  if (iree_status_is_ok(status) &&
      new_export_data_length >
          registry->executable_export_record_data_capacity) {
    status = iree_allocator_grow_array(
        registry->host_allocator,
        iree_max((iree_host_size_t)1024, new_export_data_length),
        sizeof(registry->executable_export_record_data[0]),
        &registry->executable_export_record_data_capacity,
        (void**)&registry->executable_export_record_data);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_profile_executable_record_t record =
        iree_hal_profile_executable_record_default();
    record.executable_id = executable_id;
    record.export_count = (uint32_t)export_count;
    if (code_object_hash) {
      record.flags |= IREE_HAL_PROFILE_EXECUTABLE_FLAG_CODE_OBJECT_HASH;
      record.code_object_hash[0] = code_object_hash[0];
      record.code_object_hash[1] = code_object_hash[1];
    }

    uint8_t* export_data = registry->executable_export_record_data +
                           registry->executable_export_record_data_length;
    status = iree_hal_amdgpu_profile_metadata_append_export_records(
        executable_id, export_count, export_infos, export_parameter_offsets,
        code_object_hash, host_kernel_args, export_data);
    if (iree_status_is_ok(status)) {
      registry->executable_records[registry->executable_record_count++] =
          record;
      registry->executable_export_record_data_length = new_export_data_length;
      ++registry->next_executable_id;
      *out_executable_id = executable_id;
    }
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_profile_metadata_register_command_buffer(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_host_size_t physical_device_ordinal, uint64_t* out_command_buffer_id) {
  IREE_ASSERT_ARGUMENT(out_command_buffer_id);
  *out_command_buffer_id = 0;
  if (IREE_UNLIKELY(physical_device_ordinal > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile command-buffer physical device ordinal exceeds uint32_t");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_slim_mutex_lock(&registry->mutex);

  const uint64_t command_buffer_id = registry->next_command_buffer_id;
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(command_buffer_id == UINT64_MAX)) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "profile command-buffer id space exhausted");
  }
  if (iree_status_is_ok(status) &&
      registry->command_buffer_record_count + 1 >
          registry->command_buffer_record_capacity) {
    status = iree_allocator_grow_array(
        registry->host_allocator,
        iree_max((iree_host_size_t)16,
                 registry->command_buffer_record_count + 1),
        sizeof(registry->command_buffer_records[0]),
        &registry->command_buffer_record_capacity,
        (void**)&registry->command_buffer_records);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_profile_command_buffer_record_t record =
        iree_hal_profile_command_buffer_record_default();
    record.command_buffer_id = command_buffer_id;
    record.mode = mode;
    record.command_categories = command_categories;
    record.queue_affinity = queue_affinity;
    record.physical_device_ordinal = (uint32_t)physical_device_ordinal;
    registry->command_buffer_records[registry->command_buffer_record_count++] =
        record;
    ++registry->next_command_buffer_id;
    *out_command_buffer_id = command_buffer_id;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_hal_amdgpu_profile_metadata_register_command_operations(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_host_size_t operation_count,
    const iree_hal_profile_command_operation_record_t* operations) {
  if (operation_count == 0) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(!operations)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile command-operation records are required");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation_count);

  iree_slim_mutex_lock(&registry->mutex);

  iree_host_size_t new_record_count = 0;
  iree_host_size_t byte_length = 0;
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(registry->command_operation_record_count,
                                      operation_count, &new_record_count))) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "profile command-operation metadata count overflow");
  }
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        0, &byte_length,
        IREE_STRUCT_FIELD(operation_count,
                          iree_hal_profile_command_operation_record_t, NULL));
  }
  if (iree_status_is_ok(status) &&
      new_record_count > registry->command_operation_record_capacity) {
    status = iree_allocator_grow_array(
        registry->host_allocator,
        iree_max((iree_host_size_t)64, new_record_count),
        sizeof(registry->command_operation_records[0]),
        &registry->command_operation_record_capacity,
        (void**)&registry->command_operation_records);
  }
  if (iree_status_is_ok(status)) {
    memcpy(registry->command_operation_records +
               registry->command_operation_record_count,
           operations, byte_length);
    registry->command_operation_record_count = new_record_count;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdgpu_profile_metadata_snapshot_deinitialize(
    iree_hal_amdgpu_profile_metadata_snapshot_t* snapshot) {
  iree_allocator_free(snapshot->host_allocator,
                      snapshot->command_operation_records);
  iree_allocator_free(snapshot->host_allocator,
                      snapshot->command_buffer_records);
  iree_allocator_free(snapshot->host_allocator,
                      snapshot->executable_export_record_data);
  iree_allocator_free(snapshot->host_allocator, snapshot->executable_records);
  memset(snapshot, 0, sizeof(*snapshot));
}

static iree_status_t iree_hal_amdgpu_profile_metadata_snapshot_copy(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    const iree_hal_amdgpu_profile_metadata_cursor_t* cursor,
    iree_hal_amdgpu_profile_metadata_snapshot_t* out_snapshot) {
  memset(out_snapshot, 0, sizeof(*out_snapshot));
  out_snapshot->host_allocator = registry->host_allocator;

  iree_slim_mutex_lock(&registry->mutex);

  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(cursor->executable_record_count >
                        registry->executable_record_count ||
                    cursor->executable_export_record_data_length >
                        registry->executable_export_record_data_length ||
                    cursor->command_buffer_record_count >
                        registry->command_buffer_record_count ||
                    cursor->command_operation_record_count >
                        registry->command_operation_record_count)) {
    status =
        iree_make_status(IREE_STATUS_INTERNAL,
                         "profile metadata cursor is outside registry bounds");
  }

  iree_host_size_t executable_record_count = 0;
  iree_host_size_t executable_export_record_data_length = 0;
  iree_host_size_t command_buffer_record_count = 0;
  iree_host_size_t command_operation_record_count = 0;
  if (iree_status_is_ok(status)) {
    executable_record_count =
        registry->executable_record_count - cursor->executable_record_count;
    executable_export_record_data_length =
        registry->executable_export_record_data_length -
        cursor->executable_export_record_data_length;
    command_buffer_record_count = registry->command_buffer_record_count -
                                  cursor->command_buffer_record_count;
    command_operation_record_count = registry->command_operation_record_count -
                                     cursor->command_operation_record_count;
  }

  if (iree_status_is_ok(status) && executable_record_count > 0) {
    iree_host_size_t byte_length = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &byte_length,
        IREE_STRUCT_FIELD(executable_record_count,
                          iree_hal_profile_executable_record_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(registry->host_allocator, byte_length,
                                     (void**)&out_snapshot->executable_records);
    }
    if (iree_status_is_ok(status)) {
      memcpy(out_snapshot->executable_records,
             registry->executable_records + cursor->executable_record_count,
             byte_length);
      out_snapshot->executable_record_count = executable_record_count;
    }
  }

  if (iree_status_is_ok(status) && executable_export_record_data_length > 0) {
    status = iree_allocator_malloc(
        registry->host_allocator, executable_export_record_data_length,
        (void**)&out_snapshot->executable_export_record_data);
    if (iree_status_is_ok(status)) {
      memcpy(out_snapshot->executable_export_record_data,
             registry->executable_export_record_data +
                 cursor->executable_export_record_data_length,
             executable_export_record_data_length);
      out_snapshot->executable_export_record_data_length =
          executable_export_record_data_length;
    }
  }

  if (iree_status_is_ok(status) && command_buffer_record_count > 0) {
    iree_host_size_t byte_length = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &byte_length,
        IREE_STRUCT_FIELD(command_buffer_record_count,
                          iree_hal_profile_command_buffer_record_t, NULL));
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(registry->host_allocator, byte_length,
                                (void**)&out_snapshot->command_buffer_records);
    }
    if (iree_status_is_ok(status)) {
      memcpy(out_snapshot->command_buffer_records,
             registry->command_buffer_records +
                 cursor->command_buffer_record_count,
             byte_length);
      out_snapshot->command_buffer_record_count = command_buffer_record_count;
    }
  }

  if (iree_status_is_ok(status) && command_operation_record_count > 0) {
    iree_host_size_t byte_length = 0;
    status = IREE_STRUCT_LAYOUT(
        0, &byte_length,
        IREE_STRUCT_FIELD(command_operation_record_count,
                          iree_hal_profile_command_operation_record_t, NULL));
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(
          registry->host_allocator, byte_length,
          (void**)&out_snapshot->command_operation_records);
    }
    if (iree_status_is_ok(status)) {
      memcpy(out_snapshot->command_operation_records,
             registry->command_operation_records +
                 cursor->command_operation_record_count,
             byte_length);
      out_snapshot->command_operation_record_count =
          command_operation_record_count;
    }
  }

  if (iree_status_is_ok(status)) {
    out_snapshot->end_cursor.executable_record_count =
        registry->executable_record_count;
    out_snapshot->end_cursor.executable_export_record_data_length =
        registry->executable_export_record_data_length;
    out_snapshot->end_cursor.command_buffer_record_count =
        registry->command_buffer_record_count;
    out_snapshot->end_cursor.command_operation_record_count =
        registry->command_operation_record_count;
  }

  iree_slim_mutex_unlock(&registry->mutex);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_profile_metadata_snapshot_deinitialize(out_snapshot);
  }
  return status;
}

iree_status_t iree_hal_amdgpu_profile_metadata_write(
    iree_hal_amdgpu_profile_metadata_registry_t* registry,
    iree_hal_profile_sink_t* sink, uint64_t session_id, iree_string_view_t name,
    iree_hal_amdgpu_profile_metadata_cursor_t* cursor) {
  if (!sink) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdgpu_profile_metadata_snapshot_t snapshot;
  iree_status_t status = iree_hal_amdgpu_profile_metadata_snapshot_copy(
      registry, cursor, &snapshot);

  if (iree_status_is_ok(status) && snapshot.executable_record_count > 0) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLES;
    metadata.name = name;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec =
        iree_make_const_byte_span(snapshot.executable_records,
                                  snapshot.executable_record_count *
                                      sizeof(snapshot.executable_records[0]));
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  if (iree_status_is_ok(status) &&
      snapshot.executable_export_record_data_length > 0) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_EXECUTABLE_EXPORTS;
    metadata.name = name;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec = iree_make_const_byte_span(
        snapshot.executable_export_record_data,
        snapshot.executable_export_record_data_length);
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  if (iree_status_is_ok(status) && snapshot.command_buffer_record_count > 0) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_BUFFERS;
    metadata.name = name;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec = iree_make_const_byte_span(
        snapshot.command_buffer_records,
        snapshot.command_buffer_record_count *
            sizeof(snapshot.command_buffer_records[0]));
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  if (iree_status_is_ok(status) &&
      snapshot.command_operation_record_count > 0) {
    iree_hal_profile_chunk_metadata_t metadata =
        iree_hal_profile_chunk_metadata_default();
    metadata.content_type = IREE_HAL_PROFILE_CONTENT_TYPE_COMMAND_OPERATIONS;
    metadata.name = name;
    metadata.session_id = session_id;
    iree_const_byte_span_t iovec = iree_make_const_byte_span(
        snapshot.command_operation_records,
        snapshot.command_operation_record_count *
            sizeof(snapshot.command_operation_records[0]));
    status = iree_hal_profile_sink_write(sink, &metadata, 1, &iovec);
  }

  if (iree_status_is_ok(status)) {
    *cursor = snapshot.end_cursor;
  }
  iree_hal_amdgpu_profile_metadata_snapshot_deinitialize(&snapshot);

  IREE_TRACE_ZONE_END(z0);
  return status;
}
