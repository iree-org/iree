// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_command_buffer.h"

#include <string.h>

#include "iree/hal/replay/recorder_buffer.h"
#include "iree/hal/replay/recorder_executable.h"
#include "iree/hal/replay/recorder_record.h"

#define IREE_HAL_REPLAY_VTABLE_DISPATCH(resource, type_prefix, method_name) \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource))  \
       ->vtable)                                                            \
      ->method_name

typedef struct iree_hal_replay_recorder_command_buffer_t {
  // HAL command buffer header for the recording wrapper command buffer.
  iree_hal_command_buffer_t base;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying command buffer receiving forwarded HAL calls.
  iree_hal_command_buffer_t* base_command_buffer;
  // Session-local device object id associated with this command buffer.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this command buffer.
  iree_hal_replay_object_id_t command_buffer_id;
} iree_hal_replay_recorder_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_replay_recorder_command_buffer_vtable;

static bool iree_hal_replay_recorder_command_buffer_isa(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_hal_resource_is(base_command_buffer,
                              &iree_hal_replay_recorder_command_buffer_vtable);
}

static iree_hal_replay_recorder_command_buffer_t*
iree_hal_replay_recorder_command_buffer_cast(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_HAL_ASSERT_TYPE(base_command_buffer,
                       &iree_hal_replay_recorder_command_buffer_vtable);
  return (iree_hal_replay_recorder_command_buffer_t*)base_command_buffer;
}

void iree_hal_replay_recorder_command_buffer_make_object_payload(
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_hal_replay_command_buffer_object_payload_t* out_payload) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->mode = mode;
  out_payload->command_categories = command_categories;
  out_payload->queue_affinity = queue_affinity;
  out_payload->binding_capacity = binding_capacity;
}

iree_hal_command_buffer_t* iree_hal_replay_recorder_command_buffer_base_or_self(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_replay_recorder_command_buffer_isa(command_buffer)
             ? iree_hal_replay_recorder_command_buffer_cast(command_buffer)
                   ->base_command_buffer
             : command_buffer;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_command_buffer_id_or_none(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_replay_recorder_command_buffer_isa(command_buffer)
             ? iree_hal_replay_recorder_command_buffer_cast(command_buffer)
                   ->command_buffer_id
             : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

iree_status_t iree_hal_replay_recorder_command_buffer_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t command_buffer_id,
    iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_t* base_command_buffer,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(base_command_buffer);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  iree_host_size_t validation_state_size =
      iree_hal_command_buffer_validation_state_size(
          iree_hal_command_buffer_mode(base_command_buffer),
          base_command_buffer->binding_capacity);
  iree_host_size_t total_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          sizeof(iree_hal_replay_recorder_command_buffer_t),
          validation_state_size, &total_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay command buffer size overflow");
  }

  iree_hal_replay_recorder_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, total_size,
                                             (void**)&command_buffer));
  memset(command_buffer, 0, total_size);

  void* validation_state =
      validation_state_size ? (uint8_t*)command_buffer + sizeof(*command_buffer)
                            : NULL;
  iree_hal_command_buffer_initialize(
      device_allocator, iree_hal_command_buffer_mode(base_command_buffer),
      iree_hal_command_buffer_allowed_categories(base_command_buffer),
      iree_hal_command_buffer_queue_affinity(base_command_buffer),
      base_command_buffer->binding_capacity, validation_state,
      &iree_hal_replay_recorder_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->recorder = recorder;
  iree_hal_replay_recorder_retain(command_buffer->recorder);
  command_buffer->base_command_buffer = base_command_buffer;
  iree_hal_command_buffer_retain(command_buffer->base_command_buffer);
  command_buffer->device_id = device_id;
  command_buffer->command_buffer_id = command_buffer_id;

  *out_command_buffer = &command_buffer->base;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_command_buffer_begin_operation(
    iree_hal_replay_recorder_command_buffer_t* command_buffer,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      command_buffer->recorder, command_buffer->device_id,
      command_buffer->command_buffer_id, IREE_HAL_REPLAY_OBJECT_ID_NONE,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, operation_code, payload_type,
      out_pending_record);
}

static void iree_hal_replay_recorder_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_command_buffer_release(command_buffer->base_command_buffer);
  iree_hal_replay_recorder_release(command_buffer->recorder);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_replay_recorder_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_begin_operation(
      command_buffer, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_begin(command_buffer->base_command_buffer));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_begin_operation(
      command_buffer, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_end(command_buffer->base_command_buffer));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_passthrough(
    iree_hal_replay_recorder_command_buffer_t* command_buffer,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_command_buffer_begin_operation(
      command_buffer, operation_code, IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE,
      out_pending_record);
}

static iree_status_t iree_hal_replay_recorder_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer,
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN_DEBUG_GROUP,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_begin_debug_group(
          command_buffer->base_command_buffer, label, label_color, location));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer,
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END_DEBUG_GROUP,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_command_buffer_end_debug_group(
                           command_buffer->base_command_buffer));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_command_buffer_execution_barrier_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.source_stage_mask = source_stage_mask;
  payload.target_stage_mask = target_stage_mask;
  payload.flags = flags;
  payload.memory_barrier_count = memory_barrier_count;
  payload.buffer_barrier_count = buffer_barrier_count;

  iree_hal_replay_memory_barrier_payload_t* memory_payloads = NULL;
  iree_host_size_t memory_payloads_size = 0;
  iree_hal_replay_buffer_barrier_payload_t* buffer_payloads = NULL;
  iree_host_size_t buffer_payloads_size = 0;
  iree_hal_buffer_barrier_t* base_buffer_barriers = NULL;
  iree_hal_buffer_t** temporary_buffers = NULL;
  iree_status_t status = iree_ok_status();
  if (memory_barrier_count) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(memory_barrier_count,
                                                  sizeof(*memory_payloads),
                                                  &memory_payloads_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay command buffer memory barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(command_buffer->host_allocator,
                                memory_payloads_size, (void**)&memory_payloads);
    }
    if (iree_status_is_ok(status)) {
      for (iree_host_size_t i = 0; i < memory_barrier_count; ++i) {
        memory_payloads[i].source_scope = memory_barriers[i].source_scope;
        memory_payloads[i].target_scope = memory_barriers[i].target_scope;
      }
    }
  }
  if (iree_status_is_ok(status) && buffer_barrier_count) {
    iree_host_size_t base_buffer_barriers_size = 0;
    iree_host_size_t temporary_buffers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(buffer_barrier_count,
                                                  sizeof(*buffer_payloads),
                                                  &buffer_payloads_size) ||
                      !iree_host_size_checked_mul(buffer_barrier_count,
                                                  sizeof(*base_buffer_barriers),
                                                  &base_buffer_barriers_size) ||
                      !iree_host_size_checked_mul(buffer_barrier_count,
                                                  sizeof(*temporary_buffers),
                                                  &temporary_buffers_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay command buffer buffer barrier count overflow");
    }
    if (iree_status_is_ok(status)) {
      status =
          iree_allocator_malloc(command_buffer->host_allocator,
                                buffer_payloads_size, (void**)&buffer_payloads);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(command_buffer->host_allocator,
                                     base_buffer_barriers_size,
                                     (void**)&base_buffer_barriers);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(command_buffer->host_allocator,
                                     temporary_buffers_size,
                                     (void**)&temporary_buffers);
    }
    if (iree_status_is_ok(status)) {
      memset(temporary_buffers, 0, temporary_buffers_size);
      memcpy(base_buffer_barriers, buffer_barriers, base_buffer_barriers_size);
      for (iree_host_size_t i = 0;
           i < buffer_barrier_count && iree_status_is_ok(status); ++i) {
        buffer_payloads[i].source_scope = buffer_barriers[i].source_scope;
        buffer_payloads[i].target_scope = buffer_barriers[i].target_scope;
        iree_hal_replay_recorder_buffer_ref_make_payload(
            buffer_barriers[i].buffer_ref, &buffer_payloads[i].buffer_ref);
        if (base_buffer_barriers[i].buffer_ref.buffer) {
          status = iree_hal_replay_recorder_buffer_unwrap_for_call(
              base_buffer_barriers[i].buffer_ref.buffer,
              command_buffer->host_allocator,
              &base_buffer_barriers[i].buffer_ref.buffer,
              &temporary_buffers[i]);
        }
      }
    }
  }

  iree_const_byte_span_t iovecs[3] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(memory_payloads, memory_payloads_size),
      iree_make_const_byte_span(buffer_payloads, buffer_payloads_size),
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_begin_operation(
        command_buffer,
        IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER,
        &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_command_buffer_execution_barrier(
            command_buffer->base_command_buffer, source_stage_mask,
            target_stage_mask, flags, memory_barrier_count, memory_barriers,
            buffer_barrier_count,
            base_buffer_barriers ? base_buffer_barriers : buffer_barriers),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (temporary_buffers) {
    for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(temporary_buffers[i]);
    }
  }
  iree_allocator_free(command_buffer->host_allocator, temporary_buffers);
  iree_allocator_free(command_buffer->host_allocator, base_buffer_barriers);
  iree_allocator_free(command_buffer->host_allocator, buffer_payloads);
  iree_allocator_free(command_buffer->host_allocator, memory_payloads);
  return status;
}

static iree_status_t iree_hal_replay_recorder_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer,
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_SIGNAL_EVENT,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_signal_event(command_buffer->base_command_buffer,
                                           event, source_stage_mask));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_RESET_EVENT,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_reset_event(command_buffer->base_command_buffer,
                                          event, source_stage_mask));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_WAIT_EVENTS,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_wait_events(
          command_buffer->base_command_buffer, event_count, events,
          source_stage_mask, target_stage_mask, memory_barrier_count,
          memory_barriers, buffer_barrier_count, buffer_barriers));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer,
      IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ADVISE_BUFFER,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_command_buffer_advise_buffer(command_buffer->base_command_buffer,
                                            buffer_ref, flags, arg0, arg1));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_command_buffer_fill_buffer_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(target_ref,
                                                   &payload.target_ref);
  payload.flags = flags;
  payload.pattern_length = pattern_length;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(pattern, pattern_length),
  };

  iree_hal_buffer_ref_t base_target_ref = target_ref;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  iree_status_t status = iree_ok_status();
  if (target_ref.buffer) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_ref.buffer, command_buffer->host_allocator,
        &base_target_ref.buffer, &temporary_target_buffer);
  }
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_begin_operation(
        command_buffer,
        IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER,
        &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_command_buffer_fill_buffer(command_buffer->base_command_buffer,
                                            base_target_ref, pattern,
                                            pattern_length, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  return status;
}

static iree_status_t iree_hal_replay_recorder_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  if (IREE_UNLIKELY(target_ref.length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "replay command buffer update length exceeds host size");
  }
  if (IREE_UNLIKELY(target_ref.length != 0 && !source_buffer)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replay command buffer update source buffer is required");
  }

  iree_hal_replay_command_buffer_update_buffer_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(target_ref,
                                                   &payload.target_ref);
  payload.flags = flags;
  payload.source_offset = source_offset;
  payload.data_length = target_ref.length;
  const uint8_t* source_data =
      target_ref.length ? (const uint8_t*)source_buffer + source_offset : NULL;
  iree_const_byte_span_t iovecs[2] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(source_data,
                                (iree_host_size_t)payload.data_length),
  };

  iree_hal_buffer_ref_t base_target_ref = target_ref;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  iree_status_t status = iree_ok_status();
  if (target_ref.buffer) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_ref.buffer, command_buffer->host_allocator,
        &base_target_ref.buffer, &temporary_target_buffer);
  }
  iree_hal_replay_pending_record_t pending_record = {0};
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_begin_operation(
        command_buffer,
        IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER,
        &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_command_buffer_update_buffer(
            command_buffer->base_command_buffer, source_buffer, source_offset,
            base_target_ref, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  return status;
}

static iree_status_t iree_hal_replay_recorder_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_command_buffer_copy_buffer_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  iree_hal_replay_recorder_buffer_ref_make_payload(source_ref,
                                                   &payload.source_ref);
  iree_hal_replay_recorder_buffer_ref_make_payload(target_ref,
                                                   &payload.target_ref);
  payload.flags = flags;
  iree_const_byte_span_t payload_iovec =
      iree_make_const_byte_span(&payload, sizeof(payload));

  iree_hal_buffer_ref_t base_source_ref = source_ref;
  iree_hal_buffer_t* temporary_source_buffer = NULL;
  iree_hal_buffer_ref_t base_target_ref = target_ref;
  iree_hal_buffer_t* temporary_target_buffer = NULL;
  iree_status_t status = iree_ok_status();
  if (source_ref.buffer) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        source_ref.buffer, command_buffer->host_allocator,
        &base_source_ref.buffer, &temporary_source_buffer);
  }
  if (iree_status_is_ok(status) && target_ref.buffer) {
    status = iree_hal_replay_recorder_buffer_unwrap_for_call(
        target_ref.buffer, command_buffer->host_allocator,
        &base_target_ref.buffer, &temporary_target_buffer);
  }
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_begin_operation(
        command_buffer,
        IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COPY_BUFFER,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER,
        &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_command_buffer_copy_buffer(command_buffer->base_command_buffer,
                                            base_source_ref, base_target_ref,
                                            flags),
        1, &payload_iovec);
  }
  iree_hal_replay_recorder_buffer_release_temporary(temporary_target_buffer);
  iree_hal_replay_recorder_buffer_release_temporary(temporary_source_buffer);
  return status;
}

static iree_status_t iree_hal_replay_recorder_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_command_buffer_passthrough(
      command_buffer, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COLLECTIVE,
      &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_command_buffer_collective(
                           command_buffer->base_command_buffer, channel, op,
                           param, send_ref, recv_ref, element_count));
}

static iree_status_t iree_hal_replay_recorder_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_replay_recorder_command_buffer_t* command_buffer =
      iree_hal_replay_recorder_command_buffer_cast(base_command_buffer);

  iree_hal_replay_dispatch_payload_t payload;
  memset(&payload, 0, sizeof(payload));
  payload.executable_id =
      iree_hal_replay_recorder_executable_id_or_none(executable);
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
  payload.constants_length = constants.data_length;
  payload.binding_count = bindings.count;

  iree_hal_replay_buffer_ref_payload_t* binding_payloads = NULL;
  iree_host_size_t binding_payloads_size = 0;
  iree_hal_buffer_ref_t* base_binding_storage = NULL;
  iree_hal_buffer_t** temporary_buffers = NULL;
  iree_hal_buffer_ref_list_t base_bindings = bindings;
  iree_status_t status = iree_ok_status();
  if (bindings.count) {
    iree_host_size_t base_binding_storage_size = 0;
    iree_host_size_t temporary_buffers_size = 0;
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*binding_payloads),
                                                  &binding_payloads_size) ||
                      !iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*base_binding_storage),
                                                  &base_binding_storage_size) ||
                      !iree_host_size_checked_mul(bindings.count,
                                                  sizeof(*temporary_buffers),
                                                  &temporary_buffers_size))) {
      status = iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "replay command buffer dispatch binding count overflow");
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(command_buffer->host_allocator,
                                     binding_payloads_size,
                                     (void**)&binding_payloads);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(command_buffer->host_allocator,
                                     base_binding_storage_size,
                                     (void**)&base_binding_storage);
    }
    if (iree_status_is_ok(status)) {
      status = iree_allocator_malloc(command_buffer->host_allocator,
                                     temporary_buffers_size,
                                     (void**)&temporary_buffers);
    }
    if (iree_status_is_ok(status)) {
      memset(temporary_buffers, 0, temporary_buffers_size);
      memcpy(base_binding_storage, bindings.values, base_binding_storage_size);
      for (iree_host_size_t i = 0;
           i < bindings.count && iree_status_is_ok(status); ++i) {
        iree_hal_replay_recorder_buffer_ref_make_payload(bindings.values[i],
                                                         &binding_payloads[i]);
        if (base_binding_storage[i].buffer) {
          status = iree_hal_replay_recorder_buffer_unwrap_for_call(
              base_binding_storage[i].buffer, command_buffer->host_allocator,
              &base_binding_storage[i].buffer, &temporary_buffers[i]);
        }
      }
      base_bindings.values = base_binding_storage;
    }
  }

  iree_const_byte_span_t iovecs[5] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(NULL, 0),
      iree_make_const_byte_span(NULL, 0),
      constants,
      iree_make_const_byte_span(binding_payloads, binding_payloads_size),
  };
  iree_hal_replay_pending_record_t pending_record;
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_command_buffer_begin_operation(
        command_buffer, IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH, &pending_record);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation_with_payload(
        &pending_record,
        iree_hal_command_buffer_dispatch(
            command_buffer->base_command_buffer,
            iree_hal_replay_recorder_executable_base_or_self(executable),
            export_ordinal, config, constants, base_bindings, flags),
        IREE_ARRAYSIZE(iovecs), iovecs);
  }

  if (temporary_buffers) {
    for (iree_host_size_t i = 0; i < bindings.count; ++i) {
      iree_hal_replay_recorder_buffer_release_temporary(temporary_buffers[i]);
    }
  }
  iree_allocator_free(command_buffer->host_allocator, temporary_buffers);
  iree_allocator_free(command_buffer->host_allocator, base_binding_storage);
  iree_allocator_free(command_buffer->host_allocator, binding_payloads);
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_replay_recorder_command_buffer_vtable = {
        .destroy = iree_hal_replay_recorder_command_buffer_destroy,
        .begin = iree_hal_replay_recorder_command_buffer_begin,
        .end = iree_hal_replay_recorder_command_buffer_end,
        .begin_debug_group =
            iree_hal_replay_recorder_command_buffer_begin_debug_group,
        .end_debug_group =
            iree_hal_replay_recorder_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_replay_recorder_command_buffer_execution_barrier,
        .signal_event = iree_hal_replay_recorder_command_buffer_signal_event,
        .reset_event = iree_hal_replay_recorder_command_buffer_reset_event,
        .wait_events = iree_hal_replay_recorder_command_buffer_wait_events,
        .advise_buffer = iree_hal_replay_recorder_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_replay_recorder_command_buffer_fill_buffer,
        .update_buffer = iree_hal_replay_recorder_command_buffer_update_buffer,
        .copy_buffer = iree_hal_replay_recorder_command_buffer_copy_buffer,
        .collective = iree_hal_replay_recorder_command_buffer_collective,
        .dispatch = iree_hal_replay_recorder_command_buffer_dispatch,
};
