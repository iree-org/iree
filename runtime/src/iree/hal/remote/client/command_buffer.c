// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/command_buffer.h"

#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/protocol/commands.h"
#include "iree/hal/remote/protocol/control.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_command_buffer_t
//===----------------------------------------------------------------------===//

// Initial stream buffer capacity. Covers small recordings (a few dispatches
// with barriers) without reallocation.
#define IREE_HAL_REMOTE_CB_INITIAL_CAPACITY 4096

static const iree_hal_command_buffer_vtable_t
    iree_hal_remote_client_command_buffer_vtable;

typedef struct iree_hal_remote_client_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;

  // Serialized command stream (iree_hal_remote_cmd_header_t sequence).
  uint8_t* stream_data;
  iree_host_size_t stream_length;
  iree_host_size_t stream_capacity;

  // Server resource ID (reusable only). Initially provisional (from end()),
  // resolved to canonical when upload response arrives.
  iree_hal_remote_resource_id_t resource_id;
} iree_hal_remote_client_command_buffer_t;

static iree_hal_remote_client_command_buffer_t*
iree_hal_remote_client_command_buffer_cast(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_HAL_ASSERT_TYPE(base_command_buffer,
                       &iree_hal_remote_client_command_buffer_vtable);
  return (iree_hal_remote_client_command_buffer_t*)base_command_buffer;
}

//===----------------------------------------------------------------------===//
// Buffer reference helpers
//===----------------------------------------------------------------------===//

// Resolves a buffer reference for wire serialization. Returns the resource_id
// of the root allocation and adjusts the offset to be absolute within that
// allocation (adding the subspan byte_offset if the buffer is a subspan).
static iree_hal_remote_resource_id_t
iree_hal_remote_client_cb_resolve_buffer_ref(iree_hal_buffer_ref_t ref,
                                             iree_device_size_t* out_offset) {
  if (!ref.buffer) {
    *out_offset = ref.offset;
    return 0;
  }
  // Add the buffer's byte_offset (accumulated from subspans) to convert
  // from "relative to this buffer" to "absolute within the allocation".
  *out_offset = ref.offset + iree_hal_buffer_byte_offset(ref.buffer);
  return iree_hal_remote_client_buffer_resource_id(ref.buffer);
}

//===----------------------------------------------------------------------===//
// Stream writing helpers
//===----------------------------------------------------------------------===//

// Ensures the stream buffer has room for |additional_bytes| more bytes.
static iree_status_t iree_hal_remote_client_cb_ensure_capacity(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_host_size_t additional_bytes) {
  iree_host_size_t required = 0;
  if (!iree_host_size_checked_add(command_buffer->stream_length,
                                  additional_bytes, &required)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command stream size overflow");
  }
  if (required <= command_buffer->stream_capacity) return iree_ok_status();

  // Grow by at least 2x to amortize reallocation cost.
  iree_host_size_t new_capacity = command_buffer->stream_capacity * 2;
  if (new_capacity < required) new_capacity = required;

  return iree_allocator_realloc(command_buffer->host_allocator, new_capacity,
                                (void**)&command_buffer->stream_data);
}

// Returns a pointer to |size| bytes at the current write position, growing
// the buffer if needed. Advances stream_length. The returned memory is zeroed.
static iree_status_t iree_hal_remote_client_cb_append(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_host_size_t size, void** out_ptr) {
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_cb_ensure_capacity(command_buffer, size));
  uint8_t* ptr = command_buffer->stream_data + command_buffer->stream_length;
  memset(ptr, 0, size);
  command_buffer->stream_length += size;
  *out_ptr = ptr;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_remote_client_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;

  // Release server-side resource for reusable command buffers.
  if (command_buffer->resource_id != 0) {
    struct {
      iree_hal_remote_control_envelope_t envelope;
      iree_hal_remote_resource_release_batch_t batch;
      iree_hal_remote_resource_id_t resource_ids[1];
    } message;
    memset(&message, 0, sizeof(message));
    message.envelope.message_type =
        IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH;
    message.envelope.message_flags =
        IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;
    message.batch.resource_count = 1;
    message.resource_ids[0] = command_buffer->resource_id;
    iree_status_ignore(iree_hal_remote_client_device_send_fire_and_forget(
        command_buffer->device,
        iree_make_const_byte_span(&message, sizeof(message))));
  }

  iree_allocator_free(host_allocator, command_buffer->stream_data);
  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t iree_hal_remote_client_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  // Re-recording: release old server resource if present.
  if (command_buffer->resource_id != 0) {
    struct {
      iree_hal_remote_control_envelope_t envelope;
      iree_hal_remote_resource_release_batch_t batch;
      iree_hal_remote_resource_id_t resource_ids[1];
    } message;
    memset(&message, 0, sizeof(message));
    message.envelope.message_type =
        IREE_HAL_REMOTE_CONTROL_RESOURCE_RELEASE_BATCH;
    message.envelope.message_flags =
        IREE_HAL_REMOTE_CONTROL_FLAG_FIRE_AND_FORGET;
    message.batch.resource_count = 1;
    message.resource_ids[0] = command_buffer->resource_id;
    iree_status_ignore(iree_hal_remote_client_device_send_fire_and_forget(
        command_buffer->device,
        iree_make_const_byte_span(&message, sizeof(message))));
    command_buffer->resource_id = 0;
  }

  // Reset recording position (keeps the allocation for reuse).
  command_buffer->stream_length = 0;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  // For reusable command buffers, upload the recorded stream to the server.
  // The upload is async via control RPC. We assign a provisional resource ID
  // that can be referenced immediately in queue_execute.
  if (!iree_all_bits_set(base_command_buffer->mode,
                         IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    // Build COMMAND_BUFFER_UPLOAD request.
    iree_host_size_t header_size =
        sizeof(iree_hal_remote_control_envelope_t) +
        sizeof(iree_hal_remote_command_buffer_upload_request_t);
    iree_host_size_t message_size = 0;
    if (!iree_host_size_checked_add(header_size, command_buffer->stream_length,
                                    &message_size)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "command buffer upload size overflow");
    }

    uint8_t* message_buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        command_buffer->host_allocator, message_size, (void**)&message_buffer));
    memset(message_buffer, 0, header_size);

    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message_buffer;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_COMMAND_BUFFER_UPLOAD;

    iree_hal_remote_command_buffer_upload_request_t* request =
        (iree_hal_remote_command_buffer_upload_request_t*)(envelope + 1);
    request->provisional_id = IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
        IREE_HAL_REMOTE_RESOURCE_TYPE_COMMAND_BUFFER, 0);
    request->mode = (uint32_t)base_command_buffer->mode;
    request->categories = (uint32_t)base_command_buffer->allowed_categories;
    request->binding_capacity = (uint16_t)base_command_buffer->binding_capacity;
    request->upload_flags = IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA;
    request->data_length = command_buffer->stream_length;

    // Copy stream data after the header.
    if (command_buffer->stream_length > 0) {
      memcpy(message_buffer + header_size, command_buffer->stream_data,
             command_buffer->stream_length);
    }

    // Send RPC and get resolved resource ID.
    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    iree_status_t status = iree_hal_remote_client_device_control_rpc(
        command_buffer->device,
        iree_make_const_byte_span(message_buffer, message_size),
        &response_payload, &response_lease);

    iree_allocator_free(command_buffer->host_allocator, message_buffer);

    if (iree_status_is_ok(status)) {
      if (response_payload.data_length <
          sizeof(iree_hal_remote_command_buffer_upload_response_t)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "COMMAND_BUFFER_UPLOAD response too short: %" PRIhsz " bytes",
            response_payload.data_length);
      }
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_remote_command_buffer_upload_response_t* response =
          (const iree_hal_remote_command_buffer_upload_response_t*)
              response_payload.data;
      command_buffer->resource_id = response->resolved_id;
    }

    iree_async_buffer_lease_release(&response_lease);
    IREE_RETURN_IF_ERROR(status);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Debug groups
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  iree_host_size_t label_padded = iree_host_align(label.size, 8);
  iree_host_size_t total_size = sizeof(iree_hal_remote_debug_group_begin_cmd_t);
  if (!iree_host_size_checked_add(total_size, label_padded, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "debug group size overflow");
  }

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_cb_append(command_buffer, total_size, &ptr));

  iree_hal_remote_debug_group_begin_cmd_t* cmd =
      (iree_hal_remote_debug_group_begin_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN;
  cmd->header.length = (uint16_t)total_size;
  cmd->label_color = label_color.value;
  cmd->label_length = (uint16_t)label.size;

  if (label.size > 0) {
    memcpy((uint8_t*)(cmd + 1), label.data, label.size);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_cb_append(
      command_buffer, sizeof(iree_hal_remote_debug_group_end_cmd_t), &ptr));

  iree_hal_remote_debug_group_end_cmd_t* cmd =
      (iree_hal_remote_debug_group_end_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END;
  cmd->header.length = (uint16_t)sizeof(*cmd);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  iree_host_size_t memory_barriers_size = 0;
  iree_host_size_t buffer_barriers_size = 0;
  if (!iree_host_size_checked_mul(memory_barrier_count,
                                  sizeof(iree_hal_remote_memory_barrier_t),
                                  &memory_barriers_size) ||
      !iree_host_size_checked_mul(buffer_barrier_count,
                                  sizeof(iree_hal_remote_buffer_barrier_t),
                                  &buffer_barriers_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "barrier size overflow");
  }

  iree_host_size_t total_size = sizeof(iree_hal_remote_execution_barrier_cmd_t);
  if (!iree_host_size_checked_add(total_size, memory_barriers_size,
                                  &total_size) ||
      !iree_host_size_checked_add(total_size, buffer_barriers_size,
                                  &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "barrier command size overflow");
  }
  total_size = iree_host_align(total_size, 8);

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_cb_append(command_buffer, total_size, &ptr));

  iree_hal_remote_execution_barrier_cmd_t* cmd =
      (iree_hal_remote_execution_barrier_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER;
  cmd->header.length = (uint16_t)total_size;
  cmd->source_stage_mask = (uint32_t)source_stage_mask;
  cmd->target_stage_mask = (uint32_t)target_stage_mask;
  cmd->memory_barrier_count = (uint16_t)memory_barrier_count;
  cmd->buffer_barrier_count = (uint16_t)buffer_barrier_count;

  uint8_t* cursor = (uint8_t*)(cmd + 1);

  // Serialize memory barriers.
  for (iree_host_size_t i = 0; i < memory_barrier_count; ++i) {
    iree_hal_remote_memory_barrier_t* wire =
        (iree_hal_remote_memory_barrier_t*)cursor;
    wire->source_scope = (uint32_t)memory_barriers[i].source_scope;
    wire->target_scope = (uint32_t)memory_barriers[i].target_scope;
    cursor += sizeof(*wire);
  }

  // Serialize buffer barriers (resolve buffer IDs + subspan offsets).
  for (iree_host_size_t i = 0; i < buffer_barrier_count; ++i) {
    iree_hal_remote_buffer_barrier_t* wire =
        (iree_hal_remote_buffer_barrier_t*)cursor;
    wire->source_scope = (uint32_t)buffer_barriers[i].source_scope;
    wire->target_scope = (uint32_t)buffer_barriers[i].target_scope;
    iree_device_size_t barrier_offset = 0;
    wire->buffer_id = iree_hal_remote_client_cb_resolve_buffer_ref(
        buffer_barriers[i].buffer_ref, &barrier_offset);
    wire->offset = barrier_offset;
    wire->length = buffer_barriers[i].buffer_ref.length;
    cursor += sizeof(*wire);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "events not supported on remote device");
}

static iree_status_t iree_hal_remote_client_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "events not supported on remote device");
}

static iree_status_t iree_hal_remote_client_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "events not supported on remote device");
}

//===----------------------------------------------------------------------===//
// Transfer operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_cb_append(
      command_buffer, sizeof(iree_hal_remote_buffer_fill_cmd_t), &ptr));

  iree_hal_remote_buffer_fill_cmd_t* cmd =
      (iree_hal_remote_buffer_fill_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_BUFFER_FILL;
  cmd->header.length = (uint16_t)sizeof(*cmd);
  iree_device_size_t target_offset = 0;
  cmd->target_buffer_id =
      iree_hal_remote_client_cb_resolve_buffer_ref(target_ref, &target_offset);
  cmd->target_offset = target_offset;
  cmd->target_length = target_ref.length;
  cmd->pattern_length = (uint8_t)pattern_length;
  cmd->fill_flags = (uint32_t)flags;

  // Zero-extend pattern into uint32_t.
  memcpy(&cmd->pattern, pattern,
         iree_min(pattern_length, sizeof(cmd->pattern)));

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  iree_host_size_t data_length = (iree_host_size_t)target_ref.length;
  iree_host_size_t data_padded = iree_host_align(data_length, 8);
  iree_host_size_t total_size = 0;
  if (!iree_host_size_checked_add(sizeof(iree_hal_remote_buffer_update_cmd_t),
                                  data_padded, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "update buffer command size overflow");
  }

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_cb_append(command_buffer, total_size, &ptr));

  iree_hal_remote_buffer_update_cmd_t* cmd =
      (iree_hal_remote_buffer_update_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_BUFFER_UPDATE;
  cmd->header.length = (uint16_t)total_size;
  iree_device_size_t update_target_offset = 0;
  cmd->target_buffer_id = iree_hal_remote_client_cb_resolve_buffer_ref(
      target_ref, &update_target_offset);
  cmd->target_offset = update_target_offset;
  cmd->target_length = target_ref.length;
  cmd->update_flags = (uint32_t)flags;

  // Deep-copy the source data into the stream.
  memcpy((uint8_t*)(cmd + 1), (const uint8_t*)source_buffer + source_offset,
         data_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_cb_append(
      command_buffer, sizeof(iree_hal_remote_buffer_copy_cmd_t), &ptr));

  iree_hal_remote_buffer_copy_cmd_t* cmd =
      (iree_hal_remote_buffer_copy_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_BUFFER_COPY;
  cmd->header.length = (uint16_t)sizeof(*cmd);
  iree_device_size_t copy_source_offset = 0;
  iree_device_size_t copy_target_offset = 0;
  cmd->source_buffer_id = iree_hal_remote_client_cb_resolve_buffer_ref(
      source_ref, &copy_source_offset);
  cmd->source_offset = copy_source_offset;
  cmd->target_buffer_id = iree_hal_remote_client_cb_resolve_buffer_ref(
      target_ref, &copy_target_offset);
  cmd->target_offset = copy_target_offset;
  cmd->length = source_ref.length;
  cmd->copy_flags = (uint32_t)flags;

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_memory_advise_flags_t advise_flags, uint64_t arg0, uint64_t arg1) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_cb_append(
      command_buffer, sizeof(iree_hal_remote_buffer_advise_cmd_t), &ptr));

  iree_hal_remote_buffer_advise_cmd_t* cmd =
      (iree_hal_remote_buffer_advise_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_BUFFER_ADVISE;
  cmd->header.length = (uint16_t)sizeof(*cmd);
  iree_device_size_t advise_offset = 0;
  cmd->buffer_id =
      iree_hal_remote_client_cb_resolve_buffer_ref(buffer_ref, &advise_offset);
  cmd->offset = advise_offset;
  cmd->length = buffer_ref.length;
  cmd->advise_flags = (uint32_t)advise_flags;
  cmd->argument0 = arg0;
  cmd->argument1 = arg1;

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Collective operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not supported on remote device");
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t entry_point,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  uint16_t constant_count =
      (uint16_t)(constants.data_length / sizeof(uint32_t));
  uint16_t binding_count = (uint16_t)bindings.count;

  iree_host_size_t constants_size = 0;
  iree_host_size_t bindings_size = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)constant_count,
                                  sizeof(uint32_t), &constants_size) ||
      !iree_host_size_checked_mul((iree_host_size_t)binding_count,
                                  sizeof(iree_hal_remote_binding_t),
                                  &bindings_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch command size overflow");
  }
  iree_host_size_t constants_padded = iree_host_align(constants_size, 8);

  iree_host_size_t total_size = sizeof(iree_hal_remote_dispatch_cmd_t);
  if (!iree_host_size_checked_add(total_size, constants_padded, &total_size) ||
      !iree_host_size_checked_add(total_size, bindings_size, &total_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "dispatch command size overflow");
  }
  total_size = iree_host_align(total_size, 8);

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_remote_client_cb_append(command_buffer, total_size, &ptr));

  iree_hal_remote_dispatch_cmd_t* cmd = (iree_hal_remote_dispatch_cmd_t*)ptr;
  cmd->header.type = IREE_HAL_REMOTE_CMD_DISPATCH;
  cmd->header.length = (uint16_t)total_size;
  cmd->executable_id =
      iree_hal_remote_client_executable_resource_id(executable);
  cmd->export_ordinal = entry_point;
  memcpy(cmd->config.workgroup_size, config.workgroup_size,
         sizeof(config.workgroup_size));
  memcpy(cmd->config.workgroup_count, config.workgroup_count,
         sizeof(config.workgroup_count));
  cmd->config.dynamic_workgroup_local_memory =
      config.dynamic_workgroup_local_memory;
  cmd->constant_count = constant_count;
  cmd->binding_count = binding_count;
  cmd->dispatch_flags = flags;

  // Constants (padded to 8-byte alignment).
  uint8_t* cursor = (uint8_t*)(cmd + 1);
  if (constants_size > 0) {
    memcpy(cursor, constants.data, constants_size);
  }
  cursor += constants_padded;

  // Bindings.
  iree_hal_remote_binding_t* wire_bindings = (iree_hal_remote_binding_t*)cursor;
  for (uint16_t i = 0; i < binding_count; ++i) {
    const iree_hal_buffer_ref_t* ref = &bindings.values[i];
    iree_device_size_t binding_offset = 0;
    wire_bindings[i].buffer_id =
        iree_hal_remote_client_cb_resolve_buffer_ref(*ref, &binding_offset);
    wire_bindings[i].offset = binding_offset;
    wire_bindings[i].length = ref->length;
    wire_bindings[i].buffer_slot = ref->buffer_slot;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_remote_client_command_buffer_vtable = {
        .destroy = iree_hal_remote_client_command_buffer_destroy,
        .begin = iree_hal_remote_client_command_buffer_begin,
        .end = iree_hal_remote_client_command_buffer_end,
        .begin_debug_group =
            iree_hal_remote_client_command_buffer_begin_debug_group,
        .end_debug_group =
            iree_hal_remote_client_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_remote_client_command_buffer_execution_barrier,
        .signal_event = iree_hal_remote_client_command_buffer_signal_event,
        .reset_event = iree_hal_remote_client_command_buffer_reset_event,
        .wait_events = iree_hal_remote_client_command_buffer_wait_events,
        .fill_buffer = iree_hal_remote_client_command_buffer_fill_buffer,
        .update_buffer = iree_hal_remote_client_command_buffer_update_buffer,
        .copy_buffer = iree_hal_remote_client_command_buffer_copy_buffer,
        .advise_buffer = iree_hal_remote_client_command_buffer_advise_buffer,
        .collective = iree_hal_remote_client_command_buffer_collective,
        .dispatch = iree_hal_remote_client_command_buffer_dispatch,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_remote_client_command_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  iree_host_size_t validation_size =
      iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
  iree_host_size_t total_size =
      sizeof(iree_hal_remote_client_command_buffer_t) + validation_size;

  iree_hal_remote_client_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(host_allocator, total_size,
                                               (void**)&command_buffer);

  if (iree_status_is_ok(status)) {
    // Allocate initial stream buffer.
    status = iree_allocator_malloc(host_allocator,
                                   IREE_HAL_REMOTE_CB_INITIAL_CAPACITY,
                                   (void**)&command_buffer->stream_data);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        iree_hal_device_allocator((iree_hal_device_t*)device), mode,
        command_categories, queue_affinity, binding_capacity,
        (uint8_t*)command_buffer + sizeof(*command_buffer),
        &iree_hal_remote_client_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    command_buffer->device = device;
    command_buffer->stream_length = 0;
    command_buffer->stream_capacity = IREE_HAL_REMOTE_CB_INITIAL_CAPACITY;
    command_buffer->resource_id = 0;
    *out_command_buffer = &command_buffer->base;
  } else {
    if (command_buffer) {
      iree_allocator_free(host_allocator, command_buffer->stream_data);
      iree_allocator_free(host_allocator, command_buffer);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_remote_client_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(command_buffer,
                              &iree_hal_remote_client_command_buffer_vtable);
}

iree_const_byte_span_t iree_hal_remote_client_command_buffer_stream(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  return iree_make_const_byte_span(command_buffer->stream_data,
                                   command_buffer->stream_length);
}

iree_hal_remote_resource_id_t iree_hal_remote_client_command_buffer_resource_id(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  return command_buffer->resource_id;
}
