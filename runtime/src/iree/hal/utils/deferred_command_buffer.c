// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/deferred_command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// Command recording structures
//===----------------------------------------------------------------------===//

typedef enum iree_hal_command_type_e {
  IREE_HAL_CMD_BEGIN_DEBUG_GROUP = 0,
  IREE_HAL_CMD_END_DEBUG_GROUP,
  IREE_HAL_CMD_EXECUTION_BARRIER,
  IREE_HAL_CMD_SIGNAL_EVENT,
  IREE_HAL_CMD_RESET_EVENT,
  IREE_HAL_CMD_WAIT_EVENTS,
  IREE_HAL_CMD_ADVISE_BUFFER,
  IREE_HAL_CMD_FILL_BUFFER,
  IREE_HAL_CMD_UPDATE_BUFFER,
  IREE_HAL_CMD_COPY_BUFFER,
  IREE_HAL_CMD_COLLECTIVE,
  IREE_HAL_CMD_DISPATCH,
} iree_hal_cmd_type_t;

// Header prefixed to all commands, forming a linked-list.
//
// Each command is allocated from the arena and does *not* retain any resources;
// the command buffer has a resource set that does lifetime tracking.
//
// We could elide some of these commands by keeping local state however that
// requires knowing more about the target device (pipeline layouts, etc) and
// prevents using this as a way to debug or benchmark command buffers. The
// intent is that each command captures the exact information passed during the
// call such that the target command buffer cannot tell they were deferred.
//
// As each command is variable sized we store pointers to the following command
// to allow us to walk the list during replay. Storing just a size would be
// insufficient as commands may be spread across many arena blocks from the
// block pool.
typedef struct iree_hal_cmd_header_t {
  // Next command in the list or NULL if the end.
  struct iree_hal_cmd_header_t* next;
  // Type of the command that follows.
  iree_hal_cmd_type_t type;
} iree_hal_cmd_header_t;

typedef iree_status_t (*iree_hal_cmd_apply_fn_t)(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_cmd_header_t* cmd_header);

//===----------------------------------------------------------------------===//
// Command list allocation and storage
//===----------------------------------------------------------------------===//

// A singly-linked list of commands allocated from an arena.
typedef struct iree_hal_cmd_list_t {
  // Arena used to hold the recorded commands using block_pool for storage.
  // Will be reset as the command buffer is re-recorded.
  iree_arena_allocator_t arena;

  // Head of the command list.
  iree_hal_cmd_header_t* head;
  // Tail of the command list (may be head).
  iree_hal_cmd_header_t* tail;
} iree_hal_cmd_list_t;

// Initializes a new command list that allocates from the given |block_pool|.
// Upon return the command list is ready for recording.
static void iree_hal_cmd_list_initialize(iree_arena_block_pool_t* block_pool,
                                         iree_hal_cmd_list_t* out_cmd_list) {
  iree_arena_initialize(block_pool, &out_cmd_list->arena);
  out_cmd_list->head = NULL;
  out_cmd_list->tail = NULL;
}

// Returns true if the |cmd_list| is empty.
static bool iree_hal_cmd_list_is_empty(const iree_hal_cmd_list_t* cmd_list) {
  return cmd_list->head == NULL;
}

// Resets the command list and returns all arena blocks back to the block pool.
// Upon return the command list is ready for recording.
static void iree_hal_cmd_list_reset(iree_hal_cmd_list_t* cmd_list) {
  // We could make reset retain a single block so as we know that we'll be
  // adding more commands on this path and it would remove a round-trip through
  // the pool.
  iree_arena_reset(&cmd_list->arena);
  cmd_list->head = NULL;
  cmd_list->tail = NULL;
}

// Deinitializes the command list, preparing for destruction.
static void iree_hal_cmd_list_deinitialize(iree_hal_cmd_list_t* cmd_list) {
  iree_hal_cmd_list_reset(cmd_list);
}

// Appends a new command to the command list and returns the base pointer to its
// storage. Callers must cast to the appropriate type and populate all fields.
static iree_status_t iree_hal_cmd_list_append_command(
    iree_hal_cmd_list_t* cmd_list, iree_hal_cmd_type_t command_type,
    iree_host_size_t command_size, void** out_cmd) {
  iree_hal_cmd_header_t* header = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&cmd_list->arena, command_size, (void**)&header));
  header->next = NULL;
  header->type = command_type;
  if (!cmd_list->head) {
    cmd_list->head = header;
  } else if (cmd_list->tail) {
    cmd_list->tail->next = header;
  }
  cmd_list->tail = header;
  *out_cmd = header;
  return iree_ok_status();
}

// Clones a source buffer and returns the pointer into the arena.
static iree_status_t iree_hal_cmd_list_clone_data(iree_hal_cmd_list_t* cmd_list,
                                                  const void* source_data,
                                                  iree_host_size_t data_length,
                                                  void** out_target_data) {
  void* target_data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(&cmd_list->arena, data_length, &target_data));
  memcpy(target_data, source_data, data_length);
  *out_target_data = target_data;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_deferred_command_buffer_t implementation
//===----------------------------------------------------------------------===//

typedef struct iree_hal_deferred_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  // Maintains a reference to all resources used within the command buffer.
  // Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // All commands in encoding order.
  iree_hal_cmd_list_t cmd_list;
} iree_hal_deferred_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_deferred_command_buffer_vtable;

static iree_hal_deferred_command_buffer_t*
iree_hal_deferred_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_deferred_command_buffer_vtable);
  return (iree_hal_deferred_command_buffer_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_deferred_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_deferred_command_buffer_t* command_buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      sizeof(*command_buffer) +
          iree_hal_command_buffer_validation_state_size(mode, binding_capacity),
      (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
        &iree_hal_deferred_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    iree_hal_cmd_list_initialize(block_pool, &command_buffer->cmd_list);

    if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
      status = iree_hal_resource_set_allocate(block_pool,
                                              &command_buffer->resource_set);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_destroy(&command_buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_deferred_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cmd_list_deinitialize(&command_buffer->cmd_list);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT bool iree_hal_deferred_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_deferred_command_buffer_vtable);
}

static iree_status_t iree_hal_deferred_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  if (!iree_hal_cmd_list_is_empty(&command_buffer->cmd_list)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer cannot be re-recorded");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_resource_set_freeze(command_buffer->resource_set);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_BEGIN_DEBUG_GROUP
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_begin_debug_group_t {
  iree_hal_cmd_header_t header;
  iree_string_view_t label;
  iree_hal_label_color_t label_color;
  // NOTE: we assume iree_hal_label_location_t stays valid - not great, though.
  // It'd be better to copy but that can get expensive fast. Tracy currently
  // requires that we don't ever deallocate these locations (which sucks), so
  // we leak that requirement here.
  const iree_hal_label_location_t* location;
} iree_hal_cmd_begin_debug_group_t;

static iree_status_t iree_hal_deferred_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_cmd_list_t* cmd_list =
      &iree_hal_deferred_command_buffer_cast(base_command_buffer)->cmd_list;
  iree_hal_cmd_begin_debug_group_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_BEGIN_DEBUG_GROUP, sizeof(*cmd) + label.size,
      (void**)&cmd));
  char* label_storage = (char*)cmd + sizeof(*cmd);
  memcpy(label_storage, label.data, label.size);
  cmd->label = iree_make_string_view(label_storage, label.size);
  cmd->label_color = label_color;
  cmd->location = location;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_begin_debug_group(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_begin_debug_group_t* cmd) {
  return iree_hal_command_buffer_begin_debug_group(
      target_command_buffer, cmd->label, cmd->label_color, cmd->location);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_END_DEBUG_GROUP
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_end_debug_group_t {
  iree_hal_cmd_header_t header;
} iree_hal_cmd_end_debug_group_t;

static iree_status_t iree_hal_deferred_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cmd_list_t* cmd_list =
      &iree_hal_deferred_command_buffer_cast(base_command_buffer)->cmd_list;
  iree_hal_cmd_end_debug_group_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_END_DEBUG_GROUP, sizeof(*cmd), (void**)&cmd));
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_end_debug_group(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_end_debug_group_t* cmd) {
  return iree_hal_command_buffer_end_debug_group(target_command_buffer);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_EXECUTION_BARRIER
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_execution_barrier_t {
  iree_hal_cmd_header_t header;
  iree_hal_execution_stage_t source_stage_mask;
  iree_hal_execution_stage_t target_stage_mask;
  iree_hal_execution_barrier_flags_t flags;
  iree_host_size_t memory_barrier_count;
  const iree_hal_memory_barrier_t* memory_barriers;
  iree_host_size_t buffer_barrier_count;
  const iree_hal_buffer_barrier_t* buffer_barriers;
} iree_hal_cmd_execution_barrier_t;

static iree_status_t iree_hal_deferred_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_cmd_list_t* cmd_list =
      &iree_hal_deferred_command_buffer_cast(base_command_buffer)->cmd_list;
  iree_hal_cmd_execution_barrier_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_EXECUTION_BARRIER, sizeof(*cmd), (void**)&cmd));
  cmd->source_stage_mask = source_stage_mask;
  cmd->target_stage_mask = target_stage_mask;
  cmd->flags = flags;
  cmd->memory_barrier_count = memory_barrier_count;
  cmd->memory_barriers = NULL;
  cmd->buffer_barrier_count = buffer_barrier_count;
  cmd->buffer_barriers = NULL;
  if (memory_barrier_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_list_clone_data(
        cmd_list, memory_barriers,
        sizeof(memory_barriers[0]) * memory_barrier_count,
        (void**)&cmd->memory_barriers));
  }
  if (buffer_barrier_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_list_clone_data(
        cmd_list, buffer_barriers,
        sizeof(buffer_barriers[0]) * buffer_barrier_count,
        (void**)&cmd->buffer_barriers));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_execution_barrier(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_execution_barrier_t* cmd) {
  return iree_hal_command_buffer_execution_barrier(
      target_command_buffer, cmd->source_stage_mask, cmd->target_stage_mask,
      cmd->flags, cmd->memory_barrier_count, cmd->memory_barriers,
      cmd->buffer_barrier_count, cmd->buffer_barriers);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_SIGNAL_EVENT
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_signal_event_t {
  iree_hal_cmd_header_t header;
  iree_hal_event_t* event;
  iree_hal_execution_stage_t source_stage_mask;
} iree_hal_cmd_signal_event_t;

static iree_status_t iree_hal_deferred_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &event));
  iree_hal_cmd_signal_event_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_SIGNAL_EVENT, sizeof(*cmd), (void**)&cmd));
  cmd->event = event;
  cmd->source_stage_mask = source_stage_mask;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_signal_event(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_signal_event_t* cmd) {
  return iree_hal_command_buffer_signal_event(target_command_buffer, cmd->event,
                                              cmd->source_stage_mask);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_RESET_EVENT
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_reset_event_t {
  iree_hal_cmd_header_t header;
  iree_hal_event_t* event;
  iree_hal_execution_stage_t source_stage_mask;
} iree_hal_cmd_reset_event_t;

static iree_status_t iree_hal_deferred_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  IREE_RETURN_IF_ERROR(
      iree_hal_resource_set_insert(command_buffer->resource_set, 1, &event));
  iree_hal_cmd_reset_event_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_RESET_EVENT, sizeof(*cmd), (void**)&cmd));
  cmd->event = event;
  cmd->source_stage_mask = source_stage_mask;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_reset_event(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_reset_event_t* cmd) {
  return iree_hal_command_buffer_reset_event(target_command_buffer, cmd->event,
                                             cmd->source_stage_mask);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_WAIT_EVENTS
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_wait_events_t {
  iree_hal_cmd_header_t header;
  iree_host_size_t event_count;
  iree_hal_execution_stage_t source_stage_mask;
  iree_hal_execution_stage_t target_stage_mask;
  iree_host_size_t memory_barrier_count;
  const iree_hal_memory_barrier_t* memory_barriers;
  iree_host_size_t buffer_barrier_count;
  const iree_hal_buffer_barrier_t* buffer_barriers;
  iree_hal_event_t* events[];
} iree_hal_cmd_wait_events_t;

static iree_status_t iree_hal_deferred_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, event_count, events));
  iree_hal_cmd_wait_events_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_WAIT_EVENTS,
      sizeof(*cmd) + sizeof(cmd->events[0]) * event_count, (void**)&cmd));
  cmd->event_count = event_count;
  cmd->source_stage_mask = source_stage_mask;
  cmd->target_stage_mask = target_stage_mask;
  cmd->memory_barrier_count = memory_barrier_count;
  cmd->memory_barriers = NULL;
  cmd->buffer_barrier_count = buffer_barrier_count;
  cmd->buffer_barriers = NULL;
  memcpy(cmd->events, events, sizeof(cmd->events[0]) * event_count);
  if (memory_barrier_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_list_clone_data(
        cmd_list, memory_barriers,
        sizeof(memory_barriers[0]) * memory_barrier_count,
        (void**)&cmd->memory_barriers));
  }
  if (buffer_barrier_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_cmd_list_clone_data(
        cmd_list, buffer_barriers,
        sizeof(buffer_barriers[0]) * buffer_barrier_count,
        (void**)&cmd->buffer_barriers));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_wait_events(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_wait_events_t* cmd) {
  return iree_hal_command_buffer_wait_events(
      target_command_buffer, cmd->event_count,
      (const iree_hal_event_t**)cmd->events, cmd->source_stage_mask,
      cmd->target_stage_mask, cmd->memory_barrier_count, cmd->memory_barriers,
      cmd->buffer_barrier_count, cmd->buffer_barriers);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_ADVISE_BUFFER
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_advise_buffer_t {
  iree_hal_cmd_header_t header;
  iree_hal_buffer_ref_t buffer_ref;
  iree_hal_memory_advise_flags_t flags;
  uint64_t arg0;
  uint64_t arg1;
} iree_hal_cmd_advise_buffer_t;

static iree_status_t iree_hal_deferred_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  if (buffer_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &buffer_ref.buffer));
  }
  iree_hal_cmd_advise_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_ADVISE_BUFFER, sizeof(*cmd), (void**)&cmd));
  cmd->buffer_ref = buffer_ref;
  cmd->flags = flags;
  cmd->arg0 = arg0;
  cmd->arg1 = arg1;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_advise_buffer(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_advise_buffer_t* cmd) {
  iree_hal_buffer_ref_t buffer_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->buffer_ref, &buffer_ref));
  return iree_hal_command_buffer_advise_buffer(
      target_command_buffer, buffer_ref, cmd->flags, cmd->arg0, cmd->arg1);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_FILL_BUFFER
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_fill_buffer_t {
  iree_hal_cmd_header_t header;
  iree_hal_buffer_ref_t target_ref;
  uint64_t pattern;
  iree_host_size_t pattern_length;
  iree_hal_fill_flags_t flags;
} iree_hal_cmd_fill_buffer_t;

static iree_status_t iree_hal_deferred_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  iree_hal_cmd_fill_buffer_t* cmd = NULL;
  if (pattern_length > sizeof(cmd->pattern)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill patterns must be < 8 bytes");
  }
  if (target_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &target_ref.buffer));
  }
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_FILL_BUFFER, sizeof(*cmd), (void**)&cmd));
  cmd->target_ref = target_ref;
  memcpy(&cmd->pattern, pattern, pattern_length);
  cmd->pattern_length = pattern_length;
  cmd->flags = flags;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_fill_buffer(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_fill_buffer_t* cmd) {
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->target_ref, &target_ref));
  return iree_hal_command_buffer_fill_buffer(target_command_buffer, target_ref,
                                             (void**)&cmd->pattern,
                                             cmd->pattern_length, cmd->flags);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_UPDATE_BUFFER
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_update_buffer_t {
  iree_hal_cmd_header_t header;
  iree_hal_buffer_ref_t target_ref;
  iree_hal_update_flags_t flags;
  uint8_t source_buffer[];
} iree_hal_cmd_update_buffer_t;

static iree_status_t iree_hal_deferred_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  if (target_ref.buffer) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &target_ref.buffer));
  }
  iree_hal_cmd_update_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_UPDATE_BUFFER,
      sizeof(*cmd) + sizeof(cmd->source_buffer[0]) * target_ref.length,
      (void**)&cmd));
  cmd->target_ref = target_ref;
  cmd->flags = flags;
  memcpy(cmd->source_buffer, (const uint8_t*)source_buffer + source_offset,
         sizeof(cmd->source_buffer[0]) * target_ref.length);
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_update_buffer(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_update_buffer_t* cmd) {
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->target_ref, &target_ref));
  return iree_hal_command_buffer_update_buffer(
      target_command_buffer, cmd->source_buffer, 0, target_ref, cmd->flags);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_COPY_BUFFER
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_copy_buffer_t {
  iree_hal_cmd_header_t header;
  iree_hal_buffer_ref_t source_ref;
  iree_hal_buffer_ref_t target_ref;
  iree_hal_copy_flags_t flags;
} iree_hal_cmd_copy_buffer_t;

static iree_status_t iree_hal_deferred_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  iree_host_size_t resource_count = 0;
  const void* resources[2] = {NULL, NULL};
  if (source_ref.buffer) {
    resources[resource_count++] = source_ref.buffer;
  }
  if (target_ref.buffer) {
    resources[resource_count++] = target_ref.buffer;
  }
  if (resource_count > 0) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, resource_count, resources));
  }
  iree_hal_cmd_copy_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_COPY_BUFFER, sizeof(*cmd), (void**)&cmd));
  cmd->source_ref = source_ref;
  cmd->target_ref = target_ref;
  cmd->flags = flags;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_copy_buffer(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_copy_buffer_t* cmd) {
  iree_hal_buffer_ref_t source_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->source_ref, &source_ref));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->target_ref, &target_ref));
  return iree_hal_command_buffer_copy_buffer(target_command_buffer, source_ref,
                                             target_ref, cmd->flags);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_COLLECTIVE
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_collective_t {
  iree_hal_cmd_header_t header;
  iree_hal_channel_t* channel;
  iree_hal_collective_op_t op;
  uint32_t param;
  iree_hal_buffer_ref_t send_ref;
  iree_hal_buffer_ref_t recv_ref;
  iree_device_size_t element_count;
} iree_hal_cmd_collective_t;

static iree_status_t iree_hal_deferred_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;
  iree_host_size_t resource_count = 0;
  const void* resources[3] = {NULL, NULL, NULL};
  resources[resource_count++] = channel;
  if (send_ref.buffer) resources[resource_count++] = send_ref.buffer;
  if (recv_ref.buffer) resources[resource_count++] = recv_ref.buffer;
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, resource_count, resources));
  iree_hal_cmd_collective_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      cmd_list, IREE_HAL_CMD_COLLECTIVE, sizeof(*cmd), (void**)&cmd));
  cmd->channel = channel;
  cmd->op = op;
  cmd->param = param;
  cmd->send_ref = send_ref;
  cmd->recv_ref = recv_ref;
  cmd->element_count = element_count;
  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_collective(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_collective_t* cmd) {
  iree_hal_buffer_ref_t send_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->send_ref, &send_ref));
  iree_hal_buffer_ref_t recv_ref;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->recv_ref, &recv_ref));
  return iree_hal_command_buffer_collective(target_command_buffer, cmd->channel,
                                            cmd->op, cmd->param, send_ref,
                                            recv_ref, cmd->element_count);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_CMD_DISPATCH
//===----------------------------------------------------------------------===//

typedef struct iree_hal_cmd_dispatch_t {
  iree_hal_cmd_header_t header;
  iree_hal_executable_t* executable;
  iree_hal_executable_export_ordinal_t export_ordinal;
  iree_hal_dispatch_config_t config;
  iree_const_byte_span_t constants;
  iree_hal_buffer_ref_list_t bindings;
  iree_hal_dispatch_flags_t flags;
} iree_hal_cmd_dispatch_t;

static iree_status_t iree_hal_deferred_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);

  iree_host_size_t resource_count = 0;
  const void* resources[2] = {NULL, NULL};
  resources[resource_count++] = executable;
  if (iree_hal_dispatch_uses_indirect_parameters(flags) &&
      config.workgroup_count_ref.buffer) {
    resources[resource_count++] = config.workgroup_count_ref.buffer;
  }
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, resource_count, resources));

  iree_hal_cmd_dispatch_t* cmd = NULL;
  iree_host_size_t total_size =
      sizeof(*cmd) + iree_host_align(constants.data_length, iree_max_align_t) +
      bindings.count * sizeof(bindings.values[0]);
  IREE_RETURN_IF_ERROR(iree_hal_cmd_list_append_command(
      &command_buffer->cmd_list, IREE_HAL_CMD_DISPATCH, total_size,
      (void**)&cmd));
  cmd->executable = executable;
  cmd->export_ordinal = export_ordinal;
  memcpy(&cmd->config, &config, sizeof(cmd->config));
  cmd->flags = flags;

  uint8_t* cmd_ptr = (uint8_t*)cmd;
  cmd_ptr += sizeof(*cmd);

  memcpy(cmd_ptr, constants.data, constants.data_length);
  cmd->constants = iree_make_const_byte_span(cmd_ptr, constants.data_length);
  cmd_ptr += iree_host_align(constants.data_length, iree_max_align_t);

  cmd->bindings.count = bindings.count;
  memcpy(cmd_ptr, bindings.values, bindings.count * sizeof(bindings.values[0]));
  cmd->bindings.values = (iree_hal_buffer_ref_t*)cmd_ptr;
  cmd_ptr += bindings.count * sizeof(bindings.values[0]);
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  return iree_ok_status();
}

static iree_status_t iree_hal_deferred_command_buffer_apply_dispatch(
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_cmd_dispatch_t* cmd) {
  iree_hal_dispatch_config_t config = cmd->config;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      binding_table, cmd->config.workgroup_count_ref,
      &config.workgroup_count_ref));
  iree_hal_buffer_ref_t* binding_refs = (iree_hal_buffer_ref_t*)iree_alloca(
      cmd->bindings.count * sizeof(iree_hal_buffer_ref_t));
  for (iree_host_size_t i = 0; i < cmd->bindings.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
        binding_table, cmd->bindings.values[i], &binding_refs[i]));
  }
  const iree_hal_buffer_ref_list_t binding_ref_list = {
      .count = cmd->bindings.count,
      .values = binding_refs,
  };
  return iree_hal_command_buffer_dispatch(
      target_command_buffer, cmd->executable, cmd->export_ordinal, config,
      cmd->constants, binding_ref_list, cmd->flags);
}

//===----------------------------------------------------------------------===//
// Dynamic replay dispatch
//===----------------------------------------------------------------------===//

static const iree_hal_cmd_apply_fn_t iree_hal_cmd_apply_table[] = {
    [IREE_HAL_CMD_BEGIN_DEBUG_GROUP] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_begin_debug_group,
    [IREE_HAL_CMD_END_DEBUG_GROUP] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_end_debug_group,
    [IREE_HAL_CMD_EXECUTION_BARRIER] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_execution_barrier,
    [IREE_HAL_CMD_SIGNAL_EVENT] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_signal_event,
    [IREE_HAL_CMD_RESET_EVENT] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_reset_event,
    [IREE_HAL_CMD_WAIT_EVENTS] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_wait_events,
    [IREE_HAL_CMD_ADVISE_BUFFER] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_advise_buffer,
    [IREE_HAL_CMD_FILL_BUFFER] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_fill_buffer,
    [IREE_HAL_CMD_UPDATE_BUFFER] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_update_buffer,
    [IREE_HAL_CMD_COPY_BUFFER] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_copy_buffer,
    [IREE_HAL_CMD_COLLECTIVE] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_collective,
    [IREE_HAL_CMD_DISPATCH] = (iree_hal_cmd_apply_fn_t)
        iree_hal_deferred_command_buffer_apply_dispatch,
};

IREE_API_EXPORT iree_status_t iree_hal_deferred_command_buffer_apply(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_buffer_binding_table_t binding_table) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_deferred_command_buffer_t* command_buffer =
      iree_hal_deferred_command_buffer_cast(base_command_buffer);
  iree_hal_cmd_list_t* cmd_list = &command_buffer->cmd_list;

  iree_status_t status = iree_hal_command_buffer_begin(target_command_buffer);
  if (iree_status_is_ok(status)) {
    for (iree_hal_cmd_header_t* cmd = cmd_list->head; cmd != NULL;
         cmd = cmd->next) {
      status = iree_hal_cmd_apply_table[cmd->type](target_command_buffer,
                                                   binding_table, cmd);
      if (!iree_status_is_ok(status)) break;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_command_buffer_end(target_command_buffer);
  }

  // One-shot command buffers can't be replayed so we can drop the memory
  // immediately. As command buffers must remain live for the duration of their
  // execution this prevents us from hanging on to the commands we will never
  // use again.
  if (iree_status_is_ok(status) &&
      iree_all_bits_set(command_buffer->base.mode,
                        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    iree_hal_cmd_list_reset(cmd_list);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_deferred_command_buffer_vtable = {
        .destroy = iree_hal_deferred_command_buffer_destroy,
        .begin = iree_hal_deferred_command_buffer_begin,
        .end = iree_hal_deferred_command_buffer_end,
        .begin_debug_group = iree_hal_deferred_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_deferred_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_deferred_command_buffer_execution_barrier,
        .signal_event = iree_hal_deferred_command_buffer_signal_event,
        .reset_event = iree_hal_deferred_command_buffer_reset_event,
        .wait_events = iree_hal_deferred_command_buffer_wait_events,
        .advise_buffer = iree_hal_deferred_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_deferred_command_buffer_fill_buffer,
        .update_buffer = iree_hal_deferred_command_buffer_update_buffer,
        .copy_buffer = iree_hal_deferred_command_buffer_copy_buffer,
        .collective = iree_hal_deferred_command_buffer_collective,
        .dispatch = iree_hal_deferred_command_buffer_dispatch,
};
