// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/local_task/block_command_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/local_task/block_builder.h"
#include "iree/hal/drivers/local_task/block_command_ops.h"
#include "iree/hal/drivers/local_task/block_isa.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/transient_buffer.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_block_command_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_block_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  iree_arena_block_pool_t* block_pool;

  // Retains resources (buffers, executables) used during recording.
  iree_hal_resource_set_t* resource_set;

  // Block builder compiling HAL commands into block ISA.
  iree_hal_cmd_block_builder_t builder;

  // Recording produced by end(). Consumed by the queue or released on destroy.
  iree_hal_cmd_block_recording_t recording;

  // Direct transient bindings that must be mapped after queue waits resolve.
  iree_hal_buffer_binding_t* late_bindings;

  // Number of entries in late_bindings.
  iree_host_size_t late_binding_count;

  // Total allocated capacity of late_bindings.
  iree_host_size_t late_binding_capacity;

  // Profiling sideband metadata collected during command recording.
  struct {
    // Distinct executable metadata required by recorded dispatch commands.
    struct {
      // First distinct executable referenced by dispatch commands.
      iree_hal_executable_t* first_executable;

      // Profile id for |first_executable|, or 0 when absent.
      uint64_t first_profile_id;

      // Heap list used when more than one distinct executable is referenced.
      // When allocated, index 0 aliases |first_executable|.
      iree_hal_executable_t** list;

      // Open-addressed set of executable profile IDs in |list|.
      uint64_t* profile_id_set;

      // Number of distinct executables, including |first_executable|.
      iree_host_size_t count;

      // Power-of-two slot count in |profile_id_set| and allocated length of
      // |list|.
      iree_host_size_t capacity;
    } executables;

    // Command-buffer operation metadata required to explain replay timing.
    struct {
      // Dense command-buffer-global operation records in recording order.
      iree_hal_profile_command_operation_record_t* records;

      // Number of valid entries in |records|.
      iree_host_size_t count;

      // Allocated entry capacity of |records|.
      iree_host_size_t capacity;
    } operations;
  } profile;
} iree_hal_block_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_block_command_buffer_vtable;

static iree_hal_block_command_buffer_t* iree_hal_block_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_block_command_buffer_vtable);
  return (iree_hal_block_command_buffer_t*)base_value;
}

//===----------------------------------------------------------------------===//
// Profiling helpers
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_hal_block_command_buffer_profile_hash_id(
    uint64_t executable_id, iree_host_size_t capacity) {
  executable_id ^= executable_id >> 33;
  executable_id *= 0xff51afd7ed558ccdull;
  executable_id ^= executable_id >> 33;
  executable_id *= 0xc4ceb9fe1a85ec53ull;
  executable_id ^= executable_id >> 33;
  return (iree_host_size_t)executable_id & (capacity - 1);
}

static bool iree_hal_block_command_buffer_profile_find_id_slot(
    const uint64_t* profile_id_set, iree_host_size_t capacity,
    uint64_t executable_id, iree_host_size_t* out_slot) {
  iree_host_size_t slot =
      iree_hal_block_command_buffer_profile_hash_id(executable_id, capacity);
  while (profile_id_set[slot] != 0) {
    if (profile_id_set[slot] == executable_id) {
      *out_slot = slot;
      return true;
    }
    slot = (slot + 1) & (capacity - 1);
  }
  *out_slot = slot;
  return false;
}

static iree_status_t iree_hal_block_command_buffer_profile_reserve_executables(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_host_size_t minimum_count) {
  if (minimum_count <= 1) return iree_ok_status();
  if (IREE_UNLIKELY(minimum_count > IREE_HOST_SIZE_MAX / 2)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile executable set for block command buffer is too large");
  }
  iree_host_size_t required_capacity = minimum_count * 2;
  required_capacity = iree_max((iree_host_size_t)16, required_capacity);
  required_capacity = iree_host_size_next_power_of_two(required_capacity);
  if (IREE_UNLIKELY(!iree_host_size_is_power_of_two(required_capacity))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile executable set for block command buffer is too large");
  }
  if (required_capacity <= command_buffer->profile.executables.capacity) {
    return iree_ok_status();
  }

  iree_host_size_t profile_id_set_byte_length = 0;
  iree_host_size_t list_byte_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          required_capacity,
          sizeof(*command_buffer->profile.executables.profile_id_set),
          &profile_id_set_byte_length)) ||
      IREE_UNLIKELY(!iree_host_size_checked_mul(
          required_capacity, sizeof(*command_buffer->profile.executables.list),
          &list_byte_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile executable set for block command buffer is too large");
  }

  uint64_t* new_profile_id_set = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(command_buffer->host_allocator,
                                             profile_id_set_byte_length,
                                             (void**)&new_profile_id_set));
  memset(new_profile_id_set, 0, profile_id_set_byte_length);

  iree_hal_executable_t** new_list = NULL;
  iree_status_t status = iree_allocator_malloc(
      command_buffer->host_allocator, list_byte_length, (void**)&new_list);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(command_buffer->host_allocator, new_profile_id_set);
    return status;
  }
  memset(new_list, 0, list_byte_length);

  for (iree_host_size_t i = 0; i < command_buffer->profile.executables.count;
       ++i) {
    iree_hal_executable_t* executable =
        i == 0 ? command_buffer->profile.executables.first_executable
               : command_buffer->profile.executables.list[i];
    const uint64_t executable_id =
        i == 0 ? command_buffer->profile.executables.first_profile_id
               : iree_hal_local_executable_profile_id(
                     iree_hal_local_executable_cast(executable));
    iree_host_size_t slot = 0;
    iree_hal_block_command_buffer_profile_find_id_slot(
        new_profile_id_set, required_capacity, executable_id, &slot);
    new_profile_id_set[slot] = executable_id;
    new_list[i] = executable;
  }

  iree_allocator_free(command_buffer->host_allocator,
                      command_buffer->profile.executables.profile_id_set);
  iree_allocator_free(command_buffer->host_allocator,
                      command_buffer->profile.executables.list);
  command_buffer->profile.executables.profile_id_set = new_profile_id_set;
  command_buffer->profile.executables.list = new_list;
  command_buffer->profile.executables.capacity = required_capacity;
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_profile_track_executable(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable) {
  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);
  const uint64_t executable_id =
      iree_hal_local_executable_profile_id(local_executable);
  if (command_buffer->profile.executables.count == 0) {
    command_buffer->profile.executables.first_executable = executable;
    command_buffer->profile.executables.first_profile_id = executable_id;
    command_buffer->profile.executables.count = 1;
    return iree_ok_status();
  }
  if (executable_id == command_buffer->profile.executables.first_profile_id) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_hal_block_command_buffer_profile_reserve_executables(
          command_buffer, command_buffer->profile.executables.count + 1));

  iree_host_size_t slot = 0;
  if (iree_hal_block_command_buffer_profile_find_id_slot(
          command_buffer->profile.executables.profile_id_set,
          command_buffer->profile.executables.capacity, executable_id, &slot)) {
    return iree_ok_status();
  }

  command_buffer->profile.executables.profile_id_set[slot] = executable_id;
  iree_host_size_t index = command_buffer->profile.executables.count++;
  command_buffer->profile.executables.list[index] = executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_profile_reserve_operations(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_host_size_t minimum_count) {
  if (minimum_count <= command_buffer->profile.operations.capacity) {
    return iree_ok_status();
  }
  if (IREE_UNLIKELY(minimum_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profile command-buffer operation count exceeds uint32_t");
  }
  return iree_allocator_grow_array(
      command_buffer->host_allocator,
      iree_max((iree_host_size_t)64, minimum_count),
      sizeof(command_buffer->profile.operations.records[0]),
      &command_buffer->profile.operations.capacity,
      (void**)&command_buffer->profile.operations.records);
}

static iree_hal_profile_command_operation_record_t*
iree_hal_block_command_buffer_profile_append_operation(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_profile_command_operation_type_t type) {
  const uint32_t command_index =
      (uint32_t)command_buffer->profile.operations.count;
  iree_hal_profile_command_operation_record_t* record =
      &command_buffer->profile.operations
           .records[command_buffer->profile.operations.count++];
  *record = iree_hal_profile_command_operation_record_default();
  record->type = type;
  record->command_index = command_index;
  record->command_buffer_id =
      iree_hal_command_buffer_profile_id(&command_buffer->base);
  return record;
}

static iree_hal_profile_command_operation_flags_t
iree_hal_block_command_buffer_profile_binding_flags(
    iree_host_size_t ref_count, const iree_hal_buffer_ref_t* refs) {
  iree_hal_profile_command_operation_flags_t flags =
      IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_NONE;
  for (iree_host_size_t i = 0; i < ref_count; ++i) {
    flags |= refs[i].buffer
                 ? IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_STATIC_BINDINGS
                 : IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_DYNAMIC_BINDINGS;
  }
  return flags;
}

static uint32_t iree_hal_block_command_buffer_profile_buffer_ordinal(
    iree_hal_buffer_ref_t ref) {
  return ref.buffer ? UINT32_MAX : ref.buffer_slot;
}

static void iree_hal_block_command_buffer_profile_append_barrier(
    iree_hal_block_command_buffer_t* command_buffer) {
  iree_hal_profile_command_operation_record_t* record =
      iree_hal_block_command_buffer_profile_append_operation(
          command_buffer, IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_BARRIER);
  record->flags |= IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_EXECUTION_BARRIER;
}

static void iree_hal_block_command_buffer_profile_append_fill(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t target_ref) {
  iree_hal_profile_command_operation_record_t* record =
      iree_hal_block_command_buffer_profile_append_operation(
          command_buffer, IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_FILL);
  record->flags |=
      iree_hal_block_command_buffer_profile_binding_flags(1, &target_ref);
  record->target_offset = target_ref.offset;
  record->length = target_ref.length;
  record->target_ordinal =
      iree_hal_block_command_buffer_profile_buffer_ordinal(target_ref);
}

static void iree_hal_block_command_buffer_profile_append_update(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t target_ref) {
  iree_hal_profile_command_operation_record_t* record =
      iree_hal_block_command_buffer_profile_append_operation(
          command_buffer, IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_UPDATE);
  record->flags |=
      iree_hal_block_command_buffer_profile_binding_flags(1, &target_ref);
  record->target_offset = target_ref.offset;
  record->length = target_ref.length;
  record->target_ordinal =
      iree_hal_block_command_buffer_profile_buffer_ordinal(target_ref);
}

static void iree_hal_block_command_buffer_profile_append_copy(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  const iree_hal_buffer_ref_t refs[2] = {source_ref, target_ref};
  iree_hal_profile_command_operation_record_t* record =
      iree_hal_block_command_buffer_profile_append_operation(
          command_buffer, IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_COPY);
  record->flags |= iree_hal_block_command_buffer_profile_binding_flags(
      IREE_ARRAYSIZE(refs), refs);
  record->source_offset = source_ref.offset;
  record->target_offset = target_ref.offset;
  record->length = target_ref.length;
  record->source_ordinal =
      iree_hal_block_command_buffer_profile_buffer_ordinal(source_ref);
  record->target_ordinal =
      iree_hal_block_command_buffer_profile_buffer_ordinal(target_ref);
}

static void iree_hal_block_command_buffer_profile_append_dispatch(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_cmd_dispatch_t* dispatch, iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_profile_command_operation_record_t* record =
      iree_hal_block_command_buffer_profile_append_operation(
          command_buffer, IREE_HAL_PROFILE_COMMAND_OPERATION_TYPE_DISPATCH);
  dispatch->profile.command_index = record->command_index;
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    record->flags |=
        IREE_HAL_PROFILE_COMMAND_OPERATION_FLAG_INDIRECT_PARAMETERS;
  } else {
    memcpy(record->workgroup_count, config.workgroup_count,
           sizeof(record->workgroup_count));
  }
  record->flags |= iree_hal_block_command_buffer_profile_binding_flags(
      bindings.count, bindings.values);
  record->executable_id = iree_hal_local_executable_profile_id(
      iree_hal_local_executable_cast(executable));
  record->export_ordinal = export_ordinal;
  record->binding_count =
      (uint32_t)iree_min(bindings.count, (iree_host_size_t)UINT32_MAX);
  record->workgroup_size[0] =
      config.workgroup_size[0] ? config.workgroup_size[0] : 1;
  record->workgroup_size[1] =
      config.workgroup_size[1] ? config.workgroup_size[1] : 1;
  record->workgroup_size[2] =
      config.workgroup_size[2] ? config.workgroup_size[2] : 1;
}

//===----------------------------------------------------------------------===//
// Binding helpers
//===----------------------------------------------------------------------===//

// Returns true when |buffer| must be mapped at queue-execute time instead of
// command-buffer record time. Transient buffers are queue-owned memory with a
// dealloca operation, so their mappings must be scoped to queue execution even
// if the backing is already committed while the command buffer records.
static bool iree_hal_block_command_buffer_needs_late_binding(
    iree_hal_buffer_t* buffer) {
  return iree_hal_local_transient_buffer_isa(buffer);
}

// Finds or appends a late direct binding for a transient buffer. Late slots are
// placed after the command buffer's external binding capacity so ordinary
// indirect binding table slots keep their original indices.
static iree_status_t iree_hal_block_command_buffer_get_late_binding_slot(
    iree_hal_block_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t buffer_ref, uint16_t* out_slot) {
  for (iree_host_size_t i = 0; i < command_buffer->late_binding_count; ++i) {
    const iree_hal_buffer_binding_t* binding =
        &command_buffer->late_bindings[i];
    if (binding->buffer == buffer_ref.buffer &&
        binding->offset == buffer_ref.offset &&
        binding->length == buffer_ref.length) {
      const uint64_t slot = (uint64_t)command_buffer->base.binding_capacity + i;
      if (IREE_UNLIKELY(slot > UINT16_MAX)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "late binding slot %" PRIu64 " exceeds the block ISA limit", slot);
      }
      *out_slot = (uint16_t)slot;
      return iree_ok_status();
    }
  }

  if (command_buffer->late_binding_count ==
      command_buffer->late_binding_capacity) {
    const iree_host_size_t new_capacity = iree_max(
        (iree_host_size_t)4, command_buffer->late_binding_capacity * 2);
    iree_hal_buffer_binding_t* new_bindings = command_buffer->late_bindings;
    IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
        command_buffer->host_allocator, new_capacity, sizeof(*new_bindings),
        (void**)&new_bindings));
    command_buffer->late_bindings = new_bindings;
    command_buffer->late_binding_capacity = new_capacity;
  }

  const uint64_t slot = (uint64_t)command_buffer->base.binding_capacity +
                        command_buffer->late_binding_count;
  if (IREE_UNLIKELY(slot > UINT16_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "late binding slot %" PRIu64 " exceeds the block ISA limit", slot);
  }
  command_buffer->late_bindings[command_buffer->late_binding_count++] =
      (iree_hal_buffer_binding_t){
          .buffer = buffer_ref.buffer,
          .offset = buffer_ref.offset,
          .length = buffer_ref.length,
      };
  *out_slot = (uint16_t)slot;
  return iree_ok_status();
}

const iree_hal_buffer_binding_t* iree_hal_block_command_buffer_late_bindings(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t* out_late_binding_count) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  *out_late_binding_count = command_buffer->late_binding_count;
  return command_buffer->late_bindings;
}

// Resolves |count| buffer references into fixup entries. For each ref:
//   - Direct ready non-transient buffer: maps the buffer persistently and
//     stores the host pointer inline in the fixup.
//   - Direct transient buffer: records a late binding slot that the queue maps
//     after waits resolve.
//   - Indirect buffer: records the binding table slot and offset for runtime
//     resolution by the processor.
//
// Direct non-transient buffers use PERSISTENT mapping: the buffer is retained
// by the resource_set for the CB's lifetime, so the pointer is stable.
// Queue-owned transients are retained but mapped only during queue execution so
// dealloca can be ordered against the scoped mapping lifetime.
//
// The fixup data_index fields are pre-filled by the builder and preserved.
static iree_status_t iree_hal_block_command_buffer_resolve_refs(
    iree_hal_block_command_buffer_t* command_buffer, iree_host_size_t count,
    const iree_hal_buffer_ref_t* buffer_refs, iree_hal_cmd_fixup_t* fixups) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    // data_index is pre-filled by the builder; write only the other fields.
    fixups[i].flags = IREE_HAL_CMD_FIXUP_FLAG_NONE;
    if (buffer_refs[i].buffer) {
      if (iree_hal_block_command_buffer_needs_late_binding(
              buffer_refs[i].buffer)) {
        uint16_t slot = 0;
        IREE_RETURN_IF_ERROR(
            iree_hal_block_command_buffer_get_late_binding_slot(
                command_buffer, buffer_refs[i], &slot));
        fixups[i].host_ptr = NULL;
        fixups[i].offset = 0;
        fixups[i].length = buffer_refs[i].length;
        fixups[i].slot = slot;
        continue;
      }

      // Direct: map the buffer now.
      iree_hal_buffer_mapping_t mapping = {{0}};
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          buffer_refs[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_ANY, buffer_refs[i].offset,
          buffer_refs[i].length, &mapping));
      fixups[i].host_ptr = mapping.contents.data;
      fixups[i].offset = 0;  // map_range already applied the offset.
      fixups[i].length = mapping.contents.data_length;
      fixups[i].slot = 0;
    } else {
      // Indirect: record binding table slot for runtime resolution.
      fixups[i].host_ptr = NULL;
      fixups[i].offset = buffer_refs[i].offset;
      fixups[i].length = buffer_refs[i].length;
      fixups[i].slot = (uint16_t)buffer_refs[i].buffer_slot;
    }
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_block_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer);

iree_status_t iree_hal_block_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (iree_any_bit_set(mode,
                       IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION) &&
      !iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ALLOW_INLINE_EXECUTION requires ONE_SHOT mode");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t total_size = 0;
  iree_host_size_t validation_state_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_hal_block_command_buffer_t), &total_size,
              IREE_STRUCT_FIELD(iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                                uint8_t, &validation_state_offset)));

  iree_hal_block_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&command_buffer));
  memset(command_buffer, 0, sizeof(*command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + validation_state_offset,
      &iree_hal_block_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->block_pool = block_pool;
  iree_hal_cmd_block_builder_initialize(block_pool, &command_buffer->builder);
  memset(&command_buffer->recording, 0, sizeof(command_buffer->recording));
  command_buffer->late_bindings = NULL;
  command_buffer->late_binding_count = 0;
  command_buffer->late_binding_capacity = 0;

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_block_command_buffer_destroy(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_block_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cmd_block_recording_release(&command_buffer->recording);
  iree_allocator_free(host_allocator, command_buffer->late_bindings);
  iree_allocator_free(host_allocator,
                      command_buffer->profile.executables.profile_id_set);
  iree_allocator_free(host_allocator, command_buffer->profile.executables.list);
  iree_allocator_free(host_allocator,
                      command_buffer->profile.operations.records);
  iree_hal_cmd_block_builder_deinitialize(&command_buffer->builder);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_block_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_block_command_buffer_vtable);
}

const iree_hal_cmd_block_recording_t* iree_hal_block_command_buffer_recording(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return &command_buffer->recording;
}

iree_hal_executable_t* const* iree_hal_block_command_buffer_profile_executables(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t* out_executable_count) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  *out_executable_count = command_buffer->profile.executables.count;
  return command_buffer->profile.executables.count == 1
             ? &command_buffer->profile.executables.first_executable
             : command_buffer->profile.executables.list;
}

iree_host_size_t iree_hal_block_command_buffer_profile_operation_count(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return command_buffer->profile.operations.count;
}

void iree_hal_block_command_buffer_profile_metadata(
    iree_hal_command_buffer_t* base_command_buffer,
    uint32_t physical_device_ordinal,
    iree_hal_profile_command_buffer_record_t* out_command_buffer,
    const iree_hal_profile_command_operation_record_t** out_operations,
    iree_host_size_t* out_operation_count) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  iree_hal_profile_command_buffer_record_t record =
      iree_hal_profile_command_buffer_record_default();
  record.command_buffer_id =
      iree_hal_command_buffer_profile_id(base_command_buffer);
  record.mode = iree_hal_command_buffer_mode(base_command_buffer);
  record.command_categories =
      iree_hal_command_buffer_allowed_categories(base_command_buffer);
  record.queue_affinity =
      iree_hal_command_buffer_queue_affinity(base_command_buffer);
  record.physical_device_ordinal = physical_device_ordinal;

  *out_command_buffer = record;
  *out_operations = command_buffer->profile.operations.records;
  *out_operation_count = command_buffer->profile.operations.count;
}

//===----------------------------------------------------------------------===//
// Recording session
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return iree_hal_cmd_block_builder_begin(&command_buffer->builder);
}

static iree_status_t iree_hal_block_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  return iree_hal_cmd_block_builder_end(&command_buffer->builder,
                                        &command_buffer->recording);
}

//===----------------------------------------------------------------------===//
// Debug groups (no-op)
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Barriers and events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  // Block ISA barriers are global: all prior work in the region must complete
  // before the next region begins. Fine-grained memory/buffer barriers are
  // not applicable (CPU execution is cache-coherent).
  IREE_RETURN_IF_ERROR(iree_hal_block_command_buffer_profile_reserve_operations(
      command_buffer, command_buffer->profile.operations.count + 1));
  IREE_RETURN_IF_ERROR(
      iree_hal_cmd_block_builder_barrier(&command_buffer->builder));
  iree_hal_block_command_buffer_profile_append_barrier(command_buffer);
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_ok_status();
}

static iree_status_t iree_hal_block_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // Treat event waits as global barriers (same as the task CB).
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);
  IREE_RETURN_IF_ERROR(iree_hal_block_command_buffer_profile_reserve_operations(
      command_buffer, command_buffer->profile.operations.count + 1));
  IREE_RETURN_IF_ERROR(
      iree_hal_cmd_block_builder_barrier(&command_buffer->builder));
  iree_hal_block_command_buffer_profile_append_barrier(command_buffer);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Buffer advise
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, 1, &target_ref,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_block_command_buffer_profile_reserve_operations(
      command_buffer, command_buffer->profile.operations.count + 1));
  IREE_RETURN_IF_ERROR(
      iree_hal_cmd_build_fill(&command_buffer->builder, target_ref.length,
                              pattern, pattern_length, &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, 1, &target_ref, fixups);
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  } else {
    iree_hal_block_command_buffer_profile_append_fill(command_buffer,
                                                      target_ref);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, 1, &target_ref,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_block_command_buffer_profile_reserve_operations(
      command_buffer, command_buffer->profile.operations.count + 1));
  IREE_RETURN_IF_ERROR(iree_hal_cmd_build_update(
      &command_buffer->builder, source_buffer, source_offset, target_ref.length,
      &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, 1, &target_ref, fixups);
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  } else {
    iree_hal_block_command_buffer_profile_append_update(command_buffer,
                                                        target_ref);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  // Retain direct buffer references. Indirect refs (buffer == NULL) are
  // skipped by the strided insert and resolved from the binding table at
  // submit time.
  const iree_hal_buffer_ref_t refs[2] = {source_ref, target_ref};
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, IREE_ARRAYSIZE(refs), refs,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_block_command_buffer_profile_reserve_operations(
      command_buffer, command_buffer->profile.operations.count + 1));
  IREE_RETURN_IF_ERROR(iree_hal_cmd_build_copy(
      &command_buffer->builder, target_ref.length, &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, IREE_ARRAYSIZE(refs), refs, fixups);
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  } else {
    iree_hal_block_command_buffer_profile_append_copy(command_buffer,
                                                      source_ref, target_ref);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_collective
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on the block ISA");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_block_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_block_command_buffer_t* command_buffer =
      iree_hal_block_command_buffer_cast(base_command_buffer);

  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
      command_buffer->resource_set, 1, &executable));
  IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert_strided(
      command_buffer->resource_set, bindings.count, bindings.values,
      offsetof(iree_hal_buffer_ref_t, buffer), sizeof(iree_hal_buffer_ref_t)));
  if (iree_hal_dispatch_uses_indirect_parameters(flags)) {
    IREE_RETURN_IF_ERROR(iree_hal_resource_set_insert(
        command_buffer->resource_set, 1, &config.workgroup_count_ref.buffer));
  }

  iree_hal_cmd_fixup_t* fixups = NULL;
  iree_hal_cmd_build_token_t token;
  IREE_RETURN_IF_ERROR(iree_hal_block_command_buffer_profile_reserve_operations(
      command_buffer, command_buffer->profile.operations.count + 1));
  IREE_RETURN_IF_ERROR(iree_hal_cmd_build_dispatch(
      &command_buffer->builder, executable, export_ordinal, config, constants,
      bindings.count, flags, &fixups, &token));

  iree_status_t status = iree_hal_block_command_buffer_resolve_refs(
      command_buffer, bindings.count, bindings.values, fixups);
  if (iree_status_is_ok(status) &&
      iree_hal_dispatch_uses_indirect_parameters(flags)) {
    status = iree_hal_block_command_buffer_resolve_refs(
        command_buffer, 1, &config.workgroup_count_ref,
        &fixups[bindings.count]);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_block_command_buffer_profile_track_executable(
        command_buffer, executable);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_cmd_build_rollback(&command_buffer->builder, token);
  } else {
    iree_hal_block_command_buffer_profile_append_dispatch(
        command_buffer, (iree_hal_cmd_dispatch_t*)token.command, executable,
        export_ordinal, config, bindings, flags);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_block_command_buffer_vtable = {
        .destroy = iree_hal_block_command_buffer_destroy,
        .begin = iree_hal_block_command_buffer_begin,
        .end = iree_hal_block_command_buffer_end,
        .begin_debug_group = iree_hal_block_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_block_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_block_command_buffer_execution_barrier,
        .signal_event = iree_hal_block_command_buffer_signal_event,
        .reset_event = iree_hal_block_command_buffer_reset_event,
        .wait_events = iree_hal_block_command_buffer_wait_events,
        .advise_buffer = iree_hal_block_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_block_command_buffer_fill_buffer,
        .update_buffer = iree_hal_block_command_buffer_update_buffer,
        .copy_buffer = iree_hal_block_command_buffer_copy_buffer,
        .collective = iree_hal_block_command_buffer_collective,
        .dispatch = iree_hal_block_command_buffer_dispatch,
};
