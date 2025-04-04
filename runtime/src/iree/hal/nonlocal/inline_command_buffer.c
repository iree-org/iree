// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/local/inline_command_buffer.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/fpu_state.h"
#include "iree/base/internal/math.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_inline_command_buffer_t
//===----------------------------------------------------------------------===//

// Inline synchronous one-shot command "buffer".
typedef struct iree_hal_inline_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  struct {
    // Cached and initialized dispatch state reused for all dispatches.
    // Individual dispatches must populate the dynamically changing fields like
    // constant_count and binding_count.
    iree_alignas(64) iree_hal_executable_dispatch_state_v0_t dispatch_state;
    // Persistent storage for binding pointers used by dispatch_state.
    void* binding_ptr_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];
    // Persistent storage for binding lengths used by dispatch_state.
    size_t binding_length_storage[IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT];

    // An opaque tag used to reduce the cost of processor ID queries.
    iree_cpu_processor_tag_t processor_tag;
    // Guess at the current processor ID.
    iree_cpu_processor_id_t processor_id;
  } state;
} iree_hal_inline_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_inline_command_buffer_vtable;

static iree_hal_inline_command_buffer_t* iree_hal_inline_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_inline_command_buffer_vtable);
  return (iree_hal_inline_command_buffer_t*)base_value;
}

static void iree_hal_inline_command_buffer_reset(
    iree_hal_inline_command_buffer_t* command_buffer) {
  memset(&command_buffer->state, 0, sizeof(command_buffer->state));

  // Setup the cached dispatch state pointers that don't change.
  iree_hal_executable_dispatch_state_v0_t* dispatch_state =
      &command_buffer->state.dispatch_state;
  dispatch_state->binding_ptrs = command_buffer->state.binding_ptr_storage;
  dispatch_state->binding_lengths =
      command_buffer->state.binding_length_storage;
}

iree_host_size_t iree_hal_inline_command_buffer_size(
    iree_hal_command_buffer_mode_t mode, iree_host_size_t binding_capacity) {
  return sizeof(iree_hal_inline_command_buffer_t) +
         iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
}

iree_status_t iree_hal_inline_command_buffer_initialize(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator, iree_byte_span_t storage,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (!iree_all_bits_set(
          mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
                    IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION)) {
    // This implementation only supports command buffers that are allowed to
    // execute inline. This mode is a contract with the caller that it is ok if
    // we begin executing prior to submission.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "inline command buffers must have a mode with ALLOW_INLINE_EXECUTION");
  }
  if (binding_capacity > 0) {
    // We execute as we record and can't use binding tables to do that.
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "indirect command buffers do not support binding tables");
  }
  if (storage.data_length <
      iree_hal_inline_command_buffer_size(mode, binding_capacity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "storage must have at least the capacity as "
                            "defined by iree_hal_inline_command_buffer_size");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_inline_command_buffer_t* command_buffer =
      (iree_hal_inline_command_buffer_t*)storage.data;
  memset(command_buffer, 0, sizeof(*command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_inline_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  iree_hal_inline_command_buffer_reset(command_buffer);

  *out_command_buffer = &command_buffer->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_inline_command_buffer_deinitialize(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);
  iree_hal_inline_command_buffer_reset(command_buffer);
}

iree_status_t iree_hal_inline_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  uint8_t* storage = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator,
      iree_hal_inline_command_buffer_size(mode, binding_capacity),
      (void**)&storage);
  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_inline_command_buffer_initialize(
        device_allocator, mode, command_categories, queue_affinity,
        binding_capacity, host_allocator,
        iree_make_byte_span(storage, iree_hal_inline_command_buffer_size(
                                         mode, binding_capacity)),
        &command_buffer);
  }

  if (iree_status_is_ok(status)) {
    *out_command_buffer = command_buffer;
  } else {
    iree_allocator_free(host_allocator, storage);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_inline_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_inline_command_buffer_deinitialize(base_command_buffer);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_inline_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_inline_command_buffer_vtable);
}

//===----------------------------------------------------------------------===//
// iree_hal_inline_command_buffer_t recording
//===----------------------------------------------------------------------===//

// Updates the cached processor ID field in the command buffer.
static void iree_hal_inline_command_buffer_update_processor_id(
    iree_hal_inline_command_buffer_t* command_buffer) {
  iree_cpu_requery_processor_id(&command_buffer->state.processor_tag,
                                &command_buffer->state.processor_id);
}

static iree_status_t iree_hal_inline_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);
  iree_hal_inline_command_buffer_reset(command_buffer);

  // Query the processor ID we start out on. We may update it during execution.
  iree_hal_inline_command_buffer_update_processor_id(command_buffer);

  return iree_ok_status();
}

static iree_status_t iree_hal_inline_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);
  iree_hal_inline_command_buffer_reset(command_buffer);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_inline_command_buffer_t debug utilities
//===----------------------------------------------------------------------===//

static void iree_hal_inline_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO(benvanik): tracy event stack.
}

static void iree_hal_inline_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO(benvanik): tracy event stack.
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_execution_barrier
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_signal_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_reset_event
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_wait_events
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // No-op; we execute synchronously.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_discard_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  // Could be treated as a cache invalidation as it indicates we won't be using
  // the existing buffer contents again.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length) {
  return iree_hal_buffer_map_fill(target_ref.buffer, target_ref.offset,
                                  target_ref.length, pattern, pattern_length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref) {
  return iree_hal_buffer_map_write(
      target_ref.buffer, target_ref.offset,
      (const uint8_t*)source_buffer + source_offset, target_ref.length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  return iree_hal_buffer_map_copy(source_ref.buffer, source_ref.offset,
                                  target_ref.buffer, target_ref.offset,
                                  target_ref.length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_collective
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on CPU");
}

static iree_status_t iree_hal_inline_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const uint32_t workgroup_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);

  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);

  iree_hal_executable_dispatch_attrs_v0_t dispatch_attrs = {0};
  if (local_executable->dispatch_attrs) {
    dispatch_attrs = local_executable->dispatch_attrs[entry_point];
  }
  const iree_host_size_t local_memory_size =
      dispatch_attrs.local_memory_pages *
      IREE_HAL_EXECUTABLE_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE;

  // Update the ID of the processor we are running on.
  // We don't know how much time has passed since we last updated as we are
  // running inline with the user program; if we knew we were going to be
  // handling a batch of dispatches we could reduce the amount of times we call
  // this - but that's what the task system is for.
  iree_hal_inline_command_buffer_update_processor_id(command_buffer);

  iree_hal_executable_dispatch_state_v0_t* dispatch_state =
      &command_buffer->state.dispatch_state;

  // TODO(benvanik): expose on API or keep fixed on executable.
  dispatch_state->workgroup_size_x = 1;
  dispatch_state->workgroup_size_y = 1;
  dispatch_state->workgroup_size_z = 1;
  dispatch_state->workgroup_count_x = workgroup_count[0];
  dispatch_state->workgroup_count_y = workgroup_count[1];
  dispatch_state->workgroup_count_z = workgroup_count[2];

  // Single-threaded.
  dispatch_state->max_concurrency = 1;

  // Push constants are pulled directly from the args. Note that we require 4
  // byte alignment and if the input buffer is not aligned we have to fail.
  if (IREE_UNLIKELY((constants.data_length % sizeof(uint32_t)) != 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "constants must be 4-byte aligned");
  } else if (IREE_UNLIKELY(constants.data_length !=
                           dispatch_attrs.constant_count * sizeof(uint32_t))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "constant count mismatch, expected %u but was provided %" PRIhsz,
        (uint32_t)dispatch_attrs.constant_count,
        constants.data_length / sizeof(uint32_t));
  }
  dispatch_state->constant_count = dispatch_attrs.constant_count;
  dispatch_state->constants = (const uint32_t*)constants.data;

  // Produce the dense binding list based on the declared bindings used.
  //
  // Note that we are just directly setting the binding data pointers here with
  // no ownership/retaining/etc - it's part of the HAL contract that buffers are
  // kept valid for the duration they may be in use.
  if (IREE_UNLIKELY(bindings.count != dispatch_attrs.binding_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "binding count mismatch, expected %u but was provided %" PRIhsz,
        (uint32_t)dispatch_attrs.binding_count, bindings.count);
  }
  dispatch_state->binding_count = bindings.count;
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
    iree_hal_buffer_mapping_t buffer_mapping = {{0}};
    if (IREE_LIKELY(bindings.values[i].buffer)) {
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          bindings.values[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_ANY, bindings.values[i].offset,
          bindings.values[i].length, &buffer_mapping));
    } else {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "required binding %" PRIhsz
          " is NULL; all bindings must have a valid pointer",
          i);
    }
    command_buffer->state.binding_ptr_storage[i] = buffer_mapping.contents.data;
    command_buffer->state.binding_length_storage[i] =
        buffer_mapping.contents.data_length;
  }

  // TODO(benvanik): plumb through an arena or fixed-size reservation to use.
  // For now when deploying to devices where you want something like the
  // inline command buffer you probably don't want 256KB of transient memory
  // getting allocated and retained implicitly - this should be a compiler
  // option. For now we just malloc here to make things work and strongly
  // encourage the kind of user who wants synchronous inline execution to not
  // also want tons of scratch memory.
  iree_byte_span_t local_memory = iree_make_byte_span(NULL, local_memory_size);
  if (local_memory_size > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(command_buffer->host_allocator,
                                               local_memory_size,
                                               (void**)&local_memory.data));
  }

  // Since we are running on a borrowed thread, we know nothing about the
  // floating point state. Reset it.
  iree_fpu_state_t fpu_state =
      iree_fpu_state_push(IREE_FPU_STATE_FLAG_FLUSH_DENORMALS_TO_ZERO);
  iree_status_t status = iree_hal_local_executable_issue_dispatch_inline(
      local_executable, entry_point, dispatch_state,
      command_buffer->state.processor_id, local_memory);
  iree_fpu_state_pop(fpu_state);

  if (local_memory.data) {
    iree_allocator_free(command_buffer->host_allocator, local_memory.data);
  }
  return status;
}

typedef union iree_hal_vec3_t {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t value[3];
} iree_hal_vec3_t;

static iree_status_t iree_hal_inline_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
  iree_hal_buffer_mapping_t buffer_mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      workgroups_ref.buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ, workgroups_ref.offset, 3 * sizeof(uint32_t),
      &buffer_mapping));
  iree_hal_vec3_t workgroup_count =
      *(const iree_hal_vec3_t*)buffer_mapping.contents.data;
  return iree_hal_inline_command_buffer_dispatch(
      base_command_buffer, executable, entry_point, workgroup_count.value,
      constants, bindings, flags);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_vtable_t
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_inline_command_buffer_vtable = {
        .destroy = iree_hal_inline_command_buffer_destroy,
        .begin = iree_hal_inline_command_buffer_begin,
        .end = iree_hal_inline_command_buffer_end,
        .begin_debug_group = iree_hal_inline_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_inline_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_inline_command_buffer_execution_barrier,
        .signal_event = iree_hal_inline_command_buffer_signal_event,
        .reset_event = iree_hal_inline_command_buffer_reset_event,
        .wait_events = iree_hal_inline_command_buffer_wait_events,
        .discard_buffer = iree_hal_inline_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_inline_command_buffer_fill_buffer,
        .update_buffer = iree_hal_inline_command_buffer_update_buffer,
        .copy_buffer = iree_hal_inline_command_buffer_copy_buffer,
        .collective = iree_hal_inline_command_buffer_collective,
        .dispatch = iree_hal_inline_command_buffer_dispatch,
        .dispatch_indirect = iree_hal_inline_command_buffer_dispatch_indirect,
};
