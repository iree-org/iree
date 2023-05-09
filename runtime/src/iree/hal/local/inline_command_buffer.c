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
#include "iree/base/tracing.h"
#include "iree/hal/local/executable_library.h"
#include "iree/hal/local/local_executable.h"
#include "iree/hal/local/local_pipeline_layout.h"

//===----------------------------------------------------------------------===//
// iree_hal_inline_command_buffer_t
//===----------------------------------------------------------------------===//

// Inline synchronous one-shot command "buffer".
typedef struct iree_hal_inline_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  struct {
    // A flattened list of all available descriptor set bindings.
    // As descriptor sets are pushed/bound the bindings will be updated to
    // represent the fully-translated binding data pointer.
    //
    // TODO(benvanik): support proper mapping semantics and track the
    // iree_hal_buffer_mapping_t and map/unmap where appropriate.
    void* full_bindings[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                        IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];
    size_t full_binding_lengths[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                                IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];

    // Packed bindings scratch space used during dispatch. Executable bindings
    // are packed into a dense list with unused bindings removed.
    void* packed_bindings[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                          IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];
    size_t packed_binding_lengths[IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT *
                                  IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT];

    // All available push constants updated each time push_constants is called.
    // Reset only with the command buffer and otherwise will maintain its values
    // during recording to allow for partial push_constants updates.
    uint32_t push_constants[IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT];

    // Cached and initialized dispatch state reused for all dispatches.
    // Individual dispatches must populate the dynamically changing fields like
    // push_constant_count and binding_count.
    iree_alignas(64) iree_hal_executable_dispatch_state_v0_t dispatch_state;

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
  dispatch_state->push_constants = command_buffer->state.push_constants;
  dispatch_state->binding_ptrs = command_buffer->state.packed_bindings;
  dispatch_state->binding_lengths =
      command_buffer->state.packed_binding_lengths;
}

iree_host_size_t iree_hal_inline_command_buffer_size(void) {
  return sizeof(iree_hal_inline_command_buffer_t);
}

iree_status_t iree_hal_inline_command_buffer_initialize(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
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
  if (storage.data_length < iree_hal_inline_command_buffer_size()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "storage must have at least the capacity as "
                            "defined by iree_hal_inline_command_buffer_size");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_inline_command_buffer_t* command_buffer =
      (iree_hal_inline_command_buffer_t*)storage.data;
  memset(command_buffer, 0, sizeof(*command_buffer));

  iree_hal_command_buffer_initialize(
      device, mode, command_categories, queue_affinity, binding_capacity,
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
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  uint8_t* storage = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, iree_hal_inline_command_buffer_size(), (void**)&storage);
  iree_hal_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_inline_command_buffer_initialize(
        device, mode, command_categories, queue_affinity, binding_capacity,
        host_allocator,
        iree_make_byte_span(storage, iree_hal_inline_command_buffer_size()),
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
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // Could be treated as a cache invalidation as it indicates we won't be using
  // the existing buffer contents again.
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_fill_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  return iree_hal_buffer_map_fill(target_buffer, target_offset, length, pattern,
                                  pattern_length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_update_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_hal_buffer_map_write(
      target_buffer, target_offset,
      (const uint8_t*)source_buffer + source_offset, length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_copy_buffer
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  return iree_hal_buffer_map_copy(source_buffer, source_offset, target_buffer,
                                  target_offset, length);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_collective
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet implemented on CPU");
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_push_constants
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_inline_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);

  if (IREE_UNLIKELY(offset + values_length >=
                    sizeof(command_buffer->state.push_constants))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant range %zu (length=%zu) out of range",
                            offset, values_length);
  }

  memcpy((uint8_t*)&command_buffer->state.push_constants + offset, values,
         values_length);

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_push_descriptor_set
//===----------------------------------------------------------------------===//
// NOTE: command buffer state change only; enqueues no tasks.

static iree_status_t iree_hal_inline_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);

  if (IREE_UNLIKELY(set >= IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "set %u out of bounds", set);
  }

  iree_host_size_t binding_base =
      set * IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT;
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    if (IREE_UNLIKELY(bindings[i].binding >=
                      IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffer binding index out of bounds");
    }
    iree_host_size_t binding_ordinal = binding_base + bindings[i].binding;

    // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
    iree_hal_buffer_mapping_t buffer_mapping = {{0}};
    if (bindings[i].buffer) {
      IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
          bindings[i].buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
          IREE_HAL_MEMORY_ACCESS_ANY, bindings[i].offset, bindings[i].length,
          &buffer_mapping));
    }
    command_buffer->state.full_bindings[binding_ordinal] =
        buffer_mapping.contents.data;
    command_buffer->state.full_binding_lengths[binding_ordinal] =
        buffer_mapping.contents.data_length;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_inline_command_buffer_t* command_buffer =
      iree_hal_inline_command_buffer_cast(base_command_buffer);

  iree_hal_local_executable_t* local_executable =
      iree_hal_local_executable_cast(executable);
  if (IREE_UNLIKELY(!local_executable->pipeline_layouts)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "layouts not provided during executable creation; cannot dispatch");
  }

  iree_hal_local_pipeline_layout_t* local_layout =
      (iree_hal_local_pipeline_layout_t*)
          local_executable->pipeline_layouts[entry_point];
  iree_host_size_t local_memory_size =
      local_executable->dispatch_attrs
          ? local_executable->dispatch_attrs[entry_point].local_memory_pages *
                IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE
          : 0;

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
  dispatch_state->workgroup_count_x = workgroup_x;
  dispatch_state->workgroup_count_y = workgroup_y;
  dispatch_state->workgroup_count_z = workgroup_z;

  // Single-threaded.
  dispatch_state->max_concurrency = 1;

  // Push constants are pulled directly from the command buffer state, but we
  // only allow the dispatch to read what we know is initialized based on the
  // layout.
  dispatch_state->push_constant_count = local_layout->push_constants;

  // Produce the dense binding list based on the declared bindings used.
  // This allows us to change the descriptor sets and bindings counts supported
  // in the HAL independent of any executable as each executable just gets the
  // flat dense list and doesn't care about our descriptor set stuff.
  //
  // Note that we are just directly setting the binding data pointers here with
  // no ownership/retaining/etc - it's part of the HAL contract that buffers are
  // kept valid for the duration they may be in use.
  iree_hal_local_binding_mask_t used_binding_mask = local_layout->used_bindings;
  iree_host_size_t used_binding_count =
      iree_math_count_ones_u64(used_binding_mask);
  dispatch_state->binding_count = used_binding_count;
  void** binding_ptrs = (void**)dispatch_state->binding_ptrs;
  size_t* binding_lengths = (size_t*)dispatch_state->binding_lengths;
  iree_host_size_t binding_base = 0;
  for (iree_host_size_t i = 0; i < used_binding_count; ++i) {
    int mask_offset = iree_math_count_trailing_zeros_u64(used_binding_mask);
    int binding_ordinal = binding_base + mask_offset;
    binding_base += mask_offset + 1;
    used_binding_mask = iree_shr(used_binding_mask, mask_offset + 1);
    binding_ptrs[i] = command_buffer->state.full_bindings[binding_ordinal];
    if (!binding_ptrs[i]) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "(flat) binding %d is NULL", binding_ordinal);
    }
    binding_lengths[i] =
        command_buffer->state.full_binding_lengths[binding_ordinal];
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
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  // TODO(benvanik): track mapping so we can properly map/unmap/flush/etc.
  iree_hal_buffer_mapping_t buffer_mapping = {{0}};
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      workgroups_buffer, IREE_HAL_MAPPING_MODE_PERSISTENT,
      IREE_HAL_MEMORY_ACCESS_READ, workgroups_offset, 3 * sizeof(uint32_t),
      &buffer_mapping));
  iree_hal_vec3_t workgroup_count =
      *(const iree_hal_vec3_t*)buffer_mapping.contents.data;
  return iree_hal_inline_command_buffer_dispatch(
      base_command_buffer, executable, entry_point, workgroup_count.x,
      workgroup_count.y, workgroup_count.z);
}

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_execute_commands
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_inline_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  // TODO(#10144): decide how to execute the inline command buffer; it is
  // definitely a deferred command buffer but we don't want to force that
  // dependency here. We could allow injection of a function to call to execute
  // command buffers so that the device can decide how it wants to handle them.
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
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
        .push_constants = iree_hal_inline_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_inline_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_inline_command_buffer_dispatch,
        .dispatch_indirect = iree_hal_inline_command_buffer_dispatch_indirect,
        .execute_commands = iree_hal_inline_command_buffer_execute_commands,
};
