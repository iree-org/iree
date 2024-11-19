// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/null/command_buffer.h"

#include "iree/hal/drivers/null/buffer.h"
#include "iree/hal/drivers/null/channel.h"
#include "iree/hal/drivers/null/executable.h"

//===----------------------------------------------------------------------===//
// iree_hal_null_command_buffer_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_null_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
} iree_hal_null_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_null_command_buffer_vtable;

static iree_hal_null_command_buffer_t* iree_hal_null_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_null_command_buffer_vtable);
  return (iree_hal_null_command_buffer_t*)base_value;
}

iree_status_t iree_hal_null_command_buffer_create(
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  iree_hal_null_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_null_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;

  // TODO(null): allocate any additional resources for managing command buffer
  // state. Some implementations may have their own command buffer/command list
  // APIs this can route to or may need to implement it all themselves using
  // iree_arena_t/block pools. Implementations should also retain any resources
  // used during the recording and can use iree_hal_resource_set_t* to make that
  // easier.
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "command buffers not yet implemented");

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_null_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // TODO(null): release any implementation resources and
  // iree_hal_resource_set_t.

  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_null_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_null_command_buffer_vtable);
}

static iree_status_t iree_hal_null_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): if the implementation needs to route the begin to the
  // implementation it can be done here. Note that creation may happen much
  // earlier than recording and any expensive work should be deferred until this
  // point to make profiling easier.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                       "command buffer recording start not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): if recording requires multiple passes any fixup/linking can
  // happen here. Recording-only resources are no longer needed after this point
  // and can be disposed.
  (void)command_buffer;
  iree_status_t status = iree_make_status(
      IREE_STATUS_UNIMPLEMENTED, "command buffer finalization not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): begin a nested debug group (push) if the implementation has a
  // way to insert markers. This is informational and can be ignored.
  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t iree_hal_null_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): end a nested debug group (pop). Always called 1:1 in stack
  // order with begin_debug_group.
  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t iree_hal_null_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): barriers split the execution sequence into all operations that
  // did happen before the barrier and all that will happen after. In
  // implementations that have no concurrency this can be a no-op. This is
  // effectively just a signal_event followed by a wait_event.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "execution barriers not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): WIP API and may change; signals the given event allowing
  // waiters to proceed.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): WIP API and may change; resets the given event to unsignaled.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): WIP API and may change; waits on the list of events and enacts
  // the specified set of barriers. Implementations without fine-grained
  // tracking can treat this as an execution_barrier and ignore the
  // memory/buffer barriers provided.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "events not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): WIP API and may change; this is likely to become an
  // madvise-like command that can be used to control prefetching and other
  // cache behavior. The current discard behavior is a hint that the buffer
  // contents will never be used again and that if they are in a cache they need
  // not be written back to global memory.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "discard buffer not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): memset on the buffer. The pattern_length is 1, 2, or 4 bytes.
  // Note that the buffer may be a reference to a binding table slot in which
  // case it will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "fill buffer not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): embed and copy a small (~64KB) chunk of host memory to the
  // target buffer. The source_buffer contents must be captured as they may
  // change/be freed after this call completes.
  // Note that the target buffer may be a reference to a binding table slot in
  // which case it will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "update buffer not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): memcpy between two buffers. The buffers must both be
  // device-visible but may reside on either the host or device.
  // Note that either buffer may be a reference to a binding table slot in
  // which case it will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "copy buffer not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): perform the collective operation defined by op. See the headers
  // for more information. The channel is fixed for a particular recording but
  // note that either buffer may be a reference to a binding table slot in
  // which case it will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "collectives not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const uint32_t workgroup_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): dispatch the specified executable entry point with the given
  // workgroup count. The constants must be copied into the command buffer as
  // they may be mutated or freed after this call returns.
  // Note that any of the bindings may be references to binding table slots in
  // which case they will be provided during submission to a queue.
  (void)command_buffer;
  iree_status_t status =
      iree_make_status(IREE_STATUS_UNIMPLEMENTED, "dispatch not implemented");

  return status;
}

static iree_status_t iree_hal_null_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_null_command_buffer_t* command_buffer =
      iree_hal_null_command_buffer_cast(base_command_buffer);

  // TODO(null): dispatch the specified executable entry point with a workgroup
  // count that is stored in the given workgroup count buffer as a uint32_t[3].
  // The workgroup count may change up until immediately prior to the dispatch.
  // The constants must be copied into the command buffer as they may be mutated
  // or freed after this call returns. Note that any of the bindings may be
  // references to binding table slots in which case they will be provided
  // during submission to a queue.
  (void)command_buffer;
  iree_status_t status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                          "indirect dispatch not implemented");

  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_null_command_buffer_vtable = {
        .destroy = iree_hal_null_command_buffer_destroy,
        .begin = iree_hal_null_command_buffer_begin,
        .end = iree_hal_null_command_buffer_end,
        .begin_debug_group = iree_hal_null_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_null_command_buffer_end_debug_group,
        .execution_barrier = iree_hal_null_command_buffer_execution_barrier,
        .signal_event = iree_hal_null_command_buffer_signal_event,
        .reset_event = iree_hal_null_command_buffer_reset_event,
        .wait_events = iree_hal_null_command_buffer_wait_events,
        .advise_buffer = iree_hal_null_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_null_command_buffer_fill_buffer,
        .update_buffer = iree_hal_null_command_buffer_update_buffer,
        .copy_buffer = iree_hal_null_command_buffer_copy_buffer,
        .collective = iree_hal_null_command_buffer_collective,
        .dispatch = iree_hal_null_command_buffer_dispatch,
        .dispatch_indirect = iree_hal_null_command_buffer_dispatch_indirect,
};
