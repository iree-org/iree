// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/hip_multi_queue_command_buffer.h"

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/hal/drivers/hip/context_util.h"
#include "iree/hal/drivers/hip/status_util.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_hip_multi_queue_command_buffer_t implementation
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hip_multi_queue_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_host_size_t command_buffer_count;
  iree_hal_hip_device_topology_t topology;
  const iree_hal_hip_dynamic_symbols_t* hip_symbols;
  iree_hal_command_buffer_t* child_buffers[];
} iree_hal_hip_multi_queue_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_hip_multi_queue_command_buffer_vtable;

static iree_hal_hip_multi_queue_command_buffer_t*
iree_hal_hip_multi_queue_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value,
                       &iree_hal_hip_multi_queue_command_buffer_vtable);
  return (iree_hal_hip_multi_queue_command_buffer_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_hip_multi_queue_command_buffer_create(
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t** in_command_buffers,
    iree_hal_allocator_t* device_allocator, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_hip_dynamic_symbols_t* hip_symbols,
    iree_hal_hip_device_topology_t topology, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (iree_math_count_ones_u64(queue_affinity) != command_buffer_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected one command buffer per enabled queue");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_multi_queue_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(
              host_allocator,
              sizeof(*command_buffer) +
                  command_buffer_count * sizeof(iree_hal_command_buffer_t*) +
                  iree_hal_command_buffer_validation_state_size(
                      mode, binding_capacity),
              (void**)&command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity,
      (uint8_t*)command_buffer + sizeof(*command_buffer) +
          command_buffer_count * sizeof(iree_hal_command_buffer_t*),
      &iree_hal_hip_multi_queue_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->command_buffer_count = command_buffer_count;
  command_buffer->topology = topology;
  command_buffer->hip_symbols = hip_symbols;
  for (iree_host_size_t i = 0; i < command_buffer_count; ++i) {
    command_buffer->child_buffers[i] = in_command_buffers[i];
    iree_hal_resource_retain(command_buffer->child_buffers[i]);
  }

  *out_command_buffer = (iree_hal_command_buffer_t*)command_buffer;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_multi_queue_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  for (iree_host_size_t i = 0; i < command_buffer->command_buffer_count; ++i) {
    iree_hal_resource_release(command_buffer->child_buffers[i]);
  }
  iree_allocator_free(command_buffer->host_allocator, command_buffer);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT bool iree_hal_hip_multi_queue_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_hip_multi_queue_command_buffer_vtable);
}

IREE_API_EXPORT iree_status_t iree_hal_hip_multi_queue_command_buffer_get(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  *out_command_buffer = NULL;
  if (iree_math_count_ones_u64(queue_affinity) != 1) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "one and only one device may be specified.");
  }
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  if (!(command_buffer->base.queue_affinity & queue_affinity)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no command buffer for affinity %" PRIu64,
                            queue_affinity);
  }
  int index = iree_math_count_ones_u64(command_buffer->base.queue_affinity &
                                       (queue_affinity - 1));

  *out_command_buffer = command_buffer->child_buffers[index];
  return iree_ok_status();
}

// Use |command_buffer_index| in the command to index into the correct
// command buffer, within the given command.
#define CALL_COMMAND(status, command)                                    \
  do {                                                                   \
    iree_hal_queue_affinity_t queue_affinity =                           \
        command_buffer->base.queue_affinity;                             \
    int command_buffer_index = 0;                                        \
    int device_ordinal = 0;                                              \
    while (queue_affinity && IREE_LIKELY(iree_status_is_ok(status))) {   \
      int count = iree_math_count_trailing_zeros_u64(queue_affinity);    \
      device_ordinal += count;                                           \
      status = iree_hal_hip_set_context(                                 \
          command_buffer->hip_symbols,                                   \
          command_buffer->topology.devices[device_ordinal].hip_context); \
      if (!iree_status_is_ok(status)) {                                  \
        break;                                                           \
      }                                                                  \
      status = command;                                                  \
      queue_affinity >>= (count + 1);                                    \
      device_ordinal += 1;                                               \
      ++command_buffer_index;                                            \
    }                                                                    \
  } while (false)

static iree_status_t iree_hal_hip_multi_queue_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status,
               iree_hal_command_buffer_begin(
                   command_buffer->child_buffers[command_buffer_index]));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status,
               iree_hal_command_buffer_end(
                   command_buffer->child_buffers[command_buffer_index]));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_execution_barrier(
                           command_buffer->child_buffers[command_buffer_index],
                           source_stage_mask, target_stage_mask, flags,
                           memory_barrier_count, memory_barriers,
                           buffer_barrier_count, buffer_barriers));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_signal_event(
                           command_buffer->child_buffers[command_buffer_index],
                           event, source_stage_mask));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_reset_event(
                           command_buffer->child_buffers[command_buffer_index],
                           event, source_stage_mask));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(
      status,
      iree_hal_command_buffer_wait_events(
          command_buffer->child_buffers[command_buffer_index], event_count,
          events, source_stage_mask, target_stage_mask, memory_barrier_count,
          memory_barriers, buffer_barrier_count, buffer_barriers));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags,
    uint64_t arg0, uint64_t arg1) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_advise_buffer(
                           command_buffer->child_buffers[command_buffer_index],
                           buffer_ref, flags, arg0, arg1));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_fill_buffer(
                           command_buffer->child_buffers[command_buffer_index],
                           target_ref, pattern, pattern_length, flags));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_update_buffer(
                           command_buffer->child_buffers[command_buffer_index],
                           source_buffer, source_offset, target_ref, flags));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status, iree_hal_command_buffer_copy_buffer(
                           command_buffer->child_buffers[command_buffer_index],
                           source_ref, target_ref, flags));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(status,
               iree_hal_command_buffer_collective(
                   command_buffer->child_buffers[command_buffer_index], channel,
                   op, param, send_ref, recv_ref, element_count));
  return status;
}

static iree_status_t iree_hal_hip_multi_queue_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_hip_multi_queue_command_buffer_t* command_buffer =
      iree_hal_hip_multi_queue_command_buffer_cast(base_command_buffer);
  iree_status_t status = iree_ok_status();
  CALL_COMMAND(
      status, iree_hal_command_buffer_dispatch(
                  command_buffer->child_buffers[command_buffer_index],
                  executable, entry_point, config, constants, bindings, flags));
  return status;
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hip_multi_queue_command_buffer_vtable = {
        .destroy = iree_hal_hip_multi_queue_command_buffer_destroy,
        .begin = iree_hal_hip_multi_queue_command_buffer_begin,
        .end = iree_hal_hip_multi_queue_command_buffer_end,
        .execution_barrier =
            iree_hal_hip_multi_queue_command_buffer_execution_barrier,
        .signal_event = iree_hal_hip_multi_queue_command_buffer_signal_event,
        .reset_event = iree_hal_hip_multi_queue_command_buffer_reset_event,
        .wait_events = iree_hal_hip_multi_queue_command_buffer_wait_events,
        .advise_buffer = iree_hal_hip_multi_queue_command_buffer_advise_buffer,
        .fill_buffer = iree_hal_hip_multi_queue_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hip_multi_queue_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hip_multi_queue_command_buffer_copy_buffer,
        .collective = iree_hal_hip_multi_queue_command_buffer_collective,
        .dispatch = iree_hal_hip_multi_queue_command_buffer_dispatch,
};

#undef CALL_COMMAND
