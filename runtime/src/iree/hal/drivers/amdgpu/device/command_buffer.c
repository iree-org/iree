// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/device/command_buffer.h"

// TODO(benvanik): bring in scheduler implementation. For now we define the
// methods we use to produce valid builds.
static void
iree_hal_amdgpu_device_queue_scheduler_reschedule_from_execution_queue(
    iree_hal_amdgpu_device_queue_scheduler_t* scheduler,
    uint64_t scheduler_queue_entry) {}
static void iree_hal_amdgpu_device_queue_scheduler_retire_from_execution_queue(
    iree_hal_amdgpu_device_queue_scheduler_t* scheduler,
    uint64_t scheduler_queue_entry) {}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Returns the packet pointer in the execution queue with the given |packet_id|.
static iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT
iree_hal_amdgpu_device_cmd_resolve_dispatch_packet(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const uint64_t packet_id) {
  const uint64_t queue_mask = state->execution_queue.size - 1;  // power of two
  return state->execution_queue.base_address + (packet_id & queue_mask) * 64;
}

// Makes all bits of an AQL packet header except the type.
// The caller must OR in the type before setting the packet.
static inline IREE_AMDGPU_ATTRIBUTE_ALWAYS_INLINE uint16_t
iree_hal_amdgpu_device_make_cmd_packet_header(
    const iree_hal_amdgpu_device_cmd_header_t* IREE_AMDGPU_RESTRICT cmd_header,
    iree_hal_amdgpu_device_execution_flags_t execution_flags) {
  const bool force_barrier =
      (execution_flags & IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_SERIALIZE) != 0;
  const bool force_uncached =
      (execution_flags & IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_UNCACHED) != 0;

  // Translate command flags; they're mostly just bit-packed header bits.
  const bool barrier =
      force_barrier ||
      ((cmd_header->flags &
        IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER) != 0);
  const iree_hsa_fence_scope_t scacquire_fence_scope =
      force_uncached ? IREE_HSA_FENCE_SCOPE_SYSTEM
                     : ((cmd_header->flags >>
                         IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_ACQUIRE_BIT) &
                        0x3);
  const iree_hsa_fence_scope_t screlease_fence_scope =
      force_uncached ? IREE_HSA_FENCE_SCOPE_SYSTEM
                     : ((cmd_header->flags >>
                         IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_FENCE_RELEASE_BIT) &
                        0x3);

  // Form the header word.
  return (barrier << IREE_HSA_PACKET_HEADER_BARRIER) |
         (scacquire_fence_scope
          << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
         (screlease_fence_scope
          << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
}

// Emplaces a lightweight barrier packet (no cache management, no-op wait)
// and associates the optional |completion_signal|. The packet processor will
// populate the timestamps on the signal after the packet has retired.
static void iree_hal_amdgpu_device_cmd_emplace_barrier(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_cmd_header_t* IREE_AMDGPU_RESTRICT cmd_header,
    const uint64_t packet_id, iree_hsa_signal_t completion_signal) {
  const uint64_t queue_mask = state->execution_queue.size - 1;  // power of two
  iree_hsa_barrier_or_packet_t* IREE_AMDGPU_RESTRICT packet =
      state->execution_queue.base_address + (packet_id & queue_mask) * 64;

  // No signals to make this a no-op.
  for (size_t i = 0; i < IREE_AMDGPU_ARRAYSIZE(packet->dep_signal); ++i) {
    packet->dep_signal[i] = iree_hsa_signal_null();
  }

  // Chain the provided signal, which is likely an trace query.
  packet->completion_signal = completion_signal;

  // Form the header word.
  // NOTE: uint16_t high is reserved0.
  const uint32_t barrier_header =
      iree_hal_amdgpu_device_make_cmd_packet_header(cmd_header, state->flags);

  // Swap header to enable the packet.
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)packet, barrier_header,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_device);
}

// Commits a CFG control packet.
// These are assumed to run on a single thread.
static void iree_hal_amdgpu_device_cmd_commit_cfg_packet(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_cmd_header_t* IREE_AMDGPU_RESTRICT cmd_header,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id,
    const iree_hal_amdgpu_device_kernel_args_t* IREE_AMDGPU_RESTRICT
        kernel_args,
    void* IREE_AMDGPU_RESTRICT kernarg_ptr) {
  // Emplace a packet in the execution queue but leave the header uninitialized.
  const uint64_t queue_mask = state->execution_queue.size - 1;  // power of two
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      (iree_hsa_kernel_dispatch_packet_t*)state->execution_queue.base_address +
      (packet_id & queue_mask);
  packet->setup = kernel_args->setup;
  packet->workgroup_size[0] = kernel_args->workgroup_size[0];
  packet->workgroup_size[1] = kernel_args->workgroup_size[1];
  packet->workgroup_size[2] = kernel_args->workgroup_size[2];
  packet->reserved0 = 0;
  packet->grid_size[0] = 1;
  packet->grid_size[1] = 1;
  packet->grid_size[2] = 1;
  packet->private_segment_size = kernel_args->private_segment_size;
  packet->group_segment_size = kernel_args->group_segment_size;
  packet->kernel_object = kernel_args->kernel_object;
  packet->kernarg_address = kernarg_ptr;
  packet->reserved2 = 0;

#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  // Enqueue tracing event and get a query signal used for timing.
  if (execution_query_id != IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    packet->completion_signal =
        iree_hal_amdgpu_device_trace_execution_zone_dispatch(
            state->trace_buffer,
            IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_INTERNAL,
            kernel_args->trace_src_loc, execution_query_id);
  } else {
    packet->completion_signal = iree_hsa_signal_null();
  }
#else
  packet->completion_signal = iree_hsa_signal_null();
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // Populate the header and release the packet to the queue.
  const uint16_t header =
      (IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH << IREE_HSA_PACKET_HEADER_TYPE) |
      iree_hal_amdgpu_device_make_cmd_packet_header(cmd_header, state->flags);
  const uint32_t header_setup = header | (uint32_t)(packet->setup << 16);
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)packet, header_setup,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_device);

  iree_hsa_signal_store(state->execution_queue.doorbell_signal, packet_id,
                        iree_amdgpu_memory_order_release);
}

// Updates a dispatch packet header and optional tracing query signal.
// The returned packet will still have an INVALID type and that will need to be
// OR'ed in by the caller.
static void iree_hal_amdgpu_device_cmd_update_dispatch_packet(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_cmd_header_t* IREE_AMDGPU_RESTRICT cmd_header,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet,
    const iree_hal_amdgpu_trace_execution_zone_type_t execution_zone_type,
    const uint64_t export_loc,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  // Enqueue tracing event and get a query signal used for timing.
  if (execution_query_id != IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    packet->completion_signal =
        iree_hal_amdgpu_device_trace_execution_zone_dispatch(
            state->trace_buffer, execution_zone_type, export_loc,
            execution_query_id);
  } else {
    packet->completion_signal = iree_hsa_signal_null();
  }
#else
  packet->completion_signal = iree_hsa_signal_null();
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION

  // Populate the header and release the packet to the queue.
  // NOTE: we don't assign the packet type yet - the commit needs to do that
  // only when the packet has been full formed.
  packet->header =
      (IREE_HSA_PACKET_TYPE_INVALID << IREE_HSA_PACKET_HEADER_TYPE) |
      iree_hal_amdgpu_device_make_cmd_packet_header(cmd_header, state->flags);
}

// Commits a dispatch or transfer packet.
static void iree_hal_amdgpu_device_cmd_commit_dispatch_packet(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_cmd_header_t* IREE_AMDGPU_RESTRICT cmd_header,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet,
    const iree_hal_amdgpu_trace_execution_zone_type_t execution_zone_type,
    const uint64_t export_loc,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Update the packet with the required information.
  iree_hal_amdgpu_device_cmd_update_dispatch_packet(
      state, cmd_header, packet, execution_zone_type, export_loc,
      execution_query_id);

  // Update the header from INVALID to KERNEL_DISPATCH so the packet processor
  // can begin executing it.
  const uint16_t header =
      (packet->header &
       ~(IREE_HSA_PACKET_TYPE_INVALID << IREE_HSA_PACKET_HEADER_TYPE)) |
      (IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH << IREE_HSA_PACKET_HEADER_TYPE);
  const uint32_t header_setup = header | (uint32_t)(packet->setup << 16);
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)packet, header_setup,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_device);
}

// Flushes all outstanding tracing queries from the current block.
// If the caller is running on the execution queue it will not be included (as
// its query has not yet been resolved).
//
// TODO(benvanik): support resolving the final terminator time.
static void iree_hal_amdgpu_device_flush_execution_queries(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state) {
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION)
  iree_hal_amdgpu_trace_agent_time_range_t* IREE_AMDGPU_RESTRICT time_ranges =
      iree_hal_amdgpu_device_trace_execution_zone_notify_batch(
          state->trace_buffer, state->trace_block_query_base_id,
          state->trace_block_query_count);
  for (uint16_t i = 0; i < state->trace_block_query_count; ++i) {
    iree_amd_signal_t* IREE_AMDGPU_RESTRICT signal =
        (iree_amd_signal_t*)
            iree_hal_amdgpu_device_query_ringbuffer_signal_for_id(
                &state->trace_buffer->query_ringbuffer,
                state->trace_block_query_base_id + i)
                .handle;
    time_ranges[i] = (iree_hal_amdgpu_trace_agent_time_range_t){
        .begin = signal->start_ts,
        .end = signal->end_ts,
    };
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_EXECUTION
}

//===----------------------------------------------------------------------===//
// Device-side Enqueuing
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    uint32_t command_ordinal, uint64_t base_packet_id);

static void iree_hal_amdgpu_device_command_buffer_enqueue_next_block_serial(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state) {
  // Reserve space for all of the execution packets. We will populate them all
  // in the loop below.
  const uint64_t base_packet_id = iree_hsa_queue_add_write_index(
      &state->execution_queue, state->block->max_packet_count,
      iree_amdgpu_memory_order_relaxed);
  while (base_packet_id -
             iree_hsa_queue_load_read_index(&state->execution_queue,
                                            iree_amdgpu_memory_order_acquire) >=
         state->execution_queue.size) {
    iree_amdgpu_yield();  // spinning
  }

  // Signal the execution queue doorbell immediately even though we haven't
  // populated the packets yet: it should kick it into waking and spinning while
  // we populate the packets. Since we write packets in order the span from this
  // moment to the first packet execution should be as small as possible when
  // hopping queues.
  iree_hsa_signal_store(state->execution_queue.doorbell_signal,
                        base_packet_id + state->block->max_packet_count,
                        iree_amdgpu_memory_order_relaxed);

  // Issue all packets to the execution queue.
  for (uint32_t i = 0; i < state->block->command_count; ++i) {
    iree_hal_amdgpu_device_cmd_issue(state, state->block, i, base_packet_id);
  }
}

static void iree_hal_amdgpu_device_command_buffer_enqueue_next_block_parallel(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state) {
  // Reserve the next packet in the control queue for the issue_block kernel.
  const iree_amd_cached_queue_t* control_queue = state->control_queue;
  const uint64_t control_packet_id = iree_hsa_queue_add_write_index(
      control_queue, 1u, iree_amdgpu_memory_order_relaxed);
  while (control_packet_id -
             iree_hsa_queue_load_read_index(control_queue,
                                            iree_amdgpu_memory_order_acquire) >=
         control_queue->size) {
    iree_amdgpu_yield();  // spinning
  }

  // Reserve space for all of the execution packets.
  // We need to ensure we have this entire range for the issue_block kernel to
  // populate prior to launching it.
  //
  // NOTE: we do this after we insert the control packet as the control and
  // execution queues may be the same: we must issue the control packet prior to
  // any execution packets that are reserved as INVALID and that will block the
  // packet processor.
  const uint64_t base_packet_id = iree_hsa_queue_add_write_index(
      &state->execution_queue, state->block->max_packet_count,
      iree_amdgpu_memory_order_relaxed);
  while (base_packet_id -
             iree_hsa_queue_load_read_index(&state->execution_queue,
                                            iree_amdgpu_memory_order_acquire) >=
         state->execution_queue.size) {
    iree_amdgpu_yield();  // spinning
  }

  // Kernel arguments stored in the shared control kernarg storage. There should
  // only be one control dispatch enqueued at a time.
  uint64_t* kernarg_ptr = (uint64_t*)state->control_kernarg_storage;
  kernarg_ptr[0] = (uint64_t)state;
  kernarg_ptr[1] = (uint64_t)state->block;
  kernarg_ptr[2] = base_packet_id;

  // Construct the control packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const iree_hal_amdgpu_device_kernel_args_t control_args =
      state->kernels->iree_hal_amdgpu_device_cmd_block_issue;
  const uint64_t queue_mask = control_queue->size - 1;  // power of two
  iree_hsa_kernel_dispatch_packet_t* control_packet =
      (iree_hsa_kernel_dispatch_packet_t*)control_queue->base_address +
      (control_packet_id & queue_mask);
  control_packet->setup = control_args.setup;
  control_packet->workgroup_size[0] = control_args.workgroup_size[0];
  control_packet->workgroup_size[1] = control_args.workgroup_size[1];
  control_packet->workgroup_size[2] = control_args.workgroup_size[2];
  control_packet->reserved0 = 0;
  control_packet->grid_size[0] = state->block->command_count;
  control_packet->grid_size[1] = 1;
  control_packet->grid_size[2] = 1;
  control_packet->private_segment_size = control_args.private_segment_size;
  control_packet->group_segment_size = control_args.group_segment_size;
  control_packet->kernel_object = control_args.kernel_object;
  control_packet->kernarg_address = kernarg_ptr;
  control_packet->reserved2 = 0;
  control_packet->completion_signal.handle = 0;

  // Populate the header and release the packet to the queue.
  uint16_t control_header = IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH
                            << IREE_HSA_PACKET_HEADER_TYPE;

  // Force a barrier while performing the issue so that any shared resources we
  // use will not have hazards.
  control_header |= 1 << IREE_HSA_PACKET_HEADER_BARRIER;

  // We scope to the agent as we should be scheduled and targeting execution
  // queues on the same one.
  control_header |= IREE_HSA_FENCE_SCOPE_AGENT
                    << IREE_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  control_header |= IREE_HSA_FENCE_SCOPE_AGENT
                    << IREE_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

  // Mark the control packet as ready to execute. The hardware command processor
  // may begin executing it immediately after performing the atomic swap.
  const uint32_t control_header_setup =
      control_header | (uint32_t)(control_packet->setup << 16);
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)control_packet, control_header_setup,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_device);

  // Signal the queue doorbell indicating the packet has been updated.
  iree_hsa_signal_store(control_queue->doorbell_signal, control_packet_id,
                        iree_amdgpu_memory_order_relaxed);
}

void iree_hal_amdgpu_device_command_buffer_enqueue_next_block(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state) {
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)
  // Reserve a query ID range for the commands in the block.
  // We take up to the maximum required for each tracing mode we may be in but
  // the block may not use them all.
  uint16_t query_count = 0;
  if (state->flags & IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH) {
    query_count = state->block->query_map.max_dispatch_query_count;
  } else if (state->flags &
             IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL) {
    query_count = state->block->query_map.max_control_query_count;
  }
  state->trace_block_query_count = query_count;
  if (query_count > 0) {
    state->trace_block_query_base_id =
        iree_hal_amdgpu_device_query_ringbuffer_acquire_range(
            &state->trace_buffer->query_ringbuffer, query_count);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // The execution request decides whether we issue serially here on the control
  // queue or in parallel via a dispatch to the control queue. The dispatch has
  // higher latency but greater throughput and is something we only want to use
  // if that throughput is required (lots of commands).
  if (state->flags & IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_ISSUE_SERIALLY) {
    return iree_hal_amdgpu_device_command_buffer_enqueue_next_block_serial(
        state);
  } else {
    return iree_hal_amdgpu_device_command_buffer_enqueue_next_block_parallel(
        state);
  }
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_debug_group_begin_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_debug_group_begin_t* IREE_AMDGPU_RESTRICT
        cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // If tracing is enabled then get the signal used to query timestamps.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)
  if (execution_query_id != IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    completion_signal = iree_hal_amdgpu_device_trace_execution_zone_begin(
        state->trace_buffer, execution_query_id, cmd->src_loc);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // Emit a lightweight barrier packet (no cache management, no-op wait) to
  // force the command buffer to execute as if we were capturing timing even if
  // we aren't. This can be useful for native debugging tools and also lets us
  // more easily detect the overhead of tracing.
  return iree_hal_amdgpu_device_cmd_emplace_barrier(
      state, &cmd->header, packet_id, completion_signal);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_debug_group_end_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_debug_group_end_t* IREE_AMDGPU_RESTRICT
        cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // If tracing is enabled then get the signal used to query timestamps.
  iree_hsa_signal_t completion_signal = iree_hsa_signal_null();
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)
  if (execution_query_id != IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    completion_signal = iree_hal_amdgpu_device_trace_execution_zone_end(
        state->trace_buffer, execution_query_id);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // Emit a lightweight barrier packet (no cache management, no-op wait) to
  // force the command buffer to execute as if we were capturing timing even if
  // we aren't. This can be useful for native debugging tools and also lets us
  // more easily detect the overhead of tracing.
  return iree_hal_amdgpu_device_cmd_emplace_barrier(
      state, &cmd->header, packet_id, completion_signal);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_barrier_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_barrier_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): derive scope from command header.
  return iree_hal_amdgpu_device_cmd_emplace_barrier(
      state, &cmd->header, packet_id, iree_hsa_signal_null());
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_signal_event_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_signal_event_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): HSA signal handling.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_reset_event_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_reset_event_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): HSA signal handling.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_wait_events_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_wait_events_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): HSA signal handling.
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_fill_buffer_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_fill_buffer_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Resolve bindings.
  void* target_ptr = iree_hal_amdgpu_device_buffer_ref_resolve(cmd->target_ref,
                                                               state->bindings);
  const uint64_t length = cmd->target_ref.length;

  // Emplace a packet in the execution queue but leave the header uninitialized.
  uint64_t* kernarg_ptr =
      (uint64_t*)(state->execution_kernarg_storage + cmd->kernarg_offset);
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_fill_emplace_reserve(
          &state->transfer_context, target_ptr, length, cmd->pattern,
          cmd->pattern_length, kernarg_ptr, packet_id);

  // Commit the packet.
  return iree_hal_amdgpu_device_cmd_commit_dispatch_packet(
      state, &cmd->header, packet,
      IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_FILL, 0, execution_query_id);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_device_cmd_copy_buffer_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_copy_buffer_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Resolve bindings.
  const void* source_ptr = iree_hal_amdgpu_device_buffer_ref_resolve(
      cmd->source_ref, state->bindings);
  void* target_ptr = iree_hal_amdgpu_device_buffer_ref_resolve(cmd->target_ref,
                                                               state->bindings);
  const uint64_t length = cmd->target_ref.length;

  // Emplace a packet in the execution queue but leave the header uninitialized.
  uint64_t* kernarg_ptr =
      (uint64_t*)(state->execution_kernarg_storage + cmd->kernarg_offset);
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_buffer_copy_emplace_reserve(
          &state->transfer_context, source_ptr, target_ptr, length, kernarg_ptr,
          packet_id);

  // Commit the packet.
  return iree_hal_amdgpu_device_cmd_commit_dispatch_packet(
      state, &cmd->header, packet,
      IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_COPY, 0, execution_query_id);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH
//===----------------------------------------------------------------------===//

static iree_hsa_kernel_dispatch_packet_t*
iree_hal_amdgpu_device_cmd_dispatch_reserve(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_dispatch_t* IREE_AMDGPU_RESTRICT cmd,
    uint8_t* IREE_AMDGPU_RESTRICT kernarg_base, const uint64_t packet_id) {
  const iree_hal_amdgpu_device_kernel_args_t* dispatch_args =
      cmd->config.kernel_args;

  // Populate bindings and constants in the reserved kernarg storage.
  for (uint16_t i = 0; i < dispatch_args->binding_count; ++i) {
    ((uint64_t*)kernarg_base)[i] =
        (uint64_t)iree_hal_amdgpu_device_buffer_ref_resolve(cmd->bindings[i],
                                                            state->bindings);
  }
  uint8_t* kernarg_ptr =
      kernarg_base + dispatch_args->binding_count * sizeof(void*);
  iree_amdgpu_memcpy(kernarg_ptr, cmd->constants,
                     dispatch_args->constant_count * sizeof(uint32_t));

  // Construct the dispatch packet based on the template embedded in the command
  // buffer. Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  iree_hsa_kernel_dispatch_packet_t* packet =
      iree_hal_amdgpu_device_cmd_resolve_dispatch_packet(state, packet_id);
  packet->setup = dispatch_args->setup;
  packet->workgroup_size[0] = dispatch_args->workgroup_size[0];
  packet->workgroup_size[1] = dispatch_args->workgroup_size[1];
  packet->workgroup_size[2] = dispatch_args->workgroup_size[2];
  packet->reserved0 = 0;
  packet->private_segment_size = dispatch_args->private_segment_size;
  packet->group_segment_size =
      dispatch_args->group_segment_size + cmd->config.dynamic_lds_size;
  packet->kernel_object = dispatch_args->kernel_object;
  packet->kernarg_address = kernarg_ptr;
  packet->reserved2 = 0;

  // Resolve the workgroup count (if possible).
  const uint32_t* workgroup_count_ptr = NULL;
  if (cmd->config.flags &
      IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_INDIRECT_STATIC) {
    // Workgroup count is indirect but statically available and can be resolved
    // during issue. This is the common case where the workgroup count is stored
    // in a uniform buffer by the launcher and it allows us to avoid any
    // additional dispatch overhead.
    workgroup_count_ptr =
        iree_hal_amdgpu_device_workgroup_count_buffer_ref_resolve(
            cmd->config.workgroup_count.ref, state->bindings);
  } else {
    // Workgroup count is constant.
    workgroup_count_ptr = cmd->config.workgroup_count.dims;
  }
  packet->grid_size[0] = workgroup_count_ptr[0] * packet->workgroup_size[0];
  packet->grid_size[1] = workgroup_count_ptr[1] * packet->workgroup_size[1];
  packet->grid_size[2] = workgroup_count_ptr[2] * packet->workgroup_size[2];

  // If the dispatch requires implicit args then populate them now.
  // Some of these are static and could be precomputed while others are
  // dependent on where we are running. We only need to produce until we meet
  // the implicit kernarg size (the remainder after our own explicit args).
  //
  // TODO(benvanik): once we have real kernels coming through see what we
  // actually use. We can also check if it makes sense to try to be efficient or
  // always splat in everything to keep things more uniform. The kernarg
  // reservation being larger than required is sad but so is the need to check
  // each field individually.
  //
  // Today we hardcode any kernel needing implicit args to attach a
  // IREE_AMDGPU_KERNEL_IMPLICIT_ARGS_SIZE-byte suffix that we initialize
  // unconditionally. Detecting what we need to populate is significantly more
  // expensive than just always reserving the space. Probably. This entire
  // design makes me sad as it's *never* required in our generated kernels and
  // the best we can hope for is that we don't accidentally use any device
  // library code that triggers the implicit args to be required.
  if (cmd->header.flags & IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_IMPLICIT_ARGS) {
    kernarg_ptr +=
        iree_amdgpu_align(dispatch_args->constant_count * sizeof(uint32_t), 8);
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args =
        (iree_amdgpu_kernel_implicit_args_t*)kernarg_ptr;
    // This information is redundant with the dispatch packet and it's sad that
    // it is required by HIP.
    implicit_args->block_count[0] = workgroup_count_ptr[0];
    implicit_args->block_count[1] = workgroup_count_ptr[1];
    implicit_args->block_count[2] = workgroup_count_ptr[2];
    implicit_args->group_size[0] = dispatch_args->workgroup_size[0];
    implicit_args->group_size[1] = dispatch_args->workgroup_size[1];
    implicit_args->group_size[2] = dispatch_args->workgroup_size[2];
    // Hardcoded to 0 in HIP.
    implicit_args->remainder[0] = 0;
    implicit_args->remainder[1] = 0;
    implicit_args->remainder[2] = 0;
    // Hardcoded to 0 in HIP.
    implicit_args->global_offset[0] = 0;
    implicit_args->global_offset[1] = 0;
    implicit_args->global_offset[2] = 0;
    // Hardcoded to 3 in HIP.
    implicit_args->grid_dims = 3;
    // TODO(benvanik): support printf_buffer (and maybe hostcall_buffer).
    // Today we set to NULL so we get a segfault if a kernel happens to use
    // them.
    implicit_args->printf_buffer = NULL;
    implicit_args->hostcall_buffer = NULL;
    // We don't currently use dynamic LDS but may allow HIP kernels to do so.
    implicit_args->dynamic_lds_size = cmd->config.dynamic_lds_size;
  }

  // NOTE: we return the packet without having updated the header. The caller
  // is responsible for calling iree_hal_amdgpu_device_cmd_dispatch_mark_ready
  // when it is ready for the hardware command processor to pick up the packet.
  return packet;
}

static void iree_hal_amdgpu_device_cmd_dispatch_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_dispatch_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Enqueue the dispatch packet but do not mark it as ready yet.
  uint8_t* kernarg_ptr = state->execution_kernarg_storage + cmd->kernarg_offset;
  iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet =
      iree_hal_amdgpu_device_cmd_dispatch_reserve(state, block, cmd,
                                                  kernarg_ptr, packet_id);

  // Mark the dispatch as complete and allow the hardware command processor to
  // process it.
  return iree_hal_amdgpu_device_cmd_commit_dispatch_packet(
      state, &cmd->header, packet,
      IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_DISPATCH,
      cmd->config.kernel_args->trace_src_loc, execution_query_id);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC
//===----------------------------------------------------------------------===//

IREE_AMDGPU_ATTRIBUTE_KERNEL IREE_AMDGPU_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_cmd_dispatch_update(
    const iree_hal_amdgpu_device_cmd_dispatch_t* IREE_AMDGPU_RESTRICT cmd,
    const uint32_t* IREE_AMDGPU_RESTRICT workgroups_ptr,
    iree_hsa_kernel_dispatch_packet_t* IREE_AMDGPU_RESTRICT packet) {
  // Read the uint32_t[3] workgroup count buffer and update the packet in-place.
  packet->grid_size[0] = workgroups_ptr[0] * packet->workgroup_size[0];
  packet->grid_size[1] = workgroups_ptr[1] * packet->workgroup_size[0];
  packet->grid_size[2] = workgroups_ptr[2] * packet->workgroup_size[0];

  // If the kernel has implicit args then update those as well.
  // This results from an unfortunate design decision and is only needed for
  // compatibility with HIP kernels that use the device library routines that
  // fetch implicit args instead of using the builtins that reference registers.
  if (cmd->header.flags & IREE_HAL_AMDGPU_DEVICE_DISPATCH_FLAG_IMPLICIT_ARGS) {
    const iree_hal_amdgpu_device_kernel_args_t* dispatch_args =
        cmd->config.kernel_args;
    uint8_t* implicit_args_ptr =
        (uint8_t*)packet->kernarg_address +
        dispatch_args->binding_count * sizeof(void*) +
        iree_amdgpu_align(dispatch_args->constant_count * sizeof(uint32_t), 8);
    iree_amdgpu_kernel_implicit_args_t* IREE_AMDGPU_RESTRICT implicit_args =
        (iree_amdgpu_kernel_implicit_args_t*)implicit_args_ptr;
    implicit_args->block_count[0] = workgroups_ptr[0];
    implicit_args->block_count[1] = workgroups_ptr[1];
    implicit_args->block_count[2] = workgroups_ptr[2];
  }

  // Now that the packet (and maybe kernargs) have been updated we can mark it
  // as ready so that the hardware command processor can take it. Since the
  // execution queue has already had its doorbell updated we don't need to do
  // that - it _should_ be spinning on the packet.
  const uint16_t header =
      (packet->header &
       ~(IREE_HSA_PACKET_TYPE_INVALID << IREE_HSA_PACKET_HEADER_TYPE)) |
      (IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH << IREE_HSA_PACKET_HEADER_TYPE);
  const uint32_t header_setup = header | (uint32_t)(packet->setup << 16);
  iree_amdgpu_scoped_atomic_store(
      (iree_amdgpu_scoped_atomic_uint32_t*)packet, header_setup,
      iree_amdgpu_memory_order_release, iree_amdgpu_memory_scope_device);
}

static void iree_hal_amdgpu_device_cmd_dispatch_indirect_dynamic_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_dispatch_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  const uint32_t update_id = packet_id;
  const uint32_t dispatch_id = update_id + 1;

  // Enqueue the dispatch packet but do not mark it as ready yet.
  // We do this first so that if the workgroup count update dispatch begins
  // executing while we're still running we want it to have valid data to
  // manipulate.
  uint8_t* IREE_AMDGPU_RESTRICT dispatch_kernarg_ptr =
      state->execution_kernarg_storage + cmd->kernarg_offset +
      IREE_HAL_AMDGPU_DEVICE_WORKGROUP_COUNT_UPDATE_KERNARG_SIZE;
  iree_hsa_kernel_dispatch_packet_t* dispatch_packet =
      iree_hal_amdgpu_device_cmd_dispatch_reserve(
          state, block, cmd, dispatch_kernarg_ptr, dispatch_id);

  // Update the dispatch packet with the required scheduling information.
  // It is not yet committed and will not have its type set until the indirect
  // update executes.
  iree_hal_amdgpu_device_cmd_update_dispatch_packet(
      state, &cmd->header, dispatch_packet,
      IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_DISPATCH_INDIRECT,
      cmd->config.kernel_args->trace_src_loc, execution_query_id);

  // Workgroup count is dynamic and must be resolved just prior to executing
  // the dispatch. There's no native AQL dispatch behavior to enable this so
  // we have to emulate it by enqueuing a builtin that performs the
  // indirection and overwrites the packet memory directly.
  uint64_t* IREE_AMDGPU_RESTRICT update_kernarg_ptr =
      (uint64_t*)(state->execution_kernarg_storage + cmd->kernarg_offset);
  update_kernarg_ptr[0] = (uint64_t)cmd;
  update_kernarg_ptr[1] =
      (uint64_t)iree_hal_amdgpu_device_workgroup_count_buffer_ref_resolve(
          cmd->config.workgroup_count.ref, state->bindings);
  update_kernarg_ptr[2] = (uint64_t)dispatch_packet;

  // Construct the update packet.
  // Note that the header is not written until the end so that the
  // hardware command processor stalls until we're done writing.
  const iree_hal_amdgpu_device_kernel_args_t update_args =
      state->kernels->iree_hal_amdgpu_device_cmd_dispatch_update;
  iree_hsa_kernel_dispatch_packet_t* update_packet =
      iree_hal_amdgpu_device_cmd_resolve_dispatch_packet(state, packet_id);
  update_packet->setup = update_args.setup;
  update_packet->workgroup_size[0] = update_args.workgroup_size[0];
  update_packet->workgroup_size[1] = update_args.workgroup_size[1];
  update_packet->workgroup_size[2] = update_args.workgroup_size[2];
  update_packet->reserved0 = 0;
  update_packet->grid_size[0] = 1;
  update_packet->grid_size[1] = 1;
  update_packet->grid_size[2] = 1;
  update_packet->private_segment_size = update_args.private_segment_size;
  update_packet->group_segment_size = update_args.group_segment_size;
  update_packet->kernel_object = update_args.kernel_object;
  update_packet->kernarg_address = update_kernarg_ptr;
  update_packet->reserved2 = 0;

  // Mark the update packet as ready to execute. The hardware command processor
  // may begin executing it immediately after performing the atomic swap.
  //
  // NOTE: the following dispatch packet is still marked INVALID and is only
  // changed after the update dispatch completes. The hardware command processor
  // should process the update (as we change it from INVALID here) and then
  // block before reading the contents of the dispatch packet.
  return iree_hal_amdgpu_device_cmd_commit_dispatch_packet(
      state, &cmd->header, update_packet,
      IREE_HAL_AMDGPU_TRACE_EXECUTION_ZONE_TYPE_INTERNAL, 0,
      IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH
//===----------------------------------------------------------------------===//

// Enqueues the command buffer rescheduling on the scheduler queue.
// The next tick will move the parent queue entry to the ready list and attempt
// to schedule the next block as set on the state.
IREE_AMDGPU_ATTRIBUTE_KERNEL IREE_AMDGPU_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_cmd_branch(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT
        next_block) {
  // Flush trace zones if any were used.
  // Note that this won't include this kernel as it is still running.
  // TODO(benvanik): find a way to include the time for the terminator.
  iree_hal_amdgpu_device_flush_execution_queries(state);

  // Set the target block of the branch.
  // When the execution state is rescheduled it will resume at the new block.
  state->block = next_block;

  // TODO(benvanik): evaluate or make a mode bit to control whether command
  // buffers yield for rescheduling or if they directly enqueue the issue block.
  // Rescheduling would allow for better QoS as older but newly-runnable entries
  // would be allowed to execute ahead of the continuation point. The downside
  // of rescheduling is that we'll have at least one hop to do the scheduler
  // tick and then one more to do the block issue.
  //
  // NOTE: the rescheduling may happen immediately and we cannot use any
  // execution state.
  const bool direct_issue_block = true;
  if (direct_issue_block) {
    // Enqueue the command buffer issue on the control queue.
    // It'll continue executing at the state block set above.
    return iree_hal_amdgpu_device_command_buffer_enqueue_next_block(state);
  } else {
    // Enqueue the parent queue scheduler tick.
    // It will move the queue entry to the ready list and may immediately begin
    // issuing the next block.
    return iree_hal_amdgpu_device_queue_scheduler_reschedule_from_execution_queue(
        state->scheduler, state->scheduler_queue_entry);
  }
}

static void iree_hal_amdgpu_device_cmd_branch_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_branch_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Direct branches are like tail calls and can simply begin issuing the
  // following block. The kernargs are stored in state->control_kernarg_storage
  // so that the issue_block can completely overwrite the values.
  // Command buffer issue has already bumped the write_index and all we need to
  // do is populate the packet.
  //
  // NOTE: we implicitly assume
  // IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER but need not do so
  // (technically) when continuing within the same command buffer. Performing a
  // barrier is a more conservative operation and may mask compiler/command
  // buffer construction issues with the more strict execution model but in
  // practice is unlikely to have an appreciable effect on latency.

  // Pass target block to the branch op.
  // For other CFG commands (like conditional branches) we pass the information
  // required to evaluate the condition and calculate the target block.
  uint64_t* kernarg_ptr =
      (uint64_t*)(state->execution_kernarg_storage + cmd->kernarg_offset);
  kernarg_ptr[0] = (uint64_t)state;
  kernarg_ptr[1] = (uint64_t)&state->command_buffer->blocks[cmd->target_block];

  // Emplace and ready the CFG packet.
  return iree_hal_amdgpu_device_cmd_commit_cfg_packet(
      state, &cmd->header, packet_id, execution_query_id,
      &state->kernels->iree_hal_amdgpu_device_cmd_branch, kernarg_ptr);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_COND_BRANCH
//===----------------------------------------------------------------------===//

static inline bool iree_hal_amdgpu_device_evaluate_cond(
    uint64_t lhs, iree_hal_amdgpu_device_cmd_cond_t cond, uint64_t rhs) {
  switch (cond) {
    default:
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_EQ:
      return (int64_t)lhs == (int64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_NE:
      return (int64_t)lhs != (int64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_SLT:
      return (int64_t)lhs < (int64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_SLE:
      return (int64_t)lhs <= (int64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_SGT:
      return (int64_t)lhs > (int64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_SGE:
      return (int64_t)lhs >= (int64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_ULT:
      return (uint64_t)lhs < (uint64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_ULE:
      return (uint64_t)lhs <= (uint64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_UGT:
      return (uint64_t)lhs > (uint64_t)rhs;
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_UGE:
      return (uint64_t)lhs >= (uint64_t)rhs;
  }
}

// Enqueues the command buffer rescheduling on the scheduler queue.
// The next tick will move the parent queue entry to the ready list and attempt
// to schedule the next block as set on the state. The block selected is based
// on the provided dynamic and static values:
//   next_block = *ref_ptr <cond> value ? true_block : false_block
IREE_AMDGPU_ATTRIBUTE_KERNEL IREE_AMDGPU_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_cmd_cond_branch(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const uint64_t* IREE_AMDGPU_RESTRICT ref_ptr,
    iree_hal_amdgpu_device_cmd_cond_t cond, uint64_t value,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT
        true_block,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT
        false_block) {
  // Flush trace zones if any were used.
  // Note that this won't include this kernel as it is still running.
  // TODO(benvanik): find a way to include the time for the terminator.
  iree_hal_amdgpu_device_flush_execution_queries(state);

  // Evaluate condition.
  //
  // TODO(benvanik): evaluate whether we should be doing this like a dynamic
  // indirect dispatch: since the AMD LLVMGPU backend (/hardware) is pretty
  // garbage at function calls having this kernel may cause a pretty extreme
  // amount of bloat as the compiler desperately tries to inline every function
  // to avoid dealing with the 👻 behavior of 👋 functions and stacks 👋. We
  // could instead always issue an unconditional branch but have the update
  // dispatch overwrite the kernarg for the `next_block` with the result of the
  // condition. This is a lot clearer and more flexible, though, so hopefully
  // that's not required. It's probably required. Sigh.
  const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT
      next_block = iree_hal_amdgpu_device_evaluate_cond(ref_ptr[0], cond, value)
                       ? true_block
                       : false_block;

  // Set the target block of the branch.
  // When the execution state is rescheduled it will resume at the new block.
  state->block = next_block;

  // TODO(benvanik): evaluate or make a mode bit to control whether command
  // buffers yield for rescheduling or if they directly enqueue the issue block.
  // Rescheduling would allow for better QoS as older but newly-runnable entries
  // would be allowed to execute ahead of the continuation point. The downside
  // of rescheduling is that we'll have at least one hop to do the scheduler
  // tick and then one more to do the block issue.
  //
  // NOTE: the rescheduling may happen immediately and we cannot use any
  // execution state.
  const bool direct_issue_block = true;
  if (direct_issue_block) {
    // Enqueue the command buffer issue on the control queue.
    // It'll continue executing at the state block set above.
    return iree_hal_amdgpu_device_command_buffer_enqueue_next_block(state);
  } else {
    // Enqueue the parent queue scheduler tick.
    // It will move the queue entry to the ready list and may immediately begin
    // issuing the next block.
    return iree_hal_amdgpu_device_queue_scheduler_reschedule_from_execution_queue(
        state->scheduler, state->scheduler_queue_entry);
  }
}

static void iree_hal_amdgpu_device_cmd_cond_branch_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_cond_branch_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // Direct branches are like tail calls and can simply begin issuing the
  // following block. The kernargs are stored in state->control_kernarg_storage
  // so that the issue_block can completely overwrite the values.
  // Command buffer issue has already bumped the write_index and all we need to
  // do is populate the packet.
  //
  // NOTE: we implicitly assume
  // IREE_HAL_AMDGPU_DEVICE_CMD_FLAG_QUEUE_AWAIT_BARRIER but need not do so
  // (technically) when continuing within the same command buffer. Performing a
  // barrier is a more conservative operation and may mask compiler/command
  // buffer construction issues with the more strict execution model but in
  // practice is unlikely to have an appreciable effect on latency.

  // Pass the condition, its inputs, and the true/false blocks to the kernel.
  // Which block is evaluated at the time the command is executed but since
  // block pointers are immutable we can resolve them now to make the kernel
  // simpler.
  uint64_t* kernarg_ptr =
      (uint64_t*)(state->execution_kernarg_storage + cmd->kernarg_offset);
  kernarg_ptr[0] = (uint64_t)state;
  kernarg_ptr[1] = (uint64_t)iree_hal_amdgpu_device_uint64_buffer_ref_resolve(
      cmd->ref, state->bindings);
  kernarg_ptr[2] = cmd->cond;
  kernarg_ptr[3] = cmd->value;
  kernarg_ptr[4] = (uint64_t)&state->command_buffer->blocks[cmd->true_block];
  kernarg_ptr[5] = (uint64_t)&state->command_buffer->blocks[cmd->false_block];

  // Emplace and ready the CFG packet.
  return iree_hal_amdgpu_device_cmd_commit_cfg_packet(
      state, &cmd->header, packet_id, execution_query_id,
      &state->kernels->iree_hal_amdgpu_device_cmd_cond_branch, kernarg_ptr);
}

//===----------------------------------------------------------------------===//
// IREE_HAL_AMDGPU_DEVICE_CMD_RETURN
//===----------------------------------------------------------------------===//

// Enqueues the command buffer retirement on the parent scheduler.
// The execution state may be deallocated immediately.
IREE_AMDGPU_ATTRIBUTE_KERNEL IREE_AMDGPU_ATTRIBUTE_SINGLE_WORK_ITEM void
iree_hal_amdgpu_device_cmd_return(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state) {
  // Flush trace zones if any were used.
  // Note that this won't include this kernel as it is still running.
  // TODO(benvanik): find a way to include the time for the terminator.
  iree_hal_amdgpu_device_flush_execution_queries(state);

  // Clear block to indicate execution has completed.
  //
  // TODO(benvanik): does this need to be atomic release?
  state->block = NULL;

  // Enqueue the parent queue scheduler tick.
  // It will clean up the command buffer execution state and resume
  // processing queue entries.
  //
  // NOTE: the retire may immediately reclaim the execution state and we cannot
  // do anything else with it.
  return iree_hal_amdgpu_device_queue_scheduler_retire_from_execution_queue(
      state->scheduler, state->scheduler_queue_entry);
}

static void iree_hal_amdgpu_device_cmd_return_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const iree_hal_amdgpu_device_cmd_return_t* IREE_AMDGPU_RESTRICT cmd,
    const uint64_t packet_id,
    const iree_hal_amdgpu_trace_execution_query_id_t execution_query_id) {
  // TODO(benvanik): handle call stacks when nesting command buffers. For now a
  // return is always going back to the queue scheduler and can be enqueued as
  // such.

  // Pass just the state; the returns will always return to the scheduler.
  uint64_t* kernarg_ptr =
      (uint64_t*)(state->execution_kernarg_storage + cmd->kernarg_offset);
  kernarg_ptr[0] = (uint64_t)state;

  // Emplace and ready the CFG packet.
  return iree_hal_amdgpu_device_cmd_commit_cfg_packet(
      state, &cmd->header, packet_id, execution_query_id,
      &state->kernels->iree_hal_amdgpu_device_cmd_return, kernarg_ptr);
}

//===----------------------------------------------------------------------===//
// Command Issue
//===----------------------------------------------------------------------===//

// Issues a block of commands in parallel.
// Each work item processes a single command. Each command in the block contains
// a relative offset into the queue where AQL packets should be placed and must
// fill all packets that were declared when the command buffer was recorded
// (even if they are no-oped).
//
// This relies on the AQL queue mechanics defined in section 2.8.3 of the HSA
// System Architecture Specification. The parent enqueuing this kernel reserves
// sufficient queue space for all AQL packets and bumps the write_index to the
// end of the block. Each command processed combines the base queue index
// provided with the per-command relative offset and performs the required queue
// masking to get the final packet pointer. Packets are written by populating
// all kernel arguments (if any), populating the packet fields, and finally
// atomically changing the packet type from INVALID to (likely) KERNEL_DISPATCH.
// Even though the write_index of the queue was bumped to the end the queue
// processor is required to block on the first packet it finds with an INVALID
// type and as such we don't require ordering guarantees on the packet
// population. It's of course better if the first packet complete population
// first so that the queue processor can launch it and that will often be the
// case given that HSA mandates that workgroups with lower indices are scheduled
// to resources before those with higher ones.
//
// Of course, the spec _could_ be wrong and not match reality. At this point I'm
// _sure_ it doesn't match on many closely related aspects. Let's see what we
// can get away with for the next short while. I immediately recognize I will
// regret both making this implementation decision and writing these words.
// Future me: deal ;) Workarounds are to issue all from a single thread or issue
// with chunked reservations per thread. We could also do an atomic MAX on the
// write index to keep the window between nearly-populated packets (those who
// have bumped the write index but not yet changed from INVALID) and
// fully-populated packets (with INVALID transitioned to a packet type) as small
// as possible, however PCIe atomics don't support MAX and though we _shouldn't_
// need that here (device->device) it does close off some design space we may
// need to explore to get around _other_ wonkiness in various hardware (I'm
// looking at you, HDP ಠ_ಠ).
IREE_AMDGPU_ATTRIBUTE_KERNEL void iree_hal_amdgpu_device_cmd_block_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    const uint64_t base_packet_id) {
  // Each invocation handles a single command in the block.
  const uint32_t command_ordinal = iree_hal_amdgpu_device_global_id_x();
  if (command_ordinal >= block->command_count) return;
  return iree_hal_amdgpu_device_cmd_issue(state, block, command_ordinal,
                                          base_packet_id);  // tail ideal
}

// Issues a single command packet in a block.
//
// NOTE: this should be tail-called to avoid non-trivial stack management
// overhead (as the AMD LLVMGPU backend is very poor at function calls). Not
// correctly tail-calling such that LLVM can recognize it can easily double
// binary size. Unfortunately today this often doesn't happen due to ABI
// mismatch (the caller is device_kernel, the target is cdecl). We should find a
// way to make these line up so the caller can turn into a jump instead of
// having to deal with 👻 function calls 👻.
static void iree_hal_amdgpu_device_cmd_issue(
    iree_hal_amdgpu_device_execution_state_t* IREE_AMDGPU_RESTRICT state,
    const iree_hal_amdgpu_device_command_block_t* IREE_AMDGPU_RESTRICT block,
    uint32_t command_ordinal, uint64_t base_packet_id) {
  // When device control or dispatch tracing is enabled we need to pass a query
  // signal with any work we do. Prior to the block starting execution we
  // acquire a range for all commands on the scheduler queue and store it in
  // state->trace_block_query_base_id. Here we then take that base ID and add a
  // relative offset that was precomputed when the command buffer was recorded.
  // This allows us to support sparse/partial queries and still issue in
  // parallel while respecting the required query ordering.
  //
  // There's probably a much simpler way of doing this - not needing all this
  // branching per command or the precomputed query map would be nice.
  iree_hal_amdgpu_trace_execution_query_id_t execution_query_id =
      IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID;
#if IREE_HAL_AMDGPU_HAS_TRACING_FEATURE( \
    IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL)
  const iree_hal_amdgpu_device_command_query_id_t command_query_id =
      block->query_map.query_ids[command_ordinal];
  if ((state->flags & IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_DISPATCH) &&
      command_query_id.dispatch_id !=
          IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    execution_query_id = iree_hal_amdgpu_device_query_ringbuffer_query_id(
        &state->trace_buffer->query_ringbuffer,
        state->trace_block_query_base_id + command_query_id.dispatch_id);
  } else if ((state->flags &
              IREE_HAL_AMDGPU_DEVICE_EXECUTION_FLAG_TRACE_CONTROL) &&
             command_query_id.control_id !=
                 IREE_HAL_AMDGPU_TRACE_EXECUTION_QUERY_ID_INVALID) {
    execution_query_id = iree_hal_amdgpu_device_query_ringbuffer_query_id(
        &state->trace_buffer->query_ringbuffer,
        state->trace_block_query_base_id + command_query_id.control_id);
  }
#endif  // IREE_HAL_AMDGPU_TRACING_FEATURE_DEVICE_CONTROL

  // Tail-call into the command handler.
  const iree_hal_amdgpu_device_cmd_t* IREE_AMDGPU_RESTRICT cmd =
      &block->commands[command_ordinal];
  const uint64_t packet_id = base_packet_id + cmd->header.packet_offset;
  switch (cmd->header.type) {
    default:
      return;  // no-op
    case IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_BEGIN:
      return iree_hal_amdgpu_device_cmd_debug_group_begin_issue(
          state, block,
          (const iree_hal_amdgpu_device_cmd_debug_group_begin_t*)cmd, packet_id,
          execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_DEBUG_GROUP_END:
      return iree_hal_amdgpu_device_cmd_debug_group_end_issue(
          state, block,
          (const iree_hal_amdgpu_device_cmd_debug_group_end_t*)cmd, packet_id,
          execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_BARRIER:
      return iree_hal_amdgpu_device_cmd_barrier_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_barrier_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_SIGNAL_EVENT:
      return iree_hal_amdgpu_device_cmd_signal_event_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_signal_event_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_RESET_EVENT:
      return iree_hal_amdgpu_device_cmd_reset_event_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_reset_event_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_WAIT_EVENTS:
      return iree_hal_amdgpu_device_cmd_wait_events_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_wait_events_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_FILL_BUFFER:
      return iree_hal_amdgpu_device_cmd_fill_buffer_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_fill_buffer_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_COPY_BUFFER:
      return iree_hal_amdgpu_device_cmd_copy_buffer_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_copy_buffer_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH:
      return iree_hal_amdgpu_device_cmd_dispatch_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_dispatch_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_DISPATCH_INDIRECT_DYNAMIC:
      return iree_hal_amdgpu_device_cmd_dispatch_indirect_dynamic_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_dispatch_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_BRANCH:
      return iree_hal_amdgpu_device_cmd_branch_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_branch_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_COND_BRANCH:
      return iree_hal_amdgpu_device_cmd_cond_branch_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_cond_branch_t*)cmd,
          packet_id, execution_query_id);
    case IREE_HAL_AMDGPU_DEVICE_CMD_RETURN:
      return iree_hal_amdgpu_device_cmd_return_issue(
          state, block, (const iree_hal_amdgpu_device_cmd_return_t*)cmd,
          packet_id, execution_query_id);
  }
  // NOTE: we need the above switch to end in tail calls in all cases. It
  // doesn't today. But it should. If the stars align and we can make that
  // happen it eliminates the (currently) only requirement for expanded shared
  // memory faults in the entire library. At last check it also would halve the
  // binary size: the AMD LLVMGPU backend inlines this function in any kernel
  // that performs an issue.
}
