// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_block.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/aql_block_processor.h"
#include "iree/hal/drivers/amdgpu/aql_block_processor_profile.h"
#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/timestamp.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_packet.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_command_buffer_scratch.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_timestamp.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/utils/resource_set.h"

static iree_status_t iree_hal_amdgpu_host_queue_ensure_command_buffer_scratch(
    iree_hal_amdgpu_host_queue_t* queue) {
  if (queue->command_buffer_scratch) return iree_ok_status();
  iree_hal_amdgpu_host_queue_command_buffer_scratch_t* scratch = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      queue->host_allocator, sizeof(*scratch), (void**)&scratch));
  memset(scratch, 0, sizeof(*scratch));
  queue->command_buffer_scratch = scratch;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_binding_base_ptr(
    const iree_hal_buffer_binding_t* binding, uint64_t* out_binding_ptr) {
  *out_binding_ptr = 0;
  if (IREE_UNLIKELY(!binding->buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "dispatch binding table entry is NULL");
  }
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(binding->buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  const iree_device_size_t binding_length =
      binding->length == IREE_HAL_WHOLE_BUFFER ? 0 : binding->length;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      binding->buffer, binding->offset, binding_length));

  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(binding->buffer);
  void* device_ptr = iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "dispatch binding table entry must be backed by an AMDGPU allocation");
  }
  iree_device_size_t device_offset = 0;
  if (IREE_UNLIKELY(!iree_device_size_checked_add(
          iree_hal_buffer_byte_offset(binding->buffer), binding->offset,
          &device_offset))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch binding table device pointer offset overflows device size");
  }
  if (IREE_UNLIKELY(device_offset > UINTPTR_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch binding table device pointer offset exceeds host pointer "
        "size");
  }
  *out_binding_ptr = (uint64_t)((uintptr_t)device_ptr + device_offset);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_binding_ptrs(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_arena_allocator_t* overflow_arena, const uint64_t** out_binding_ptrs) {
  *out_binding_ptrs = NULL;
  const uint32_t binding_count = command_buffer->binding_count;
  if (binding_count == 0) return iree_ok_status();
  uint64_t* binding_ptrs = NULL;
  if (binding_count <=
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_BINDING_SCRATCH_CAPACITY) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_ensure_command_buffer_scratch(queue));
    binding_ptrs = queue->command_buffer_scratch->bindings.ptrs;
  } else {
    iree_host_size_t binding_ptr_bytes = 0;
    IREE_RETURN_IF_ERROR(
        IREE_STRUCT_LAYOUT(0, &binding_ptr_bytes,
                           IREE_STRUCT_FIELD(binding_count, uint64_t, NULL)));
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, binding_count);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(overflow_arena, binding_ptr_bytes,
                                (void**)&binding_ptrs));
    IREE_TRACE_ZONE_END(z0);
  }

  iree_status_t status =
      iree_hal_amdgpu_host_queue_resolve_command_buffer_binding_ptrs(
          command_buffer, binding_table, binding_ptrs);
  if (iree_status_is_ok(status)) {
    *out_binding_ptrs = binding_ptrs;
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_resolve_command_buffer_binding_ptrs(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table, uint64_t* out_binding_ptrs) {
  const uint32_t binding_count = command_buffer->binding_count;
  if (binding_count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(binding_table.count < binding_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "queue_execute binding table count %" PRIhsz
                            " is smaller than command-buffer binding count %u",
                            binding_table.count, binding_count);
  }
  if (IREE_UNLIKELY(!binding_table.bindings)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "queue_execute binding table storage is NULL for %u bindings",
        binding_count);
  }

  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < binding_count && iree_status_is_ok(status); ++i) {
    status = iree_hal_amdgpu_host_queue_resolve_binding_base_ptr(
        &binding_table.bindings[i], &out_binding_ptrs[i]);
    if (!iree_status_is_ok(status)) {
      status = iree_status_annotate_f(status, "binding_table[%" PRIu32 "]", i);
    }
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_prepare_command_buffer_packet_metadata(
    iree_hal_amdgpu_host_queue_t* queue, uint32_t packet_count,
    iree_arena_allocator_t* scratch_arena, uint16_t** out_packet_headers,
    uint16_t** out_packet_setups) {
  *out_packet_headers = NULL;
  *out_packet_setups = NULL;

  uint16_t* packet_headers = NULL;
  uint16_t* packet_setups = NULL;
  if (packet_count <=
      IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_SCRATCH_CAPACITY) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_ensure_command_buffer_scratch(queue));
    packet_headers = queue->command_buffer_scratch->packets.headers;
    packet_setups = queue->command_buffer_scratch->packets.setups;
  } else {
    iree_host_size_t packet_metadata_bytes = 0;
    IREE_RETURN_IF_ERROR(
        IREE_STRUCT_LAYOUT(0, &packet_metadata_bytes,
                           IREE_STRUCT_FIELD(packet_count, uint16_t, NULL)));
    IREE_TRACE_ZONE_BEGIN(z0);
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, packet_count);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(scratch_arena, packet_metadata_bytes,
                                (void**)&packet_headers));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(scratch_arena, packet_metadata_bytes,
                                (void**)&packet_setups));
    IREE_TRACE_ZONE_END(z0);
  }

  // The block processors overwrite every reserved metadata slot before
  // reporting success; avoid pre-zeroing this hot submission span.
  *out_packet_headers = packet_headers;
  *out_packet_setups = packet_setups;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
    const iree_hal_amdgpu_wait_resolution_t* resolution, uint32_t packet_index,
    iree_hal_amdgpu_host_queue_command_buffer_packet_flags_t packet_flags) {
  return iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_EXECUTION_BARRIER) ||
         iree_any_bit_set(
             packet_flags,
             IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL) ||
         (packet_index == 0 && resolution->barrier_count > 0) ||
         (packet_index == 0 &&
          resolution->inline_acquire_scope != IREE_HSA_FENCE_SCOPE_NONE);
}

static iree_hsa_fence_scope_t
iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_scope(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  if (block->kernarg_length == 0) return IREE_HSA_FENCE_SCOPE_NONE;
  return iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
      queue, IREE_HSA_FENCE_SCOPE_NONE);
}

static uint32_t
iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_packet_count(
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint32_t packet_index_base, iree_hsa_fence_scope_t payload_acquire_scope) {
  if (payload_acquire_scope == IREE_HSA_FENCE_SCOPE_NONE) return 0;
  if (block->aql_packet_count == 0) return 0;
  if (iree_hal_amdgpu_host_queue_command_buffer_packet_has_barrier(
          resolution, packet_index_base,
          block->aql_packet_count == 1
              ? IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL
              : IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE)) {
    return 1;
  }
  return block->initial_barrier_packet_count;
}

static uint32_t iree_hal_amdgpu_host_queue_aql_packet_header_field(
    uint16_t header, uint32_t field_shift, uint32_t field_width) {
  return (header >> field_shift) & ((1u << field_width) - 1u);
}

static iree_hsa_packet_type_t iree_hal_amdgpu_host_queue_aql_packet_header_type(
    uint16_t header) {
  return (iree_hsa_packet_type_t)
      iree_hal_amdgpu_host_queue_aql_packet_header_field(
          header, IREE_HSA_PACKET_HEADER_TYPE,
          IREE_HSA_PACKET_HEADER_WIDTH_TYPE);
}

static iree_status_t iree_hal_amdgpu_host_queue_write_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    const uint64_t* binding_ptrs, uint64_t first_payload_packet_id,
    uint32_t packet_index_base, iree_hal_amdgpu_kernarg_block_t* kernarg_blocks,
    uint16_t* packet_headers, uint16_t* packet_setups,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    uint32_t emitted_packet_count, uint32_t profile_counter_set_count,
    uint32_t profile_trace_packet_count,
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources,
    iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t
        profile_dispatches) {
  const iree_hsa_fence_scope_t payload_acquire_scope =
      iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_scope(
          queue, block);
  const bool use_base_processor = profile_events.event_count == 0 &&
                                  profile_counter_set_count == 0 &&
                                  profile_trace_packet_count == 0;
  if (use_base_processor) {
    iree_hal_amdgpu_aql_block_processor_t processor;
    const iree_hal_amdgpu_aql_block_processor_t processor_params = {
        .transfer_context = queue->transfer_context,
        .command_buffer = command_buffer,
        .bindings =
            {
                .table = binding_table,
                .ptrs = binding_ptrs,
            },
        .packets =
            {
                .ring = &queue->aql_ring,
                .first_id = first_payload_packet_id,
                .index_base = packet_index_base,
                .count = emitted_packet_count,
                .headers = packet_headers,
                .setups = packet_setups,
            },
        .kernargs =
            {
                .blocks = kernarg_blocks,
                .count = (uint32_t)iree_host_size_ceil_div(
                    block->kernarg_length,
                    sizeof(iree_hal_amdgpu_kernarg_block_t)),
            },
        .submission =
            {
                .wait_barrier_count = resolution->barrier_count,
                .inline_acquire_scope = resolution->inline_acquire_scope,
                .signal_release_scope =
                    iree_hal_amdgpu_host_queue_signal_list_release_scope(
                        queue, signal_semaphore_list),
            },
        .payload =
            {
                .acquire_scope = payload_acquire_scope,
            },
        .flags =
            packet_index_base == 0
                ? IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_FINAL_PAYLOAD_PACKET
                : IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_FLAG_NONE,
    };
    iree_hal_amdgpu_aql_block_processor_initialize(&processor_params,
                                                   &processor);
    iree_hal_amdgpu_aql_block_processor_result_t result;
    iree_status_t status =
        iree_hal_amdgpu_aql_block_processor_invoke(&processor, block, &result);
    iree_hal_amdgpu_aql_block_processor_deinitialize(&processor);
    return status;
  }
  // Per-dispatch counter and trace packets are emitted before the recorded
  // payload packet they wrap. Do not let a submit-time barrier on logical
  // packet 0 shrink the recorded payload acquire span when that first logical
  // packet is profiling metadata instead of the recorded payload stream.
  const uint32_t first_recorded_packet_index_base =
      profile_counter_set_count == 0 && profile_trace_packet_count == 0
          ? packet_index_base
          : 1u;
  const uint32_t payload_acquire_packet_count =
      iree_hal_amdgpu_host_queue_command_buffer_block_payload_acquire_packet_count(
          resolution, block, first_recorded_packet_index_base,
          payload_acquire_scope);
  iree_hal_amdgpu_aql_block_processor_profile_flags_t profile_flags =
      IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_NONE;
  if (profile_events.event_count != 0) {
    profile_flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_DISPATCH_PACKETS;
  }
  if (packet_index_base != 0) {
    profile_flags |=
        IREE_HAL_AMDGPU_AQL_BLOCK_PROCESSOR_PROFILE_FLAG_QUEUE_DEVICE_EVENT;
  }
  iree_hal_amdgpu_aql_block_processor_profile_t processor;
  const iree_hal_amdgpu_aql_block_processor_profile_t processor_params = {
      .queue = queue,
      .command_buffer = command_buffer,
      .block = block,
      .submission =
          {
              .resolution = resolution,
              .signal_semaphore_list = signal_semaphore_list,
          },
      .bindings =
          {
              .table = binding_table,
              .ptrs = binding_ptrs,
          },
      .packets =
          {
              .first_payload_id = first_payload_packet_id,
              .index_base = packet_index_base,
              .count = emitted_packet_count,
              .headers = packet_headers,
              .setups = packet_setups,
          },
      .kernargs =
          {
              .blocks = kernarg_blocks,
              .count = (uint32_t)iree_host_size_ceil_div(
                  block->kernarg_length,
                  sizeof(iree_hal_amdgpu_kernarg_block_t)),
          },
      .payload =
          {
              .acquire_scope = payload_acquire_scope,
              .acquire_packet_count = payload_acquire_packet_count,
          },
      .profile =
          {
              .dispatches = profile_dispatches,
              .dispatch_events = profile_events,
              .harvest_sources = profile_harvest_sources,
              .command_buffer_id =
                  iree_hal_amdgpu_aql_command_buffer_profile_id(command_buffer),
              .counter_set_count = profile_counter_set_count,
              .trace_packet_count = profile_trace_packet_count,
          },
      .flags = profile_flags,
  };
  iree_hal_amdgpu_aql_block_processor_profile_initialize(&processor_params,
                                                         &processor);
  iree_hal_amdgpu_aql_block_processor_profile_result_t result;
  iree_status_t status =
      iree_hal_amdgpu_aql_block_processor_profile_invoke(&processor, &result);
  iree_hal_amdgpu_aql_block_processor_profile_deinitialize(&processor);
  return status;
}

typedef uint32_t
    iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flags_t;
enum iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_NONE = 0u,
  // The non-profiling path needs a trailing barrier packet to own queue
  // completion.
  IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET =
      1u << 0,
};

typedef struct iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t {
  // Reserved dispatch timestamp events for profiled commands in this block.
  iree_hal_amdgpu_profile_dispatch_event_reservation_t dispatch_events;
  // Reserved whole-block queue-device timestamp event for this execute.
  iree_hal_amdgpu_profile_queue_device_event_reservation_t queue_device_events;
  // Host queue event metadata shared by host and device queue-event records.
  iree_hal_amdgpu_host_queue_profile_event_info_t queue_event_info;
  // Optional harvest dispatch emitted after profiled dispatch payloads.
  struct {
    // Explicit queue barrier packet emitted immediately before |packet|.
    iree_hal_amdgpu_aql_packet_t* barrier_packet;
    // Dispatch packet for harvesting dispatch timestamp records.
    iree_hal_amdgpu_aql_packet_t* packet;
    // Setup bits for |packet| when present.
    uint16_t setup;
  } harvest;
  // Flags from
  // iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flag_bits_t.
  iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flags_t flags;
} iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t;

// Publishes the non-profiling terminal barrier packet for a replayed
// command-buffer block. Payload packets keep their recorded final-payload
// barriers, but queue completion is signaled from this trailing packet so
// software observes block completion only after the CP reaches the end of the
// replay span.
static void iree_hal_amdgpu_host_queue_commit_command_buffer_completion_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list, uint64_t packet_id,
    uint32_t packet_index) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const uint16_t header = iree_hal_amdgpu_aql_emit_nop(
      &packet->barrier_and,
      iree_hal_amdgpu_host_queue_command_buffer_packet_control(
          queue, resolution, signal_semaphore_list, packet_index,
          IREE_HSA_FENCE_SCOPE_NONE,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL),
      iree_hal_amdgpu_notification_ring_epoch_signal(
          &queue->notification_ring));
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
}

static uint64_t iree_hal_amdgpu_host_queue_finish_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t emitted_packet_count,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_kernel_submission_t* submission,
    const uint16_t* packet_headers, const uint16_t* packet_setups,
    iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t* profile) {
  submission->pre_signal_action = pre_signal_action;
  iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, resolution,
                                                           submission);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_kernel_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, inout_binding_resource_set,
          submission_flags, submission);

  iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
      iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
          queue, profile->queue_device_events, &profile->queue_event_info);
  if (queue_device_event) {
    submission->reclaim_entry->queue_device_event_first_position =
        profile->queue_device_events.first_event_position;
    submission->reclaim_entry->queue_device_event_count =
        profile->queue_device_events.event_count;
    queue_device_event->submission_id = submission_epoch;
  }

  uint16_t profile_harvest_header = 0;
  if (profile->dispatch_events.event_count != 0) {
    submission->reclaim_entry->profile_event_first_position =
        profile->dispatch_events.first_event_position;
    submission->reclaim_entry->profile_event_count =
        profile->dispatch_events.event_count;
    for (uint32_t i = 0; i < profile->dispatch_events.event_count; ++i) {
      iree_hal_amdgpu_profile_dispatch_event_t* event =
          iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
              queue, profile->dispatch_events.first_event_position + i);
      event->submission_id = submission_epoch;
    }
    profile->harvest.packet->dispatch.completion_signal =
        queue_device_event ? iree_hsa_signal_null()
                           : iree_hal_amdgpu_notification_ring_epoch_signal(
                                 &queue->notification_ring);
    const iree_hsa_fence_scope_t profile_harvest_acquire_scope =
        iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
            queue, IREE_HSA_FENCE_SCOPE_AGENT);
    profile_harvest_header = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                profile_harvest_acquire_scope,
                resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_SYSTEM));
  }

  const uint32_t profile_queue_device_prefix_packet_count =
      queue_device_event ? 1u : 0u;
  const uint32_t profile_harvest_packet_count =
      profile->dispatch_events.event_count != 0 ? 2u : 0u;
  const uint64_t first_payload_packet_id =
      submission->first_packet_id + resolution->barrier_count +
      profile_queue_device_prefix_packet_count;
  iree_hal_amdgpu_host_queue_publish_submission_kernargs(queue, submission);
  if (queue_device_event) {
    const uint64_t start_packet_id =
        submission->first_packet_id + resolution->barrier_count;
    iree_hal_amdgpu_host_queue_commit_timestamp_start(
        queue, start_packet_id,
        iree_hal_amdgpu_host_queue_command_buffer_packet_control(
            queue, resolution, signal_semaphore_list, /*packet_index=*/0,
            IREE_HSA_FENCE_SCOPE_NONE,
            IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_NONE),
        &queue_device_event->start_tick);
  }

  for (uint32_t i = 0; i < emitted_packet_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
        &queue->aql_ring, first_payload_packet_id + i);
    if (iree_hal_amdgpu_host_queue_aql_packet_header_type(packet_headers[i]) !=
        IREE_HSA_PACKET_TYPE_INVALID) {
      iree_hal_amdgpu_aql_ring_commit(packet, packet_headers[i],
                                      packet_setups[i]);
    }
  }
  if (profile->dispatch_events.event_count != 0) {
    const uint16_t profile_harvest_barrier_header =
        iree_hal_amdgpu_aql_emit_nop(
            &profile->harvest.barrier_packet->barrier_and,
            iree_hal_amdgpu_aql_packet_control_barrier(
                IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_NONE),
            iree_hsa_signal_null());
    iree_hal_amdgpu_aql_ring_commit(profile->harvest.barrier_packet,
                                    profile_harvest_barrier_header,
                                    /*setup=*/0);
    iree_hal_amdgpu_aql_ring_commit(profile->harvest.packet,
                                    profile_harvest_header,
                                    profile->harvest.setup);
  }
  if (iree_any_bit_set(
          profile->flags,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET)) {
    const uint64_t completion_packet_id = first_payload_packet_id +
                                          emitted_packet_count +
                                          profile_harvest_packet_count;
    const uint32_t completion_packet_index =
        profile_queue_device_prefix_packet_count + emitted_packet_count +
        profile_harvest_packet_count;
    iree_hal_amdgpu_host_queue_commit_command_buffer_completion_packet(
        queue, resolution, signal_semaphore_list, completion_packet_id,
        completion_packet_index);
  }
  if (queue_device_event) {
    const uint64_t end_packet_id = first_payload_packet_id +
                                   emitted_packet_count +
                                   profile_harvest_packet_count;
    const uint32_t end_packet_index = profile_queue_device_prefix_packet_count +
                                      emitted_packet_count +
                                      profile_harvest_packet_count;
    iree_hal_amdgpu_host_queue_commit_timestamp_end(
        queue, end_packet_id,
        iree_hal_amdgpu_host_queue_command_buffer_packet_control(
            queue, resolution, signal_semaphore_list, end_packet_index,
            IREE_HSA_FENCE_SCOPE_NONE,
            IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PACKET_FLAG_FINAL),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring),
        &queue_device_event->end_tick);
  }
  iree_hal_amdgpu_aql_ring_doorbell(
      &queue->aql_ring,
      submission->first_packet_id + submission->packet_count - 1);
  profile->queue_event_info.submission_id = submission_epoch;
  memset(submission, 0, sizeof(*submission));
  return submission_epoch;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table, const uint64_t* binding_ptrs,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_resource_set_t** inout_binding_resource_set,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready) {
  *out_ready = false;
  const uint64_t command_buffer_id =
      iree_hal_amdgpu_aql_command_buffer_profile_id(command_buffer);
  const uint32_t kernarg_block_count = (uint32_t)iree_host_size_ceil_div(
      block->kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(queue->block_pool, &scratch_arena);
  iree_hal_amdgpu_aql_block_processor_profile_dispatch_list_t
      profile_dispatches = {0};
  iree_status_t status = iree_ok_status();
  if (command_buffer_id != 0 && queue->profiling.dispatch_profiling_enabled) {
    status =
        iree_hal_amdgpu_host_queue_select_command_buffer_profile_dispatches(
            queue, command_buffer, block, &scratch_arena, &profile_dispatches);
    if (!iree_status_is_ok(status)) {
      iree_arena_deinitialize(&scratch_arena);
      return status;
    }
  }
  const uint32_t profile_dispatch_event_count = profile_dispatches.count;
  iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events = {0};
  if (profile_dispatch_event_count != 0) {
    status = iree_hal_amdgpu_host_queue_reserve_profile_dispatch_events(
        queue, profile_dispatch_event_count, &profile_events);
    if (!iree_status_is_ok(status)) {
      iree_arena_deinitialize(&scratch_arena);
      return status;
    }
  }
  const uint32_t profile_counter_set_count =
      profile_events.event_count != 0
          ? iree_hal_amdgpu_host_queue_profile_counter_set_count(queue,
                                                                 profile_events)
          : 0u;
  const uint32_t profile_counter_packet_count =
      profile_events.event_count != 0
          ? iree_hal_amdgpu_host_queue_profile_counter_packet_count(
                queue, profile_events)
          : 0u;
  const uint32_t profile_trace_packet_count =
      profile_events.event_count != 0
          ? iree_hal_amdgpu_host_queue_profile_trace_packet_count(
                queue, profile_events)
          : 0u;
  if (IREE_UNLIKELY(profile_trace_packet_count >
                    UINT32_MAX - profile_counter_packet_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_arena_deinitialize(&scratch_arena);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profiled command-buffer block packet count overflow");
  }
  const uint32_t extra_profile_packet_count =
      profile_counter_packet_count + profile_trace_packet_count;
  if (IREE_UNLIKELY(block->aql_packet_count >
                    UINT32_MAX - extra_profile_packet_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_arena_deinitialize(&scratch_arena);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profiled command-buffer block packet count overflow");
  }
  const uint32_t emitted_packet_count =
      block->aql_packet_count + extra_profile_packet_count;
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events = {0};
  if (iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue)) {
    iree_status_t reserve_status =
        iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
            queue, /*event_count=*/1, &profile_queue_device_events);
    if (!iree_status_is_ok(reserve_status)) {
      iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                                profile_events);
      iree_arena_deinitialize(&scratch_arena);
      return reserve_status;
    }
  }
  const uint32_t profile_harvest_packet_count =
      profile_events.event_count != 0 ? 2u : 0u;
  const uint32_t profile_queue_device_packet_count =
      profile_queue_device_events.event_count != 0 ? 2u : 0u;
  iree_hal_amdgpu_host_queue_command_buffer_profile_submission_flags_t
      profile_submission_flags =
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_NONE;
  if (profile_events.event_count == 0 &&
      profile_queue_device_packet_count == 0) {
    profile_submission_flags |=
        IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET;
  }
  const uint32_t trailing_completion_packet_count =
      iree_any_bit_set(
          profile_submission_flags,
          IREE_HAL_AMDGPU_HOST_QUEUE_COMMAND_BUFFER_PROFILE_SUBMISSION_FLAG_TRAILING_COMPLETION_PACKET)
          ? 1u
          : 0u;
  if (IREE_UNLIKELY(emitted_packet_count >
                        UINT32_MAX - profile_harvest_packet_count ||
                    emitted_packet_count + profile_harvest_packet_count >
                        UINT32_MAX - profile_queue_device_packet_count ||
                    emitted_packet_count + profile_harvest_packet_count +
                            profile_queue_device_packet_count >
                        UINT32_MAX - trailing_completion_packet_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
    iree_arena_deinitialize(&scratch_arena);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "profiled command-buffer block packet count overflow");
  }
  const uint32_t payload_packet_count =
      emitted_packet_count + profile_harvest_packet_count +
      profile_queue_device_packet_count + trailing_completion_packet_count;
  const uint32_t profile_harvest_kernarg_block_count =
      profile_events.event_count != 0
          ? (uint32_t)iree_host_size_ceil_div(
                iree_hal_amdgpu_device_timestamp_dispatch_harvest_kernarg_length(
                    profile_events.event_count),
                sizeof(iree_hal_amdgpu_kernarg_block_t))
          : 0u;

  const uint64_t* block_binding_ptrs = binding_ptrs;
  if (!block_binding_ptrs) {
    status = iree_hal_amdgpu_host_queue_prepare_command_buffer_binding_ptrs(
        queue, command_buffer, binding_table, &scratch_arena,
        &block_binding_ptrs);
  }
  if (iree_status_is_ok(status) && profile_counter_packet_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_counter_samples(
        queue, profile_events);
  }
  if (iree_status_is_ok(status) && profile_trace_packet_count != 0) {
    status = iree_hal_amdgpu_host_queue_prepare_profile_traces(queue,
                                                               profile_events);
  }
  if (iree_status_is_ok(status) && profile_trace_packet_count != 0) {
    status =
        iree_hal_amdgpu_host_queue_prepare_command_buffer_profile_trace_code_objects(
            queue, profile_dispatches, profile_events);
  }

  iree_hal_amdgpu_host_queue_kernel_submission_t submission;
  memset(&submission, 0, sizeof(submission));
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
        queue, resolution, signal_semaphore_list, operation_resource_count,
        payload_packet_count,
        kernarg_block_count + profile_harvest_kernarg_block_count, out_ready,
        &submission);
  }
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                              profile_events);
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
  }
  if (iree_status_is_ok(status) && *out_ready) {
    iree_hal_amdgpu_aql_packet_t* profile_harvest_packet = NULL;
    iree_hal_amdgpu_aql_packet_t* profile_harvest_barrier_packet = NULL;
    iree_hal_amdgpu_profile_dispatch_harvest_source_t* profile_harvest_sources =
        NULL;
    uint16_t profile_harvest_setup = 0;
    const uint32_t profile_queue_device_prefix_packet_count =
        profile_queue_device_events.event_count != 0 ? 1u : 0u;
    const uint64_t first_payload_packet_id =
        submission.first_packet_id + resolution->barrier_count +
        profile_queue_device_prefix_packet_count;
    if (profile_events.event_count != 0) {
      profile_harvest_barrier_packet = iree_hal_amdgpu_aql_ring_packet(
          &queue->aql_ring, first_payload_packet_id + emitted_packet_count);
      profile_harvest_packet = iree_hal_amdgpu_aql_ring_packet(
          &queue->aql_ring, first_payload_packet_id + emitted_packet_count + 1);
      profile_harvest_sources =
          iree_hal_amdgpu_device_timestamp_emplace_dispatch_harvest(
              &queue->transfer_context->kernels
                   ->iree_hal_amdgpu_device_timestamp_harvest_dispatch_records,
              profile_events.event_count, &profile_harvest_packet->dispatch,
              submission.kernargs.blocks[kernarg_block_count].data);
      profile_harvest_setup = profile_harvest_packet->dispatch.setup;
    }
    uint16_t* packet_headers = NULL;
    uint16_t* packet_setups = NULL;
    status = iree_hal_amdgpu_host_queue_prepare_command_buffer_packet_metadata(
        queue, emitted_packet_count, &scratch_arena, &packet_headers,
        &packet_setups);
    if (iree_status_is_ok(status)) {
      status = iree_hal_amdgpu_host_queue_write_command_buffer_block(
          queue, resolution, signal_semaphore_list, command_buffer,
          binding_table, block, block_binding_ptrs, first_payload_packet_id,
          profile_queue_device_prefix_packet_count, submission.kernargs.blocks,
          packet_headers, packet_setups, profile_events, emitted_packet_count,
          profile_counter_set_count, profile_trace_packet_count,
          profile_harvest_sources, profile_dispatches);
    }
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_host_queue_command_buffer_profile_submission_t
          profile_submission = {
              .dispatch_events = profile_events,
              .queue_device_events = profile_queue_device_events,
              .queue_event_info =
                  {
                      .type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_EXECUTE,
                      .command_buffer_id = command_buffer_id,
                      .operation_count = block->command_count,
                  },
              .harvest =
                  {
                      .barrier_packet = profile_harvest_barrier_packet,
                      .packet = profile_harvest_packet,
                      .setup = profile_harvest_setup,
                  },
              .flags = profile_submission_flags,
          };
      iree_hal_amdgpu_host_queue_finish_command_buffer_block(
          queue, resolution, signal_semaphore_list, emitted_packet_count,
          inout_binding_resource_set, pre_signal_action, operation_resources,
          operation_resource_count, submission_flags, &submission,
          packet_headers, packet_setups, &profile_submission);
      iree_hal_amdgpu_host_queue_record_profile_queue_event(
          queue, resolution, signal_semaphore_list,
          &profile_submission.queue_event_info);
    } else {
      iree_hal_amdgpu_host_queue_fail_kernel_submission(queue, &submission);
      iree_hal_amdgpu_host_queue_cancel_profile_dispatch_events(queue,
                                                                profile_events);
      iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
          queue, profile_queue_device_events);
    }
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
