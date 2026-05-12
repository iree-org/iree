// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/device/timestamp.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile.h"
#include "iree/hal/drivers/amdgpu/host_queue_profile_events.h"
#include "iree/hal/drivers/amdgpu/profile_counters.h"
#include "iree/hal/drivers/amdgpu/profile_traces.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

// Returns true if |semaphore| has the strict private stream contract that lets
// the signal path publish only a producer queue epoch instead of accumulating a
// full multi-producer semaphore frontier.
static bool iree_hal_amdgpu_host_queue_is_private_stream_signal(
    const iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_semaphore_t* semaphore) {
  return iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      semaphore,
      (const iree_hal_amdgpu_logical_device_t*)queue->logical_device);
}

static uint64_t iree_hal_amdgpu_host_queue_last_drained_epoch(
    const iree_hal_amdgpu_host_queue_t* queue) {
  return (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_acquire);
}

static bool iree_hal_amdgpu_host_queue_should_push_frontier_snapshot(
    const iree_hal_amdgpu_host_queue_t* queue,
    bool span_needs_frontier_snapshot, uint64_t span_epoch,
    uint64_t last_drained_epoch) {
  return queue->can_publish_frontier && span_needs_frontier_snapshot &&
         span_epoch > last_drained_epoch;
}

// Returns a conservative upper bound on the number of frontier snapshots that
// commit_signals will push for |signal_semaphore_list|.
//
// Caller must hold submission_mutex.
static iree_host_size_t iree_hal_amdgpu_host_queue_count_frontier_snapshots(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  const uint64_t last_drained_epoch =
      iree_hal_amdgpu_host_queue_last_drained_epoch(queue);
  iree_host_size_t snapshot_count = 0;
  iree_async_semaphore_t* last_semaphore = queue->last_signal.semaphore;
  uint64_t last_semaphore_epoch = queue->last_signal.epoch;
  bool last_needs_frontier_snapshot =
      queue->last_signal.needs_frontier_snapshot;
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* hal_semaphore = signal_semaphore_list.semaphores[i];
    iree_async_semaphore_t* semaphore = (iree_async_semaphore_t*)hal_semaphore;
    const bool needs_frontier_snapshot =
        queue->can_publish_frontier &&
        !iree_hal_amdgpu_host_queue_is_private_stream_signal(queue,
                                                             hal_semaphore);
    if (semaphore != last_semaphore) {
      if (last_semaphore != NULL &&
          iree_hal_amdgpu_host_queue_should_push_frontier_snapshot(
              queue, last_needs_frontier_snapshot, last_semaphore_epoch,
              last_drained_epoch)) {
        ++snapshot_count;
      }
      last_semaphore = semaphore;
      last_needs_frontier_snapshot = needs_frontier_snapshot;
    }
    // Any later transition within this submission is necessarily pending, even
    // if the previous same-semaphore span had already drained.
    last_semaphore_epoch = UINT64_MAX;
  }
  return snapshot_count;
}

static void iree_hal_amdgpu_host_queue_push_frontier_snapshot_if_pending(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_async_frontier_t* queue_frontier) {
  if (queue->last_signal.semaphore == NULL) return;
  if (!iree_hal_amdgpu_host_queue_should_push_frontier_snapshot(
          queue, queue->last_signal.needs_frontier_snapshot,
          queue->last_signal.epoch,
          iree_hal_amdgpu_host_queue_last_drained_epoch(queue))) {
    return;
  }
  iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
      &queue->notification_ring, queue->last_signal.epoch, queue_frontier);
}

iree_status_t iree_hal_amdgpu_host_queue_count_reclaim_resources(
    iree_host_size_t signal_semaphore_count,
    iree_host_size_t operation_resource_count,
    uint16_t* out_reclaim_resource_count) {
  IREE_ASSERT_ARGUMENT(out_reclaim_resource_count);
  if (signal_semaphore_count > UINT16_MAX ||
      operation_resource_count > UINT16_MAX ||
      signal_semaphore_count > UINT16_MAX - operation_resource_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "submission retains too many resources (signals=%" PRIhsz
        ", operation_resources=%" PRIhsz ", max=%u)",
        signal_semaphore_count, operation_resource_count, UINT16_MAX);
  }
  *out_reclaim_resource_count =
      (uint16_t)(signal_semaphore_count + operation_resource_count);
  return iree_ok_status();
}

// Writes |packet_count| no-op barrier packets into already-reserved AQL slots.
// The caller controls doorbell timing so these packets can plug failure-path
// reservations without leaving INVALID packets visible to the CP.
static void iree_hal_amdgpu_host_queue_fill_noop_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t first_packet_id,
    uint32_t packet_count) {
  for (uint32_t i = 0; i < packet_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, first_packet_id + i);
    uint16_t header = iree_hal_amdgpu_aql_emit_nop(
        &packet->barrier_and,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_NONE,
                                                   IREE_HSA_FENCE_SCOPE_NONE),
        iree_hsa_signal_null());
    iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  }
}

// Emits |packet_count| no-op barrier packets and rings the doorbell. Used only
// to plug already-reserved AQL slots on an internal failure path so the CP
// never stalls on an INVALID header.
static void iree_hal_amdgpu_host_queue_emit_noop_packets(
    iree_hal_amdgpu_host_queue_t* queue, uint64_t first_packet_id,
    uint32_t packet_count) {
  IREE_ASSERT(packet_count > 0, "must plug at least one reserved AQL packet");
  iree_hal_amdgpu_host_queue_fill_noop_packets(queue, first_packet_id,
                                               packet_count);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                    first_packet_id + packet_count - 1);
}

// Publishes an internal completion epoch for a failed submission that already
// reserved kernarg space. User-visible semaphores are not signaled, but the
// normal notification drain can reclaim the kernarg ring in queue order.
static void iree_hal_amdgpu_host_queue_emit_reclaim_noop_packets(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_reclaim_entry_t* reclaim_entry, uint64_t first_packet_id,
    uint32_t packet_count, uint64_t kernarg_write_position,
    uint64_t queue_upload_write_position) {
  reclaim_entry->kernarg_write_position = kernarg_write_position;
  reclaim_entry->queue_upload_write_position = queue_upload_write_position;
  reclaim_entry->count = 0;
  iree_hal_amdgpu_notification_ring_advance_epoch(&queue->notification_ring);
  for (uint32_t i = 0; i < packet_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, first_packet_id + i);
    const bool is_final_packet = i + 1 == packet_count;
    uint16_t header = iree_hal_amdgpu_aql_emit_nop(
        &packet->barrier_and,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_NONE,
                                                   IREE_HSA_FENCE_SCOPE_NONE),
        is_final_packet ? iree_hal_amdgpu_notification_ring_epoch_signal(
                              &queue->notification_ring)
                        : iree_hsa_signal_null());
    iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  }
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                    first_packet_id + packet_count - 1);
}

// Returns the packet control for the final dispatch packet in a submission.
// Direct host-queue submissions keep BARRIER set so the queue epoch remains an
// ordered prefix-completion clock. Dispatches carry at least AGENT acquire so
// device-side packet execution observes host-populated kernargs.
static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_final_dispatch_packet_control(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hsa_fence_scope_t minimum_release_scope) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_max_fence_scope(
              IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
          minimum_acquire_scope),
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_signal_list_release_scope(
              queue, signal_semaphore_list),
          minimum_release_scope));
}

// Returns the packet control for a dispatch packet followed by a trailing
// queue-completion packet. The dispatch no longer signals the queue/user epoch,
// so it keeps only operation-local visibility requirements; the trailing packet
// owns signal-list release visibility.
static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_payload_dispatch_packet_control(
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hsa_fence_scope_t minimum_acquire_scope,
    iree_hsa_fence_scope_t minimum_release_scope) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      iree_hal_amdgpu_host_queue_max_fence_scope(
          iree_hal_amdgpu_host_queue_max_fence_scope(
              IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
          minimum_acquire_scope),
      minimum_release_scope);
}

// Returns the packet control for a final no-op/barrier completion packet.
static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_final_barrier_packet_control(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      resolution->inline_acquire_scope,
      iree_hal_amdgpu_host_queue_signal_list_release_scope(
          queue, signal_semaphore_list));
}

// Returns the packet control for a final PM4-IB payload packet. PM4 IB payloads
// are host-populated memory consumed by the CP, so they carry the same minimum
// AGENT acquire as dispatch packets.
static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_final_pm4_ib_packet_control(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      iree_hal_amdgpu_host_queue_max_fence_scope(
          IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
      iree_hal_amdgpu_host_queue_signal_list_release_scope(
          queue, signal_semaphore_list));
}

// Returns the packet control for a non-final PM4-IB packet in a larger
// submission. The final packet owns user-visible release and queue completion.
static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_payload_pm4_ib_packet_control(
    const iree_hal_amdgpu_wait_resolution_t* resolution) {
  return iree_hal_amdgpu_aql_packet_control_barrier(
      iree_hal_amdgpu_host_queue_max_fence_scope(
          IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
      IREE_HSA_FENCE_SCOPE_NONE);
}

void iree_hal_amdgpu_host_queue_commit_queue_device_start_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution, uint64_t packet_id,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(pm4_ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_start_to_memory(
          &builder, queue->pm4_timestamp_strategy,
          &queue_device_event->start_tick);
  IREE_ASSERT(did_emit, "PM4 start timestamp must fit profiling IB slot");
  (void)did_emit;
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib(
      &packet->pm4_ib, pm4_ib_slot,
      iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
      iree_hal_amdgpu_host_queue_payload_pm4_ib_packet_control(resolution),
      iree_hsa_signal_null(), &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

void iree_hal_amdgpu_host_queue_commit_queue_device_end_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list, uint64_t packet_id,
    iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event) {
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
      &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
  iree_hal_amdgpu_pm4_ib_builder_t builder;
  iree_hal_amdgpu_pm4_ib_builder_initialize(pm4_ib_slot, &builder);
  const bool did_emit =
      iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
          &builder, queue->pm4_timestamp_strategy,
          &queue_device_event->end_tick);
  IREE_ASSERT(did_emit, "PM4 end timestamp must fit profiling IB slot");
  (void)did_emit;
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib(
      &packet->pm4_ib, pm4_ib_slot,
      iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
      iree_hal_amdgpu_host_queue_final_pm4_ib_packet_control(
          queue, resolution, signal_semaphore_list),
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring),
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
}

uint16_t iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
    iree_hsa_kernel_dispatch_packet_t* IREE_RESTRICT dispatch_packet,
    const iree_hsa_kernel_dispatch_packet_t* IREE_RESTRICT
        dispatch_packet_template,
    void* kernarg_address, iree_hsa_signal_t completion_signal) {
  dispatch_packet->workgroup_size[0] =
      dispatch_packet_template->workgroup_size[0];
  dispatch_packet->workgroup_size[1] =
      dispatch_packet_template->workgroup_size[1];
  dispatch_packet->workgroup_size[2] =
      dispatch_packet_template->workgroup_size[2];
  dispatch_packet->reserved0 = dispatch_packet_template->reserved0;
  dispatch_packet->grid_size[0] = dispatch_packet_template->grid_size[0];
  dispatch_packet->grid_size[1] = dispatch_packet_template->grid_size[1];
  dispatch_packet->grid_size[2] = dispatch_packet_template->grid_size[2];
  dispatch_packet->private_segment_size =
      dispatch_packet_template->private_segment_size;
  dispatch_packet->group_segment_size =
      dispatch_packet_template->group_segment_size;
  dispatch_packet->kernel_object = dispatch_packet_template->kernel_object;
  dispatch_packet->kernarg_address = kernarg_address;
  dispatch_packet->reserved2 = dispatch_packet_template->reserved2;
  dispatch_packet->completion_signal = completion_signal;
  return dispatch_packet_template->setup;
}

iree_status_t iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count, uint32_t payload_packet_count,
    uint32_t kernarg_block_count, bool* out_ready,
    iree_hal_amdgpu_host_queue_kernel_submission_t* out_submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(out_ready);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_ready = false;
  memset(out_submission, 0, sizeof(*out_submission));

  if (IREE_UNLIKELY(payload_packet_count == 0)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "kernel submission requires at least one payload "
                            "packet");
  }

  const uint64_t packet_count =
      (uint64_t)resolution->barrier_count + payload_packet_count;
  const uint64_t aql_queue_capacity = (uint64_t)queue->aql_ring.mask + 1;
  if (IREE_UNLIKELY(packet_count > aql_queue_capacity ||
                    packet_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "kernel submission requires %" PRIu64
        " AQL packets (%u barriers + %u payload packets) but queue capacity is "
        "%" PRIu64,
        packet_count, resolution->barrier_count, payload_packet_count,
        aql_queue_capacity);
  }
  if (IREE_UNLIKELY(kernarg_block_count > queue->kernarg_ring.capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "kernel submission requires %u kernarg blocks but ring capacity is %u",
        kernarg_block_count, queue->kernarg_ring.capacity);
  }
  if (IREE_UNLIKELY(signal_semaphore_list.count >
                    queue->notification_ring.capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel submission requires %" PRIhsz
                            " notification entries but ring capacity is %u",
                            signal_semaphore_list.count,
                            queue->notification_ring.capacity);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list.count, operation_resource_count,
      &out_submission->reclaim_resource_count));

  const iree_host_size_t frontier_snapshot_count =
      iree_hal_amdgpu_host_queue_count_frontier_snapshots(
          queue, signal_semaphore_list);
  if (!iree_hal_amdgpu_notification_ring_can_reserve(
          &queue->notification_ring, signal_semaphore_list.count,
          frontier_snapshot_count)) {
    return iree_ok_status();
  }
  if (kernarg_block_count > 0 &&
      !iree_hal_amdgpu_kernarg_ring_can_allocate(&queue->kernarg_ring,
                                                 kernarg_block_count)) {
    return iree_ok_status();
  }

  uint64_t first_packet_id = 0;
  if (!iree_hal_amdgpu_aql_ring_try_reserve(
          &queue->aql_ring, (uint32_t)packet_count, &first_packet_id)) {
    return iree_ok_status();
  }

  out_submission->reclaim_entry =
      iree_hal_amdgpu_notification_ring_reclaim_entry(
          &queue->notification_ring);
  iree_status_t status = iree_hal_amdgpu_reclaim_entry_prepare(
      out_submission->reclaim_entry, queue->block_pool,
      out_submission->reclaim_resource_count,
      &out_submission->reclaim_resources);
  if (iree_status_is_ok(status)) {
    out_submission->packet_count = (uint32_t)packet_count;
    out_submission->first_packet_id = first_packet_id;
    if (kernarg_block_count > 0) {
      out_submission->kernargs.blocks = iree_hal_amdgpu_kernarg_ring_allocate(
          &queue->kernarg_ring, kernarg_block_count,
          &out_submission->kernargs.write_position);
      if (IREE_UNLIKELY(!out_submission->kernargs.blocks)) {
        iree_hal_amdgpu_host_queue_emit_noop_packets(
            queue, out_submission->first_packet_id,
            out_submission->packet_count);
        iree_hal_amdgpu_reclaim_entry_release(out_submission->reclaim_entry,
                                              queue->block_pool);
        status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                  "kernarg ring allocation failed after AQL "
                                  "reservation; queue sizing invariant was "
                                  "violated");
      }
    } else {
      out_submission->kernargs.write_position = (uint64_t)iree_atomic_load(
          &queue->kernarg_ring.write_position, iree_memory_order_relaxed);
    }
  } else {
    iree_hal_amdgpu_host_queue_emit_noop_packets(queue, first_packet_id,
                                                 (uint32_t)packet_count);
  }
  if (iree_status_is_ok(status)) {
    *out_ready = true;
  } else {
    memset(out_submission, 0, sizeof(*out_submission));
  }
  return status;
}

static uint32_t iree_hal_amdgpu_host_queue_barrier_packet_count(
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    bool profile_queue_device_event) {
  if (profile_queue_device_event) {
    return (uint32_t)resolution->barrier_count + 1u;
  }
  return resolution->barrier_count > 0 ? resolution->barrier_count : 1u;
}

iree_status_t iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    bool* out_ready,
    iree_hal_amdgpu_host_queue_barrier_submission_t* out_submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(out_ready);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_ready = false;
  memset(out_submission, 0, sizeof(*out_submission));

  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  const bool profile_queue_device_event =
      profile_event_info &&
      iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue);
  const uint64_t packet_count = iree_hal_amdgpu_host_queue_barrier_packet_count(
      resolution, profile_queue_device_event);
  const uint64_t aql_queue_capacity = (uint64_t)queue->aql_ring.mask + 1;
  if (IREE_UNLIKELY(packet_count > aql_queue_capacity ||
                    packet_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "barrier submission requires %" PRIu64
        " AQL packets (%u wait barriers) but queue capacity is %" PRIu64,
        packet_count, resolution->barrier_count, aql_queue_capacity);
  }
  if (IREE_UNLIKELY(signal_semaphore_list.count >
                    queue->notification_ring.capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "barrier submission requires %" PRIhsz
                            " notification entries but ring capacity is %u",
                            signal_semaphore_list.count,
                            queue->notification_ring.capacity);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list.count, operation_resource_count,
      &out_submission->reclaim_resource_count));

  if (profile_queue_device_event) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
            queue, /*event_count=*/1,
            &out_submission->profile_queue_device_events));
  }

  const iree_host_size_t frontier_snapshot_count =
      iree_hal_amdgpu_host_queue_count_frontier_snapshots(
          queue, signal_semaphore_list);
  if (!iree_hal_amdgpu_notification_ring_can_reserve(
          &queue->notification_ring, signal_semaphore_list.count,
          frontier_snapshot_count)) {
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, out_submission->profile_queue_device_events);
    memset(out_submission, 0, sizeof(*out_submission));
    return iree_ok_status();
  }

  uint64_t first_packet_id = 0;
  if (!iree_hal_amdgpu_aql_ring_try_reserve(
          &queue->aql_ring, (uint32_t)packet_count, &first_packet_id)) {
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, out_submission->profile_queue_device_events);
    memset(out_submission, 0, sizeof(*out_submission));
    return iree_ok_status();
  }

  out_submission->reclaim_entry =
      iree_hal_amdgpu_notification_ring_reclaim_entry(
          &queue->notification_ring);
  iree_status_t status = iree_hal_amdgpu_reclaim_entry_prepare(
      out_submission->reclaim_entry, queue->block_pool,
      out_submission->reclaim_resource_count,
      &out_submission->reclaim_resources);
  if (iree_status_is_ok(status)) {
    out_submission->packet_count = (uint32_t)packet_count;
    out_submission->first_packet_id = first_packet_id;
    *out_ready = true;
  } else {
    iree_hal_amdgpu_host_queue_emit_noop_packets(queue, first_packet_id,
                                                 (uint32_t)packet_count);
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, out_submission->profile_queue_device_events);
    memset(out_submission, 0, sizeof(*out_submission));
  }
  return status;
}

void iree_hal_amdgpu_host_queue_fail_kernel_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_kernel_submission_t* submission) {
  iree_hal_amdgpu_host_queue_emit_reclaim_noop_packets(
      queue, submission->reclaim_entry, submission->first_packet_id,
      submission->packet_count, submission->kernargs.write_position,
      submission->queue_upload.write_position);
  memset(submission, 0, sizeof(*submission));
}

void iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_pm4_ib_submission_t* submission) {
  iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
      queue, submission->profile_queue_device_events);
  iree_hal_amdgpu_host_queue_fail_kernel_submission(queue, &submission->kernel);
  memset(submission, 0, sizeof(*submission));
}

void iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_amdgpu_host_queue_kernel_submission_t* submission) {
  iree_hal_amdgpu_host_queue_emit_barriers(queue, resolution,
                                           submission->first_packet_id);
}

iree_status_t iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count, uint32_t kernarg_block_count,
    iree_hal_amdgpu_profile_dispatch_event_reservation_t profile_events,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    bool* out_ready,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* out_submission) {
  IREE_ASSERT_ARGUMENT(out_ready);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_ready = false;
  memset(out_submission, 0, sizeof(*out_submission));

  const bool use_profiling_completion_signal = profile_events.event_count != 0;
  const bool profile_queue_device_event =
      profile_queue_event_info &&
      iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue);
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events = {0};
  if (profile_queue_device_event) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
            queue, /*event_count=*/1, &profile_queue_device_events));
  }
  uint32_t profile_counter_set_count = 0;
  uint32_t profile_counter_packet_count = 0;
  uint32_t profile_trace_packet_count = 0;
  uint32_t profile_trace_start_packet_count = 0;
  if (use_profiling_completion_signal) {
    profile_counter_set_count =
        iree_hal_amdgpu_host_queue_profile_counter_set_count(queue,
                                                             profile_events);
    profile_counter_packet_count =
        iree_hal_amdgpu_host_queue_profile_counter_packet_count(queue,
                                                                profile_events);
    profile_trace_packet_count =
        iree_hal_amdgpu_host_queue_profile_trace_packet_count(queue,
                                                              profile_events);
    profile_trace_start_packet_count =
        iree_hal_amdgpu_host_queue_profile_trace_start_packet_count(
            queue, profile_events);
  }
  const uint32_t profile_queue_device_packet_count =
      profile_queue_device_events.event_count != 0 ? 2u : 0u;
  const uint32_t payload_packet_count =
      1u + profile_counter_packet_count + profile_trace_packet_count +
      (use_profiling_completion_signal ? 1u : 0u) +
      profile_queue_device_packet_count;
  const uint32_t profile_harvest_kernarg_block_count =
      use_profiling_completion_signal
          ? (uint32_t)iree_host_size_ceil_div(
                iree_hal_amdgpu_device_timestamp_dispatch_harvest_kernarg_length(
                    profile_events.event_count),
                sizeof(iree_hal_amdgpu_kernarg_block_t))
          : 0u;
  iree_status_t status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      payload_packet_count,
      kernarg_block_count + profile_harvest_kernarg_block_count, out_ready,
      &out_submission->kernel);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
  }
  if (iree_status_is_ok(status) && *out_ready) {
    out_submission->profile_queue_device_events = profile_queue_device_events;
    const uint32_t profile_queue_device_prefix_packet_count =
        profile_queue_device_events.event_count != 0 ? 1u : 0u;
    const uint32_t profile_queue_device_suffix_packet_count =
        profile_queue_device_events.event_count != 0 ? 1u : 0u;
    const uint64_t dispatch_packet_id =
        out_submission->kernel.first_packet_id + resolution->barrier_count +
        profile_queue_device_prefix_packet_count + profile_counter_set_count +
        profile_trace_start_packet_count;
    out_submission->dispatch_packet_id = dispatch_packet_id;
    out_submission->dispatch_slot =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, dispatch_packet_id);
    out_submission->dispatch_completion_signal =
        profile_queue_device_events.event_count != 0
            ? iree_hsa_signal_null()
            : iree_hal_amdgpu_notification_ring_epoch_signal(
                  &queue->notification_ring);
    if (use_profiling_completion_signal) {
      out_submission->profile_events = profile_events;
      out_submission->profile_counter_set_count = profile_counter_set_count;
      out_submission->profile_trace_start_packet_count =
          profile_trace_start_packet_count;
      out_submission->dispatch_completion_signal =
          iree_hal_amdgpu_host_queue_profiling_completion_signal(
              queue, profile_events.first_event_position);
      out_submission->profile_harvest_slot = iree_hal_amdgpu_aql_ring_packet(
          &queue->aql_ring, out_submission->kernel.first_packet_id +
                                out_submission->kernel.packet_count - 1 -
                                profile_queue_device_suffix_packet_count);
      out_submission->profile_harvest_kernarg_blocks =
          &out_submission->kernel.kernargs.blocks[kernarg_block_count];
      out_submission->minimum_release_scope =
          iree_hal_amdgpu_host_queue_max_fence_scope(
              out_submission->minimum_release_scope,
              IREE_HSA_FENCE_SCOPE_AGENT);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_try_begin_pm4_ib_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    bool* out_ready,
    iree_hal_amdgpu_host_queue_pm4_ib_submission_t* out_submission) {
  IREE_ASSERT_ARGUMENT(out_ready);
  IREE_ASSERT_ARGUMENT(out_submission);
  *out_ready = false;
  memset(out_submission, 0, sizeof(*out_submission));

  if (IREE_UNLIKELY(!queue->pm4_ib_slots)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "PM4 IB slots are not available");
  }

  const bool profile_queue_device_event =
      profile_queue_event_info &&
      iree_hal_amdgpu_host_queue_should_profile_queue_device_events(queue);
  iree_hal_amdgpu_profile_queue_device_event_reservation_t
      profile_queue_device_events = {0};
  if (profile_queue_device_event) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_reserve_profile_queue_device_events(
            queue, /*event_count=*/1, &profile_queue_device_events));
  }
  iree_status_t status = iree_hal_amdgpu_host_queue_try_begin_kernel_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      /*payload_packet_count=*/1, /*kernarg_block_count=*/0, out_ready,
      &out_submission->kernel);
  if (!iree_status_is_ok(status) || !*out_ready) {
    iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
        queue, profile_queue_device_events);
  }
  if (iree_status_is_ok(status) && *out_ready) {
    out_submission->profile_queue_device_events = profile_queue_device_events;
    const uint64_t packet_id =
        out_submission->kernel.first_packet_id + resolution->barrier_count;
    out_submission->pm4_ib_packet_slot =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
    out_submission->pm4_ib_slot =
        &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
    iree_hal_amdgpu_pm4_ib_builder_initialize(out_submission->pm4_ib_slot,
                                              &out_submission->pm4_ib_builder);
    if (profile_queue_device_events.event_count != 0) {
      iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
          iree_hal_amdgpu_host_queue_profile_queue_device_event_at(
              queue, profile_queue_device_events.first_event_position);
      const bool did_emit =
          iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_start_to_memory(
              &out_submission->pm4_ib_builder, queue->pm4_timestamp_strategy,
              &queue_device_event->start_tick);
      if (IREE_UNLIKELY(!did_emit)) {
        iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue,
                                                          out_submission);
        *out_ready = false;
        return iree_make_status(
            IREE_STATUS_INTERNAL,
            "PM4 queue-device start timestamp does not fit PM4 IB slot");
      }
    }
  }
  return status;
}

// Commits the signal/frontier side of an AQL submission. Called after all
// dispatch packet fields, kernargs, and any prefix barrier headers are written,
// but before the final dispatch header is committed and the doorbell is rung.
// The caller must reserve notification-ring space and prepare the next reclaim
// entry before this call. Caller must hold submission_mutex.
static uint64_t iree_hal_amdgpu_host_queue_commit_signals(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  // Advance epoch and merge this queue's axis into the accumulated frontier.
  uint64_t epoch = iree_hal_amdgpu_notification_ring_advance_epoch(
      &queue->notification_ring);
  iree_async_single_frontier_t self_frontier;
  iree_async_single_frontier_initialize(&self_frontier, queue->axis, epoch);
  if (IREE_UNLIKELY(!iree_async_frontier_merge(
          iree_hal_amdgpu_host_queue_frontier(queue),
          IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY,
          iree_async_single_frontier_as_const_frontier(&self_frontier)))) {
    // The queue frontier was full of foreign axes and did not contain this
    // queue's own axis. Collapse to the current self axis as a safe lower bound
    // and permanently disable frontier publication so later waits defer instead
    // of observing under-attributed dependencies.
    iree_async_frontier_initialize(iree_hal_amdgpu_host_queue_frontier(queue),
                                   /*entry_count=*/1);
    queue->frontier.entries[0] = self_frontier.entries[0];
    queue->can_publish_frontier = false;
  }

  const iree_async_frontier_t* queue_frontier =
      iree_hal_amdgpu_host_queue_const_frontier(queue);

  // A submission with no user-visible signal semaphores still consumes one
  // queue-private epoch and reclaim entry. Leave last_signal unchanged so a
  // later signaled submission can still flush the previous same-semaphore span;
  // any intervening zero-signal epochs are conservatively included in that
  // frontier snapshot.
  if (signal_semaphore_list.count == 0) {
    return epoch;
  }

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_hal_semaphore_t* hal_semaphore = signal_semaphore_list.semaphores[i];
    uint64_t value = signal_semaphore_list.payload_values[i];
    iree_async_semaphore_t* async_semaphore =
        (iree_async_semaphore_t*)hal_semaphore;
    bool is_amdgpu_semaphore = iree_hal_amdgpu_semaphore_isa(hal_semaphore);
    const bool is_private_stream_signal =
        iree_hal_amdgpu_host_queue_is_private_stream_signal(queue,
                                                            hal_semaphore);

    // Detect semaphore transition for frontier snapshot recording.
    if (async_semaphore != queue->last_signal.semaphore) {
      iree_hal_amdgpu_host_queue_push_frontier_snapshot_if_pending(
          queue, queue_frontier);
      queue->last_signal.semaphore = async_semaphore;
      queue->last_signal.needs_frontier_snapshot =
          queue->can_publish_frontier && !is_private_stream_signal;
    }

    // Push notification entry for drain -> signal_untainted on completion.
    const iree_hal_amdgpu_notification_entry_flags_t notification_flags =
        (is_private_stream_signal || !queue->can_publish_frontier)
            ? IREE_HAL_AMDGPU_NOTIFICATION_ENTRY_FLAG_OMIT_FRONTIER_SNAPSHOT
            : IREE_HAL_AMDGPU_NOTIFICATION_ENTRY_FLAG_NONE;
    iree_hal_amdgpu_notification_ring_push(&queue->notification_ring, epoch,
                                           async_semaphore, value,
                                           notification_flags);
    queue->last_signal.epoch = epoch;

    if (is_private_stream_signal) {
      iree_hal_amdgpu_semaphore_publish_private_stream_signal(
          hal_semaphore, queue->axis, epoch, value);
      continue;
    }

    // Submission-time causal marker: merge queue's frontier into the
    // semaphore's frontier so same-queue and already-dominated cross-queue
    // waits can resolve before GPU completion under the current submission
    // boundary barrier policy.
    bool did_publish_frontier = queue->can_publish_frontier;
    if (did_publish_frontier) {
      if (is_amdgpu_semaphore) {
        did_publish_frontier = iree_hal_amdgpu_semaphore_publish_signal(
            hal_semaphore, queue->axis, queue_frontier, epoch, value);
      } else {
        did_publish_frontier = iree_async_semaphore_merge_frontier(
            async_semaphore, queue_frontier);
      }
    }
    if (!did_publish_frontier) {
      // The semaphore's frontier storage overflowed, so its frontier is no
      // longer a conservative summary of this signal's causal dependencies.
      // Clear the last-signal cache to force future waits down the software
      // deferral path instead of unsafely eliding or under-barriering them.
      if (is_amdgpu_semaphore) {
        iree_hal_amdgpu_semaphore_clear_last_signal(hal_semaphore);
      }
      continue;
    }
  }
  return epoch;
}

uint64_t iree_hal_amdgpu_host_queue_finish_kernel_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_resource_set_t** inout_resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_kernel_submission_t* submission) {
  const bool retain_submission_resources = iree_any_bit_set(
      submission_flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    submission->reclaim_resources[i] =
        (iree_hal_resource_t*)signal_semaphore_list.semaphores[i];
    if (retain_submission_resources) {
      iree_hal_resource_retain(submission->reclaim_resources[i]);
    }
  }
  for (iree_host_size_t i = 0; i < operation_resource_count; ++i) {
    iree_hal_resource_t* resource = operation_resources[i];
    submission->reclaim_resources[signal_semaphore_list.count + i] = resource;
    if (retain_submission_resources) {
      iree_hal_resource_retain(resource);
    }
  }
  submission->reclaim_entry->resource_set =
      inout_resource_set ? *inout_resource_set : NULL;
  if (inout_resource_set) {
    *inout_resource_set = NULL;
  }
  submission->reclaim_entry->kernarg_write_position =
      submission->kernargs.write_position;
  submission->reclaim_entry->queue_upload_write_position =
      submission->queue_upload.write_position;
  submission->reclaim_entry->count = submission->reclaim_resource_count;
  submission->reclaim_entry->pre_signal_action = submission->pre_signal_action;
  iree_hal_amdgpu_host_queue_merge_barrier_axes(queue, resolution);
  return iree_hal_amdgpu_host_queue_commit_signals(queue,
                                                   signal_semaphore_list);
}

uint64_t iree_hal_amdgpu_host_queue_finish_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* submission) {
  iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, resolution,
                                                           &submission->kernel);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_kernel_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, /*inout_resource_set=*/NULL,
          submission_flags, &submission->kernel);

  iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
      iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
          queue, submission->profile_queue_device_events,
          profile_queue_event_info);
  if (queue_device_event) {
    submission->kernel.reclaim_entry->queue_device_event_first_position =
        submission->profile_queue_device_events.first_event_position;
    submission->kernel.reclaim_entry->queue_device_event_count =
        submission->profile_queue_device_events.event_count;
    queue_device_event->submission_id = submission_epoch;
  }

  uint16_t profile_harvest_header = 0;
  const iree_hsa_fence_scope_t dispatch_minimum_acquire_scope =
      submission->kernel.kernargs.blocks
          ? iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
                queue, submission->minimum_acquire_scope)
          : submission->minimum_acquire_scope;
  iree_hal_amdgpu_aql_packet_control_t dispatch_packet_control =
      iree_hal_amdgpu_host_queue_final_dispatch_packet_control(
          queue, resolution, signal_semaphore_list,
          dispatch_minimum_acquire_scope, submission->minimum_release_scope);
  if (queue_device_event || submission->profile_harvest_slot) {
    dispatch_packet_control =
        iree_hal_amdgpu_host_queue_payload_dispatch_packet_control(
            resolution, dispatch_minimum_acquire_scope,
            submission->minimum_release_scope);
  }
  if (submission->profile_harvest_slot) {
    submission->kernel.reclaim_entry->profile_event_first_position =
        submission->profile_events.first_event_position;
    submission->kernel.reclaim_entry->profile_event_count =
        submission->profile_events.event_count;
    for (uint32_t i = 0; i < submission->profile_events.event_count; ++i) {
      iree_hal_amdgpu_profile_dispatch_event_t* event =
          iree_hal_amdgpu_host_queue_profile_dispatch_event_at(
              queue, submission->profile_events.first_event_position + i);
      event->submission_id = submission_epoch;
    }
    submission->profile_harvest_slot->dispatch.completion_signal =
        queue_device_event ? iree_hsa_signal_null()
                           : iree_hal_amdgpu_notification_ring_epoch_signal(
                                 &queue->notification_ring);
    const iree_hsa_fence_scope_t profile_harvest_acquire_scope =
        iree_hal_amdgpu_host_queue_kernarg_acquire_scope(
            queue, IREE_HSA_FENCE_SCOPE_AGENT);
    profile_harvest_header = iree_hal_amdgpu_aql_make_header(
        IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
        queue_device_event
            ? iree_hal_amdgpu_aql_packet_control_barrier(
                  iree_hal_amdgpu_host_queue_max_fence_scope(
                      profile_harvest_acquire_scope,
                      resolution->inline_acquire_scope),
                  IREE_HSA_FENCE_SCOPE_SYSTEM)
            : iree_hal_amdgpu_host_queue_final_dispatch_packet_control(
                  queue, resolution, signal_semaphore_list,
                  profile_harvest_acquire_scope, IREE_HSA_FENCE_SCOPE_SYSTEM));
  }
  const uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH, dispatch_packet_control);
  const uint32_t profile_queue_device_prefix_packet_count =
      queue_device_event ? 1u : 0u;
  iree_hal_amdgpu_host_queue_publish_submission_kernargs(queue,
                                                         &submission->kernel);
  if (queue_device_event) {
    iree_hal_amdgpu_host_queue_commit_queue_device_start_packet(
        queue, resolution,
        submission->kernel.first_packet_id + resolution->barrier_count,
        queue_device_event);
  }
  if (submission->profile_counter_set_count != 0) {
    iree_hal_amdgpu_host_queue_commit_profile_counter_start_packets(
        queue, submission->profile_events.first_event_position,
        submission->profile_counter_set_count,
        submission->kernel.first_packet_id + resolution->barrier_count +
            profile_queue_device_prefix_packet_count,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_AGENT));
  }
  if (submission->profile_trace_start_packet_count != 0) {
    iree_hal_amdgpu_host_queue_commit_profile_trace_start_packet(
        queue, submission->profile_events.first_event_position,
        submission->kernel.first_packet_id + resolution->barrier_count +
            profile_queue_device_prefix_packet_count +
            submission->profile_counter_set_count,
        iree_hal_amdgpu_aql_packet_control_barrier(
            iree_hal_amdgpu_host_queue_max_fence_scope(
                IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope),
            IREE_HSA_FENCE_SCOPE_AGENT));
    iree_hal_amdgpu_host_queue_commit_profile_trace_code_object_packet(
        queue, submission->profile_events.first_event_position,
        submission->kernel.first_packet_id + resolution->barrier_count +
            profile_queue_device_prefix_packet_count +
            submission->profile_counter_set_count + 1u,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));
  }
  iree_hal_amdgpu_aql_ring_commit(submission->dispatch_slot, dispatch_header,
                                  submission->dispatch_setup);
  if (submission->profile_trace_start_packet_count != 0) {
    iree_hal_amdgpu_host_queue_commit_profile_trace_stop_packet(
        queue, submission->profile_events.first_event_position,
        submission->dispatch_packet_id + 1,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));
  }
  if (submission->profile_counter_set_count != 0) {
    const uint32_t profile_trace_stop_packet_count =
        submission->profile_trace_start_packet_count != 0
            ? submission->profile_events.event_count
            : 0u;
    iree_hal_amdgpu_host_queue_commit_profile_counter_read_stop_packets(
        queue, submission->profile_events.first_event_position,
        submission->profile_counter_set_count,
        submission->dispatch_packet_id + 1 + profile_trace_stop_packet_count,
        iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_AGENT,
                                                   IREE_HSA_FENCE_SCOPE_AGENT));
  }
  if (submission->profile_harvest_slot) {
    iree_hal_amdgpu_aql_ring_commit(submission->profile_harvest_slot,
                                    profile_harvest_header,
                                    submission->profile_harvest_setup);
  }
  if (queue_device_event) {
    iree_hal_amdgpu_host_queue_commit_queue_device_end_packet(
        queue, resolution, signal_semaphore_list,
        submission->kernel.first_packet_id + submission->kernel.packet_count -
            1,
        queue_device_event);
  }
  iree_hal_amdgpu_aql_ring_doorbell(
      &queue->aql_ring,
      submission->kernel.first_packet_id + submission->kernel.packet_count - 1);
  memset(submission, 0, sizeof(*submission));
  return submission_epoch;
}

iree_status_t iree_hal_amdgpu_host_queue_finish_pm4_ib_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_pm4_ib_submission_t* submission,
    uint64_t* out_submission_epoch) {
  if (out_submission_epoch) *out_submission_epoch = 0;

  iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
      iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
          queue, submission->profile_queue_device_events,
          profile_queue_event_info);
  if (queue_device_event) {
    const bool did_emit =
        iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_end_to_memory(
            &submission->pm4_ib_builder, queue->pm4_timestamp_strategy,
            &queue_device_event->end_tick);
    if (IREE_UNLIKELY(!did_emit)) {
      iree_hal_amdgpu_host_queue_fail_pm4_ib_submission(queue, submission);
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "PM4 queue-device end timestamp does not fit PM4 IB slot");
    }
  }

  iree_hal_amdgpu_host_queue_emit_kernel_submission_prefix(queue, resolution,
                                                           &submission->kernel);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_kernel_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, /*inout_resource_set=*/NULL,
          submission_flags, &submission->kernel);

  if (queue_device_event) {
    submission->kernel.reclaim_entry->queue_device_event_first_position =
        submission->profile_queue_device_events.first_event_position;
    submission->kernel.reclaim_entry->queue_device_event_count =
        submission->profile_queue_device_events.event_count;
    queue_device_event->submission_id = submission_epoch;
  }

  uint16_t pm4_ib_setup = 0;
  uint16_t pm4_ib_header = iree_hal_amdgpu_aql_emit_pm4_ib(
      &submission->pm4_ib_packet_slot->pm4_ib, submission->pm4_ib_slot,
      iree_hal_amdgpu_pm4_ib_builder_dword_count(&submission->pm4_ib_builder),
      iree_hal_amdgpu_host_queue_final_pm4_ib_packet_control(
          queue, resolution, signal_semaphore_list),
      iree_hal_amdgpu_notification_ring_epoch_signal(&queue->notification_ring),
      &pm4_ib_setup);
  iree_hal_amdgpu_aql_ring_commit(submission->pm4_ib_packet_slot, pm4_ib_header,
                                  pm4_ib_setup);
  iree_hal_amdgpu_aql_ring_doorbell(
      &queue->aql_ring,
      submission->kernel.first_packet_id + submission->kernel.packet_count - 1);
  memset(submission, 0, sizeof(*submission));
  if (out_submission_epoch) *out_submission_epoch = submission_epoch;
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_dispatch_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hsa_kernel_dispatch_packet_t* dispatch_packet_template,
    const void* kernargs, iree_host_size_t kernarg_length,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t*
        profile_queue_event_info,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready, uint64_t* out_submission_id) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(dispatch_packet_template);
  IREE_ASSERT_ARGUMENT(kernargs);
  IREE_ASSERT_ARGUMENT(out_ready);
  IREE_ASSERT_LE(kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  *out_ready = false;
  if (out_submission_id) *out_submission_id = 0;

  iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      /*kernarg_block_count=*/1,
      (iree_hal_amdgpu_profile_dispatch_event_reservation_t){0},
      profile_queue_event_info, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  memcpy(submission.kernel.kernargs.blocks->data, kernargs, kernarg_length);
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, dispatch_packet_template,
          submission.kernel.kernargs.blocks->data,
          submission.dispatch_completion_signal);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_dispatch_submission(
          queue, resolution, signal_semaphore_list, operation_resources,
          operation_resource_count, profile_queue_event_info, submission_flags,
          &submission);
  if (out_submission_id) *out_submission_id = submission_epoch;
  return iree_ok_status();
}

uint64_t iree_hal_amdgpu_host_queue_finish_barrier_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    iree_hal_amdgpu_host_queue_post_commit_callback_t post_commit_callback,
    iree_hal_resource_set_t* resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_barrier_submission_t* submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(submission);
  const bool retain_submission_resources = iree_any_bit_set(
      submission_flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);

  const bool complete_with_queue_device_event =
      submission->profile_queue_device_events.event_count != 0;
  const bool complete_with_wait_barrier =
      resolution->barrier_count > 0 && !complete_with_queue_device_event;
  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry = submission->reclaim_entry;
  iree_hal_resource_t** reclaim_resources = submission->reclaim_resources;
  reclaim_entry->pre_signal_action = pre_signal_action;
  reclaim_entry->resource_set = resource_set;

  const uint32_t aql_packet_count = submission->packet_count;
  const uint64_t first_packet_id = submission->first_packet_id;

  uint16_t completion_header = 0;
  uint16_t completion_setup = 0;
  iree_hal_amdgpu_aql_packet_t* completion_slot =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring,
                                      first_packet_id + aql_packet_count - 1);
  if (complete_with_queue_device_event) {
    iree_hal_amdgpu_host_queue_emit_barriers(queue, resolution,
                                             first_packet_id);
  } else if (complete_with_wait_barrier) {
    for (uint8_t i = 0; i + 1 < resolution->barrier_count; ++i) {
      iree_hal_amdgpu_aql_packet_t* barrier_packet =
          iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring,
                                          first_packet_id + i);
      uint16_t barrier_setup = 0;
      uint16_t barrier_header =
          iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
              queue, &resolution->barriers[i], first_packet_id + i,
              iree_hsa_signal_null(), resolution->barrier_acquire_scope,
              IREE_HSA_FENCE_SCOPE_NONE, barrier_packet, &barrier_setup);
      iree_hal_amdgpu_aql_ring_commit(barrier_packet, barrier_header,
                                      barrier_setup);
    }
  }
  iree_hal_amdgpu_profile_queue_device_event_t* queue_device_event =
      iree_hal_amdgpu_host_queue_initialize_profile_queue_device_event(
          queue, submission->profile_queue_device_events, profile_event_info);

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    reclaim_resources[i] =
        (iree_hal_resource_t*)signal_semaphore_list.semaphores[i];
    if (retain_submission_resources) {
      iree_hal_resource_retain(reclaim_resources[i]);
    }
  }
  for (iree_host_size_t i = 0; i < operation_resource_count; ++i) {
    iree_hal_resource_t* resource = operation_resources[i];
    reclaim_resources[signal_semaphore_list.count + i] = resource;
    if (retain_submission_resources) {
      iree_hal_resource_retain(resource);
    }
  }
  reclaim_entry->kernarg_write_position = (uint64_t)iree_atomic_load(
      &queue->kernarg_ring.write_position, iree_memory_order_relaxed);
  reclaim_entry->count = submission->reclaim_resource_count;

  iree_hal_amdgpu_host_queue_merge_barrier_axes(queue, resolution);
  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_commit_signals(queue, signal_semaphore_list);
  if (queue_device_event) {
    reclaim_entry->queue_device_event_first_position =
        submission->profile_queue_device_events.first_event_position;
    reclaim_entry->queue_device_event_count =
        submission->profile_queue_device_events.event_count;
    queue_device_event->submission_id = submission_epoch;
  }
  if (post_commit_callback.fn) {
    post_commit_callback.fn(post_commit_callback.user_data,
                            iree_hal_amdgpu_host_queue_const_frontier(queue),
                            submission_epoch);
  }
  if (complete_with_queue_device_event) {
    iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slot =
        &queue->pm4_ib_slots[(first_packet_id + aql_packet_count - 1) &
                             queue->aql_ring.mask];
    iree_hal_amdgpu_pm4_ib_builder_t builder;
    iree_hal_amdgpu_pm4_ib_builder_initialize(pm4_ib_slot, &builder);
    const bool did_emit =
        iree_hal_amdgpu_pm4_ib_builder_emit_timestamp_range_to_memory(
            &builder, queue->pm4_timestamp_strategy,
            &queue_device_event->start_tick, &queue_device_event->end_tick);
    IREE_ASSERT(did_emit, "PM4 timestamp range must fit profiling IB slot");
    (void)did_emit;
    completion_header = iree_hal_amdgpu_aql_emit_pm4_ib(
        &completion_slot->pm4_ib, pm4_ib_slot,
        iree_hal_amdgpu_pm4_ib_builder_dword_count(&builder),
        iree_hal_amdgpu_host_queue_final_pm4_ib_packet_control(
            queue, resolution, signal_semaphore_list),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring),
        &completion_setup);
  } else if (complete_with_wait_barrier) {
    const iree_hsa_fence_scope_t release_scope =
        iree_hal_amdgpu_host_queue_signal_list_release_scope(
            queue, signal_semaphore_list);
    completion_header =
        iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
            queue, &resolution->barriers[resolution->barrier_count - 1],
            first_packet_id + aql_packet_count - 1,
            iree_hal_amdgpu_notification_ring_epoch_signal(
                &queue->notification_ring),
            resolution->barrier_acquire_scope, release_scope, completion_slot,
            &completion_setup);
  } else {
    completion_header = iree_hal_amdgpu_aql_emit_nop(
        &completion_slot->barrier_and,
        iree_hal_amdgpu_host_queue_final_barrier_packet_control(
            queue, resolution, signal_semaphore_list),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring));
  }
  iree_hal_amdgpu_aql_ring_commit(completion_slot, completion_header,
                                  completion_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                    first_packet_id + aql_packet_count - 1);
  memset(submission, 0, sizeof(*submission));
  return submission_epoch;
}

void iree_hal_amdgpu_host_queue_fail_barrier_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_host_queue_barrier_submission_t* submission) {
  iree_hal_amdgpu_host_queue_emit_noop_packets(
      queue, submission->first_packet_id, submission->packet_count);
  iree_hal_amdgpu_host_queue_cancel_profile_queue_device_events(
      queue, submission->profile_queue_device_events);
  iree_hal_amdgpu_reclaim_entry_release(submission->reclaim_entry,
                                        queue->block_pool);
  memset(submission, 0, sizeof(*submission));
}

iree_status_t iree_hal_amdgpu_host_queue_try_submit_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    const iree_hal_amdgpu_host_queue_profile_event_info_t* profile_event_info,
    iree_hal_amdgpu_host_queue_post_commit_callback_t post_commit_callback,
    iree_hal_resource_set_t* resource_set,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    bool* out_ready, uint64_t* out_submission_id) {
  IREE_ASSERT_ARGUMENT(out_ready);
  *out_ready = false;
  if (out_submission_id) *out_submission_id = 0;

  iree_hal_amdgpu_host_queue_barrier_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_try_begin_barrier_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      profile_event_info, out_ready, &submission));
  if (!*out_ready) return iree_ok_status();

  const uint64_t submission_epoch =
      iree_hal_amdgpu_host_queue_finish_barrier_submission(
          queue, resolution, signal_semaphore_list, pre_signal_action,
          operation_resources, operation_resource_count, profile_event_info,
          post_commit_callback, resource_set, submission_flags, &submission);
  if (out_submission_id) *out_submission_id = submission_epoch;
  return iree_ok_status();
}
