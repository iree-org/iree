// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/aql_command_buffer.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/blit.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

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
// The caller controls doorbell timing so these packets can be used either as
// normal submission padding or as failure-path slot plugging.
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
    uint32_t packet_count, uint64_t kernarg_write_position) {
  reclaim_entry->kernarg_write_position = kernarg_write_position;
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

iree_status_t iree_hal_amdgpu_host_queue_begin_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t operation_resource_count, uint32_t kernarg_block_count,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* out_submission) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(out_submission);
  memset(out_submission, 0, sizeof(*out_submission));

  if (IREE_UNLIKELY(kernarg_block_count == 0)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "dispatch submission requires at least one "
                            "kernarg block");
  }

  const uint64_t packet_count =
      (uint64_t)resolution->barrier_count + kernarg_block_count;
  const uint64_t aql_queue_capacity = (uint64_t)queue->aql_ring.mask + 1;
  if (IREE_UNLIKELY(packet_count > aql_queue_capacity ||
                    packet_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch submission requires %" PRIu64
        " AQL packets (%u barriers + %u kernarg blocks) but queue capacity is "
        "%" PRIu64,
        packet_count, resolution->barrier_count, kernarg_block_count,
        aql_queue_capacity);
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list.count, operation_resource_count,
      &out_submission->reclaim_resource_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_notification_ring_reserve(
      &queue->notification_ring, signal_semaphore_list.count,
      iree_hal_amdgpu_host_queue_count_frontier_snapshots(
          queue, signal_semaphore_list)));

  out_submission->reclaim_entry =
      iree_hal_amdgpu_notification_ring_reclaim_entry(
          &queue->notification_ring);
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_reclaim_entry_prepare(
      out_submission->reclaim_entry, queue->block_pool,
      out_submission->reclaim_resource_count,
      &out_submission->reclaim_resources));

  out_submission->packet_count = (uint32_t)packet_count;
  out_submission->first_packet_id = iree_hal_amdgpu_aql_ring_reserve(
      &queue->aql_ring, out_submission->packet_count);
  out_submission->kernarg_blocks = iree_hal_amdgpu_kernarg_ring_allocate(
      &queue->kernarg_ring, kernarg_block_count,
      &out_submission->kernarg_write_position);
  if (IREE_UNLIKELY(!out_submission->kernarg_blocks)) {
    iree_hal_amdgpu_host_queue_emit_noop_packets(
        queue, out_submission->first_packet_id, out_submission->packet_count);
    iree_hal_amdgpu_reclaim_entry_release(out_submission->reclaim_entry,
                                          queue->block_pool);
    memset(out_submission, 0, sizeof(*out_submission));
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernarg ring allocation failed after AQL reservation; queue sizing "
        "invariant was violated");
  }

  iree_hal_amdgpu_host_queue_emit_barriers(queue, resolution,
                                           out_submission->first_packet_id);

  const uint32_t noop_packet_count = kernarg_block_count - 1;
  if (noop_packet_count > 0) {
    iree_hal_amdgpu_host_queue_fill_noop_packets(
        queue, out_submission->first_packet_id + resolution->barrier_count,
        noop_packet_count);
  }
  out_submission->dispatch_slot = iree_hal_amdgpu_aql_ring_packet(
      &queue->aql_ring,
      out_submission->first_packet_id + out_submission->packet_count - 1);
  return iree_ok_status();
}

// Commits the signal/frontier side of an AQL submission. Called after all
// dispatch packet fields, kernargs, and any prefix barrier headers are written,
// but before the final dispatch header is committed and the doorbell is rung.
// The caller must reserve notification-ring space and prepare the next reclaim
// entry before this call. Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_commit_signals(
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
    return;
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
    // waits can resolve before GPU completion under the current all-barrier
    // AQL queue policy.
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
}

void iree_hal_amdgpu_host_queue_finish_dispatch_submission(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_host_queue_dispatch_submission_t* submission) {
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
  submission->reclaim_entry->kernarg_write_position =
      submission->kernarg_write_position;
  submission->reclaim_entry->count = submission->reclaim_resource_count;
  submission->reclaim_entry->pre_signal_action = submission->pre_signal_action;

  iree_hal_amdgpu_host_queue_merge_barrier_axes(queue, resolution);
  iree_hal_amdgpu_host_queue_commit_signals(queue, signal_semaphore_list);

  uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_host_queue_final_dispatch_packet_control(
          queue, resolution, signal_semaphore_list,
          submission->minimum_acquire_scope,
          submission->minimum_release_scope));
  iree_hal_amdgpu_aql_ring_commit(submission->dispatch_slot, dispatch_header,
                                  submission->dispatch_setup);
  iree_hal_amdgpu_aql_ring_doorbell(
      &queue->aql_ring,
      submission->first_packet_id + submission->packet_count - 1);
  memset(submission, 0, sizeof(*submission));
}

iree_status_t iree_hal_amdgpu_host_queue_submit_dispatch_packet(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const iree_hsa_kernel_dispatch_packet_t* dispatch_packet_template,
    const void* kernargs, iree_host_size_t kernarg_length,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  IREE_ASSERT_ARGUMENT(dispatch_packet_template);
  IREE_ASSERT_ARGUMENT(kernargs);
  IREE_ASSERT_LE(kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));

  iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resource_count,
      /*kernarg_block_count=*/1, &submission));
  memcpy(submission.kernarg_blocks->data, kernargs, kernarg_length);
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, dispatch_packet_template,
          submission.kernarg_blocks->data,
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &queue->notification_ring));
  iree_hal_amdgpu_host_queue_finish_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      operation_resource_count, submission_flags, &submission);
  return iree_ok_status();
}

iree_status_t iree_hal_amdgpu_host_queue_submit_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_post_commit_fn_t post_commit_fn,
    void* post_commit_user_data,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  const bool retain_submission_resources = iree_any_bit_set(
      submission_flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);

  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  const bool complete_with_wait_barrier = resolution->barrier_count > 0;
  const uint64_t packet_count =
      complete_with_wait_barrier ? (uint64_t)resolution->barrier_count : 1ull;
  const uint64_t aql_queue_capacity = (uint64_t)queue->aql_ring.mask + 1;
  if (IREE_UNLIKELY(packet_count > aql_queue_capacity ||
                    packet_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "barrier submission requires %" PRIu64
        " AQL packets (%u wait barriers) but queue capacity is %" PRIu64,
        packet_count, resolution->barrier_count, aql_queue_capacity);
  }

  uint16_t reclaim_resource_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list.count, operation_resource_count,
      &reclaim_resource_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_notification_ring_reserve(
      &queue->notification_ring, signal_semaphore_list.count,
      iree_hal_amdgpu_host_queue_count_frontier_snapshots(
          queue, signal_semaphore_list)));

  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry =
      iree_hal_amdgpu_notification_ring_reclaim_entry(
          &queue->notification_ring);
  iree_hal_resource_t** reclaim_resources = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_reclaim_entry_prepare(
      reclaim_entry, queue->block_pool, reclaim_resource_count,
      &reclaim_resources));
  reclaim_entry->pre_signal_action = pre_signal_action;

  const uint32_t aql_packet_count = (uint32_t)packet_count;
  const uint64_t first_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, aql_packet_count);

  uint16_t completion_header = 0;
  uint16_t completion_setup = 0;
  iree_hal_amdgpu_aql_packet_t* completion_slot =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring,
                                      first_packet_id + aql_packet_count - 1);
  if (complete_with_wait_barrier) {
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
  reclaim_entry->count = reclaim_resource_count;

  iree_hal_amdgpu_host_queue_merge_barrier_axes(queue, resolution);
  iree_hal_amdgpu_host_queue_commit_signals(queue, signal_semaphore_list);
  if (post_commit_fn) {
    post_commit_fn(post_commit_user_data,
                   iree_hal_amdgpu_host_queue_const_frontier(queue));
  }
  if (complete_with_wait_barrier) {
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
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_static_buffer_ptr(
    iree_hal_command_buffer_t* command_buffer, const uint32_t ordinal,
    const uint64_t offset, uint8_t** out_device_ptr) {
  *out_device_ptr = NULL;
  iree_hal_buffer_t* buffer =
      iree_hal_amdgpu_aql_command_buffer_static_buffer(command_buffer, ordinal);
  if (IREE_UNLIKELY(!buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AQL command-buffer static buffer ordinal %" PRIu32
                            " is invalid",
                            ordinal);
  }
  iree_hal_buffer_t* allocated_buffer =
      iree_hal_buffer_allocated_buffer(buffer);
  uint8_t* device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_buffer);
  if (IREE_UNLIKELY(!device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "static command-buffer buffer must be backed by an AMDGPU allocation");
  }
  *out_device_ptr = device_ptr + iree_hal_buffer_byte_offset(buffer) + offset;
  return iree_ok_status();
}

static iree_hal_amdgpu_aql_packet_control_t
iree_hal_amdgpu_host_queue_command_buffer_packet_control(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    uint32_t packet_index, bool is_final_packet) {
  const iree_hsa_fence_scope_t acquire_scope =
      packet_index == 0
          ? iree_hal_amdgpu_host_queue_max_fence_scope(
                IREE_HSA_FENCE_SCOPE_AGENT, resolution->inline_acquire_scope)
          : IREE_HSA_FENCE_SCOPE_AGENT;
  const iree_hsa_fence_scope_t release_scope =
      is_final_packet ? iree_hal_amdgpu_host_queue_signal_list_release_scope(
                            queue, signal_semaphore_list)
                      : IREE_HSA_FENCE_SCOPE_AGENT;
  return iree_hal_amdgpu_aql_packet_control_barrier(acquire_scope,
                                                    release_scope);
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_fill_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_fill_command_t* fill_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  if (IREE_UNLIKELY(fill_command->target_kind !=
                    IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AQL command-buffer dynamic fill bindings not yet wired");
  }

  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_static_buffer_ptr(
      command_buffer, fill_command->target_ordinal, fill_command->target_offset,
      &target_ptr));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_emplace(
          queue->transfer_context, &packet->dispatch, target_ptr,
          fill_command->length, fill_command->pattern,
          fill_command->pattern_length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer fill dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_copy_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_copy_command_t* copy_command,
    iree_hal_amdgpu_aql_packet_t* packet,
    iree_hal_amdgpu_kernarg_block_t* kernarg_block,
    iree_hsa_signal_t completion_signal, uint16_t* out_setup) {
  if (IREE_UNLIKELY(copy_command->source_kind !=
                        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC ||
                    copy_command->target_kind !=
                        IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AQL command-buffer dynamic copy bindings not yet wired");
  }

  uint8_t* source_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_static_buffer_ptr(
      command_buffer, copy_command->source_ordinal, copy_command->source_offset,
      &source_ptr));
  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_static_buffer_ptr(
      command_buffer, copy_command->target_ordinal, copy_command->target_offset,
      &target_ptr));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &packet->dispatch, source_ptr, target_ptr,
          copy_command->length, kernarg_block->data))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer copy dispatch shape");
  }
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_host_size_t iree_hal_amdgpu_host_queue_update_kernarg_length(
    uint32_t source_length) {
  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  return source_payload_offset + (iree_host_size_t)source_length;
}

static iree_status_t iree_hal_amdgpu_host_queue_replay_update_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint8_t* kernarg_data,
    iree_host_size_t kernarg_length, iree_hsa_signal_t completion_signal,
    uint16_t* out_setup) {
  if (IREE_UNLIKELY(update_command->target_kind !=
                    IREE_HAL_AMDGPU_COMMAND_BUFFER_BINDING_KIND_STATIC)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "AQL command-buffer dynamic update bindings not yet wired");
  }

  uint8_t* target_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_resolve_static_buffer_ptr(
      command_buffer, update_command->target_ordinal,
      update_command->target_offset, &target_ptr));
  const uint8_t* source_bytes = iree_hal_amdgpu_aql_command_buffer_rodata(
      command_buffer, update_command->rodata_offset, update_command->length);
  if (IREE_UNLIKELY(!source_bytes)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer update rodata range is invalid");
  }

  const iree_host_size_t source_payload_offset =
      IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_OFFSET;
  const iree_host_size_t required_kernarg_length =
      source_payload_offset + (iree_host_size_t)update_command->length;
  if (IREE_UNLIKELY(required_kernarg_length > kernarg_length)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer update kernarg range is too small");
  }

  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &packet->dispatch,
          (const void*)(uintptr_t)
              IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_STAGED_SOURCE_ALIGNMENT,
          target_ptr, update_command->length, &kernargs))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported command-buffer update dispatch shape");
  }

  uint8_t* staged_source_bytes = kernarg_data + source_payload_offset;
  memcpy(kernarg_data, &kernargs, sizeof(kernargs));
  ((iree_hal_amdgpu_device_buffer_copy_kernargs_t*)kernarg_data)->source_ptr =
      staged_source_bytes;
  memcpy(staged_source_bytes, source_bytes, update_command->length);
  packet->dispatch.kernarg_address = kernarg_data;
  packet->dispatch.completion_signal = completion_signal;
  *out_setup = packet->dispatch.setup;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_validate_metadata_commands(
    const iree_hal_amdgpu_aql_program_t* program) {
  const iree_hal_amdgpu_command_buffer_block_header_t* block =
      program->first_block;
  bool reached_return = false;
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) && !reached_return && block) {
    const iree_hal_amdgpu_command_buffer_command_header_t* command =
        iree_hal_amdgpu_command_buffer_block_commands_const(block);
    bool advanced_block = false;
    for (uint16_t i = 0;
         i < block->command_count && iree_status_is_ok(status) &&
         !reached_return && !advanced_block;
         ++i) {
      switch (command->opcode) {
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
          break;
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH: {
          const iree_hal_amdgpu_command_buffer_branch_command_t*
              branch_command =
                  (const iree_hal_amdgpu_command_buffer_branch_command_t*)
                      command;
          iree_hal_amdgpu_command_buffer_block_header_t* next_block =
              iree_hal_amdgpu_aql_program_block_next(program->block_pool,
                                                     block);
          if (IREE_UNLIKELY(!next_block ||
                            branch_command->target_block_ordinal !=
                                next_block->block_ordinal)) {
            status = iree_make_status(
                IREE_STATUS_UNIMPLEMENTED,
                "non-linear AQL command-buffer branch replay not yet wired");
          } else {
            block = next_block;
            advanced_block = true;
          }
          break;
        }
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
          reached_return = true;
          break;
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
        case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
          status = iree_make_status(
              IREE_STATUS_UNIMPLEMENTED,
              "AQL command-buffer opcode %u replay not yet wired",
              command->opcode);
          break;
        default:
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "malformed AQL command-buffer opcode %u",
                                    command->opcode);
          break;
      }
      if (iree_status_is_ok(status) && !reached_return && !advanced_block) {
        command = iree_hal_amdgpu_command_buffer_command_next_const(command);
      }
    }
    if (iree_status_is_ok(status) && !reached_return && !advanced_block) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "AQL command-buffer block %" PRIu32
                                " has no terminator",
                                block->block_ordinal);
    }
  }
  if (iree_status_is_ok(status) && !reached_return) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer program has no return");
  }
  return status;
}

#if !defined(NDEBUG)
static iree_status_t iree_hal_amdgpu_host_queue_check_update_packet_command(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_update_command_t* update_command,
    iree_hal_amdgpu_aql_packet_t* packet, uint16_t* out_setup) {
  const iree_host_size_t kernarg_length =
      iree_hal_amdgpu_host_queue_update_kernarg_length(update_command->length);
  const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  iree_host_size_t kernarg_block_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          kernarg_block_count, sizeof(iree_hal_amdgpu_kernarg_block_t),
          &kernarg_block_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AQL command-buffer update debug scratch size overflow");
  }

  iree_hal_amdgpu_kernarg_block_t* update_kernarg_blocks = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(queue->host_allocator,
                                             kernarg_block_length,
                                             (void**)&update_kernarg_blocks));
  memset(update_kernarg_blocks, 0, kernarg_block_length);
  iree_status_t status = iree_hal_amdgpu_host_queue_replay_update_packet_body(
      queue, command_buffer, update_command, packet,
      update_kernarg_blocks->data, kernarg_block_length, iree_hsa_signal_null(),
      out_setup);
  iree_allocator_free(queue->host_allocator, update_kernarg_blocks);
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_check_packet_commands(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block) {
  iree_hal_amdgpu_aql_packet_t packet;
  iree_hal_amdgpu_kernarg_block_t kernarg_block;
  memset(&packet, 0, sizeof(packet));
  memset(&kernarg_block, 0, sizeof(kernarg_block));

  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  bool reached_return = false;
  uint32_t packet_count = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < block->command_count && iree_status_is_ok(status) && !reached_return;
       ++i) {
    uint16_t setup = 0;
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
        status = iree_hal_amdgpu_host_queue_replay_fill_packet_body(
            queue, command_buffer,
            (const iree_hal_amdgpu_command_buffer_fill_command_t*)command,
            &packet, &kernarg_block, iree_hsa_signal_null(), &setup);
        if (iree_status_is_ok(status)) ++packet_count;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY:
        status = iree_hal_amdgpu_host_queue_replay_copy_packet_body(
            queue, command_buffer,
            (const iree_hal_amdgpu_command_buffer_copy_command_t*)command,
            &packet, &kernarg_block, iree_hsa_signal_null(), &setup);
        if (iree_status_is_ok(status)) ++packet_count;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE: {
        const iree_hal_amdgpu_command_buffer_update_command_t* update_command =
            (const iree_hal_amdgpu_command_buffer_update_command_t*)command;
        status = iree_hal_amdgpu_host_queue_check_update_packet_command(
            queue, command_buffer, update_command, &packet, &setup);
        if (iree_status_is_ok(status)) ++packet_count;
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        reached_return = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "multi-block packet-bearing AQL command-buffer replay not yet "
            "wired");
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AQL command-buffer opcode %u replay not yet wired",
            command->opcode);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malformed AQL command-buffer opcode %u",
                                  command->opcode);
        break;
    }
    if (iree_status_is_ok(status) && !reached_return) {
      command = iree_hal_amdgpu_command_buffer_command_next_const(command);
    }
  }
  if (iree_status_is_ok(status) && !reached_return) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no return terminator",
                              block->block_ordinal);
  }
  if (iree_status_is_ok(status) && packet_count != block->aql_packet_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " validates %" PRIu32
        " packets but declares %" PRIu32,
        block->block_ordinal, packet_count, block->aql_packet_count);
  }
  return status;
}
#endif  // !defined(NDEBUG)

static iree_status_t iree_hal_amdgpu_host_queue_write_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    uint64_t first_packet_id, iree_hal_amdgpu_kernarg_block_t* kernarg_blocks,
    uint32_t extra_noop_packet_count, uint16_t* packet_headers,
    uint16_t* packet_setups) {
  const iree_hal_amdgpu_command_buffer_command_header_t* command =
      iree_hal_amdgpu_command_buffer_block_commands_const(block);
  bool reached_return = false;
  uint32_t packet_index = 0;
  uint32_t kernarg_block_index = 0;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < block->command_count && iree_status_is_ok(status) && !reached_return;
       ++i) {
    switch (command->opcode) {
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BARRIER:
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COPY: {
        const bool is_final_packet =
            packet_index + 1 == block->aql_packet_count;
        iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
            &queue->aql_ring, first_packet_id + resolution->barrier_count +
                                  extra_noop_packet_count + packet_index);
        const iree_hsa_signal_t completion_signal =
            is_final_packet ? iree_hal_amdgpu_notification_ring_epoch_signal(
                                  &queue->notification_ring)
                            : iree_hsa_signal_null();
        if (command->opcode == IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_FILL) {
          status = iree_hal_amdgpu_host_queue_replay_fill_packet_body(
              queue, command_buffer,
              (const iree_hal_amdgpu_command_buffer_fill_command_t*)command,
              packet, &kernarg_blocks[kernarg_block_index], completion_signal,
              &packet_setups[packet_index]);
        } else {
          status = iree_hal_amdgpu_host_queue_replay_copy_packet_body(
              queue, command_buffer,
              (const iree_hal_amdgpu_command_buffer_copy_command_t*)command,
              packet, &kernarg_blocks[kernarg_block_index], completion_signal,
              &packet_setups[packet_index]);
        }
        if (iree_status_is_ok(status)) {
          packet_headers[packet_index] = iree_hal_amdgpu_aql_make_header(
              IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
              iree_hal_amdgpu_host_queue_command_buffer_packet_control(
                  queue, resolution, signal_semaphore_list, packet_index,
                  is_final_packet));
          ++packet_index;
          ++kernarg_block_index;
        }
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_UPDATE: {
        const iree_hal_amdgpu_command_buffer_update_command_t* update_command =
            (const iree_hal_amdgpu_command_buffer_update_command_t*)command;
        const bool is_final_packet =
            packet_index + 1 == block->aql_packet_count;
        iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
            &queue->aql_ring, first_packet_id + resolution->barrier_count +
                                  extra_noop_packet_count + packet_index);
        const iree_hsa_signal_t completion_signal =
            is_final_packet ? iree_hal_amdgpu_notification_ring_epoch_signal(
                                  &queue->notification_ring)
                            : iree_hsa_signal_null();
        const iree_host_size_t kernarg_length =
            iree_hal_amdgpu_host_queue_update_kernarg_length(
                update_command->length);
        const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
            kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
        status = iree_hal_amdgpu_host_queue_replay_update_packet_body(
            queue, command_buffer, update_command, packet,
            kernarg_blocks[kernarg_block_index].data,
            kernarg_block_count * sizeof(iree_hal_amdgpu_kernarg_block_t),
            completion_signal, &packet_setups[packet_index]);
        if (iree_status_is_ok(status)) {
          packet_headers[packet_index] = iree_hal_amdgpu_aql_make_header(
              IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
              iree_hal_amdgpu_host_queue_command_buffer_packet_control(
                  queue, resolution, signal_semaphore_list, packet_index,
                  is_final_packet));
          ++packet_index;
          kernarg_block_index += (uint32_t)kernarg_block_count;
        }
        break;
      }
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_RETURN:
        reached_return = true;
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "multi-block packet-bearing AQL command-buffer replay not yet "
            "wired");
        break;
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_DISPATCH:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_PROFILE_MARKER:
      case IREE_HAL_AMDGPU_COMMAND_BUFFER_OPCODE_COND_BRANCH:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "AQL command-buffer opcode %u replay not yet wired",
            command->opcode);
        break;
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "malformed AQL command-buffer opcode %u",
                                  command->opcode);
        break;
    }
    if (iree_status_is_ok(status) && !reached_return) {
      command = iree_hal_amdgpu_command_buffer_command_next_const(command);
    }
  }
  if (iree_status_is_ok(status) && !reached_return) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AQL command-buffer block %" PRIu32
                              " has no return terminator",
                              block->block_ordinal);
  }
  if (iree_status_is_ok(status) && packet_index != block->aql_packet_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
        " packets but declares %" PRIu32,
        block->block_ordinal, packet_index, block->aql_packet_count);
  }
  if (iree_status_is_ok(status) &&
      kernarg_block_index !=
          iree_host_size_ceil_div(block->kernarg_length,
                                  sizeof(iree_hal_amdgpu_kernarg_block_t))) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AQL command-buffer block %" PRIu32 " emitted %" PRIu32
        " kernarg blocks but declares %" PRIu32 " kernarg bytes",
        block->block_ordinal, kernarg_block_index, block->kernarg_length);
  }
  return status;
}

static void iree_hal_amdgpu_host_queue_finish_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_reclaim_entry_t* reclaim_entry,
    iree_hal_resource_t** reclaim_resources, uint16_t reclaim_resource_count,
    uint64_t kernarg_write_position, uint64_t first_packet_id,
    uint32_t aql_packet_count, const uint16_t* packet_headers,
    const uint16_t* packet_setups, uint32_t extra_noop_packet_count) {
  const bool retain_submission_resources = iree_any_bit_set(
      submission_flags,
      IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);

  iree_hal_amdgpu_host_queue_emit_barriers(queue, resolution, first_packet_id);
  if (extra_noop_packet_count > 0) {
    iree_hal_amdgpu_host_queue_fill_noop_packets(
        queue, first_packet_id + resolution->barrier_count,
        extra_noop_packet_count);
  }

  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    reclaim_resources[i] =
        (iree_hal_resource_t*)signal_semaphore_list.semaphores[i];
    if (retain_submission_resources) {
      iree_hal_resource_retain(reclaim_resources[i]);
    }
  }
  reclaim_resources[signal_semaphore_list.count] =
      (iree_hal_resource_t*)command_buffer;
  if (retain_submission_resources) {
    iree_hal_resource_retain(reclaim_resources[signal_semaphore_list.count]);
  }
  reclaim_entry->kernarg_write_position = kernarg_write_position;
  reclaim_entry->count = reclaim_resource_count;

  iree_hal_amdgpu_host_queue_merge_barrier_axes(queue, resolution);
  iree_hal_amdgpu_host_queue_commit_signals(queue, signal_semaphore_list);
  for (uint32_t i = 0; i < block->aql_packet_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet = iree_hal_amdgpu_aql_ring_packet(
        &queue->aql_ring, first_packet_id + resolution->barrier_count +
                              extra_noop_packet_count + i);
    iree_hal_amdgpu_aql_ring_commit(packet, packet_headers[i],
                                    packet_setups[i]);
  }
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                    first_packet_id + aql_packet_count - 1);
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer_block(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    const iree_hal_amdgpu_command_buffer_block_header_t* block,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  const uint32_t kernarg_block_count = (uint32_t)iree_host_size_ceil_div(
      block->kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (IREE_UNLIKELY(kernarg_block_count < block->aql_packet_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command-buffer block declares %" PRIu32
                            " AQL packets but only %" PRIu32 " kernarg blocks",
                            block->aql_packet_count, kernarg_block_count);
  }
  const uint32_t extra_noop_packet_count =
      kernarg_block_count - block->aql_packet_count;
  const uint64_t packet_count =
      (uint64_t)resolution->barrier_count + kernarg_block_count;
  const uint64_t aql_queue_capacity = (uint64_t)queue->aql_ring.mask + 1;
  if (IREE_UNLIKELY(packet_count > aql_queue_capacity ||
                    packet_count > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command-buffer block requires %" PRIu64
                            " AQL packets (%u wait barriers + %" PRIu32
                            " kernarg blocks) but queue capacity is %" PRIu64,
                            packet_count, resolution->barrier_count,
                            kernarg_block_count, aql_queue_capacity);
  }
#if !defined(NDEBUG)
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_check_packet_commands(
      queue, command_buffer, block));
#endif  // !defined(NDEBUG)

  uint16_t reclaim_resource_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list.count,
      /*operation_resource_count=*/1, &reclaim_resource_count));
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_notification_ring_reserve(
      &queue->notification_ring, signal_semaphore_list.count,
      iree_hal_amdgpu_host_queue_count_frontier_snapshots(
          queue, signal_semaphore_list)));

  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry =
      iree_hal_amdgpu_notification_ring_reclaim_entry(
          &queue->notification_ring);
  iree_hal_resource_t** reclaim_resources = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_reclaim_entry_prepare(
      reclaim_entry, queue->block_pool, reclaim_resource_count,
      &reclaim_resources));

  const uint32_t aql_packet_count = (uint32_t)packet_count;
  const uint64_t first_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, aql_packet_count);
  uint64_t kernarg_write_position = 0;
  iree_hal_amdgpu_kernarg_block_t* kernarg_blocks =
      iree_hal_amdgpu_kernarg_ring_allocate(
          &queue->kernarg_ring, kernarg_block_count, &kernarg_write_position);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(!kernarg_blocks)) {
    iree_hal_amdgpu_host_queue_emit_noop_packets(queue, first_packet_id,
                                                 aql_packet_count);
    iree_hal_amdgpu_reclaim_entry_release(reclaim_entry, queue->block_pool);
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "kernarg ring allocation failed after AQL reservation; queue sizing "
        "invariant was violated");
  }

  if (iree_status_is_ok(status)) {
    uint16_t* packet_headers =
        (uint16_t*)iree_alloca(block->aql_packet_count * sizeof(uint16_t));
    uint16_t* packet_setups =
        (uint16_t*)iree_alloca(block->aql_packet_count * sizeof(uint16_t));
    memset(packet_headers, 0, block->aql_packet_count * sizeof(uint16_t));
    memset(packet_setups, 0, block->aql_packet_count * sizeof(uint16_t));
    status = iree_hal_amdgpu_host_queue_write_command_buffer_block(
        queue, resolution, signal_semaphore_list, command_buffer, block,
        first_packet_id, kernarg_blocks, extra_noop_packet_count,
        packet_headers, packet_setups);
    if (iree_status_is_ok(status)) {
      iree_hal_amdgpu_host_queue_finish_command_buffer_block(
          queue, resolution, signal_semaphore_list, command_buffer, block,
          submission_flags, reclaim_entry, reclaim_resources,
          reclaim_resource_count, kernarg_write_position, first_packet_id,
          aql_packet_count, packet_headers, packet_setups,
          extra_noop_packet_count);
    } else {
      iree_hal_amdgpu_host_queue_emit_reclaim_noop_packets(
          queue, reclaim_entry, first_packet_id, aql_packet_count,
          kernarg_write_position);
    }
  }
  return status;
}

iree_status_t iree_hal_amdgpu_host_queue_submit_command_buffer(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(resolution);
  (void)binding_table;

  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }
  if (IREE_UNLIKELY(!command_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command buffer is required");
  }
  if (IREE_UNLIKELY(!iree_hal_amdgpu_aql_command_buffer_isa(command_buffer))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command buffer is not an AMDGPU AQL command "
                            "buffer");
  }

  const iree_hal_amdgpu_aql_program_t* program =
      iree_hal_amdgpu_aql_command_buffer_program(command_buffer);
  if (IREE_UNLIKELY(!program->first_block)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer has not been finalized");
  }

  if (program->max_block_aql_packet_count == 0) {
    IREE_RETURN_IF_ERROR(
        iree_hal_amdgpu_host_queue_validate_metadata_commands(program));

    // A metadata-only replay still needs one completion packet so notification
    // drain can advance the queue epoch and publish the user-visible signals.
    iree_hal_resource_t* command_buffer_resource =
        (iree_hal_resource_t*)command_buffer;
    return iree_hal_amdgpu_host_queue_submit_barrier(
        queue, resolution, signal_semaphore_list,
        (iree_hal_amdgpu_reclaim_action_t){0}, &command_buffer_resource,
        /*operation_resource_count=*/1,
        /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
        submission_flags);
  }

  if (IREE_UNLIKELY(program->block_count != 1)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "multi-block packet-bearing AQL command-buffer replay not yet wired");
  }
  return iree_hal_amdgpu_host_queue_submit_command_buffer_block(
      queue, resolution, signal_semaphore_list, command_buffer,
      program->first_block, submission_flags);
}
