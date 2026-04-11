// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_submission.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

// Returns a conservative upper bound on the number of frontier snapshots that
// commit_signals will push for |signal_semaphore_list|.
//
// Caller must hold submission_mutex.
static iree_host_size_t iree_hal_amdgpu_host_queue_count_frontier_snapshots(
    const iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list) {
  iree_host_size_t snapshot_count = 0;
  iree_async_semaphore_t* last_semaphore = queue->last_signal.semaphore;
  for (iree_host_size_t i = 0; i < signal_semaphore_list.count; ++i) {
    iree_async_semaphore_t* semaphore =
        (iree_async_semaphore_t*)signal_semaphore_list.semaphores[i];
    if (semaphore == last_semaphore) continue;
    if (last_semaphore != NULL) {
      ++snapshot_count;
    }
    last_semaphore = semaphore;
  }
  return snapshot_count;
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
        iree_hal_amdgpu_aql_packet_control_barrier_system(),
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

    // Detect semaphore transition for frontier snapshot recording.
    if (async_semaphore != queue->last_signal.semaphore) {
      if (queue->last_signal.semaphore != NULL && queue->can_publish_frontier) {
        iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
            &queue->notification_ring, queue->last_signal.epoch,
            queue_frontier);
      }
      queue->last_signal.semaphore = async_semaphore;
    }

    // Push notification entry for drain -> signal_untainted on completion.
    iree_hal_amdgpu_notification_ring_push(&queue->notification_ring, epoch,
                                           async_semaphore, value);

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

  queue->last_signal.epoch = epoch;
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

  iree_hal_amdgpu_host_queue_merge_barrier_axes(queue, resolution);
  iree_hal_amdgpu_host_queue_commit_signals(queue, signal_semaphore_list);

  uint16_t dispatch_header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_aql_packet_control_barrier_system());
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
              iree_hsa_signal_null(), barrier_packet, &barrier_setup);
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
    completion_header =
        iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
            queue, &resolution->barriers[resolution->barrier_count - 1],
            first_packet_id + aql_packet_count - 1,
            iree_hal_amdgpu_notification_ring_epoch_signal(
                &queue->notification_ring),
            completion_slot, &completion_setup);
  } else {
    completion_header = iree_hal_amdgpu_aql_emit_nop(
        &completion_slot->barrier_and,
        iree_hal_amdgpu_aql_packet_control_barrier_system(),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &queue->notification_ring));
  }
  iree_hal_amdgpu_aql_ring_commit(completion_slot, completion_header,
                                  completion_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring,
                                    first_packet_id + aql_packet_count - 1);
  return iree_ok_status();
}
