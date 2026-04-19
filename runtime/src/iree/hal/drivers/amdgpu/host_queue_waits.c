// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_waits.h"

#include <string.h>

#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/pm4_emitter.h"

// Returns true if |frontier| contains |axis| at an epoch >= |target_epoch|.
// Frontier entries are sorted by axis, so the scan can stop as soon as the
// target axis has been passed.
static bool iree_hal_amdgpu_frontier_dominates_axis(
    const iree_async_frontier_t* frontier, iree_async_axis_t axis,
    uint64_t target_epoch) {
  for (uint8_t i = 0; i < frontier->entry_count; ++i) {
    const iree_async_frontier_entry_t* entry = &frontier->entries[i];
    if (entry->axis < axis) continue;
    return entry->axis == axis && entry->epoch >= target_epoch;
  }
  return false;
}

// Appends a tier-2 barrier to |resolution| while preserving ascending axis
// order and deduplicating repeated axes across multiple waits. If the barrier
// budget is exhausted, returns false so the caller can fall back to software
// deferral instead of relying on a debug-only assert.
//
// Caller must hold submission_mutex.
static bool iree_hal_amdgpu_host_queue_append_wait_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_wait_resolution_t* resolution, iree_async_axis_t axis,
    hsa_signal_t epoch_signal, uint64_t target_epoch,
    iree_hsa_fence_scope_t acquire_scope) {
  if (queue->wait_barrier_strategy ==
      IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER) {
    return false;
  }

  resolution->barrier_acquire_scope =
      iree_hal_amdgpu_host_queue_max_fence_scope(
          resolution->barrier_acquire_scope, acquire_scope);

  uint8_t insert_ordinal = 0;
  while (insert_ordinal < resolution->barrier_count &&
         resolution->barriers[insert_ordinal].axis < axis) {
    ++insert_ordinal;
  }

  if (insert_ordinal < resolution->barrier_count &&
      resolution->barriers[insert_ordinal].axis == axis) {
    if (target_epoch > resolution->barriers[insert_ordinal].target_epoch) {
      resolution->barriers[insert_ordinal].target_epoch = target_epoch;
      resolution->barriers[insert_ordinal].epoch_signal = epoch_signal;
    }
    return true;
  }

  if (resolution->barrier_count >= IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY) {
    return false;
  }

  for (uint8_t i = resolution->barrier_count; i > insert_ordinal; --i) {
    resolution->barriers[i] = resolution->barriers[i - 1];
  }

  iree_hal_amdgpu_wait_barrier_t* barrier =
      &resolution->barriers[insert_ordinal];
  barrier->axis = axis;
  barrier->epoch_signal = epoch_signal;
  barrier->target_epoch = target_epoch;
  ++resolution->barrier_count;
  return true;
}

// Resolves a single (semaphore, value) wait. Appends tier 2 barriers to
// |resolution| if needed. Returns true if the wait is resolved (satisfied or
// barriers appended), false if deferral is needed.
//
// Tier 0: timeline_value >= value -> already completed.
// Tier 1a: signal submitted by this queue -> elide directly from last_signal
//   under this strategy's current all-barrier AQL policy, no semaphore-frontier
//   mutex/copy.
// Tier 1b: signal submitted by a producer epoch that exactly covers the
//   semaphore frontier, and this queue already dominates that producer epoch
//   -> elide directly from last_signal.
// Tier 1c: signal submitted + queue frontier dominates -> no barrier needed.
// Tier 2a: signal submitted by a local producer epoch that exactly covers the
//   semaphore frontier -> append one barrier directly from last_signal.
// Tier 2b: signal submitted + local queue axes from semaphore frontier ->
//   barriers appended from the undominated frontier entries.
// Tier 3: anything else -> deferral.
//
// Caller must hold submission_mutex.
static bool iree_hal_amdgpu_host_queue_resolve_wait(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_hal_amdgpu_wait_resolution_t* resolution) {
  iree_async_semaphore_t* async_semaphore = (iree_async_semaphore_t*)semaphore;
  iree_hsa_fence_scope_t acquire_scope =
      iree_hal_amdgpu_host_queue_wait_acquire_scope(queue, semaphore);

  // A failed semaphore must take the software deferral path so the timepoint
  // callback propagates the failure to this op's signal semaphores.
  if (iree_async_semaphore_query_status(async_semaphore) != IREE_STATUS_OK) {
    return false;
  }

  // Tier 0: already completed. Cheapest check (one atomic load).
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) {
    resolution->inline_acquire_scope =
        iree_hal_amdgpu_host_queue_max_fence_scope(
            resolution->inline_acquire_scope, acquire_scope);
    return true;
  }

  // Not completed. Must be an AMDGPU semaphore for device-side resolution.
  if (!iree_hal_amdgpu_semaphore_isa(semaphore)) return false;

  // Has the signal for |value| been submitted? The last_signal cache records
  // the most recent signal's value. If it hasn't reached |value|, the signal
  // hasn't been submitted yet (wait-before-signal) and the frontier does not
  // reflect the signal's causal context - frontier dominance would be a
  // false positive.
  iree_hal_amdgpu_last_signal_flags_t signal_flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t signal_axis = 0;
  uint64_t signal_epoch = 0;
  uint64_t signal_value = 0;
  if (!iree_hal_amdgpu_last_signal_load(
          iree_hal_amdgpu_semaphore_last_signal(semaphore), &signal_flags,
          &signal_axis, &signal_epoch, &signal_value) ||
      signal_value < value) {
    return false;
  }

  // Tier 1a: same-queue elision from the last_signal cache alone.
  // HAL queues are not FIFO: user-visible order comes from semaphore edges.
  // This shortcut is valid only because AMDGPU queue submissions represent
  // inline waits on the first payload packet with AQL BARRIER set, so submission
  // order under submission_mutex creates a single in-queue dependency chain. If
  // that policy is relaxed for independent HIP streams, this branch must emit
  // an explicit same-queue dependency edge instead of returning purely from
  // producer axis identity.
  if (signal_axis == queue->axis) {
    resolution->inline_acquire_scope =
        iree_hal_amdgpu_host_queue_max_fence_scope(
            resolution->inline_acquire_scope, acquire_scope);
    return true;
  }

  // Tier 1b/2a: when the semaphore cache says the producer queue's epoch
  // exactly covers the unresolved semaphore frontier, resolve directly from
  // that producer axis/epoch snapshot. This avoids the semaphore-frontier
  // mutex/copy on common cross-queue handoffs while still refusing to guess
  // on TP fan-in semaphores with independent producers.
  if (signal_flags & IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT) {
    if (iree_hal_amdgpu_frontier_dominates_axis(
            iree_hal_amdgpu_host_queue_const_frontier(queue), signal_axis,
            signal_epoch)) {
      resolution->inline_acquire_scope =
          iree_hal_amdgpu_host_queue_max_fence_scope(
              resolution->inline_acquire_scope, acquire_scope);
      return true;
    }
    hsa_signal_t peer_signal;
    if (!iree_hal_amdgpu_epoch_signal_table_lookup(queue->epoch_table,
                                                   signal_axis, &peer_signal)) {
      return false;
    }
    return iree_hal_amdgpu_host_queue_append_wait_barrier(
        queue, resolution, signal_axis, peer_signal, signal_epoch,
        acquire_scope);
  }

  // Signal submitted, not completed. Copy the semaphore's frontier and find
  // axes that our queue doesn't dominate.
  iree_hal_amdgpu_fixed_frontier_t semaphore_frontier;
  iree_async_semaphore_query_frontier(
      async_semaphore,
      iree_hal_amdgpu_fixed_frontier_as_frontier(&semaphore_frontier),
      IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY);

  iree_async_frontier_entry_t
      undominated[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
  uint8_t undominated_count = iree_async_frontier_find_undominated(
      iree_hal_amdgpu_host_queue_const_frontier(queue),
      iree_hal_amdgpu_fixed_frontier_as_frontier(&semaphore_frontier),
      IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY, undominated);

  // Tier 1c: all axes dominated -> no additional barrier needed.
  if (undominated_count == 0) {
    resolution->inline_acquire_scope =
        iree_hal_amdgpu_host_queue_max_fence_scope(
            resolution->inline_acquire_scope, acquire_scope);
    return true;
  }

  // Tier 2b: look up each undominated axis in the epoch signal table.
  // If any axis is not a local queue (remote, collective, host), defer.
  for (uint8_t i = 0; i < undominated_count; ++i) {
    hsa_signal_t peer_signal;
    if (!iree_hal_amdgpu_epoch_signal_table_lookup(
            queue->epoch_table, undominated[i].axis, &peer_signal)) {
      return false;
    }
    if (!iree_hal_amdgpu_host_queue_append_wait_barrier(
            queue, resolution, undominated[i].axis, peer_signal,
            undominated[i].epoch, acquire_scope)) {
      return false;
    }
  }
  return true;
}

void iree_hal_amdgpu_host_queue_resolve_waits(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_wait_resolution_t* out_resolution) {
  out_resolution->barrier_count = 0;
  out_resolution->needs_deferral = false;
  memset(out_resolution->reserved, 0, sizeof(out_resolution->reserved));
  out_resolution->inline_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;
  out_resolution->barrier_acquire_scope = IREE_HSA_FENCE_SCOPE_NONE;

  for (iree_host_size_t i = 0; i < wait_semaphore_list.count; ++i) {
    if (!iree_hal_amdgpu_host_queue_resolve_wait(
            queue, wait_semaphore_list.semaphores[i],
            wait_semaphore_list.payload_values[i], out_resolution)) {
      out_resolution->needs_deferral = true;
      out_resolution->barrier_count = 0;
      return;
    }
  }
}

uint16_t iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_barrier_t* barrier, uint64_t packet_id,
    hsa_signal_t completion_signal, iree_hsa_fence_scope_t acquire_scope,
    iree_hsa_fence_scope_t release_scope, iree_hal_amdgpu_aql_packet_t* packet,
    uint16_t* out_setup) {
  // The epoch signal starts at INITIAL_VALUE and is decremented by 1 per
  // completion. A submission at one-based epoch N is complete when the queue's
  // current epoch has reached N, so the barrier fires when:
  //   signal_load(s) <= INITIAL_VALUE - target_epoch
  // BARRIER_VALUE only supports LT, so encode <= as:
  //   signal_load(s) < INITIAL_VALUE - target_epoch + 1
  //
  // Epochs are one-based by construction (see notification_ring_advance_epoch).
  // Plugging target_epoch == 0 into the formula collapses to
  // "signal < INITIAL_VALUE + 1", which is trivially true and would let a wait
  // for "no submission yet" fire immediately. Reserving zero for that state
  // keeps the wait formula safe without any special-case branches here.
  iree_hsa_signal_value_t compare_value =
      (iree_hsa_signal_value_t)(IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE -
                                barrier->target_epoch + 1);

  switch (queue->wait_barrier_strategy) {
    case IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE:
      return iree_hal_amdgpu_aql_emit_barrier_value(
          &packet->barrier_value,
          (iree_hsa_signal_t){.handle = barrier->epoch_signal.handle},
          IREE_HSA_SIGNAL_CONDITION_LT, compare_value,
          (iree_hsa_signal_value_t)INT64_MAX,
          iree_hal_amdgpu_aql_packet_control_barrier(acquire_scope,
                                                     release_scope),
          completion_signal, out_setup);
    case IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64: {
      IREE_ASSERT(queue->pm4_ib_slots != NULL);
      iree_hal_amdgpu_pm4_ib_slot_t* ib_slot =
          &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
      uint32_t ib_dword_count = iree_hal_amdgpu_pm4_emit_wait_reg_mem64(
          ib_slot, (iree_hsa_signal_t){.handle = barrier->epoch_signal.handle},
          compare_value, (iree_hsa_signal_value_t)INT64_MAX);
      return iree_hal_amdgpu_aql_emit_pm4_ib(
          &packet->pm4_ib, ib_slot, ib_dword_count,
          iree_hal_amdgpu_aql_packet_control_barrier(acquire_scope,
                                                     release_scope),
          completion_signal, out_setup);
    }
    case IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER:
    default:
      IREE_ASSERT(false,
                  "resolved wait barriers require a device-side strategy");
      *out_setup = 0;
      return 0;
  }
}

void iree_hal_amdgpu_host_queue_emit_barriers(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    uint64_t first_packet_id) {
  for (uint8_t i = 0; i < resolution->barrier_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, first_packet_id + i);
    uint16_t setup = 0;
    uint16_t header = iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
        queue, &resolution->barriers[i], first_packet_id + i,
        iree_hsa_signal_null(), resolution->barrier_acquire_scope,
        IREE_HSA_FENCE_SCOPE_NONE, packet, &setup);
    iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  }
}

void iree_hal_amdgpu_host_queue_merge_barrier_axes(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution) {
  if (resolution->barrier_count == 0 || !queue->can_publish_frontier) return;
  iree_hal_amdgpu_fixed_frontier_t barrier_frontier;
  iree_async_frontier_initialize(
      iree_hal_amdgpu_fixed_frontier_as_frontier(&barrier_frontier),
      resolution->barrier_count);
  for (uint8_t i = 0; i < resolution->barrier_count; ++i) {
    barrier_frontier.entries[i].axis = resolution->barriers[i].axis;
    barrier_frontier.entries[i].epoch = resolution->barriers[i].target_epoch;
  }
  if (!iree_async_frontier_merge(
          iree_hal_amdgpu_host_queue_frontier(queue),
          IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY,
          iree_hal_amdgpu_fixed_frontier_as_frontier(&barrier_frontier))) {
    queue->can_publish_frontier = false;
  }
}

const iree_async_frontier_t* iree_hal_amdgpu_host_queue_pool_requester_frontier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_amdgpu_fixed_frontier_t* storage) {
  const iree_async_frontier_t* queue_frontier =
      iree_hal_amdgpu_host_queue_const_frontier(queue);
  if (resolution->barrier_count == 0) return queue_frontier;

  memcpy(storage, &queue->frontier, sizeof(*storage));
  iree_hal_amdgpu_fixed_frontier_t barrier_frontier;
  iree_async_frontier_initialize(
      iree_hal_amdgpu_fixed_frontier_as_frontier(&barrier_frontier),
      resolution->barrier_count);
  for (uint8_t i = 0; i < resolution->barrier_count; ++i) {
    barrier_frontier.entries[i].axis = resolution->barriers[i].axis;
    barrier_frontier.entries[i].epoch = resolution->barriers[i].target_epoch;
  }
  if (!iree_async_frontier_merge(
          iree_hal_amdgpu_fixed_frontier_as_frontier(storage),
          IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY,
          iree_hal_amdgpu_fixed_frontier_as_frontier(&barrier_frontier))) {
    return queue_frontier;
  }
  return iree_hal_amdgpu_fixed_frontier_as_frontier(storage);
}

bool iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_async_frontier_t* requester_frontier,
    const iree_async_frontier_t* wait_frontier,
    iree_hal_amdgpu_wait_resolution_t* resolution) {
  if (!wait_frontier) return false;
  for (uint8_t i = 0; i < wait_frontier->entry_count; ++i) {
    const iree_async_frontier_entry_t* entry = &wait_frontier->entries[i];
    if (iree_hal_amdgpu_frontier_dominates_axis(requester_frontier, entry->axis,
                                                entry->epoch)) {
      continue;
    }
    hsa_signal_t peer_signal;
    if (!iree_hal_amdgpu_epoch_signal_table_lookup(queue->epoch_table,
                                                   entry->axis, &peer_signal)) {
      return false;
    }
    if (!iree_hal_amdgpu_host_queue_append_wait_barrier(
            queue, resolution, entry->axis, peer_signal, entry->epoch,
            iree_hal_amdgpu_host_queue_axis_acquire_scope(queue,
                                                          entry->axis))) {
      return false;
    }
  }
  return true;
}
