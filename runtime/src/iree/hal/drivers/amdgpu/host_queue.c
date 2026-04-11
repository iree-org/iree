// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdgpu/buffer.h"
#include "iree/hal/drivers/amdgpu/device/dispatch.h"
#include "iree/hal/drivers/amdgpu/executable.h"
#include "iree/hal/drivers/amdgpu/semaphore.h"
#include "iree/hal/drivers/amdgpu/transient_buffer.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"

// The inline frontier on host_queue_t uses the same {entry_count, reserved[7],
// entries[N]} layout as iree_async_frontier_t + FAM. Verify the entries field
// starts at the FAM offset so iree_hal_amdgpu_host_queue_frontier() is valid.
static_assert(offsetof(iree_hal_amdgpu_host_queue_t, frontier.entries) -
                      offsetof(iree_hal_amdgpu_host_queue_t, frontier) ==
                  sizeof(iree_async_frontier_t),
              "inline frontier entries must align with frontier_t FAM offset");

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable;
static void iree_hal_amdgpu_host_queue_commit_signals(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list);

typedef void(IREE_API_PTR* iree_hal_amdgpu_host_queue_post_commit_fn_t)(
    void* user_data, const iree_async_frontier_t* queue_frontier);

// Flags controlling submission helper ownership transfers.
typedef uint32_t iree_hal_amdgpu_host_queue_submission_flags_t;
enum iree_hal_amdgpu_host_queue_submission_flag_bits_t {
  IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE = 0u,
  // Retains signal semaphores and operation resources into the reclaim entry.
  // When omitted, the helper transfers one existing retain for each resource
  // from the caller into the reclaim entry on success.
  IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES = 1u << 0,
};

//===----------------------------------------------------------------------===//
// Fixed-capacity frontier storage
//===----------------------------------------------------------------------===//

// Stack-allocable frontier with fixed capacity. Layout-compatible with
// iree_async_frontier_t when cast — same {entry_count, reserved, entries[]}
// field layout with a fixed-size array in place of the FAM.
typedef struct iree_hal_amdgpu_fixed_frontier_t {
  uint8_t entry_count;
  uint8_t reserved[7];
  iree_async_frontier_entry_t entries[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
} iree_hal_amdgpu_fixed_frontier_t;

static_assert(offsetof(iree_hal_amdgpu_fixed_frontier_t, entries) ==
                  sizeof(iree_async_frontier_t),
              "fixed frontier entries must align with frontier_t FAM offset");

static inline iree_async_frontier_t* iree_hal_amdgpu_fixed_frontier_as_frontier(
    iree_hal_amdgpu_fixed_frontier_t* storage) {
  return (iree_async_frontier_t*)storage;
}

//===----------------------------------------------------------------------===//
// Wait resolution
//===----------------------------------------------------------------------===//

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

// A single barrier to emit as an AQL barrier-value packet. Produced by
// resolve_waits for each undominated axis that maps to a local queue.
typedef struct iree_hal_amdgpu_wait_barrier_t {
  iree_async_axis_t axis;
  hsa_signal_t epoch_signal;
  uint64_t target_epoch;
} iree_hal_amdgpu_wait_barrier_t;

// Queue-private PM4 IB slot. These are allocated only for queues using the
// PM4 WAIT_REG_MEM64 fallback. The slot count always matches the AQL ring
// capacity, so AQL packet id N uses pm4_ib_slots[N & aql_ring.mask].
struct iree_alignas(64) iree_hal_amdgpu_pm4_ib_slot_t {
  uint32_t dwords[16];
};
static_assert(sizeof(iree_hal_amdgpu_pm4_ib_slot_t) == 64,
              "PM4 IB slot must be exactly one cache line");

enum {
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER = 0x3F,
  IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64 = 0x93,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_LESS_THAN = 1,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY = 1,
  IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM = 0,
};

static const uint32_t
    IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE = 0x80000000u;

static inline uint32_t iree_hal_amdgpu_pm4_make_header(uint32_t opcode,
                                                       uint32_t dword_count) {
  return (3u << 30) | (opcode << 8) | ((dword_count - 2u) << 16);
}

static inline uint32_t iree_hal_amdgpu_pm4_addr_lo(uintptr_t address) {
  return (uint32_t)(address & 0xFFFFFFFCu);
}

static inline uint32_t iree_hal_amdgpu_pm4_addr_lo_8(uintptr_t address) {
  return (uint32_t)(address & 0xFFFFFFF8u);
}

static inline uint32_t iree_hal_amdgpu_pm4_addr_hi(uintptr_t address) {
  return (uint32_t)(address >> 32);
}

static inline uint32_t iree_hal_amdgpu_pm4_ib_addr_hi(uintptr_t address) {
  return (uint32_t)((address >> 32) & 0xFFFFu);
}

static inline uint32_t iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
    uint32_t function, uint32_t mem_space, uint32_t operation) {
  return (function & 0x7u) | ((mem_space & 0x3u) << 4) |
         ((operation & 0x3u) << 6);
}

// Result of resolving a wait_semaphore_list. Either all waits are resolved
// (barrier_count barriers to emit) or deferral is needed.
//
// barrier_count and needs_deferral are at the top so they share a cache line
// with barriers[0] — the caller checks deferral then iterates barriers[0..N].
typedef struct iree_hal_amdgpu_wait_resolution_t {
  uint8_t barrier_count;
  bool needs_deferral;
  uint8_t reserved[6];
  iree_hal_amdgpu_wait_barrier_t
      barriers[IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY];
} iree_hal_amdgpu_wait_resolution_t;

// Appends a tier-2 barrier to |resolution| while preserving ascending axis
// order and deduplicating repeated axes across multiple waits. If the barrier
// budget is exhausted, returns false so the caller can fall back to software
// deferral instead of relying on a debug-only assert.
//
// Caller must hold submission_mutex.
static bool iree_hal_amdgpu_host_queue_append_wait_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    iree_hal_amdgpu_wait_resolution_t* resolution, iree_async_axis_t axis,
    hsa_signal_t epoch_signal, uint64_t target_epoch) {
  if (queue->wait_barrier_strategy ==
      IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_DEFER) {
    return false;
  }

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
// Tier 0: timeline_value >= value → already completed.
// Tier 1a: signal submitted by this queue → elide directly from last_signal
//   under this strategy's current all-barrier AQL policy, no semaphore-frontier
//   mutex/copy.
// Tier 1b: signal submitted by a producer epoch that exactly covers the
//   semaphore frontier, and this queue already dominates that producer epoch
//   → elide directly from last_signal.
// Tier 1c: signal submitted + queue frontier dominates → no barrier needed.
// Tier 2a: signal submitted by a local producer epoch that exactly covers the
//   semaphore frontier → append one barrier directly from last_signal.
// Tier 2b: signal submitted + local queue axes from semaphore frontier →
//   barriers appended from the undominated frontier entries.
// Tier 3: anything else → deferral.
//
// Caller must hold submission_mutex.
static bool iree_hal_amdgpu_host_queue_resolve_wait(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_semaphore_t* semaphore,
    uint64_t value, iree_hal_amdgpu_wait_resolution_t* resolution) {
  iree_async_semaphore_t* async_semaphore = (iree_async_semaphore_t*)semaphore;

  // A failed semaphore must take the software deferral path so the timepoint
  // callback propagates the failure to this op's signal semaphores.
  if (iree_async_semaphore_query_status(async_semaphore) != IREE_STATUS_OK) {
    return false;
  }

  // Tier 0: already completed. Cheapest check (one atomic load).
  uint64_t current_value = (uint64_t)iree_atomic_load(
      &async_semaphore->timeline_value, iree_memory_order_acquire);
  if (current_value >= value) return true;

  // Not completed. Must be an AMDGPU semaphore for device-side resolution.
  if (!iree_hal_amdgpu_semaphore_isa(semaphore)) return false;

  // Has the signal for |value| been submitted? The last_signal cache records
  // the most recent signal's value. If it hasn't reached |value|, the signal
  // hasn't been submitted yet (wait-before-signal) and the frontier does not
  // reflect the signal's causal context — frontier dominance would be a
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
  // This shortcut is valid only because the current AMDGPU host queue emits all
  // AQL packets with BARRIER set, so submission order under submission_mutex
  // creates a single in-queue dependency chain. If that policy is relaxed for
  // independent HIP streams, this branch must emit an explicit same-queue
  // dependency edge instead of returning purely from producer axis identity.
  if (signal_axis == queue->axis) return true;

  // Tier 1b/2a: when the semaphore cache says the producer queue's epoch
  // exactly covers the unresolved semaphore frontier, resolve directly from
  // that producer axis/epoch snapshot. This avoids the semaphore-frontier
  // mutex/copy on common cross-queue handoffs while still refusing to guess
  // on TP fan-in semaphores with independent producers.
  if (signal_flags & IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT) {
    if (iree_hal_amdgpu_frontier_dominates_axis(
            iree_hal_amdgpu_host_queue_const_frontier(queue), signal_axis,
            signal_epoch)) {
      return true;
    }
    hsa_signal_t peer_signal;
    if (!iree_hal_amdgpu_epoch_signal_table_lookup(queue->epoch_table,
                                                   signal_axis, &peer_signal)) {
      return false;
    }
    return iree_hal_amdgpu_host_queue_append_wait_barrier(
        queue, resolution, signal_axis, peer_signal, signal_epoch);
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

  // Tier 1c: all axes dominated → no additional barrier needed.
  if (undominated_count == 0) return true;

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
            undominated[i].epoch)) {
      return false;
    }
  }
  return true;
}

// Resolves a wait_semaphore_list into barriers or deferral.
//
// All-or-nothing: if ANY wait is tier 3, sets needs_deferral and returns
// immediately. No partial barriers are emitted.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_resolve_waits(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_amdgpu_wait_resolution_t* out_resolution) {
  out_resolution->barrier_count = 0;
  out_resolution->needs_deferral = false;
  memset(out_resolution->reserved, 0, sizeof(out_resolution->reserved));

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

static uint32_t iree_hal_amdgpu_host_queue_emit_pm4_wait_reg_mem64(
    iree_hal_amdgpu_pm4_ib_slot_t* slot, hsa_signal_t epoch_signal,
    iree_hsa_signal_value_t compare_value, iree_hsa_signal_value_t mask) {
  memset(slot, 0, sizeof(*slot));
  iree_amd_signal_t* signal_abi = (iree_amd_signal_t*)epoch_signal.handle;
  volatile iree_hsa_signal_value_t* value_address = &signal_abi->value;
  const uintptr_t address = (uintptr_t)value_address;
  uint32_t* dword = slot->dwords;
  dword[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WAIT_REG_MEM64, 9);
  dword[1] = iree_hal_amdgpu_pm4_wait_reg_mem_dw1(
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_FUNC_LESS_THAN,
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_SPACE_MEMORY,
      IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPERATION_WAIT_REG_MEM);
  dword[2] = iree_hal_amdgpu_pm4_addr_lo_8(address);
  dword[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  dword[4] = (uint32_t)compare_value;
  dword[5] = (uint32_t)((uint64_t)compare_value >> 32);
  dword[6] = (uint32_t)mask;
  dword[7] = (uint32_t)((uint64_t)mask >> 32);
  dword[8] = 4 | IREE_HAL_AMDGPU_PM4_WAIT_REG_MEM_OPTIMIZE_ACE_OFFLOAD_MODE;
  return 9;
}

static uint16_t iree_hal_amdgpu_host_queue_write_pm4_ib_packet_body(
    iree_hsa_amd_aql_pm4_ib_packet_t* packet,
    const iree_hal_amdgpu_pm4_ib_slot_t* ib_slot, uint32_t ib_dword_count,
    iree_hal_amdgpu_aql_packet_control_t packet_control,
    hsa_signal_t completion_signal, uint16_t* out_setup) {
  const uintptr_t ib_address = (uintptr_t)ib_slot->dwords;
  packet->ib_jump_cmd[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_INDIRECT_BUFFER, 4);
  packet->ib_jump_cmd[1] = iree_hal_amdgpu_pm4_addr_lo(ib_address);
  packet->ib_jump_cmd[2] = iree_hal_amdgpu_pm4_ib_addr_hi(ib_address);
  packet->ib_jump_cmd[3] = (ib_dword_count & 0xFFFFFu) | (1u << 23);
  packet->dw_cnt_remain = 0xA;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(packet->reserved); ++i) {
    packet->reserved[i] = 0;
  }
  packet->completion_signal = completion_signal;
  *out_setup = IREE_HSA_AMD_AQL_FORMAT_PM4_IB;
  return iree_hal_amdgpu_aql_make_header(IREE_HSA_PACKET_TYPE_VENDOR_SPECIFIC,
                                         packet_control);
}

static iree_status_t iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t gpu_agent,
    hsa_amd_memory_pool_t pm4_ib_pool, uint32_t aql_queue_capacity,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  iree_host_size_t pm4_ib_size = 0;
  if (!iree_host_size_checked_mul(aql_queue_capacity,
                                  sizeof(iree_hal_amdgpu_pm4_ib_slot_t),
                                  &pm4_ib_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 IB slot buffer size overflow");
  }
  if (IREE_UNLIKELY(!pm4_ib_pool.handle)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 IB memory pool is required");
  }
  iree_hal_amdgpu_pm4_ib_slot_t* pm4_ib_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(libhsa), pm4_ib_pool, pm4_ib_size,
      HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, (void**)&pm4_ib_slots));
  iree_status_t status = iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(libhsa), /*num_agents=*/1, &gpu_agent, /*flags=*/NULL,
      pm4_ib_slots);
  if (iree_status_is_ok(status)) {
    memset(pm4_ib_slots, 0, pm4_ib_size);
    out_queue->pm4_ib_slots = pm4_ib_slots;
  } else {
    IREE_IGNORE_ERROR(
        iree_hsa_amd_memory_pool_free(IREE_LIBHSA(libhsa), pm4_ib_slots));
  }
  return status;
}

// Writes one device-side wait barrier packet body and returns the header/setup
// bits that will publish it. The barrier halts the CP until the peer queue's
// epoch signal reaches the target epoch.
//
// Caller must commit the packet header after this returns. Caller must hold
// submission_mutex.
static uint16_t iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_barrier_t* barrier, uint64_t packet_id,
    hsa_signal_t completion_signal, iree_hal_amdgpu_aql_packet_t* packet,
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
          iree_hal_amdgpu_aql_packet_control_barrier_system(),
          completion_signal, out_setup);
    case IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64: {
      IREE_ASSERT(queue->pm4_ib_slots != NULL);
      iree_hal_amdgpu_pm4_ib_slot_t* ib_slot =
          &queue->pm4_ib_slots[packet_id & queue->aql_ring.mask];
      uint32_t ib_dword_count =
          iree_hal_amdgpu_host_queue_emit_pm4_wait_reg_mem64(
              ib_slot, barrier->epoch_signal, compare_value,
              (iree_hsa_signal_value_t)INT64_MAX);
      return iree_hal_amdgpu_host_queue_write_pm4_ib_packet_body(
          &packet->pm4_ib, ib_slot, ib_dword_count,
          iree_hal_amdgpu_aql_packet_control_barrier_system(),
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

// Emits device-side wait barrier packets for resolved waits. Each barrier
// halts the CP until the peer queue's epoch signal reaches the target epoch.
//
// |first_packet_id| is the first AQL slot reserved for barriers. The caller
// must have reserved resolution->barrier_count consecutive slots starting here.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_emit_barriers(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    uint64_t first_packet_id) {
  for (uint8_t i = 0; i < resolution->barrier_count; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, first_packet_id + i);
    uint16_t setup = 0;
    uint16_t header = iree_hal_amdgpu_host_queue_write_wait_barrier_packet_body(
        queue, &resolution->barriers[i], first_packet_id + i,
        iree_hsa_signal_null(), packet, &setup);
    iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  }
}

// Merges the waited axes from resolved barriers into the queue's accumulated
// frontier. This records that the queue has barrier'd on these axes, allowing
// future submissions to skip redundant barriers for producer epochs already
// known to precede this queue's current AQL frontier.
//
// The barrier entries are maintained in ascending axis order by
// append_wait_barrier(), satisfying the frontier's sorted invariant.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_merge_barrier_axes(
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

// Returns the number of retained resources required for a submission with
// |signal_semaphore_count| user-visible signal semaphores and
// |operation_resource_count| additional operation-owned resources.
static iree_status_t iree_hal_amdgpu_host_queue_count_reclaim_resources(
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

// Returns the queue-order frontier to use for pool acquire_reservation() after
// accounting for dependency barriers in |resolution|.
//
// If no barriers were resolved, returns queue->frontier directly. Otherwise
// copies queue->frontier into |storage| and merges the barrier axes so pool
// dominance checks see the post-wait queue position. If that temporary merge
// overflows, returns the current queue frontier as a conservative lower bound.
static const iree_async_frontier_t*
iree_hal_amdgpu_host_queue_pool_requester_frontier(
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

// Imports a pool-owned death frontier into the queue's AQL dependency list.
// Entries already dominated by |requester_frontier| are skipped; remaining
// local-queue axes become device-side wait barriers in |resolution|.
static bool iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
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
            queue, resolution, entry->axis, peer_signal, entry->epoch)) {
      return false;
    }
  }
  return true;
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

// One in-flight kernel-dispatch submission assembled under submission_mutex.
// All queue-private resource transfer and reclaim bookkeeping flows through
// this struct so fill/copy/update/dispatch do not accidentally grow divergent
// kernel-ring retirement mechanisms.
typedef struct iree_hal_amdgpu_host_queue_dispatch_submission_t {
  iree_hal_amdgpu_reclaim_entry_t* reclaim_entry;
  iree_hal_resource_t** reclaim_resources;
  iree_hal_amdgpu_aql_packet_t* dispatch_slot;
  iree_hal_amdgpu_kernarg_block_t* kernarg_blocks;
  uint64_t first_packet_id;
  uint64_t kernarg_write_position;
  uint32_t packet_count;
  uint16_t reclaim_resource_count;
  uint16_t dispatch_setup;
} iree_hal_amdgpu_host_queue_dispatch_submission_t;

// Writes one final dispatch packet body into an AQL slot in forward field
// order and returns the setup bits that must be published with the header.
// The first dword (header + setup) remains untouched so the CP continues to
// observe an INVALID packet until aql_ring_commit() performs the release store.
//
// Queue-owned kernarg/completion-signal addresses are substituted as part of
// the single write pass so each ring slot field is written exactly once.
static uint16_t iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
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

// Begins one kernel-dispatch submission by reserving notification/reclaim
// state, AQL slots, and |kernarg_block_count| queue-owned kernarg blocks.
//
// The submission reserves one final dispatch packet plus one prefix no-op
// packet for each extra kernarg block beyond the first. That preserves the
// kernarg-ring backpressure invariant for variable-sized updates: each reserved
// AQL packet accounts for at most one queued kernarg block.
//
// Prefix dependency barriers are emitted before any no-op padding packets, and
// |out_submission->dispatch_slot| points at the final uncommitted dispatch
// slot. If kernarg allocation fails after AQL reservation, all reserved slots
// are plugged with no-op packets and the empty reclaim entry is released.
//
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_begin_dispatch_submission(
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

// Finishes a submission by transferring retained resources to the reclaim
// entry, publishing the queue/semaphore frontier state, committing the final
// dispatch header, and ringing the doorbell.
//
// |dispatch_slot->dispatch| must already be populated except for the first
// dword (header + setup), |submission->dispatch_setup| must contain the setup
// bits to publish, and the queue-owned kernarg bytes must already be written
// into |submission->kernarg_blocks|.
//
// Caller must hold submission_mutex.
static void iree_hal_amdgpu_host_queue_finish_dispatch_submission(
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

//===----------------------------------------------------------------------===//
// Pending operations (deferred submission)
//===----------------------------------------------------------------------===//
//
// LOCKING PROTOCOL
//
// submission_mutex protects all submission-path state: AQL ring reservation,
// kernarg allocation, packet emission, commit_signals, frontier mutation,
// notification ring push, and the pending list (link/unlink).
//
// The proactor thread (drain, error check) does NOT acquire submission_mutex.
// It reads the notification ring (SPSC consumer) and the atomic error_status.
//
// Deferred operations use a two-phase protocol:
//
//   Phase 1 (under submission_mutex): resolve_waits, allocate pending_op,
//     capture operation parameters, link to pending list.
//
//   Phase 2 (WITHOUT submission_mutex): register timepoints via enqueue_waits.
//     Timepoint callbacks may fire synchronously during acquire_timepoint
//     (when the semaphore value is already reached or the semaphore is already
//     failed). The last callback to fire calls pending_op_issue or
//     pending_op_fail, both of which acquire submission_mutex internally.
//     This is safe because Phase 1 released the mutex before Phase 2 began.
//
// pending_op_issue: acquires submission_mutex to emit AQL packets, transfer
//   retained resources to the reclaim ring, commit signals, and unlink.
//
// pending_op_fail: acquires submission_mutex to unlink. Semaphore failure
//   and resource release happen outside the lock.
//
// pending_op_destroy_under_lock: for capture-time failures (arena allocation
//   errors after pending_op_allocate). Caller already holds submission_mutex.
//   Does NOT re-acquire — unlinks and cleans up directly.

// Operation types corresponding to virtual queue vtable entries. Each type
// has a per-operation capture struct in the pending_op_t union.
typedef enum iree_hal_amdgpu_pending_op_type_e {
  IREE_HAL_AMDGPU_PENDING_OP_FILL,
  IREE_HAL_AMDGPU_PENDING_OP_COPY,
  IREE_HAL_AMDGPU_PENDING_OP_UPDATE,
  IREE_HAL_AMDGPU_PENDING_OP_DISPATCH,
  IREE_HAL_AMDGPU_PENDING_OP_EXECUTE,
  IREE_HAL_AMDGPU_PENDING_OP_ALLOCA,
  IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA,
  IREE_HAL_AMDGPU_PENDING_OP_HOST_CALL,
} iree_hal_amdgpu_pending_op_type_t;

// Completion ownership for a deferred operation.
typedef enum iree_hal_amdgpu_pending_op_lifecycle_e {
  // Waiting callbacks may still resolve the op.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING = 0,
  // Queue shutdown claimed cancellation ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING = 1,
  // The last wait callback claimed completion ownership.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING = 2,
  // The issuing thread is registering a cold alloca memory-readiness wait.
  // Cancellation only claims PENDING ops; the arming thread publishes PENDING
  // after registration or observes a synchronous callback as COMPLETING.
  IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT = 3,
} iree_hal_amdgpu_pending_op_lifecycle_t;

// Per-wait timepoint entry, arena-allocated (one per unsatisfied wait).
// The timepoint callback decrements the operation's atomic wait counter;
// the last callback to fire issues (or fails) the operation.
typedef struct iree_hal_amdgpu_wait_entry_t {
  iree_async_semaphore_timepoint_t timepoint;
  iree_hal_amdgpu_pending_op_t* operation;
  // Set to 1 after the callback's final access to this entry/op completes.
  // Queue shutdown spins on this for callbacks that were already detached from
  // the semaphore before cancel_timepoint() ran.
  iree_atomic_int32_t callback_complete;
} iree_hal_amdgpu_wait_entry_t;

typedef enum iree_hal_amdgpu_alloca_memory_wait_kind_e {
  // No active memory wait or held reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE = 0,
  // Waiting for a copied pool death frontier while holding a reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER = 1,
  // Waiting for a pool release notification before retrying reservation.
  IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION = 2,
} iree_hal_amdgpu_alloca_memory_wait_kind_t;

// Cold-path alloca memory-readiness wait. Allocated inside a pending op's
// arena only after user semaphore waits have resolved and the pool cannot
// produce immediately-usable bytes.
typedef struct iree_hal_amdgpu_alloca_memory_wait_t {
  // Active wait source.
  iree_hal_amdgpu_alloca_memory_wait_kind_t kind;

  // Set to 1 after the callback's final access to this wait/op completes.
  iree_atomic_int32_t callback_complete;

  // State for a held reservation blocked on a pool death frontier.
  struct {
    // Queue-owned reservation held while waiting for its death frontier.
    iree_hal_pool_reservation_t reservation;

    // Arena-owned copy of the pool-owned death frontier.
    iree_async_frontier_t* wait_frontier;

    // Tracker waiter storage for |wait_frontier|.
    iree_async_frontier_waiter_t waiter;
  } frontier;

  // State for reservation retry after pool release notifications.
  struct {
    // Borrowed notification returned by the pool.
    iree_async_notification_t* notification;

    // Notification epoch observed before the reservation retry.
    uint32_t wait_token;

    // Wait operations rotated so a callback can arm a retry before returning.
    iree_async_notification_wait_operation_t wait_ops[2];

    // Index of the active wait operation in |wait_ops|.
    uint8_t wait_slot;
  } pool_notification;
} iree_hal_amdgpu_alloca_memory_wait_t;

// A deferred queue operation waiting for its waits to become satisfiable.
// Arena-allocated from the queue's block pool. All variable-size captured
// data (semaphore lists, constants, bindings, update source data) lives in
// the arena alongside this struct.
struct iree_hal_amdgpu_pending_op_t {
  // Arena backing this operation and all captured data. Deinitialized on
  // completion, failure, or cancellation — returns blocks to the pool.
  iree_arena_allocator_t arena;

  // Owning queue. Used to acquire submission_mutex and emit AQL packets
  // when all waits are satisfied.
  iree_hal_amdgpu_host_queue_t* queue;

  // Intrusive linked list for the queue's pending list (cleanup/shutdown).
  iree_hal_amdgpu_pending_op_t* next;
  iree_hal_amdgpu_pending_op_t** prev_next;

  // Number of outstanding wait timepoints. Atomically decremented by each
  // wait_entry callback. When this reaches zero, the operation is ready.
  iree_atomic_int32_t wait_count;

  // Completion/cancellation owner. Exactly one path transitions this away from
  // PENDING and then destroys the operation.
  iree_atomic_int32_t lifecycle_state;

  // First error from a failed wait. CAS from 0; the winner owns the status.
  iree_atomic_intptr_t error_status;

  // Wakes cancellation when a detached wait callback finishes touching the op.
  iree_notification_t callback_notification;

  // Wait semaphore list (arena-allocated clone, semaphores retained by the
  // clone). Released when the op is destroyed.
  iree_hal_semaphore_list_t wait_semaphore_list;

  // Signal semaphore list. The semaphores[] pointer points into the first
  // entries of retained_resources (the semaphore pointers are shared, not
  // separately retained). The payload_values[] is a separate arena-allocated
  // array. Used for commit_signals and semaphore_list_fail.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Wait entries registered with the wait semaphores, one per
  // wait_semaphore_list entry. Arena-allocated as one contiguous array so queue
  // shutdown can cancel and await each callback before freeing the arena.
  iree_hal_amdgpu_wait_entry_t* wait_entries;

  // Flat array of all retained HAL resources (arena-allocated). Signal
  // semaphores are stored first (signal_semaphore_list.semaphores aliases
  // this region), followed by operation-specific resources (buffers,
  // executables, command buffers). On successful issue, ownership transfers
  // to the reclaim ring. On failure/cancel, released directly.
  iree_hal_resource_t** retained_resources;

  // Number of entries currently owned in |retained_resources|.
  uint16_t retained_resource_count;

  // Operation payload selector.
  iree_hal_amdgpu_pending_op_type_t type;
  union {
    struct {
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      uint64_t pattern_bits;
      iree_host_size_t pattern_length;
      iree_hal_fill_flags_t flags;
    } fill;
    struct {
      iree_hal_buffer_t* source_buffer;
      iree_device_size_t source_offset;
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      iree_hal_copy_flags_t flags;
    } copy;
    struct {
      // Source data is copied into the arena (not a borrowed pointer).
      const void* source_data;
      iree_hal_buffer_t* target_buffer;
      iree_device_size_t target_offset;
      iree_device_size_t length;
      iree_hal_update_flags_t flags;
    } update;
    struct {
      iree_hal_executable_t* executable;
      iree_hal_executable_export_ordinal_t export_ordinal;
      iree_hal_dispatch_config_t config;
      iree_const_byte_span_t constants;     // Arena-allocated copy.
      iree_hal_buffer_ref_list_t bindings;  // Arena-allocated copy.
      iree_hal_dispatch_flags_t flags;
    } dispatch;
    struct {
      iree_hal_command_buffer_t* command_buffer;
      iree_hal_buffer_binding_table_t binding_table;  // Arena-allocated copy.
      iree_hal_execute_flags_t flags;
    } execute;
    struct {
      // Borrowed pool resolved during queue_alloca capture. The pool owner
      // must outlive all queued transient allocations.
      iree_hal_pool_t* pool;

      // Buffer parameters captured from queue_alloca.
      iree_hal_buffer_params_t params;

      // Requested allocation size in bytes.
      iree_device_size_t allocation_size;

      // Pool reservation flags derived from queue_alloca flags.
      iree_hal_pool_reserve_flags_t reserve_flags;

      // Transient buffer returned to the caller and committed on success.
      iree_hal_buffer_t* buffer;

      // Cold memory-readiness sidecar allocated only after user waits resolve.
      iree_hal_amdgpu_alloca_memory_wait_t* memory_wait;
    } alloca_op;
    struct {
      iree_hal_buffer_t* buffer;
    } dealloca;
    struct {
      iree_hal_host_call_t call;
      uint64_t args[4];
      iree_hal_host_call_flags_t flags;
    } host_call;
  };
};

static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op);
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status);
static iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op);
static iree_status_t iree_hal_amdgpu_host_queue_submit_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);
static iree_status_t iree_hal_amdgpu_host_queue_submit_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);
static iree_status_t iree_hal_amdgpu_host_queue_submit_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);
static iree_status_t iree_hal_amdgpu_host_queue_submit_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);
static iree_status_t iree_hal_amdgpu_host_queue_submit_barrier(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdgpu_reclaim_action_t pre_signal_action,
    iree_hal_resource_t* const* operation_resources,
    iree_host_size_t operation_resource_count,
    iree_hal_amdgpu_host_queue_post_commit_fn_t post_commit_fn,
    void* post_commit_user_data,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags);

// Links a pending op into the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_link(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  op->next = queue->pending_head;
  op->prev_next = &queue->pending_head;
  if (queue->pending_head) {
    queue->pending_head->prev_next = &op->next;
  }
  queue->pending_head = op;
}

// Unlinks a pending op from the queue's pending list. Caller must hold
// submission_mutex.
static void iree_hal_amdgpu_pending_op_unlink(
    iree_hal_amdgpu_pending_op_t* op) {
  *op->prev_next = op->next;
  if (op->next) {
    op->next->prev_next = op->prev_next;
  }
  op->next = NULL;
  op->prev_next = NULL;
}

// Retains a resource and appends it to the pending op's retained_resources
// array. The caller must have allocated sufficient capacity in the array
// via the max_resource_count parameter to pending_op_allocate.
static inline void iree_hal_amdgpu_pending_op_retain(
    iree_hal_amdgpu_pending_op_t* op, iree_hal_resource_t* resource) {
  if (IREE_LIKELY(resource)) {
    iree_hal_resource_retain(resource);
    op->retained_resources[op->retained_resource_count++] = resource;
  }
}

// Releases all retained HAL resources in the flat array. Used on failure
// and cancellation paths. On success, retained_resources are transferred
// to the reclaim ring instead (no release here).
static void iree_hal_amdgpu_pending_op_release_retained(
    iree_hal_amdgpu_pending_op_t* op) {
  for (uint16_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  op->retained_resource_count = 0;
}

// Releases any queue-owned alloca memory-readiness reservation. This runs only
// on failure/cancellation paths or after ownership has not transferred into the
// transient buffer.
static void iree_hal_amdgpu_pending_op_release_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_pool_release_reservation(op->alloca_op.pool,
                                        &wait->frontier.reservation,
                                        wait->frontier.wait_frontier);
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static bool iree_hal_amdgpu_alloca_memory_wait_callback_is_complete(
    void* user_data) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait =
      (iree_hal_amdgpu_alloca_memory_wait_t*)user_data;
  return iree_atomic_load(&wait->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&op->callback_notification, IREE_ALL_WAITERS);
}

// Cancels any active alloca memory-readiness wait before destroying the op.
static void iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  if (op->type != IREE_HAL_AMDGPU_PENDING_OP_ALLOCA) return;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) return;

  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER: {
      const bool cancelled = iree_async_frontier_tracker_cancel_wait(
          op->queue->frontier_tracker, &wait->frontier.waiter);
      if (!cancelled) {
        iree_notification_await(
            &op->callback_notification,
            iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, wait,
            iree_infinite_timeout());
      }
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION: {
      // Shutdown is allowed to prod the pool notification: it is a broad wake,
      // but prevents teardown from depending on a future dealloca. The callback
      // observes the CANCELLING lifecycle state and only publishes completion.
      iree_async_notification_signal(wait->pool_notification.notification,
                                     INT32_MAX);
      iree_notification_await(
          &op->callback_notification,
          iree_hal_amdgpu_alloca_memory_wait_callback_is_complete, wait,
          iree_infinite_timeout());
      break;
    }
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      break;
  }
}

static bool iree_hal_amdgpu_wait_entry_callback_is_complete(void* user_data) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  return iree_atomic_load(&entry->callback_complete,
                          iree_memory_order_acquire) != 0;
}

static void iree_hal_amdgpu_wait_entry_publish_callback_complete(
    iree_hal_amdgpu_wait_entry_t* entry) {
  iree_atomic_store(&entry->callback_complete, 1, iree_memory_order_release);
  iree_notification_post(&entry->operation->callback_notification,
                         IREE_ALL_WAITERS);
}

// Records the first asynchronous wait failure. Takes ownership of |status|,
// storing it for the completion owner or dropping it if another failure won.
static void iree_hal_amdgpu_pending_op_record_error_status(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &op->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_ignore(status);
  }
}

// Destroys a pending operation that failed during capture (arena allocation
// error after pending_op_allocate but before enqueue_waits). Caller MUST hold
// submission_mutex — the op is linked to the pending list by allocate and
// needs the mutex for unlinking.
//
// Unlike pending_op_fail (which acquires submission_mutex internally), this
// function assumes the caller already holds it. This is necessary because the
// capture phase runs under the mutex (Phase 1 of the two-phase protocol).
static void iree_hal_amdgpu_pending_op_destroy_under_lock(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  // Fail signal semaphores so downstream waiters get the error.
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  // Unlink from the pending list (caller holds submission_mutex).
  iree_hal_amdgpu_pending_op_unlink(op);
  // Tear down callback wake state before returning arena blocks to the pool.
  iree_notification_deinitialize(&op->callback_notification);
  // Return arena blocks to the pool.
  iree_arena_deinitialize(&op->arena);
}

// Timepoint callback fired when a wait semaphore reaches its target value
// (or fails). Each pending operation has one wait_entry per unsatisfied wait;
// each entry's callback atomically decrements wait_count. The last callback
// to fire (wait_count reaches zero) issues the operation or fails it.
static void iree_hal_amdgpu_wait_entry_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_hal_amdgpu_wait_entry_t* entry =
      (iree_hal_amdgpu_wait_entry_t*)user_data;
  iree_hal_amdgpu_pending_op_t* op = entry->operation;

  iree_hal_amdgpu_pending_op_record_error_status(op, status);

  // Decrement wait count. The last callback to fire triggers issue or fail.
  int32_t previous_count =
      iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel);
  if (previous_count == 1) {
    int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      // Publish callback completion before destroying the op arena.
      iree_hal_amdgpu_wait_entry_publish_callback_complete(entry);
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_hal_amdgpu_pending_op_fail(op, error);
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
      return;
    }
  }

  iree_hal_amdgpu_wait_entry_publish_callback_complete(entry);
}

// Registers timepoints for all waits in the operation's wait semaphore list.
// Sets wait_count and registers one timepoint per wait — callbacks may fire
// synchronously during registration. When all waits are satisfied (the last
// callback fires), the operation is issued or failed.
//
// The wait_semaphore_list on the op (cloned into the arena by allocate)
// retains all semaphores for the lifetime of the op. Wait entries do not
// independently retain semaphores.
static iree_status_t iree_hal_amdgpu_pending_op_enqueue_waits(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_semaphore_list_t wait_semaphores = op->wait_semaphore_list;
  if (wait_semaphores.count == 0) {
    iree_hal_amdgpu_pending_op_issue(op);
    return iree_ok_status();
  }

  iree_host_size_t wait_entry_bytes = 0;
  if (!iree_host_size_checked_mul(wait_semaphores.count,
                                  sizeof(*op->wait_entries),
                                  &wait_entry_bytes)) {
    iree_hal_amdgpu_pending_op_fail(
        op, iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "pending op wait entry allocation overflow"));
    return iree_ok_status();
  }
  iree_status_t status = iree_arena_allocate(&op->arena, wait_entry_bytes,
                                             (void**)&op->wait_entries);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return iree_ok_status();
  }
  memset(op->wait_entries, 0, wait_entry_bytes);

  // Set wait_count before registering any timepoints. A timepoint callback
  // may fire synchronously during acquire_timepoint.
  iree_atomic_store(&op->wait_count, (int32_t)wait_semaphores.count,
                    iree_memory_order_release);

  for (iree_host_size_t i = 0; i < wait_semaphores.count; ++i) {
    iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
    entry->operation = op;
    iree_atomic_store(&entry->callback_complete, 0, iree_memory_order_relaxed);
    entry->timepoint.callback = iree_hal_amdgpu_wait_entry_resolved;
    entry->timepoint.user_data = entry;
    status = iree_async_semaphore_acquire_timepoint(
        (iree_async_semaphore_t*)wait_semaphores.semaphores[i],
        wait_semaphores.payload_values[i], &entry->timepoint);

    if (!iree_status_is_ok(status)) {
      // Registration failed at index i. Timepoints 0..i-1 are already
      // registered and their callbacks will fire asynchronously — we cannot
      // destroy the op here. Record the error and subtract the unregistered
      // count so the existing callbacks drain and destroy the op.
      iree_hal_amdgpu_pending_op_record_error_status(op, status);
      int32_t unregistered = (int32_t)(wait_semaphores.count - i);
      int32_t previous_count = iree_atomic_fetch_sub(
          &op->wait_count, unregistered, iree_memory_order_acq_rel);
      if (previous_count == unregistered) {
        iree_status_t error = (iree_status_t)iree_atomic_exchange(
            &op->error_status, 0, iree_memory_order_acquire);
        iree_hal_amdgpu_pending_op_fail(op, error);
        return iree_ok_status();
      }
      return iree_ok_status();
    }
  }

  return iree_ok_status();
}

static void iree_hal_amdgpu_alloca_memory_wait_resolved(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (iree_status_is_ok(status) &&
      wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION) {
    wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  }

  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_record_error_status(op, status);
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    return;
  }

  expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdgpu_pending_op_fail(op, status);
    } else {
      iree_hal_amdgpu_pending_op_issue(op);
    }
    return;
  }

  iree_hal_amdgpu_pending_op_record_error_status(op, status);
  iree_hal_amdgpu_alloca_memory_wait_publish_callback_complete(op);
}

static void iree_hal_amdgpu_alloca_frontier_wait_resolved(
    void* user_data, iree_status_t status) {
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_alloca_pool_notification_wait_resolved(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_amdgpu_alloca_memory_wait_resolved(
      (iree_hal_amdgpu_pending_op_t*)user_data, status);
}

static void iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(
    iree_hal_amdgpu_pending_op_t* op, iree_status_t status) {
  int32_t expected_state =
      IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT;
  if (iree_status_is_ok(status)) {
    if (iree_atomic_compare_exchange_strong(
            &op->lifecycle_state, &expected_state,
            IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      return;
    }
    if (expected_state == IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING) {
      iree_status_t error = (iree_status_t)iree_atomic_exchange(
          &op->error_status, 0, iree_memory_order_acquire);
      if (!iree_status_is_ok(error)) {
        iree_hal_amdgpu_pending_op_fail(op, error);
      } else {
        iree_hal_amdgpu_pending_op_issue(op);
      }
    }
    return;
  }

  if (iree_atomic_compare_exchange_strong(
          &op->lifecycle_state, &expected_state,
          IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_COMPLETING,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    iree_hal_amdgpu_pending_op_fail(op, status);
    return;
  }
  iree_hal_amdgpu_pending_op_record_error_status(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status = iree_async_frontier_tracker_wait(
      queue->frontier_tracker, wait->frontier.wait_frontier,
      iree_hal_amdgpu_alloca_frontier_wait_resolved, op,
      &wait->frontier.waiter);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  iree_async_notification_wait_operation_t* wait_op =
      &wait->pool_notification.wait_ops[wait->pool_notification.wait_slot];
  iree_async_operation_zero(&wait_op->base, sizeof(*wait_op));
  iree_async_operation_initialize(
      &wait_op->base, IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_hal_amdgpu_alloca_pool_notification_wait_resolved, op);
  wait_op->notification = wait->pool_notification.notification;
  wait_op->wait_flags = IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN;
  wait_op->wait_token = wait->pool_notification.wait_token;

  iree_atomic_store(&wait->callback_complete, 0, iree_memory_order_relaxed);
  iree_status_t status =
      iree_async_proactor_submit_one(op->queue->proactor, &wait_op->base);
  iree_hal_amdgpu_pending_op_finish_alloca_memory_wait_enqueue(op, status);
}

static void iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  switch (wait->kind) {
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER:
      iree_hal_amdgpu_pending_op_enqueue_alloca_frontier_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION:
      iree_hal_amdgpu_pending_op_enqueue_alloca_pool_notification_wait(op);
      break;
    case IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE:
      iree_hal_amdgpu_pending_op_fail(
          op, iree_make_status(IREE_STATUS_INTERNAL,
                               "pending alloca has no memory wait to enqueue"));
      break;
  }
}

// Allocates and initializes a pending operation from a fresh arena.
// Clones the wait semaphore list. Allocates the retained_resources array
// with |max_resource_count| capacity and populates the first entries with
// signal semaphores (retained). The signal_semaphore_list.semaphores pointer
// aliases into retained_resources so that commit_signals and
// semaphore_list_fail can use it directly.
//
// The caller must push operation-specific resources into retained_resources
// (via the returned op) before calling enqueue_waits.
//
// On failure, the arena is cleaned up and *out_op is set to NULL.
static iree_status_t iree_hal_amdgpu_pending_op_allocate(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_amdgpu_pending_op_type_t type, uint16_t max_resource_count,
    iree_hal_amdgpu_pending_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;

  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_hal_amdgpu_pending_op_t* op = NULL;
  iree_status_t status = iree_arena_allocate(&arena, sizeof(*op), (void**)&op);
  if (!iree_status_is_ok(status)) {
    iree_arena_deinitialize(&arena);
    return status;
  }

  memset(op, 0, sizeof(*op));
  memcpy(&op->arena, &arena, sizeof(arena));
  op->queue = queue;
  op->type = type;
  iree_atomic_store(&op->wait_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING,
                    iree_memory_order_relaxed);
  iree_atomic_store(&op->error_status, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&op->callback_notification);

  iree_allocator_t arena_allocator = iree_arena_allocator(&op->arena);

  // Clone the wait semaphore list (retains each wait semaphore).
  status = iree_hal_semaphore_list_clone(wait_semaphore_list, arena_allocator,
                                         &op->wait_semaphore_list);

  // Allocate the retained resources array and the signal payload values.
  if (iree_status_is_ok(status) && max_resource_count > 0) {
    status = iree_arena_allocate(
        &op->arena, max_resource_count * sizeof(iree_hal_resource_t*),
        (void**)&op->retained_resources);
  }
  uint64_t* signal_payload_values = NULL;
  if (iree_status_is_ok(status) && signal_semaphore_list->count > 0) {
    status = iree_arena_allocate(
        &op->arena, signal_semaphore_list->count * sizeof(uint64_t),
        (void**)&signal_payload_values);
  }

  if (iree_status_is_ok(status)) {
    // Signal semaphores occupy the first entries of retained_resources.
    // The signal_semaphore_list.semaphores pointer aliases this region.
    for (iree_host_size_t i = 0; i < signal_semaphore_list->count; ++i) {
      op->retained_resources[i] =
          (iree_hal_resource_t*)signal_semaphore_list->semaphores[i];
      iree_hal_resource_retain(op->retained_resources[i]);
      signal_payload_values[i] = signal_semaphore_list->payload_values[i];
    }
    op->retained_resource_count = (uint16_t)signal_semaphore_list->count;
    op->signal_semaphore_list.count = signal_semaphore_list->count;
    op->signal_semaphore_list.semaphores =
        (iree_hal_semaphore_t**)op->retained_resources;
    op->signal_semaphore_list.payload_values = signal_payload_values;

    iree_hal_amdgpu_pending_op_link(op);
    *out_op = op;
  } else {
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
  return status;
}

// Issues a deferred operation after all waits are satisfied. All waits are
// tier 0 (timeline_value >= waited_value) — the GPU work producing those
// values has completed. No barriers are needed.
//
// Called from the last wait_entry callback (any thread). Acquires
// submission_mutex to emit AQL packets and commit signals.
static void iree_hal_amdgpu_pending_op_issue(iree_hal_amdgpu_pending_op_t* op) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  iree_slim_mutex_lock(&queue->submission_mutex);

  iree_status_t status = iree_ok_status();
  if (queue->is_shutting_down) {
    status = iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  } else {
    // All waits are tier 0 — emit operation packets with no dependency
    // barriers.
    iree_hal_amdgpu_wait_resolution_t resolution;
    resolution.barrier_count = 0;
    resolution.needs_deferral = false;
    memset(resolution.reserved, 0, sizeof(resolution.reserved));
    switch (op->type) {
      case IREE_HAL_AMDGPU_PENDING_OP_FILL:
        status = iree_hal_amdgpu_host_queue_submit_fill(
            queue, &resolution, op->signal_semaphore_list,
            op->fill.target_buffer, op->fill.target_offset, op->fill.length,
            op->fill.pattern_bits, op->fill.pattern_length, op->fill.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_COPY:
        status = iree_hal_amdgpu_host_queue_submit_copy(
            queue, &resolution, op->signal_semaphore_list,
            op->copy.source_buffer, op->copy.source_offset,
            op->copy.target_buffer, op->copy.target_offset, op->copy.length,
            op->copy.flags, IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_UPDATE:
        status = iree_hal_amdgpu_host_queue_submit_update(
            queue, &resolution, op->signal_semaphore_list,
            op->update.source_data, /*source_offset=*/0,
            op->update.target_buffer, op->update.target_offset,
            op->update.length, op->update.flags,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_EXECUTE:
        if (op->execute.command_buffer) {
          status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                    "pending command buffer execute not yet "
                                    "wired up");
          break;
        }
        status = iree_hal_amdgpu_host_queue_submit_barrier(
            queue, &resolution, op->signal_semaphore_list,
            (iree_hal_amdgpu_reclaim_action_t){0},
            /*operation_resources=*/NULL,
            /*operation_resource_count=*/0,
            /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      case IREE_HAL_AMDGPU_PENDING_OP_ALLOCA: {
        iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
        status = iree_hal_amdgpu_host_queue_submit_alloca(
            queue, &resolution, op->signal_semaphore_list, op->alloca_op.pool,
            op->alloca_op.params, op->alloca_op.allocation_size,
            op->alloca_op.reserve_flags, op->alloca_op.buffer,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE, op,
            &memory_wait_op);
        if (iree_status_is_ok(status) && memory_wait_op) {
          iree_slim_mutex_unlock(&queue->submission_mutex);
          iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
          return;
        }
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      }
      case IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA:
        status = iree_hal_amdgpu_host_queue_submit_dealloca(
            queue, &resolution, op->signal_semaphore_list, op->dealloca.buffer,
            IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_NONE);
        if (iree_status_is_ok(status)) {
          op->retained_resource_count = 0;
        }
        break;
      default:
        status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                  "pending op issue not yet wired up");
        break;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
  }

  // Clean up the pending op. Wait semaphore list is released (the clone
  // holds separate retains). Signal semaphore list is NOT released — the
  // semaphore pointers are in retained_resources (either transferred to
  // reclaim or released above).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);

  iree_slim_mutex_unlock(&queue->submission_mutex);
}

// Fails a deferred operation. Propagates the error to all signal semaphores
// so downstream waiters receive the failure instead of hanging. Takes
// ownership of |status|.
static void iree_hal_amdgpu_pending_op_fail(iree_hal_amdgpu_pending_op_t* op,
                                            iree_status_t status) {
  iree_hal_amdgpu_host_queue_t* queue = op->queue;
  // Fail signal semaphores (records error, does not release our retains).
  iree_hal_semaphore_list_fail(op->signal_semaphore_list, status);
  // Release any queue-owned memory reservation before releasing op resources.
  iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
  // Release all retained resources (signal semaphores + op resources).
  iree_hal_amdgpu_pending_op_release_retained(op);
  // Release wait semaphores (separately retained by the clone).
  iree_hal_semaphore_list_release(op->wait_semaphore_list);
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_pending_op_unlink(op);
  iree_slim_mutex_unlock(&queue->submission_mutex);
  iree_notification_deinitialize(&op->callback_notification);
  iree_arena_deinitialize(&op->arena);
}

// Cancels all pending operations on a queue with the given failure details.
// Creates a status only for operations that do not already carry a wait error.
// Called during deinitialize or on unrecoverable GPU fault.
// Caller must ensure no concurrent submissions (shutdown path).
static void iree_hal_amdgpu_host_queue_cancel_pending(
    iree_hal_amdgpu_host_queue_t* queue, iree_status_code_t status_code,
    const char* status_message) {
  iree_slim_mutex_lock(&queue->submission_mutex);
  queue->is_shutting_down = true;
  iree_slim_mutex_unlock(&queue->submission_mutex);

  for (;;) {
    iree_hal_amdgpu_pending_op_t* op = NULL;
    iree_slim_mutex_lock(&queue->submission_mutex);
    for (iree_hal_amdgpu_pending_op_t* candidate = queue->pending_head;
         candidate != NULL; candidate = candidate->next) {
      int32_t expected_state = IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_PENDING;
      if (iree_atomic_compare_exchange_strong(
              &candidate->lifecycle_state, &expected_state,
              IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_CANCELLING,
              iree_memory_order_acq_rel, iree_memory_order_acquire)) {
        iree_hal_amdgpu_pending_op_unlink(candidate);
        op = candidate;
        break;
      }
    }
    bool has_pending_ops = queue->pending_head != NULL;
    iree_slim_mutex_unlock(&queue->submission_mutex);

    if (op == NULL) {
      if (!has_pending_ops) break;
      iree_thread_yield();
      continue;
    }

    for (iree_host_size_t i = 0; i < op->wait_semaphore_list.count; ++i) {
      iree_hal_amdgpu_wait_entry_t* entry = &op->wait_entries[i];
      if (iree_async_semaphore_cancel_timepoint(entry->timepoint.semaphore,
                                                &entry->timepoint)) {
        continue;
      }
      iree_notification_await(&op->callback_notification,
                              iree_hal_amdgpu_wait_entry_callback_is_complete,
                              entry, iree_infinite_timeout());
    }
    iree_hal_amdgpu_pending_op_cancel_alloca_memory_wait(op);

    iree_status_t op_status = (iree_status_t)iree_atomic_exchange(
        &op->error_status, 0, iree_memory_order_acquire);
    if (iree_status_is_ok(op_status)) {
      op_status = iree_make_status(status_code, "%s", status_message);
    }
    iree_hal_semaphore_list_fail(op->signal_semaphore_list, op_status);
    iree_hal_amdgpu_pending_op_release_alloca_memory_wait(op);
    iree_hal_amdgpu_pending_op_release_retained(op);
    iree_hal_semaphore_list_release(op->wait_semaphore_list);
    iree_notification_deinitialize(&op->callback_notification);
    iree_arena_deinitialize(&op->arena);
  }
}

//===----------------------------------------------------------------------===//
// Initialization / deinitialization
//===----------------------------------------------------------------------===//

// Proactor progress callback. Drains completed notification entries and
// reclaims kernarg space on each proactor iteration. If the GPU queue has
// faulted (error_status is set), fails all pending entries instead of
// draining normally.
static iree_host_size_t iree_hal_amdgpu_host_queue_progress_fn(
    void* user_data) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)user_data;

  // Check for GPU queue error (set by the HSA error callback on another
  // thread). If the queue has faulted, no further epochs will advance —
  // fail all pending entries so waiters get the actual GPU error instead
  // of hanging or timing out.
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  const uint64_t previous_epoch = (uint64_t)iree_atomic_load(
      &queue->notification_ring.epoch.last_drained, iree_memory_order_relaxed);
  uint64_t kernarg_reclaim_position = 0;
  iree_host_size_t count = 0;
  if (IREE_UNLIKELY(error)) {
    count = iree_hal_amdgpu_notification_ring_fail_all(
        &queue->notification_ring, error, &kernarg_reclaim_position);
    iree_async_frontier_tracker_fail_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(iree_status_code(error)));
  } else {
    count = iree_hal_amdgpu_notification_ring_drain(&queue->notification_ring,
                                                    /*fallback_frontier=*/NULL,
                                                    &kernarg_reclaim_position);
    const uint64_t current_epoch =
        (uint64_t)iree_atomic_load(&queue->notification_ring.epoch.last_drained,
                                   iree_memory_order_acquire);
    if (current_epoch > previous_epoch) {
      iree_async_frontier_tracker_advance(queue->frontier_tracker, queue->axis,
                                          current_epoch);
    }
  }
  if (kernarg_reclaim_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(&queue->kernarg_ring,
                                         kernarg_reclaim_position);
  }
  return count;
}

// HSA queue error callback. Called by the HSA runtime (on an internal thread)
// when the queue encounters an unrecoverable error (page fault, invalid AQL
// packet, ECC error). Stores the error atomically on the queue so the proactor
// progress callback can fail pending semaphores with the actual GPU error.
static void iree_hal_amdgpu_host_queue_error_callback(hsa_status_t status,
                                                      hsa_queue_t* source,
                                                      void* data) {
  iree_hal_amdgpu_host_queue_t* queue = (iree_hal_amdgpu_host_queue_t*)data;

  // Convert the HSA error to an IREE status with diagnostic information.
  iree_status_t error = iree_status_from_hsa_status(
      __FILE__, __LINE__, status, "hsa_queue_error_callback",
      "GPU queue encountered an unrecoverable error");

  // First-error-wins: store the error with release semantics so the status
  // payload (heap-allocated string, backtrace) is visible to any thread that
  // loads with acquire. If another error already won the race, free ours.
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &queue->error_status, &expected, (intptr_t)error,
          iree_memory_order_release, iree_memory_order_acquire)) {
    iree_status_free(error);
    return;
  }

  // Wake the proactor so it picks up the error promptly. The proactor already
  // busy-polls when progress callbacks are registered, but it may be inside a
  // blocking wait at the exact moment of the fault.
  iree_async_proactor_wake(queue->proactor);
}

iree_status_t iree_hal_amdgpu_host_queue_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_hal_device_t* logical_device,
    iree_async_proactor_t* proactor, hsa_agent_t gpu_agent,
    hsa_amd_memory_pool_t kernarg_pool, hsa_amd_memory_pool_t pm4_ib_pool,
    iree_async_frontier_tracker_t* frontier_tracker, iree_async_axis_t axis,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_amdgpu_wait_barrier_strategy_t wait_barrier_strategy,
    iree_hal_amdgpu_epoch_signal_table_t* epoch_table,
    iree_arena_block_pool_t* block_pool,
    const iree_hal_amdgpu_device_buffer_transfer_context_t* transfer_context,
    iree_hal_pool_t* default_pool, iree_host_size_t device_ordinal,
    uint32_t aql_queue_capacity, uint32_t notification_capacity,
    uint32_t kernarg_capacity_in_blocks, iree_allocator_t host_allocator,
    iree_hal_amdgpu_host_queue_t* out_queue) {
  IREE_ASSERT_ARGUMENT(libhsa);
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_ASSERT_ARGUMENT(epoch_table);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(transfer_context);
  IREE_ASSERT_ARGUMENT(default_pool);
  IREE_ASSERT_ARGUMENT(out_queue);

  if (!iree_host_size_is_power_of_two(aql_queue_capacity) ||
      !iree_host_size_is_power_of_two(notification_capacity) ||
      !iree_host_size_is_power_of_two(kernarg_capacity_in_blocks)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "all capacities must be powers of two");
  }
  if (kernarg_capacity_in_blocks / 2u < aql_queue_capacity) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernarg ring capacity must be at least 2x the AQL ring capacity "
        "to cover one tail-padding gap at wrap (got kernarg_blocks=%u, "
        "aql_packets=%u)",
        kernarg_capacity_in_blocks, aql_queue_capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->base.vtable = &iree_hal_amdgpu_host_queue_vtable;
  out_queue->libhsa = libhsa;
  out_queue->logical_device = logical_device;
  out_queue->proactor = proactor;
  out_queue->frontier_tracker = frontier_tracker;
  out_queue->host_allocator = host_allocator;

  // Submission pipeline state.
  iree_slim_mutex_initialize(&out_queue->submission_mutex);
  out_queue->axis = axis;
  out_queue->wait_barrier_strategy = wait_barrier_strategy;
  out_queue->queue_affinity = queue_affinity;
  out_queue->last_signal.semaphore = NULL;
  out_queue->last_signal.epoch = 0;
  out_queue->block_pool = block_pool;
  out_queue->can_publish_frontier = true;
  out_queue->transfer_context = transfer_context;
  out_queue->default_pool = default_pool;
  out_queue->device_ordinal = device_ordinal;
  out_queue->pending_head = NULL;
  iree_async_frontier_initialize(iree_hal_amdgpu_host_queue_frontier(out_queue),
                                 /*entry_count=*/0);

  // The optional tracker semaphore is an iree_async_semaphore_t bridge for
  // CPU-side wait integration. The queue's GPU-visible HSA epoch signal is
  // created by the notification ring below and registered in the epoch table.
  iree_status_t status = iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL);

  // Create the HSA hardware AQL queue.
  //
  // HSA_QUEUE_TYPE_MULTI is required (not just an optimization). Once command
  // buffers start performing device-side enqueue, the CP itself becomes a
  // concurrent producer alongside the host submission path, so the queue must
  // permit multiple concurrent producers. The host-side reserve already uses
  // an atomic fetch_add on the write index, which is well-defined only on
  // MULTI queues.
  hsa_queue_t* hardware_queue = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hsa_queue_create(
        IREE_LIBHSA(libhsa), gpu_agent, aql_queue_capacity,
        HSA_QUEUE_TYPE_MULTI, iree_hal_amdgpu_host_queue_error_callback,
        /*data=*/out_queue,
        /*private_segment_size=*/UINT32_MAX,
        /*group_segment_size=*/UINT32_MAX, &hardware_queue);
  }

  // Initialize the AQL ring from the hardware queue.
  if (iree_status_is_ok(status)) {
    out_queue->hardware_queue = hardware_queue;
    iree_hal_amdgpu_aql_ring_initialize((iree_amd_queue_t*)hardware_queue,
                                        &out_queue->aql_ring);
  }

  // Initialize the kernarg ring from the HSA kernarg memory pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_kernarg_ring_initialize(
        libhsa, gpu_agent, kernarg_pool, kernarg_capacity_in_blocks,
        &out_queue->kernarg_ring);
  }

  // Initialize the optional PM4 IB slot buffer. The buffer is indexed by AQL
  // packet id and inherits AQL ring backpressure/reuse; there is no separate
  // PM4 producer or reclaim position.
  if (iree_status_is_ok(status) &&
      wait_barrier_strategy ==
          IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_PM4_WAIT_REG_MEM64) {
    status = iree_hal_amdgpu_host_queue_allocate_pm4_ib_slots(
        libhsa, gpu_agent, pm4_ib_pool, aql_queue_capacity, out_queue);
  }

  // Initialize the notification ring (creates epoch signal + entry buffer).
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_notification_ring_initialize(
        libhsa, block_pool, notification_capacity, host_allocator,
        &out_queue->notification_ring);
  }

  // Register this queue's epoch signal in the shared table for cross-queue
  // barrier emission lookups. Must happen after notification ring init (which
  // creates the epoch signal) and before any submissions.
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_epoch_signal_table_register(
        epoch_table, iree_async_axis_device_index(axis),
        iree_async_axis_queue_index(axis),
        iree_hal_amdgpu_notification_ring_epoch_signal(
            &out_queue->notification_ring));
    out_queue->epoch_table = epoch_table;
  }

  if (iree_status_is_ok(status)) {
    memset(&out_queue->progress_entry, 0, sizeof(out_queue->progress_entry));
    out_queue->progress_entry.fn = iree_hal_amdgpu_host_queue_progress_fn;
    out_queue->progress_entry.user_data = out_queue;
    iree_async_proactor_register_progress(proactor, &out_queue->progress_entry);
    // The proactor pool's runner thread may already be blocked in poll() before
    // this cold-path registration happens. Wake it so the proactor observes the
    // non-empty progress list and switches to the non-blocking poll loop that
    // drives notification-ring drains.
    iree_async_proactor_wake(proactor);
  } else {
    iree_hal_amdgpu_host_queue_deinitialize(out_queue);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_amdgpu_host_queue_deinitialize(
    iree_hal_amdgpu_host_queue_t* queue) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_unregister_progress(queue->proactor,
                                          &queue->progress_entry);

  // Cancel all pending (deferred) operations. Their signal semaphores are
  // failed with CANCELLED so downstream waiters don't hang.
  if (queue->pending_head) {
    iree_hal_amdgpu_host_queue_cancel_pending(queue, IREE_STATUS_CANCELLED,
                                              "queue shutting down");
  }

  // Process any remaining notification entries before destroying resources.
  // If the GPU faulted, fail all pending entries so waiters get the actual
  // error. Otherwise drain normally (entries completed but not yet processed).
  iree_status_t error = (iree_status_t)iree_atomic_load(
      &queue->error_status, iree_memory_order_acquire);
  uint64_t kernarg_reclaim_position = 0;
  if (!iree_status_is_ok(error)) {
    iree_hal_amdgpu_notification_ring_fail_all(&queue->notification_ring, error,
                                               &kernarg_reclaim_position);
    iree_status_free(error);
  } else {
    iree_hal_amdgpu_notification_ring_drain(&queue->notification_ring,
                                            /*fallback_frontier=*/NULL,
                                            &kernarg_reclaim_position);
  }
  if (kernarg_reclaim_position > 0) {
    iree_hal_amdgpu_kernarg_ring_reclaim(&queue->kernarg_ring,
                                         kernarg_reclaim_position);
  }

  // Deregister from the epoch signal table before destroying the notification
  // ring (which owns the epoch signal). Guarded by epoch_table != NULL to
  // handle partial initialization (init failed before registration).
  if (queue->epoch_table) {
    iree_hal_amdgpu_epoch_signal_table_deregister(
        queue->epoch_table, iree_async_axis_device_index(queue->axis),
        iree_async_axis_queue_index(queue->axis));
    queue->epoch_table = NULL;
  }

  if (queue->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
        queue->frontier_tracker, queue->axis,
        iree_status_from_code(IREE_STATUS_CANCELLED));
    queue->frontier_tracker = NULL;
    queue->axis = 0;
  }

  iree_hal_amdgpu_notification_ring_deinitialize(&queue->notification_ring);

  iree_hal_amdgpu_kernarg_ring_deinitialize(queue->libhsa,
                                            &queue->kernarg_ring);

  if (queue->pm4_ib_slots) {
    IREE_IGNORE_ERROR(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(queue->libhsa),
                                                    queue->pm4_ib_slots));
    queue->pm4_ib_slots = NULL;
  }

  if (queue->hardware_queue) {
    IREE_IGNORE_ERROR(iree_hsa_queue_destroy(IREE_LIBHSA(queue->libhsa),
                                             queue->hardware_queue));
    queue->hardware_queue = NULL;
  }

  iree_slim_mutex_deinitialize(&queue->submission_mutex);

  IREE_TRACE_ZONE_END(z0);
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

    // Push notification entry for drain → signal_untainted on completion.
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

// Emits one kernel-dispatch submission using an already-prepared packet shape
// and kernargs blob. All queue-private resource transfer and reclaim
// bookkeeping flows through this helper so fill/copy/update/dispatch do not
// accidentally grow divergent kernel-ring retirement mechanisms.
//
// |dispatch_packet_template| must have its packet body populated. Queue-owned
// kernarg_address and completion_signal values are substituted while writing
// the reserved ring slot so each packet field is stored exactly once before the
// final packet header is committed.
//
// |submission_flags| controls whether signal_semaphore_list.semaphores and
// |operation_resources| are retained into the reclaim entry or transferred from
// existing caller-owned retains on success.
//
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_submit_dispatch_packet(
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

static void iree_hal_amdgpu_host_queue_trim(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {}

//===----------------------------------------------------------------------===//
// Queue operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags);

static void iree_hal_amdgpu_host_queue_commit_transient_buffer(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  if (!iree_status_is_ok(status)) return;
  iree_hal_amdgpu_transient_buffer_commit((iree_hal_buffer_t*)user_data);
}

static void iree_hal_amdgpu_host_queue_decommit_transient_buffer(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status) {
  (void)entry;
  if (!iree_status_is_ok(status)) return;
  iree_hal_amdgpu_transient_buffer_decommit((iree_hal_buffer_t*)user_data);
}

static void iree_hal_amdgpu_host_queue_release_transient_buffer_reservation(
    void* user_data, const iree_async_frontier_t* queue_frontier) {
  iree_hal_amdgpu_transient_buffer_release_reservation(
      (iree_hal_buffer_t*)user_data, queue_frontier);
}

static iree_status_t iree_hal_amdgpu_host_queue_resolve_pool(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    iree_hal_pool_t** out_pool) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(out_pool);
  *out_pool = pool ? pool : queue->default_pool;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_prepare_alloca_wrapper(
    iree_hal_amdgpu_host_queue_t* queue, iree_hal_pool_t* pool,
    iree_hal_buffer_params_t* params, iree_device_size_t allocation_size,
    iree_hal_alloca_flags_t flags, iree_hal_pool_t** out_allocation_pool,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_allocation_pool);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_allocation_pool = NULL;
  *out_buffer = NULL;

  if (IREE_UNLIKELY(iree_any_bit_set(
          flags, ~(IREE_HAL_ALLOCA_FLAG_NONE |
                   IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME |
                   IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported alloca flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(allocation_size == 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue_alloca allocation_size must be non-zero");
  }

  iree_hal_pool_t* allocation_pool = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_host_queue_resolve_pool(queue, pool, &allocation_pool));

  iree_hal_buffer_params_canonicalize(params);
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(allocation_pool, &capabilities);
  if (iree_any_bit_set(params->type, IREE_HAL_MEMORY_TYPE_OPTIMAL)) {
    params->type &= ~IREE_HAL_MEMORY_TYPE_OPTIMAL;
    params->type |= capabilities.memory_type;
  }
  if (IREE_UNLIKELY(
          !iree_all_bits_set(capabilities.memory_type, params->type))) {
    iree_bitfield_string_temp_t requested_type_string;
    iree_bitfield_string_temp_t pool_type_string;
    iree_string_view_t requested_type =
        iree_hal_memory_type_format(params->type, &requested_type_string);
    iree_string_view_t pool_type = iree_hal_memory_type_format(
        capabilities.memory_type, &pool_type_string);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation pool does not support requested memory type %.*s "
        "(pool memory type %.*s)",
        (int)requested_type.size, requested_type.data, (int)pool_type.size,
        pool_type.data);
  }
  if (IREE_UNLIKELY(
          !iree_all_bits_set(capabilities.supported_usage, params->usage))) {
    iree_bitfield_string_temp_t requested_usage_string;
    iree_bitfield_string_temp_t pool_usage_string;
    iree_string_view_t requested_usage =
        iree_hal_buffer_usage_format(params->usage, &requested_usage_string);
    iree_string_view_t pool_usage = iree_hal_buffer_usage_format(
        capabilities.supported_usage, &pool_usage_string);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "allocation pool does not support requested buffer usage %.*s "
        "(pool usage %.*s)",
        (int)requested_usage.size, requested_usage.data, (int)pool_usage.size,
        pool_usage.data);
  }

  iree_hal_buffer_placement_t placement = {
      .device = queue->logical_device,
      .queue_affinity = queue->queue_affinity,
      .flags = IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS,
  };
  if (iree_all_bits_set(flags, IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    placement.flags |= IREE_HAL_BUFFER_PLACEMENT_FLAG_INDETERMINATE_LIFETIME;
  }

  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_transient_buffer_create(
      placement, *params, allocation_size, allocation_size,
      queue->host_allocator, out_buffer));
  *out_allocation_pool = allocation_pool;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_ALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->alloca_op.pool = pool;
  op->alloca_op.params = params;
  op->alloca_op.allocation_size = allocation_size;
  op->alloca_op.reserve_flags = reserve_flags;
  op->alloca_op.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

typedef enum iree_hal_amdgpu_alloca_reservation_readiness_e {
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY = 0,
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT = 1,
  IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION = 2,
} iree_hal_amdgpu_alloca_reservation_readiness_t;

typedef struct iree_hal_amdgpu_alloca_reservation_t {
  // Scheduler action required before the reservation can be submitted.
  iree_hal_amdgpu_alloca_reservation_readiness_t readiness;

  // Pool acquisition result that produced |reservation|.
  iree_hal_pool_acquire_result_t acquire_result;

  // Pool-owned byte range reserved for this alloca operation.
  iree_hal_pool_reservation_t reservation;

  // Borrowed metadata returned with |reservation|.
  iree_hal_pool_acquire_info_t acquire_info;

  // Queue wait resolution to use when publishing the alloca signal.
  iree_hal_amdgpu_wait_resolution_t wait_resolution;
} iree_hal_amdgpu_alloca_reservation_t;

static iree_status_t iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags,
    iree_hal_amdgpu_alloca_reservation_t* out_reservation) {
  IREE_ASSERT_ARGUMENT(out_reservation);
  memset(out_reservation, 0, sizeof(*out_reservation));
  out_reservation->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
  out_reservation->acquire_result = IREE_HAL_POOL_ACQUIRE_EXHAUSTED;
  out_reservation->wait_resolution = *resolution;

  iree_hal_amdgpu_fixed_frontier_t requester_frontier_storage;
  const iree_async_frontier_t* requester_frontier =
      iree_hal_amdgpu_host_queue_pool_requester_frontier(
          queue, resolution, &requester_frontier_storage);

  IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservation(
      allocation_pool, allocation_size,
      params.min_alignment ? params.min_alignment : 1, requester_frontier,
      reserve_flags, &out_reservation->reservation,
      &out_reservation->acquire_info, &out_reservation->acquire_result));

  switch (out_reservation->acquire_result) {
    case IREE_HAL_POOL_ACQUIRE_OK:
    case IREE_HAL_POOL_ACQUIRE_OK_FRESH:
      return iree_ok_status();
    case IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT:
      // A waitable pool reservation is legal whenever the HAL alloca flag
      // permits one. Appending device-side barriers is only one representation;
      // non-local, over-capacity, or forced-DEFER frontiers must route to the
      // cold host-gated memory-readiness path.
      if (iree_hal_amdgpu_host_queue_append_pool_wait_frontier_barriers(
              queue, requester_frontier,
              out_reservation->acquire_info.wait_frontier,
              &out_reservation->wait_resolution)) {
        out_reservation->readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY;
      } else {
        out_reservation->readiness =
            IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT;
      }
      return iree_ok_status();
    case IREE_HAL_POOL_ACQUIRE_EXHAUSTED:
    case IREE_HAL_POOL_ACQUIRE_OVER_BUDGET:
      out_reservation->readiness =
          IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized pool acquire result %u",
                              out_reservation->acquire_result);
  }
}

static iree_status_t iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(
    iree_hal_amdgpu_pending_op_t* op,
    iree_hal_amdgpu_alloca_memory_wait_t** out_wait) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = op->alloca_op.memory_wait;
  if (!wait) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(&op->arena, sizeof(*wait), (void**)&wait));
    memset(wait, 0, sizeof(*wait));
    iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
    op->alloca_op.memory_wait = wait;
  }
  *out_wait = wait;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
    iree_hal_amdgpu_pending_op_t* op,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation) {
  const iree_async_frontier_t* wait_frontier =
      alloca_reservation->acquire_info.wait_frontier;
  if (IREE_UNLIKELY(!wait_frontier)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca waitable pool reservation did not provide a frontier");
  }

  iree_host_size_t wait_frontier_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_frontier_size(wait_frontier->entry_count,
                                                &wait_frontier_size));
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  iree_async_frontier_t* wait_frontier_copy = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(&op->arena, wait_frontier_size,
                                           (void**)&wait_frontier_copy));

  memcpy(wait_frontier_copy, wait_frontier, wait_frontier_size);
  wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
  wait->frontier.reservation = alloca_reservation->reservation;
  wait->frontier.wait_frontier = wait_frontier_copy;
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT,
                    iree_memory_order_release);
  return iree_ok_status();
}

static iree_status_t
iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
    iree_hal_amdgpu_pending_op_t* op, iree_async_notification_t* notification,
    uint32_t wait_token) {
  iree_hal_amdgpu_alloca_memory_wait_t* wait = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_pending_op_ensure_alloca_memory_wait(op, &wait));
  wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_POOL_NOTIFICATION;
  iree_atomic_store(&wait->callback_complete, 1, iree_memory_order_relaxed);
  wait->pool_notification.notification = notification;
  wait->pool_notification.wait_token = wait_token;
  wait->pool_notification.wait_slot =
      (uint8_t)((wait->pool_notification.wait_slot + 1u) & 1u);
  iree_atomic_store(&op->lifecycle_state,
                    IREE_HAL_AMDGPU_PENDING_OP_LIFECYCLE_ARMING_MEMORY_WAIT,
                    iree_memory_order_release);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_alloca_reservation(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  const iree_async_frontier_t* reservation_failure_frontier =
      alloca_reservation->acquire_result == IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT
          ? alloca_reservation->acquire_info.wait_frontier
          : NULL;
  iree_hal_amdgpu_transient_buffer_attach_reservation(
      buffer, allocation_pool, &alloca_reservation->reservation);

  iree_hal_buffer_t* backing_buffer = NULL;
  iree_status_t status = iree_hal_pool_materialize_reservation(
      allocation_pool, params, &alloca_reservation->reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &backing_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_stage_backing(buffer, backing_buffer);
  }
  iree_hal_buffer_release(backing_buffer);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_release_reservation(
        buffer, reservation_failure_frontier);
    return status;
  }

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)buffer,
  };
  status = iree_hal_amdgpu_host_queue_submit_barrier(
      queue, &alloca_reservation->wait_resolution, signal_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_host_queue_commit_transient_buffer,
          .user_data = buffer,
      },
      operation_resources, IREE_ARRAYSIZE(operation_resources),
      /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
      submission_flags);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_release_reservation(
        buffer, reservation_failure_frontier);
  }
  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_alloca_memory_wait_t* memory_wait) {
  iree_hal_amdgpu_alloca_reservation_t alloca_reservation = {
      .readiness = IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY,
      .acquire_result = IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT,
      .reservation = memory_wait->frontier.reservation,
      .acquire_info =
          {
              .wait_frontier = memory_wait->frontier.wait_frontier,
          },
      .wait_resolution = *resolution,
  };
  memory_wait->kind = IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_NONE;
  return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
      queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
      params, buffer, submission_flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_pending_op_t* pending_op,
    bool* out_allocated_memory_wait_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_allocated_memory_wait_op = false;
  *out_memory_wait_op = NULL;
  if (pending_op) {
    *out_memory_wait_op = pending_op;
    return iree_ok_status();
  }

  iree_hal_semaphore_list_t empty_wait_list = iree_hal_semaphore_list_empty();
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_defer_alloca(
      queue, &empty_wait_list, &signal_semaphore_list, allocation_pool, params,
      allocation_size, reserve_flags, buffer, out_memory_wait_op));
  *out_allocated_memory_wait_op = true;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    const iree_hal_amdgpu_alloca_reservation_t* alloca_reservation,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  bool allocated_memory_wait_op = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, params, allocation_size,
      reserve_flags, buffer, pending_op, &allocated_memory_wait_op,
      &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_frontier_wait(
        memory_wait_op, alloca_reservation);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else {
    iree_hal_pool_release_reservation(
        allocation_pool, &alloca_reservation->reservation,
        alloca_reservation->acquire_info.wait_frontier);
    if (allocated_memory_wait_op) {
      iree_hal_amdgpu_pending_op_destroy_under_lock(memory_wait_op,
                                                    iree_status_clone(status));
    }
  }

  return status;
}

static iree_status_t
iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  iree_async_notification_t* notification =
      iree_hal_pool_notification(allocation_pool);
  if (IREE_UNLIKELY(!notification)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue_alloca exhausted pool did not provide a notification");
  }

  const uint32_t wait_token = iree_async_notification_query_epoch(notification);
  iree_hal_amdgpu_alloca_reservation_t alloca_reservation;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
      queue, resolution, allocation_pool, params, allocation_size,
      reserve_flags, &alloca_reservation));
  switch (alloca_reservation.readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
      return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
          queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
          params, buffer, submission_flags);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, params,
          allocation_size, reserve_flags, buffer, &alloca_reservation,
          pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
      break;
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              alloca_reservation.readiness);
  }

  iree_hal_amdgpu_pending_op_t* memory_wait_op = pending_op;
  bool allocated_memory_wait_op = false;
  iree_status_t status = iree_hal_amdgpu_host_queue_get_alloca_memory_wait_op(
      queue, signal_semaphore_list, allocation_pool, params, allocation_size,
      reserve_flags, buffer, pending_op, &allocated_memory_wait_op,
      &memory_wait_op);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdgpu_pending_op_prepare_alloca_pool_notification_wait(
        memory_wait_op, notification, wait_token);
  }
  if (iree_status_is_ok(status)) {
    *out_memory_wait_op = memory_wait_op;
  } else if (allocated_memory_wait_op) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(memory_wait_op,
                                                  iree_status_clone(status));
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_alloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* allocation_pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    iree_hal_pool_reserve_flags_t reserve_flags, iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags,
    iree_hal_amdgpu_pending_op_t* pending_op,
    iree_hal_amdgpu_pending_op_t** out_memory_wait_op) {
  *out_memory_wait_op = NULL;
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hal_amdgpu_alloca_memory_wait_t* memory_wait =
      pending_op ? pending_op->alloca_op.memory_wait : NULL;
  if (memory_wait &&
      memory_wait->kind == IREE_HAL_AMDGPU_ALLOCA_MEMORY_WAIT_FRONTIER) {
    return iree_hal_amdgpu_host_queue_submit_alloca_held_frontier_wait(
        queue, resolution, signal_semaphore_list, allocation_pool, params,
        buffer, submission_flags, memory_wait);
  }

  iree_hal_amdgpu_alloca_reservation_t alloca_reservation;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_acquire_alloca_reservation(
      queue, resolution, allocation_pool, params, allocation_size,
      reserve_flags, &alloca_reservation));
  switch (alloca_reservation.readiness) {
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_READY:
      return iree_hal_amdgpu_host_queue_submit_alloca_reservation(
          queue, &alloca_reservation, signal_semaphore_list, allocation_pool,
          params, buffer, submission_flags);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_FRONTIER_WAIT:
      return iree_hal_amdgpu_host_queue_defer_alloca_frontier_wait(
          queue, signal_semaphore_list, allocation_pool, params,
          allocation_size, reserve_flags, buffer, &alloca_reservation,
          pending_op, out_memory_wait_op);
    case IREE_HAL_AMDGPU_ALLOCA_RESERVATION_NEEDS_POOL_NOTIFICATION:
      return iree_hal_amdgpu_host_queue_defer_alloca_pool_notification_wait(
          queue, resolution, signal_semaphore_list, allocation_pool, params,
          allocation_size, reserve_flags, buffer, submission_flags, pending_op,
          out_memory_wait_op);
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unrecognized alloca reservation readiness %u",
                              alloca_reservation.readiness);
  }
}

static iree_status_t iree_hal_amdgpu_host_queue_alloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size, iree_hal_alloca_flags_t flags,
    iree_hal_buffer_t** IREE_RESTRICT out_buffer) {
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;

  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_hal_pool_t* allocation_pool = NULL;
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_alloca_wrapper(
      queue, pool, &params, allocation_size, flags, &allocation_pool, &buffer));
  // The HAL alloca flag is semantic permission to consume a pool death-frontier
  // dependency. The selected queue wait strategy only decides how that hidden
  // dependency is represented; it must not cause the pool to skip an otherwise
  // legal waitable reservation and report exhaustion instead.
  const iree_hal_pool_reserve_flags_t reserve_flags =
      iree_all_bits_set(flags, IREE_HAL_ALLOCA_FLAG_ALLOW_POOL_WAIT_FRONTIER)
          ? IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER
          : IREE_HAL_POOL_RESERVE_FLAG_NONE;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  iree_hal_amdgpu_pending_op_t* memory_wait_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_alloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, allocation_pool,
        params, allocation_size, reserve_flags, buffer, &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_alloca(
        queue, &resolution, signal_semaphore_list, allocation_pool, params,
        allocation_size, reserve_flags, buffer,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES,
        /*pending_op=*/NULL, &memory_wait_op);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  if (iree_status_is_ok(status) && memory_wait_op) {
    iree_hal_amdgpu_pending_op_enqueue_alloca_memory_wait(memory_wait_op);
  }

  if (iree_status_is_ok(status)) {
    *out_buffer = buffer;
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_defer_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DEALLOCA, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)buffer);
  op->dealloca.buffer = buffer;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_dealloca(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)buffer,
  };
  return iree_hal_amdgpu_host_queue_submit_barrier(
      queue, resolution, signal_semaphore_list,
      (iree_hal_amdgpu_reclaim_action_t){
          .fn = iree_hal_amdgpu_host_queue_decommit_transient_buffer,
          .user_data = buffer,
      },
      operation_resources, IREE_ARRAYSIZE(operation_resources),
      iree_hal_amdgpu_host_queue_release_transient_buffer_reservation, buffer,
      submission_flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_dealloca(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  if (IREE_UNLIKELY(
          iree_any_bit_set(flags, ~(IREE_HAL_DEALLOCA_FLAG_NONE |
                                    IREE_HAL_DEALLOCA_FLAG_PREFER_ORIGIN)))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported dealloca flags: 0x%" PRIx64, flags);
  }

  // iree_hal_device_queue_dealloca() applies PREFER_ORIGIN before vtable
  // dispatch by rewriting the device and queue affinity from the buffer's
  // allocation placement. Transient wrappers created by queue_alloca carry this
  // queue's one-bit affinity in that placement, so this host-queue path can use
  // |base_queue| directly.
  if (!iree_hal_amdgpu_transient_buffer_isa(buffer)) {
    return iree_hal_amdgpu_host_queue_execute(
        base_queue, wait_semaphore_list, signal_semaphore_list,
        /*command_buffer=*/NULL, iree_hal_buffer_binding_table_empty(),
        IREE_HAL_EXECUTE_FLAG_NONE);
  }

  if (IREE_UNLIKELY(!iree_hal_amdgpu_transient_buffer_begin_dealloca(buffer))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "transient buffer has already been queued for deallocation");
  }

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_dealloca(
        queue, &wait_semaphore_list, &signal_semaphore_list, buffer,
        &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_dealloca(
        queue, &resolution, signal_semaphore_list, buffer,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_transient_buffer_abort_dealloca(buffer);
  }
  return status;
}

// Captures a fill operation into a pending op. Does NOT call enqueue_waits —
// the caller must release submission_mutex before calling enqueue_waits on the
// returned op (Phase 2 of the two-phase protocol).
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_FILL, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->fill.target_buffer = target_buffer;
  op->fill.target_offset = target_offset;
  op->fill.length = length;
  op->fill.pattern_bits = pattern_bits;
  op->fill.pattern_length = pattern_length;
  op->fill.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static_assert(IREE_HAL_AMDGPU_DEVICE_BUFFER_FILL_KERNARG_SIZE <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "fill kernargs must fit in one kernarg ring block");

// Prepares a fill dispatch packet and kernargs in stack-local storage without
// touching queue rings. All user-input validation must happen here so the
// caller can avoid reserving AQL slots before the packet shape is known-valid.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_fill_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    uint64_t pattern_bits, iree_host_size_t pattern_length,
    iree_hal_fill_flags_t flags,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_device_buffer_fill_kernargs_t* out_kernargs) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_ASSERT_ARGUMENT(out_dispatch_packet);
  IREE_ASSERT_ARGUMENT(out_kernargs);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  if (IREE_UNLIKELY(pattern_length != 1 && pattern_length != 2 &&
                    pattern_length != 4)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill patterns must be 1, 2, or 4 bytes (got %" PRIhsz ")",
        pattern_length);
  }
  if (IREE_UNLIKELY(flags != IREE_HAL_FILL_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported fill flags: 0x%" PRIx64, flags);
  }

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_fill_emplace(
          queue->transfer_context, &dispatch_packet, target_device_ptr, length,
          pattern_bits, (uint8_t)pattern_length, &kernargs))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported fill dispatch shape (target_offset=%" PRIdsz
        ", length=%" PRIdsz ", pattern_length=%" PRIhsz ")",
        target_offset, length, pattern_length);
  }
  dispatch_packet.kernarg_address = NULL;

  *out_dispatch_packet = dispatch_packet;
  *out_kernargs = kernargs;
  return iree_ok_status();
}

// Emits an inline or deferred fill submission on |queue|.
//
// |submission_flags| controls whether signal_semaphore_list.semaphores and
// |target_buffer| are retained into the reclaim entry or transferred from
// existing caller-owned retains on success.
//
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_submit_fill(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_fill_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_fill_dispatch(
      queue, target_buffer, target_offset, length, pattern_bits, pattern_length,
      flags, &dispatch_packet, &kernargs));

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_fill(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, uint64_t pattern_bits,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  // Phase 1 (under mutex): resolve waits and capture if deferred.
  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_fill(
        queue, &wait_semaphore_list, &signal_semaphore_list, target_buffer,
        target_offset, length, pattern_bits, pattern_length, flags,
        &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_fill(
        queue, &resolution, signal_semaphore_list, target_buffer, target_offset,
        length, pattern_bits, pattern_length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  // Phase 2 (without mutex): register timepoints. Callbacks may fire
  // synchronously and re-acquire submission_mutex via pending_op_issue/fail.
  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Captures a copy operation into a pending op.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/2, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_COPY, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)source_buffer);
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);
  op->copy.source_buffer = source_buffer;
  op->copy.source_offset = source_offset;
  op->copy.target_buffer = target_buffer;
  op->copy.target_offset = target_offset;
  op->copy.length = length;
  op->copy.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static_assert(IREE_HAL_AMDGPU_DEVICE_BUFFER_COPY_KERNARG_SIZE <=
                  sizeof(iree_hal_amdgpu_kernarg_block_t),
              "copy kernargs must fit in one kernarg ring block");

// Prepares a copy dispatch packet and kernargs in stack-local storage without
// touching queue rings. Overlapping ranges within the same buffer are rejected
// here because the builtin copy kernels implement memcpy semantics, not
// memmove.
static iree_status_t iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
    const iree_hal_amdgpu_host_queue_t* queue, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    iree_hal_copy_flags_t flags,
    iree_hsa_kernel_dispatch_packet_t* out_dispatch_packet,
    iree_hal_amdgpu_device_buffer_copy_kernargs_t* out_kernargs) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(source_buffer);
  IREE_ASSERT_ARGUMENT(target_buffer);
  IREE_ASSERT_ARGUMENT(out_dispatch_packet);
  IREE_ASSERT_ARGUMENT(out_kernargs);

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  if (IREE_UNLIKELY(flags != IREE_HAL_COPY_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported copy flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(iree_hal_buffer_test_overlap(source_buffer, source_offset,
                                                 length, target_buffer,
                                                 target_offset, length) !=
                    IREE_HAL_BUFFER_OVERLAP_DISJOINT)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges must not overlap within the same buffer");
  }

  iree_hal_buffer_t* allocated_source_buffer =
      iree_hal_buffer_allocated_buffer(source_buffer);
  const uint8_t* source_device_ptr =
      (const uint8_t*)iree_hal_amdgpu_buffer_device_pointer(
          allocated_source_buffer);
  if (IREE_UNLIKELY(!source_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source buffer must be backed by an AMDGPU allocation");
  }
  source_device_ptr +=
      iree_hal_buffer_byte_offset(source_buffer) + source_offset;

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &dispatch_packet, source_device_ptr,
          target_device_ptr, length, &kernargs))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported copy dispatch shape (source_offset=%" PRIdsz
        ", target_offset=%" PRIdsz ", length=%" PRIdsz ")",
        source_offset, target_offset, length);
  }
  dispatch_packet.kernarg_address = NULL;

  *out_dispatch_packet = dispatch_packet;
  *out_kernargs = kernargs;
  return iree_ok_status();
}

// Emits an inline or deferred copy submission on |queue|.
//
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_submit_copy(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_copy_dispatch(
      queue, source_buffer, source_offset, target_buffer, target_offset, length,
      flags, &dispatch_packet, &kernargs));

  iree_hal_resource_t* operation_resources[2] = {
      (iree_hal_resource_t*)source_buffer,
      (iree_hal_resource_t*)target_buffer,
  };
  return iree_hal_amdgpu_host_queue_submit_dispatch_packet(
      queue, resolution, signal_semaphore_list, &dispatch_packet, &kernargs,
      sizeof(kernargs), operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags);
}

static iree_status_t iree_hal_amdgpu_host_queue_copy(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_copy_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_copy(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_copy(
        queue, &resolution, signal_semaphore_list, source_buffer, source_offset,
        target_buffer, target_offset, length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

#define IREE_HAL_AMDGPU_HOST_QUEUE_UPDATE_SOURCE_ALIGNMENT 16

// Validates a queue_update request and resolves the source host span and target
// device pointer. The source host pointer is captured by the caller either into
// the pending-op arena (deferred path) or into the queue-owned kernarg ring
// (inline issue path).
static iree_status_t iree_hal_amdgpu_host_queue_prepare_update_copy(
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    const uint8_t** out_source_bytes, iree_host_size_t* out_source_length,
    uint8_t** out_target_device_ptr) {
  IREE_ASSERT_ARGUMENT(out_source_bytes);
  IREE_ASSERT_ARGUMENT(out_source_length);
  IREE_ASSERT_ARGUMENT(out_target_device_ptr);
  *out_source_bytes = NULL;
  *out_source_length = 0;
  *out_target_device_ptr = NULL;

  if (IREE_UNLIKELY(!source_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source buffer must be non-null");
  }
  if (IREE_UNLIKELY(!target_buffer)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target buffer must be non-null");
  }

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  if (IREE_UNLIKELY(flags != IREE_HAL_UPDATE_FLAG_NONE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported update flags: 0x%" PRIx64, flags);
  }
  if (IREE_UNLIKELY(length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "update length %" PRIdsz
                            " exceeds host addressable size %" PRIhsz,
                            length, IREE_HOST_SIZE_MAX);
  }
  const iree_host_size_t source_length = (iree_host_size_t)length;
  iree_host_size_t source_end = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(source_offset, source_length,
                                                &source_end))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update source span overflows host size (offset=%" PRIhsz
        ", length=%" PRIhsz ")",
        source_offset, source_length);
  }
  (void)source_end;

  iree_hal_buffer_t* allocated_target_buffer =
      iree_hal_buffer_allocated_buffer(target_buffer);
  uint8_t* target_device_ptr =
      (uint8_t*)iree_hal_amdgpu_buffer_device_pointer(allocated_target_buffer);
  if (IREE_UNLIKELY(!target_device_ptr)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target buffer must be backed by an AMDGPU allocation");
  }
  target_device_ptr +=
      iree_hal_buffer_byte_offset(target_buffer) + target_offset;

  *out_source_bytes = (const uint8_t*)source_buffer + source_offset;
  *out_source_length = source_length;
  *out_target_device_ptr = target_device_ptr;
  return iree_ok_status();
}

// Captures an update operation into a pending op. Copies the source host data
// into the arena (the caller's buffer may be freed after this returns).
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  const uint8_t* source_bytes = NULL;
  iree_host_size_t source_length = 0;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_update_copy(
      target_buffer, target_offset, source_buffer, source_offset, length, flags,
      &source_bytes, &source_length, &target_device_ptr));
  (void)target_device_ptr;

  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count,
      /*operation_resource_count=*/1, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_UPDATE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)target_buffer);

  // Copy the source host data into the arena. The caller's buffer may be
  // freed after this call returns.
  void* source_copy = NULL;
  iree_status_t status =
      iree_arena_allocate(&op->arena, source_length, &source_copy);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_update");
  }
  memcpy(source_copy, source_bytes, source_length);
  op->update.source_data = source_copy;
  op->update.target_buffer = target_buffer;
  op->update.target_offset = target_offset;
  op->update.length = length;
  op->update.flags = flags;
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_submit_update(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_amdgpu_wait_resolution_t* resolution,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags,
    iree_hal_amdgpu_host_queue_submission_flags_t submission_flags) {
  if (IREE_UNLIKELY(queue->is_shutting_down)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "queue shutting down");
  }

  const uint8_t* source_bytes = NULL;
  iree_host_size_t source_length = 0;
  uint8_t* target_device_ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_prepare_update_copy(
      target_buffer, target_offset, source_buffer, source_offset, length, flags,
      &source_bytes, &source_length, &target_device_ptr));

  const iree_host_size_t source_payload_offset =
      iree_host_align(sizeof(iree_hal_amdgpu_device_buffer_copy_kernargs_t),
                      IREE_HAL_AMDGPU_HOST_QUEUE_UPDATE_SOURCE_ALIGNMENT);
  iree_host_size_t kernarg_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          source_payload_offset, source_length, &kernarg_length))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update staging payload overflows host size (offset=%" PRIhsz
        ", source_length=%" PRIhsz ")",
        source_payload_offset, source_length);
  }
  const iree_host_size_t kernarg_block_count = iree_host_size_ceil_div(
      kernarg_length, sizeof(iree_hal_amdgpu_kernarg_block_t));
  if (IREE_UNLIKELY(kernarg_block_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "update staging payload requires too many kernarg blocks (%" PRIhsz
        ", max=%u)",
        kernarg_block_count, UINT32_MAX);
  }

  iree_hsa_kernel_dispatch_packet_t dispatch_packet;
  memset(&dispatch_packet, 0, sizeof(dispatch_packet));
  iree_hal_amdgpu_device_buffer_copy_kernargs_t kernargs;
  memset(&kernargs, 0, sizeof(kernargs));
  // The eventual staged source pointer is 16-byte aligned by construction. Use
  // a synthetic aligned pointer for pre-reservation packet-shape selection,
  // then patch source_ptr to the real ring address after allocation succeeds.
  if (IREE_UNLIKELY(!iree_hal_amdgpu_device_buffer_copy_emplace(
          queue->transfer_context, &dispatch_packet,
          (const void*)(uintptr_t)
              IREE_HAL_AMDGPU_HOST_QUEUE_UPDATE_SOURCE_ALIGNMENT,
          target_device_ptr, length, &kernargs))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "unsupported update dispatch shape (target_offset=%" PRIdsz
        ", length=%" PRIdsz ", source_payload_alignment=%d)",
        target_offset, length,
        IREE_HAL_AMDGPU_HOST_QUEUE_UPDATE_SOURCE_ALIGNMENT);
  }

  iree_hal_amdgpu_host_queue_dispatch_submission_t submission;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_begin_dispatch_submission(
      queue, resolution, signal_semaphore_list,
      /*operation_resource_count=*/1, (uint32_t)kernarg_block_count,
      &submission));

  uint8_t* staged_source_bytes =
      (uint8_t*)submission.kernarg_blocks + source_payload_offset;
  memcpy(submission.kernarg_blocks->data, &kernargs, sizeof(kernargs));
  ((iree_hal_amdgpu_device_buffer_copy_kernargs_t*)
       submission.kernarg_blocks->data)
      ->source_ptr = staged_source_bytes;
  memcpy(staged_source_bytes, source_bytes, source_length);
  submission.dispatch_setup =
      iree_hal_amdgpu_host_queue_write_dispatch_packet_body(
          &submission.dispatch_slot->dispatch, &dispatch_packet,
          submission.kernarg_blocks->data,
          iree_hal_amdgpu_notification_ring_epoch_signal(
              &queue->notification_ring));

  iree_hal_resource_t* operation_resources[1] = {
      (iree_hal_resource_t*)target_buffer,
  };
  iree_hal_amdgpu_host_queue_finish_dispatch_submission(
      queue, resolution, signal_semaphore_list, operation_resources,
      IREE_ARRAYSIZE(operation_resources), submission_flags, &submission);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_update(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    const void* source_buffer, iree_host_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_update_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_update(
        queue, &wait_semaphore_list, &signal_semaphore_list, source_buffer,
        source_offset, target_buffer, target_offset, length, flags,
        &deferred_op);
  } else {
    status = iree_hal_amdgpu_host_queue_submit_update(
        queue, &resolution, signal_semaphore_list, source_buffer, source_offset,
        target_buffer, target_offset, length, flags,
        IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Captures a dispatch operation into a pending op. Copies constants and
// bindings into the arena.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_dispatch(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags,
    iree_hal_amdgpu_pending_op_t** out_op) {
  // 1 executable + up to bindings.count buffers.
  iree_host_size_t operation_resource_count = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(1, bindings.count,
                                                &operation_resource_count))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch retains too many resources (bindings=%" PRIhsz ")",
        bindings.count);
  }
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_DISPATCH, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)executable);
  op->dispatch.executable = executable;
  op->dispatch.export_ordinal = export_ordinal;
  op->dispatch.config = config;
  op->dispatch.flags = flags;

  // Copy constants into the arena.
  iree_status_t status = iree_ok_status();
  if (constants.data_length > 0) {
    void* constants_copy = NULL;
    status =
        iree_arena_allocate(&op->arena, constants.data_length, &constants_copy);
    if (iree_status_is_ok(status)) {
      memcpy(constants_copy, constants.data, constants.data_length);
      op->dispatch.constants.data = (const uint8_t*)constants_copy;
      op->dispatch.constants.data_length = constants.data_length;
    }
  }

  // Copy bindings array and retain all bound buffers.
  if (iree_status_is_ok(status) && bindings.count > 0) {
    iree_hal_buffer_ref_t* bindings_copy = NULL;
    status = iree_arena_allocate(&op->arena,
                                 bindings.count * sizeof(iree_hal_buffer_ref_t),
                                 (void**)&bindings_copy);
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, bindings.values,
             bindings.count * sizeof(iree_hal_buffer_ref_t));
      for (iree_host_size_t i = 0; i < bindings.count; ++i) {
        iree_hal_amdgpu_pending_op_retain(
            op, (iree_hal_resource_t*)bindings_copy[i].buffer);
      }
      op->dispatch.bindings.count = bindings.count;
      op->dispatch.bindings.values = bindings_copy;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_dispatch");
  }
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_dispatch(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_list_t bindings,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_dispatch(
        queue, &wait_semaphore_list, &signal_semaphore_list, executable,
        export_ordinal, config, constants, bindings, flags, &deferred_op);
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "inline dispatch AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

// Submits a barrier-only execute operation with no command buffer payload.
// This is the host-queue path for iree_hal_device_queue_barrier().
//
// The queue consumes one completion-producing AQL packet per submission so
// no-signal barrier submissions retire through the same notification/reclaim
// mechanism as kernel dispatches. If the submission already has wait barriers,
// the queue epoch completion signal is attached to the final wait barrier;
// otherwise a standalone completion packet is emitted.
//
// Caller must hold submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_submit_barrier(
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

// Captures an execute operation into a pending op. Copies binding table into
// the arena.
// Caller must hold submission_mutex. Caller must call enqueue_waits after
// releasing submission_mutex.
static iree_status_t iree_hal_amdgpu_host_queue_defer_execute(
    iree_hal_amdgpu_host_queue_t* queue,
    const iree_hal_semaphore_list_t* wait_semaphore_list,
    const iree_hal_semaphore_list_t* signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags, iree_hal_amdgpu_pending_op_t** out_op) {
  if (IREE_UNLIKELY(!command_buffer && binding_table.count != 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "barrier-only queue_execute must not provide a binding table "
        "(count=%" PRIhsz ")",
        binding_table.count);
  }

  // Optional command buffer + up to binding_table.count buffers.
  iree_host_size_t operation_resource_count = command_buffer ? 1 : 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(operation_resource_count,
                                                binding_table.count,
                                                &operation_resource_count))) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "execute retains too many resources (bindings=%" PRIhsz ")",
        binding_table.count);
  }
  uint16_t max_resources = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_host_queue_count_reclaim_resources(
      signal_semaphore_list->count, operation_resource_count, &max_resources));
  iree_hal_amdgpu_pending_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pending_op_allocate(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDGPU_PENDING_OP_EXECUTE, max_resources, &op));
  iree_hal_amdgpu_pending_op_retain(op, (iree_hal_resource_t*)command_buffer);
  op->execute.command_buffer = command_buffer;
  op->execute.flags = flags;

  // Copy binding table and retain all bound buffers.
  iree_status_t status = iree_ok_status();
  if (binding_table.count > 0) {
    iree_hal_buffer_binding_t* bindings_copy = NULL;
    status = iree_arena_allocate(
        &op->arena, binding_table.count * sizeof(iree_hal_buffer_binding_t),
        (void**)&bindings_copy);
    if (iree_status_is_ok(status)) {
      memcpy(bindings_copy, binding_table.bindings,
             binding_table.count * sizeof(iree_hal_buffer_binding_t));
      for (iree_host_size_t i = 0; i < binding_table.count; ++i) {
        iree_hal_amdgpu_pending_op_retain(
            op, (iree_hal_resource_t*)bindings_copy[i].buffer);
      }
      op->execute.binding_table.count = binding_table.count;
      op->execute.binding_table.bindings = bindings_copy;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_hal_amdgpu_pending_op_destroy_under_lock(op, status);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "arena allocation failed during defer_execute");
  }
  *out_op = op;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdgpu_host_queue_execute(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_buffer_binding_table_t binding_table,
    iree_hal_execute_flags_t flags) {
  iree_hal_amdgpu_host_queue_t* queue =
      (iree_hal_amdgpu_host_queue_t*)base_queue;

  iree_slim_mutex_lock(&queue->submission_mutex);
  iree_hal_amdgpu_wait_resolution_t resolution;
  iree_hal_amdgpu_host_queue_resolve_waits(queue, wait_semaphore_list,
                                           &resolution);
  iree_status_t status = iree_ok_status();
  iree_hal_amdgpu_pending_op_t* deferred_op = NULL;
  if (resolution.needs_deferral) {
    status = iree_hal_amdgpu_host_queue_defer_execute(
        queue, &wait_semaphore_list, &signal_semaphore_list, command_buffer,
        binding_table, flags, &deferred_op);
  } else if (!command_buffer) {
    if (IREE_UNLIKELY(binding_table.count != 0)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "barrier-only queue_execute must not provide a binding table "
          "(count=%" PRIhsz ")",
          binding_table.count);
    } else {
      status = iree_hal_amdgpu_host_queue_submit_barrier(
          queue, &resolution, signal_semaphore_list,
          (iree_hal_amdgpu_reclaim_action_t){0},
          /*operation_resources=*/NULL,
          /*operation_resource_count=*/0,
          /*post_commit_fn=*/NULL, /*post_commit_user_data=*/NULL,
          IREE_HAL_AMDGPU_HOST_QUEUE_SUBMISSION_FLAG_RETAIN_RESOURCES);
    }
  } else {
    status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "inline execute AQL emission");
  }
  iree_slim_mutex_unlock(&queue->submission_mutex);

  if (iree_status_is_ok(status) && deferred_op) {
    status = iree_hal_amdgpu_pending_op_enqueue_waits(deferred_op);
  }
  return status;
}

static iree_status_t iree_hal_amdgpu_host_queue_read(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue read not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_write(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue write not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_host_call(
    iree_hal_amdgpu_virtual_queue_t* base_queue,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_host_call_t call, const uint64_t args[4],
    iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "host_queue host_call not yet implemented");
}

static iree_status_t iree_hal_amdgpu_host_queue_flush(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Virtual queue vtable
//===----------------------------------------------------------------------===//

static void iree_hal_amdgpu_host_queue_deinitialize_vtable(
    iree_hal_amdgpu_virtual_queue_t* base_queue) {
  iree_hal_amdgpu_host_queue_deinitialize(
      (iree_hal_amdgpu_host_queue_t*)base_queue);
}

static const iree_hal_amdgpu_virtual_queue_vtable_t
    iree_hal_amdgpu_host_queue_vtable = {
        .deinitialize = iree_hal_amdgpu_host_queue_deinitialize_vtable,
        .trim = iree_hal_amdgpu_host_queue_trim,
        .alloca = iree_hal_amdgpu_host_queue_alloca,
        .dealloca = iree_hal_amdgpu_host_queue_dealloca,
        .fill = iree_hal_amdgpu_host_queue_fill,
        .update = iree_hal_amdgpu_host_queue_update,
        .copy = iree_hal_amdgpu_host_queue_copy,
        .read = iree_hal_amdgpu_host_queue_read,
        .write = iree_hal_amdgpu_host_queue_write,
        .host_call = iree_hal_amdgpu_host_queue_host_call,
        .dispatch = iree_hal_amdgpu_host_queue_dispatch,
        .execute = iree_hal_amdgpu_host_queue_execute,
        .flush = iree_hal_amdgpu_host_queue_flush,
};
