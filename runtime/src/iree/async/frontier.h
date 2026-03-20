// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_FRONTIER_H_
#define IREE_ASYNC_FRONTIER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Causal domain
//===----------------------------------------------------------------------===//

// Identifies the category of timeline participant represented by an axis.
// Each domain has its own ordinal encoding scheme within the axis (see
// iree_async_axis_t documentation for the full encoding).
enum iree_async_causal_domain_e {
  // Hardware or virtual device queue.
  IREE_ASYNC_CAUSAL_DOMAIN_QUEUE = 0,
  // Collective communication channel.
  IREE_ASYNC_CAUSAL_DOMAIN_COLLECTIVE = 1,
  // Epilogue kernel timeline.
  IREE_ASYNC_CAUSAL_DOMAIN_EPILOGUE = 2,
  // Host-side virtual queue (thread or execution context).
  IREE_ASYNC_CAUSAL_DOMAIN_HOST = 3,
};
typedef uint8_t iree_async_causal_domain_t;

//===----------------------------------------------------------------------===//
// Axis
//===----------------------------------------------------------------------===//

// A 64-bit identifier for a timeline participant in the causal ordering system.
//
// Every independent source of progress in the system — every GPU queue, every
// collective channel, every host thread — is an "axis." Each axis has a
// monotonically increasing timeline value (epoch). An axis at epoch N means
// "all work submitted to this participant up to sequence point N has
// completed."
//
// Axes are the building blocks of frontiers. A frontier says "axis A is at
// epoch 5, axis B is at epoch 12" — meaning the referenced event depends on
// both A having completed 5 operations and B having completed 12.
//
// The encoding is structured for debuggability and deterministic allocation:
//
//   ┌────────────┬────────────┬────────────┬──────────────────────────────┐
//   │ Bits 63-56 │ Bits 55-48 │ Bits 47-40 │         Bits 39-0            │
//   ├────────────┼────────────┼────────────┼──────────────────────────────┤
//   │  Session   │  Machine   │   Domain   │           Ordinal            │
//   │   Epoch    │   Index    │            │     (structured by domain)   │
//   │   (8 b)    │   (8 b)    │   (8 b)    │           (40 b)             │
//   └────────────┴────────────┴────────────┴──────────────────────────────┘
//
//   Session epoch: Increments per session creation. Prevents ABA problems
//     across sessions — an axis from session 3 can never be confused with
//     one from session 4 even if machine/domain/ordinal happen to match.
//
//   Machine index: 0-255 within a session. Assigned by the coordinator at
//     session join time. Identifies which physical machine this axis lives on.
//
//   Domain: Which category of timeline participant (see
//   iree_async_causal_domain_t).
//
//   Ordinal: Domain-specific structured identifier:
//     QUEUE domain:
//       [39:32] device_index (8 bits) — 256 devices per machine
//       [31:24] queue_index  (8 bits) — 256 queues per device
//       [23:0]  reserved     (24 bits)
//     COLLECTIVE domain:
//       [39:0]  channel_id   (40 bits) — coordinator-assigned
//     HOST domain:
//       [39:0]  context_id   (40 bits) — thread/context identifier
//
// Axes are computed deterministically from logical topology: the same topology
// description always produces the same axis values, regardless of which
// physical GPU fills each slot. This makes frontiers directly comparable across
// machines without translation.
//
// Example axis values (session 1, machine 3):
//   GPU queue 0 on device 1:  0x01_03_00_01_00_000000
//   Collective channel 42:    0x01_03_01_00_00_00002A
//   Host context 7:           0x01_03_03_00_00_000007
typedef uint64_t iree_async_axis_t;

// Constructs an axis from its four components. The ordinal is masked to 40
// bits; the caller is responsible for encoding domain-specific structure within
// the ordinal.
static inline iree_async_axis_t iree_async_axis_make(
    uint8_t session_epoch, uint8_t machine_index,
    iree_async_causal_domain_t domain, uint64_t ordinal) {
  return ((uint64_t)session_epoch << 56) | ((uint64_t)machine_index << 48) |
         ((uint64_t)domain << 40) | (ordinal & 0xFFFFFFFFFFull);
}

// Constructs a QUEUE-domain axis from device and queue indices.
// This is the most common axis construction — one per GPU queue in the system.
static inline iree_async_axis_t iree_async_axis_make_queue(
    uint8_t session_epoch, uint8_t machine_index, uint8_t device_index,
    uint8_t queue_index) {
  uint64_t ordinal =
      ((uint64_t)device_index << 32) | ((uint64_t)queue_index << 24);
  return iree_async_axis_make(session_epoch, machine_index,
                              IREE_ASYNC_CAUSAL_DOMAIN_QUEUE, ordinal);
}

// Extracts the session epoch from an axis.
static inline uint8_t iree_async_axis_session(iree_async_axis_t axis) {
  return (uint8_t)(axis >> 56);
}

// Extracts the machine index from an axis.
static inline uint8_t iree_async_axis_machine(iree_async_axis_t axis) {
  return (uint8_t)(axis >> 48);
}

// Extracts the causal domain from an axis.
static inline iree_async_causal_domain_t iree_async_axis_domain(
    iree_async_axis_t axis) {
  return (iree_async_causal_domain_t)((axis >> 40) & 0xFF);
}

// Extracts the raw 40-bit ordinal from an axis. Interpretation depends on the
// domain (see iree_async_axis_t encoding documentation).
static inline uint64_t iree_async_axis_ordinal(iree_async_axis_t axis) {
  return axis & 0xFFFFFFFFFFull;
}

// Extracts the device index from a QUEUE-domain axis.
// Undefined behavior if the axis is not in the QUEUE domain.
static inline uint8_t iree_async_axis_device_index(iree_async_axis_t axis) {
  return (uint8_t)(iree_async_axis_ordinal(axis) >> 32);
}

// Extracts the queue index from a QUEUE-domain axis.
// Undefined behavior if the axis is not in the QUEUE domain.
static inline uint8_t iree_async_axis_queue_index(iree_async_axis_t axis) {
  return (uint8_t)(iree_async_axis_ordinal(axis) >> 24);
}

//===----------------------------------------------------------------------===//
// Frontier
//===----------------------------------------------------------------------===//

// A single (axis, epoch) pair within a frontier. States that the given axis
// must have reached at least the given epoch for this entry to be satisfied.
typedef struct iree_async_frontier_entry_t {
  iree_async_axis_t axis;
  uint64_t epoch;
} iree_async_frontier_entry_t;

// A frontier is a vector clock: a set of (axis, epoch) pairs that together
// define a causal position in the distributed system. A frontier answers the
// question: "what must have happened before this point?"
//
// ## The problem frontiers solve
//
// In a multi-device, multi-machine system, work proceeds on many independent
// timelines simultaneously. A single buffer might be:
//   - Produced by GPU0's compute queue (dispatch 47)
//   - Transferred via network (send sequence 12)
//   - Have its metadata updated by the host (context 3, step 8)
//
// Before you can safely read that buffer, ALL THREE of those timelines must
// have reached their respective points. A single semaphore tracks one timeline.
// A frontier tracks all of them at once.
//
// ## Visual intuition
//
// Think of time flowing left-to-right on multiple parallel tracks:
//
//   GPU0 queue0: ──①──②──③──④──⑤──⑥──⑦──▶
//   GPU0 queue1: ──①──②──③──④──▶
//   GPU1 queue0: ──①──②──③──④──⑤──▶
//   Network:     ──①──②──③──▶
//
// A frontier is a vertical "cut" across these timelines:
//
//   GPU0 queue0: ──①──②──③──④──⑤──│──⑥──⑦──▶
//   GPU0 queue1: ──①──②──③──│──④──▶
//   GPU1 queue0: ──①──②──③──④──│──⑤──▶
//   Network:     ──①──②──│──③──▶
//                              │
//                         frontier F = {
//                           (GPU0/q0, 5),
//                           (GPU0/q1, 3),
//                           (GPU1/q0, 4),
//                           (Network, 2),
//                         }
//
// Everything to the LEFT of the cut has happened. Everything to the RIGHT
// may or may not have happened yet. The frontier IS the cut.
//
// ## Causal ordering (happens-before)
//
// Two frontiers define a partial order:
//
//   F1 BEFORE F2: Every axis in F1 is at the same or earlier epoch than in F2,
//                 and at least one is strictly earlier. F1's "cut" is entirely
//                 to the left of F2's. Anything that depends on F1 being
//                 satisfied also satisfies F2 (but not vice versa).
//
//   F1 AFTER F2:  The reverse. F2 happens-before F1.
//
//   CONCURRENT:   Neither dominates. F1 is ahead on some axes, F2 is ahead on
//                 others. There is no causal relationship — these events could
//                 have happened in either order (or simultaneously).
//
//   EQUAL:        Same cut. Same causal position.
//
// Example:
//   F1 = {(A, 3), (B, 5)}
//   F2 = {(A, 4), (B, 7)}     → F1 BEFORE F2 (both axes advanced)
//   F3 = {(A, 5), (B, 3)}     → F1 CONCURRENT with F3 (A advanced, B went back)
//
// ## Relationship to semaphores
//
// Each axis in a frontier corresponds to one iree_async_semaphore_t timeline.
// A frontier {(axis_A, 5), (axis_B, 12)} is equivalent to:
//   "wait for semaphore_A >= 5 AND semaphore_B >= 12"
//
// The frontier is a COMPACT REPRESENTATION of a multi-semaphore wait. It adds:
//   - Causal reasoning (compare, merge) that individual semaphores don't have.
//   - Wire-format efficiency (one frontier vs. N semaphore references).
//   - Transitive causality propagation (merge preserves all dependencies).
//
// ## Merge (join in the lattice)
//
// Merging two frontiers produces the LEAST UPPER BOUND: the earliest point
// that is guaranteed to be after both inputs. This is component-wise max:
//
//   merge({(A,3), (B,5)}, {(A,7), (C,2)}) = {(A,7), (B,5), (C,2)}
//
// Merge is used when an operation depends on two independent predecessors:
//
//   GPU produces buffer    → frontier {(GPU/q0, 47)}
//   Network receives data  → frontier {(Net, 12)}
//   Operation needs both   → merge = {(GPU/q0, 47), (Net, 12)}
//
// Merge is associative and commutative (it's a lattice join).
//
// ## Wire format usage
//
// Frontiers are the core of the remoting protocol's consistency guarantees.
// Every command sent across the network carries a frontier:
//
//   "Execute this dispatch, but only after frontier F is satisfied."
//
// The receiver maps the frontier's axes to local semaphores (via the axis
// table) and waits for all of them. This is how GPU→GPU synchronization works
// across machines without a centralized lock server.
//
// ## Storage and allocation
//
// Frontiers are variable-length value types (VLA). Callers provide storage:
//   - Stack: for small frontiers (≤4 entries is the common case).
//   - Slab: via iree_async_frontier_size() for overflow-checked sizing.
//   - Inline: embedded in protocol messages with trailing entry data.
//
// Entries MUST be sorted by axis (ascending) for O(n) comparison and merge.
// The entry_count is uint8_t: maximum 255 entries per frontier (sufficient
// for rack-scale systems with hundreds of queues).
typedef struct iree_async_frontier_t {
  uint8_t entry_count;
  uint8_t reserved[7];
  iree_async_frontier_entry_t entries[];
} iree_async_frontier_t;

// Computes the total byte size needed to store a frontier with |entry_count|
// entries using overflow-checked arithmetic. Use this for slab or heap
// allocation:
//
//   iree_host_size_t size = 0;
//   IREE_RETURN_IF_ERROR(iree_async_frontier_size(3, &size));
//   iree_async_frontier_t* f = NULL;
//   IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator, size, (void**)&f));
//   iree_async_frontier_initialize(f, 3);
//   f->entries[0] = (iree_async_frontier_entry_t){gpu_axis, 5};
//   f->entries[1] = (iree_async_frontier_entry_t){net_axis, 12};
//   f->entries[2] = (iree_async_frontier_entry_t){host_axis, 8};
static inline iree_status_t iree_async_frontier_size(
    iree_host_size_t entry_count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_frontier_t), out_size,
      IREE_STRUCT_FIELD_FAM(entry_count, iree_async_frontier_entry_t));
}

// Initializes a pre-allocated frontier with the given entry count.
// After this call, the caller must fill entries[0..entry_count-1] with
// (axis, epoch) pairs IN SORTED ORDER BY AXIS (ascending).
//
// WARNING: The sorted-entries invariant is critical for correctness.
// iree_async_frontier_compare() and iree_async_frontier_merge() use a
// merge-scan algorithm that relies on sorted order. Unsorted entries cause
// SILENT INCORRECT CAUSAL ORDERING — operations may execute before their
// dependencies complete, leading to data corruption. There is no runtime
// check on the hot path (compare/merge are O(n) and must stay cheap).
// Use iree_async_frontier_validate() in debug/test code to catch violations.
static inline void iree_async_frontier_initialize(
    iree_async_frontier_t* frontier, uint8_t entry_count) {
  frontier->entry_count = entry_count;
  memset(frontier->reserved, 0, sizeof(frontier->reserved));
}

// Validates that a frontier's entries satisfy the sorted-by-axis invariant.
// Returns iree_ok_status() if valid, or an error describing the violation.
// This is intended for debug assertions and test validation — NOT for hot
// paths. Call after constructing a frontier to catch programming errors early.
//
// Checks:
//   - Entries are in strictly ascending axis order (no duplicates).
//   - No zero-epoch entries (epoch 0 is meaningless — the axis hasn't
//     advanced, so it should not be in the frontier at all).
//
// Usage:
//   iree_async_frontier_initialize(f, 3);
//   f->entries[0] = ...;
//   f->entries[1] = ...;
//   f->entries[2] = ...;
//   IREE_ASSERT_OK(iree_async_frontier_validate(f));  // Debug only.
iree_status_t iree_async_frontier_validate(
    const iree_async_frontier_t* frontier);

//===----------------------------------------------------------------------===//
// Frontier comparison
//===----------------------------------------------------------------------===//

// Result of comparing two frontiers. Defines the causal relationship between
// the events they represent.
//
// The comparison follows vector clock semantics:
//   - BEFORE: a is dominated by b (a happens-before b).
//   - AFTER: b is dominated by a (b happens-before a).
//   - CONCURRENT: neither dominates (no causal relationship).
//   - EQUAL: identical causal position.
//
// Axes present in one frontier but not the other are treated as epoch 0 in
// the missing frontier. This means a frontier that mentions an axis is always
// AFTER a frontier that doesn't mention it (assuming epoch > 0).
enum iree_async_frontier_comparison_e {
  IREE_ASYNC_FRONTIER_EQUAL = 0,
  IREE_ASYNC_FRONTIER_BEFORE = 1,
  IREE_ASYNC_FRONTIER_AFTER = 2,
  IREE_ASYNC_FRONTIER_CONCURRENT = 3,
};
typedef uint8_t iree_async_frontier_comparison_t;

// Compares two frontiers for causal ordering. Both must have entries sorted by
// axis. Returns the relationship of |a| relative to |b|:
//   BEFORE means "a happens-before b"
//   AFTER means "b happens-before a"
//
// Runs in O(n + m) time where n and m are the entry counts (merge-scan of two
// sorted arrays).
//
// Example:
//   a = {(axis_X, 3), (axis_Y, 5)}
//   b = {(axis_X, 3), (axis_Y, 7)}
//   compare(a, b) → BEFORE (Y advanced from 5 to 7, X stayed)
iree_async_frontier_comparison_t iree_async_frontier_compare(
    const iree_async_frontier_t* a, const iree_async_frontier_t* b);

// Merges |source| into |target| by taking the component-wise maximum epoch
// for each axis. Axes in |source| not present in |target| are added. Both
// must have entries sorted by axis.
//
// The result is the LEAST UPPER BOUND (join) in the causal lattice: the
// earliest frontier that is causally after both |target| and |source|.
//
// |target_capacity| is the maximum number of entries |target|'s allocation can
// hold. If the merged result would exceed capacity, the merge fails with
// IREE_STATUS_RESOURCE_EXHAUSTED and |target| is left unchanged. Callers must
// size capacity to accommodate worst-case merge (sum of unique axes across all
// sources). 255 entries (uint8_t max) is sufficient for rack-scale systems.
//
// Performance: three paths in order of expected frequency:
//   1. Source empty: O(1) no-op.
//   2. Identical axis sets: O(n) epoch-max in place, zero entry movement.
//   3. Different axis sets: O(n+m) in-place right-to-left merge, no scratch.
//
// Returns iree_ok_status() on success (|target| modified, entry_count updated).
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if merged count exceeds capacity.
//
// Example:
//   target = {(A, 3), (B, 5)}       capacity = 4
//   source = {(A, 7), (C, 2)}
//   after merge: target = {(A, 7), (B, 5), (C, 2)}   entry_count = 3
iree_status_t iree_async_frontier_merge(iree_async_frontier_t* target,
                                        uint8_t target_capacity,
                                        const iree_async_frontier_t* source);

// Returns true if every entry in |frontier| is satisfied by the corresponding
// epoch in |current_epochs|. A frontier is satisfied when all of its axes have
// reached at least their target epoch values.
//
// |current_epochs| is a sorted array of (axis, epoch) pairs representing the
// current progress of each axis in the system. Axes in |frontier| not found in
// |current_epochs| are treated as epoch 0 (unsatisfied unless target is 0).
//
// This is the core predicate for "can this operation execute now?":
//   if (iree_async_frontier_is_satisfied(cmd->prerequisite, tracker_epochs, n))
//     execute(cmd);
bool iree_async_frontier_is_satisfied(
    const iree_async_frontier_t* frontier,
    const iree_async_frontier_entry_t* current_epochs,
    iree_host_size_t current_epochs_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_FRONTIER_H_
