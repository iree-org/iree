// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Epoch-driven notification ring for mapping GPU submission completions to
// async semaphore signals. Each queue has one notification ring; the ring
// maps monotonic submission epochs to pending semaphore signals that the
// host queue drains when the GPU advances the epoch.
//
// The epoch signal is a single hsa_signal_t initialized to a large value
// and decremented by 1 on each submission's last AQL packet completion.
// The current epoch (count of completed submissions) is:
//   INITIAL_VALUE - hsa_signal_load(epoch_signal)
//
// The ring uses a hot/cold split for cache-friendly drain:
//   - Hot entries (32 bytes each): semaphore, value, epoch, and reserved
//     padding. Stored in a power-of-two ring buffer, dense and L1-resident for
//     the coalescing scan.
//   - Cold frontier snapshots (variable-size): written to a byte ring only
//     at semaphore transition points that still have an undrained span. The
//     drain reads snapshots only when flushing a span that actually has a
//     transition snapshot, avoiding per-entry frontier overhead.
//
// The drain coalesces consecutive same-semaphore entries into a single
// signal call using a single-slot accumulator. For N dispatches on the same
// stream semaphore, this produces 1 signal instead of N.

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_NOTIFICATION_RING_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_NOTIFICATION_RING_H_

#include "iree/async/frontier.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_resource_set_t iree_hal_resource_set_t;

// Initial value for the epoch signal. The CP decrements by 1 on each
// submission's last packet completion. The epoch (number of completed
// submissions) is: INITIAL_VALUE - hsa_signal_load(epoch_signal).
// INT64_MAX/2 gives ~4.6e18 decrements before overflow (~146 years at 1
// billion submissions/second).
#define IREE_HAL_AMDGPU_EPOCH_INITIAL_VALUE (INT64_MAX / 2)

// Default notification ring capacity.
#define IREE_HAL_AMDGPU_DEFAULT_NOTIFICATION_CAPACITY 1024

// Sentinel value in frontier_snapshot_t::entry_count indicating the reader
// should wrap to byte 0 of the frontier ring. Written when a snapshot doesn't
// fit in the remaining buffer space.
#define IREE_HAL_AMDGPU_FRONTIER_SNAPSHOT_SENTINEL 0xFF

// Maximum number of frontier entries in one snapshot.
//
// This must be >= the queue frontier capacity used by host_queue.c, because
// transition snapshots serialize the queue's accumulated frontier verbatim.
#define IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT 64

// Maximum size of a single frontier snapshot in bytes.
#define IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_SIZE     \
  (sizeof(iree_hal_amdgpu_frontier_snapshot_t) +       \
   IREE_HAL_AMDGPU_MAX_FRONTIER_SNAPSHOT_ENTRY_COUNT * \
       sizeof(iree_async_frontier_entry_t))

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_notification_entry_t (hot, 32 bytes)
//===----------------------------------------------------------------------===//

typedef uint32_t iree_hal_amdgpu_notification_entry_flags_t;
enum iree_hal_amdgpu_notification_entry_flag_bits_t {
  IREE_HAL_AMDGPU_NOTIFICATION_ENTRY_FLAG_NONE = 0u,
  // This entry's same-semaphore span does not own a cold frontier snapshot.
  // Drain must signal the semaphore with |fallback_frontier| instead of
  // consuming from the cold frontier ring.
  IREE_HAL_AMDGPU_NOTIFICATION_ENTRY_FLAG_OMIT_FRONTIER_SNAPSHOT = 1u << 0,
};

// A pending semaphore signal associated with a submission epoch. Contains
// only the data needed for the drain coalescing scan — frontier data is
// stored separately in the frontier snapshot ring.
//
// Entries are stored in a power-of-two ring buffer (32 bytes each, two per
// 64-byte cache line).
typedef struct iree_hal_amdgpu_notification_entry_t {
  // Semaphore to signal when the epoch is reached. Not retained — the caller
  // ensures the semaphore outlives the notification (queue teardown waits for
  // all in-flight work and drains before destroying semaphores).
  iree_async_semaphore_t* semaphore;
  // Timeline value to signal the semaphore to.
  uint64_t timeline_value;
  // One-based submission epoch on this queue. When the queue's current epoch
  // reaches this value (current_epoch >= submission_epoch), this entry is ready
  // to drain.
  uint64_t submission_epoch;
  // Flags controlling how this entry is drained.
  iree_hal_amdgpu_notification_entry_flags_t flags;
  // Reserved padding to keep hot entries at 32 bytes (2 per 64-byte cache
  // line).
  uint32_t reserved0;
} iree_hal_amdgpu_notification_entry_t;
static_assert(sizeof(iree_hal_amdgpu_notification_entry_t) == 32,
              "notification entries must remain 32 bytes");

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_frontier_snapshot_t (cold, variable-size)
//===----------------------------------------------------------------------===//

// Frontier snapshot written at each semaphore transition point. Records the
// queue's accumulated frontier at the end of a same-semaphore span. The
// drain reads one snapshot per coalesced flush.
//
// Variable-size: the header is followed by entry_count frontier entries.
// Total size: sizeof(header) + entry_count *
// sizeof(iree_async_frontier_entry_t).
typedef struct iree_hal_amdgpu_frontier_snapshot_t {
  // Epoch at the end of the same-semaphore span this snapshot covers.
  uint64_t epoch;
  // Number of frontier entries following this header. 0xFF is a sentinel
  // indicating the reader should wrap to byte 0 (not a real snapshot).
  uint8_t entry_count;
  uint8_t reserved[7];
  // Followed by entry_count x iree_async_frontier_entry_t.
} iree_hal_amdgpu_frontier_snapshot_t;

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_reclaim_entry_t (cold, epoch-indexed)
//===----------------------------------------------------------------------===//

// Number of resource pointers stored inline in each reclaim entry. Covers
// the common case of 1 signal semaphore + up to 7 operation resources
// (buffers, executables, command buffers) without any block pool allocation.
// Dispatches with more than 7 bindings spill to a block-pool-allocated array.
#define IREE_HAL_AMDGPU_RECLAIM_INLINE_CAPACITY 8

typedef struct iree_hal_amdgpu_reclaim_entry_t iree_hal_amdgpu_reclaim_entry_t;

// Infallible callback executed for one completed epoch before that epoch's
// user-visible semaphore signals are published.
//
// This is the pre-signal state-transition lane for operations like transient
// buffer commit/decommit. |status| is OK for normal GPU completion and a
// borrowed queue/device failure status when the queue fails outstanding work.
// Any object referenced by |user_data| must also be retained in the reclaim
// entry's post-signal |resources| array if its lifetime must extend past
// callback execution.
typedef void(IREE_API_PTR* iree_hal_amdgpu_reclaim_action_fn_t)(
    iree_hal_amdgpu_reclaim_entry_t* entry, void* user_data,
    iree_status_t status);

typedef struct iree_hal_amdgpu_reclaim_action_t {
  iree_hal_amdgpu_reclaim_action_fn_t fn;
  void* user_data;
} iree_hal_amdgpu_reclaim_action_t;

// Per-epoch resource reclaim entry. Stores retained HAL resource pointers
// that are released when the epoch completes (drain time). One entry per
// advance_epoch call, indexed by epoch & (capacity - 1).
//
// Resources include signal semaphores (the notification entry stores
// unretained semaphore pointers — the reclaim entry keeps them alive)
// and operation-specific resources (buffers, executables, command buffers).
struct iree_hal_amdgpu_reclaim_entry_t {
  // Pointer to the retained-resource pointer array. Points to inline_resources
  // when count <= INLINE_CAPACITY, otherwise to a block-pool-allocated array.
  iree_hal_resource_t** resources;
  // Optional resource set released with this entry after user signals publish.
  iree_hal_resource_set_t* resource_set;
  // One bounded pre-signal action for this epoch. Executed before any
  // user-visible signal publication for the epoch when drain observes normal
  // completion, and during fail_all with the failure status before resources
  // are released.
  iree_hal_amdgpu_reclaim_action_t pre_signal_action;
  // Kernarg ring write position at the time of this submission. Drain/fail_all
  // report the highest position across retired epochs so the caller can reclaim
  // kernarg blocks. 0 means no kernarg was allocated.
  uint64_t kernarg_write_position;
  uint16_t count;
  uint16_t reserved[3];
  iree_hal_resource_t*
      inline_resources[IREE_HAL_AMDGPU_RECLAIM_INLINE_CAPACITY];
};

// Prepares a reclaim entry for |count| resources. If count fits inline,
// sets |*out_resources| to the entry's inline storage. Otherwise acquires
// a block from |block_pool| and sets |*out_resources| to point into it.
// The caller fills the array with retained resource pointers, sets
// entry->kernarg_write_position, and sets entry->count before advancing the
// submission epoch.
iree_status_t iree_hal_amdgpu_reclaim_entry_prepare(
    iree_hal_amdgpu_reclaim_entry_t* entry, iree_arena_block_pool_t* block_pool,
    uint16_t count, iree_hal_resource_t*** out_resources);

// Releases all resources in the entry and returns any overflow block to
// the pool. Zeros entry->count.
void iree_hal_amdgpu_reclaim_entry_release(
    iree_hal_amdgpu_reclaim_entry_t* entry,
    iree_arena_block_pool_t* block_pool);

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_notification_ring_t
//===----------------------------------------------------------------------===//

// Epoch-driven notification ring with hot/cold split storage.
//
// The hot entry ring stores 32-byte entries for the drain coalescing scan.
// The cold frontier ring stores variable-size frontier snapshots written only
// at semaphore transition points — one snapshot per same-semaphore span.
// Both arrays are allocated in a single contiguous block.
typedef struct iree_hal_amdgpu_notification_ring_t {
  // HSA API handle for signal operations. Not retained.
  const iree_hal_amdgpu_libhsa_t* libhsa;
  iree_allocator_t host_allocator;

  // Monotonic completion counter.
  struct {
    // Per-queue hsa_signal_t created at init, destroyed at deinit. Set as
    // completion_signal on the last AQL packet of each submission; the CP
    // decrements it by 1 on completion.
    hsa_signal_t signal;
    // Next epoch to assign. Incremented by 1 per submission.
    uint64_t next_submission;
    // Last epoch observed by drain. The consumer stores with release after
    // releasing reclaim entries; the submission path acquires this in reserve()
    // to avoid reusing a still-live reclaim slot for a zero-signal epoch.
    iree_atomic_int64_t last_drained;
  } epoch;

  // Hot entry ring (32 bytes per entry, cache-friendly for drain scan).
  iree_hal_amdgpu_notification_entry_t* entries;
  // Producer index (submission path advances with a release store after
  // writing entries). The consumer acquires this before reading entries.
  iree_atomic_int64_t write;
  // Consumer index (drain/fail_all advances with a release store after
  // consuming entries). The producer acquires this before capacity checks.
  iree_atomic_int64_t read;
  // Power-of-two ring capacity. Indices are masked by (capacity - 1).
  uint32_t capacity;

  // Cold frontier snapshot byte ring (variable-size, sparse).
  // Written at semaphore transition points by the submission path via
  // push_frontier_snapshot. Read sequentially by drain when a completed span
  // reaches a different next semaphore; late snapshots for already-drained
  // spans are discarded before processing new completions.
  struct {
    uint8_t* data;
    // Power-of-two byte capacity. Monotonic byte positions are masked by
    // (capacity - 1) to derive in-buffer offsets.
    iree_host_size_t capacity;
    // Monotonic byte positions, not modulo offsets. Same SPSC release/acquire
    // contract as the hot entry ring indices.
    iree_atomic_int64_t write;
    iree_atomic_int64_t read;
  } frontier_ring;

  // Block pool for overflow reclaim allocations. Borrowed from the physical
  // device (NUMA-pinned); valid for the lifetime of the ring.
  iree_arena_block_pool_t* block_pool;

  // Per-epoch resource reclaim entries. Indexed by
  // epoch.next_submission & (capacity - 1) on the submission path, drained
  // in lockstep with the notification entries. Same capacity as the hot
  // entry ring (one reclaim entry per epoch, bounded by notification capacity).
  iree_hal_amdgpu_reclaim_entry_t* reclaim_entries;

  // Pointer to the base of the single allocation backing the entry array,
  // frontier ring buffer, and reclaim entries. Freed in deinitialize.
  void* storage;
} iree_hal_amdgpu_notification_ring_t;

// Initializes a notification ring. Creates the epoch signal and allocates
// the hot entry array, frontier snapshot byte ring, and reclaim entries in
// a single allocation.
//
// |block_pool| is used for overflow reclaim allocations (dispatches with
// more than IREE_HAL_AMDGPU_RECLAIM_INLINE_CAPACITY resources). Must
// outlive the ring.
//
// |capacity| must be a power of two.
iree_status_t iree_hal_amdgpu_notification_ring_initialize(
    const iree_hal_amdgpu_libhsa_t* libhsa, iree_arena_block_pool_t* block_pool,
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_hal_amdgpu_notification_ring_t* out_ring);

// Deinitializes the notification ring. Destroys the epoch signal and frees
// the backing storage.
//
// All in-flight work must have completed and been drained before calling.
void iree_hal_amdgpu_notification_ring_deinitialize(
    iree_hal_amdgpu_notification_ring_t* ring);

// Returns the epoch signal for use as completion_signal on AQL packets.
hsa_signal_t iree_hal_amdgpu_notification_ring_epoch_signal(
    const iree_hal_amdgpu_notification_ring_t* ring);

// Advances the submission epoch counter and returns the assigned one-based
// frontier epoch. Called by the submission path after all AQL packets for a
// submission have been written to the hardware queue.
//
// Epochs are one-based because the device-side wait formula in
// host_queue.c (compare_value = INITIAL_VALUE - target_epoch + 1) collapses
// to "signal < INITIAL_VALUE + 1" for target_epoch == 0, which is
// trivially true for any signal value. With one-based epochs, target == 0
// is reserved for "no submission has happened yet" and the formula only
// fires once at least one completion has been observed.
uint64_t iree_hal_amdgpu_notification_ring_advance_epoch(
    iree_hal_amdgpu_notification_ring_t* ring);

// Verifies that the ring has enough space for |entry_count| notification
// entries and up to |frontier_snapshot_count| max-size frontier snapshots.
//
// The snapshot reservation is conservative: it includes one extra max-size
// snapshot worth of bytes for a wrap sentinel/tail gap. Callers should check
// this before emitting AQL packets and calling push/push_frontier_snapshot so
// debug-only overflow asserts remain programmer-error-only.
iree_status_t iree_hal_amdgpu_notification_ring_reserve(
    const iree_hal_amdgpu_notification_ring_t* ring,
    iree_host_size_t entry_count, iree_host_size_t frontier_snapshot_count);

// Returns the reclaim entry for the next submission. Reclaim entries are
// indexed by the zero-based completion interval, so callers must fill this
// before calling advance_epoch.
static inline iree_hal_amdgpu_reclaim_entry_t*
iree_hal_amdgpu_notification_ring_reclaim_entry(
    iree_hal_amdgpu_notification_ring_t* ring) {
  return &ring->reclaim_entries[ring->epoch.next_submission &
                                (ring->capacity - 1)];
}

// Pushes a notification entry for a semaphore signal at the given epoch.
//
// The caller must ensure the ring has capacity by calling
// iree_hal_amdgpu_notification_ring_reserve() before publishing AQL packets and
// then pushing the corresponding notification entries.
//
// Frontier data is NOT stored per-entry. The caller must separately call
// push_frontier_snapshot at semaphore transition points.
void iree_hal_amdgpu_notification_ring_push(
    iree_hal_amdgpu_notification_ring_t* ring, uint64_t submission_epoch,
    iree_async_semaphore_t* semaphore, uint64_t timeline_value,
    iree_hal_amdgpu_notification_entry_flags_t flags);

// Pushes a frontier snapshot to the frontier byte ring. Called by the
// submission path when the signal semaphore changes between consecutive
// submissions (semaphore transition). |epoch| is the epoch of the last entry
// in the ending same-semaphore span. |frontier| is the queue's accumulated
// frontier at that point (before merging new dependencies). Must be a valid
// frontier (entry_count may be 0).
//
// The drain reads snapshots when flushing a completed span that has reached a
// different next semaphore. A final flush with no visible transition uses the
// fallback_frontier provided to drain, and late snapshots whose covered epoch
// has already drained are discarded on the next drain.
void iree_hal_amdgpu_notification_ring_push_frontier_snapshot(
    iree_hal_amdgpu_notification_ring_t* ring, uint64_t epoch,
    const iree_async_frontier_t* frontier);

// Drains all completed notification entries, coalescing consecutive
// same-semaphore entries into a single signal call (single-slot accumulator).
//
// |fallback_frontier| is used for the final coalesced flush when no frontier
// snapshot exists for the last same-semaphore span. It may be NULL if the
// caller has already merged frontier state into the semaphore at submission
// time and only needs completion-time timeline advancement/untainting.
//
// Stores the highest kernarg_write_position across all retired epochs in
// |out_kernarg_reclaim_position|. Set to 0 if no epochs were retired.
//
// Returns the number of entries drained.
iree_host_size_t iree_hal_amdgpu_notification_ring_drain(
    iree_hal_amdgpu_notification_ring_t* ring,
    const iree_async_frontier_t* fallback_frontier,
    uint64_t* out_kernarg_reclaim_position);

// Fails all pending notification entries with |error_status|.
// Each unique semaphore is failed exactly once; duplicate entries for the same
// semaphore skip the clone+fail (check-before-clone: status objects contain
// stack traces and are not free to clone).
//
// |error_status| is borrowed, not consumed — the caller retains ownership.
//
// Stores the highest kernarg_write_position across all failed entries in
// |out_kernarg_reclaim_position| (same semantics as drain).
//
// Returns the number of entries failed.
iree_host_size_t iree_hal_amdgpu_notification_ring_fail_all(
    iree_hal_amdgpu_notification_ring_t* ring, iree_status_t error_status,
    uint64_t* out_kernarg_reclaim_position);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_NOTIFICATION_RING_H_
