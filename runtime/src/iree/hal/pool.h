// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_POOL_H_
#define IREE_HAL_POOL_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_notification_t iree_async_notification_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// Result of a pool reservation acquisition operation. This is a result enum,
// NOT an iree_status_t, because acquire_reservation() is a hot-path operation
// that regularly returns EXHAUSTED or OVER_BUDGET during normal operation
// (pool growth, steady-state pipelining where releases lag reservations by one
// epoch).
// Using iree_make_status() for these cases would capture backtraces on every
// transient exhaustion; expensive and misleading.
//
// The iree_status_t return from acquire_reservation() is reserved for
// infrastructure failures: invalid arguments, internal corruption, slab
// provider errors. Those are exceptional and warrant backtraces.
typedef uint32_t iree_hal_pool_acquire_result_t;
enum iree_hal_pool_acquire_result_e {
  // Block reserved successfully. The death frontier from the recycled block
  // was dominated by the requester's frontier; zero-sync reuse. The memory
  // is safe for immediate use without any device synchronization.
  IREE_HAL_POOL_ACQUIRE_OK = 0,

  // Block reserved from previously unused offset space (first use of this
  // region, or the block was freed with a NULL frontier). No death frontier
  // to check. Equivalent to OK for callers; no synchronization needed.
  IREE_HAL_POOL_ACQUIRE_OK_FRESH = 1,

  // Block reserved, but the death frontier was NOT dominated by the
  // requester's frontier. The block IS reserved (offset assigned), but the
  // queue scheduler must add a hidden wait on the death frontier's axes before
  // the reservation's bytes are used. The block's death frontier is returned
  // via |out_info->wait_frontier| from
  // iree_hal_pool_acquire_reservation(), and generic
  // metadata about that dependency is returned via |out_info->flags|.
  //
  // This occurs when try-before-fence fails: prior work on this block may
  // still be executing, and the requester's frontier does not transitively
  // cover it. Queue implementations decide how to represent that hidden wait
  // in their own scheduler state; user-facing queue_alloca APIs must not
  // surface this as a transient caller-visible branch.
  IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT = 2,

  // No blocks available. All blocks are allocated, or all free blocks have
  // non-dominated death frontiers and the pool chose not to hand one out
  // as NEEDS_WAIT.
  //
  // The caller should observe the pool's notification, retry the reservation,
  // and wait on the observed token only if the pool remains exhausted. No
  // reservation was made; |out_reservation| is not modified.
  IREE_HAL_POOL_ACQUIRE_EXHAUSTED = 3,

  // The pool's budget limit would be exceeded by this reservation. The
  // request is valid and blocks may be physically available, but the budget
  // policy prevents the allocation.
  //
  // The caller should free other reservations from this pool, adjust the
  // budget, or use a different pool. No reservation was made;
  // |out_reservation| is not modified.
  IREE_HAL_POOL_ACQUIRE_OVER_BUDGET = 4,
};

// A reservation from a pool. Returned by
// iree_hal_pool_acquire_reservation() and passed to
// iree_hal_pool_release_reservation().
//
// This is a pure value type (32 bytes, no ownership). It lives on the stack
// during queue submission or is stored in the buffer that wraps it. The pool
// knows how to interpret its fields; callers treat it as opaque.
typedef struct iree_hal_pool_reservation_t {
  // Offset within the pool's managed range.
  iree_device_size_t offset;

  // Actual allocated length in bytes. May exceed the requested length due to
  // alignment rounding or because the block could not be split (the remainder
  // would have been smaller than the minimum block size).
  iree_device_size_t length;

  // Pool-internal opaque handle for returning the block on release.
  // Interpretation is strategy-specific: a TLSF release-node pointer,
  // fixed-block block index, pass-through reservation-state pointer, etc.
  // 64-bit to accommodate pointer-sized handles on all platforms.
  uint64_t block_handle;

  // Which slab within the pool (for multi-slab pools in slab mode).
  // 0 for single-slab or VMM pools.
  uint16_t slab_index;

  // Reserved for future expansion. Must be zero.
  uint16_t reserved[3];
} iree_hal_pool_reservation_t;

// Flags controlling pool reservation acquisition.
typedef uint32_t iree_hal_pool_reserve_flags_t;
enum iree_hal_pool_reserve_flag_bits_e {
  IREE_HAL_POOL_RESERVE_FLAG_NONE = 0u,

  // Allows the pool to return IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT when a
  // recycled block is available but its death frontier is not dominated by the
  // requester frontier. Callers setting this flag must either insert an
  // internal dependency on out_info->wait_frontier before the bytes are used or
  // release the reservation with out_info->wait_frontier to preserve the
  // block's dependency metadata.
  //
  // Callers that cannot model queue-owned hidden memory dependencies must omit
  // this flag. Such calls should receive only immediately-usable reservations
  // or transient EXHAUSTED/OVER_BUDGET results from well-behaved pools.
  IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER = 1u << 0,
};

// Generic metadata flags returned by a pool reservation acquisition.
typedef uint32_t iree_hal_pool_acquire_flags_t;
enum iree_hal_pool_acquire_flag_bits_e {
  IREE_HAL_POOL_ACQUIRE_FLAG_NONE = 0u,

  // The returned wait frontier is tainted: at least one writer freed the block
  // with no death frontier or an invalid frontier, so zero-sync reuse was
  // intentionally disabled for safety. Queue implementations should treat this
  // as a conservative dependency edge, not proof of precise happens-before.
  IREE_HAL_POOL_ACQUIRE_FLAG_WAIT_FRONTIER_TAINTED = 1u << 0,
};

// Generic metadata returned by a pool reservation acquisition.
//
// |wait_frontier| is a borrowed pointer to the selected block's death frontier
// when |out_result| is IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT. The pointer remains
// valid until the matching reservation is released. It is NULL for
// IREE_HAL_POOL_ACQUIRE_OK and IREE_HAL_POOL_ACQUIRE_OK_FRESH.
//
// If a caller declines an IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT reservation and
// immediately releases it, passing |wait_frontier| back to
// iree_hal_pool_release_reservation() must preserve the block's dependency
// metadata. Concrete pools must therefore tolerate |death_frontier| aliasing
// the reservation's own pool-owned frontier storage in that path.
typedef struct iree_hal_pool_acquire_info_t {
  // Borrowed dependency frontier for IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT.
  // NULL for success paths that require no wait.
  const iree_async_frontier_t* wait_frontier;

  // Generic metadata bits describing the selected reservation.
  iree_hal_pool_acquire_flags_t flags;

  uint32_t reserved;
} iree_hal_pool_acquire_info_t;

// Controls how a concrete buffer object/view is materialized for a reservation.
typedef uint32_t iree_hal_pool_materialize_flags_t;
enum iree_hal_pool_materialize_flag_bits_e {
  IREE_HAL_POOL_MATERIALIZE_FLAG_NONE = 0u,

  // Transfers reservation ownership to the returned buffer. When that buffer
  // is destroyed its release callback must return |reservation| to |pool|
  // with a NULL death frontier.
  //
  // Without this flag, the returned buffer is only a borrowed view of the
  // reserved bytes and the caller remains responsible for calling
  // iree_hal_pool_release_reservation() exactly once.
  IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP = 1u << 0,
};

// Describes the memory capabilities of a pool. Computed at pool creation time
// from the slab provider's properties and the pool's strategy constraints.
// Used by iree_hal_pool_set_t for routing allocation requests to compatible
// pools.
typedef struct iree_hal_pool_capabilities_t {
  // Memory type properties provided by this pool's slab provider. Checked
  // against the required bits in iree_hal_buffer_params_t.type.
  iree_hal_memory_type_t memory_type;

  // Buffer usages this pool supports. A pool backed by DEVICE_LOCAL memory
  // that isn't host-visible can't serve MAPPING usage.
  iree_hal_buffer_usage_t supported_usage;

  // Minimum allocation size in bytes. Fixed-block pools use their block size.
  // Suballocating pools may round internally and report 0 or 1 when they have
  // no practical lower bound.
  iree_device_size_t min_allocation_size;

  // Strategy-specific maximum single reservation in bytes. Fixed-block pools
  // use their block size, TLSF pools use their slab size, and pass-through
  // pools report 0 for no strategy limit. Budgets are reported separately and
  // enforced by acquire_reservation().
  iree_device_size_t max_allocation_size;
} iree_hal_pool_capabilities_t;

// Running statistics for a pool. All values are atomic snapshots; they may
// be momentarily inconsistent under concurrent modifications. Querying stats
// is O(1) with no internal walks or locks.
typedef struct iree_hal_pool_stats_t {
  // Total bytes currently occupied by live reservations.
  iree_device_size_t bytes_reserved;

  // Total bytes in free blocks or available for reservation.
  iree_device_size_t bytes_free;

  // Total physical memory committed (slabs or VMM pages).
  iree_device_size_t bytes_committed;

  // Budget limit in bytes (0 = unlimited).
  iree_device_size_t budget_limit;

  // Number of currently live reservations.
  uint32_t reservation_count;

  // Number of slabs from the slab provider.
  uint32_t slab_count;

  // Total successful acquire_reservation() calls.
  uint64_t reserve_count;

  // Total release_reservation() calls.
  uint64_t release_count;

  // Reserves that hit frontier-dominated reuse.
  uint64_t reuse_count;

  // Reserves where the dominance check failed.
  uint64_t reuse_miss_count;

  // Reserves from fresh, never-used offset space.
  uint64_t fresh_count;

  // Reserves that returned EXHAUSTED.
  uint64_t exhausted_count;

  // Reserves that returned OVER_BUDGET.
  uint64_t over_budget_count;

  // Reserves that returned NEEDS_WAIT.
  uint64_t wait_count;
} iree_hal_pool_stats_t;

// Callback for try-before-fence epoch queries. When a death-frontier dominance
// check fails in acquire_reservation(), the pool calls this to check whether
// the timeline has actually advanced past the death frontier's epoch on a
// specific axis, even though the requester's frontier hasn't imported the
// update yet.
//
// Returns true if |axis| has reached at least |epoch| (the work has
// completed). This is a host-side read of the semaphore's current value;
// no device interaction, no blocking. The pool uses this to avoid
// unnecessarily skipping reusable blocks when completion notifications are
// batched by the proactor.
//
// Set at pool creation time. Pools created with |epoch_query.fn| == NULL skip
// the try-before-fence optimization; non-dominated blocks are treated as
// genuinely unavailable for zero-sync reuse.
typedef bool(IREE_API_PTR* iree_hal_pool_epoch_query_fn_t)(
    void* user_data, iree_async_axis_t axis, uint64_t epoch);

// Bound epoch query callback and user data.
typedef struct iree_hal_pool_epoch_query_t {
  // Callback used to query whether an axis has reached an epoch.
  iree_hal_pool_epoch_query_fn_t fn;

  // User data passed to |fn|.
  void* user_data;
} iree_hal_pool_epoch_query_t;

// Returns a null epoch query callback.
static inline iree_hal_pool_epoch_query_t iree_hal_pool_epoch_query_null(void) {
  iree_hal_pool_epoch_query_t query = {
      .fn = NULL,
      .user_data = NULL,
  };
  return query;
}

//===----------------------------------------------------------------------===//
// iree_hal_pool_t
//===----------------------------------------------------------------------===//

// A memory pool that manages a region of offset space backed by one or more
// slabs of physical memory. Pools are the primary allocation interface in IREE:
// both synchronous (allocate_buffer) and asynchronous (queue_alloca/dealloca)
// allocation paths go through pools.
//
// Pools are ref-counted HAL resources, but allocations/wrapped buffers borrow
// their source pool instead of retaining it. Pool owners must ensure a pool
// outlives every reservation and buffer allocated from it. That keeps
// queue_alloca/dealloca hot paths free of per-allocation pool refcount traffic
// and matches the intended "application-scoped allocation policy" lifetime
// model.
//
// Pools can be shared across queues and threads (concurrency is handled
// internally per pool type).
//
// The pool base type is abstract; concrete pools are created by type-specific
// factory functions (iree_hal_tlsf_pool_create,
// iree_hal_fixed_block_pool_create, etc.) and used through this common
// interface.
//
// ## Allocation protocol
//
// Asynchronous (queue-ordered) allocation:
//
//   submit queue_alloca             use bytes             submit queue_dealloca
// ┌─────────────────────┐      ┌────────────────┐      ┌──────────────────────┐
// │ acquire_reservation │─────▶│ borrowed view  │─────▶│ release_reservation  │
// │ checks death edge   │      │ no pool retain │      │ records death edge   │
// └─────────────────────┘      └────────────────┘      └──────────────────────┘
//
//   1. acquire_reservation() at submit time: finds a free block, checks death
//      frontier dominance, returns a reservation with offset and length.
//   2. materialize_reservation() without ownership transfer: creates a
//      backing buffer view whose lifetime is independent from the reservation.
//   3. release_reservation() at dealloca submit time: returns the block to
//      the pool's free list, tagged with a death frontier.
//
// Synchronous allocation:
//   iree_hal_pool_allocate_buffer(): calls acquire_reservation() +
//   materialize_reservation(TRANSFER_RESERVATION_OWNERSHIP) in a loop, waiting
//   on the pool's notification if exhausted. This is a shared utility, not a
//   vtable method.
//
// ## Death frontier integration
//
// Free blocks carry death frontiers: causal snapshots from when they were
// last freed. Reserve checks whether the requester's frontier dominates the
// death frontier for zero-sync reuse. This enables buffer recycling without
// any device synchronization in steady-state pipelined workloads.
typedef struct iree_hal_pool_t iree_hal_pool_t;

// Retains the given |pool| for the caller.
IREE_API_EXPORT void iree_hal_pool_retain(iree_hal_pool_t* pool);

// Releases the given |pool| from the caller.
IREE_API_EXPORT void iree_hal_pool_release(iree_hal_pool_t* pool);

// Acquires a reservation from the pool for a future allocation.
//
// |size| is the minimum number of bytes needed. |alignment| is the required
// alignment for the returned offset (must be a power of two and > 0). The
// actual allocated length may exceed |size| due to alignment rounding or
// block splitting constraints.
//
// |requester_frontier| is the queue scheduler's current causal position, used
// for death frontier dominance checking. Pass NULL to skip dominance checking
// (appropriate for synchronous allocations that don't participate in
// queue-ordered frontier tracking).
//
// |flags| controls whether the caller can accept queue-owned dependency work
// as part of the reservation. In particular, callers must set
// IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER before a pool may return
// IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT.
//
// On success (iree_ok_status()), |out_result| indicates the specific outcome:
//   OK / OK_FRESH: reservation succeeded, memory is safe for immediate use.
//   OK_NEEDS_WAIT: reservation succeeded, but the queue scheduler must add a
//     hidden wait on |out_info->wait_frontier| before the reservation's bytes
//     are used.
//   EXHAUSTED: no reservation made, pool has no suitable blocks.
//   OVER_BUDGET: no reservation made, budget limit would be exceeded.
//
// |out_info| is always initialized on success. For EXHAUSTED/OVER_BUDGET it is
// set to an empty record. For OK_NEEDS_WAIT, |out_info->wait_frontier| is
// borrowed pool storage owned by the reservation and remains valid until the
// reservation is released.
//
// Returns an error status (with backtrace) only for infrastructure failures:
// invalid arguments (size 0, non-power-of-two alignment), internal
// corruption, or slab provider errors. These are exceptional.
IREE_API_EXPORT iree_status_t iree_hal_pool_acquire_reservation(
    iree_hal_pool_t* pool, iree_device_size_t size,
    iree_device_size_t alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result);

// Releases a reservation back to the pool's free list.
//
// |reservation| is the reservation returned by a prior
// iree_hal_pool_acquire_reservation() call on this pool. |death_frontier| is
// the causal snapshot to attach to the freed block; typically the queue's
// frontier at dealloca submit time. Pass NULL for an empty frontier (the block
// is immediately available for zero-sync reuse by any requester).
//
// The reservation's offset is returned to the pool's free list immediately.
// The memory is available for future acquire_reservation() calls from that
// point forward, even though the GPU may still be executing prior work;
// death frontier dominance checking gates actual reuse safety.
//
// Publishes the pool's notification epoch. Releases that occur with no known
// waiter may skip platform wake work; waiters use an observe-check-wait
// protocol so this cannot lose wakeups.
IREE_API_EXPORT void iree_hal_pool_release_reservation(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier);

// Materializes a concrete buffer object/view for a reservation from this pool.
//
// |params| describes the buffer's usage, access, and memory type properties.
// |reservation| is the reservation returned by
// iree_hal_pool_acquire_reservation().
//
// If |flags| includes
// IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, the returned
// buffer stores the reservation and a borrowed pointer to |pool| and releases
// that reservation with a NULL death frontier when destroyed.
//
// Otherwise the returned buffer is only a borrowed view and the caller keeps
// ownership of |reservation|. This is the queue-allocation path's backing
// materialization primitive: a transient wrapper can commit/decommit the
// borrowed view while releasing the reservation independently at a
// queue-ordered dealloca point.
//
// |pool| must outlive both the reservation and the returned buffer.
//
// The concrete pool owns reservation bookkeeping and release callbacks, but
// provider-specific buffer materialization must flow through that pool's slab
// provider. Generic pools must not dereference slab payload fields directly;
// they pass a slab plus offset range to iree_hal_slab_provider_wrap_buffer().
IREE_API_EXPORT iree_status_t iree_hal_pool_materialize_reservation(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer);

// Queries the memory capabilities of the pool. O(1); reads cached fields
// computed at pool creation time.
IREE_API_EXPORT void iree_hal_pool_query_capabilities(
    const iree_hal_pool_t* pool,
    iree_hal_pool_capabilities_t* out_capabilities);

// Queries the pool's running statistics. O(1); atomic snapshots of
// incrementally maintained counters. Values may be momentarily inconsistent
// under concurrent modifications.
IREE_API_EXPORT void iree_hal_pool_query_stats(
    const iree_hal_pool_t* pool, iree_hal_pool_stats_t* out_stats);

// Releases unused physical memory back to the slab provider.
// VMM mode: decommit pages with no live reservations.
// Slab mode: free slabs with no live reservations.
// The pool remains valid after trimming; it can grow again on demand.
IREE_API_EXPORT iree_status_t iree_hal_pool_trim(iree_hal_pool_t* pool);

// Returns the pool's notification. Callers waiting for blocks to become
// available can use this to sleep efficiently instead of polling.
//
// The notification is advisory over pool state. Callers must observe the
// notification epoch before checking the pool state, and then wait on that
// token only if the checked state still requires a wakeup. Releases may skip
// platform wake work when no wait observer exists.
IREE_API_EXPORT iree_async_notification_t* iree_hal_pool_notification(
    iree_hal_pool_t* pool);

// Allocates a buffer from the pool synchronously.
//
// This is a shared utility (NOT a vtable method) that calls
// acquire_reservation() + materialize_reservation() in a loop. If
// acquire_reservation() returns EXHAUSTED or OVER_BUDGET, the function waits
// on the pool's notification for |timeout| and retries.
//
// |requester_frontier| is passed to acquire_reservation() for dominance
// checking. Pass NULL to skip dominance checking (appropriate for persistent
// buffers that aren't queue-ordered).
//
// This helper is synchronous-only. Queue implementations must not call it for
// queue_alloca, because queue-owned memory-frontier waits and pool-notification
// retries are scheduler state, not host-thread blocking in this helper.
//
// |timeout| controls how long to wait for a free block. Converted to an
// absolute deadline internally so retries after spurious wakes use a
// consistent cutoff:
//   iree_make_timeout_ms(0): try once, fail immediately if exhausted.
//   iree_infinite_timeout(): block until a block becomes available.
//   iree_make_timeout_ms(N): wait up to N milliseconds.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |timeout| is reached before an
// immediately-usable reservation can be acquired. This uses iree_make_status()
// because it represents a terminal failure visible to the application, not a
// transient hot-path condition.
IREE_API_EXPORT iree_status_t iree_hal_pool_allocate_buffer(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    const iree_async_frontier_t* requester_frontier, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_pool_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_pool_vtable_t {
  // Destroys a concrete pool implementation.
  void(IREE_API_PTR* destroy)(iree_hal_pool_t* pool);

  // Acquires reservation metadata from the concrete pool implementation.
  iree_status_t(IREE_API_PTR* acquire_reservation)(
      iree_hal_pool_t* pool, iree_device_size_t size,
      iree_device_size_t alignment,
      const iree_async_frontier_t* requester_frontier,
      iree_hal_pool_reserve_flags_t flags,
      iree_hal_pool_reservation_t* out_reservation,
      iree_hal_pool_acquire_info_t* out_info,
      iree_hal_pool_acquire_result_t* out_result);

  // Releases reservation metadata back to the concrete pool implementation.
  void(IREE_API_PTR* release_reservation)(
      iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
      const iree_async_frontier_t* death_frontier);

  // Materializes a concrete buffer object or view for a reservation.
  iree_status_t(IREE_API_PTR* materialize_reservation)(
      iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
      const iree_hal_pool_reservation_t* reservation,
      iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer);

  // Queries cached memory capabilities for routing.
  void(IREE_API_PTR* query_capabilities)(
      const iree_hal_pool_t* pool,
      iree_hal_pool_capabilities_t* out_capabilities);

  // Queries running pool statistics.
  void(IREE_API_PTR* query_stats)(const iree_hal_pool_t* pool,
                                  iree_hal_pool_stats_t* out_stats);

  // Releases unused physical memory back to the concrete provider.
  iree_status_t(IREE_API_PTR* trim)(iree_hal_pool_t* pool);

  // Returns the notification used for pool availability changes.
  iree_async_notification_t*(IREE_API_PTR* notification)(iree_hal_pool_t* pool);
} iree_hal_pool_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_pool_vtable_t);

IREE_API_EXPORT void iree_hal_pool_destroy(iree_hal_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_POOL_H_
