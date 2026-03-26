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

// Result of a pool reserve operation. This is a result enum, NOT an
// iree_status_t, because reserve() is a hot-path operation that regularly
// returns EXHAUSTED or OVER_BUDGET during normal operation (pool growth,
// steady-state pipelining where releases lag reservations by one epoch).
// Using iree_make_status() for these cases would capture backtraces on every
// transient exhaustion — expensive and misleading.
//
// The iree_status_t return from reserve() is reserved for infrastructure
// failures: invalid arguments, internal corruption, slab provider errors.
// Those are exceptional and warrant backtraces.
typedef uint32_t iree_hal_pool_reserve_result_t;
enum iree_hal_pool_reserve_result_e {
  // Block reserved successfully. The death frontier from the recycled block
  // was dominated by the requester's frontier — zero-sync reuse. The memory
  // is safe for immediate use without any device synchronization.
  IREE_HAL_POOL_RESERVE_OK = 0,

  // Block reserved from previously unused offset space (first use of this
  // region, or the block was freed with a NULL frontier). No death frontier
  // to check. Equivalent to OK for callers — no synchronization needed.
  IREE_HAL_POOL_RESERVE_OK_FRESH = 1,

  // Block reserved, but the death frontier was NOT dominated by the
  // requester's frontier. The block IS reserved (offset assigned), but the
  // caller must add a semaphore wait on the death frontier's axes before the
  // GPU uses this memory. The block's death frontier is available from the
  // pool via the reservation's block_handle.
  //
  // This occurs when try-before-fence fails: prior work on this block may
  // still be executing, and the requester's frontier does not transitively
  // cover it. The caller decides whether to accept the wait dependency or
  // release the reservation and retry.
  IREE_HAL_POOL_RESERVE_OK_NEEDS_WAIT = 2,

  // No blocks available. All blocks are allocated, or all free blocks have
  // non-dominated death frontiers and the pool chose not to hand one out
  // as NEEDS_WAIT.
  //
  // The caller should wait on the pool's notification (signaled when any
  // release occurs) and retry, or trigger slab growth. No reservation was
  // made — |out_reservation| is not modified.
  IREE_HAL_POOL_RESERVE_EXHAUSTED = 3,

  // The pool's budget limit would be exceeded by this reservation. The
  // request is valid and blocks may be physically available, but the budget
  // policy prevents the allocation.
  //
  // The caller should free other reservations from this pool, adjust the
  // budget, or use a different pool. No reservation was made —
  // |out_reservation| is not modified.
  IREE_HAL_POOL_RESERVE_OVER_BUDGET = 4,
};

// A reservation from a pool. Returned by iree_hal_pool_reserve() and passed
// to iree_hal_pool_release_reservation().
//
// This is a pure value type (32 bytes, no ownership). It lives on the stack
// during queue submission or is stored in the buffer that wraps it. The pool
// knows how to interpret its fields — callers treat it as opaque.
typedef struct iree_hal_pool_reservation_t {
  // Offset within the pool's managed range.
  iree_device_size_t offset;

  // Actual allocated length in bytes. May exceed the requested length due to
  // alignment rounding or because the block could not be split (the remainder
  // would have been smaller than the minimum block size).
  iree_device_size_t length;

  // Pool-internal opaque handle for returning the block on release.
  // Interpretation is strategy-specific: TLSF block_index, block pool
  // block_index, slab base_ptr for pass-through pools, etc. 64-bit to
  // accommodate pointer-sized handles on all platforms.
  uint64_t block_handle;

  // Which slab within the pool (for multi-slab pools in slab mode).
  // 0 for single-slab or VMM pools.
  uint16_t slab_index;

  uint16_t reserved[3];
} iree_hal_pool_reservation_t;

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

  // Minimum allocation size in bytes. Block pools: block_size. TLSF:
  // alignment. Arena: 1. 0 means no minimum.
  iree_device_size_t min_allocation_size;

  // Maximum allocation size in bytes. Block pools: block_size (or
  // block_count * block_size for multi-block). TLSF: range_length.
  // 0 means no maximum.
  iree_device_size_t max_allocation_size;
} iree_hal_pool_capabilities_t;

// Running statistics for a pool. All values are atomic snapshots — they may
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

  // Cumulative counters (monotonically increasing over pool lifetime).
  uint64_t reserve_count;      // Total successful reserve() calls.
  uint64_t release_count;      // Total release_reservation() calls.
  uint64_t reuse_count;        // Reserves that hit frontier-dominated reuse.
  uint64_t reuse_miss_count;   // Reserves where dominance check failed.
  uint64_t fresh_count;        // Reserves from fresh (never-used) space.
  uint64_t exhausted_count;    // Reserves that returned EXHAUSTED.
  uint64_t over_budget_count;  // Reserves that returned OVER_BUDGET.
  uint64_t wait_count;         // Reserves that returned NEEDS_WAIT.
} iree_hal_pool_stats_t;

// Callback for try-before-fence epoch queries. When a death frontier
// dominance check fails in reserve(), the pool calls this to check whether
// the timeline has actually advanced past the death frontier's epoch on a
// specific axis — even though the requester's frontier hasn't imported the
// update yet.
//
// Returns true if |axis| has reached at least |epoch| (the work has
// completed). This is a host-side read of the semaphore's current value —
// no device interaction, no blocking. The pool uses this to avoid
// unnecessarily skipping reusable blocks when completion notifications are
// batched by the proactor.
//
// Set at pool creation time. Pools created without a callback (NULL) skip
// the try-before-fence optimization — non-dominated blocks are treated as
// genuinely unavailable for zero-sync reuse.
typedef bool (*iree_hal_pool_epoch_query_fn_t)(void* user_data,
                                               iree_async_axis_t axis,
                                               uint64_t epoch);

//===----------------------------------------------------------------------===//
// iree_hal_pool_t
//===----------------------------------------------------------------------===//

// A memory pool that manages a region of offset space backed by one or more
// slabs of physical memory. Pools are the primary allocation interface in IREE:
// both synchronous (allocate_buffer) and asynchronous (queue_alloca/dealloca)
// allocation paths go through pools.
//
// Pools are ref-counted HAL resources. Buffers allocated from a pool retain it,
// ensuring the pool outlives its allocations. Pools can be shared across queues
// and threads (concurrency is handled internally per pool type).
//
// The pool base type is abstract — concrete pools are created by type-specific
// factory functions (iree_hal_tlsf_pool_create, iree_hal_block_pool_create,
// etc.) and used through this common interface.
//
// ## Allocation protocol
//
// Asynchronous (queue-ordered) allocation:
//   1. reserve() at submit time — finds a free block, checks death frontier
//      dominance, returns a reservation with offset and length.
//   2. wrap_reservation() — creates a buffer backed by the reservation.
//   3. release_reservation() at dealloca submit time — returns the block to
//      the pool's free list, tagged with a death frontier.
//
// Synchronous allocation:
//   iree_hal_pool_allocate_buffer() — calls reserve + wrap_reservation in a
//   loop, waiting on the pool's notification if exhausted. This is a shared
//   utility, not a vtable method.
//
// ## Death frontier integration
//
// Free blocks carry death frontiers — causal snapshots from when they were
// last freed. Reserve checks whether the requester's frontier dominates the
// death frontier for zero-sync reuse. This enables buffer recycling without
// any device synchronization in steady-state pipelined workloads.
typedef struct iree_hal_pool_t iree_hal_pool_t;

// Retains the given |pool| for the caller.
IREE_API_EXPORT void iree_hal_pool_retain(iree_hal_pool_t* pool);

// Releases the given |pool| from the caller.
IREE_API_EXPORT void iree_hal_pool_release(iree_hal_pool_t* pool);

// Reserves a block from the pool for a future allocation.
//
// |size| is the minimum number of bytes needed. |alignment| is the required
// alignment for the returned offset (must be a power of two and > 0). The
// actual allocated length may exceed |size| due to alignment rounding or
// block splitting constraints.
//
// |requester_frontier| is the caller's current causal position, used for
// death frontier dominance checking. Pass NULL to skip dominance checking
// (appropriate for synchronous allocations that don't participate in
// queue-ordered frontier tracking).
//
// On success (iree_ok_status()), |out_result| indicates the specific outcome:
//   OK / OK_FRESH — reservation succeeded, memory is safe for immediate use.
//   OK_NEEDS_WAIT — reservation succeeded, but the caller must add a
//     semaphore wait before the GPU uses this memory.
//   EXHAUSTED — no reservation made, pool has no suitable blocks.
//   OVER_BUDGET — no reservation made, budget limit would be exceeded.
//
// Returns an error status (with backtrace) only for infrastructure failures:
// invalid arguments (size 0, non-power-of-two alignment), internal
// corruption, or slab provider errors. These are exceptional.
IREE_API_EXPORT iree_status_t
iree_hal_pool_reserve(iree_hal_pool_t* pool, iree_device_size_t size,
                      iree_device_size_t alignment,
                      const iree_async_frontier_t* requester_frontier,
                      iree_hal_pool_reservation_t* out_reservation,
                      iree_hal_pool_reserve_result_t* out_result);

// Releases a reservation back to the pool's free list.
//
// |reservation| is the reservation returned by a prior iree_hal_pool_reserve()
// call on this pool. |death_frontier| is the causal snapshot to attach to the
// freed block — typically the queue's frontier at dealloca submit time. Pass
// NULL for an empty frontier (the block is immediately available for
// zero-sync reuse by any requester).
//
// The reservation's offset is returned to the pool's free list immediately.
// The memory is available for future reserve() calls from that point forward,
// even though the GPU may still be executing prior work — death frontier
// dominance checking gates actual reuse safety.
//
// Signals the pool's notification (wakes threads waiting in
// iree_hal_pool_allocate_buffer).
IREE_API_EXPORT void iree_hal_pool_release_reservation(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier);

// Creates a buffer wrapping a reservation from this pool.
//
// |params| describes the buffer's usage, access, and memory type properties.
// |reservation| is the reservation returned by iree_hal_pool_reserve().
// The returned buffer retains the pool and stores the reservation internally.
// When the buffer is destroyed (ref count reaches 0), the reservation is
// released back to the pool.
//
// This is pool-type-specific because the buffer must reference the pool's
// slab(s) for memory access (host mapping, GPU dispatch, etc.).
IREE_API_EXPORT iree_status_t iree_hal_pool_wrap_reservation(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_buffer_t** out_buffer);

// Queries the memory capabilities of the pool. O(1) — reads cached fields
// computed at pool creation time.
IREE_API_EXPORT void iree_hal_pool_query_capabilities(
    const iree_hal_pool_t* pool,
    iree_hal_pool_capabilities_t* out_capabilities);

// Queries the pool's running statistics. O(1) — atomic snapshots of
// incrementally maintained counters. Values may be momentarily inconsistent
// under concurrent modifications.
IREE_API_EXPORT void iree_hal_pool_query_stats(
    const iree_hal_pool_t* pool, iree_hal_pool_stats_t* out_stats);

// Releases unused physical memory back to the slab provider.
// VMM mode: decommit pages with no live reservations.
// Slab mode: free slabs with no live reservations.
// The pool remains valid after trimming — it can grow again on demand.
IREE_API_EXPORT iree_status_t iree_hal_pool_trim(iree_hal_pool_t* pool);

// Returns the pool's notification, signaled on every release_reservation().
// Callers waiting for blocks to become available can use this to sleep
// efficiently instead of polling.
IREE_API_EXPORT iree_async_notification_t* iree_hal_pool_notification(
    iree_hal_pool_t* pool);

// Allocates a buffer from the pool synchronously.
//
// This is a shared utility (NOT a vtable method) that calls reserve() +
// wrap_reservation() in a loop. If reserve returns EXHAUSTED or OVER_BUDGET,
// the function waits on the pool's notification for |timeout| and retries.
//
// |requester_frontier| is passed to reserve() for dominance checking. Pass
// NULL to skip dominance checking (appropriate for persistent buffers that
// aren't queue-ordered).
//
// |timeout| controls how long to wait for a free block. Converted to an
// absolute deadline internally so retries after spurious wakes use a
// consistent cutoff:
//   iree_make_timeout_ms(0) — try once, fail immediately if exhausted.
//   iree_infinite_timeout() — block until a block becomes available.
//   iree_make_timeout_ms(N) — wait up to N milliseconds.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if the pool is exhausted and
// |timeout| expired. Returns IREE_STATUS_DEADLINE_EXCEEDED if the timeout
// was reached. These use iree_make_status() because they represent a
// terminal failure visible to the application, not a transient hot-path
// condition.
IREE_API_EXPORT iree_status_t iree_hal_pool_allocate_buffer(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    iree_device_size_t allocation_size,
    const iree_async_frontier_t* requester_frontier, iree_timeout_t timeout,
    iree_hal_buffer_t** out_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_pool_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_pool_vtable_t {
  void(IREE_API_PTR* destroy)(iree_hal_pool_t* pool);

  iree_status_t(IREE_API_PTR* reserve)(
      iree_hal_pool_t* pool, iree_device_size_t size,
      iree_device_size_t alignment,
      const iree_async_frontier_t* requester_frontier,
      iree_hal_pool_reservation_t* out_reservation,
      iree_hal_pool_reserve_result_t* out_result);

  void(IREE_API_PTR* release_reservation)(
      iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
      const iree_async_frontier_t* death_frontier);

  iree_status_t(IREE_API_PTR* wrap_reservation)(
      iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
      const iree_hal_pool_reservation_t* reservation,
      iree_hal_buffer_t** out_buffer);

  void(IREE_API_PTR* query_capabilities)(
      const iree_hal_pool_t* pool,
      iree_hal_pool_capabilities_t* out_capabilities);

  void(IREE_API_PTR* query_stats)(const iree_hal_pool_t* pool,
                                  iree_hal_pool_stats_t* out_stats);

  iree_status_t(IREE_API_PTR* trim)(iree_hal_pool_t* pool);

  iree_async_notification_t*(IREE_API_PTR* notification)(iree_hal_pool_t* pool);
} iree_hal_pool_vtable_t;
IREE_HAL_ASSERT_VTABLE_LAYOUT(iree_hal_pool_vtable_t);

IREE_API_EXPORT void iree_hal_pool_destroy(iree_hal_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_POOL_H_
