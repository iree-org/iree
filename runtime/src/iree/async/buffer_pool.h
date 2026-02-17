// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lock-free buffer pool for zero-copy send operations.
//
// A buffer pool is a lock-free freelist over a registered region. Buffers are
// acquired for filling and sending, then returned after the send completes.
// Both acquire and release are O(1) operations.
//
// ## Separation of concerns
//
// Buffer pools sit at the top of the slab/region/pool hierarchy:
//
//   Slab    Physical memory. Ref-counted. No backend knowledge.
//     |
//   Region  Registration handle. Created by proactor via register_slab.
//     |     Contains backend-specific handles (io_uring fixed buffers, etc).
//     |
//   Pool    Lock-free freelist over a region. Send-side acquire/release.
//
// The pool does not allocate memory or interact with the proactor. It operates
// purely on the region passed to it at creation. The typical workflow is:
//
//   // 1. Create slab with NUMA/huge page options.
//   iree_async_slab_t* slab = NULL;
//   IREE_RETURN_IF_ERROR(iree_async_slab_create(options, allocator, &slab));
//
//   // 2. Register slab with proactor for zero-copy I/O.
//   iree_async_region_t* region = NULL;
//   IREE_RETURN_IF_ERROR(iree_async_proactor_register_slab(
//       proactor, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));
//   iree_async_slab_release(slab);  // Region holds a ref; slab can be
//   released.
//
//   // 3. Create pool over region.
//   iree_async_buffer_pool_t* pool = NULL;
//   IREE_RETURN_IF_ERROR(iree_async_buffer_pool_allocate(region, allocator,
//                                                        &pool));
//
//   // 4. Use pool for send operations.
//   iree_async_buffer_lease_t lease;
//   IREE_RETURN_IF_ERROR(iree_async_buffer_pool_acquire(pool, &lease));
//   // Fill buffer, submit send, wait for completion...
//   iree_async_buffer_lease_release(&lease);
//
//   // 5. Cleanup.
//   iree_async_buffer_pool_free(pool);
//   iree_async_region_release(region);
//
// ## Lease semantics
//
// Callers acquire buffers from the pool as leases. A lease identifies a
// specific buffer (via span) and must be returned via
// iree_async_buffer_lease_release() after use. The release function is
// polymorphic, allowing the same lease type to be used for both send pools
// (freelist push) and recv completions (PBUF_RING recycle).
//
// ## Thread safety
//
// Both acquire and release are thread-safe. Multiple threads may concurrently
// acquire buffers for sending and release them after completions arrive.
//
// ## Singleton constraint
//
// For io_uring send operations (READ access), only one region may be registered
// per proactor (kernel limitation: single fixed buffer table per ring). This
// constraint is enforced by register_slab, not by the pool.

#ifndef IREE_ASYNC_BUFFER_POOL_H_
#define IREE_ASYNC_BUFFER_POOL_H_

#include "iree/async/region.h"
#include "iree/async/span.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_buffer_pool_t iree_async_buffer_pool_t;

// Index of a buffer within a pool (0..buffer_count-1).
// Used for O(1) freelist operations and backend buffer identification
// (io_uring buffer_id within a group, RDMA WQE slot index).
typedef uint32_t iree_async_buffer_index_t;

//===----------------------------------------------------------------------===//
// Buffer lease
//===----------------------------------------------------------------------===//

// A buffer acquired from a pool or received from the kernel.
//
// Value type — the pool tracks availability by buffer index in an internal
// freelist, not by lease pointer. Callers may copy or embed leases freely.
//
// Release is polymorphic: call iree_async_buffer_lease_release() to return
// the buffer to its source (pool freelist or recv ring).
typedef struct iree_async_buffer_lease_t {
  // Span referencing the leased buffer within the registered region.
  // span.region points to the region (valid for the region's lifetime).
  // span.offset = index * buffer_size.
  // span.length = buffer_size.
  iree_async_span_t span;

  // Polymorphic release callback and context.
  // For pool leases: release.fn pushes to freelist, release_context is pool.
  // For recv leases: release.fn recycles to PBUF_RING, release_context varies.
  iree_async_buffer_recycle_callback_t release;

  // Buffer index within the region (0..buffer_count-1).
  // Used for O(1) return and backend-specific buffer identification.
  iree_async_buffer_index_t buffer_index;
} iree_async_buffer_lease_t;

// Releases a buffer lease, returning it to its source.
// For pool leases: pushes the buffer back to the freelist.
// For recv leases: recycles the buffer to the provided buffer ring.
// The caller must not access the lease's span data after this call.
//
// This function is idempotent: calling it multiple times on the same lease
// is safe (subsequent calls are no-ops). This simplifies error handling paths.
static inline void iree_async_buffer_lease_release(
    iree_async_buffer_lease_t* lease) {
  if (!lease) return;
  iree_async_buffer_recycle_callback_t release = lease->release;
  if (release.fn) {
    iree_async_buffer_index_t buffer_index = lease->buffer_index;
    // Clear before calling to make idempotent (callback may reuse lease).
    lease->release = iree_async_buffer_recycle_callback_null();
    release.fn(release.user_data, buffer_index);
  }
}

//===----------------------------------------------------------------------===//
// Pool lifecycle
//===----------------------------------------------------------------------===//

// Allocates a buffer pool over a registered region.
//
// The pool provides lock-free acquire/release over the buffers described by
// the region. The region must have been created via register_slab and must
// outlive the pool (the pool retains a reference).
//
// The region's buffer_count and buffer_size are used to configure the pool.
// All buffers start as available in the freelist.
//
// On failure, no resources are leaked and |out_pool| is set to NULL.
IREE_API_EXPORT iree_status_t iree_async_buffer_pool_allocate(
    iree_async_region_t* region, iree_allocator_t allocator,
    iree_async_buffer_pool_t** out_pool);

// Frees a buffer pool. The caller must ensure:
//   1. All leases have been returned (freelist is full).
//   2. All I/O operations using buffers from this pool have completed.
//
// Freeing a pool with outstanding leases is a programming error and will
// trigger an assertion failure.
//
// Releases the region reference acquired during allocation.
IREE_API_EXPORT void iree_async_buffer_pool_free(
    iree_async_buffer_pool_t* pool);

//===----------------------------------------------------------------------===//
// Acquire / Release
//===----------------------------------------------------------------------===//

// Acquires a buffer from the pool for filling and sending.
//
// On success, |out_lease| is populated with a span covering the acquired
// buffer, a release callback, and the buffer index. The span's region is the
// pool's region (valid for the pool's lifetime).
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if all buffers are currently leased.
// This is non-blocking: the caller should implement backpressure (e.g., defer
// the operation, apply flow control) rather than spin-waiting.
//
// Thread-safe: may be called from any thread concurrently with other acquire
// or release calls.
//
// O(1) operation: pops from the internal lock-free freelist.
IREE_API_EXPORT iree_status_t iree_async_buffer_pool_acquire(
    iree_async_buffer_pool_t* pool, iree_async_buffer_lease_t* out_lease);

//===----------------------------------------------------------------------===//
// Query
//===----------------------------------------------------------------------===//

// Returns the number of buffers currently available (not leased).
IREE_API_EXPORT iree_host_size_t
iree_async_buffer_pool_available(const iree_async_buffer_pool_t* pool);

// Returns the total number of buffers in the pool (leased + available).
IREE_API_EXPORT iree_host_size_t
iree_async_buffer_pool_capacity(const iree_async_buffer_pool_t* pool);

// Returns the per-buffer size in bytes.
IREE_API_EXPORT iree_host_size_t
iree_async_buffer_pool_buffer_size(const iree_async_buffer_pool_t* pool);

// Returns the pool's registered region.
// The region contains backend-specific handles for zero-copy I/O.
IREE_API_EXPORT iree_async_region_t* iree_async_buffer_pool_region(
    const iree_async_buffer_pool_t* pool);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_BUFFER_POOL_H_
