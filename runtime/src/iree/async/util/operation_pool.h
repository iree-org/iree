// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_UTIL_OPERATION_POOL_H_
#define IREE_ASYNC_UTIL_OPERATION_POOL_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Operation pool
//===----------------------------------------------------------------------===//

// A size-class block pool for async operation structs.
//
// Operations are allocated from the pool and returned after their final
// completion callback fires. The pool eliminates per-operation malloc/free
// traffic in steady-state, which matters because high-throughput network
// servers may process millions of operations per second.
//
// ## Size classes
//
// The pool maintains power-of-two size classes (64, 128, 256, ... up to a
// configurable maximum). Each size class has its own freelist. Acquiring an
// operation rounds up to the next size class and pops from that freelist
// (O(1) in the common case). When a freelist is empty, the pool allocates a
// new block and carves it into slots for that size class.
//
// Requests larger than the maximum pooled size are allocated directly from
// the allocator and freed on release.
//
// ## Block allocation
//
// Rather than allocating individual operations, the pool allocates blocks
// (default 64KB) and divides them into fixed-size slots. This amortizes
// allocation overhead and improves cache locality for operations in the
// same block.
//
// ## NUMA awareness
//
// The pool allocates from the provided allocator, which should be NUMA-local
// to the proactor thread that will use these operations. When operations are
// filled and submitted from the same NUMA node where their memory lives,
// cache line bouncing is minimized.
//
// ## Automatic return
//
// Operations have an optional |pool| field. When set, the proactor
// automatically calls iree_async_operation_pool_release() after the final
// completion callback returns. This eliminates manual release bookkeeping:
//
//   iree_async_operation_t* op = NULL;
//   IREE_RETURN_IF_ERROR(iree_async_operation_pool_acquire(
//       pool, sizeof(iree_async_socket_recv_operation_t), &op));
//   op->pool = pool;  // Auto-release after final callback.
//   // ... fill subtype fields, submit ...
//   // After final callback fires, op is back in the pool. Don't touch it.
//
// ## Thread safety
//
// The pool is thread-safe. Freelists use lock-free atomic operations for
// push/pop, allowing concurrent acquire/release from multiple threads.
// This is important for pools shared across proactors or when completions
// fire on different threads than submissions.
//
// ## Lifecycle
//
// Allocate the pool, acquire/release operations during steady-state, then
// free the pool when shutting down. Freeing the pool releases all memory in
// all blocks. Operations that are still in-flight (not yet returned to the
// pool) are the caller's responsibility — they must be cancelled and their
// callbacks must fire before the pool is freed.

typedef struct iree_async_operation_pool_t iree_async_operation_pool_t;

// Options for pool creation.
typedef struct iree_async_operation_pool_options_t {
  // Block size for new allocations. Larger blocks mean fewer allocations but
  // potentially more wasted memory if the pool shrinks.
  // 0 = use default (64KB).
  iree_host_size_t block_size;

  // Maximum operation size to pool. Requests larger than this are allocated
  // directly from the allocator and freed on release.
  // 0 = use default (16KB).
  iree_host_size_t max_pooled_size;
} iree_async_operation_pool_options_t;

// Returns default pool options.
static inline iree_async_operation_pool_options_t
iree_async_operation_pool_options_default(void) {
  iree_async_operation_pool_options_t options = {0};
  return options;
}

// Allocates an operation pool with the given options.
// The pool allocates operation structs from |allocator| as needed and
// maintains freelists for reuse.
iree_status_t iree_async_operation_pool_allocate(
    iree_async_operation_pool_options_t options, iree_allocator_t allocator,
    iree_async_operation_pool_t** out_pool);

// Frees an operation pool, releasing all cached operation structs.
// All operations must have been released back to the pool (or their memory
// ownership transferred elsewhere) before calling this. Operations still
// in-flight will leak.
void iree_async_operation_pool_free(iree_async_operation_pool_t* pool);

// Acquires an operation of at least |size| bytes from the pool.
// The returned operation is zeroed and ready to fill. The base operation
// fields are NOT pre-initialized; the caller must set type, completion_fn,
// user_data, and flags before submitting.
//
// Set operation->pool = pool for automatic release after the final callback.
//
// Returns from the freelist if available (O(1)), otherwise allocates from
// a new or existing block.
//
// For fixed-size operations, use sizeof(subtype):
//   acquire(pool, sizeof(iree_async_socket_recv_operation_t), &op);
//
// For variable-size operations, compute size with the _size() helper:
//   iree_host_size_t size = 0;
//   IREE_RETURN_IF_ERROR(
//       iree_async_sequence_operation_size(step_count, &size));
//   acquire(pool, size, &op);
iree_status_t iree_async_operation_pool_acquire(
    iree_async_operation_pool_t* pool, iree_host_size_t size,
    iree_async_operation_t** out_operation);

// Releases an operation back to the pool for reuse.
// The operation is returned to the freelist for its size class. The caller
// must not access the operation after this call.
//
// This is called automatically by the proactor after the final completion
// callback if operation->pool is set. Manual calls are needed only when
// the pool field is NULL or when an operation is being abandoned without
// submission (e.g., error during setup after acquire).
void iree_async_operation_pool_release(iree_async_operation_pool_t* pool,
                                       iree_async_operation_t* operation);

// Trims the pool by releasing cached operations back to the allocator.
// Currently a no-op (deferred implementation). Returns 0.
iree_host_size_t iree_async_operation_pool_trim(
    iree_async_operation_pool_t* pool, iree_host_size_t max_to_trim);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_UTIL_OPERATION_POOL_H_
