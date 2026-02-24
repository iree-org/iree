// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// io_uring ring buffer management.
//
// Wraps the kernel's memory-mapped ring buffers with a clean interface for:
//   - Ring setup (io_uring_setup syscall + mmap)
//   - SQE acquisition and submission
//   - CQE processing with proper barriers
//   - Ring teardown (munmap + close)
//
// Thread safety:
//   SQ operations (get_sqe, fill, flush) are protected by an atomic spinlock
//   (sq_lock). Callers must hold the lock during the entire get_sqe -> fill ->
//   unlock sequence to prevent partially-filled SQEs from being flushed.
//
//   io_uring_enter is called OUTSIDE the lock and ONLY from the poll thread.
//   This satisfies IORING_SETUP_SINGLE_ISSUER (only one thread may call
//   io_uring_enter). Cross-thread submitters fill SQEs under the lock and wake
//   the poll thread to flush via io_uring_enter.
//
//   CQ operations (peek, advance) are poll-thread-only and do not require the
//   lock.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_URING_H_
#define IREE_ASYNC_PLATFORM_IO_URING_URING_H_

#include "iree/async/platform/io_uring/defs.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Ring state
//===----------------------------------------------------------------------===//

// State for an io_uring instance. Manages the ring fd and memory mappings.
typedef struct iree_io_uring_ring_t {
  // Ring file descriptor from io_uring_setup.
  int ring_fd;

  // Features reported by the kernel during setup.
  uint32_t features;

  // Submission queue state.
  void* sq_ring_ptr;
  size_t sq_ring_size;
  uint32_t* sq_head;       // Kernel-updated (acquire load).
  uint32_t* sq_tail;       // User-updated (release store).
  uint32_t* sq_ring_mask;  // Constant after setup.
  uint32_t* sq_ring_entries;
  uint32_t* sq_flags;
  uint32_t* sq_array;  // Index array into sqes.

  // SQE array (separate mmap).
  iree_io_uring_sqe_t* sqes;
  size_t sqes_size;

  // Completion queue state.
  void* cq_ring_ptr;
  size_t cq_ring_size;
  uint32_t* cq_head;  // User-updated (release store).
  uint32_t* cq_tail;  // Kernel-updated (acquire load).
  uint32_t* cq_ring_mask;
  uint32_t* cq_ring_entries;
  uint32_t* cq_overflow;
  iree_io_uring_cqe_t* cqes;

  // Cached values for fast access.
  uint32_t sq_mask;
  uint32_t cq_mask;
  uint32_t sq_entries;
  uint32_t cq_entries;

  // Local SQ tail for two-phase commit. get_sqe() increments this locally;
  // submit() flushes it to the kernel-visible *sq_tail.
  // This separation allows rollback on encoding failure and ensures we always
  // know how many SQEs are pending submission.
  // Protected by sq_lock.
  uint32_t sq_local_tail;

  // Spinlock protecting SQ mutations (sq_local_tail, *sq_tail, SQE fills).
  // 0 = unlocked, 1 = locked.
  iree_atomic_int32_t sq_lock;
} iree_io_uring_ring_t;

//===----------------------------------------------------------------------===//
// SQ spinlock
//===----------------------------------------------------------------------===//

// Acquires the SQ spinlock. Must be held during get_sqe, SQE fill, rollback,
// and sq_flush sequences. The lock is NOT held during io_uring_enter.
static inline void iree_io_uring_ring_sq_lock(iree_io_uring_ring_t* ring) {
  while (iree_atomic_exchange(&ring->sq_lock, 1, iree_memory_order_acquire)) {
    // Spin until unlocked. Contention is rare (cross-thread submit vs poll
    // thread internal operations) and the critical section is short (SQE
    // pointer bump + memcpy).
  }
}

// Releases the SQ spinlock.
static inline void iree_io_uring_ring_sq_unlock(iree_io_uring_ring_t* ring) {
  iree_atomic_store(&ring->sq_lock, 0, iree_memory_order_release);
}

//===----------------------------------------------------------------------===//
// Ring lifecycle
//===----------------------------------------------------------------------===//

// Setup options.
typedef struct iree_io_uring_ring_options_t {
  // Desired number of SQ entries. Will be rounded up to power of 2.
  // Zero uses a reasonable default (256).
  uint32_t sq_entries;

  // Setup flags to request (IORING_SETUP_*). We automatically try fallbacks
  // if certain flags aren't supported.
  uint32_t setup_flags;
} iree_io_uring_ring_options_t;

// Returns default ring options.
static inline iree_io_uring_ring_options_t iree_io_uring_ring_options_default(
    void) {
  iree_io_uring_ring_options_t options = {0};
  options.sq_entries = 256;
  options.setup_flags =
      IREE_IORING_SETUP_SINGLE_ISSUER | IREE_IORING_SETUP_DEFER_TASKRUN;
  return options;
}

// Initializes an io_uring ring.
// Creates the ring via io_uring_setup and maps the ring buffers.
//
// Returns IREE_STATUS_UNAVAILABLE if io_uring is not supported on this system
// (kernel too old, disabled, or permission denied).
iree_status_t iree_io_uring_ring_initialize(
    iree_io_uring_ring_options_t options, iree_io_uring_ring_t* out_ring);

// Deinitializes an io_uring ring.
// Unmaps ring buffers and closes the ring fd. Safe to call on a zero-
// initialized or partially initialized ring.
void iree_io_uring_ring_deinitialize(iree_io_uring_ring_t* ring);

//===----------------------------------------------------------------------===//
// Submission queue operations
//===----------------------------------------------------------------------===//

// Returns the number of available SQ slots.
static inline uint32_t iree_io_uring_ring_sq_space_left(
    iree_io_uring_ring_t* ring) {
  // sq_head is updated by kernel, needs acquire load.
  uint32_t head = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                   iree_memory_order_acquire);
  // Use local tail (includes uncommitted SQEs).
  return ring->sq_entries - (ring->sq_local_tail - head);
}

// Returns the number of SQEs prepared but not yet submitted to kernel.
static inline uint32_t iree_io_uring_ring_sq_pending(
    iree_io_uring_ring_t* ring) {
  return ring->sq_local_tail - *ring->sq_tail;
}

// Rolls back |count| uncommitted SQEs from the local tail.
// Use this to undo get_sqe() calls when encoding fails partway through a batch.
static inline void iree_io_uring_ring_sq_rollback(iree_io_uring_ring_t* ring,
                                                  uint32_t count) {
  ring->sq_local_tail -= count;
}

// Gets the next SQE slot for filling. Returns NULL if SQ is full.
// The returned SQE is zeroed and ready to fill with operation details.
// After filling, call iree_io_uring_ring_submit() to submit pending SQEs.
iree_io_uring_sqe_t* iree_io_uring_ring_get_sqe(iree_io_uring_ring_t* ring);

// Submits all pending SQEs that were acquired via get_sqe().
// Automatically calculates the number of SQEs to submit from ring state.
// |min_complete| is the minimum number of CQEs to wait for (0 for non-blocking)
// |flags| are IORING_ENTER_* flags.
iree_status_t iree_io_uring_ring_submit(iree_io_uring_ring_t* ring,
                                        uint32_t min_complete, uint32_t flags);

//===----------------------------------------------------------------------===//
// Completion queue operations
//===----------------------------------------------------------------------===//

// Returns true if there are CQEs ready to process.
static inline bool iree_io_uring_ring_cq_ready(iree_io_uring_ring_t* ring) {
  // cq_tail is updated by kernel, needs acquire load.
  uint32_t tail = iree_atomic_load((iree_atomic_int32_t*)ring->cq_tail,
                                   iree_memory_order_acquire);
  return *ring->cq_head != tail;
}

// Returns the number of CQEs ready to process.
static inline uint32_t iree_io_uring_ring_cq_count(iree_io_uring_ring_t* ring) {
  uint32_t tail = iree_atomic_load((iree_atomic_int32_t*)ring->cq_tail,
                                   iree_memory_order_acquire);
  return tail - *ring->cq_head;
}

// Gets the next CQE for processing. Returns NULL if CQ is empty.
// The CQE remains valid until iree_io_uring_ring_cq_advance() is called.
static inline iree_io_uring_cqe_t* iree_io_uring_ring_peek_cqe(
    iree_io_uring_ring_t* ring) {
  uint32_t tail = iree_atomic_load((iree_atomic_int32_t*)ring->cq_tail,
                                   iree_memory_order_acquire);
  uint32_t head = *ring->cq_head;
  if (head == tail) return NULL;
  return &ring->cqes[head & ring->cq_mask];
}

// Advances the CQ head by |count| entries after processing CQEs.
// This makes the processed CQE slots available to the kernel.
static inline void iree_io_uring_ring_cq_advance(iree_io_uring_ring_t* ring,
                                                 uint32_t count) {
  iree_atomic_store((iree_atomic_int32_t*)ring->cq_head, *ring->cq_head + count,
                    iree_memory_order_release);
}

// Waits for at least |min_complete| CQEs with timeout.
// |flush_pending| controls whether to submit pending SQEs before waiting.
// |timeout_ns| is the timeout in nanoseconds (IREE_DURATION_ZERO for
// non-blocking, IREE_DURATION_INFINITE for infinite wait).
//
// Returns OK if completions are available, DEADLINE_EXCEEDED on timeout.
iree_status_t iree_io_uring_ring_wait_cqe(iree_io_uring_ring_t* ring,
                                          uint32_t min_complete,
                                          bool flush_pending,
                                          iree_duration_t timeout_ns);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_URING_H_
