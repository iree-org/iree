// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// High-performance event pool for efficient 1:1 signaling.
//
// Optimized for the proactor polling pattern where:
//   - The polling thread (latency-critical) releases events
//   - Random producer threads acquire events
//
// Design goals:
//   - Nanosecond-scale acquire/release without global locks
//   - Zero contention on polling thread (lock-free release)
//   - NUMA-aware allocation via caller-provided block pool
//   - Good cache utilization with minimal cache line bouncing
//
// Implementation uses producer-consumer separated stacks:
//   - return_stack: lock-free Treiber stack for polling thread pushes
//   - acquire_stack: mutex-protected stack for producer pops
//   - Lazy migration moves events from return_stack to acquire_stack
//
// Reset syscall happens on acquire (not release), keeping the polling
// thread completely syscall-free during release.

#ifndef IREE_ASYNC_EVENT_POOL_H_
#define IREE_ASYNC_EVENT_POOL_H_

#include "iree/async/event.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Event pool
//===----------------------------------------------------------------------===//

// Lock-free return stack for polling thread releases.
// Aligned to cache line to prevent false sharing with acquire path.
typedef struct iree_alignas(iree_hardware_destructive_interference_size)
    iree_async_event_pool_return_stack_t {
  // Treiber stack head (lock-free push/pop via CAS).
  iree_atomic_intptr_t head;
} iree_async_event_pool_return_stack_t;

// High-performance event pool with producer-consumer separation.
//
// The pool separates release (polling thread) and acquire (producer threads)
// into different data structures to minimize contention:
//   - Release: lock-free Treiber stack push (~10-20ns)
//   - Acquire: mutex + pop + optional migration + reset (~150-500ns)
//
// Thread-safe: multiple producers can acquire concurrently; the polling thread
// can release concurrently with acquires.
typedef struct iree_async_event_pool_t {
  // Proactor that owns the events (for primitive creation/destruction).
  iree_async_proactor_t* proactor;

  // Allocator for event struct allocation. Pool grows by allocating events
  // from here; only fails on OOM.
  iree_allocator_t allocator;

  // Lock-free return stack (polling thread pushes here).
  // Separate cache line to avoid false sharing with acquire path.
  iree_async_event_pool_return_stack_t return_stack;

  // Mutex-protected acquire stack (producers pop here).
  iree_slim_mutex_t acquire_mutex;
  iree_async_event_t* acquire_head;

  // Track all allocated events for cleanup (singly-linked via pool_next).
  // Protected by acquire_mutex.
  iree_async_event_t* all_events_head;
} iree_async_event_pool_t;

// Initializes an event pool for efficient 1:1 signaling with reusable events.
// The pool is embedded in the caller's struct (proactor, scheduler, etc.).
//
// |proactor| is used for creating event primitives (eventfd, pipe, etc.).
// |allocator| is used for event struct allocation.
// For NUMA-aware allocation, pass an allocator backed by NUMA-local memory.
//
// |initial_capacity| events are pre-created during initialization, amortizing
// the eventfd syscall cost. The pool grows dynamically via |allocator| when
// exhausted.
//
// Thread-safe: multiple producers can acquire concurrently; the polling thread
// can release concurrently with acquires.
IREE_API_EXPORT iree_status_t iree_async_event_pool_initialize(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_host_size_t initial_capacity, iree_async_event_pool_t* out_pool);

// Deinitializes the pool and destroys all events in it.
// Requires all events to have been returned to pool (no in-flight operations).
IREE_API_EXPORT void iree_async_event_pool_deinitialize(
    iree_async_event_pool_t* pool);

// Acquires an event from the pool.
//
// The returned event is ready for use in a single wait operation. On io_uring,
// events are automatically drained by a linked READ when the wait completes,
// so no explicit reset is performed here. On other backends, the event may be
// reset during acquire if needed.
//
// Contract: Each acquired event should be used for exactly one wait operation
// before being released. Acquiring an event, signaling it, and releasing it
// WITHOUT submitting a wait operation may leave the event in a signaled state
// on some backends.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if allocation or eventfd creation
// fails.
//
// Thread-safe: multiple producers can acquire concurrently.
IREE_API_EXPORT iree_status_t iree_async_event_pool_acquire(
    iree_async_event_pool_t* pool, iree_async_event_t** out_event);

// Returns an event to the pool.
// The event is NOT reset here; reset happens on the next acquire.
// This keeps the polling thread completely syscall-free.
//
// Thread-safe: lock-free for the polling thread (single-producer path).
IREE_API_EXPORT void iree_async_event_pool_release(
    iree_async_event_pool_t* pool, iree_async_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_EVENT_POOL_H_
