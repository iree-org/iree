// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/event_pool.h"

#include "iree/async/event.h"

//===----------------------------------------------------------------------===//
// Internal: Pool growth
//===----------------------------------------------------------------------===//

// Allocates a new event and adds it to the pool's all_events list.
// Called under acquire_mutex when both stacks are empty.
static iree_status_t iree_async_event_pool_grow_locked(
    iree_async_event_pool_t* pool, iree_async_event_t** out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a new event via the proactor (eventfd on Linux, pipe on macOS, etc.)
  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_event_create(pool->proactor, &event));

  // Set the event's home pool for release routing.
  event->pool = pool;

  // Add to all_events list for cleanup during deinitialize.
  // Use pool_all_next (not pool_next) to keep this list independent of the
  // acquire_stack and return_stack which use pool_next.
  event->pool_all_next = pool->all_events_head;
  pool->all_events_head = event;

  *out_event = event;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Event pool
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_async_event_pool_initialize(
    iree_async_proactor_t* proactor, iree_allocator_t allocator,
    iree_host_size_t initial_capacity, iree_async_event_pool_t* out_pool) {
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(out_pool);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_pool, 0, sizeof(*out_pool));
  out_pool->proactor = proactor;
  out_pool->allocator = allocator;
  iree_atomic_store(&out_pool->return_stack.head, 0, iree_memory_order_relaxed);
  iree_slim_mutex_initialize(&out_pool->acquire_mutex);
  out_pool->acquire_head = NULL;
  out_pool->all_events_head = NULL;

  // Pre-create initial_capacity events to amortize eventfd syscall cost.
  // Events go directly to acquire_head (no need for migration on first use).
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < initial_capacity && iree_status_is_ok(status); ++i) {
    iree_async_event_t* event = NULL;
    status = iree_async_event_pool_grow_locked(out_pool, &event);
    if (iree_status_is_ok(status)) {
      // Add to acquire stack (already in all_events from grow_locked).
      event->pool_next = out_pool->acquire_head;
      out_pool->acquire_head = event;
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_async_event_pool_deinitialize(out_pool);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_async_event_pool_deinitialize(
    iree_async_event_pool_t* pool) {
  if (!pool) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Destroy all events in the all_events list.
  // This list uses pool_all_next linkage (independent of pool_next used by
  // acquire_stack and return_stack).
  iree_async_event_t* event = pool->all_events_head;
  while (event) {
    iree_async_event_t* next = event->pool_all_next;
    // Clear pool references before destroying to avoid double-free issues.
    event->pool = NULL;
    event->pool_next = NULL;
    event->pool_all_next = NULL;
    // Release the event (will call proactor vtable destroy_event).
    iree_async_event_release(event);
    event = next;
  }
  pool->all_events_head = NULL;
  pool->acquire_head = NULL;
  iree_atomic_store(&pool->return_stack.head, 0, iree_memory_order_relaxed);

  iree_slim_mutex_deinitialize(&pool->acquire_mutex);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_async_event_pool_acquire(
    iree_async_event_pool_t* pool, iree_async_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;

  iree_slim_mutex_lock(&pool->acquire_mutex);

  iree_async_event_t* event = NULL;
  iree_status_t status = iree_ok_status();

  // Fast path: pop from acquire stack.
  if (pool->acquire_head) {
    event = pool->acquire_head;
    pool->acquire_head = event->pool_next;
  } else {
    // Slow path: migrate from return stack.
    iree_async_event_t* return_list = (iree_async_event_t*)iree_atomic_exchange(
        &pool->return_stack.head, 0, iree_memory_order_acquire);

    if (!return_list) {
      // Both stacks empty: grow the pool.
      status = iree_async_event_pool_grow_locked(pool, &event);
    } else {
      // Reverse the list for FIFO fairness (events returned first are acquired
      // first). This also ensures good cache locality since recently-used
      // events are acquired next.
      iree_async_event_t* reversed = NULL;
      while (return_list) {
        iree_async_event_t* next = return_list->pool_next;
        return_list->pool_next = reversed;
        reversed = return_list;
        return_list = next;
      }
      // Pop one for the caller, rest becomes acquire_head.
      event = reversed;
      pool->acquire_head = reversed->pool_next;
    }
  }

  iree_slim_mutex_unlock(&pool->acquire_mutex);

  if (iree_status_is_ok(status)) {
    // No reset needed: events are one-shot. The proactor's event wait operation
    // drains the underlying primitive (eventfd/pipe) as part of completion, so
    // events are always clean when returned to the pool.

    // Clear pool_next to avoid dangling pointers while event is in use.
    event->pool_next = NULL;
    *out_event = event;
  }
  return status;
}

IREE_API_EXPORT void iree_async_event_pool_release(
    iree_async_event_pool_t* pool, iree_async_event_t* event) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(event);

  // Lock-free push to return stack (Treiber stack).
  // This is the critical path for the polling thread - must be fast.
  //
  // Use intptr_t for the expected value to avoid strict aliasing violations.
  // The CAS atomically updates head if it still equals expected, otherwise
  // reloads expected with the current value for retry.
  intptr_t expected;
  do {
    expected =
        iree_atomic_load(&pool->return_stack.head, iree_memory_order_relaxed);
    event->pool_next = (iree_async_event_t*)expected;
  } while (!iree_atomic_compare_exchange_weak(
      &pool->return_stack.head, &expected, (intptr_t)event,
      iree_memory_order_release, iree_memory_order_relaxed));
}
