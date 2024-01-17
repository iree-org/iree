// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/event_pool.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_event_t
//===----------------------------------------------------------------------===//

struct iree_hal_cuda2_event_t {
  // A reference count used to manage resource lifetime. Its value range:
  // * 1 - when inside the event pool and to be acquired;
  // * >= 1 - when acquired outside of the event pool;
  // * 0 - when before releasing back to the pool or destruction.
  iree_atomic_ref_count_t ref_count;

  // The allocator used to create the event.
  iree_allocator_t host_allocator;

  // The symbols used to create and destroy CUevent objects.
  const iree_hal_event_impl_symtable_t* symbols;
  // User data to the symbol table functions.
  void* symbol_user_data;

  // The event pool that owns this event. This cannot be NULL. We retain it to
  // make sure the event outlive the pool.
  iree_hal_cuda2_event_pool_t* pool;
  // The underlying CUevent object.
  iree_hal_event_impl_t event_impl;
};

iree_hal_event_impl_t iree_hal_cuda2_event_handle(
    const iree_hal_cuda2_event_t* event) {
  return event->event_impl;
}

static inline void iree_hal_cuda2_event_destroy(iree_hal_cuda2_event_t* event) {
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT_REF_COUNT_ZERO(&event->ref_count);
  event->symbols->destroy(event->symbol_user_data, event->event_impl);
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

static inline iree_status_t iree_hal_cuda2_event_create(
    const iree_hal_event_impl_symtable_t* symbols, void* symbol_user_data,
    iree_hal_cuda2_event_pool_t* pool, iree_allocator_t host_allocator,
    iree_hal_cuda2_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(symbol_user_data);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));
  iree_atomic_ref_count_init(&event->ref_count);  // -> 1
  event->host_allocator = host_allocator;
  event->symbols = symbols;
  event->symbol_user_data = symbol_user_data;
  event->pool = pool;
  event->event_impl = NULL;

  iree_status_t status =
      event->symbols->create(event->symbol_user_data, &event->event_impl);
  if (iree_status_is_ok(status)) {
    *out_event = event;
  } else {
    iree_atomic_ref_count_dec(&event->ref_count);  // -> 0
    iree_hal_cuda2_event_destroy(event);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda2_event_retain(iree_hal_cuda2_event_t* event) {
  iree_atomic_ref_count_inc(&event->ref_count);
}

static void iree_hal_cuda2_event_pool_release_event(
    iree_hal_cuda2_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_cuda2_event_t** events);

void iree_hal_cuda2_event_release(iree_hal_cuda2_event_t* event) {
  if (iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    iree_hal_cuda2_event_pool_t* pool = event->pool;
    // Release back to the pool if the reference count becomes 0.
    iree_hal_cuda2_event_pool_release_event(pool, 1, &event);
    // Drop our reference to the pool itself when we return event to it.
    iree_hal_cuda2_event_pool_release(pool);  // -1
  }
}

//===----------------------------------------------------------------------===//
// iree_hal_cuda2_event_pool_t
//===----------------------------------------------------------------------===//

struct iree_hal_cuda2_event_pool_t {
  // A reference count used to manage resource lifetime.
  iree_atomic_ref_count_t ref_count;

  // The allocator used to create the event pool.
  iree_allocator_t host_allocator;

  // The symbols used to create and destroy CUevent objects.
  const iree_hal_event_impl_symtable_t* symbols;
  // User data to the symbol table functions.
  void* symbol_user_data;

  // Guards event related fields in the pool. We don't expect a performant
  // program to frequently allocate events for synchronization purposes; the
  // traffic to this pool should be low. So it should be fine to use mutex to
  // guard here.
  iree_slim_mutex_t event_mutex;

  // Maximum number of event objects that will be maintained in the pool.
  // More events may be allocated at any time, but they will be disposed
  // directly when they are no longer needed.
  iree_host_size_t available_capacity IREE_GUARDED_BY(event_mutex);
  // Total number of currently available event objects.
  iree_host_size_t available_count IREE_GUARDED_BY(event_mutex);
  // The list of available_count event objects.
  iree_hal_cuda2_event_t* available_list[] IREE_GUARDED_BY(event_mutex);
};
// + Additional inline allocation for holding events up to the capacity.

static void iree_hal_cuda2_event_pool_free(
    iree_hal_cuda2_event_pool_t* event_pool);

iree_status_t iree_hal_cuda2_event_pool_allocate(
    const iree_hal_event_impl_symtable_t* symbols, void* symbol_user_data,
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    iree_hal_cuda2_event_pool_t** out_event_pool) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(symbol_user_data);
  IREE_ASSERT_ARGUMENT(out_event_pool);
  *out_event_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda2_event_pool_t* event_pool = NULL;
  iree_host_size_t total_size =
      sizeof(*event_pool) +
      available_capacity * sizeof(*event_pool->available_list);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&event_pool));
  iree_atomic_ref_count_init(&event_pool->ref_count);  // -> 1
  event_pool->host_allocator = host_allocator;
  event_pool->symbols = symbols;
  event_pool->symbol_user_data = symbol_user_data;
  iree_slim_mutex_initialize(&event_pool->event_mutex);
  event_pool->available_capacity = available_capacity;
  event_pool->available_count = 0;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < available_capacity; ++i) {
    status = iree_hal_cuda2_event_create(
        event_pool->symbols, symbol_user_data, event_pool, host_allocator,
        &event_pool->available_list[event_pool->available_count++]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_event_pool = event_pool;
  } else {
    iree_hal_cuda2_event_pool_free(event_pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda2_event_pool_free(
    iree_hal_cuda2_event_pool_t* event_pool) {
  iree_allocator_t host_allocator = event_pool->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < event_pool->available_count; ++i) {
    iree_hal_cuda2_event_t* event = event_pool->available_list[i];
    iree_atomic_ref_count_dec(&event->ref_count);  // -> 0
    iree_hal_cuda2_event_destroy(event);
  }
  IREE_ASSERT_REF_COUNT_ZERO(&event_pool->ref_count);

  iree_slim_mutex_deinitialize(&event_pool->event_mutex);
  iree_allocator_free(host_allocator, event_pool);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_cuda2_event_pool_retain(iree_hal_cuda2_event_pool_t* event_pool) {
  iree_atomic_ref_count_inc(&event_pool->ref_count);
}

void iree_hal_cuda2_event_pool_release(
    iree_hal_cuda2_event_pool_t* event_pool) {
  if (iree_atomic_ref_count_dec(&event_pool->ref_count) == 1) {
    iree_hal_cuda2_event_pool_free(event_pool);
  }
}

iree_status_t iree_hal_cuda2_event_pool_acquire(
    iree_hal_cuda2_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_cuda2_event_t** out_events) {
  IREE_ASSERT_ARGUMENT(event_pool);
  if (!event_count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(out_events);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We'll try to get what we can from the pool and fall back to initializing
  // new iree_hal_cuda2_event_t objects.
  iree_host_size_t remaining_count = event_count;

  // Try first to grab from the pool.
  iree_slim_mutex_lock(&event_pool->event_mutex);
  iree_host_size_t from_pool_count =
      iree_min(event_pool->available_count, event_count);
  if (from_pool_count > 0) {
    iree_host_size_t pool_base_index =
        event_pool->available_count - from_pool_count;
    memcpy(out_events, &event_pool->available_list[pool_base_index],
           from_pool_count * sizeof(*event_pool->available_list));
    event_pool->available_count -= from_pool_count;
    remaining_count -= from_pool_count;
  }
  iree_slim_mutex_unlock(&event_pool->event_mutex);

  // Allocate the rest of the events.
  if (remaining_count > 0) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "event-pool-unpooled-acquire");
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      status = iree_hal_cuda2_event_create(
          event_pool->symbols, event_pool->symbol_user_data, event_pool,
          event_pool->host_allocator, &out_events[from_pool_count + i]);
      if (!iree_status_is_ok(status)) {
        // Must release all events we've acquired so far.
        iree_hal_cuda2_event_pool_release_event(event_pool, from_pool_count + i,
                                                out_events);
        IREE_TRACE_ZONE_END(z1);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
    IREE_TRACE_ZONE_END(z1);
  }

  // Retain a reference to a pool when we pass event to the caller. When the
  // caller returns event back to the pool they'll release the reference.
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    iree_hal_cuda2_event_pool_retain(out_events[i]->pool);  // +1
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_cuda2_event_pool_release_event(
    iree_hal_cuda2_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_cuda2_event_t** events) {
  IREE_ASSERT_ARGUMENT(event_pool);
  if (!event_count) return;
  IREE_ASSERT_ARGUMENT(events);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We'll try to release all we can back to the pool and then deinitialize
  // the ones that won't fit.
  iree_host_size_t remaining_count = event_count;

  // Try first to release to the pool.
  iree_slim_mutex_lock(&event_pool->event_mutex);
  iree_host_size_t to_pool_count =
      iree_min(event_pool->available_capacity - event_pool->available_count,
               event_count);
  if (to_pool_count > 0) {
    for (iree_host_size_t i = 0; i < to_pool_count; ++i) {
      IREE_ASSERT_REF_COUNT_ZERO(&events[i]->ref_count);
      iree_hal_cuda2_event_retain(events[i]);  // -> 1
    }
    iree_host_size_t pool_base_index = event_pool->available_count;
    memcpy(&event_pool->available_list[pool_base_index], events,
           to_pool_count * sizeof(*event_pool->available_list));
    event_pool->available_count += to_pool_count;
    remaining_count -= to_pool_count;
  }
  iree_slim_mutex_unlock(&event_pool->event_mutex);

  // Deallocate the rest of the events. We don't bother resetting them as we are
  // getting rid of them.
  if (remaining_count > 0) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "event-pool-unpooled-release");
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      iree_hal_cuda2_event_destroy(events[to_pool_count + i]);
    }
    IREE_TRACE_ZONE_END(z1);
  }
  IREE_TRACE_ZONE_END(z0);
}
