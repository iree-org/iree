// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/event_pool.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/hip/context_util.h"
#include "iree/hal/drivers/hip/dynamic_symbols.h"
#include "iree/hal/drivers/hip/status_util.h"

//===----------------------------------------------------------------------===//
// iree_hal_hip_event_t
//===----------------------------------------------------------------------===//

struct iree_hal_hip_event_t {
  // A reference count used to manage resource lifetime. Its value range:
  // * 1 - when inside the event pool and to be acquired;
  // * >= 1 - when acquired outside of the event pool;
  // * 0 - when before releasing back to the pool or destruction.
  iree_atomic_ref_count_t ref_count;

  // The allocator used to create the event.
  iree_allocator_t host_allocator;
  // The symbols used to create and destroy hipEvent_t objects.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  // This event was imported, and should be disposed instead of
  // pooled.
  bool imported;

  // This event has been exported, therefore we should drop it
  // instead of re-adding it to the pool. We should not delete
  // it.
  bool exported;

  // The event pool that owns this event. This cannot be NULL. We retain it to
  // make sure the event outlive the pool.
  iree_hal_hip_event_pool_t* pool;
  // The underlying hipEvent_t object.
  hipEvent_t hip_event;
};

hipEvent_t iree_hal_hip_event_handle(const iree_hal_hip_event_t* event) {
  return event->hip_event;
}

static inline void iree_hal_hip_event_destroy(iree_hal_hip_event_t* event) {
  iree_allocator_t host_allocator = event->host_allocator;
  const iree_hal_hip_dynamic_symbols_t* symbols = event->symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT_REF_COUNT_ZERO(&event->ref_count);
  // If this event was exported, then we do not want to destroy it
  // as ownership is transferred out.
  if (!event->exported) {
    IREE_HIP_IGNORE_ERROR(symbols, hipEventDestroy(event->hip_event));
  }
  iree_allocator_free(host_allocator, event);

  IREE_TRACE_ZONE_END(z0);
}

static inline iree_status_t iree_hal_hip_event_create(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_hal_hip_event_pool_t* pool, iree_allocator_t host_allocator,
    hipEvent_t imported_event, iree_hal_hip_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, sizeof(*event), (void**)&event));
  iree_atomic_ref_count_init(&event->ref_count);  // -> 1
  event->host_allocator = host_allocator;
  event->symbols = symbols;
  event->pool = pool;
  event->hip_event = imported_event;
  event->exported = false;

  iree_status_t status = iree_ok_status();

  if (event->hip_event) {
    event->imported = true;
  } else {
    status = IREE_HIP_CALL_TO_STATUS(
        symbols,
        hipEventCreateWithFlags(&event->hip_event, hipEventDisableTiming),
        "hipEventCreateWithFlags");
  }
  if (iree_status_is_ok(status)) {
    *out_event = event;
  } else {
    iree_atomic_ref_count_dec(&event->ref_count);  // -> 0
    iree_hal_hip_event_destroy(event);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hip_event_retain(iree_hal_hip_event_t* event) {
  iree_atomic_ref_count_inc(&event->ref_count);
}

static void iree_hal_hip_event_pool_release_event(
    iree_hal_hip_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_hip_event_t** events);

void iree_hal_hip_event_release(iree_hal_hip_event_t* event) {
  if (!event) {
    return;
  }
  if (iree_atomic_ref_count_dec(&event->ref_count) == 1) {
    iree_hal_hip_event_pool_t* pool = event->pool;
    // Release back to the pool if the reference count becomes 0.
    iree_hal_hip_event_pool_release_event(pool, 1, &event);
    // Drop our reference to the pool itself when we return event to it.
    iree_hal_hip_event_pool_release(pool);  // -1
  }
}

iree_status_t iree_hal_hip_event_export(iree_hal_hip_event_t* event) {
  if (event->imported) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Cannot export an imported event");
  }
  event->exported = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// iree_hal_hip_event_pool_t
//===----------------------------------------------------------------------===//

struct iree_hal_hip_event_pool_t {
  // A reference count used to manage resource lifetime.
  iree_atomic_ref_count_t ref_count;

  // The allocator used to create the event pool.
  iree_allocator_t host_allocator;
  // The symbols used to create and destroy hipEvent_t objects.
  const iree_hal_hip_dynamic_symbols_t* symbols;

  hipCtx_t device_context;

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
  iree_hal_hip_event_t* available_list[] IREE_GUARDED_BY(event_mutex);
};
// + Additional inline allocation for holding events up to the capacity.

static void iree_hal_hip_event_pool_free(iree_hal_hip_event_pool_t* event_pool);

iree_status_t iree_hal_hip_event_pool_allocate(
    const iree_hal_hip_dynamic_symbols_t* symbols,
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    hipCtx_t device_context, iree_hal_hip_event_pool_t** out_event_pool) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(out_event_pool);
  *out_event_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hip_event_pool_t* event_pool = NULL;
  iree_host_size_t total_size =
      sizeof(*event_pool) +
      available_capacity * sizeof(*event_pool->available_list);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&event_pool));
  iree_atomic_ref_count_init(&event_pool->ref_count);  // -> 1
  event_pool->host_allocator = host_allocator;
  event_pool->symbols = symbols;
  iree_slim_mutex_initialize(&event_pool->event_mutex);
  event_pool->available_capacity = available_capacity;
  event_pool->available_count = 0;
  event_pool->device_context = device_context;

  iree_status_t status = iree_hal_hip_set_context(symbols, device_context);
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < available_capacity; ++i) {
      status = iree_hal_hip_event_create(
          symbols, event_pool, host_allocator, NULL,
          &event_pool->available_list[event_pool->available_count++]);
      if (!iree_status_is_ok(status)) break;
    }
  }

  if (iree_status_is_ok(status)) {
    *out_event_pool = event_pool;
  } else {
    iree_hal_hip_event_pool_free(event_pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Imports an external hip_event_t to the pool.
iree_status_t iree_hal_hip_event_pool_import(
    iree_hal_hip_event_pool_t* event_pool, hipEvent_t event,
    iree_hal_hip_event_t** out_event) {
  iree_status_t status =
      iree_hal_hip_event_create(event_pool->symbols, event_pool,
                                event_pool->host_allocator, event, out_event);
  if (iree_status_is_ok(status)) {
    iree_hal_hip_event_pool_retain(event_pool);
  }
  return status;
}

static void iree_hal_hip_event_pool_free(
    iree_hal_hip_event_pool_t* event_pool) {
  iree_allocator_t host_allocator = event_pool->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < event_pool->available_count; ++i) {
    iree_hal_hip_event_t* event = event_pool->available_list[i];
    iree_atomic_ref_count_dec(&event->ref_count);  // -> 0
    iree_hal_hip_event_destroy(event);
  }
  IREE_ASSERT_REF_COUNT_ZERO(&event_pool->ref_count);

  iree_slim_mutex_deinitialize(&event_pool->event_mutex);
  iree_allocator_free(host_allocator, event_pool);

  IREE_TRACE_ZONE_END(z0);
}

void iree_hal_hip_event_pool_retain(iree_hal_hip_event_pool_t* event_pool) {
  iree_atomic_ref_count_inc(&event_pool->ref_count);
}

void iree_hal_hip_event_pool_release(iree_hal_hip_event_pool_t* event_pool) {
  if (!event_pool) {
    return;
  }
  if (iree_atomic_ref_count_dec(&event_pool->ref_count) == 1) {
    iree_hal_hip_event_pool_free(event_pool);
  }
}

iree_status_t iree_hal_hip_event_pool_acquire(
    iree_hal_hip_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_hip_event_t** out_events) {
  IREE_ASSERT_ARGUMENT(event_pool);
  if (!event_count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(out_events);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We'll try to get what we can from the pool and fall back to initializing
  // new iree_hal_hip_event_t objects.
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
    IREE_TRACE_ZONE_APPEND_TEXT(z0, "unpooled acquire");
    IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, (int64_t)remaining_count);

    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hip_set_context(event_pool->symbols,
                                     event_pool->device_context));
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      iree_status_t status = iree_hal_hip_event_create(
          event_pool->symbols, event_pool, event_pool->host_allocator, NULL,
          &out_events[from_pool_count + i]);
      if (!iree_status_is_ok(status)) {
        // Must release all events we've acquired so far.
        iree_hal_hip_event_pool_release_event(event_pool, from_pool_count + i,
                                              out_events);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
  }

  // Retain a reference to a pool when we pass event to the caller. When the
  // caller returns event back to the pool they'll release the reference.
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    iree_hal_hip_event_pool_retain(out_events[i]->pool);  // +1
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hip_event_pool_release_event(
    iree_hal_hip_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_hal_hip_event_t** events) {
  IREE_ASSERT_ARGUMENT(event_pool);
  if (!event_count) return;
  IREE_ASSERT_ARGUMENT(events);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Try first to release to the pool.
  iree_slim_mutex_lock(&event_pool->event_mutex);
  for (iree_host_size_t i = 0; i < event_count; ++i) {
    if (!events[i]->imported && !events[i]->exported &&
        event_pool->available_capacity > event_pool->available_count) {
      iree_hal_hip_event_retain(events[i]);
      event_pool->available_list[event_pool->available_count] = events[i];
      event_pool->available_count += 1;
    } else {
      iree_hal_hip_event_destroy(events[i]);
    }
  }
  iree_slim_mutex_unlock(&event_pool->event_mutex);
  IREE_TRACE_ZONE_END(z0);
}
