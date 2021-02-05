// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/hal/local/event_pool.h"

#include "iree/base/internal/debugging.h"
#include "iree/base/synchronization.h"
#include "iree/base/tracing.h"

struct iree_hal_local_event_pool_s {
  // Allocator used to create the event pool.
  iree_allocator_t host_allocator;
  // Guards the pool. Since this pool is used to get operating system-level
  // event objects that will be signaled and waited on using syscalls it's got
  // relatively low contention: callers are rate limited by how fast they can
  // signal and wait on the events they get.
  iree_slim_mutex_t mutex;
  // Maximum number of events that will be maintained in the pool. More events
  // may be allocated at any time but when they are no longer needed they will
  // be disposed directly.
  iree_host_size_t available_capacity;
  // Total number of available
  iree_host_size_t available_count;
  // Dense left-aligned list of available_count events.
  iree_event_t available_list[];
};

iree_status_t iree_hal_local_event_pool_allocate(
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    iree_hal_local_event_pool_t** out_event_pool) {
  IREE_ASSERT_ARGUMENT(out_event_pool);
  *out_event_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_local_event_pool_t* event_pool = NULL;
  iree_host_size_t total_size =
      sizeof(*event_pool) +
      available_capacity * sizeof(event_pool->available_list[0]);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&event_pool));
  event_pool->host_allocator = host_allocator;
  event_pool->available_capacity = available_capacity;
  event_pool->available_count = 0;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < available_capacity; ++i) {
    status = iree_event_initialize(
        /*initial_state=*/false,
        &event_pool->available_list[event_pool->available_count++]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_event_pool = event_pool;
  } else {
    iree_hal_local_event_pool_free(event_pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_local_event_pool_free(iree_hal_local_event_pool_t* event_pool) {
  iree_allocator_t host_allocator = event_pool->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < event_pool->available_count; ++i) {
    iree_event_deinitialize(&event_pool->available_list[i]);
  }
  iree_slim_mutex_deinitialize(&event_pool->mutex);
  iree_allocator_free(host_allocator, event_pool);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_local_event_pool_acquire(
    iree_hal_local_event_pool_t* event_pool, iree_host_size_t event_count,
    iree_event_t* out_events) {
  IREE_ASSERT_ARGUMENT(event_pool);
  if (!event_count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(out_events);

  // We'll try to get what we can from the pool and fall back to initializing
  // new events.
  iree_host_size_t remaining_count = event_count;

  // Try first to grab from the pool.
  iree_slim_mutex_lock(&event_pool->mutex);
  iree_host_size_t from_pool_count =
      iree_min(event_pool->available_count, event_count);
  if (from_pool_count > 0) {
    iree_host_size_t pool_base_index =
        event_pool->available_count - from_pool_count;
    memcpy(out_events, &event_pool->available_list[pool_base_index],
           from_pool_count * sizeof(iree_event_t));
    event_pool->available_count -= from_pool_count;
    remaining_count -= from_pool_count;
  }
  iree_slim_mutex_unlock(&event_pool->mutex);

  // Allocate the rest of the events.
  if (remaining_count > 0) {
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      status = iree_event_initialize(/*initial_state=*/false,
                                     &out_events[from_pool_count + i]);
      if (!iree_status_is_ok(status)) {
        // Must release all events we've acquired so far.
        iree_hal_local_event_pool_release(event_pool, from_pool_count + i,
                                          out_events);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
    IREE_TRACE_ZONE_END(z0);
  }

  return iree_ok_status();
}

void iree_hal_local_event_pool_release(iree_hal_local_event_pool_t* event_pool,
                                       iree_host_size_t event_count,
                                       iree_event_t* events) {
  IREE_ASSERT_ARGUMENT(event_pool);
  if (!event_count) return;
  IREE_ASSERT_ARGUMENT(events);

  // We'll try to release all we can back to the pool and then deinitialize
  // the ones that won't fit.
  iree_host_size_t remaining_count = event_count;

  // Try first to release to the pool.
  // Note that we reset the events we add back to the pool so that they are
  // ready to be acquired again.
  iree_slim_mutex_lock(&event_pool->mutex);
  iree_host_size_t to_pool_count =
      iree_min(event_pool->available_capacity - event_pool->available_count,
               event_count);
  if (to_pool_count > 0) {
    iree_host_size_t pool_base_index = event_pool->available_count;
    for (iree_host_size_t i = 0; i < to_pool_count; ++i) {
      iree_event_reset(&events[i]);
    }
    memcpy(&event_pool->available_list[pool_base_index], events,
           to_pool_count * sizeof(iree_event_t));
    event_pool->available_count += to_pool_count;
    remaining_count -= to_pool_count;
  }
  iree_slim_mutex_unlock(&event_pool->mutex);

  // Deallocate the rest of the events. We don't bother resetting them as we are
  // getting rid of them.
  if (remaining_count > 0) {
    IREE_TRACE_ZONE_BEGIN(z0);
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      iree_event_deinitialize(&events[to_pool_count + i]);
    }
    IREE_TRACE_ZONE_END(z0);
  }
}
