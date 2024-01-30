// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/cuda/timepoint_pool.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/event_pool.h"
#include "iree/base/internal/synchronization.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/cuda_dynamic_symbols.h"
#include "iree/hal/drivers/cuda/cuda_status_util.h"
#include "iree/hal/drivers/cuda/event_pool.h"
#include "iree/hal/utils/semaphore_base.h"

//===----------------------------------------------------------------------===//
// iree_hal_cuda_timepoint_t
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_cuda_timepoint_allocate(
    iree_hal_cuda_timepoint_pool_t* pool, iree_allocator_t host_allocator,
    iree_hal_cuda_timepoint_t** out_timepoint) {
  IREE_ASSERT_ARGUMENT(pool);
  IREE_ASSERT_ARGUMENT(out_timepoint);
  *out_timepoint = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_timepoint_t* timepoint = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*timepoint),
                                (void**)&timepoint));
  // iree_allocator_malloc zeros out the whole struct.
  timepoint->host_allocator = host_allocator;
  timepoint->pool = pool;

  *out_timepoint = timepoint;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Clears all data fields in the given |timepoint| except the original host
// allocator and owning pool.
static void iree_hal_cuda_timepoint_clear(
    iree_hal_cuda_timepoint_t* timepoint) {
  iree_allocator_t host_allocator = timepoint->host_allocator;
  iree_hal_cuda_timepoint_pool_t* pool = timepoint->pool;
  memset(timepoint, 0, sizeof(*timepoint));
  timepoint->host_allocator = host_allocator;
  timepoint->pool = pool;
}

static void iree_hal_cuda_timepoint_free(iree_hal_cuda_timepoint_t* timepoint) {
  iree_allocator_t host_allocator = timepoint->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT(timepoint->kind == IREE_HAL_CUDA_TIMEPOINT_KIND_NONE);
  iree_allocator_free(host_allocator, timepoint);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// iree_hal_cuda_timepoint_pool_t
//===----------------------------------------------------------------------===//

struct iree_hal_cuda_timepoint_pool_t {
  // The allocator used to create the timepoint pool.
  iree_allocator_t host_allocator;

  // The pool to acquire host events.
  iree_event_pool_t* host_event_pool;
  // The pool to acquire device events. Internally synchronized.
  iree_hal_cuda_event_pool_t* device_event_pool;

  // Note that the above pools are internally synchronized; so we don't and
  // shouldn't use the following mutex to guard access to them.

  // Guards timepoint related fields this pool. We don't expect a performant
  // program to frequently allocate timepoints for synchronization purposes; the
  // traffic to this pool should be low. So it should be fine to use mutex to
  // guard here.
  iree_slim_mutex_t timepoint_mutex;

  // Maximum number of timepoint objects that will be maintained in the pool.
  // More timepoints may be allocated at any time, but they will be disposed
  // directly when they are no longer needed.
  iree_host_size_t available_capacity IREE_GUARDED_BY(timepoint_mutex);
  // Total number of currently available timepoint objects.
  iree_host_size_t available_count IREE_GUARDED_BY(timepoint_mutex);
  // The list of available_count timepoint objects.
  iree_hal_cuda_timepoint_t* available_list[] IREE_GUARDED_BY(timepoint_mutex);
};
// + Additional inline allocation for holding timepoints up to the capacity.

iree_status_t iree_hal_cuda_timepoint_pool_allocate(
    iree_event_pool_t* host_event_pool,
    iree_hal_cuda_event_pool_t* device_event_pool,
    iree_host_size_t available_capacity, iree_allocator_t host_allocator,
    iree_hal_cuda_timepoint_pool_t** out_timepoint_pool) {
  IREE_ASSERT_ARGUMENT(host_event_pool);
  IREE_ASSERT_ARGUMENT(device_event_pool);
  IREE_ASSERT_ARGUMENT(out_timepoint_pool);
  *out_timepoint_pool = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_timepoint_pool_t* timepoint_pool = NULL;
  iree_host_size_t total_size =
      sizeof(*timepoint_pool) +
      available_capacity * sizeof(*timepoint_pool->available_list);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size,
                                (void**)&timepoint_pool));
  timepoint_pool->host_allocator = host_allocator;
  timepoint_pool->host_event_pool = host_event_pool;
  timepoint_pool->device_event_pool = device_event_pool;

  iree_slim_mutex_initialize(&timepoint_pool->timepoint_mutex);
  timepoint_pool->available_capacity = available_capacity;
  timepoint_pool->available_count = 0;

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < available_capacity; ++i) {
    status = iree_hal_cuda_timepoint_allocate(
        timepoint_pool, host_allocator,
        &timepoint_pool->available_list[timepoint_pool->available_count++]);
    if (!iree_status_is_ok(status)) break;
  }

  if (iree_status_is_ok(status)) {
    *out_timepoint_pool = timepoint_pool;
  } else {
    iree_hal_cuda_timepoint_pool_free(timepoint_pool);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_cuda_timepoint_pool_free(
    iree_hal_cuda_timepoint_pool_t* timepoint_pool) {
  iree_allocator_t host_allocator = timepoint_pool->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < timepoint_pool->available_count; ++i) {
    iree_hal_cuda_timepoint_free(timepoint_pool->available_list[i]);
  }
  iree_slim_mutex_deinitialize(&timepoint_pool->timepoint_mutex);
  iree_allocator_free(host_allocator, timepoint_pool);

  IREE_TRACE_ZONE_END(z0);
}

// Acquires |timepoint_count| timepoints from the given |timepoint_pool|.
// The |out_timepoints| needs to be further initialized with proper kind and
// payload values.
static iree_status_t iree_hal_cuda_timepoint_pool_acquire_internal(
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_cuda_timepoint_t** out_timepoints) {
  IREE_ASSERT_ARGUMENT(timepoint_pool);
  if (!timepoint_count) return iree_ok_status();
  IREE_ASSERT_ARGUMENT(out_timepoints);
  IREE_TRACE_ZONE_BEGIN(z0);

  // We'll try to get what we can from the pool and fall back to initializing
  // new iree_hal_cuda_timepoint_t objects.
  iree_host_size_t remaining_count = timepoint_count;

  // Try first to grab from the pool.
  iree_slim_mutex_lock(&timepoint_pool->timepoint_mutex);
  iree_host_size_t from_pool_count =
      iree_min(timepoint_pool->available_count, timepoint_count);
  if (from_pool_count > 0) {
    iree_host_size_t pool_base_index =
        timepoint_pool->available_count - from_pool_count;
    memcpy(out_timepoints, &timepoint_pool->available_list[pool_base_index],
           from_pool_count * sizeof(*timepoint_pool->available_list));
    timepoint_pool->available_count -= from_pool_count;
    remaining_count -= from_pool_count;
  }
  iree_slim_mutex_unlock(&timepoint_pool->timepoint_mutex);

  // Allocate the rest of the timepoints.
  if (remaining_count > 0) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "timepoint-pool-unpooled-acquire");
    iree_status_t status = iree_ok_status();
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      status = iree_hal_cuda_timepoint_allocate(
          timepoint_pool, timepoint_pool->host_allocator,
          &out_timepoints[from_pool_count + i]);
      if (!iree_status_is_ok(status)) {
        // Must release all timepoints we've acquired so far.
        iree_hal_cuda_timepoint_pool_release(
            timepoint_pool, from_pool_count + i, out_timepoints);
        IREE_TRACE_ZONE_END(z1);
        IREE_TRACE_ZONE_END(z0);
        return status;
      }
    }
    IREE_TRACE_ZONE_END(z1);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_timepoint_pool_acquire_host_wait(
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_cuda_timepoint_t** out_timepoints) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Acquire host events to wrap up. This should happen before acquiring the
  // timepoints to avoid nested locks.
  iree_event_t* host_events = iree_alloca(
      timepoint_count * sizeof((*out_timepoints)->timepoint.host_wait));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_event_pool_acquire(timepoint_pool->host_event_pool,
                                  timepoint_count, host_events));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_timepoint_pool_acquire_internal(
              timepoint_pool, timepoint_count, out_timepoints));
  for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
    out_timepoints[i]->kind = IREE_HAL_CUDA_TIMEPOINT_KIND_HOST_WAIT;
    out_timepoints[i]->timepoint.host_wait = host_events[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_timepoint_pool_acquire_device_signal(
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_cuda_timepoint_t** out_timepoints) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Acquire device events to wrap up. This should happen before acquiring the
  // timepoints to avoid nested locks.
  iree_hal_cuda_event_t** device_events = iree_alloca(
      timepoint_count * sizeof((*out_timepoints)->timepoint.device_signal));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_event_pool_acquire(timepoint_pool->device_event_pool,
                                           timepoint_count, device_events));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_timepoint_pool_acquire_internal(
              timepoint_pool, timepoint_count, out_timepoints));
  for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
    out_timepoints[i]->kind = IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_SIGNAL;
    out_timepoints[i]->timepoint.device_signal = device_events[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_cuda_timepoint_pool_acquire_device_wait(
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count,
    iree_hal_cuda_timepoint_t** out_timepoints) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Acquire device events to wrap up. This should happen before acquiring the
  // timepoints to avoid nested locks.
  iree_hal_cuda_event_t** device_events = iree_alloca(
      timepoint_count * sizeof((*out_timepoints)->timepoint.device_wait));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_event_pool_acquire(timepoint_pool->device_event_pool,
                                           timepoint_count, device_events));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_cuda_timepoint_pool_acquire_internal(
              timepoint_pool, timepoint_count, out_timepoints));
  for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
    out_timepoints[i]->kind = IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_WAIT;
    out_timepoints[i]->timepoint.device_wait = device_events[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_hal_cuda_timepoint_pool_release(
    iree_hal_cuda_timepoint_pool_t* timepoint_pool,
    iree_host_size_t timepoint_count, iree_hal_cuda_timepoint_t** timepoints) {
  IREE_ASSERT_ARGUMENT(timepoint_pool);
  if (!timepoint_count) return;
  IREE_ASSERT_ARGUMENT(timepoints);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Release the wrapped host/device events. This should happen before acquiring
  // the timepoint pool's lock given that the host/device event pool its
  // internal lock too.
  // TODO: Release in batch to avoid lock overhead from separate calls.
  for (iree_host_size_t i = 0; i < timepoint_count; ++i) {
    switch (timepoints[i]->kind) {
      case IREE_HAL_CUDA_TIMEPOINT_KIND_HOST_WAIT:
        iree_event_pool_release(timepoint_pool->host_event_pool, 1,
                                &timepoints[i]->timepoint.host_wait);
        break;
      case IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_SIGNAL:
        iree_hal_cuda_event_release(timepoints[i]->timepoint.device_signal);
        break;
      case IREE_HAL_CUDA_TIMEPOINT_KIND_DEVICE_WAIT:
        iree_hal_cuda_event_release(timepoints[i]->timepoint.device_wait);
        break;
      default:
        break;
    }
  }

  // We'll try to release all we can back to the pool and then deinitialize
  // the ones that won't fit.
  iree_host_size_t remaining_count = timepoint_count;

  // Try first to release to the pool.
  iree_slim_mutex_lock(&timepoint_pool->timepoint_mutex);
  iree_host_size_t to_pool_count = iree_min(
      timepoint_pool->available_capacity - timepoint_pool->available_count,
      timepoint_count);
  if (to_pool_count > 0) {
    for (iree_host_size_t i = 0; i < to_pool_count; ++i) {
      iree_hal_cuda_timepoint_clear(timepoints[i]);
    }
    iree_host_size_t pool_base_index = timepoint_pool->available_count;
    memcpy(&timepoint_pool->available_list[pool_base_index], timepoints,
           to_pool_count * sizeof(*timepoint_pool->available_list));
    timepoint_pool->available_count += to_pool_count;
    remaining_count -= to_pool_count;
  }
  iree_slim_mutex_unlock(&timepoint_pool->timepoint_mutex);

  // Deallocate the rest of the timepoints. We don't bother resetting them as we
  // are getting rid of them.
  if (remaining_count > 0) {
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "timepoint-pool-unpooled-release");
    for (iree_host_size_t i = 0; i < remaining_count; ++i) {
      iree_hal_cuda_timepoint_clear(timepoints[to_pool_count + i]);
      iree_hal_cuda_timepoint_free(timepoints[to_pool_count + i]);
    }
    IREE_TRACE_ZONE_END(z1);
  }
  IREE_TRACE_ZONE_END(z0);
}
