// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off: must be included before all other headers.
#include "iree/base/internal/wait_handle_impl.h"
// clang-format on

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/target_platform.h"

// This implementation uses iree_notification_t - backed by a futex in most
// cases - to simulate system wait handles. When using a single handle such as
// an iree_event_t and waiting on it with iree_wait_one things behave just as
// the base iree_notification_t: threads can block and wait for the event to
// be signaled. Multi-wait, however, requires some trickery as we need to be
// able to wake when one or more events are signaled and unfortunately there are
// no multi-wait futex APIs. To get around this we have a shared notification
// that is posted every time an event is signaled and multi-waits await that.
// This can lead to spurious wakes when under heavy load as disparate events may
// wake unrelated multi-waiters, however by design in IREE we tend to avoid that
// and centralize waits via things like the task system poller such that this
// isn't so bad. The cases that are likely to suffer are heavy multi-tenant
// workloads in the same process but those should be using a real wait handle
// implementation instead of this bare-metal friendly one anyway.
#if IREE_WAIT_API == IREE_WAIT_API_INPROC

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

typedef struct iree_futex_handle_t {
  iree_atomic_int64_t value;
  iree_notification_t notification;
} iree_futex_handle_t;

static bool iree_wait_primitive_compare_identical(iree_wait_handle_t* lhs,
                                                  iree_wait_handle_t* rhs) {
  return lhs->type == rhs->type &&
         memcmp(&lhs->value, &rhs->value, sizeof(lhs->value)) == 0;
}

void iree_wait_handle_close(iree_wait_handle_t* handle) {
  switch (handle->type) {
#if defined(IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX)
    case IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX: {
      iree_futex_handle_t* futex =
          (iree_futex_handle_t*)handle->value.local_futex;
      iree_notification_deinitialize(&futex->notification);
      iree_allocator_free(iree_allocator_system(), futex);
      break;
    }
#endif  // IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX
    default:
      break;
  }
  iree_wait_handle_deinitialize(handle);
}

//===----------------------------------------------------------------------===//
// Multi-wait emulation
//===----------------------------------------------------------------------===//

// Returns a notification that is shared with all waiters in the process.
// Waiting on the notification will cause a wake whenever any event is set.
static iree_notification_t* iree_wait_multi_notification(void) {
  static iree_notification_t shared_notification = IREE_NOTIFICATION_INIT;
  return &shared_notification;
}

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

struct iree_wait_set_t {
  iree_allocator_t allocator;

  // Total capacity of handles in the set (including duplicates).
  // This defines the capacity of handles to ensure that we don't get insanely
  // hard to debug behavioral differences when some handles happen to be
  // duplicates vs all being unique.
  //
  // If you added 1000 duplicate handles to the set you'd need a capacity
  // of 1000 even though handle_count (expluding duplicates) would be 1.
  iree_host_size_t capacity;

  // Total number of handles in the set (including duplicates).
  // We use this to ensure that we provide consistent capacity errors;
  iree_host_size_t total_handle_count;

  // Number of handles in the set (excluding duplicates), defining the valid
  // size of the dense handles list.
  iree_host_size_t handle_count;

  // De-duped user-provided handles. iree_wait_handle_t::set_internal.dupe_count
  // is used to indicate how many additional duplicates there are of a
  // particular handle. For example, dupe_count=0 means that there are no
  // duplicates.
  iree_wait_handle_t handles[];
};

iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set) {
  // Be reasonable; 64K objects is too high.
  if (capacity >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wait set capacity of %zu is unreasonably large",
                            capacity);
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE(z0, (int64_t)capacity);
  *out_set = NULL;

  iree_wait_set_t* set = NULL;
  iree_status_t status = iree_allocator_malloc(
      allocator, sizeof(*set) + capacity * sizeof(iree_wait_handle_t),
      (void**)&set);
  if (iree_status_is_ok(status)) {
    set->allocator = allocator;
    set->capacity = capacity;
    iree_wait_set_clear(set);
  }

  *out_set = set;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_wait_set_free(iree_wait_set_t* set) {
  if (!set) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_t allocator = set->allocator;
  iree_allocator_free(allocator, set);
  IREE_TRACE_ZONE_END(z0);
}

bool iree_wait_set_is_empty(const iree_wait_set_t* set) {
  return set->handle_count != 0;
}

iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle) {
  if (set->total_handle_count + 1 > set->capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "wait set capacity %" PRIhsz
                            " reached; no more wait handles available",
                            set->capacity);
  } else if (handle.type != IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unimplemented primitive type %d (expected LOCAL_FUTEX)",
        (int)handle.type);
  }

  // First check to see if we already have the handle in the set; most native
  // system APIs don't allow duplicates so we match that behavior here to be
  // consistent. It also helps in cases where the same event is waited on
  // multiple times (such as when joining on a semaphore) as they can be routed
  // to the much more efficient iree_wait_one.
  for (iree_host_size_t i = 0; i < set->handle_count; ++i) {
    iree_wait_handle_t* existing_handle = &set->handles[i];
    if (iree_wait_primitive_compare_identical(existing_handle, &handle)) {
      // Handle already exists in the set; just increment the reference count.
      ++existing_handle->set_internal.dupe_count;
      ++set->total_handle_count;
      return iree_ok_status();
    }
  }

  ++set->total_handle_count;
  iree_host_size_t index = set->handle_count++;
  iree_wait_handle_t* stored_handle = &set->handles[index];
  iree_wait_handle_wrap_primitive(handle.type, handle.value, stored_handle);
  stored_handle->set_internal.dupe_count = 0;  // just us so far

  return iree_ok_status();
}

void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle) {
  // Find the user handle in the set. This either requires a linear scan to
  // find the matching user handle or - if valid - we can use the native index
  // set after an iree_wait_any wake to do a quick lookup.
  iree_host_size_t index = handle.set_internal.index;
  if (IREE_UNLIKELY(index >= set->handle_count) ||
      IREE_UNLIKELY(!iree_wait_primitive_compare_identical(&set->handles[index],
                                                           &handle))) {
    // Fallback to a linear scan of (hopefully) a small list.
    for (iree_host_size_t i = 0; i < set->handle_count; ++i) {
      if (iree_wait_primitive_compare_identical(&set->handles[i], &handle)) {
        index = i;
        break;
      }
    }
  }

  // Decrement reference count.
  iree_wait_handle_t* existing_handle = &set->handles[index];
  if (existing_handle->set_internal.dupe_count-- > 0) {
    // Still one or more remaining in the set; leave it in the handle list.
    --set->total_handle_count;
    return;
  }

  // No more references remaining; remove from both handle lists.
  // Since we make no guarantees about the order of the lists we can just swap
  // with the last value.
  int tail_index = (int)set->handle_count - 1;
  if (tail_index > index) {
    memcpy(&set->handles[index], &set->handles[tail_index],
           sizeof(*set->handles));
  }
  --set->total_handle_count;
  --set->handle_count;
}

void iree_wait_set_clear(iree_wait_set_t* set) {
  memset(&set->handles[0], 0, set->handle_count * sizeof(iree_wait_handle_t));
  set->total_handle_count = 0;
  set->handle_count = 0;
}

typedef struct {
  iree_wait_set_t* set;
  iree_wait_handle_t* wake_handle;  // if set then wait-any
} iree_wait_set_check_params_t;

static bool iree_wait_set_check(const iree_wait_set_check_params_t* params) {
  iree_host_size_t ready_count = 0;
  for (iree_host_size_t i = 0; i < params->set->handle_count; ++i) {
    iree_wait_handle_t* wait_handle = &params->set->handles[i];
    iree_futex_handle_t* futex =
        (iree_futex_handle_t*)wait_handle->value.local_futex;
    if (iree_atomic_load_int64(&futex->value, iree_memory_order_acquire) != 0) {
      ++ready_count;
      if (params->wake_handle) {
        *params->wake_handle = *wait_handle;
        return true;
      }
    }
  }
  return ready_count == params->set->handle_count;
}

static iree_status_t iree_wait_multi(iree_wait_set_t* set,
                                     iree_time_t deadline_ns,
                                     iree_wait_handle_t* out_wake_handle) {
  if (set->handle_count == 0) return iree_ok_status();  // no-op
  if (set->handle_count == 1) {
    // It's much more efficient to use a wait-one as then we will only wake if
    // the specific handle is signaled; otherwise we will use the multi-wait
    // notification and potentially wake many times.
    return iree_wait_one(&set->handles[0], deadline_ns);
  }

  iree_wait_set_check_params_t params = {
      .set = set,
      .wake_handle = out_wake_handle,
  };
  if (!iree_notification_await(iree_wait_multi_notification(),
                               (iree_condition_fn_t)iree_wait_set_check,
                               &params, iree_make_deadline(deadline_ns))) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return iree_ok_status();
}

iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = iree_wait_multi(set, deadline_ns,
                                         /*out_wake_handle=*/NULL);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_wake_handle, 0, sizeof(*out_wake_handle));
  iree_status_t status = iree_wait_multi(set, deadline_ns, out_wake_handle);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static bool iree_futex_handle_check(iree_futex_handle_t* futex) {
  return iree_atomic_load_int64(&futex->value, iree_memory_order_acquire) != 0;
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  if (handle->type == IREE_WAIT_PRIMITIVE_TYPE_NONE) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  if (handle->type == IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX) {
    iree_futex_handle_t* futex =
        (iree_futex_handle_t*)handle->value.local_futex;
    if (!iree_notification_await(&futex->notification,
                                 (iree_condition_fn_t)iree_futex_handle_check,
                                 futex, iree_make_deadline(deadline_ns))) {
      status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
  } else {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unhandled primitive type");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_event, 0, sizeof(*out_event));

  iree_futex_handle_t* futex = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               sizeof(*futex), (void**)&futex);
  if (iree_status_is_ok(status)) {
    out_event->type = IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX;
    out_event->value.local_futex = (void*)futex;
    iree_atomic_store_int64(&futex->value, initial_state ? 1 : 0,
                            iree_memory_order_release);
    iree_notification_initialize(&futex->notification);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_event_deinitialize(iree_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_wait_handle_close(event);
  IREE_TRACE_ZONE_END(z0);
}

void iree_event_set(iree_event_t* event) {
  if (!event) return;
  iree_futex_handle_t* futex = (iree_futex_handle_t*)event->value.local_futex;
  if (!futex) return;

  // Try to transition from unset -> set.
  // No-op if already set and otherwise we successfully signaled the event and
  // need to notify all waiters.
  if (iree_atomic_exchange_int64(&futex->value, 1, iree_memory_order_release) ==
      0) {
    // Notify those waiting on just this event.
    iree_notification_post(&futex->notification, IREE_ALL_WAITERS);
    // Notify any multi-waits that may have this event as part of their set.
    iree_notification_post(iree_wait_multi_notification(), IREE_ALL_WAITERS);
  }
}

void iree_event_reset(iree_event_t* event) {
  if (!event) return;
  iree_futex_handle_t* futex = (iree_futex_handle_t*)event->value.local_futex;
  if (!futex) return;
  iree_atomic_store_int64(&futex->value, 0, iree_memory_order_release);
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_INPROC
