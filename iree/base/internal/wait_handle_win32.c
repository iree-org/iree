// Copyright 2020 Google LLC
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

// NOTE: must be first to ensure that we can define settings for all includes.
#include "iree/base/internal/wait_handle_impl.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include "iree/base/tracing.h"

//===----------------------------------------------------------------------===//
// Platform utilities
//===----------------------------------------------------------------------===//

static_assert(
    sizeof(iree_wait_primitive_value_t) == sizeof(HANDLE),
    "win32 HANDLE type must match uintptr size in wait primitive struct");

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

// Clones a wait handle such that both the |source_handle| and new
// |out_target_handle| both reference the same wait primitive. The handle must
// be closed with iree_wait_primitive_close as if it had been created.
static iree_status_t iree_wait_primitive_clone(
    iree_wait_handle_t* source_handle, iree_wait_handle_t* out_target_handle) {
  if (source_handle->type != IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source wait handle must be a win32 HANDLE");
  }

  iree_wait_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  HANDLE process = GetCurrentProcess();
  if (!DuplicateHandle(process, (HANDLE)source_handle->value.win32.handle,
                       process, (LPHANDLE)&value.win32.handle, 0, FALSE,
                       DUPLICATE_SAME_ACCESS)) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "unable to duplicate HANDLE; possibly out of process handles");
  }
  return iree_wait_handle_wrap_primitive(IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE,
                                         value, out_target_handle);
}

// Closes an existing handle that was either created manually or via
// iree_wait_primitive_clone. Must not be called while there are any waiters on
// the handle.
static void iree_wait_primitive_close(iree_wait_handle_t* handle) {
  if (IREE_LIKELY(handle->value.win32.handle != 0)) {
    CloseHandle((HANDLE)handle->value.win32.handle);
  }
  iree_wait_handle_deinitialize(handle);
}

// Returns true if the two handles share the same underlying primitive object.
static bool iree_wait_primitive_compare(const iree_wait_handle_t* lhs,
                                        const iree_wait_handle_t* rhs) {
  if (lhs->type != rhs->type) return false;
  bool handles_match =
      memcmp(&lhs->value, &rhs->value, sizeof(lhs->value)) == 0;
  switch (lhs->type) {
    case IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE:
      // Note that multiple HANDLEs may point at the same underlying object
      // (such as if they have been cloned).
      return handles_match ||
                     CompareObjectHandles((HANDLE)lhs->value.win32.handle,
                                          (HANDLE)rhs->value.win32.handle)
                 ? true
                 : false;
    default:
      return handles_match;
  }
}

// Returns true if the two handles are identical in representation.
// Note that two unique handles may point to the same underlying primitive
// object (such as when they have been cloned); if testing for duplicate
// primitives prefer iree_wait_primitive_compare.
static bool iree_wait_primitive_compare_identical(
    const iree_wait_handle_t* lhs, const iree_wait_handle_t* rhs) {
  return lhs->type == rhs->type &&
         memcmp(&lhs->value, &rhs->value, sizeof(lhs->value)) == 0;
}

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

struct iree_wait_set_s {
  iree_allocator_t allocator;

  // Total capacity of handles in the set (including duplicates).
  // This defines the capacity of user_handles and native_handles and to ensure
  // that we don't get insanely hard to debug behavioral differences when some
  // handles happen to be duplicates we track the total count against this total
  // capacity including duplicates.
  //
  // If you added 1000 duplicate handles to the set you'd need a handle_capacity
  // of 1000 even though handle_count (expluding duplicates) would be 1.
  iree_host_size_t handle_capacity;

  // Total number of handles in the set (including duplicates).
  // We use this to ensure that we provide consistent capacity errors;
  iree_host_size_t total_handle_count;

  // Number of handles in the set (excluding duplicates), defining the valid
  // size of both user_handles and native_handles.
  iree_host_size_t handle_count;

  // De-duped user-provided handles. iree_wait_handle_t::set_internal.dupe_count
  // is used to indicate how many additional duplicates there are of a
  // particular handle. For example, dupe_count=0 means that there are no
  // duplicates.
  iree_wait_handle_t* user_handles;

  // Native list of win32 HANDLE we will pass directly to WFMO.
  // This list may be smaller than the total_handle_count if handles have been
  // deduplicated.
  HANDLE* native_handles;
};

iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set) {
  // Be reasonable; 64 MAXIMUM_WAIT_OBJECTS is low, but 64K objects is too high.
  if (capacity >= UINT16_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wait set capacity of %zu is unreasonably large",
                            capacity);
  }

  iree_host_size_t user_handle_list_size =
      capacity * sizeof(iree_wait_handle_t);
  iree_host_size_t native_handle_list_size = capacity * sizeof(HANDLE);
  iree_host_size_t total_size =
      sizeof(iree_wait_set_t) + user_handle_list_size + native_handle_list_size;

  iree_wait_set_t* set = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, total_size, (void**)&set));
  set->allocator = allocator;
  set->handle_capacity = capacity;
  iree_wait_set_clear(set);

  set->user_handles =
      (iree_wait_handle_t*)((uint8_t*)set + sizeof(iree_wait_set_t));
  set->native_handles =
      (HANDLE*)((uint8_t*)set->user_handles + user_handle_list_size);

  *out_set = set;
  return iree_ok_status();
}

void iree_wait_set_free(iree_wait_set_t* set) {
  iree_allocator_free(set->allocator, set);
}

iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle) {
  if (set->total_handle_count + 1 > set->handle_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "wait set capacity reached");
  }

  // First check to see if we already have the handle in the set; since APIs
  // like WFMO don't allow duplicate handles in their arguments this is our
  // workaround (with the benefit of also reducing the native handle count).
  for (iree_host_size_t i = 0; i < set->handle_count; ++i) {
    iree_wait_handle_t* existing_handle = &set->user_handles[i];
    if (iree_wait_primitive_compare_identical(existing_handle, &handle)) {
      // Handle already exists in the set; just increment the reference count.
      ++existing_handle->set_internal.dupe_count;
      ++set->total_handle_count;
      return iree_ok_status();
    }
  }

  HANDLE native_handle = NULL;
  if (IREE_LIKELY(handle.type == IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE)) {
    // Our normal handle type; pass-through below.
    native_handle = (HANDLE)handle.value.win32.handle;
  } else {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unimplemented primitive type %d (expected PERMANENT/WIN32_HANDLE)",
        (int)handle.type);
  }

  // There's a max of 64 waitable handles. If we want to support more than that
  // we can spawn threads to wait on 64 objects and then wait on all those
  // threads. For example:
  //   iree_wait_multi(...180 handles...):
  //     -> spawn th0 and wait on handles 0-63 (64 handles)
  //     -> spawn th1 and wait on handles 64-127 (64 handles)
  //     wait on [th0, th1, handles 128-179] (threads + 52 remaining handles)
  //
  // At the point you're multiwaiting on that many things, though, it indicates
  // that there may be higher level coalescing that can be done by the
  // application itself (by, say, multiplexing sockets onto a single fd instead
  // of trying to wait on every unique socket handle via this API).
  if (native_handle &&
      IREE_UNLIKELY(set->handle_count + 1 > MAXIMUM_WAIT_OBJECTS)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "max wait objects exceeded; only up to %d native "
                            "wait handles are supported in WFMO",
                            (int)MAXIMUM_WAIT_OBJECTS);
  }

  ++set->total_handle_count;
  iree_host_size_t index = set->handle_count++;
  iree_wait_handle_t* user_handle = &set->user_handles[index];
  IREE_IGNORE_ERROR(
      iree_wait_handle_wrap_primitive(handle.type, handle.value, user_handle));
  user_handle->set_internal.dupe_count = 0;  // just us so far
  set->native_handles[index] = native_handle;

  return iree_ok_status();
}

void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle) {
  // Find the user handle in the set. This either requires a linear scan to
  // find the matching user handle or - if valid - we can use the native index
  // set after an iree_wait_any wake to do a quick lookup.
  iree_host_size_t index = handle.set_internal.index;
  if (IREE_UNLIKELY(index >= set->handle_count) ||
      IREE_UNLIKELY(!iree_wait_primitive_compare_identical(
          &set->user_handles[index], &handle))) {
    // Fallback to a linear scan of (hopefully) a small list.
    for (iree_host_size_t i = 0; i < set->handle_count; ++i) {
      if (iree_wait_primitive_compare_identical(&set->user_handles[i],
                                                &handle)) {
        index = i;
        break;
      }
    }
  }

  // Decrement reference count.
  iree_wait_handle_t* existing_handle = &set->user_handles[index];
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
    memcpy(&set->native_handles[index], &set->native_handles[tail_index],
           sizeof(*set->native_handles));
    memcpy(&set->user_handles[index], &set->user_handles[tail_index],
           sizeof(*set->user_handles));
  }
  --set->total_handle_count;
  --set->handle_count;
}

void iree_wait_set_clear(iree_wait_set_t* set) {
  set->total_handle_count = 0;
  set->handle_count = 0;
}

static iree_status_t iree_wait_multi(iree_wait_set_t* set, bool require_all,
                                     iree_time_t deadline_ns,
                                     iree_wait_handle_t* out_wake_handle) {
  // TODO(benvanik): see if we can use tracy's mutex tracking to make waits
  // nicer (at least showing signal->wait relations).

  // Early-exit when there's nothing to wait on.
  if (set->handle_count == 0) {
    if (out_wake_handle) memset(out_wake_handle, 0, sizeof(*out_wake_handle));
    return iree_ok_status();
  }

  // Remap absolute timeout to relative timeout, handling special values as
  // needed.
  DWORD timeout_ms =
      (DWORD)(iree_absolute_deadline_to_timeout_ns(deadline_ns) / 1000000ull);

  // Perform the wait; this is allowed to yield the calling thread even if the
  // timeout_ms is 0 to indicate a poll.
  DWORD result =
      WaitForMultipleObjectsEx(set->handle_count, set->native_handles,
                               /*bWaitAll=*/(require_all ? TRUE : FALSE),
                               timeout_ms, /*bAlertable=*/FALSE);

  if (result == WAIT_TIMEOUT) {
    // Timeout elapsed while waiting; note that the timeout may have been 0 to
    // force a poll and be an expected result. We avoid a full status object
    // here as we don't want to track all that in non-exceptional cases.
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  } else if (result >= WAIT_OBJECT_0 &&
             result < WAIT_OBJECT_0 + set->handle_count) {
    // One (or more) handles were signaled sucessfully.
    if (out_wake_handle) {
      DWORD wake_index = result - WAIT_OBJECT_0;
      iree_wait_primitive_value_t wake_value;
      memset(&wake_value, 0, sizeof(wake_value));
      wake_value.win32.handle = (uintptr_t)set->native_handles[wake_index];
      iree_wait_handle_wrap_primitive(IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE,
                                      wake_value, out_wake_handle);

      // Optimization for wait-wake-erase; this lets us avoid scanning the
      // native handle list (the kernel already did that for us!).
      out_wake_handle->set_internal.index = wake_index;
    }
    return iree_ok_status();
  } else if (result >= WAIT_ABANDONED_0 &&
             result < WAIT_ABANDONED_0 + set->handle_count) {
    // One (or more) mutex handles were abandonded during the wait.
    // This happens when a thread holding the mutex dies without releasing it.
    // This is less common in-process and more for the cross-process situations
    // where we have duped/opened a remote handle and the remote process dies.
    // That's a pretty situation but not quite unheard of in sandboxing impls
    // where death is a feature.
    //
    // NOTE: we shouldn't get abandoned handles in regular cases - both because
    // we don't really use mutex handles (though users may provide them) and
    // that mutex abandonment is exceptional. If you see this you are probably
    // going to want to look for thread exit messages or zombie processes.
    DWORD wake_index = result - WAIT_ABANDONED_0;
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "mutex native handle %lu abanonded; shared state is "
        "(likely) inconsistent",
        wake_index);
  } else if (result == WAIT_FAILED) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "WFMO failed");
  } else {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "WFMO internal error (unimplemented APC?)");
  }
}

iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_wait_multi(set, /*require_all=*/true, deadline_ns, NULL);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status =
      iree_wait_multi(set, /*require_all=*/false, deadline_ns, out_wake_handle);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Remap absolute timeout to relative timeout, handling special values as
  // needed.
  DWORD timeout_ms =
      (DWORD)(iree_absolute_deadline_to_timeout_ns(deadline_ns) / 1000000ull);

  // Perform the wait; this is allowed to yield the calling thread even if the
  // timeout_ms is 0 to indicate a poll.
  DWORD result =
      WaitForSingleObjectEx((HANDLE)handle->value.win32.handle, timeout_ms,
                            /*bAlertable=*/FALSE);

  iree_status_t status;
  if (result == WAIT_TIMEOUT) {
    // Timeout elapsed while waiting; note that the timeout may have been 0 to
    // force a poll and be an expected result. We avoid a full status object
    // here as we don't want to track all that in non-exceptional cases.
    status = iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  } else if (result == WAIT_OBJECT_0) {
    // Handle was signaled sucessfully.
    status = iree_ok_status();
  } else if (result == WAIT_ABANDONED_0) {
    // The mutex handle was abandonded during the wait.
    // This happens when a thread holding the mutex dies without releasing it.
    // This is less common in-process and more for the cross-process situations
    // where we have duped/opened a remote handle and the remote process dies.
    // That's a pretty situation but not quite unheard of in sandboxing impls
    // where death is a feature.
    //
    // NOTE: we shouldn't get abandoned handles in regular cases - both because
    // we don't really use mutex handles (though users may provide them) and
    // that mutex abandonment is exceptional. If you see this you are probably
    // going to want to look for thread exit messages or zombie processes.
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "mutex native handle abanonded; shared state is "
                              "(likely) inconsistent");
  } else if (result == WAIT_FAILED) {
    status = iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                              "WFSO failed");
  } else {
    status = iree_make_status(IREE_STATUS_INTERNAL,
                              "WFSO internal error (unimplemented APC?)");
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event) {
  memset(out_event, 0, sizeof(*out_event));
  iree_wait_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  value.win32.handle =
      (uintptr_t)CreateEvent(NULL, TRUE, initial_state ? TRUE : FALSE, NULL);
  if (!value.win32.handle) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "unable to create event");
  }
  return iree_wait_handle_wrap_primitive(IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE,
                                         value, out_event);
}

void iree_event_deinitialize(iree_event_t* event) {
  iree_wait_primitive_close(event);
}

void iree_event_set(iree_event_t* event) {
  SetEvent((HANDLE)event->value.win32.handle);
}

void iree_event_reset(iree_event_t* event) {
  ResetEvent((HANDLE)event->value.win32.handle);
}

#endif  // IREE_PLATFORM_WINDOWS
