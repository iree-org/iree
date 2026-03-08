// Copyright 2020 The IREE Authors
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

#include "iree/base/internal/wait_handle.h"

#if IREE_WAIT_API == IREE_WAIT_API_WIN32

//===----------------------------------------------------------------------===//
// Platform utilities
//===----------------------------------------------------------------------===//

static_assert(
    sizeof(iree_wait_primitive_value_t) == sizeof(HANDLE),
    "win32 HANDLE type must match uintptr size in wait primitive struct");

//===----------------------------------------------------------------------===//
// iree_wait_primitive_* raw calls
//===----------------------------------------------------------------------===//

// Closes an existing handle. Must not be called while there are any waiters on
// the handle.
void iree_wait_handle_close(iree_wait_handle_t* handle) {
  if (IREE_LIKELY(handle->value.win32.handle != 0)) {
    CloseHandle((HANDLE)handle->value.win32.handle);
  }
  iree_wait_handle_deinitialize(handle);
}

iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns) {
  if (handle->type == IREE_WAIT_PRIMITIVE_TYPE_NONE) {
    return iree_ok_status();
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Remap absolute timeout to relative timeout, handling special values as
  // needed.
  DWORD timeout_ms = iree_absolute_deadline_to_timeout_ms(deadline_ns);

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
    // Handle was signaled successfully.
    status = iree_ok_status();
  } else if (result == WAIT_ABANDONED_0) {
    // The mutex handle was abandoned during the wait.
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
                              "mutex native handle abandoned; shared state is "
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
  iree_wait_handle_wrap_primitive(IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE, value,
                                  out_event);
  return iree_ok_status();
}

void iree_event_deinitialize(iree_event_t* event) {
  iree_wait_handle_close(event);
}

void iree_event_set(iree_event_t* event) {
  HANDLE handle = (HANDLE)event->value.win32.handle;
  if (handle) SetEvent(handle);
}

void iree_event_reset(iree_event_t* event) {
  HANDLE handle = (HANDLE)event->value.win32.handle;
  if (handle) ResetEvent(handle);
}

#endif  // IREE_WAIT_API == IREE_WAIT_API_WIN32
