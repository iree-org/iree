// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_INTERNAL_WAIT_HANDLE_H_
#define IREE_BASE_INTERNAL_WAIT_HANDLE_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_wait_handle_t
//===----------------------------------------------------------------------===//

// Non-owning handle reference to a waitable object.
typedef struct iree_wait_handle_t {
  // Reserved padding for alignment with iree_wait_primitive_type_t.
  uint8_t reserved[3];
  // Inlined iree_wait_primitive_t to get better packing:
  iree_wait_primitive_type_t type;  // uint8_t
  iree_wait_primitive_value_t value;
} iree_wait_handle_t;
static_assert(sizeof(iree_wait_handle_t) <= sizeof(uint64_t) * 2,
              "iree_wait_handle_t must fit in 16-bytes so it can be stored in "
              "other data structures");

// Returns a wait handle that is immediately resolved.
static inline iree_wait_handle_t iree_wait_handle_immediate(void) {
  iree_wait_handle_t wait_handle;
  memset(&wait_handle, 0, sizeof(wait_handle));
  return wait_handle;
}

// Returns true if the wait |handle| is resolved immediately (empty).
static inline bool iree_wait_handle_is_immediate(iree_wait_handle_t handle) {
  return handle.type == IREE_WAIT_PRIMITIVE_TYPE_NONE;
}

// Initializes a wait handle with the given primitive type and value.
// Wait handles do not retain the provided primitives and they must be kept
// valid (allocated and open) for the duration any wait handle references them.
void iree_wait_handle_wrap_primitive(
    iree_wait_primitive_type_t primitive_type,
    iree_wait_primitive_value_t primitive_value,
    iree_wait_handle_t* out_handle);

// Deinitializes a wait handle.
// Note that wait handles do not retain the underlying wait primitive and
// deinitializing a handle will not close the resource.
void iree_wait_handle_deinitialize(iree_wait_handle_t* handle);

// Closes a wait handle and resets |handle|.
void iree_wait_handle_close(iree_wait_handle_t* handle);

// Resolve function for wait sources backed by system wait handles.
iree_status_t iree_wait_handle_resolve(
    iree_wait_source_t wait_source, iree_timeout_t timeout,
    iree_wait_source_resolve_callback_t callback, void* user_data);

// Returns a pointer to the wait handle in |wait_source| if it is using
// iree_wait_handle_resolve and otherwise NULL.
static inline iree_wait_handle_t* iree_wait_handle_from_source(
    iree_wait_source_t* wait_source) {
  return wait_source->resolve == iree_wait_handle_resolve
             ? (iree_wait_handle_t*)wait_source->storage
             : NULL;
}

// Blocks the caller until the given wait handle is signaled or |deadline_ns|
// elapses.
//
// A deadline of IREE_TIME_INFINITE_PAST will act as a poll and not block the
// caller. IREE_TIME_INFINITE_FUTURE can be used to block until signaled.
//
// Returns success if the handle was signaled either prior to the call or
// during the wait.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the deadline elapses without the
// handle having been signaled.
iree_status_t iree_wait_one(iree_wait_handle_t* handle,
                            iree_time_t deadline_ns);

//===----------------------------------------------------------------------===//
// iree_event_t
//===----------------------------------------------------------------------===//

// A manual reset event (aka binary semaphore).
// https://docs.microsoft.com/en-us/windows/win32/sync/event-objects
//
// Events are much heavier than iree_notification_t but are waitable objects.
// Prefer iree_notification_t when only single-handle waiting is needed.
//
// Which primitive is used will depend on the current platform.
typedef iree_wait_handle_t iree_event_t;

// Initializes an event in either the signaled or unsignaled state.
// The event must be closed with iree_event_deinitialize.
iree_status_t iree_event_initialize(bool initial_state,
                                    iree_event_t* out_event);

// Deinitializes an event.
void iree_event_deinitialize(iree_event_t* event);

// Sets the event object to the signaled state.
// The event stays signaled until iree_event_reset is called. Multiple waiters
// will be woken and attempted waits while the event is set will succeed
// immediately.
void iree_event_set(iree_event_t* event);

// Resets the event object to the unsignaled state.
// Resetting an event that is already reset has no effect.
void iree_event_reset(iree_event_t* event);

// Returns a wait source reference to |event|.
// The event must be kept live for as long as the reference is live.
iree_wait_source_t iree_event_await(iree_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_WAIT_HANDLE_H_
