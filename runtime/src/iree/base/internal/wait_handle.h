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
// TODO(benvanik): packing to ensure we are getting the expected alignments.
typedef struct iree_wait_handle_t {
  union {
    // Used by iree_wait_set_t storage to track the number of duplicate
    // instances of a particular handle within the set to avoid needing to store
    // them all separately. A dupe_count of 0 means there is one unique handle.
    uint32_t dupe_count : 16;
    // Used by iree_wait_any and iree_wait_set_erase to optimize the
    // wait-wake-erase pattern by avoiding the need to scan the internal storage
    // list to erase a handle.
    uint32_t index : 16;
    // (3 bytes total available)
    uint8_t storage[3];
  } set_internal;
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

// iree_wait_source_t control function.
iree_status_t iree_wait_handle_ctl(iree_wait_source_t wait_source,
                                   iree_wait_source_command_t command,
                                   const void* params, void** inout_ptr);

// Returns a pointer to the wait handle in |wait_source| if it is using
// iree_wait_handle_ctl and otherwise NULL.
static inline iree_wait_handle_t* iree_wait_handle_from_source(
    iree_wait_source_t* wait_source) {
  return wait_source->ctl == iree_wait_handle_ctl
             ? (iree_wait_handle_t*)wait_source->storage
             : NULL;
}

//===----------------------------------------------------------------------===//
// iree_wait_set_t
//===----------------------------------------------------------------------===//

// A platform-specific cache of wait handles that can be multi-waited.
// By caching callers don't need to build the list each wait and implementations
// can store acceleration information or kernel API data structures and either
// optimize or make compliant sets such as by deduplicating or sorting by
// primitive type to perform a multi-api muli-wait.
//
// Certain handle types may also gain benefits: when syncfile is used we can use
// sync_merge to coalesce wait handles when performing a wait-all on multiple
// handles.
//
// This cache shines when handles are persistent (such as sockets/eventfds/etc)
// and the set will rarely be changing relative to how many times it will be
// waited on. It's not as optimal in the cases of one-shot waits on small
// numbers of handles but those are also the cases where the set overhead is
// small (2 set insertions all touching hot cache lines is fine) and we gain
// the benefits of a unified code path and nice error handling/validation.
//
// Thread-compatible; only one thread may be manipulating or waiting on a
// particular set at any time.
typedef struct iree_wait_set_t iree_wait_set_t;

// Allocates a wait set with the maximum |capacity| of unique handles.
iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set);

// Frees a wait set. The wait set must not be being waited on.
void iree_wait_set_free(iree_wait_set_t* set);

// Returns true if there are no handles registered with the set.
bool iree_wait_set_is_empty(const iree_wait_set_t* set);

// Inserts a wait handle into the set.
// If the handle is already in the set it will be reference counted such that a
// matching number of iree_wait_set_erase calls are required.
iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle);

// Erases a single instance of a wait handle from the set.
// Decrements the reference count; if the same handle was inserted multiple
// times then it may still remain in the set after the call returns.
void iree_wait_set_erase(iree_wait_set_t* set, iree_wait_handle_t handle);

// Clears all handles from the wait set.
void iree_wait_set_clear(iree_wait_set_t* set);

// TODO(benvanik): signal/interrupt API to make a wait set wake up.
// Can be implemented with signals/QueueUserAPC/etc. The workaround is that the
// caller will need to create their own events to add to the set where for
// transient wakes we could avoid that extra overhead.

// Blocks the caller until all of the passed wait handles are signaled or the
// |deadline_ns| elapses.
//
// A deadline of IREE_DURATION_ZERO will act as a poll and not block the caller.
// IREE_DURATION_INFINITE can be used to block until signaled.
//
// Returns success if all handles were signaled either prior to the call or
// during the wait.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the deadline elapses without all
// handles having been signaled. Note that zero or more handles may have
// actually signaled even if the deadline is exceeded (such as if they signal
// while the waiting thread is resuming from the failed wait).
//
// iree_wait_set_t is thread-compatible; only one thread may be manipulating or
// waiting on a set at any time.
iree_status_t iree_wait_all(iree_wait_set_t* set, iree_time_t deadline_ns);

// Blocks the caller until at least one of the handles is signaled or the
// |deadline_ns| elapses.
//
// A deadline of IREE_TIME_INFINITE_PAST will act as a poll and not block the
// caller. IREE_TIME_INFINITE_FUTURE can be used to block until signaled.
//
// Returns success if all handles were signaled either prior to the call or
// during the wait. A handle of one of the signaled handles will be returned in
// the optional |out_wake_handle| argument; note however that one or more
// handles may have signaled and which handle is returned is unspecified.
// Callers are expected to use the handle to short-circuit scanning the handles
// list but if a full scan is going to happen regardless it can be ignored.
//
// |out_wake_handle| contains an optimization for wait-wake-erase set
// operations; it is cheap to pass the woken handle to iree_wait_set_erase if
// there are no interleaving operations that change the set layout.
//
// Returns IREE_STATUS_DEADLINE_EXCEEDED if the deadline elapses without any
// handle having been signaled.
//
// iree_wait_set_t is thread-compatible; only one thread may be manipulating or
// waiting on a set at any time.
iree_status_t iree_wait_any(iree_wait_set_t* set, iree_time_t deadline_ns,
                            iree_wait_handle_t* out_wake_handle);

// Blocks the caller until the given wait handle is signaled or |deadline_ns|
// elapses. This is functionally equivalent to iree_wait_any/iree_wait_all used
// on a set with a single handle in it but depending on the implementation may
// not require additional allocations/state tracking.
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
// Events are much heavier than iree_notification_t but are waitable objects
// that can be passed to iree_wait_all/iree_wait_any. Prefer iree_notification_t
// when multiwaiting is not required.
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
