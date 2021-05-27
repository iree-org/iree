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

#ifndef IREE_BASE_INTERNAL_WAIT_HANDLE_H_
#define IREE_BASE_INTERNAL_WAIT_HANDLE_H_

#include "iree/base/api.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)
// Though Windows can support pipes no one uses them so for simplicity we only
// exposes HANDLEs.
#define IREE_HAVE_WAIT_TYPE_WIN32_HANDLE 1
#elif defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX)
// Treat Android and modern linux as (mostly) the same.
#define IREE_HAVE_WAIT_TYPE_EVENTFD 1
#define IREE_HAVE_WAIT_TYPE_PIPE 1
#else
// BSD/Darwin/etc all have pipe.
#define IREE_HAVE_WAIT_TYPE_PIPE 1
#endif  // IREE_PLATFORM_*

// TODO(benvanik): see if we can get sync file on linux too:
#if defined(IREE_PLATFORM_ANDROID)
#define IREE_HAVE_WAIT_TYPE_SYNC_FILE 1
#endif  // IREE_PLATFORM_ANDROID

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_wait_primitive_*
//===----------------------------------------------------------------------===//

// TODO(benvanik): conditionally compile out enum values unavailable (to avoid
// runtime surprises).

// Specifies the type of a wait handle.
enum iree_wait_primitive_type_e {
  // Android/Linux eventfd handle.
  // These are akin to pipe() but require only a single handle and have
  // significantly lower overhead (equivalent if not slightly better than
  // pthreads condvars).
  //
  // eventfds support acting as both semaphores and auto reset events.
  //
  // More information:
  // http://man7.org/linux/man-pages/man2/eventfd.2.html
  IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD = 1u,

  // Android/Linux sync_file handle (aka 'sync fence').
  // The handle is allocated indirectly by the device driver via the
  // <linux/sync_file.h> API. It may be waited upon with poll(), select(), or
  // epoll() and must be closed with close() when no longer required. If
  // waiting on multiple sync_files the caller should first merge them
  // together.
  //
  // A sync_file must only be used as fences (one-shot manual reset events).
  //
  // More information:
  // https://www.kernel.org/doc/Documentation/sync_file.txt
  // https://lwn.net/Articles/702339/
  // https://source.android.com/devices/graphics/implement-vsync#explicit_synchronization
  // https://developer.android.com/ndk/reference/group/sync
  IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE = 2u,

  // Android/Linux/iOS-compatible POSIX pipe handle.
  // Two handles are generated: one for transmitting and one for receiving.
  //
  // More information:
  // http://man7.org/linux/man-pages/man2/pipe.2.html
  IREE_WAIT_PRIMITIVE_TYPE_PIPE = 3u,

  // Windows HANDLE type.
  // The HANDLE may represent a thread, event, semaphore, timer, etc.
  //
  // More information:
  // https://docs.microsoft.com/en-us/windows/win32/sysinfo/object-categories
  // https://docs.microsoft.com/en-us/windows/win32/sync/using-event-objects
  IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE = 4u,
};
typedef uint8_t iree_wait_primitive_type_t;

// A handle value whose behavior is defined by the iree_wait_primitive_type_t.
typedef union {
#if defined(IREE_HAVE_WAIT_TYPE_EVENTFD)
  // IREE_WAIT_PRIMITIVE_TYPE_EVENT_FD
  struct {
    int fd;
  } event;
#endif  // IREE_HAVE_WAIT_TYPE_EVENTFD
#if defined(IREE_HAVE_WAIT_TYPE_SYNC_FILE)
  // IREE_WAIT_PRIMITIVE_TYPE_SYNC_FILE
  struct {
    int fd;
  } sync_file;
#endif  // IREE_HAVE_WAIT_TYPE_SYNC_FILE
#if defined(IREE_HAVE_WAIT_TYPE_PIPE)
  // IREE_WAIT_PRIMITIVE_TYPE_PIPE
  union {
    struct {
      int read_fd;
      int write_fd;
    };
    int fds[2];
  } pipe;
#endif  // IREE_HAVE_WAIT_TYPE_PIPE
#if defined(IREE_HAVE_WAIT_TYPE_WIN32_HANDLE)
  // IREE_WAIT_PRIMITIVE_TYPE_WIN32_HANDLE
  struct {
    uintptr_t handle;
  } win32;
#endif  // IREE_HAVE_WAIT_TYPE_WIN32_HANDLE
} iree_wait_primitive_value_t;

//===----------------------------------------------------------------------===//
// iree_wait_handle_t
//===----------------------------------------------------------------------===//

// Non-owning handle reference to a waitable object.
// TODO(benvanik): packing to ensure we are getting the expected alignments.
typedef struct {
  iree_wait_primitive_type_t type;  // uint8_t
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
  iree_wait_primitive_value_t value;
} iree_wait_handle_t;

// Initializes a wait handle with the given primitive type and value.
// Wait handles do not retain the provided primitives and they must be kept
// valid (allocated and open) for the duration any wait handle references them.
iree_status_t iree_wait_handle_wrap_primitive(
    iree_wait_primitive_type_t primitive_type,
    iree_wait_primitive_value_t primitive_value,
    iree_wait_handle_t* out_handle);

// Deinitializes a wait handle.
// Note that wait handles do not retain the underlying wait primitive and
// deinitializing a handle will not close the resource.
void iree_wait_handle_deinitialize(iree_wait_handle_t* handle);

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
typedef struct iree_wait_set_s iree_wait_set_t;

// Allocates a wait set with the maximum |capacity| of unique handles.
iree_status_t iree_wait_set_allocate(iree_host_size_t capacity,
                                     iree_allocator_t allocator,
                                     iree_wait_set_t** out_set);

// Frees a wait set. The wait set must not be being waited on.
void iree_wait_set_free(iree_wait_set_t* set);

// Inserts a wait handle into the set.
// If the handle is already in the set it will be reference counted such that a
// matching number of iree_wait_set_erase calls are required.
iree_status_t iree_wait_set_insert(iree_wait_set_t* set,
                                   iree_wait_handle_t handle);

// Erases a single instance of a wait handle from the set.
// Decrements the reference count; if the same handle was inserted multiple
// times then the it may still remain in the set after an erase!
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

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_INTERNAL_WAIT_HANDLE_H_
