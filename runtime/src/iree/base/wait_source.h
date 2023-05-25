// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_WAIT_SOURCE_H_
#define IREE_BASE_WAIT_SOURCE_H_

#include "iree/base/attributes.h"
#include "iree/base/status.h"
#include "iree/base/target_platform.h"
#include "iree/base/time.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_wait_primitive_t
//===----------------------------------------------------------------------===//

#if IREE_SYNCHRONIZATION_DISABLE_UNSAFE
// Bare metal/no synchronization available; wait handles are no-oped.
#define IREE_WAIT_HANDLE_DISABLED 1
#elif defined(IREE_PLATFORM_WINDOWS)
// Though Windows can support pipes no one uses them so for simplicity we only
// exposes HANDLEs.
#define IREE_HAVE_WAIT_TYPE_WIN32_HANDLE 1
#elif defined(IREE_PLATFORM_EMSCRIPTEN)
// Emscripten can use JavaScript Promises (pipe also works via Emscripten's
// emulation, but Promises are platform-native primitives).
#define IREE_HAVE_WAIT_TYPE_JAVASCRIPT_PROMISE 1
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

#if !IREE_SYNCHRONIZATION_DISABLE_UNSAFE
#define IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX 1
#endif  // threading enabled

// Specifies the type of a system wait primitive.
// Enums that are unavailable on a platform are still present to allow for
// platform-independent code to still route wait primitives but actually using
// them will fail.
enum iree_wait_primitive_type_bits_t {
  // Empty handle; immediately resolved.
  IREE_WAIT_PRIMITIVE_TYPE_NONE = 0u,

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

  // Process-local futex.
  // These are only valid for multi-wait when used with an in-process wait
  // handle implementation (IREE_WAIT_API == IREE_WAIT_API_INPROC).
  IREE_WAIT_PRIMITIVE_TYPE_LOCAL_FUTEX = 5u,

  // Web platform JavaScript Promise.
  // It is not possible to block until one of these resolves.
  //
  // More information:
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise
  // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises
  IREE_WAIT_PRIMITIVE_TYPE_JAVASCRIPT_PROMISE = 6u,

  // Placeholder for wildcard queries of primitive types.
  // On an export request this indicates that the source may export any type it
  // can.
  IREE_WAIT_PRIMITIVE_TYPE_ANY = 0xFFu,
};
typedef uint8_t iree_wait_primitive_type_t;

// A handle value whose behavior is defined by the iree_wait_primitive_type_t.
// Only the primitives available on a platform are compiled in as syscalls and
// other associated operations that act on them aren't available anyway.
typedef union {
  int reserved;  // to avoid zero-sized unions
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
#if defined(IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX)
  /*iree_futex_handle_t*/ void* local_futex;
#endif  // IREE_HAVE_WAIT_TYPE_LOCAL_FUTEX
#if defined(IREE_HAVE_WAIT_TYPE_JAVASCRIPT_PROMISE)
  struct {
    int handle;
  } promise;
#endif  // IREE_HAVE_WAIT_TYPE_JAVASCRIPT_PROMISE
} iree_wait_primitive_value_t;

// A (type, value) pair describing a system wait primitive handle.
typedef struct iree_wait_primitive_t {
  iree_wait_primitive_type_t type;
  iree_wait_primitive_value_t value;
} iree_wait_primitive_t;

// Returns a wait primitive with the given (|type|, |value|).
static inline iree_wait_primitive_t iree_make_wait_primitive(
    iree_wait_primitive_type_t type, iree_wait_primitive_value_t value) {
  iree_wait_primitive_t primitive = {type, value};
  return primitive;
}

// Returns a wait primitive that will resolve immediately if waited on.
static inline iree_wait_primitive_t iree_wait_primitive_immediate(void) {
  iree_wait_primitive_value_t dummy_primitive = {0};
  return iree_make_wait_primitive(IREE_WAIT_PRIMITIVE_TYPE_NONE,
                                  dummy_primitive);
}

// Returns true if the |wait_primitive| is resolved immediately (empty).
static inline bool iree_wait_primitive_is_immediate(
    iree_wait_primitive_t wait_primitive) {
  return wait_primitive.type == IREE_WAIT_PRIMITIVE_TYPE_NONE;
}

//===----------------------------------------------------------------------===//
// iree_wait_source_t
//===----------------------------------------------------------------------===//

typedef struct iree_wait_source_t iree_wait_source_t;

// Controls the behavior of an iree_wait_source_ctl_fn_t callback function.
typedef enum iree_wait_source_command_e {
  // Queries the state of the wait source.
  // Returns IREE_STATUS_DEFERRED if the wait source is not yet resolved.
  //
  // iree_wait_source_ctl_fn_t:
  //   params: unused
  //   inout_ptr: iree_status_code_t* out_wait_status_code
  IREE_WAIT_SOURCE_COMMAND_QUERY = 0u,

  // Tries to wait for the wait source to resolve.
  // Returns IREE_STATUS_DEFERRED if the wait source does not support waiting.
  //
  // iree_wait_source_ctl_fn_t:
  //   params: iree_wait_source_wait_params_t
  //   inout_ptr: unused
  IREE_WAIT_SOURCE_COMMAND_WAIT_ONE,

  // Exports the wait source to a system wait handle.
  //
  // iree_wait_source_ctl_fn_t:
  //   params: iree_wait_source_export_params_t
  //   inout_ptr: iree_wait_primitive_t* out_wait_primitive
  IREE_WAIT_SOURCE_COMMAND_EXPORT,
} iree_wait_source_command_t;

// Parameters for IREE_WAIT_SOURCE_COMMAND_WAIT_ONE.
typedef struct iree_wait_source_wait_params_t {
  // Timeout after which the wait will return even if the wait source is not
  // resolved with IREE_STATUS_DEADLINE_EXCEEDED.
  iree_timeout_t timeout;
} iree_wait_source_wait_params_t;

// Parameters for IREE_WAIT_SOURCE_COMMAND_EXPORT.
typedef struct iree_wait_source_export_params_t {
  // Indicates the target handle type of the export operation.
  iree_wait_primitive_type_t target_type;
  // Timeout after which the export will return even if the wait source is not
  // yet available for export with IREE_STATUS_DEADLINE_EXCEEDED.
  iree_timeout_t timeout;
} iree_wait_source_export_params_t;

// Function pointer for an iree_wait_source_t control function.
// |command| provides the operation to perform. Optionally some commands may use
// |params| to pass additional operation-specific parameters. |inout_ptr| usage
// is defined by each operation.
typedef iree_status_t(IREE_API_PTR* iree_wait_source_ctl_fn_t)(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void* params, void** inout_ptr);

// A wait source instance representing some future point in time.
// Wait sources are promises for a system native wait handle that allow for
// cheaper queries and waits when the full system wait path is not required.
//
// Wait sources may have user-defined implementations or come from system wait
// handles via iree_wait_source_import.
typedef struct iree_wait_source_t {
  union {
    struct {
      // Control function data.
      void* self;
      // Implementation-defined data identifying the point in time.
      uint64_t data;
    };
    // Large enough to store an iree_wait_handle_t, used when importing a
    // system wait handle into a wait source.
    uint64_t storage[2];
  };
  // ioctl-style control function servicing wait source commands.
  // See iree_wait_source_command_t for more information.
  iree_wait_source_ctl_fn_t ctl;
} iree_wait_source_t;

// Returns a wait source that will always immediately return as resolved.
static inline iree_wait_source_t iree_wait_source_immediate(void) {
  iree_wait_source_t v = {{{NULL, 0ull}}, NULL};
  return v;
}

// Returns true if the |wait_source| is immediately resolved.
// This can be used to neuter waits in lists/sets.
static inline bool iree_wait_source_is_immediate(
    iree_wait_source_t wait_source) {
  return wait_source.ctl == NULL;
}

// Wait source control function for iree_wait_source_delay.
IREE_API_EXPORT iree_status_t iree_wait_source_delay_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void* params, void** inout_ptr);

// Returns a wait source that indicates a delay until a point in time.
// The source will remain unresolved until the |deadline_ns| is reached or
// exceeded and afterward return resolved. Export is unavailable.
static inline iree_wait_source_t iree_wait_source_delay(
    iree_time_t deadline_ns) {
  iree_wait_source_t v = {
      {{NULL, (uint64_t)deadline_ns}},
      iree_wait_source_delay_ctl,
  };
  return v;
}

// Returns true if the |wait_source| is a timed delay.
// These are sleeps that can often be handled more intelligently by platforms.
static inline bool iree_wait_source_is_delay(iree_wait_source_t wait_source) {
  return wait_source.ctl == iree_wait_source_delay_ctl;
}

// Imports a system |wait_primitive| into a wait source in |out_wait_source|.
// Ownership of the wait handle remains will the caller and it must remain valid
// for the duration the wait source is in use.
IREE_API_EXPORT iree_status_t iree_wait_source_import(
    iree_wait_primitive_t wait_primitive, iree_wait_source_t* out_wait_source);

// Exports a |wait_source| to a system wait primitive in |out_wait_primitive|.
// If the wait source is already resolved then the wait handle will be set to
// immediate and callers can check it with iree_wait_primitive_is_immediate.
// If the wait source resolved with a failure then the error status will be
// returned. The returned wait handle is owned by the wait source and will
// remain valid for the lifetime of the wait source.
//
// Exporting may require a blocking operation and |timeout| can be used to
// limit its duration.
//
// Returns IREE_STATUS_UNAVAILABLE if the requested primitive |target_type| is
// unavailable on the current platform or from the given wait source.
// Passing IREE_WAIT_PRIMITIVE_TYPE_ANY will allow the implementation to return
// any primitive that it can.
IREE_API_EXPORT iree_status_t iree_wait_source_export(
    iree_wait_source_t wait_source, iree_wait_primitive_type_t target_type,
    iree_timeout_t timeout, iree_wait_primitive_t* out_wait_primitive);

// Queries the state of a |wait_source| without waiting.
// |out_wait_status_code| will indicate the status of the source while the
// returned value indicates the status of the query. |out_wait_status_code| will
// be set to IREE_STATUS_DEFERRED if the wait source has not yet resolved and
// IREE_STATUS_OK otherwise.
IREE_API_EXPORT iree_status_t iree_wait_source_query(
    iree_wait_source_t wait_source, iree_status_code_t* out_wait_status_code);

// Blocks the caller and waits for a |wait_source| to resolve.
// Returns IREE_STATUS_DEADLINE_EXCEEDED if |timeout| is reached before the
// wait source resolves. If the wait source resolved with a failure then the
// error status will be returned.
IREE_API_EXPORT iree_status_t iree_wait_source_wait_one(
    iree_wait_source_t wait_source, iree_timeout_t timeout);

// TODO(benvanik): iree_wait_source_wait_any/all: allow multiple wait sources
// that share the same control function. The implementation can decide if it
// wants to coalesce them or not.

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_WAIT_SOURCE_H_
