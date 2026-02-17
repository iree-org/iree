// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Platform-specific async I/O handle primitives.
//
// iree_async_primitive_t is a non-owning (type, value) pair describing a
// platform handle that can be used for async operations. It is the async
// layer's equivalent of iree_wait_primitive_t and will eventually supersede it.
//
// This type is designed for proactor-based I/O rather than readiness-based
// polling. The primitive types reflect handles that can be submitted to
// io_uring, kqueue, IOCP, etc.

#ifndef IREE_ASYNC_PRIMITIVE_H_
#define IREE_ASYNC_PRIMITIVE_H_

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Platform detection
//===----------------------------------------------------------------------===//
// Each IREE_ASYNC_HAVE_* define can be set externally to override detection.
// This allows custom/embedded platforms to opt-in to features their platform
// supports (e.g., -DIREE_ASYNC_HAVE_FD=1 for a POSIX-like embedded target).

#if !defined(IREE_ASYNC_HAVE_FD)
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID) || \
    defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
#define IREE_ASYNC_HAVE_FD 1
#endif
#endif  // !IREE_ASYNC_HAVE_FD

#if !defined(IREE_ASYNC_HAVE_EVENTFD)
#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_ANDROID)
#define IREE_ASYNC_HAVE_EVENTFD 1
#endif
#endif  // !IREE_ASYNC_HAVE_EVENTFD

#if !defined(IREE_ASYNC_HAVE_MACH_PORT)
#if defined(IREE_PLATFORM_APPLE)
#define IREE_ASYNC_HAVE_MACH_PORT 1
#endif
#endif  // !IREE_ASYNC_HAVE_MACH_PORT

#if !defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
#if defined(IREE_PLATFORM_WINDOWS)
#define IREE_ASYNC_HAVE_WIN32_HANDLE 1
#endif
#endif  // !IREE_ASYNC_HAVE_WIN32_HANDLE

//===----------------------------------------------------------------------===//
// iree_async_primitive_t
//===----------------------------------------------------------------------===//

enum iree_async_primitive_type_e {
  // Empty/invalid handle.
  IREE_ASYNC_PRIMITIVE_TYPE_NONE = 0u,

  // POSIX file descriptor (Linux, macOS, BSD, Android).
  // Can represent sockets, files, eventfds, timerfd, signalfd, etc.
  IREE_ASYNC_PRIMITIVE_TYPE_FD = 1u,

  // Windows HANDLE (Event, Socket, File, Timer, etc.).
  // Usable with IOCP via CreateIoCompletionPort.
  IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE = 2u,

  // macOS/iOS Mach port.
  // Usable with kqueue via EVFILT_MACHPORT.
  IREE_ASYNC_PRIMITIVE_TYPE_MACH_PORT = 3u,
};

// Identifies the type of platform handle stored in an iree_async_primitive_t.
// Types that are unavailable on a platform are still defined for
// platform-independent routing but operations on them will fail.
typedef uint8_t iree_async_primitive_type_t;

// Platform-specific handle value.
// The active member is determined by the associated
// iree_async_primitive_type_t.
typedef union iree_async_primitive_value_t {
  // Avoids zero-sized union on platforms with no handles.
  int reserved;
#if defined(IREE_ASYNC_HAVE_FD)
  // IREE_ASYNC_PRIMITIVE_TYPE_FD
  int fd;
#endif  // IREE_ASYNC_HAVE_FD
#if defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
  // IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE
  uintptr_t win32_handle;
#endif  // IREE_ASYNC_HAVE_WIN32_HANDLE
#if defined(IREE_ASYNC_HAVE_MACH_PORT)
  // IREE_ASYNC_PRIMITIVE_TYPE_MACH_PORT
  uint32_t mach_port;
#endif  // IREE_ASYNC_HAVE_MACH_PORT
} iree_async_primitive_value_t;

// A non-owning (type, value) pair describing a platform async handle.
// Primitives do not manage the lifecycle of the underlying handle. Callers must
// ensure the handle remains valid for the duration of any operations
// referencing this primitive.
typedef struct iree_async_primitive_t {
  iree_async_primitive_type_t type;
  iree_async_primitive_value_t value;
} iree_async_primitive_t;

// Returns a primitive with the given |type| and |value|.
static inline iree_async_primitive_t iree_async_primitive_make(
    iree_async_primitive_type_t type, iree_async_primitive_value_t value) {
  iree_async_primitive_t primitive;
  primitive.type = type;
  primitive.value = value;
  return primitive;
}

// Returns an empty primitive (type NONE).
static inline iree_async_primitive_t iree_async_primitive_none(void) {
  iree_async_primitive_t primitive;
  memset(&primitive, 0, sizeof(primitive));
  return primitive;
}

// Returns true if the primitive is empty (type NONE).
static inline bool iree_async_primitive_is_none(
    iree_async_primitive_t primitive) {
  return primitive.type == IREE_ASYNC_PRIMITIVE_TYPE_NONE;
}

#if defined(IREE_ASYNC_HAVE_FD)
// Returns a primitive wrapping a POSIX file descriptor.
static inline iree_async_primitive_t iree_async_primitive_from_fd(int fd) {
  iree_async_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  value.fd = fd;
  return iree_async_primitive_make(IREE_ASYNC_PRIMITIVE_TYPE_FD, value);
}
#endif  // IREE_ASYNC_HAVE_FD

#if defined(IREE_ASYNC_HAVE_WIN32_HANDLE)
// Returns a primitive wrapping a Windows HANDLE.
static inline iree_async_primitive_t iree_async_primitive_from_win32_handle(
    uintptr_t handle) {
  iree_async_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  value.win32_handle = handle;
  return iree_async_primitive_make(IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE,
                                   value);
}
#endif  // IREE_ASYNC_HAVE_WIN32_HANDLE

#if defined(IREE_ASYNC_HAVE_MACH_PORT)
// Returns a primitive wrapping a Mach port.
static inline iree_async_primitive_t iree_async_primitive_from_mach_port(
    uint32_t port) {
  iree_async_primitive_value_t value;
  memset(&value, 0, sizeof(value));
  value.mach_port = port;
  return iree_async_primitive_make(IREE_ASYNC_PRIMITIVE_TYPE_MACH_PORT, value);
}
#endif  // IREE_ASYNC_HAVE_MACH_PORT

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PRIMITIVE_H_
