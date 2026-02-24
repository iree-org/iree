// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX backend registration for CTS tests.
//
// The POSIX backend uses various event notification mechanisms + worker threads
// for portable async I/O. It supports multishot via emulated poll-loop re-arm
// and buffer registration via userspace pool management. Tests tagged with
// zerocopy or kernel-mediated messaging capabilities will be skipped.
//
// Backends registered:
//   - posix_poll: poll() backend - always available on POSIX systems
//   - posix_epoll: epoll backend - Linux only (when implemented)
//   - posix_kqueue: kqueue backend - BSD/macOS only (when implemented)

#include "iree/async/cts/util/registry.h"
#include "iree/async/platform/posix/event_set.h"
#include "iree/async/platform/posix/proactor.h"
#include "iree/async/proactor.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

// Creates a POSIX proactor using poll() for event notification.
static iree::StatusOr<iree_async_proactor_t*> CreatePosixPollProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_posix_with_backend(
      iree_async_proactor_options_default(),
      IREE_ASYNC_POSIX_EVENT_BACKEND_POLL, iree_allocator_system(), &proactor));
  return proactor;
}

#if defined(IREE_PLATFORM_LINUX)
// Creates a POSIX proactor using epoll for event notification (Linux only).
static iree::StatusOr<iree_async_proactor_t*> CreatePosixEpollProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_posix_with_backend(
      iree_async_proactor_options_default(),
      IREE_ASYNC_POSIX_EVENT_BACKEND_EPOLL, iree_allocator_system(),
      &proactor));
  return proactor;
}
#endif  // IREE_PLATFORM_LINUX

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
// Creates a POSIX proactor using kqueue for event notification (BSD/macOS).
static iree::StatusOr<iree_async_proactor_t*> CreatePosixKqueueProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(iree_async_proactor_create_posix_with_backend(
      iree_async_proactor_options_default(),
      IREE_ASYNC_POSIX_EVENT_BACKEND_KQUEUE, iree_allocator_system(),
      &proactor));
  return proactor;
}
#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD

//===----------------------------------------------------------------------===//
// Backend registration
//===----------------------------------------------------------------------===//

// poll() backend: Always available, O(n) fd scanning.
// This is the portable baseline - works on any POSIX system.
static bool posix_poll_registered_ =
    (CtsRegistry::RegisterBackend({
         "posix_poll",
         {"posix_poll", CreatePosixPollProactor},
         {"portable", "multishot"},
     }),
     true);

#if defined(IREE_PLATFORM_LINUX)
// epoll backend: Linux-specific, O(1) ready fd enumeration.
// Currently returns UNAVAILABLE until event_set_epoll.c is implemented.
static bool posix_epoll_registered_ =
    (CtsRegistry::RegisterBackend({
         "posix_epoll",
         {"posix_epoll", CreatePosixEpollProactor},
         {"linux", "efficient", "multishot"},
     }),
     true);
#endif  // IREE_PLATFORM_LINUX

#if defined(IREE_PLATFORM_APPLE) || defined(IREE_PLATFORM_BSD)
// kqueue backend: BSD/macOS-specific, O(1) ready fd enumeration.
// Currently returns UNAVAILABLE until event_set_kqueue.c is implemented.
static bool posix_kqueue_registered_ =
    (CtsRegistry::RegisterBackend({
         "posix_kqueue",
         {"posix_kqueue", CreatePosixKqueueProactor},
         {"bsd", "efficient", "multishot"},
     }),
     true);
#endif  // IREE_PLATFORM_APPLE || IREE_PLATFORM_BSD

}  // namespace iree::async::cts
