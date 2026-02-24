// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/proactor_platform.h"

#if defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID)
#include "iree/async/platform/io_uring/api.h"
#endif  // IREE_PLATFORM_LINUX && !IREE_PLATFORM_ANDROID

#if !defined(IREE_PLATFORM_WINDOWS)
#include "iree/async/platform/posix/api.h"
#endif  // !IREE_PLATFORM_WINDOWS

#if defined(IREE_PLATFORM_WINDOWS)
#include "iree/async/platform/iocp/api.h"
#endif  // IREE_PLATFORM_WINDOWS

iree_status_t iree_async_proactor_create_platform(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  iree_status_t status = iree_status_from_code(IREE_STATUS_UNAVAILABLE);

#if defined(IREE_PLATFORM_WINDOWS)

  status = iree_async_proactor_create_iocp(options, allocator, out_proactor);

#elif defined(IREE_PLATFORM_LINUX) && !defined(IREE_PLATFORM_ANDROID)

  // Try io_uring first (kernel 5.1+, enabled). Falls back to POSIX proactor
  // if io_uring is not usable (kernel too old, blocked by seccomp/sysctl,
  // insufficient locked memory, etc.).
  status =
      iree_async_proactor_create_io_uring(options, allocator, out_proactor);
  if (iree_status_is_unavailable(status)) {
    iree_status_ignore(status);
    status = iree_async_proactor_create_posix(options, allocator, out_proactor);
  }

#else  // macOS, BSD, Android, etc.

  status = iree_async_proactor_create_posix(options, allocator, out_proactor);

#endif  // IREE_PLATFORM_*

  IREE_TRACE_ZONE_END(z0);
  return status;
}
