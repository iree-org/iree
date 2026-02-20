// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Public API for the POSIX proactor backend.
//
// This header exposes only what external users need: the create function.
// Internal implementation details are in proactor.h (not exported).

#ifndef IREE_ASYNC_PLATFORM_POSIX_API_H_
#define IREE_ASYNC_PLATFORM_POSIX_API_H_

#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a POSIX-based proactor implementation.
//
// Uses the best available event notification backend for the platform:
//   - Linux: epoll (falls back to poll if epoll unavailable)
//   - macOS/BSD: kqueue (falls back to poll if kqueue unavailable)
//   - Other POSIX: poll
//
// Returns IREE_STATUS_UNAVAILABLE if no POSIX event backend is available on
// this system (should not happen on any standard POSIX platform).
iree_status_t iree_async_proactor_create_posix(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_API_H_
