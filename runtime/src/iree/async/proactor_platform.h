// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PROACTOR_PLATFORM_H_
#define IREE_ASYNC_PROACTOR_PLATFORM_H_

#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Selects the best available backend for the current platform:
// - Linux: io_uring (kernel 5.1+), falls back to threaded
// - macOS: kqueue, falls back to threaded
// - Windows: IOCP (future), falls back to threaded
// - Other: threaded emulation
//
// The fallback is silent: callers that need specific capabilities should
// check query_capabilities() after creation.
iree_status_t iree_async_proactor_create_platform(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PROACTOR_PLATFORM_H_
