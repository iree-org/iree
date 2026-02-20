// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Public API for the io_uring proactor backend.
//
// This header exposes only what external users need: the create function.
// Internal implementation details are in proactor.h (not exported).

#ifndef IREE_ASYNC_PLATFORM_IO_URING_API_H_
#define IREE_ASYNC_PLATFORM_IO_URING_API_H_

#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an io_uring-based proactor implementation for Linux.
//
// io_uring provides true async I/O with minimal syscall overhead:
//   - Single submission queue for all operation types
//   - Completion queue drains without syscalls (memory-mapped)
//   - Linked SQEs for kernel-side operation chaining
//   - Registered buffers for zero-copy I/O
//   - Provided buffer rings for multishot receives
//
// Requires kernel 5.1+ with io_uring support enabled.
//
// Returns IREE_STATUS_UNAVAILABLE if io_uring is not usable on this system:
// kernel too old (ENOSYS), blocked by seccomp/sysctl (EPERM), insufficient
// locked memory (ENOMEM), or other io_uring_setup failures. The specific
// errno is included in the status message for diagnostics.
iree_status_t iree_async_proactor_create_io_uring(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_API_H_
