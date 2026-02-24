// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Public API for the IOCP proactor backend.
//
// This header exposes only what external users need: the create function.
// Internal implementation details are in proactor.h (not exported).

#ifndef IREE_ASYNC_PLATFORM_IOCP_API_H_
#define IREE_ASYNC_PLATFORM_IOCP_API_H_

#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates an IOCP-based proactor implementation for Windows.
//
// IOCP (I/O Completion Ports) provides completion-based async I/O:
//   - Overlapped I/O: kernel performs I/O and posts completion packets
//   - GetQueuedCompletionStatusEx: batch dequeue of completions
//   - PostQueuedCompletionStatus: cross-thread wakeup and messaging
//   - AcceptEx/ConnectEx: fully async socket operations
//   - RegisterWaitForSingleObject: event/handle monitoring
//
// No worker threads are needed: IOCP is natively completion-based, so the
// kernel performs I/O directly and posts results to the completion port.
//
// Requires Windows Vista+ (GetQueuedCompletionStatusEx, CancelIoEx).
// Sync notification waits use WaitOnAddress (Windows 8+).
//
// Returns IREE_STATUS_UNAVAILABLE if IOCP creation fails.
iree_status_t iree_async_proactor_create_iocp(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IOCP_API_H_
