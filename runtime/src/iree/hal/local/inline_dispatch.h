// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Inline single-threaded dispatch for CPU executables.
//
// Executes a dispatch synchronously on the calling thread without command
// buffer intermediation. Maps binding buffers, populates the executable
// dispatch state, iterates all workgroups, and unmaps.
//
// Used by local_sync for queue_dispatch and by local_task for inline
// dispatches (ALLOW_INLINE_EXECUTION with budget-1 processes).

#ifndef IREE_HAL_LOCAL_INLINE_DISPATCH_H_
#define IREE_HAL_LOCAL_INLINE_DISPATCH_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Executes a dispatch inline on the calling thread.
//
// Maps all binding buffers with SCOPED lifetime, populates the executable's
// dispatch state, iterates all workgroups via issue_dispatch_inline, and
// unmaps. Returns the dispatch status (first workgroup failure).
//
// The executable must be a local executable (iree_hal_local_executable_t).
// Binding buffers must support host-visible mapping.
iree_status_t iree_hal_local_executable_dispatch_inline(
    iree_hal_executable_t* executable,
    iree_hal_executable_export_ordinal_t export_ordinal,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    const iree_hal_buffer_ref_t* bindings, iree_host_size_t binding_count,
    iree_hal_dispatch_flags_t flags);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_LOCAL_INLINE_DISPATCH_H_
