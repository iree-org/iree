// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Client-side proxy semaphore for remote HAL devices.
//
// Provides a lightweight iree_hal_semaphore_t backed by a local atomic
// timeline value. Signal and query are purely local operations — the proxy
// does not communicate with the server. Instead, the client device maps
// semaphore signal/wait lists to frontier entries on the wire, and signals
// the proxy semaphore when the corresponding ADVANCE arrives.
//
// This is the right design: HAL semaphores are the application-facing
// synchronization primitive. Frontiers are the transport-layer ordering
// mechanism. The mapping between them lives in the client device, not in the
// semaphore itself.

#ifndef IREE_HAL_REMOTE_CLIENT_SEMAPHORE_H_
#define IREE_HAL_REMOTE_CLIENT_SEMAPHORE_H_

#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a client proxy semaphore with the given initial value.
//
// The semaphore embeds an iree_async_semaphore_t at offset 0 for toll-free
// bridging. Query and signal are lock-free atomic operations. Wait is
// supported via the async semaphore's timepoint mechanism.
//
// |proactor| is borrowed (must outlive the semaphore). It is used for
// timepoint dispatch on the async semaphore layer.
iree_status_t iree_hal_remote_client_semaphore_create(
    iree_async_proactor_t* proactor, uint64_t initial_value,
    iree_allocator_t host_allocator, iree_hal_semaphore_t** out_semaphore);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_SEMAPHORE_H_
