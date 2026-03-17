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
// HAL semaphores are the application-facing synchronization primitive.
// Frontiers are the transport-layer ordering mechanism. Each proxy semaphore
// maintains a (value → axis, epoch) mapping that enables the client to
// translate wait_semaphore_list entries into wait frontier entries on the wire.
// The mapping is populated during signal waiter registration (when we know
// which epoch will produce which semaphore value) and queried when building
// wait frontiers for subsequent queue operations.

#ifndef IREE_HAL_REMOTE_CLIENT_SEMAPHORE_H_
#define IREE_HAL_REMOTE_CLIENT_SEMAPHORE_H_

#include "iree/async/frontier.h"
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

// Records that signaling |semaphore| to |value| will be produced by
// submission epoch |epoch| on |axis|. Called during signal waiter
// registration so that subsequent queue operations can translate
// wait_semaphore_list entries into wait frontier entries.
//
// Values must be recorded in monotonically increasing order (which is
// guaranteed by the HAL semaphore monotonicity invariant).
void iree_hal_remote_client_semaphore_record_epoch(
    iree_hal_semaphore_t* semaphore, uint64_t value, iree_async_axis_t axis,
    uint64_t epoch);

// Looks up the (axis, epoch) that will produce |semaphore| reaching at
// least |value|. Returns true if found. Returns false if the semaphore has
// no epoch mapping for this value (host-signaled, different device, etc.).
//
// The lookup finds the first recorded signal value >= |value| and returns
// its corresponding axis and epoch. This works because signal values are
// monotonically increasing: if the semaphore will be signaled to values
// 1, 5, 10, then waiting for value 3 is satisfied when value 5 arrives
// (epoch for signal value 5).
bool iree_hal_remote_client_semaphore_lookup_epoch(
    iree_hal_semaphore_t* semaphore, uint64_t value,
    iree_async_axis_t* out_axis, uint64_t* out_epoch);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_SEMAPHORE_H_
