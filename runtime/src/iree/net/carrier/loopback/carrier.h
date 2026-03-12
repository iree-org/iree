// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loopback carrier: in-memory transport for testing.
//
// The loopback carrier provides an in-process carrier for testing without
// network dependencies. It's essential for CI environments and fast iteration
// during development.
//
// Design:
//   - Creates a connected pair of carriers that point to each other.
//   - send() copies data into a carrier-owned buffer and submits a NOP to the
//     proactor. Delivery occurs during poll() when the NOP completes. This
//     mirrors real carrier behavior where sends are async and recv handlers
//     fire from the proactor thread during poll().
//   - Data is always copied during send() (no zero-copy TX), matching real
//     carriers where send() consumes data into kernel/hardware buffers. The
//     sender's buffer can be freed immediately after send() returns.
//   - Send budget is finite (32 slots) to provide realistic backpressure.
//
// Capabilities:
//   - RELIABLE: No drops (in-memory).
//   - ORDERED: FIFO (NOP completion order matches submission order).
//
// Thread safety:
//   - send() is thread-safe (CAS-based slot claiming, same as TCP carrier).
//   - All completions fire from the proactor poll thread.

#ifndef IREE_NET_CARRIER_LOOPBACK_CARRIER_H_
#define IREE_NET_CARRIER_LOOPBACK_CARRIER_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a connected pair of loopback carriers.
//
// Both carriers share the same proactor. NOP operations are submitted through
// it so that send completions and recv handler invocations fire during poll(),
// matching real carrier timing.
//
// |callback| is optional and receives send completion notifications fired
// during poll(). Pass a callback with fn=NULL to skip completion callbacks.
//
// Typical usage:
//   1. Create pair with this function.
//   2. Set recv handlers on both carriers.
//   3. Activate both carriers.
//   4. Send data (delivered to peer during the next poll() cycle).
//   5. Deactivate and release when done.
IREE_API_EXPORT iree_status_t iree_net_loopback_carrier_create_pair(
    iree_async_proactor_t* proactor, iree_net_carrier_callback_t callback,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_client,
    iree_net_carrier_t** out_server);

// Sets a handler invoked when the peer carrier disconnects (deactivates or is
// destroyed). The notification fires asynchronously via the proactor, providing
// the loopback equivalent of TCP's ECONNRESET or SHM's peer departure
// detection.
//
// The handler receives an UNAVAILABLE status. It fires at most once per carrier
// (the peer link is cleared after notification).
//
// Set before activation so the handler is in place when the carrier becomes
// active.
IREE_API_EXPORT void iree_net_loopback_carrier_set_peer_disconnect_handler(
    iree_net_carrier_t* base_carrier,
    void (*fn)(void* user_data, iree_status_t status), void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_LOOPBACK_CARRIER_H_
