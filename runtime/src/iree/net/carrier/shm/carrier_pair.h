// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// In-process SHM carrier pair creation.
//
// Creates a connected pair of SHM carriers sharing a single shared memory
// region. Used for testing and for in-process IPC. Cross-process factories
// create carriers independently using iree_net_shm_carrier_create() and
// exchange SHM handles and event pair primitives via a control channel.

#ifndef IREE_NET_CARRIER_SHM_CARRIER_PAIR_H_
#define IREE_NET_CARRIER_SHM_CARRIER_PAIR_H_

#include "iree/net/carrier/shm/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a connected pair of SHM carriers sharing a single shared memory
// region.
//
// Each carrier is registered with its own proactor for notification-driven RX
// delivery and TX completion. Both proactors may be the same (common for
// single-threaded tests) or different (e.g., NUMA-local proactors).
//
// The shared memory region is created anonymously and can be transferred to
// another process via the SHM handle.
//
// Each carrier gets one shared notification (epoch in SHM, event pair for
// proactor integration). This is the same code path used by cross-process
// factories — the only difference is how handles are exchanged.
//
// |callback| receives send completion notifications (bytes confirmed consumed
// by the peer). Pass a callback with fn=NULL to skip completion callbacks.
//
// Typical usage:
//   1. Create pair with this function.
//   2. Set recv handlers on both carriers.
//   3. Activate both carriers.
//   4. Send data (delivered to peer via shared ring buffers).
//   5. Deactivate and release when done.
IREE_API_EXPORT iree_status_t iree_net_shm_carrier_create_pair(
    iree_async_proactor_t* client_proactor,
    iree_async_proactor_t* server_proactor,
    iree_net_shm_carrier_options_t options,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_client, iree_net_carrier_t** out_server);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_CARRIER_PAIR_H_
