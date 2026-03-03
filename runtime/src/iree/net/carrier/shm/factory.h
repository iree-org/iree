// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SHM transport factory: shared memory transport via SPSC ring buffers.
//
// The SHM factory creates connections backed by shared memory carrier pairs.
// It implements the same iree_net_transport_factory_t interface as TCP and
// loopback factories, enabling transport-agnostic code to use SHM without
// special casing.
//
// In-process connections use iree_net_shm_carrier_create_pair() to create
// carrier pairs directly. Cross-process connections will exchange SHM handles
// and notification primitives over a control channel (Unix socket / named
// pipe), creating carriers independently on each side. Both paths share the
// same connection type and endpoint adapter.
//
// Addressing uses named endpoints (e.g., "server", "worker-0"). Listeners
// register under a name, and connect looks up listeners by name in the
// factory's internal table.
//
// All callbacks (connect, accept, endpoint ready, listener stopped) are
// delivered asynchronously via the proactor, matching the behavioral contract
// of all transport factory implementations.

#ifndef IREE_NET_CARRIER_SHM_FACTORY_H_
#define IREE_NET_CARRIER_SHM_FACTORY_H_

#include "iree/base/api.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Allocates an SHM transport factory.
//
// The factory manages a table of named listeners. When connect is called,
// it looks up the listener by name and creates a carrier pair using the
// provided |options| (ring capacity, etc.).
//
// Capabilities: RELIABLE | ORDERED.
IREE_API_EXPORT iree_status_t iree_net_shm_factory_allocate(
    iree_net_shm_carrier_options_t options, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_FACTORY_H_
