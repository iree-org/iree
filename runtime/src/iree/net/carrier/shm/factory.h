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
// Addressing:
//   Named endpoints (e.g., "server", "worker-0"): In-process connections.
//   Listeners register under a name, connect looks up by name. Uses
//   iree_net_shm_carrier_create_pair() to create carrier pairs directly.
//
//   Unix domain sockets (e.g., "unix:/tmp/iree.sock"): Cross-process
//   connections via SHM handshake over a Unix domain socket. Listeners bind
//   to a socket path, clients connect by path. Each accepted connection runs
//   a handshake to exchange SHM handles and notification primitives, creating
//   independent carriers on each side.
//
// Both paths share the same connection type and endpoint adapter.
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
