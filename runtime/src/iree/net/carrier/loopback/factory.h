// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loopback transport factory: in-memory transport for testing.
//
// The loopback factory creates in-memory carrier pairs for testing without
// network dependencies. It implements the same iree_net_transport_factory_t
// interface as TCP, QUIC, and RDMA factories, validating that the transport
// abstractions are not biased toward any specific transport.
//
// Addressing uses named endpoints (e.g., "test", "server") rather than
// IP addresses. Listeners register under a name, and connect looks up
// listeners by name in the factory's internal table.
//
// All callbacks (connect, accept, carrier ready, listener stopped) are
// delivered asynchronously via the proactor, matching the behavioral contract
// of all transport factory implementations.

#ifndef IREE_NET_CARRIER_LOOPBACK_FACTORY_H_
#define IREE_NET_CARRIER_LOOPBACK_FACTORY_H_

#include "iree/base/api.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Allocates a loopback transport factory for in-memory testing.
//
// The factory manages a table of named listeners. When connect is called,
// it looks up the listener by name and creates an in-memory carrier pair.
//
// Capabilities: RELIABLE | ORDERED (same behavioral guarantees as TCP).
IREE_API_EXPORT iree_status_t
iree_net_loopback_factory_allocate(iree_allocator_t host_allocator,
                                   iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_LOOPBACK_FACTORY_H_
