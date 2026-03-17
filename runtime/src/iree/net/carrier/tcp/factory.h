// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP transport factory: creates TCP connections and listeners.
//
// Addressing uses "host:port" format. Listeners bind to a local address and
// accept incoming TCP connections. connect initiates outbound TCP
// connections to remote endpoints.
//
// The factory stores default carrier options (send slot count, recv count,
// zero-copy preferences) that are applied to all carriers created through
// connections it produces. This avoids per-connection option plumbing while
// allowing per-factory tuning.
//
// Connections build their transport stack (carrier + framing adapter) eagerly
// at creation time from the connected socket. open_endpoint() returns borrowed
// message_endpoint_t views into per-stream slots within the connection. The
// carrier and framing layer are internal to the connection and not exposed.
//
// All callbacks (connect, accept, endpoint ready, listener stopped) are
// delivered asynchronously via the proactor, matching the behavioral contract
// of all transport factory implementations.

#ifndef IREE_NET_CARRIER_TCP_FACTORY_H_
#define IREE_NET_CARRIER_TCP_FACTORY_H_

#include "iree/base/api.h"
#include "iree/net/carrier/tcp/carrier.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Allocates a TCP transport factory with the given default carrier options.
//
// The |default_options| are stored and applied to all carriers created through
// connections produced by this factory. Use
// iree_net_tcp_carrier_options_default for sensible defaults.
//
// Capabilities: RELIABLE | ORDERED (always), ZERO_COPY_TX and ZERO_COPY_RX
// depend on proactor capabilities at query time.
IREE_API_EXPORT iree_status_t
iree_net_tcp_factory_allocate(iree_net_tcp_carrier_options_t default_options,
                              iree_allocator_t host_allocator,
                              iree_net_transport_factory_t** out_factory);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_TCP_FACTORY_H_
