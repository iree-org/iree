// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Umbrella header for the iree/net/ networking API.
//
// This provides the network transport layer for IREE's remote HAL and
// distributed execution. The stack is layered:
//
//   Layer 4: Channels (control, queue, bulk) - application semantics
//   Layer 3: Pipes (framing) - channel-specific wire formats
//   Layer 2: Carriers (TCP, UDP, RDMA, loopback) - raw transport
//   Layer 1: Proactor (iree/async/) - completion-based I/O
//
// The carrier abstraction enables swapping transports (TCP, RDMA, shared
// memory) without affecting higher layers. Zero-copy paths are preserved
// through iree_async_span_t references at all layers.

#ifndef IREE_NET_API_H_
#define IREE_NET_API_H_

#include "iree/net/carrier.h"
#include "iree/net/codec.h"
#include "iree/net/message_endpoint.h"

#endif  // IREE_NET_API_H_
