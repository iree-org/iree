// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Framing adapter: wraps a byte-stream carrier to provide message-oriented
// delivery via the message_endpoint interface.
//
// The adapter sits between carriers (raw byte streams) and channels (protocols)
// by adding frame boundary detection to byte-stream transports like TCP/QUIC.
// For already message-oriented transports (UDP, RDMA SEND/RECV), see the
// native carrier-to-endpoint adapters instead.
//
// ## Zero-copy optimization
//
// When a complete frame arrives in a single receive buffer, the frame is
// delivered directly with the buffer lease for zero-copy processing. The
// handler can retain the lease to defer processing.
//
// When frames span multiple buffers (TCP stream fragmentation), the adapter
// reassembles fragments in the embedded frame_accumulator. For these copy-path
// frames, the adapter acquires a buffer from the reassembly pool and delivers
// that buffer's lease to the handler, maintaining the message_endpoint contract
// that lease is always non-NULL.
//
// ## Ownership model
//
// The adapter takes ownership of the carrier (releases it when freed). The
// reassembly pool is referenced but not owned - caller must keep it alive for
// the adapter's lifetime.
//
// ## Usage
//
// All operations after allocation go through the message_endpoint interface:
//
//   iree_net_framing_adapter_t* adapter = NULL;
//   IREE_RETURN_IF_ERROR(iree_net_framing_adapter_allocate(
//       carrier, frame_length, max_frame_size, pool, allocator, &adapter));
//   iree_net_message_endpoint_t endpoint =
//       iree_net_framing_adapter_as_endpoint(adapter);
//   iree_net_message_endpoint_set_callbacks(endpoint, callbacks);
//   IREE_RETURN_IF_ERROR(iree_net_message_endpoint_activate(endpoint));
//   // ... send/recv via endpoint ...
//   IREE_RETURN_IF_ERROR(iree_net_message_endpoint_deactivate(
//       endpoint, on_deactivated, user_data));
//   // ... after deactivation callback fires ...
//   iree_net_framing_adapter_free(adapter);

#ifndef IREE_NET_CHANNEL_UTIL_FRAMING_ADAPTER_H_
#define IREE_NET_CHANNEL_UTIL_FRAMING_ADAPTER_H_

#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/channel/util/frame_accumulator.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_framing_adapter_t iree_net_framing_adapter_t;

// Allocates a framing adapter over a byte-stream carrier.
//
// Takes ownership of |carrier|, which will be released when the adapter is
// freed. The carrier must not be activated before passing to this function.
//
// The |frame_length| callback determines frame boundaries by examining partial
// data and returning the total frame size when determinable, or 0 if more bytes
// are needed. See iree_net_frame_length_fn_t documentation for protocol
// requirements.
//
// The |reassembly_pool| is used to allocate buffers for copy-path frames (when
// frames span receive buffers). The pool is referenced but not owned - caller
// must keep it alive for the adapter's lifetime. Pool buffer size must be at
// least |max_frame_size| bytes.
//
// The |max_frame_size| limits the largest frame the adapter can handle. Frames
// larger than this return IREE_STATUS_RESOURCE_EXHAUSTED from the recv path.
iree_status_t iree_net_framing_adapter_allocate(
    iree_net_carrier_t* carrier, iree_net_frame_length_callback_t frame_length,
    iree_host_size_t max_frame_size, iree_async_buffer_pool_t* reassembly_pool,
    iree_allocator_t host_allocator, iree_net_framing_adapter_t** out_adapter);

// Frees the adapter and releases the owned carrier.
//
// The adapter must be deactivated before freeing. Freeing an active adapter is
// a programming error and triggers an assertion failure.
void iree_net_framing_adapter_free(iree_net_framing_adapter_t* adapter);

// Returns a borrowed message_endpoint view into this adapter.
//
// The returned endpoint is a lightweight handle (two pointers) that can be
// copied by value. It is valid only while the adapter is alive - there is no
// retain/release. When the adapter is freed, all endpoint references become
// invalid.
//
// All operations (set_callbacks, activate, deactivate, send, query_send_budget)
// are accessed through the returned endpoint's vtable.
iree_net_message_endpoint_t iree_net_framing_adapter_as_endpoint(
    iree_net_framing_adapter_t* adapter);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_UTIL_FRAMING_ADAPTER_H_
