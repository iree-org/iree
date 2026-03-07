// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP carrier: reliable, ordered byte stream transport.
//
// Integrates with io_uring async layer for maximum efficiency:
//   - Zero-copy sends via SEND_ZC when available
//   - Kernel-managed receive buffers via PBUF_RING
//   - Multishot receives to eliminate per-recv syscall overhead
//
// The TCP carrier uses a handler-driven receive model: call set_recv_handler()
// to register a callback, then activate() to begin receiving. Data flows to
// the handler automatically without explicit recv() calls.
//
// Thread safety:
//   - send() and query_send_budget() are safe from any thread
//   - set_recv_handler() must be called before activate() (not concurrent)
//   - All completions fire on the proactor poll thread
//
// Lifecycle:
//   1. allocate() - Create carrier with socket and options
//   2. set_recv_handler() - Register data handler
//   3. activate() - Begin receiving data
//   4. send() - Send data (any thread)
//   5. deactivate() - Drain operations, wait for callback
//   6. release() - Destroy carrier

#ifndef IREE_NET_CARRIER_TCP_CARRIER_H_
#define IREE_NET_CARRIER_TCP_CARRIER_H_

#include "iree/async/api.h"
#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Options for TCP carrier creation.
typedef struct iree_net_tcp_carrier_options_t {
  // Number of concurrent send operations. Must be power of 2.
  // Higher values reduce contention but increase memory usage.
  uint32_t send_slot_count;

  // For single-shot fallback: number of recv operations to keep posted.
  // Ignored when multishot PBUF_RING is available. Must be power of 2.
  uint32_t single_shot_recv_count;

  // If true, prefer multishot recv with PBUF_RING when available.
  bool prefer_multishot_recv;

  // If true, prefer zero-copy send when available. May have higher latency
  // for small messages.
  bool prefer_zero_copy_send;

  // Maximum message endpoints per connection. Determines the stream table size
  // for mux dispatch. A value of 1 provides a single endpoint with zero mux
  // overhead.
  uint16_t max_endpoint_count;
} iree_net_tcp_carrier_options_t;

// Returns default options for TCP carrier creation.
static inline iree_net_tcp_carrier_options_t
iree_net_tcp_carrier_options_default(void) {
  iree_net_tcp_carrier_options_t options;
  memset(&options, 0, sizeof(options));
  options.send_slot_count = 64;
  options.single_shot_recv_count = 8;
  options.prefer_multishot_recv = true;
  options.prefer_zero_copy_send = true;
  options.max_endpoint_count = 1;
  return options;
}

// Creates a TCP carrier for the given connected socket.
//
// On success, the carrier takes ownership of |socket|. On failure, the socket
// is NOT consumed and the caller retains ownership. The socket must be in
// connected state.
//
// |recv_pool| is NOT owned by the carrier. The caller must ensure the pool
// outlives the carrier. The pool is used for receive buffer allocation.
//
// Capabilities reported depend on proactor features:
//   - RELIABLE: Always set (TCP guarantees delivery)
//   - ORDERED: Always set (TCP guarantees ordering)
//   - ZERO_COPY_TX: Set if SEND_ZC available
//   - ZERO_COPY_RX: Set if PBUF_RING available
IREE_API_EXPORT iree_status_t iree_net_tcp_carrier_allocate(
    iree_async_proactor_t* proactor, iree_async_socket_t* socket,
    iree_async_buffer_pool_t* recv_pool, iree_net_tcp_carrier_options_t options,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_TCP_CARRIER_H_
