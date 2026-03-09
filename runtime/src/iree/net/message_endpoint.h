// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Message endpoint: unified interface for sending/receiving complete messages.
//
// This abstraction sits between carriers (raw bytes) and channels (protocols).
// It handles the difference between transports that need framing (TCP, QUIC)
// and those that are already message-oriented (UDP, RDMA SEND/RECV).
//
// Implementations:
//   - framing_adapter: Wraps byte-stream carrier, parses frame boundaries
//   - stream_mux: Demultiplexes by stream_id over a shared endpoint
//   - UDP/RDMA carriers: Native message-oriented, thin wrapper
//
// ## Borrowed View Semantics
//
// IMPORTANT: Message endpoints are BORROWED VIEWS, not owned objects. Callers
// never free an endpoint directly. The endpoint is valid only while the
// underlying object (framing_adapter, stream_mux, carrier) is alive.
//
// Conversion functions like framing_adapter_as_endpoint() return stack-copyable
// structs that point into the underlying object. When that object is freed,
// all endpoint references become invalid.
//
// ## Protocol Handoff
//
// During connection bootstrap, ownership of an endpoint transfers from the
// bootstrap handler to the operational protocol. Use set_callbacks() to
// atomically swap both message and error handlers, ensuring no messages are
// delivered to a stale handler.

#ifndef IREE_NET_MESSAGE_ENDPOINT_H_
#define IREE_NET_MESSAGE_ENDPOINT_H_

#include "iree/async/api.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Callbacks and parameters
//===----------------------------------------------------------------------===//

// Message handler invoked when a complete message is received.
//
// Called on the proactor thread for each complete message. The handler receives
// a view of the message data and a lease to the backing storage. The lease is
// always valid (non-NULL) whether the message came from a recv buffer or was
// reassembled from fragments.
//
// To keep the message data valid beyond the callback, retain the lease via
// iree_async_buffer_lease_retain(). Release it when done processing.
//
// Return iree_ok_status() to continue receiving. Returning an error triggers
// the endpoint's error handler and may cause deactivation.
typedef iree_status_t (*iree_net_message_endpoint_message_fn_t)(
    void* user_data, iree_const_byte_span_t message,
    iree_async_buffer_lease_t* lease);

// Error handler invoked when the endpoint encounters a transport error.
//
// Called on the proactor thread. After this callback, the endpoint may be in
// a failed state and subsequent operations may fail. The status ownership is
// transferred to the handler (must be consumed or ignored).
typedef void (*iree_net_message_endpoint_error_fn_t)(void* user_data,
                                                     iree_status_t status);

// Deactivation complete callback invoked when graceful shutdown finishes.
//
// After this callback fires, the endpoint is fully drained and operations will
// fail. The underlying object can now be safely freed.
typedef void (*iree_net_message_endpoint_deactivate_fn_t)(void* user_data);

// Bundled message and error handlers for atomic handoff.
//
// During protocol transitions (e.g., bootstrap to operational), both handlers
// must change atomically to prevent messages from being delivered to a stale
// handler. The shared user_data ensures consistency between both callbacks.
typedef struct iree_net_message_endpoint_callbacks_t {
  iree_net_message_endpoint_message_fn_t on_message;
  iree_net_message_endpoint_error_fn_t on_error;
  void* user_data;
} iree_net_message_endpoint_callbacks_t;

// Parameters for send operations.
typedef struct iree_net_message_endpoint_send_params_t {
  // Scatter-gather list of message data to send.
  iree_async_span_list_t data;
  // User data echoed to the completion callback for correlation.
  uint64_t user_data;
} iree_net_message_endpoint_send_params_t;

//===----------------------------------------------------------------------===//
// iree_net_message_endpoint_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_message_endpoint_vtable_t
    iree_net_message_endpoint_vtable_t;

// A borrowed view into a message-oriented transport endpoint.
//
// This is a lightweight handle (two pointers) that can be copied by value.
// The endpoint is valid only while the underlying object is alive. There is
// no retain/release - when the underlying object is freed, the endpoint
// becomes invalid.
typedef struct iree_net_message_endpoint_t {
  void* self;
  const iree_net_message_endpoint_vtable_t* vtable;
} iree_net_message_endpoint_t;

struct iree_net_message_endpoint_vtable_t {
  void (*set_callbacks)(void* self,
                        iree_net_message_endpoint_callbacks_t callbacks);
  iree_status_t (*activate)(void* self);
  iree_status_t (*deactivate)(
      void* self, iree_net_message_endpoint_deactivate_fn_t callback,
      void* user_data);
  iree_status_t (*send)(void* self,
                        const iree_net_message_endpoint_send_params_t* params);
  iree_net_carrier_send_budget_t (*query_send_budget)(void* self);

  // Direct-write send mode: caller writes into transport buffer.
  iree_status_t (*begin_send)(void* self, iree_host_size_t size, void** out_ptr,
                              iree_net_carrier_send_handle_t* out_handle);
  iree_status_t (*commit_send)(void* self,
                               iree_net_carrier_send_handle_t handle);
  void (*abort_send)(void* self, iree_net_carrier_send_handle_t handle);
};

// Sets message and error handlers atomically.
//
// Used for protocol handoff (e.g., bootstrap completes, operational channel
// takes over). Both handlers and user_data change in a single operation,
// ensuring no messages are delivered to a stale handler.
//
// Must be called on the proactor thread after activation, or from any thread
// before activation.
static inline void iree_net_message_endpoint_set_callbacks(
    iree_net_message_endpoint_t endpoint,
    iree_net_message_endpoint_callbacks_t callbacks) {
  endpoint.vtable->set_callbacks(endpoint.self, callbacks);
}

// Activates the endpoint, enabling message receipt.
//
// After activation, the endpoint auto-receives and delivers complete messages
// to the on_message handler. The callbacks must be set before calling this.
//
// Returns IREE_STATUS_FAILED_PRECONDITION if callbacks are not set.
static inline iree_status_t iree_net_message_endpoint_activate(
    iree_net_message_endpoint_t endpoint) {
  return endpoint.vtable->activate(endpoint.self);
}

// Begins graceful deactivation of the endpoint.
//
// This drains outstanding operations and stops receiving new messages. The
// callback fires when deactivation completes and the endpoint is safe to
// abandon. After the callback, operations on this endpoint will fail.
//
// Deactivation is non-blocking - the callback fires asynchronously on the
// proactor thread when all pending operations have completed.
static inline iree_status_t iree_net_message_endpoint_deactivate(
    iree_net_message_endpoint_t endpoint,
    iree_net_message_endpoint_deactivate_fn_t callback, void* user_data) {
  return endpoint.vtable->deactivate(endpoint.self, callback, user_data);
}

// Sends a message via the endpoint.
//
// The message data in |params->data| is sent as a complete message. For
// framing adapters, this prepends the appropriate frame header. For native
// message-oriented transports, this passes through directly.
//
// Completion is delivered via the underlying object's completion callback
// (e.g., the carrier callback for framing_adapter). The |params->user_data|
// is echoed to the completion for correlation.
//
// The data buffers must remain valid until the completion callback fires.
static inline iree_status_t iree_net_message_endpoint_send(
    iree_net_message_endpoint_t endpoint,
    const iree_net_message_endpoint_send_params_t* params) {
  return endpoint.vtable->send(endpoint.self, params);
}

// Queries send budget for backpressure management.
//
// Returns both byte budget and operation slot budget. When either reaches
// zero, the endpoint is backpressured and callers should wait for send
// completions before submitting more operations.
static inline iree_net_carrier_send_budget_t
iree_net_message_endpoint_query_send_budget(
    iree_net_message_endpoint_t endpoint) {
  return endpoint.vtable->query_send_budget(endpoint.self);
}

// Reserves space for a contiguous send of |size| bytes.
//
// On success, |*out_ptr| points to a buffer of at least |size| bytes where the
// caller writes directly. |*out_handle| receives an opaque handle that must be
// passed to either commit_send (to publish the data) or abort_send (to discard
// the reservation).
//
// For mux endpoints (TCP streams), the returned pointer is offset past the
// frame header — the caller writes only the payload, and the endpoint prepends
// the appropriate framing on commit.
//
// Between begin_send and commit/abort, the caller holds endpoint-specific
// resources. The caller must call commit_send or abort_send promptly.
//
// No completion callback fires for begin_send/commit_send — the data is fully
// consumed on commit.
//
// |size| must be > 0.
//
// Returns RESOURCE_EXHAUSTED if the transport buffer is full.
// Returns FAILED_PRECONDITION if the endpoint is not activated.
static inline iree_status_t iree_net_message_endpoint_begin_send(
    iree_net_message_endpoint_t endpoint, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  return endpoint.vtable->begin_send(endpoint.self, size, out_ptr, out_handle);
}

// Publishes a previously reserved send, making the data visible to the peer.
//
// The data written into the buffer returned by begin_send is committed to the
// transport. After this call, the handle is consumed and the caller holds no
// endpoint resources.
//
// No completion callback fires — the transport owns the data after commit.
static inline iree_status_t iree_net_message_endpoint_commit_send(
    iree_net_message_endpoint_t endpoint,
    iree_net_carrier_send_handle_t handle) {
  return endpoint.vtable->commit_send(endpoint.self, handle);
}

// Discards a previously reserved send without publishing any data.
//
// The reserved resources are released. No data is sent to the peer.
static inline void iree_net_message_endpoint_abort_send(
    iree_net_message_endpoint_t endpoint,
    iree_net_carrier_send_handle_t handle) {
  endpoint.vtable->abort_send(endpoint.self, handle);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_MESSAGE_ENDPOINT_H_
