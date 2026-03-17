// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Polymorphic connection interface for transport-agnostic endpoint creation.
//
// A connection represents a logical link to a remote endpoint. It's the result
// of a successful `connect()` from a transport factory, and the thing that
// spawns message endpoints. This abstraction enables polymorphic multi-stream
// sessions: session code doesn't need to know if it's using QUIC (one TLS
// handshake, many streams) or TCP (stream mux over single socket). It just
// opens endpoints.
//
// Each transport handles multiplexing internally:
//
//   | Transport | open_endpoint() behavior |
//   |-----------|--------------------------|
//   | QUIC      | Opens new stream on existing TLS connection |
//   | TCP       | Allocates stream slot in mux over single socket |
//   | RDMA      | Opens new QP on existing protection domain |
//
// Ownership:
//   - Connection uses create/retain/release pattern.
//   - connect() callback receives connection with ref_count=1.
//   - open_endpoint() returns borrowed-view message endpoints. The connection
//     must outlive all endpoints. Deactivate endpoints before releasing.

#ifndef IREE_NET_CONNECTION_H_
#define IREE_NET_CONNECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/net/message_endpoint.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Endpoint ready callback
//===----------------------------------------------------------------------===//

// Callback invoked when an endpoint is ready after open_endpoint().
//
// On success, |status| is OK and |endpoint| is a valid borrowed view into the
// connection's transport stack. The endpoint is valid only while the connection
// is alive. Callers must deactivate the endpoint before releasing the
// connection.
//
// On failure, |status| contains the error and |endpoint| has self=NULL.
typedef void (*iree_net_endpoint_ready_callback_t)(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint);

//===----------------------------------------------------------------------===//
// iree_net_connection_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_connection_t iree_net_connection_t;
typedef struct iree_net_connection_vtable_t iree_net_connection_vtable_t;

// A polymorphic connection to a remote endpoint.
//
// Connections are created by transport factories via connect() or
// accepted by listeners. They provide a unified interface for opening
// message endpoints regardless of the underlying transport.
//
// Concrete implementations (TCP connection, QUIC connection, RDMA connection)
// embed this structure at offset 0.
struct iree_net_connection_t {
  iree_atomic_ref_count_t ref_count;
  const iree_net_connection_vtable_t* vtable;
  iree_allocator_t host_allocator;
};

struct iree_net_connection_vtable_t {
  void (*destroy)(iree_net_connection_t* connection);
  iree_status_t (*open_endpoint)(iree_net_connection_t* connection,
                                 iree_net_endpoint_ready_callback_t callback,
                                 void* user_data);
};

// Initializes base connection fields. Called by connection implementations.
static inline void iree_net_connection_initialize(
    const iree_net_connection_vtable_t* vtable, iree_allocator_t host_allocator,
    iree_net_connection_t* out_connection) {
  iree_atomic_ref_count_init(&out_connection->ref_count);
  out_connection->vtable = vtable;
  out_connection->host_allocator = host_allocator;
}

// Retains a reference to the connection (thread-safe).
static inline void iree_net_connection_retain(
    iree_net_connection_t* connection) {
  iree_atomic_ref_count_inc(&connection->ref_count);
}

// Releases a reference to the connection (thread-safe).
// When the last reference is released, the connection is destroyed.
static inline void iree_net_connection_release(
    iree_net_connection_t* connection) {
  if (iree_atomic_ref_count_dec(&connection->ref_count) == 1) {
    connection->vtable->destroy(connection);
  }
}

// Opens a new message endpoint on this connection.
//
// The transport handles multiplexing internally — callers don't need to know
// the underlying transport's multiplexing strategy.
//
// The |callback| fires exactly once via the proactor when the endpoint is ready
// or creation fails. The callback is always delivered asynchronously, never
// synchronously from this call. On success, the callback receives a borrowed
// endpoint view that is valid for the lifetime of the connection.
//
// Multiple endpoints can be opened on a single connection for different
// channels (control, queue, bulk). Each call allocates an independent stream
// slot. Returns RESOURCE_EXHAUSTED if the maximum stream count is reached.
static inline iree_status_t iree_net_connection_open_endpoint(
    iree_net_connection_t* connection,
    iree_net_endpoint_ready_callback_t callback, void* user_data) {
  return connection->vtable->open_endpoint(connection, callback, user_data);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CONNECTION_H_
