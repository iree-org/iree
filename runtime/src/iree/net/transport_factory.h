// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transport factory: creates connections and listeners for a transport type.
//
// A factory is a stateful object for a specific transport (TCP, QUIC, RDMA).
// It holds shared state that would be expensive to create per-connection:
//
//   - TCP: Nothing special (sockets are cheap)
//   - QUIC: TLS context, connection reuse tracking
//   - RDMA: Loaded library handles, device list, protection domains
//
// Factories are registered with a transport registry at startup using
// HAL-driver style registration. See iree/net/transport_registry.h.
//
// Ownership:
//   - Factories are reference counted (create/retain/release). Any holder can
//     retain a factory to extend its lifetime independently.
//   - The transport registry retains factories on registration and releases
//     them when the registry is freed. Other holders may independently retain
//     factories obtained via registry lookup.
//   - Factories own shared resources (library handles, device lists).
//   - Factories do NOT own connections (callers own connections).

#ifndef IREE_NET_TRANSPORT_FACTORY_H_
#define IREE_NET_TRANSPORT_FACTORY_H_

#include "iree/async/api.h"
#include "iree/async/buffer_pool.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Transport capabilities
//===----------------------------------------------------------------------===//

// Capabilities advertised by a transport. Query these before connecting to
// determine what features are available and optimize the connection strategy.
typedef enum iree_net_transport_capability_bits_e {
  IREE_NET_TRANSPORT_CAPABILITY_NONE = 0u,

  // Transport guarantees delivery of all data in the order it was sent.
  // Without this, callers must implement their own reliability layer.
  IREE_NET_TRANSPORT_CAPABILITY_RELIABLE = 1u << 0,

  // Transport guarantees data arrives in the order it was sent.
  // Reliable transports are typically also ordered.
  IREE_NET_TRANSPORT_CAPABILITY_ORDERED = 1u << 1,

  // Transport supports native multiplexing (QUIC streams, RDMA QPs).
  // When set, opening additional carriers is cheap (no extra handshake).
  // When not set, a stream multiplexer layer handles carrier demuxing.
  IREE_NET_TRANSPORT_CAPABILITY_NATIVE_MUX = 1u << 2,

  // Transport can send data without copying from user buffers.
  IREE_NET_TRANSPORT_CAPABILITY_ZERO_COPY_TX = 1u << 3,

  // Transport can receive data directly into user-provided buffers.
  IREE_NET_TRANSPORT_CAPABILITY_ZERO_COPY_RX = 1u << 4,

  // Transport supports one-sided RDMA operations (read/write without
  // involving the remote CPU).
  IREE_NET_TRANSPORT_CAPABILITY_RDMA = 1u << 5,

  // Transport supports direct access to device memory (GPU-direct).
  IREE_NET_TRANSPORT_CAPABILITY_DEVICE_MEMORY = 1u << 6,

} iree_net_transport_capability_bits_t;
typedef uint32_t iree_net_transport_capabilities_t;

//===----------------------------------------------------------------------===//
// Callbacks
//===----------------------------------------------------------------------===//

typedef struct iree_net_connection_t iree_net_connection_t;

// Callback invoked when an async connect operation completes.
//
// On success, |status| is OK and |connection| is a new connection with
// ref_count=1. The caller takes ownership and must eventually release it.
//
// On failure, |status| contains the error and |connection| is NULL.
typedef void (*iree_net_transport_connect_callback_t)(
    void* user_data, iree_status_t status, iree_net_connection_t* connection);

// Callback invoked when a listener accepts a new incoming connection.
//
// On success, |status| is OK and |connection| is a new connection with
// ref_count=1. The caller takes ownership. This callback may fire multiple
// times as new connections arrive.
//
// On failure, |status| contains the error and |connection| is NULL. The
// listener continues accepting unless a fatal error occurs.
typedef void (*iree_net_listener_accept_callback_t)(
    void* user_data, iree_status_t status, iree_net_connection_t* connection);

//===----------------------------------------------------------------------===//
// iree_net_listener_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_listener_t iree_net_listener_t;
typedef struct iree_net_listener_vtable_t iree_net_listener_vtable_t;

// Callback invoked when a listener has fully stopped accepting connections.
//
// After this callback fires, no more accept callbacks will be delivered and
// iree_net_listener_free() is safe to call. This mirrors the carrier
// deactivate pattern where the callback signals safe-to-destroy.
typedef void (*iree_net_listener_stopped_fn_t)(void* user_data);
typedef struct iree_net_listener_stopped_callback_t {
  iree_net_listener_stopped_fn_t fn;
  void* user_data;
} iree_net_listener_stopped_callback_t;

// A network listener that accepts incoming connections.
//
// Listeners are created via iree_net_transport_factory_create_listener() and
// begin accepting connections immediately. Each accepted connection is
// delivered via the accept callback provided at creation time.
//
// Lifecycle:
//   LISTENING -> stop() -> STOPPING -> stopped callback -> STOPPED -> free()
//
// Calling free() before the stopped callback fires is a programming error.
// Implementations may have pending I/O operations referencing the listener's
// internal state; the stopped callback guarantees these have completed.
struct iree_net_listener_t {
  const iree_net_listener_vtable_t* vtable;
};

struct iree_net_listener_vtable_t {
  void (*free)(iree_net_listener_t* listener);
  iree_status_t (*stop)(iree_net_listener_t* listener,
                        iree_net_listener_stopped_callback_t callback);
  iree_status_t (*query_bound_address)(iree_net_listener_t* listener,
                                       iree_host_size_t buffer_capacity,
                                       char* buffer,
                                       iree_string_view_t* out_address);
};

// Frees a listener and releases all associated resources.
// The listener must be stopped (stopped callback must have fired) before
// freeing. Freeing before the stopped callback fires is a programming error.
static inline void iree_net_listener_free(iree_net_listener_t* listener) {
  listener->vtable->free(listener);
}

// Initiates graceful shutdown of the listener.
//
// This cancels any pending accept operations. The |callback| fires exactly
// once via the proactor when all pending accepts have drained and no more
// accept callbacks will be delivered. After the callback fires,
// iree_net_listener_free() is safe to call.
//
// The callback is always delivered asynchronously via the proactor, never
// synchronously from this call.
static inline iree_status_t iree_net_listener_stop(
    iree_net_listener_t* listener,
    iree_net_listener_stopped_callback_t callback) {
  return listener->vtable->stop(listener, callback);
}

// Queries the address the listener is bound to, in the same string format
// accepted by connect and create_listener.
//
// This is the round-trip companion to create_listener: after binding to a
// dynamic address (e.g., TCP port 0), this returns the actual assigned address
// that a client would pass to connect to reach this listener.
//
// The result is written into the caller-provided |buffer| of |buffer_capacity|
// bytes, and |out_address| is set to a view into that buffer.
//
// Transport-specific formats:
//   TCP:      "127.0.0.1:54321" (actual ephemeral port)
//   Loopback: "test" (the bind name, unchanged)
//   RDMA:     transport-specific addressing
static inline iree_status_t iree_net_listener_query_bound_address(
    iree_net_listener_t* listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  return listener->vtable->query_bound_address(listener, buffer_capacity,
                                               buffer, out_address);
}

//===----------------------------------------------------------------------===//
// iree_net_transport_factory_t
//===----------------------------------------------------------------------===//

typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;
typedef struct iree_net_transport_factory_vtable_t
    iree_net_transport_factory_vtable_t;

// A factory that creates connections and listeners for a specific transport.
//
// Each transport type (TCP, QUIC, RDMA, etc.) provides its own factory
// implementation. Factories hold shared state that would be expensive to
// create per-connection, such as TLS contexts or RDMA protection domains.
//
// Factories are typically registered with a transport registry and looked up
// by scheme (e.g., "tcp", "quic"). See iree/net/transport_registry.h.
//
// Concrete implementations embed this as their first member.
struct iree_net_transport_factory_t {
  iree_atomic_ref_count_t ref_count;
  const iree_net_transport_factory_vtable_t* vtable;
};

struct iree_net_transport_factory_vtable_t {
  void (*destroy)(iree_net_transport_factory_t* factory);
  iree_net_transport_capabilities_t (*query_capabilities)(
      iree_net_transport_factory_t* factory);
  iree_status_t (*connect)(iree_net_transport_factory_t* factory,
                           iree_string_view_t address,
                           iree_async_proactor_t* proactor,
                           iree_async_buffer_pool_t* recv_pool,
                           iree_net_transport_connect_callback_t callback,
                           void* user_data);
  iree_status_t (*create_listener)(
      iree_net_transport_factory_t* factory, iree_string_view_t bind_address,
      iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
      iree_net_listener_accept_callback_t accept_callback, void* user_data,
      iree_allocator_t host_allocator, iree_net_listener_t** out_listener);
};

// Retains a reference to the factory.
static inline void iree_net_transport_factory_retain(
    iree_net_transport_factory_t* factory) {
  if (IREE_LIKELY(factory)) {
    iree_atomic_ref_count_inc(&factory->ref_count);
  }
}

// Releases a reference to the factory.
// When the last reference is released, the factory is destroyed and all
// associated resources are freed.
static inline void iree_net_transport_factory_release(
    iree_net_transport_factory_t* factory) {
  if (IREE_LIKELY(factory) &&
      iree_atomic_ref_count_dec(&factory->ref_count) == 1) {
    factory->vtable->destroy(factory);
  }
}

// Returns the capabilities supported by this transport.
// Use this to determine what features are available before connecting, such
// as whether the transport supports zero-copy operations or RDMA.
static inline iree_net_transport_capabilities_t
iree_net_transport_factory_query_capabilities(
    iree_net_transport_factory_t* factory) {
  return factory->vtable->query_capabilities(factory);
}

// Initiates an asynchronous connection to the given address.
//
// The |address| format is transport-specific:
//   - TCP: "host:port" (e.g., "localhost:8080", "192.168.1.1:9000")
//   - QUIC: "host:port" with optional SNI
//   - RDMA: device-specific addressing
//
// The |proactor| handles I/O completions for this connection. All callbacks
// for this connection will fire on the proactor's thread.
//
// The |recv_pool| provides buffers for incoming data during connection setup.
// It must remain valid until the connection is released.
//
// The |callback| fires exactly once via the proactor when the connection
// succeeds or fails. The callback is always delivered asynchronously, never
// synchronously from this call. On success, the callback receives a new
// connection that the caller owns.
//
// Returns synchronous errors immediately (e.g., invalid address format,
// allocation failure). Asynchronous errors (connection refused, timeout) are
// delivered via the callback.
static inline iree_status_t iree_net_transport_factory_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  return factory->vtable->connect(factory, address, proactor, recv_pool,
                                  callback, user_data);
}

// Creates a listener that accepts incoming connections on the given address.
//
// The |bind_address| format is transport-specific:
//   - TCP: "host:port" (e.g., "0.0.0.0:8080" for all interfaces)
//   - QUIC: "host:port" with TLS configuration
//   - RDMA: device-specific addressing
//
// The |proactor| handles I/O completions. All accept callbacks fire on the
// proactor's thread.
//
// The |recv_pool| provides buffers for accepted connections' initial receives.
// It must remain valid for the lifetime of all accepted connections.
//
// The |accept_callback| fires each time a new connection is accepted. On
// success, the callback receives a new connection that the caller owns. The
// callback may fire multiple times, once per accepted connection.
//
// On success, |*out_listener| receives the new listener. The caller owns it
// and must eventually stop and free it.
static inline iree_status_t iree_net_transport_factory_create_listener(
    iree_net_transport_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  return factory->vtable->create_listener(factory, bind_address, proactor,
                                          recv_pool, accept_callback, user_data,
                                          host_allocator, out_listener);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_TRANSPORT_FACTORY_H_
