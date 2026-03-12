// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Remote HAL server: exposes local HAL devices to remote clients.
//
// The server accepts connections from remote clients and dispatches their
// operations to wrapped local HAL devices. This enables GPU resources on one
// machine to be used transparently by applications on other machines.
//
// ## Architecture
//
// The server is built on the proactor-driven iree/net/ stack:
//
//   Transport factory  →  Listener (bind + accept)
//     → Connection  →  Session (HELLO/HELLO_ACK bootstrap)
//       → Control channel (HAL commands, device queries)
//       → Queue endpoints (frontier-ordered submissions)
//       → Bulk endpoints (buffer transfers, optionally RDMA)
//
// All I/O is fully asynchronous. The server has no poll loop and no threads
// of its own — the caller's proactor drives all network I/O and callback
// dispatch. Frontiers (vector clocks) track causal dependencies across
// the distributed system without centralized coordination.
//
// ## Usage
//
//   // Set up async infrastructure.
//   iree_async_proactor_t* proactor = ...;
//   iree_async_buffer_pool_t* recv_pool = ...;
//   iree_async_frontier_tracker_t* tracker = ...;
//
//   // Set up local devices and build topology.
//   iree_hal_device_t* devices[] = {gpu0, gpu1};
//   iree_net_session_topology_t topology = ...;  // axes for device queues
//
//   // Configure and create.
//   iree_hal_remote_server_options_t options;
//   iree_hal_remote_server_options_initialize(&options);
//   options.transport_factory = tcp_factory;
//   options.bind_address = iree_make_cstring_view("0.0.0.0:5000");
//   options.local_topology = &topology;
//
//   iree_hal_remote_server_t* server = NULL;
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_create(
//       &options, devices, IREE_ARRAYSIZE(devices),
//       proactor, tracker, recv_pool,
//       host_allocator, &server));
//
//   // Start accepting connections (proactor drives everything from here).
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_start(server));
//
// ## Multi-Client Support
//
// The server can accept multiple concurrent client connections. Each client
// gets an independent session with its own resource namespace. The
// max_connections option controls the upper bound.
//
// ## Multi-Device Support
//
// The server exposes one or more local HAL devices. During session bootstrap,
// the server advertises topology (axes per device queue) so clients can
// construct correct frontiers for cross-device causal ordering.
//
// ## Thread Safety
//
// create() and the returned server pointer may be used from any thread.
// start() and stop() must be called from the proactor thread (or before the
// proactor starts polling, during setup). All session callbacks fire on the
// proactor thread.

#ifndef IREE_HAL_REMOTE_SERVER_API_H_
#define IREE_HAL_REMOTE_SERVER_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;
typedef struct iree_net_session_topology_t iree_net_session_topology_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_async_buffer_pool_t iree_async_buffer_pool_t;

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_t
//===----------------------------------------------------------------------===//

// Flags controlling remote server behavior.
enum iree_hal_remote_server_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_FLAG_NONE = 0u,
  // Enables RDMA for bulk transfers when available.
  // Falls back to the transport's default bulk transfer if RDMA is not
  // supported.
  IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA = 1u << 0,
  // Enables tracing of server operations for debugging.
  IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS = 1u << 1,
};
typedef uint32_t iree_hal_remote_server_flags_t;

// Parameters for configuring an iree_hal_remote_server_t.
// Must be initialized with iree_hal_remote_server_options_initialize prior to
// use.
typedef struct iree_hal_remote_server_options_t {
  // Transport factory for creating the server listener and accepting
  // connections. The server retains this factory on creation and releases it
  // on destroy.
  // Required — must not be NULL.
  iree_net_transport_factory_t* transport_factory;

  // Address to bind the server to.
  // Format depends on the transport:
  //   TCP/QUIC: "host:port" (e.g., "0.0.0.0:5000", "0.0.0.0:0" for dynamic)
  //   SHM: segment name (e.g., "test-server")
  // Required — must not be empty.
  iree_string_view_t bind_address;

  // Local topology to advertise during session bootstrap.
  //
  // Describes the axes (device queues) hosted by this server. Each connecting
  // client receives this topology in the HELLO_ACK and creates proxy
  // semaphores for the server's axes. The caller constructs the topology
  // from their knowledge of the wrapped devices' queue layouts.
  //
  // Typical construction: one axis per device queue, created via
  // iree_async_axis_make_queue(session_epoch, machine_index, device_index,
  // queue_index), with current_epochs set to the device's current progress.
  //
  // The server copies this data at creation time — the caller's storage does
  // not need to outlive the create() call.
  // Required — must not be NULL.
  const iree_net_session_topology_t* local_topology;

  // Maximum number of concurrent client connections.
  // Zero uses IREE_HAL_REMOTE_DEFAULT_MAX_CONNECTIONS.
  uint32_t max_connections;

  // Flags controlling server behavior.
  iree_hal_remote_server_flags_t flags;
} iree_hal_remote_server_options_t;

// Default maximum concurrent connections.
#define IREE_HAL_REMOTE_DEFAULT_MAX_CONNECTIONS 16u

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_remote_server_options_initialize(
    iree_hal_remote_server_options_t* out_options);

// Parses |params| and updates |options|.
// String views may reference strings in the original parameters; the caller
// must ensure the options struct does not outlive the parameter storage.
//
// Recognized parameters:
//   bind=<address>        Bind address (format depends on transport)
//   max_connections=<n>   Maximum concurrent connections
//   rdma=true|false       Enable/disable RDMA for bulk transfers
//   trace=true|false      Enable server operation tracing
IREE_API_EXPORT iree_status_t iree_hal_remote_server_options_parse(
    iree_hal_remote_server_options_t* options, iree_string_pair_list_t params);

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;

// Creates a remote HAL server that exposes the provided devices to clients.
//
// The server does not start listening immediately; call
// iree_hal_remote_server_start() to begin accepting connections.
//
// |devices| is an array of |device_count| HAL devices to expose. Each device
// is retained by the server. |device_count| must be >= 1.
//
// |proactor| drives all server I/O (listener, sessions, callbacks). Borrowed —
// must outlive the server.
//
// |frontier_tracker| tracks axis progress for cross-device causal ordering.
// The server registers device queue axes during creation and remote client
// axes during session bootstrap. Borrowed — must outlive the server.
//
// |recv_pool| provides buffers for incoming network data. Borrowed — must
// outlive the server.
//
// |out_server| must be released by the caller.
IREE_API_EXPORT iree_status_t iree_hal_remote_server_create(
    const iree_hal_remote_server_options_t* options,
    iree_hal_device_t* const* devices, iree_host_size_t device_count,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_remote_server_t** out_server);

// Retains a reference to the server.
IREE_API_EXPORT void iree_hal_remote_server_retain(
    iree_hal_remote_server_t* server);

// Releases a reference to the server.
// The server must be stopped before the last reference is released.
IREE_API_EXPORT void iree_hal_remote_server_release(
    iree_hal_remote_server_t* server);

// Starts the server listening for connections.
//
// Creates a listener via the transport factory, bound to the configured
// address. Incoming connections trigger session creation and bootstrap.
//
// Must be called from the proactor thread (or during setup before the proactor
// starts polling). Returns OK if the listener was created successfully.
// Asynchronous errors during accept are reported via session error callbacks.
//
// Requires STOPPED state. Returns FAILED_PRECONDITION otherwise.
IREE_API_EXPORT iree_status_t
iree_hal_remote_server_start(iree_hal_remote_server_t* server);

// Initiates graceful shutdown of the server.
//
// Sends GOAWAY to all active sessions, stops the listener, and waits for
// sessions to drain. The server transitions to STOPPING and then to STOPPED
// when all sessions have closed.
//
// The |callback| fires on the proactor thread when shutdown is complete
// (all sessions closed, listener freed). After the callback fires, the server
// is in STOPPED state and can be released.
//
// Requires RUNNING state. Returns FAILED_PRECONDITION otherwise.
typedef void (*iree_hal_remote_server_stopped_fn_t)(void* user_data);
typedef struct iree_hal_remote_server_stopped_callback_t {
  iree_hal_remote_server_stopped_fn_t fn;
  void* user_data;
} iree_hal_remote_server_stopped_callback_t;
IREE_API_EXPORT iree_status_t
iree_hal_remote_server_stop(iree_hal_remote_server_t* server,
                            iree_hal_remote_server_stopped_callback_t callback);

// Queries the address the server is bound to.
//
// After start(), this returns the actual bound address (useful when binding
// to a dynamic port like "0.0.0.0:0"). The result is written into the
// caller-provided |buffer| and |out_address| is set to a view into it.
//
// Requires RUNNING state. Returns FAILED_PRECONDITION otherwise.
IREE_API_EXPORT iree_status_t iree_hal_remote_server_query_bound_address(
    iree_hal_remote_server_t* server, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_API_H_
