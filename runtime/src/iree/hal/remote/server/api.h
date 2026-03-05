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
// ## Transports
//
// The server accepts an iree_net_transport_factory_t at creation time. The
// caller creates the appropriate factory based on the desired transport (TCP,
// SHM, etc.) and passes it in the options struct. The server retains the
// factory and releases it on destroy.
//
// ## Usage
//
//   iree_hal_device_t* local_device = ...;
//
//   iree_hal_remote_server_options_t options;
//   iree_hal_remote_server_options_initialize(&options);
//   options.bind_address = iree_make_cstring_view("0.0.0.0:5000");
//
//   iree_hal_remote_server_t* server = NULL;
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_create(
//       &options, local_device, host_allocator, &server));
//
//   // Start is mandatory — server is inert until started.
//   // start() is async: uses the proactor for non-blocking I/O.
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_start(server));
//
// ## Protocol
//
// Communication uses a custom binary protocol optimized for low latency and
// high throughput. The protocol is layered on the iree/net/ transport stack:
//
//   - Control channel: Session lifecycle, capability negotiation, errors.
//   - Queue endpoints: Frontier-ordered HAL commands with causal consistency.
//   - Bulk endpoints: Large buffer transfers, optionally via RDMA.
//
// All operations are asynchronous. Frontiers (vector clocks) track causal
// dependencies across the distributed system without centralized coordination.
//
// ## Multi-Client Support
//
// The server can accept multiple concurrent client connections. Each client
// gets an independent session with its own resource namespace. Buffer pools
// may be shared across clients within the same sharing domain for efficiency.
//
// ## Thread Safety
//
// Server configuration (create, start, stop) requires external synchronization.
// Once running, the proactor handles all I/O and callbacks on its own threads.

#ifndef IREE_HAL_REMOTE_SERVER_API_H_
#define IREE_HAL_REMOTE_SERVER_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declaration — full definition in iree/net/transport_factory.h.
typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;

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
  // on release.
  // Required — must not be NULL.
  iree_net_transport_factory_t* transport_factory;

  // Address to bind the server to.
  // Format depends on the transport:
  //   TCP/QUIC: "host:port" (e.g., "0.0.0.0:5000")
  //   SHM: "path" (e.g., "/dev/shm/iree-server")
  // Required.
  iree_string_view_t bind_address;

  // Maximum number of concurrent client connections.
  // Zero uses IREE_HAL_REMOTE_DEFAULT_MAX_CONNECTIONS.
  uint32_t max_connections;

  // Maximum size of a single message on the control channel.
  // Zero uses IREE_HAL_REMOTE_DEFAULT_MAX_CONTROL_MESSAGE_SIZE (from
  // client/api.h; shared between client and server).
  iree_host_size_t max_control_message_size;

  // Maximum size of a single frame on queue endpoints.
  // Zero uses IREE_HAL_REMOTE_DEFAULT_MAX_QUEUE_FRAME_SIZE (from client/api.h;
  // shared between client and server).
  iree_host_size_t max_queue_frame_size;

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
// must ensure options does not outlive the storage.
//
// Recognized parameters:
//   bind=<address>        Bind address (format depends on transport)
//   max_connections=<n>   Maximum concurrent connections
//   rdma=true|false       Enable/disable RDMA for bulk transfers
//   trace=true|false      Enable server operation tracing
IREE_API_EXPORT iree_status_t iree_hal_remote_server_options_parse(
    iree_hal_remote_server_options_t* options, iree_string_pair_list_t params);

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;

// Creates a remote HAL server that exposes the provided device to clients.
//
// The server does not start listening immediately; call
// iree_hal_remote_server_start() to begin accepting connections.
//
// |wrapped_device| is retained by the server and must remain valid until the
// server is destroyed.
//
// |out_server| must be released by the caller (see
// iree_hal_remote_server_release).
IREE_API_EXPORT iree_status_t iree_hal_remote_server_create(
    const iree_hal_remote_server_options_t* options,
    iree_hal_device_t* wrapped_device, iree_allocator_t host_allocator,
    iree_hal_remote_server_t** out_server);

// Retains a reference to the server.
IREE_API_EXPORT void iree_hal_remote_server_retain(
    iree_hal_remote_server_t* server);

// Releases a reference to the server.
IREE_API_EXPORT void iree_hal_remote_server_release(
    iree_hal_remote_server_t* server);

// Starts the server listening for connections.
// Returns OK if the server successfully binds to the configured address.
IREE_API_EXPORT iree_status_t
iree_hal_remote_server_start(iree_hal_remote_server_t* server);

// Stops the server, closing all active connections.
// Blocks until all connections are cleanly closed or the timeout expires.
// Pass iree_infinite_timeout() to wait indefinitely.
IREE_API_EXPORT iree_status_t iree_hal_remote_server_stop(
    iree_hal_remote_server_t* server, iree_timeout_t timeout);

// Runs the server's event loop until stopped.
// This is a convenience for simple server applications. For more control,
// use iree_hal_remote_server_poll() instead.
IREE_API_EXPORT iree_status_t
iree_hal_remote_server_run(iree_hal_remote_server_t* server);

// Polls the server for events, processing any pending work.
// Returns DEADLINE_EXCEEDED if no events occurred within the timeout.
// Pass iree_immediate_timeout() for non-blocking poll, iree_infinite_timeout()
// to block until an event occurs.
IREE_API_EXPORT iree_status_t iree_hal_remote_server_poll(
    iree_hal_remote_server_t* server, iree_timeout_t timeout);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_API_H_
