// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REVIEW: "exposes local HAL devices to remote clients" is not true, as this
// only exposes a _single_ HAL device as it is currently designed. the server
// should be created with an iree_hal_device_list_t of devices and expose them
// all. The server then would determine how many endpoints can be shared
// (multi-NIC/NUMA/etc), and in the future when we expose logical/physical
// devices with topology we'll walk that too.
//
// Remote HAL server: exposes local HAL devices to remote clients.
//
// The server accepts connections from remote clients and dispatches their
// operations to a wrapped local HAL device. This enables GPU resources on one
// machine to be used transparently by applications on other machines.
//
// ## Carriers
//
// The server listens on a specific carrier/transport:
//
// REVIEW: flag names like --bind= do not belong here - that's for
// iree-serve-device, and unrelated to any other tooling or user application -
// IREE is a toolkit, and though we have some tools we do not allow tool flags
// to leak into here. discussing the URI scheme is fine, but --bind= is not.
//
//   --bind=tcp://0.0.0.0:5000       (TCP sockets)
//   --bind=quic://0.0.0.0:5000      (QUIC/UDP, future)
//   --bind=ws://0.0.0.0:8080/iree   (WebSocket, future)
//   --bind=shm:///dev/shm/iree      (Shared memory, testing)
//
// ## Usage
//
//   // Create/obtain a local HAL device (e.g., HIP, CUDA, Vulkan).
//   iree_hal_device_t* local_device = ...;
//
//   iree_hal_remote_server_options_t options;
//   iree_hal_remote_server_options_initialize(&options);
//   options.carrier = IREE_HAL_REMOTE_SERVER_CARRIER_TCP;
//   options.bind_address = iree_make_cstring_view("0.0.0.0:5000");
//
//   iree_hal_remote_server_t* server = NULL;
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_create(
//       &options, local_device, host_allocator, &server));
//
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_start(server));
//
// REVIEW: this is all changing.
//   // Option 1: Run event loop (blocks until stopped).
//   IREE_RETURN_IF_ERROR(iree_hal_remote_server_run(server));
//
//   // Option 2: Poll manually for integration with existing event loops.
//   while (running) {
//     iree_hal_remote_server_poll(server, iree_make_duration_ms(100));
//   }
//
// ## Protocol
//
// The server uses the same binary protocol as the client:
//
//   - Control channel: Session lifecycle, capability negotiation, errors.
//   - Queue channel: Frontier-ordered HAL commands with causal consistency.
//   - Bulk channel: Large buffer transfers, optionally via RDMA.
//
// ## Multi-Client Support
//
// The server can accept multiple concurrent client connections. Each client
// gets an independent session with its own resource namespace. Buffer pools
// may be shared across clients within the same sharing domain for efficiency.
//
// ## Thread Safety
//
// Server operations are thread-safe. The event loop (run/poll) should be
// called from a single thread; callbacks fire on that thread.

#ifndef IREE_HAL_REMOTE_SERVER_API_H_
#define IREE_HAL_REMOTE_SERVER_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// REVIEW: same comments as in client api.h

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_carrier_t
//===----------------------------------------------------------------------===//

// Network carrier/transport type for server listening.
// Each carrier uses the same address scheme but different transport mechanisms.
typedef enum iree_hal_remote_server_carrier_e {
  // TCP socket transport (default, always available).
  // Address format: host:port (e.g., "0.0.0.0:5000")
  IREE_HAL_REMOTE_SERVER_CARRIER_TCP = 0,

  // QUIC/UDP transport for low-latency connections.
  // Address format: host:port (e.g., "0.0.0.0:5000")
  IREE_HAL_REMOTE_SERVER_CARRIER_QUIC = 1,

  // WebSocket transport for browser/proxy-friendly connections.
  // Address format: host:port[/path] (e.g., "0.0.0.0:8080/iree")
  IREE_HAL_REMOTE_SERVER_CARRIER_WEBSOCKET = 2,

  // Shared memory transport for local testing and benchmarking.
  // Address format: path (e.g., "/dev/shm/iree-server")
  IREE_HAL_REMOTE_SERVER_CARRIER_SHM = 3,
} iree_hal_remote_server_carrier_t;

//===----------------------------------------------------------------------===//
// iree_hal_remote_server_t
//===----------------------------------------------------------------------===//

// Flags controlling remote server behavior.
enum iree_hal_remote_server_flag_bits_e {
  IREE_HAL_REMOTE_SERVER_FLAG_NONE = 0u,
  // Enables RDMA for bulk transfers when available.
  IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA = 1u << 0,
  // Enables tracing of server operations for debugging.
  IREE_HAL_REMOTE_SERVER_FLAG_TRACE_SERVER_OPS = 1u << 1,
};
typedef uint32_t iree_hal_remote_server_flags_t;

// REVIEW: same comments as in client api.h

// Parameters for configuring an iree_hal_remote_server_t.
// Must be initialized with iree_hal_remote_server_options_initialize prior to
// use.
typedef struct iree_hal_remote_server_options_t {
  // Network carrier/transport type.
  // Determines how clients connect to this server.
  iree_hal_remote_server_carrier_t carrier;

  // Address to bind the server to.
  // Format depends on carrier:
  //   TCP/QUIC: "host:port" (e.g., "0.0.0.0:5000")
  //   WebSocket: "host:port[/path]" (e.g., "0.0.0.0:8080/iree")
  //   SHM: "path" (e.g., "/dev/shm/iree-server")
  // Required.
  iree_string_view_t bind_address;

  // Maximum number of concurrent client connections.
  // Zero uses the default (16).
  uint32_t max_connections;

  // Maximum size of a single message on the control channel.
  // Zero uses the default (64KB).
  iree_host_size_t max_control_message_size;

  // Maximum size of a single frame on the queue channel.
  // Zero uses the default (64KB).
  iree_host_size_t max_queue_frame_size;

  // Flags controlling server behavior.
  iree_hal_remote_server_flags_t flags;

} iree_hal_remote_server_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_remote_server_options_initialize(
    iree_hal_remote_server_options_t* out_options);

// Parses |params| and updates |options|.
// String views may reference strings in the original parameters; the caller
// must ensure options does not outlive the storage.
//
// Recognized parameters:
//   bind=<address>       Bind address (format depends on carrier)
//   max_connections=<n>  Maximum concurrent connections
//   rdma=true|false      Enable/disable RDMA for bulk transfers
//   trace=true|false     Enable server operation tracing
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

// REVIEW: this needs to be async, natively: start, stop, etc. 100% exclusively
// building on our iree/async/proactor.h foundation: there's no blocking here -
// if we want blocking, we build that as an iree_async_* utility (using
// notifications or something). The server is going to be run on one or more
// proactors (we can pick a default for server operations and then bind the
// devices to particular ones close to them in the NUMA hierarchy), or shove
// everything on the same proactor, etc.

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
