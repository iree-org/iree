// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Remote HAL client: presents remote GPUs as local HAL devices.
//
// This driver connects to a remote HAL server and presents the server's devices
// as local HAL devices. Users interact with the standard HAL API; the driver
// transparently handles network communication, buffer registration, and
// synchronization.
//
// ## Device URIs
//
// Remote devices are specified with a carrier-prefixed URI:
//
// REVIEW: remove the --device= - we are a toolkit, and flag names are a
// tool/framework/etc concern (if they even have flags). discussing the URI
// scheme is fine, but flag names are not.
//
//   --device=remote-tcp://server:5000      (TCP sockets)
//   --device=remote-quic://server:5000     (QUIC/UDP, future)
//   --device=remote-ws://server:8080/iree  (WebSocket, future)
//   --device=remote-shm:///dev/shm/iree    (Shared memory, testing)
//
// The carrier determines the transport mechanism while the address format
// remains consistent (host:port for network, path for local).
//
// ## Usage
//
//   iree_hal_remote_client_device_options_t options;
//   iree_hal_remote_client_device_options_initialize(&options);
//   options.carrier = IREE_HAL_REMOTE_CLIENT_CARRIER_TCP;
//   options.server_address = iree_make_cstring_view("192.168.1.100:5000");
//
//   iree_hal_device_t* device = NULL;
//   IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_create(
//       IREE_SV("remote-tcp"), &options, host_allocator, &device));
//
//   // Use device via standard HAL API - buffers, command buffers, execution
//   // all work transparently over the network.
//
// ## Connection Lifecycle
//
// REVIEW: never lazy stuff in IREE, ever - we never allow program execution to
// continue until we've established a full connection. if that means making our
// device creation async then that's fine and natural - this entire system
// should be attached to an iree_async_proactor_t which is a natural place to
// deliver callbacks.
//
// The device does not connect immediately on creation. Connection is
// established lazily on first use, or can be triggered explicitly via
// iree_hal_remote_client_device_connect(). This allows configuration and
// error handling to be separated.
//
// ## Protocol
//
// Communication uses a custom binary protocol optimized for low latency and
// high throughput. Several channel types handle different traffic patterns:
//
//   - Control channel: Session lifecycle, capability negotiation, errors.
//   - Queue channel: Frontier-ordered HAL commands with causal consistency.
//   - Bulk channel: Large buffer transfers, optionally via RDMA.
//
// All operations are asynchronous. Frontiers (vector clocks) track causal
// dependencies across the distributed system without centralized coordination.
//
// ## Thread Safety
//
// REVIEW: this is not actually how IREE thread safety works - multiple threads
// may submit work to the same HAL device concurrently - some device things
// (like swapping out an allocator shim) may not be ok, but all queue
// operations/allocations/etc are all thread-safe.
//
// Devices created by this driver follow standard HAL thread safety rules:
// operations may be called from any thread, but concurrent calls on the same
// device require external synchronization.

#ifndef IREE_HAL_REMOTE_CLIENT_API_H_
#define IREE_HAL_REMOTE_CLIENT_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_carrier_t
//===----------------------------------------------------------------------===//

// REVIEW: I'm not sure I like this being an enum - or being out of sync with
// the server. I'd rather us pass in a function pointer to a factory function or
// something. by having this enum we're leaking a set of (in this case,
// non-existent) features that may not even be compiled in the build, and to
// implement the enum switch we have to link in all the code even if it's ever
// used.

// Network carrier/transport type for remote connections.
// Each carrier uses the same address scheme but different transport mechanisms.
typedef enum iree_hal_remote_client_carrier_e {
  // TCP socket transport (default, always available).
  // Address format: host:port (e.g., "192.168.1.100:5000")
  IREE_HAL_REMOTE_CLIENT_CARRIER_TCP = 0,

  // QUIC/UDP transport for low-latency connections.
  // Address format: host:port (e.g., "192.168.1.100:5000")
  IREE_HAL_REMOTE_CLIENT_CARRIER_QUIC = 1,

  // WebSocket transport for browser/proxy-friendly connections.
  // Address format: host:port[/path] (e.g., "server:8080/iree")
  IREE_HAL_REMOTE_CLIENT_CARRIER_WEBSOCKET = 2,

  // Shared memory transport for local testing and benchmarking.
  // Address format: path (e.g., "/dev/shm/iree-server")
  IREE_HAL_REMOTE_CLIENT_CARRIER_SHM = 3,
} iree_hal_remote_client_carrier_t;

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_t
//===----------------------------------------------------------------------===//

// Flags controlling remote client device behavior.
enum iree_hal_remote_client_device_flag_bits_e {
  IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_NONE = 0u,
  // Enables RDMA for bulk transfers when available.
  // Falls back to the carrier's default bulk transfer if RDMA is not supported.
  IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA = 1u << 0,
  // Enables tracing of remote operations for debugging.
  IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_TRACE_REMOTE_OPS = 1u << 1,
};
typedef uint32_t iree_hal_remote_client_device_flags_t;

// Connection state for a remote device.
typedef enum iree_hal_remote_client_device_state_e {
  // Not connected to server. Initial state after creation.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED = 0,
  // Connection in progress. TCP handshake and capability negotiation.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING = 1,
  // Connected and ready for operations.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED = 2,
  // Unrecoverable error. Device must be destroyed and recreated.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR = 3,
} iree_hal_remote_client_device_state_t;

// Parameters configuring an iree_hal_remote_client_device_t.
// Must be initialized with iree_hal_remote_client_device_options_initialize
// prior to use.
typedef struct iree_hal_remote_client_device_options_t {
  // REVIEW: we should have an error callback (iree_hal_remote_error_callback_t
  // with a fn/user_data) in the device options that is set by the caller to get
  // state changes on disconnect/global failures/etc.

  // Network carrier/transport type.
  // Determines how the connection is established and data is transferred.
  iree_hal_remote_client_carrier_t carrier;

  // Server address to connect to.
  // Format depends on carrier:
  //   TCP/QUIC: "host:port" (e.g., "192.168.1.100:5000")
  //   WebSocket: "host:port[/path]" (e.g., "server:8080/iree")
  //   SHM: "path" (e.g., "/dev/shm/iree-server")
  // Required.
  iree_string_view_t server_address;

  // REVIEW: this should be an iree_timeout_t on our async connect - it's only
  // used once, and we don't lazy connect.

  // Connection timeout. Zero uses a reasonable default (30 seconds).
  iree_duration_t connect_timeout_ns;

  // REVIEW: never hardcode defaults in structs - you can reference defines that
  // are the default, but never exact values.
  // IREE_HAL_REMOTE_DEFAULT_MAX_CONTROL_MESSAGE_SIZE, etc.

  // Maximum size of a single message on the control channel.
  // Zero uses the default (64KB).
  iree_host_size_t max_control_message_size;

  // Maximum size of a single frame on the queue channel.
  // Zero uses the default (64KB).
  iree_host_size_t max_queue_frame_size;

  // Flags controlling device behavior.
  iree_hal_remote_client_device_flags_t flags;
} iree_hal_remote_client_device_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_remote_client_device_options_initialize(
    iree_hal_remote_client_device_options_t* out_options);

// Parses |params| and updates |options|.
// String views may reference strings in the original parameters; the caller
// must ensure options does not outlive the storage.
//
// Recognized parameters:
//   server=<address>     Server address (format depends on carrier)
//   connect_timeout=<ms> Connection timeout in milliseconds
//   rdma=true|false      Enable/disable RDMA for bulk transfers
//   trace=true|false     Enable remote operation tracing
IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_options_parse(
    iree_hal_remote_client_device_options_t* options,
    iree_string_pair_list_t params);

// Creates a remote HAL device that connects to a server at the configured
// address.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by `IREE::HAL::TargetDevice`.
//
// The device does not connect immediately; connection is established lazily
// on first use or can be triggered explicitly via
// iree_hal_remote_client_device_connect().
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

// REVIEW: this is not optional: connect *must* be called and any call before
// then has undefined behavior. connect must also be async, take an
// iree_timeout_t, and the callback should receive the status. this allows
// multiple devices to be scheduled for simultaneous connection and the client
// then gathers them up and handles management if e.g. one fails during
// connection (tears down them all).

// Explicitly initiates connection to the remote server.
//
// This is optional; the device will connect automatically on first operation.
// Use this to fail fast during initialization if the server is unavailable.
//
// Returns OK if already connected or connection succeeds within the configured
// timeout. Returns UNAVAILABLE if connection fails.
IREE_API_EXPORT iree_status_t
iree_hal_remote_client_device_connect(iree_hal_device_t* device);

// Returns the current connection state of the device.
IREE_API_EXPORT iree_hal_remote_client_device_state_t
iree_hal_remote_client_device_state(iree_hal_device_t* device);

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_driver_t
//===----------------------------------------------------------------------===//

// REVIEW: let's remove api.h and just put these device/driver in their
// respective file - same in server - keep the code together, clients/servers
// can include an extra file or two :)

// Parameters for configuring an iree_hal_remote_client_driver_t.
// Must be initialized with iree_hal_remote_client_driver_options_initialize
// prior to use.
typedef struct iree_hal_remote_client_driver_options_t {
  // Network carrier/transport type for this driver instance.
  // Set based on the driver name (remote-tcp, remote-quic, etc.).
  iree_hal_remote_client_carrier_t carrier;

  // Default device options when none are provided during device creation.
  iree_hal_remote_client_device_options_t default_device_options;
} iree_hal_remote_client_driver_options_t;

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_remote_client_driver_options_initialize(
    iree_hal_remote_client_driver_options_t* out_options);

// Parses |params| and updates |options|.
// String views may reference strings in the original parameters; the caller
// must ensure options does not outlive the storage.
IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_options_parse(
    iree_hal_remote_client_driver_options_t* options,
    iree_string_pair_list_t params);

// Creates a remote HAL driver from which devices can be created.
//
// The provided |identifier| will be used by programs to distinguish the device
// type from other HAL implementations. If compiling programs with the IREE
// compiler this must match the value used by IREE::HAL::TargetDevice.
//
// |out_driver| must be released by the caller (see iree_hal_driver_release).
IREE_API_EXPORT iree_status_t iree_hal_remote_client_driver_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_API_H_
