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
// Remote devices are specified with a transport-suffixed driver name:
//
//   --device=remote-tcp://server:5000      (TCP sockets)
//   --device=remote-shm:///dev/shm/iree    (shared memory, local testing)
//
// The suffix after "remote-" selects the transport. The registration module
// creates the appropriate iree_net_transport_factory_t based on what transports
// have been compiled in (controlled by IREE_HAVE_NET_*_TRANSPORT defines).
//
// ## Connection Lifecycle
//
// Connection is explicit and asynchronous. The device must be connected before
// any operations can be performed:
//
//   iree_hal_remote_client_device_options_t options;
//   iree_hal_remote_client_device_options_initialize(&options);
//   options.server_address = iree_make_cstring_view("192.168.1.100:5000");
//
//   iree_hal_device_t* device = NULL;
//   IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_create(
//       IREE_SV("remote"), &options, /*create_params=*/NULL,
//       proactor, frontier_tracker, recv_pool,
//       host_allocator, &device));
//
//   // Connection is mandatory — device is unusable until connected.
//   // connect() is async: fires callback on the proactor thread when ready.
//   IREE_RETURN_IF_ERROR(iree_hal_remote_client_device_connect(
//       device, connected_callback));
//
//   // Use device via standard HAL API (after callback fires with OK status).
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
// ## Thread Safety
//
// Devices created by this driver follow standard HAL thread safety rules:
// queue operations, allocations, and semaphore operations may be called from
// any thread concurrently. Device configuration (replace_device_allocator,
// replace_channel_provider) requires external synchronization.

#ifndef IREE_HAL_REMOTE_CLIENT_API_H_
#define IREE_HAL_REMOTE_CLIENT_API_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_async_buffer_pool_t iree_async_buffer_pool_t;

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_device_t
//===----------------------------------------------------------------------===//

// Flags controlling remote client device behavior.
enum iree_hal_remote_client_device_flag_bits_e {
  IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_NONE = 0u,
  // Enables RDMA for bulk transfers when available.
  // Falls back to the transport's default bulk transfer if RDMA is not
  // supported.
  IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_ENABLE_RDMA = 1u << 0,
  // Enables tracing of remote operations for debugging.
  IREE_HAL_REMOTE_CLIENT_DEVICE_FLAG_TRACE_REMOTE_OPS = 1u << 1,
};
typedef uint32_t iree_hal_remote_client_device_flags_t;

// Connection state for a remote device.
typedef enum iree_hal_remote_client_device_state_e {
  // Not connected to server. Initial state after creation.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED = 0,
  // Connection in progress (transport connect + session handshake).
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING = 1,
  // Connected and ready for operations.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED = 2,
  // Unrecoverable error. Device must be destroyed and recreated.
  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR = 3,
} iree_hal_remote_client_device_state_t;

// Callback fired when iree_hal_remote_client_device_connect() completes.
//
// |status| is OK on success. On failure, ownership is transferred to the
// callback (must be consumed or ignored).
typedef void (*iree_hal_remote_client_device_connected_fn_t)(
    void* user_data, iree_status_t status);
typedef struct iree_hal_remote_client_device_connected_callback_t {
  iree_hal_remote_client_device_connected_fn_t fn;
  void* user_data;
} iree_hal_remote_client_device_connected_callback_t;

// Callback fired when the device encounters an unrecoverable error after
// connection (e.g., transport failure, GOAWAY from server).
//
// |status| ownership is transferred to the callback (must be consumed or
// ignored). After this fires, the device is in ERROR state and unusable.
typedef void (*iree_hal_remote_client_device_error_fn_t)(void* user_data,
                                                         iree_status_t status);
typedef struct iree_hal_remote_client_device_error_callback_t {
  iree_hal_remote_client_device_error_fn_t fn;
  void* user_data;
} iree_hal_remote_client_device_error_callback_t;

// Parameters configuring an iree_hal_remote_client_device_t.
// Must be initialized with iree_hal_remote_client_device_options_initialize
// prior to use.
typedef struct iree_hal_remote_client_device_options_t {
  // Transport factory for creating connections.
  // Set by the driver from the factory it was created with. The driver retains
  // the factory and propagates it to all devices it creates. The device retains
  // its own reference during creation.
  // Required — must not be NULL.
  iree_net_transport_factory_t* transport_factory;

  // Server address to connect to.
  // Format depends on the transport:
  //   TCP/QUIC: "host:port" (e.g., "192.168.1.100:5000")
  //   SHM: "path" (e.g., "/dev/shm/iree-server")
  // Required.
  iree_string_view_t server_address;

  // Connection timeout for the async connect operation.
  // Zero uses a reasonable default (IREE_HAL_REMOTE_DEFAULT_CONNECT_TIMEOUT).
  iree_duration_t connect_timeout_ns;

  // Maximum size of a single message on the control channel.
  // Zero uses IREE_HAL_REMOTE_DEFAULT_MAX_CONTROL_MESSAGE_SIZE.
  iree_host_size_t max_control_message_size;

  // Maximum size of a single frame on queue endpoints.
  // Zero uses IREE_HAL_REMOTE_DEFAULT_MAX_QUEUE_FRAME_SIZE.
  iree_host_size_t max_queue_frame_size;

  // Flags controlling device behavior.
  iree_hal_remote_client_device_flags_t flags;

  // Callback fired when the device encounters an unrecoverable error after
  // connection. Optional but strongly recommended — without this, post-connect
  // errors are silently consumed.
  iree_hal_remote_client_device_error_callback_t error_callback;
} iree_hal_remote_client_device_options_t;

// Default connection timeout (30 seconds).
#define IREE_HAL_REMOTE_DEFAULT_CONNECT_TIMEOUT (30ull * 1000000000ull)

// Default maximum control message size (64 KB).
#define IREE_HAL_REMOTE_DEFAULT_MAX_CONTROL_MESSAGE_SIZE (64u * 1024u)

// Default maximum queue frame size (64 KB).
#define IREE_HAL_REMOTE_DEFAULT_MAX_QUEUE_FRAME_SIZE (64u * 1024u)

// Initializes |out_options| to default values.
IREE_API_EXPORT void iree_hal_remote_client_device_options_initialize(
    iree_hal_remote_client_device_options_t* out_options);

// Parses |params| and updates |options|.
// String views may reference strings in the original parameters; the caller
// must ensure options does not outlive the storage.
//
// Recognized parameters:
//   server=<address>     Server address (format depends on transport)
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
// The device does not connect automatically. Call
// iree_hal_remote_client_device_connect() to establish the connection.
// Operations performed before connection will fail with FAILED_PRECONDITION.
//
// |proactor| drives all device I/O (session, callbacks). Borrowed — must
// outlive the device.
//
// |frontier_tracker| tracks axis progress for cross-device causal ordering.
// Borrowed — must outlive the device.
//
// |recv_pool| provides buffers for incoming network data. Borrowed — must
// outlive the device.
//
// |out_device| must be released by the caller (see iree_hal_device_release).
IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_create(
    iree_string_view_t identifier,
    const iree_hal_remote_client_device_options_t* options,
    const iree_hal_device_create_params_t* create_params,
    iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

// Initiates asynchronous connection to the remote server.
//
// The |callback| fires exactly once on the proactor thread:
//   - With OK status when the session becomes OPERATIONAL.
//   - With an error status if connection or bootstrap fails.
//
// Requires DISCONNECTED state. Returns ALREADY_EXISTS if already connected,
// FAILED_PRECONDITION for any other non-DISCONNECTED state.
IREE_API_EXPORT iree_status_t iree_hal_remote_client_device_connect(
    iree_hal_device_t* device,
    iree_hal_remote_client_device_connected_callback_t callback);

// Returns the current connection state of the device.
IREE_API_EXPORT iree_hal_remote_client_device_state_t
iree_hal_remote_client_device_state(iree_hal_device_t* device);

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_driver_t
//===----------------------------------------------------------------------===//

// Parameters for configuring an iree_hal_remote_client_driver_t.
// Must be initialized with iree_hal_remote_client_driver_options_initialize
// prior to use.
typedef struct iree_hal_remote_client_driver_options_t {
  // Transport factory for creating connections. Propagated to device options.
  // The driver retains this factory and releases it on destroy.
  // Required — must not be NULL.
  iree_net_transport_factory_t* transport_factory;

  // Borrowed infrastructure propagated to all devices created by this driver.
  // Must outlive the driver and all devices created from it.
  iree_async_proactor_t* proactor;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_buffer_pool_t* recv_pool;

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
