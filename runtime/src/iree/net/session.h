// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Session: connection lifecycle, bootstrap, and endpoint provisioning.
//
// A session manages a single connection to a remote peer. It handles the
// bootstrap protocol (HELLO/HELLO_ACK topology exchange), registers remote
// axes in a shared frontier_tracker with proxy semaphores, and provides
// endpoint provisioning for application-layer traffic.
//
// ## What the session does
//
//   - Bootstrap: Exchanges topology (axes + current epochs) with the peer.
//     Creates proxy semaphores for remote axes and registers them in the
//     frontier_tracker so local waiters can depend on remote progress.
//   - Control channel lifecycle: Owns an iree_net_control_channel_t for
//     protocol messages (PING/PONG liveness, GOAWAY shutdown, ERROR
//     reporting) and application DATA frames. DATA frames are the primary
//     steady-state traffic path, carrying inline command buffer recordings
//     (potentially many megabytes per submission) to the application handler.
//   - Endpoint provisioning: Proxies connection.open_endpoint() so the
//     application can open queue and bulk endpoints on demand.
//   - Shutdown: Sends GOAWAY, drains endpoints, fails remote axes in the
//     tracker (propagating errors to all local waiters), and releases the
//     connection.
//
// ## What the session does NOT do
//
//   - Interpret application-layer traffic. Queue commands (64KB-512KB),
//     inline command buffer recordings (potentially many megabytes), and
//     bulk transfers are opaque to the session. The HAL layer handles all
//     protocol interpretation on application endpoints and control DATA.
//   - Propagate frontiers in steady state. Frontier updates flow through
//     queue channel ADVANCE frames, parsed by the HAL layer, which calls
//     frontier_tracker_advance() directly.
//   - Own the frontier_tracker. The tracker is application-scoped (spans
//     machine lifetime, survives session reconnects). The session borrows it.
//
// ## Threading
//
// All session operations happen on the proactor thread. Callbacks fire on
// the proactor thread. The session state can be queried from any thread
// (atomic acquire load).
//
// ## Ownership
//
// Sessions use create/retain/release. The session retains the transport
// factory and connection internally. The frontier_tracker is borrowed (must
// outlive the session). Proxy semaphores are owned by the session and
// released on shutdown/destroy.

#ifndef IREE_NET_SESSION_H_
#define IREE_NET_SESSION_H_

#include "iree/async/frontier.h"
#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/channel/control/frame.h"
#include "iree/net/connection.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Forward declarations.
typedef struct iree_net_transport_factory_t iree_net_transport_factory_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_buffer_pool_t iree_async_buffer_pool_t;
typedef struct iree_async_buffer_lease_t iree_async_buffer_lease_t;

//===----------------------------------------------------------------------===//
// Session state
//===----------------------------------------------------------------------===//

// Session lifecycle states.
//
// State transitions are monotonic (no backward transitions):
//   BOOTSTRAPPING -> OPERATIONAL -> DRAINING -> CLOSED
//   Any state -> ERROR (terminal)
typedef enum iree_net_session_state_e {
  // Bootstrap in progress (HELLO/HELLO_ACK exchange).
  // No application operations are allowed.
  IREE_NET_SESSION_STATE_BOOTSTRAPPING = 0,
  // Session is established. Application endpoints can be opened and used.
  IREE_NET_SESSION_STATE_OPERATIONAL = 1,
  // GOAWAY sent or received. No new endpoints can be opened.
  // Existing endpoints continue until drained.
  IREE_NET_SESSION_STATE_DRAINING = 2,
  // All endpoints drained, connection released. Terminal.
  IREE_NET_SESSION_STATE_CLOSED = 3,
  // Unrecoverable error. Terminal.
  IREE_NET_SESSION_STATE_ERROR = 4,
} iree_net_session_state_t;

//===----------------------------------------------------------------------===//
// Topology
//===----------------------------------------------------------------------===//

// Describes a set of axes for topology exchange during bootstrap.
//
// Used in two contexts:
//   - Session options: describes our local axes (sent in HELLO/HELLO_ACK).
//   - on_ready callback: describes the peer's axes (received and registered).
//
// The session copies local topology at creation time, so the caller's storage
// does not need to outlive the session creation call.
typedef struct iree_net_session_topology_t {
  // Array of axis identifiers.
  const iree_async_axis_t* axes;

  // Current epoch for each axis (parallel array with |axes|).
  const uint64_t* current_epochs;

  // Number of entries in |axes| and |current_epochs|.
  uint32_t axis_count;

  // Machine index for this side (0-255, encoded in axis identifiers).
  uint8_t machine_index;

  // Session epoch for ABA prevention in axis encoding.
  uint8_t session_epoch;

  uint8_t reserved[2];
} iree_net_session_topology_t;

//===----------------------------------------------------------------------===//
// Callbacks
//===----------------------------------------------------------------------===//

typedef struct iree_net_session_t iree_net_session_t;

// Called when bootstrap completes and the session becomes OPERATIONAL.
//
// |remote_topology| describes the peer's axes. The session has already created
// proxy semaphores and registered them in the frontier_tracker. The topology
// data is valid only for the duration of this callback.
//
// After this callback returns, the application may:
//   - Open queue/bulk endpoints via iree_net_session_open_endpoint().
//   - Send/receive HAL control messages via on_control_data and
//     iree_net_session_send_control_data().
typedef void (*iree_net_session_on_ready_fn_t)(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology);

// Called when the remote peer initiates graceful shutdown (GOAWAY received).
//
// |reason_code| is 0 for normal shutdown, nonzero for an error category.
// |message| is an optional UTF-8 reason string (may be empty).
//
// The session transitions to DRAINING before this callback fires. The
// application should drain its endpoints and then release the session.
typedef void (*iree_net_session_on_goaway_fn_t)(void* user_data,
                                                iree_net_session_t* session,
                                                uint32_t reason_code,
                                                iree_string_view_t message);

// Called when an unrecoverable error occurs.
//
// |status| ownership is transferred to the callback (must be consumed or
// ignored). The session transitions to ERROR before this callback fires.
// After this callback, the session is unusable and should be released.
typedef void (*iree_net_session_on_error_fn_t)(void* user_data,
                                               iree_net_session_t* session,
                                               iree_status_t status);

// Called when a control channel DATA frame arrives after bootstrap.
//
// |flags| are the control channel's per-frame flags (application-defined).
// |payload| is the DATA payload (control frame header already stripped).
// |lease| references the backing buffer. Retain the lease to keep payload
// data valid beyond this callback.
//
// Return iree_ok_status() to continue receiving. Returning an error
// propagates to the control channel and may cause session failure.
typedef iree_status_t (*iree_net_session_on_control_data_fn_t)(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

// Called when a send_control_data operation completes (payload buffers are
// released).
//
// |operation_user_data| echoes the value from the send_control_data call,
// allowing callers to identify which send completed and release associated
// resources.
// |status| indicates success or failure of the send.
//
// This callback only fires for application sends via
// iree_net_session_send_control_data(). Internal bootstrap sends (HELLO,
// HELLO_ACK) are managed by the session and do not fire this callback.
typedef void (*iree_net_session_on_send_complete_fn_t)(
    void* user_data, uint64_t operation_user_data, iree_status_t status);

// Bundled session callbacks.
//
// |on_ready| is required. |on_control_data| is required (the application
// must handle HAL control messages after bootstrap). |on_goaway| and
// |on_error| are optional but strongly recommended.
//
// All callbacks fire on the proactor thread. The shared |user_data| is
// passed as the first argument to each callback.
typedef struct iree_net_session_callbacks_t {
  iree_net_session_on_ready_fn_t on_ready;
  iree_net_session_on_goaway_fn_t on_goaway;
  iree_net_session_on_error_fn_t on_error;
  iree_net_session_on_control_data_fn_t on_control_data;
  iree_net_session_on_send_complete_fn_t on_send_complete;
  void* user_data;
} iree_net_session_callbacks_t;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

// Default bootstrap timeout (10 seconds).
#define IREE_NET_SESSION_DEFAULT_BOOTSTRAP_TIMEOUT_NS (10ull * 1000000000ull)

// Session configuration.
typedef struct iree_net_session_options_t {
  // Our topology to advertise during handshake.
  iree_net_session_topology_t local_topology;

  // Timeout for bootstrap to complete (HELLO -> HELLO_ACK or REJECT).
  // Zero uses IREE_NET_SESSION_DEFAULT_BOOTSTRAP_TIMEOUT_NS.
  iree_duration_t bootstrap_timeout_ns;

  // Protocol version to offer. Zero uses IREE_NET_BOOTSTRAP_PROTOCOL_VERSION.
  uint32_t protocol_version;

  // Capabilities to advertise. See iree_net_bootstrap_capability_bits_e.
  uint32_t capabilities;

  // Server-assigned session identifier. Sent to the client in HELLO_ACK.
  // Must be nonzero for server sessions (iree_net_session_accept).
  // Ignored for client sessions (assigned by the server via HELLO_ACK).
  //
  // The caller owns the ID namespace and is responsible for uniqueness.
  // Typical pattern: maintain an atomic counter scoped to the listener.
  uint64_t session_id;
} iree_net_session_options_t;

// Returns default session options.
static inline iree_net_session_options_t iree_net_session_options_default(
    void) {
  iree_net_session_options_t options;
  memset(&options, 0, sizeof(options));
  return options;
}

//===----------------------------------------------------------------------===//
// iree_net_session_t
//===----------------------------------------------------------------------===//

// Initiates a client-side session by connecting to a remote server.
//
// The session is returned immediately in |out_session| in BOOTSTRAPPING state.
// The |callbacks.on_ready| fires asynchronously on the proactor thread when
// bootstrap completes. On failure, |callbacks.on_error| fires instead.
//
// The session:
//   1. Calls factory->connect(server_address, proactor, recv_pool, ...)
//   2. Opens a control endpoint on the resulting connection.
//   3. Sends HELLO with local topology.
//   4. Receives HELLO_ACK with remote topology.
//   5. Creates proxy semaphores, registers remote axes in tracker.
//   6. Transitions to OPERATIONAL, fires on_ready.
//
// |factory| is retained by the session.
// |frontier_tracker| is borrowed (must outlive the session).
// |recv_pool| is borrowed (must outlive the session).
// |proactor| is borrowed (must outlive the session).
//
// |callbacks.on_ready| and |callbacks.on_control_data| must be non-NULL.
IREE_API_EXPORT iree_status_t iree_net_session_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t server_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_async_frontier_tracker_t* frontier_tracker,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session);

// Creates a server-side session from an accepted connection.
//
// Called by the server's accept callback when a new connection arrives. The
// session is returned immediately in |out_session| in BOOTSTRAPPING state.
// The |callbacks.on_ready| fires asynchronously on the proactor thread when
// bootstrap completes.
//
// The session:
//   1. Opens a control endpoint on the provided connection.
//   2. Waits for HELLO from the client.
//   3. Validates the HELLO (version, authentication, resource limits).
//   4. Sends HELLO_ACK with local topology.
//   5. Creates proxy semaphores, registers client axes in tracker.
//   6. Transitions to OPERATIONAL, fires on_ready.
//
// |connection| is retained by the session.
// |frontier_tracker| is borrowed (must outlive the session).
// |proactor| is borrowed (must outlive the session).
//
// |callbacks.on_ready| and |callbacks.on_control_data| must be non-NULL.
IREE_API_EXPORT iree_status_t iree_net_session_accept(
    iree_net_connection_t* connection, iree_async_proactor_t* proactor,
    iree_async_frontier_tracker_t* frontier_tracker,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session);

// Retains a reference to the session. NULL-safe.
IREE_API_EXPORT void iree_net_session_retain(iree_net_session_t* session);

// Releases a reference. Destroys the session when the last reference is
// released. NULL-safe.
IREE_API_EXPORT void iree_net_session_release(iree_net_session_t* session);

// Returns the current session state (atomic, any-thread safe).
//
// By the time the caller acts on the result, the state may have changed on
// the proactor thread. Use this for status display and coarse checks, not
// for synchronization.
IREE_API_EXPORT iree_net_session_state_t
iree_net_session_state(const iree_net_session_t* session);

// Returns the server-assigned session identifier.
//
// Returns 0 during BOOTSTRAPPING (not yet assigned). Stable after on_ready.
IREE_API_EXPORT uint64_t iree_net_session_id(const iree_net_session_t* session);

// Opens an application endpoint on this session's connection.
//
// The session proxies to connection.open_endpoint(). The application uses
// these endpoints for queue traffic, bulk transfers, etc. The session does
// not interpret traffic on application endpoints.
//
// |callback| fires exactly once on the proactor thread when the endpoint is
// ready or creation fails. On success, the callback receives a borrowed
// endpoint view valid for the lifetime of the connection.
//
// Requires OPERATIONAL state. Returns FAILED_PRECONDITION otherwise.
IREE_API_EXPORT iree_status_t iree_net_session_open_endpoint(
    iree_net_session_t* session, iree_net_endpoint_ready_callback_t callback,
    void* user_data);

// Sends a DATA frame on the control channel.
//
// |flags| are application-defined per-frame flags. |payload| is a
// scatter-gather list of application data — primarily inline command buffer
// recordings, which can be many megabytes per submission. Payload buffers
// are sent zero-copy and must remain valid until the on_send_complete
// callback fires with the matching |operation_user_data|.
//
// |operation_user_data| is echoed to on_send_complete for correlation.
// Callers typically use this to track which buffers to free on completion.
//
// Requires OPERATIONAL state. Returns FAILED_PRECONDITION in BOOTSTRAPPING
// or terminal states. On non-OK return, on_send_complete is NOT called.
IREE_API_EXPORT iree_status_t iree_net_session_send_control_data(
    iree_net_session_t* session, iree_net_control_frame_flags_t flags,
    iree_async_span_list_t payload, uint64_t operation_user_data);

// Initiates graceful shutdown.
//
// Sends GOAWAY on the control channel with the given reason. Transitions to
// DRAINING. Existing endpoints continue operating until drained; no new
// endpoints can be opened.
//
// When all endpoints are drained and the peer's GOAWAY is received, the
// session fails remote axes in the frontier_tracker, releases proxy
// semaphores, and transitions to CLOSED.
//
// |reason_code| is 0 for normal shutdown, nonzero for an error category.
// |message| is an optional UTF-8 reason string.
//
// Requires OPERATIONAL state. Returns FAILED_PRECONDITION otherwise.
IREE_API_EXPORT iree_status_t
iree_net_session_shutdown(iree_net_session_t* session, uint32_t reason_code,
                          iree_string_view_t message);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_SESSION_H_
