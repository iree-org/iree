// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/session.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/semaphore.h"
#include "iree/base/internal/atomics.h"
#include "iree/net/bootstrap.h"
#include "iree/net/channel/control/control_channel.h"
#include "iree/net/transport_factory.h"

//===----------------------------------------------------------------------===//
// Internal types
//===----------------------------------------------------------------------===//

// Session role determines bootstrap protocol behavior.
typedef enum iree_net_session_role_e {
  IREE_NET_SESSION_ROLE_CLIENT = 0,
  IREE_NET_SESSION_ROLE_SERVER = 1,
} iree_net_session_role_t;

// Internal bootstrap sub-phases within BOOTSTRAPPING state.
typedef enum iree_net_session_bootstrap_phase_e {
  // Client: factory.connect() in progress.
  IREE_NET_SESSION_BOOTSTRAP_CONNECTING = 0,
  // Both: connection.open_endpoint() for control channel in progress.
  IREE_NET_SESSION_BOOTSTRAP_OPENING_CONTROL = 1,
  // Client: HELLO sent, waiting for HELLO_ACK or REJECT.
  IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT = 2,
  // Server: control channel active, waiting for client HELLO.
  IREE_NET_SESSION_BOOTSTRAP_WAITING_HELLO = 3,
} iree_net_session_bootstrap_phase_t;

//===----------------------------------------------------------------------===//
// iree_net_session_t
//===----------------------------------------------------------------------===//

struct iree_net_session_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  iree_net_session_role_t role;
  iree_net_session_bootstrap_phase_t bootstrap_phase;

  // External state (atomic for cross-thread reads).
  iree_atomic_int32_t state;

  // Application callbacks.
  iree_net_session_callbacks_t callbacks;

  // Configuration (copied at creation).
  uint32_t protocol_version;
  uint32_t capabilities;

  // Server-assigned session identifier.
  uint64_t session_id;

  // Transport factory (retained, client path only — NULL for server).
  iree_net_transport_factory_t* transport_factory;

  // Connection (retained).
  iree_net_connection_t* connection;

  // Control channel (owned by session).
  iree_net_control_channel_t* control_channel;

  // Frontier tracker (borrowed, must outlive session).
  iree_async_frontier_tracker_t* frontier_tracker;

  // Proactor (borrowed, client path only).
  iree_async_proactor_t* proactor;

  // Buffer pool (borrowed, client path only).
  iree_async_buffer_pool_t* recv_pool;

  // Server address (copied, client path only — for factory.connect).
  char* server_address_storage;
  iree_string_view_t server_address;

  // Local topology (copied at creation).
  iree_async_axis_t* local_axes;
  uint64_t* local_epochs;
  uint32_t local_axis_count;
  uint8_t local_machine_index;
  uint8_t local_session_epoch;

  // Remote topology (allocated during bootstrap).
  iree_async_axis_t* remote_axes;
  uint64_t* remote_epochs;
  uint32_t remote_axis_count;
  uint8_t remote_machine_index;
  uint8_t remote_session_epoch;

  // Proxy semaphores for remote axes (parallel array with remote_axes).
  // Owned by the session; released on shutdown/destroy.
  iree_async_semaphore_t** proxy_semaphores;

  // Negotiated capabilities (set during bootstrap).
  iree_net_bootstrap_capabilities_t negotiated_capabilities;
};

// Forward declaration (used by fail() which precedes the definition).
static void iree_net_session_cleanup_remote_axes(iree_net_session_t* session);

//===----------------------------------------------------------------------===//
// State helpers
//===----------------------------------------------------------------------===//

static iree_net_session_state_t iree_net_session_load_state(
    const iree_net_session_t* session) {
  return (iree_net_session_state_t)iree_atomic_load(
      &((iree_net_session_t*)session)->state, iree_memory_order_acquire);
}

static void iree_net_session_set_state(iree_net_session_t* session,
                                       iree_net_session_state_t new_state) {
  iree_atomic_store(&session->state, (int32_t)new_state,
                    iree_memory_order_release);
}

// Transitions to ERROR state and fires on_error callback.
// Takes ownership of |status|.
static void iree_net_session_fail(iree_net_session_t* session,
                                  iree_status_t status) {
  iree_net_session_state_t current = iree_net_session_load_state(session);
  if (current == IREE_NET_SESSION_STATE_ERROR ||
      current == IREE_NET_SESSION_STATE_CLOSED) {
    iree_status_ignore(status);
    return;
  }
  iree_net_session_set_state(session, IREE_NET_SESSION_STATE_ERROR);

  // Fail remote axes in the tracker and release proxy semaphores immediately
  // so that waiters depending on remote axes are woken with errors rather
  // than hanging indefinitely. Idempotent — safe to call even if bootstrap
  // hasn't completed or axes were already cleaned up.
  iree_net_session_cleanup_remote_axes(session);

  if (session->callbacks.on_error) {
    session->callbacks.on_error(session->callbacks.user_data, session, status);
  } else {
    iree_status_ignore(status);
  }
}

//===----------------------------------------------------------------------===//
// Proxy semaphore management
//===----------------------------------------------------------------------===//

// Creates proxy semaphores for remote axes and registers them in the
// frontier_tracker. Called during bootstrap after receiving the peer's
// topology.
static iree_status_t iree_net_session_register_remote_axes(
    iree_net_session_t* session, const iree_net_bootstrap_axis_entry_t* entries,
    uint32_t axis_count) {
  if (axis_count == 0) return iree_ok_status();

  // Allocate parallel arrays for remote topology and proxy semaphores.
  iree_host_size_t axes_size = axis_count * sizeof(iree_async_axis_t);
  iree_host_size_t epochs_size = axis_count * sizeof(uint64_t);
  iree_host_size_t semaphores_size =
      axis_count * sizeof(iree_async_semaphore_t*);
  iree_host_size_t total_size = axes_size + epochs_size + semaphores_size;

  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(session->host_allocator,
                                             total_size, (void**)&storage));
  memset(storage, 0, total_size);

  session->remote_axes = (iree_async_axis_t*)storage;
  session->remote_epochs = (uint64_t*)(storage + axes_size);
  session->proxy_semaphores =
      (iree_async_semaphore_t**)(storage + axes_size + epochs_size);
  session->remote_axis_count = axis_count;

  // Create proxy semaphores and register axes.
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < axis_count && iree_status_is_ok(status); ++i) {
    session->remote_axes[i] = (iree_async_axis_t)entries[i].axis;
    session->remote_epochs[i] = entries[i].current_epoch;

    // Create a software proxy semaphore initialized to the remote's current
    // epoch. When the HAL layer receives ADVANCE frames, it calls
    // frontier_tracker_advance() which signals this semaphore.
    // Proxy semaphores are created without a proactor for now; the HAL remote
    // device will provide one from its proactor pool when it wires through.
    status = iree_async_semaphore_create(
        /*proactor=*/NULL, entries[i].current_epoch,
        IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, session->host_allocator,
        &session->proxy_semaphores[i]);
    if (!iree_status_is_ok(status)) break;

    // Register in the frontier tracker's axis table.
    int32_t index = iree_async_axis_table_add(
        &session->frontier_tracker->axis_table,
        (iree_async_axis_t)entries[i].axis, session->proxy_semaphores[i]);
    if (index < 0) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "frontier tracker axis table full (axis %d/%d)",
                                i, axis_count);
      break;
    }
  }

  if (!iree_status_is_ok(status)) {
    // Full cleanup: fail registered axes in the tracker (so waiters see
    // errors instead of hanging), release all created semaphores, and free the
    // combined allocation. cleanup_remote_axes is idempotent — the later call
    // from destroy() is a no-op since all pointers are NULLed.
    iree_net_session_cleanup_remote_axes(session);
  }

  return status;
}

// Fails all remote axes in the frontier tracker and releases proxy semaphores.
static void iree_net_session_cleanup_remote_axes(iree_net_session_t* session) {
  if (!session->proxy_semaphores) return;

  for (uint32_t i = 0; i < session->remote_axis_count; ++i) {
    if (session->proxy_semaphores[i]) {
      // Fail the axis in the tracker — this propagates errors to all local
      // waiters that depend on this remote axis.
      if (session->frontier_tracker) {
        iree_async_frontier_tracker_fail_axis(
            session->frontier_tracker, session->remote_axes[i],
            iree_make_status(IREE_STATUS_UNAVAILABLE,
                             "remote session disconnected"));
      }
      iree_async_semaphore_release(session->proxy_semaphores[i]);
      session->proxy_semaphores[i] = NULL;
    }
  }

  // Free the combined allocation (remote_axes is the base pointer).
  iree_allocator_free(session->host_allocator, session->remote_axes);
  session->remote_axes = NULL;
  session->remote_epochs = NULL;
  session->proxy_semaphores = NULL;
  session->remote_axis_count = 0;
}

//===----------------------------------------------------------------------===//
// Bootstrap message construction and parsing
//===----------------------------------------------------------------------===//

// Builds and sends a HELLO message on the control channel.
static iree_status_t iree_net_session_send_hello(iree_net_session_t* session) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t payload_size =
      sizeof(iree_net_bootstrap_hello_t) +
      session->local_axis_count * sizeof(iree_net_bootstrap_axis_entry_t);
  uint8_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(session->host_allocator, payload_size,
                                (void**)&buffer));

  iree_net_bootstrap_hello_t* hello = (iree_net_bootstrap_hello_t*)buffer;
  memset(hello, 0, sizeof(*hello));
  hello->header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO;
  hello->protocol_version = session->protocol_version;
  hello->capabilities = session->capabilities;
  hello->machine_index = session->local_machine_index;
  hello->session_epoch = session->local_session_epoch;
  hello->axis_count = (uint16_t)session->local_axis_count;

  iree_net_bootstrap_axis_entry_t* entries =
      (iree_net_bootstrap_axis_entry_t*)(buffer +
                                         sizeof(iree_net_bootstrap_hello_t));
  for (uint32_t i = 0; i < session->local_axis_count; ++i) {
    entries[i].axis = (uint64_t)session->local_axes[i];
    entries[i].current_epoch = session->local_epochs[i];
  }

  iree_status_t status = iree_net_control_channel_send_data(
      session->control_channel, IREE_NET_CONTROL_DATA_FLAG_NONE,
      iree_make_const_byte_span(buffer, payload_size));

  iree_allocator_free(session->host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Builds and sends a HELLO_ACK message on the control channel.
static iree_status_t iree_net_session_send_hello_ack(
    iree_net_session_t* session) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t payload_size =
      sizeof(iree_net_bootstrap_hello_ack_t) +
      session->local_axis_count * sizeof(iree_net_bootstrap_axis_entry_t);
  uint8_t* buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(session->host_allocator, payload_size,
                                (void**)&buffer));

  iree_net_bootstrap_hello_ack_t* ack = (iree_net_bootstrap_hello_ack_t*)buffer;
  memset(ack, 0, sizeof(*ack));
  ack->header.type = IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK;
  ack->session_id = session->session_id;
  ack->negotiated_capabilities = session->negotiated_capabilities;
  ack->machine_index = session->local_machine_index;
  ack->session_epoch = session->local_session_epoch;
  ack->axis_count = (uint16_t)session->local_axis_count;

  iree_net_bootstrap_axis_entry_t* entries =
      (iree_net_bootstrap_axis_entry_t*)(buffer +
                                         sizeof(
                                             iree_net_bootstrap_hello_ack_t));
  for (uint32_t i = 0; i < session->local_axis_count; ++i) {
    entries[i].axis = (uint64_t)session->local_axes[i];
    entries[i].current_epoch = session->local_epochs[i];
  }

  iree_status_t status = iree_net_control_channel_send_data(
      session->control_channel, IREE_NET_CONTROL_DATA_FLAG_NONE,
      iree_make_const_byte_span(buffer, payload_size));

  iree_allocator_free(session->host_allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Completes bootstrap by registering remote axes and transitioning to
// OPERATIONAL. Called by both client (after HELLO_ACK) and server (after
// processing HELLO and sending HELLO_ACK).
static void iree_net_session_complete_bootstrap(
    iree_net_session_t* session, const iree_net_bootstrap_axis_entry_t* entries,
    uint32_t axis_count, uint8_t remote_machine_index,
    uint8_t remote_session_epoch) {
  session->remote_machine_index = remote_machine_index;
  session->remote_session_epoch = remote_session_epoch;

  iree_status_t status =
      iree_net_session_register_remote_axes(session, entries, axis_count);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(session, status);
    return;
  }

  iree_net_session_set_state(session, IREE_NET_SESSION_STATE_OPERATIONAL);

  // Build topology descriptor for the on_ready callback.
  iree_net_session_topology_t remote_topology;
  memset(&remote_topology, 0, sizeof(remote_topology));
  remote_topology.axes = session->remote_axes;
  remote_topology.current_epochs = session->remote_epochs;
  remote_topology.axis_count = session->remote_axis_count;
  remote_topology.machine_index = session->remote_machine_index;
  remote_topology.session_epoch = session->remote_session_epoch;

  session->callbacks.on_ready(session->callbacks.user_data, session,
                              &remote_topology);
}

// Handles a received HELLO (server side).
static iree_status_t iree_net_session_handle_hello(
    iree_net_session_t* session, iree_const_byte_span_t payload) {
  if (session->role != IREE_NET_SESSION_ROLE_SERVER ||
      session->bootstrap_phase != IREE_NET_SESSION_BOOTSTRAP_WAITING_HELLO) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unexpected HELLO (role=%d, phase=%d)",
                            (int)session->role, (int)session->bootstrap_phase);
  }

  if (payload.data_length < sizeof(iree_net_bootstrap_hello_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HELLO too short: %" PRIhsz " bytes (need at least %zu)",
        payload.data_length, sizeof(iree_net_bootstrap_hello_t));
  }

  iree_net_bootstrap_hello_t hello;
  memcpy(&hello, payload.data, sizeof(hello));

  // Validate protocol version.
  if (hello.protocol_version != IREE_NET_BOOTSTRAP_PROTOCOL_VERSION) {
    // Send REJECT and fail.
    // (For now, just fail — REJECT sending is a refinement.)
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported protocol version: %u (expected %u)",
                            hello.protocol_version,
                            IREE_NET_BOOTSTRAP_PROTOCOL_VERSION);
  }

  // Validate axis entries fit in the payload.
  iree_host_size_t expected_size = sizeof(iree_net_bootstrap_hello_t) +
                                   (iree_host_size_t)hello.axis_count *
                                       sizeof(iree_net_bootstrap_axis_entry_t);
  if (payload.data_length < expected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HELLO payload too short for %u axes: %" PRIhsz
                            " bytes (need %" PRIhsz ")",
                            hello.axis_count, payload.data_length,
                            expected_size);
  }

  // Negotiate capabilities.
  session->negotiated_capabilities = hello.capabilities & session->capabilities;

  // Session ID was set from options during session_accept().

  // Send HELLO_ACK.
  iree_status_t status = iree_net_session_send_hello_ack(session);
  if (!iree_status_is_ok(status)) return status;

  // Register remote (client) axes and complete bootstrap.
  const iree_net_bootstrap_axis_entry_t* entries =
      (const iree_net_bootstrap_axis_entry_t*)(payload.data +
                                               sizeof(
                                                   iree_net_bootstrap_hello_t));
  iree_net_session_complete_bootstrap(session, entries, hello.axis_count,
                                      hello.machine_index, hello.session_epoch);

  // complete_bootstrap may have called fail() internally. Propagate the error
  // so the control channel knows to shut down.
  if (iree_net_session_load_state(session) == IREE_NET_SESSION_STATE_ERROR) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "bootstrap failed during axis registration");
  }
  return iree_ok_status();
}

// Handles a received HELLO_ACK (client side).
static iree_status_t iree_net_session_handle_hello_ack(
    iree_net_session_t* session, iree_const_byte_span_t payload) {
  if (session->role != IREE_NET_SESSION_ROLE_CLIENT ||
      session->bootstrap_phase != IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "unexpected HELLO_ACK (role=%d, phase=%d)",
                            (int)session->role, (int)session->bootstrap_phase);
  }

  if (payload.data_length < sizeof(iree_net_bootstrap_hello_ack_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HELLO_ACK too short: %" PRIhsz " bytes (need at least %zu)",
        payload.data_length, sizeof(iree_net_bootstrap_hello_ack_t));
  }

  iree_net_bootstrap_hello_ack_t ack;
  memcpy(&ack, payload.data, sizeof(ack));

  // Validate axis entries fit in the payload.
  iree_host_size_t expected_size = sizeof(iree_net_bootstrap_hello_ack_t) +
                                   (iree_host_size_t)ack.axis_count *
                                       sizeof(iree_net_bootstrap_axis_entry_t);
  if (payload.data_length < expected_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "HELLO_ACK payload too short for %u axes: %" PRIhsz
                            " bytes (need %" PRIhsz ")",
                            ack.axis_count, payload.data_length, expected_size);
  }

  session->session_id = ack.session_id;
  session->negotiated_capabilities = ack.negotiated_capabilities;

  // Register remote (server) axes and complete bootstrap.
  const iree_net_bootstrap_axis_entry_t* entries =
      (const iree_net_bootstrap_axis_entry_t*)(payload.data +
                                               sizeof(
                                                   iree_net_bootstrap_hello_ack_t));
  iree_net_session_complete_bootstrap(session, entries, ack.axis_count,
                                      ack.machine_index, ack.session_epoch);

  // complete_bootstrap may have called fail() internally.
  if (iree_net_session_load_state(session) == IREE_NET_SESSION_STATE_ERROR) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "bootstrap failed during axis registration");
  }
  return iree_ok_status();
}

// Handles a received REJECT (client side).
static iree_status_t iree_net_session_handle_reject(
    iree_net_session_t* session, iree_const_byte_span_t payload) {
  if (payload.data_length < sizeof(iree_net_bootstrap_reject_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "REJECT too short: %" PRIhsz " bytes (need at least %zu)",
        payload.data_length, sizeof(iree_net_bootstrap_reject_t));
  }

  iree_net_bootstrap_reject_t reject;
  memcpy(&reject, payload.data, sizeof(reject));

  iree_string_view_t reason =
      iree_make_string_view((const char*)payload.data + sizeof(reject),
                            payload.data_length - sizeof(reject));

  return iree_make_status((iree_status_code_t)reject.reason_code,
                          "session rejected: %.*s", (int)reason.size,
                          reason.data);
}

//===----------------------------------------------------------------------===//
// Control channel callbacks
//===----------------------------------------------------------------------===//

// Control channel DATA handler. During bootstrap, parses bootstrap messages.
// After bootstrap, forwards to the application's on_control_data handler.
//
// Retains the session for the duration of the callback to protect against
// use-after-free if application callbacks release the session.
static iree_status_t iree_net_session_on_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_retain(session);

  iree_status_t status = iree_ok_status();
  iree_net_session_state_t state = iree_net_session_load_state(session);

  if (state == IREE_NET_SESSION_STATE_OPERATIONAL ||
      state == IREE_NET_SESSION_STATE_DRAINING) {
    // Forward to application handler (on_control_data is required; validated
    // at session creation).
    IREE_ASSERT(session->callbacks.on_control_data);
    status = session->callbacks.on_control_data(session->callbacks.user_data,
                                                flags, payload, lease);
    iree_net_session_release(session);
    return status;
  }

  if (state != IREE_NET_SESSION_STATE_BOOTSTRAPPING) {
    iree_net_session_release(session);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "session in terminal state %d", (int)state);
  }

  // Parse bootstrap message header.
  if (payload.data_length < sizeof(iree_net_bootstrap_header_t)) {
    iree_net_session_release(session);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bootstrap message too short: %" PRIhsz " bytes",
                            payload.data_length);
  }

  iree_net_bootstrap_header_t header;
  memcpy(&header, payload.data, sizeof(header));

  // Validate reserved fields.
  if (header.reserved0[0] != 0 || header.reserved0[1] != 0 ||
      header.reserved0[2] != 0 || header.reserved1 != 0) {
    iree_net_session_release(session);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bootstrap header reserved fields must be 0");
  }

  switch ((iree_net_bootstrap_type_t)header.type) {
    case IREE_NET_BOOTSTRAP_TYPE_HELLO:
      status = iree_net_session_handle_hello(session, payload);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK:
      status = iree_net_session_handle_hello_ack(session, payload);
      break;
    case IREE_NET_BOOTSTRAP_TYPE_REJECT:
      status = iree_net_session_handle_reject(session, payload);
      break;
    default:
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "unknown bootstrap message type: %u", header.type);
      break;
  }

  iree_net_session_release(session);
  return status;
}

// Control channel GOAWAY handler.
// Retains the session to protect against re-entrant release from callbacks.
static void iree_net_session_on_goaway(void* user_data, uint32_t reason_code,
                                       iree_string_view_t message) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_retain(session);

  iree_net_session_set_state(session, IREE_NET_SESSION_STATE_DRAINING);

  // Clean up remote axes (fail them in tracker, release proxy semaphores).
  iree_net_session_cleanup_remote_axes(session);

  if (session->callbacks.on_goaway) {
    session->callbacks.on_goaway(session->callbacks.user_data, session,
                                 reason_code, message);
  }

  iree_net_session_release(session);
}

// Control channel ERROR handler.
// Retains the session to protect against re-entrant release from on_error.
static void iree_net_session_on_control_error(void* user_data,
                                              uint32_t error_code,
                                              iree_string_view_t message) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_retain(session);
  iree_net_session_fail(
      session,
      iree_make_status((iree_status_code_t)error_code, "remote error: %.*s",
                       (int)message.size, message.data));
  iree_net_session_release(session);
}

// Control channel transport error handler.
// Retains the session to protect against re-entrant release from on_error.
static void iree_net_session_on_transport_error(void* user_data,
                                                iree_status_t status) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_retain(session);
  iree_net_session_fail(
      session,
      iree_status_annotate(status, IREE_SV("control channel transport error")));
  iree_net_session_release(session);
}

//===----------------------------------------------------------------------===//
// Connection callbacks (async bootstrap chain)
//===----------------------------------------------------------------------===//

// Called when the control endpoint is ready. Creates the control channel,
// activates it, and begins the bootstrap protocol.
// Retains the session to protect against re-entrant release from on_error.
static void iree_net_session_on_control_endpoint_ready(
    void* user_data, iree_status_t status,
    iree_net_message_endpoint_t endpoint) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_retain(session);

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to open control endpoint")));
    iree_net_session_release(session);
    return;
  }

  // Create control channel with session as the callback target.
  iree_net_control_channel_callbacks_t channel_callbacks = {
      .on_data = iree_net_session_on_data,
      .on_goaway = iree_net_session_on_goaway,
      .on_error = iree_net_session_on_control_error,
      .on_pong = NULL,  // Session doesn't use PONG directly.
      .on_transport_error = iree_net_session_on_transport_error,
      .user_data = session,
  };

  status = iree_net_control_channel_create(
      endpoint, iree_net_control_channel_options_default(), channel_callbacks,
      session->host_allocator, &session->control_channel);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to create control channel")));
    iree_net_session_release(session);
    return;
  }

  // Activate the control channel (start receiving).
  status = iree_net_control_channel_activate(session->control_channel);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to activate control channel")));
    iree_net_session_release(session);
    return;
  }

  if (session->role == IREE_NET_SESSION_ROLE_CLIENT) {
    // Client: send HELLO and wait for HELLO_ACK.
    session->bootstrap_phase = IREE_NET_SESSION_BOOTSTRAP_HELLO_SENT;
    status = iree_net_session_send_hello(session);
    if (!iree_status_is_ok(status)) {
      iree_net_session_fail(
          session,
          iree_status_annotate(status, IREE_SV("failed to send HELLO")));
      iree_net_session_release(session);
      return;
    }
  } else {
    // Server: wait for client's HELLO.
    session->bootstrap_phase = IREE_NET_SESSION_BOOTSTRAP_WAITING_HELLO;
  }

  iree_net_session_release(session);
}

// Called when factory.connect() completes (client path only).
// Retains the session to protect against re-entrant release from on_error.
static void iree_net_session_on_connect(void* user_data, iree_status_t status,
                                        iree_net_connection_t* connection) {
  iree_net_session_t* session = (iree_net_session_t*)user_data;
  iree_net_session_retain(session);

  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session,
        iree_status_annotate(status, IREE_SV("failed to connect to server")));
    iree_net_session_release(session);
    return;
  }

  // Retain the connection.
  session->connection = connection;
  iree_net_connection_retain(connection);

  // Open the control endpoint.
  session->bootstrap_phase = IREE_NET_SESSION_BOOTSTRAP_OPENING_CONTROL;
  status = iree_net_connection_open_endpoint(
      connection, iree_net_session_on_control_endpoint_ready, session);
  if (!iree_status_is_ok(status)) {
    iree_net_session_fail(
        session, iree_status_annotate(
                     status, IREE_SV("failed to open control endpoint")));
  }

  iree_net_session_release(session);
}

//===----------------------------------------------------------------------===//
// Common initialization
//===----------------------------------------------------------------------===//

// Allocates and initializes a session with the given options.
static iree_status_t iree_net_session_create_common(
    iree_net_session_role_t role, const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_session);
  *out_session = NULL;

  // Validate required callbacks.
  if (!callbacks.on_ready) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "on_ready callback is required");
  }
  if (!callbacks.on_control_data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "on_control_data callback is required");
  }

  // Validate axis count fits in the wire format (uint16_t in HELLO/HELLO_ACK).
  uint32_t local_axis_count = options->local_topology.axis_count;
  if (local_axis_count > UINT16_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local axis count %u exceeds wire format maximum %u", local_axis_count,
        (uint32_t)UINT16_MAX);
  }

  // Compute allocation size for session + local topology arrays.
  iree_host_size_t local_axes_size =
      local_axis_count * sizeof(iree_async_axis_t);
  iree_host_size_t local_epochs_size = local_axis_count * sizeof(uint64_t);
  iree_host_size_t total_size =
      sizeof(iree_net_session_t) + local_axes_size + local_epochs_size;

  iree_net_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&session));
  memset(session, 0, total_size);

  iree_atomic_ref_count_init(&session->ref_count);
  session->host_allocator = host_allocator;
  session->role = role;
  session->bootstrap_phase = (role == IREE_NET_SESSION_ROLE_CLIENT)
                                 ? IREE_NET_SESSION_BOOTSTRAP_CONNECTING
                                 : IREE_NET_SESSION_BOOTSTRAP_OPENING_CONTROL;
  iree_atomic_store(&session->state,
                    (int32_t)IREE_NET_SESSION_STATE_BOOTSTRAPPING,
                    iree_memory_order_release);
  session->callbacks = callbacks;

  // Copy configuration.
  session->protocol_version = options->protocol_version
                                  ? options->protocol_version
                                  : IREE_NET_BOOTSTRAP_PROTOCOL_VERSION;
  session->capabilities = options->capabilities;

  // Copy local topology into trailing storage.
  uint8_t* trailing = (uint8_t*)session + sizeof(iree_net_session_t);
  session->local_axes = (iree_async_axis_t*)trailing;
  session->local_epochs = (uint64_t*)(trailing + local_axes_size);
  session->local_axis_count = local_axis_count;
  session->local_machine_index = options->local_topology.machine_index;
  session->local_session_epoch = options->local_topology.session_epoch;

  if (local_axis_count > 0) {
    memcpy(session->local_axes, options->local_topology.axes, local_axes_size);
    memcpy(session->local_epochs, options->local_topology.current_epochs,
           local_epochs_size);
  }

  *out_session = session;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Destroy
//===----------------------------------------------------------------------===//

static void iree_net_session_destroy(iree_net_session_t* session) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Clean up remote axes and proxy semaphores.
  iree_net_session_cleanup_remote_axes(session);

  // Release control channel.
  iree_net_control_channel_release(session->control_channel);

  // Release connection.
  if (session->connection) {
    iree_net_connection_release(session->connection);
  }

  // Release transport factory (client path only).
  if (session->transport_factory) {
    iree_net_transport_factory_release(session->transport_factory);
  }

  // Free server address storage (client path only).
  if (session->server_address_storage) {
    iree_allocator_free(session->host_allocator,
                        session->server_address_storage);
  }

  iree_allocator_t host_allocator = session->host_allocator;
  iree_allocator_free(host_allocator, session);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Public API: creation
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_session_connect(
    iree_net_transport_factory_t* factory, iree_string_view_t server_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_async_frontier_tracker_t* frontier_tracker,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(factory);
  IREE_ASSERT_ARGUMENT(proactor);
  IREE_ASSERT_ARGUMENT(recv_pool);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_net_session_create_common(IREE_NET_SESSION_ROLE_CLIENT, options,
                                         callbacks, host_allocator, &session));

  // Retain factory.
  session->transport_factory = factory;
  iree_net_transport_factory_retain(factory);

  // Store borrowed references.
  session->frontier_tracker = frontier_tracker;
  session->proactor = proactor;
  session->recv_pool = recv_pool;

  // Copy server address (the string_view may not outlive this call).
  if (server_address.size > 0) {
    iree_status_t status =
        iree_allocator_malloc(host_allocator, server_address.size,
                              (void**)&session->server_address_storage);
    if (!iree_status_is_ok(status)) {
      iree_net_session_destroy(session);
      IREE_TRACE_ZONE_END(z0);
      return status;
    }
    memcpy(session->server_address_storage, server_address.data,
           server_address.size);
    session->server_address = iree_make_string_view(
        session->server_address_storage, server_address.size);
  }

  // Begin async connect.
  iree_status_t status = iree_net_transport_factory_connect(
      factory, session->server_address, proactor, recv_pool,
      iree_net_session_on_connect, session);
  if (!iree_status_is_ok(status)) {
    iree_net_session_destroy(session);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_session = session;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_session_accept(
    iree_net_connection_t* connection,
    iree_async_frontier_tracker_t* frontier_tracker,
    const iree_net_session_options_t* options,
    iree_net_session_callbacks_t callbacks, iree_allocator_t host_allocator,
    iree_net_session_t** out_session) {
  IREE_ASSERT_ARGUMENT(connection);
  IREE_ASSERT_ARGUMENT(frontier_tracker);
  if (!options->session_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "server sessions require a nonzero session_id in options");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_session_t* session = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_net_session_create_common(IREE_NET_SESSION_ROLE_SERVER, options,
                                         callbacks, host_allocator, &session));

  // Copy server-assigned session ID from options.
  session->session_id = options->session_id;

  // Retain connection.
  session->connection = connection;
  iree_net_connection_retain(connection);

  // Store borrowed references.
  session->frontier_tracker = frontier_tracker;

  // Open control endpoint to begin bootstrap.
  iree_status_t status = iree_net_connection_open_endpoint(
      connection, iree_net_session_on_control_endpoint_ready, session);
  if (!iree_status_is_ok(status)) {
    iree_net_session_destroy(session);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  *out_session = session;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Public API: lifecycle
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_net_session_retain(iree_net_session_t* session) {
  if (session) iree_atomic_ref_count_inc(&session->ref_count);
}

IREE_API_EXPORT void iree_net_session_release(iree_net_session_t* session) {
  if (session && iree_atomic_ref_count_dec(&session->ref_count) == 1) {
    iree_net_session_destroy(session);
  }
}

IREE_API_EXPORT iree_net_session_state_t
iree_net_session_state(const iree_net_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return iree_net_session_load_state(session);
}

IREE_API_EXPORT uint64_t
iree_net_session_id(const iree_net_session_t* session) {
  IREE_ASSERT_ARGUMENT(session);
  return session->session_id;
}

//===----------------------------------------------------------------------===//
// Public API: operations
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_session_open_endpoint(
    iree_net_session_t* session, iree_net_endpoint_ready_callback_t callback,
    void* user_data) {
  IREE_ASSERT_ARGUMENT(session);
  iree_net_session_state_t state = iree_net_session_load_state(session);
  if (state != IREE_NET_SESSION_STATE_OPERATIONAL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot open endpoint: session state is %d (need OPERATIONAL)",
        (int)state);
  }
  return iree_net_connection_open_endpoint(session->connection, callback,
                                           user_data);
}

IREE_API_EXPORT iree_status_t iree_net_session_send_control_data(
    iree_net_session_t* session, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload) {
  IREE_ASSERT_ARGUMENT(session);
  iree_net_session_state_t state = iree_net_session_load_state(session);
  if (state != IREE_NET_SESSION_STATE_OPERATIONAL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot send control data: session state is %d (need OPERATIONAL)",
        (int)state);
  }
  return iree_net_control_channel_send_data(session->control_channel, flags,
                                            payload);
}

IREE_API_EXPORT iree_status_t
iree_net_session_shutdown(iree_net_session_t* session, uint32_t reason_code,
                          iree_string_view_t message) {
  IREE_ASSERT_ARGUMENT(session);
  iree_net_session_state_t state = iree_net_session_load_state(session);
  if (state != IREE_NET_SESSION_STATE_OPERATIONAL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot shutdown: session state is %d (need OPERATIONAL)", (int)state);
  }

  iree_status_t status = iree_net_control_channel_send_goaway(
      session->control_channel, reason_code, message);
  if (!iree_status_is_ok(status)) return status;

  iree_net_session_set_state(session, IREE_NET_SESSION_STATE_DRAINING);

  // Clean up remote axes proactively — our side is shutting down.
  iree_net_session_cleanup_remote_axes(session);

  return iree_ok_status();
}
