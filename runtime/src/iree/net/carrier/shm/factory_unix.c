// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM factory operations for POSIX platforms.
//
// Implements listener and connect over Unix domain sockets. Each accepted
// connection runs a synchronous handshake (handshake_posix.c) to exchange SHM
// handles via SCM_RIGHTS, then creates an independent SHM carrier pair.

#include "iree/net/carrier/shm/factory_internal.h"

#if !defined(IREE_PLATFORM_WINDOWS)

#include <string.h>

#include "iree/async/address.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/socket.h"
#include "iree/net/carrier/shm/handshake.h"

//===----------------------------------------------------------------------===//
// Cross-process listener (Unix domain socket)
//===----------------------------------------------------------------------===//

// State machine for the Unix socket accept loop.
typedef enum iree_net_shm_unix_listener_state_e {
  IREE_NET_SHM_UNIX_LISTENER_STATE_LISTENING = 0,
  IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING,
} iree_net_shm_unix_listener_state_t;

// Listener backed by a Unix domain stream socket. Each accepted connection
// runs a synchronous handshake to exchange SHM handles and notification
// primitives, then creates an independent SHM carrier pair. The accept loop
// supports both multishot (io_uring) and single-shot with re-arm.
typedef struct iree_net_shm_unix_listener_t {
  iree_net_listener_t base;
  iree_net_shm_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  iree_async_socket_t* listen_socket;
  struct {
    iree_net_listener_accept_callback_t fn;
    void* user_data;
  } accept;
  iree_async_socket_accept_operation_t accept_operation;
  iree_net_shm_unix_listener_state_t state;
  iree_net_listener_stopped_callback_t stopped_callback;
  iree_allocator_t host_allocator;
  // Full address string (e.g., "unix:/tmp/iree.sock") for
  // query_bound_address. Null-terminated.
  iree_host_size_t address_length;
  char address[];
} iree_net_shm_unix_listener_t;

static const iree_net_listener_vtable_t iree_net_shm_unix_listener_vtable;

static void iree_net_shm_unix_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

// Handles a successfully accepted socket: duplicates the primitive, runs the
// server-side handshake, creates a carrier, wraps in a connection, and delivers
// to the consumer. On failure at any step, reports the error to the consumer.
static void iree_net_shm_unix_listener_handle_accepted(
    iree_net_shm_unix_listener_t* listener, iree_async_socket_t* accepted) {
  // Dup the socket's primitive for the handshake (the handshake closes its
  // primitive on return; releasing the socket object separately closes the
  // original fd -- so both owners are handled cleanly).
  iree_async_primitive_t handshake_primitive;
  iree_status_t status =
      iree_async_primitive_dup(accepted->primitive, &handshake_primitive);
  iree_async_socket_release(accepted);

  // Get or create shared_wake for this proactor.
  iree_net_shm_shared_wake_t* shared_wake = NULL;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&listener->factory->mutex);
    status = iree_net_shm_factory_get_or_create_shared_wake(
        listener->factory, listener->proactor, &shared_wake);
    iree_slim_mutex_unlock(&listener->factory->mutex);
    if (!iree_status_is_ok(status)) {
      iree_async_primitive_close(&handshake_primitive);
    }
  }

  // Run the server handshake. Closes the socket primitive on return (both
  // success and error paths).
  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_server(
        handshake_primitive, shared_wake, listener->factory->options,
        listener->proactor, listener->host_allocator, &handshake_result);
  }

  // Create carrier from handshake result.
  iree_net_carrier_callback_t no_callback = {0};
  iree_net_carrier_t* carrier = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_carrier_create(&handshake_result.carrier_params,
                                         no_callback, listener->host_allocator,
                                         &carrier);
    if (!iree_status_is_ok(status)) {
      iree_net_shm_xproc_context_release(handshake_result.context);
    }
  }

  // Wrap in connection and deliver to the consumer.
  iree_net_connection_t* connection = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_connection_create(
        listener->proactor, carrier, listener->recv_pool,
        listener->host_allocator, &connection);
    if (!iree_status_is_ok(status)) {
      iree_net_carrier_release(carrier);
    }
  }

  if (iree_status_is_ok(status)) {
    listener->accept.fn(listener->accept.user_data, iree_ok_status(),
                        connection);
  } else {
    listener->accept.fn(listener->accept.user_data, status, NULL);
  }
}

// Re-arms the accept operation for single-shot proactors. Multishot proactors
// deliver IREE_ASYNC_COMPLETION_FLAG_MORE and do not need re-arming.
static void iree_net_shm_unix_listener_rearm(
    iree_net_shm_unix_listener_t* listener) {
  memset(&listener->accept_operation, 0, sizeof(listener->accept_operation));
  iree_async_operation_initialize(
      &listener->accept_operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT,
      IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_net_shm_unix_listener_accept_complete, listener);
  listener->accept_operation.listen_socket = listener->listen_socket;
  iree_status_t submit_status = iree_async_proactor_submit_one(
      listener->proactor, &listener->accept_operation.base);
  if (!iree_status_is_ok(submit_status)) {
    listener->accept.fn(listener->accept.user_data, submit_status, NULL);
  }
}

// Accept completion callback for the Unix domain socket listener.
// Runs the server-side handshake on the accepted connection, creates a carrier
// from the result, wraps it in a connection, and delivers to the consumer.
//
// The handshake is synchronous but completes in microseconds over a local Unix
// domain socket. The 5s timeout in the handshake is a safety valve for
// pathological peers. During the handshake, the proactor thread is blocked --
// acceptable for local IPC bootstrapping.
static void iree_net_shm_unix_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)user_data;
  iree_async_socket_accept_operation_t* accept_op =
      (iree_async_socket_accept_operation_t*)operation;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Handle stopping: clean up and fire stopped callback on final CQE.
  if (listener->state == IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING) {
    if (accept_op->accepted_socket) {
      iree_async_socket_release(accept_op->accepted_socket);
      accept_op->accepted_socket = NULL;
    }
    iree_status_ignore(status);
    if (!(flags & IREE_ASYNC_COMPLETION_FLAG_MORE)) {
      listener->stopped_callback.fn(listener->stopped_callback.user_data);
    }
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  if (iree_status_is_ok(status)) {
    iree_net_shm_unix_listener_handle_accepted(listener,
                                               accept_op->accepted_socket);
    accept_op->accepted_socket = NULL;
  } else {
    if (accept_op->accepted_socket) {
      iree_async_socket_release(accept_op->accepted_socket);
      accept_op->accepted_socket = NULL;
    }
    listener->accept.fn(listener->accept.user_data, status, NULL);
  }

  if (!(flags & IREE_ASYNC_COMPLETION_FLAG_MORE) &&
      listener->state == IREE_NET_SHM_UNIX_LISTENER_STATE_LISTENING) {
    iree_net_shm_unix_listener_rearm(listener);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_net_shm_unix_listener_free(
    iree_net_listener_t* base_listener) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (listener->listen_socket) {
    iree_async_socket_release(listener->listen_socket);
  }
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_allocator_free(host_allocator, listener);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_net_shm_unix_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  listener->state = IREE_NET_SHM_UNIX_LISTENER_STATE_STOPPING;
  listener->stopped_callback = callback;
  iree_status_t status = iree_async_proactor_cancel(
      listener->proactor, &listener->accept_operation.base);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_unix_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_shm_unix_listener_t* listener =
      (iree_net_shm_unix_listener_t*)base_listener;
  if (buffer_capacity < listener->address_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer too small for bound address");
  }
  memcpy(buffer, listener->address, listener->address_length);
  *out_address = iree_make_string_view(buffer, listener->address_length);
  return iree_ok_status();
}

static const iree_net_listener_vtable_t iree_net_shm_unix_listener_vtable = {
    .free = iree_net_shm_unix_listener_free,
    .stop = iree_net_shm_unix_listener_stop,
    .query_bound_address = iree_net_shm_unix_listener_query_bound_address,
};

// Creates a cross-process listener bound to a Unix domain socket path.
// Parses the address, creates the socket, binds, listens, and submits the
// initial accept operation.
iree_status_t iree_net_shm_factory_create_listener_unix(
    iree_net_shm_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_listener = NULL;

  // Parse the Unix domain socket address.
  iree_async_address_t address;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_address_from_string(bind_address, &address));

  // Create Unix domain stream socket.
  iree_async_socket_t* listen_socket = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                               IREE_ASYNC_SOCKET_OPTION_NONE, &listen_socket));

  // Bind and listen.
  iree_status_t status = iree_async_socket_bind(listen_socket, &address);
  if (iree_status_is_ok(status)) {
    status = iree_async_socket_listen(listen_socket, /*backlog=*/128);
  }

  // Allocate listener with space for the address string (null-terminated).
  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        iree_sizeof_struct(iree_net_shm_unix_listener_t), &total_size,
        IREE_STRUCT_FIELD_FAM(bind_address.size + 1, char));
  }
  iree_net_shm_unix_listener_t* listener = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&listener);
  }
  if (iree_status_is_ok(status)) {
    memset(listener, 0, total_size);
    listener->base.vtable = &iree_net_shm_unix_listener_vtable;
    listener->factory = factory;
    listener->proactor = proactor;
    listener->recv_pool = recv_pool;
    listener->listen_socket = listen_socket;
    listener->accept.fn = accept_callback;
    listener->accept.user_data = user_data;
    listener->host_allocator = host_allocator;
    listener->address_length = bind_address.size;
    memcpy(listener->address, bind_address.data, bind_address.size);
    listener->address[bind_address.size] = '\0';
  }

  // Submit the first accept operation. Use multishot if the proactor supports
  // it for reduced syscall overhead on io_uring.
  if (iree_status_is_ok(status)) {
    iree_async_proactor_capabilities_t capabilities =
        iree_async_proactor_query_capabilities(proactor);
    iree_async_operation_flags_t accept_flags = IREE_ASYNC_OPERATION_FLAG_NONE;
    if (capabilities & IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT) {
      accept_flags |= IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
    }
    iree_async_operation_initialize(
        &listener->accept_operation.base,
        IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT, accept_flags,
        iree_net_shm_unix_listener_accept_complete, listener);
    listener->accept_operation.listen_socket = listen_socket;
    status = iree_async_proactor_submit_one(proactor,
                                            &listener->accept_operation.base);
  }

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else {
    if (listener) iree_allocator_free(host_allocator, listener);
    iree_async_socket_release(listen_socket);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Cross-process connect (Unix domain socket)
//===----------------------------------------------------------------------===//

// Heap-allocated state for async Unix domain socket connect. Freed in the
// connect completion callback after the handshake and carrier creation.
typedef struct iree_net_shm_unix_connect_state_t {
  iree_async_socket_connect_operation_t connect_operation;
  struct {
    iree_net_transport_connect_callback_t fn;
    void* user_data;
  } callback;
  iree_net_shm_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  iree_async_socket_t* socket;
  iree_allocator_t host_allocator;
} iree_net_shm_unix_connect_state_t;

// Handles the connected socket: duplicates the primitive, runs the client-side
// handshake, creates a carrier, wraps in a connection, and delivers to the
// consumer. On failure at any step, reports the error to the consumer.
static void iree_net_shm_unix_connect_handle_connected(
    iree_net_shm_unix_connect_state_t* state) {
  // Dup the connected socket's primitive for the handshake (the handshake
  // closes its primitive on return; releasing the socket object separately
  // closes the original fd).
  iree_async_primitive_t handshake_primitive;
  iree_status_t status =
      iree_async_primitive_dup(state->socket->primitive, &handshake_primitive);
  iree_async_socket_release(state->socket);
  state->socket = NULL;

  // Get or create shared_wake for this proactor.
  iree_net_shm_shared_wake_t* shared_wake = NULL;
  if (iree_status_is_ok(status)) {
    iree_slim_mutex_lock(&state->factory->mutex);
    status = iree_net_shm_factory_get_or_create_shared_wake(
        state->factory, state->proactor, &shared_wake);
    iree_slim_mutex_unlock(&state->factory->mutex);
    if (!iree_status_is_ok(status)) {
      iree_async_primitive_close(&handshake_primitive);
    }
  }

  // Run the client handshake. Synchronous but completes in microseconds over
  // a local Unix domain socket. Closes the socket primitive on return.
  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_handshake_client(
        handshake_primitive, shared_wake, state->proactor,
        state->host_allocator, &handshake_result);
  }

  // Create carrier from handshake result.
  iree_net_carrier_callback_t no_callback = {0};
  iree_net_carrier_t* carrier = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_carrier_create(&handshake_result.carrier_params,
                                         no_callback, state->host_allocator,
                                         &carrier);
    if (!iree_status_is_ok(status)) {
      iree_net_shm_xproc_context_release(handshake_result.context);
    }
  }

  // Wrap in connection and deliver.
  iree_net_connection_t* connection = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_connection_create(state->proactor, carrier,
                                            state->recv_pool,
                                            state->host_allocator, &connection);
    if (!iree_status_is_ok(status)) {
      iree_net_carrier_release(carrier);
    }
  }

  if (iree_status_is_ok(status)) {
    state->callback.fn(state->callback.user_data, iree_ok_status(), connection);
  } else {
    state->callback.fn(state->callback.user_data, status, NULL);
  }
}

// Connect completion callback for cross-process SHM connect. On successful
// connect, runs the client-side handshake, creates a carrier, wraps in a
// connection, and delivers to the consumer via the callback.
static void iree_net_shm_unix_connect_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_unix_connect_state_t* state =
      (iree_net_shm_unix_connect_state_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_status_is_ok(status)) {
    iree_net_shm_unix_connect_handle_connected(state);
  } else {
    iree_async_socket_release(state->socket);
    state->callback.fn(state->callback.user_data, status, NULL);
  }

  iree_allocator_free(state->host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

// Initiates a cross-process connect to a Unix domain socket path. Parses the
// address, creates a socket, and submits an async connect operation. On
// completion, the handshake runs and the carrier is created.
iree_status_t iree_net_shm_factory_connect_unix(
    iree_net_shm_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Parse the Unix domain socket address.
  iree_async_address_t remote_address;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_address_from_string(address, &remote_address));

  // Create Unix domain stream socket.
  iree_async_socket_t* socket = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_socket_create(proactor, IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM,
                                   IREE_ASYNC_SOCKET_OPTION_NONE, &socket));

  // Allocate connect state.
  iree_net_shm_unix_connect_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(factory->host_allocator,
                                               sizeof(*state), (void**)&state);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(socket);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(state, 0, sizeof(*state));
  state->callback.fn = callback;
  state->callback.user_data = user_data;
  state->factory = factory;
  state->proactor = proactor;
  state->recv_pool = recv_pool;
  state->socket = socket;
  state->host_allocator = factory->host_allocator;

  // Submit async connect.
  iree_async_operation_initialize(&state->connect_operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_net_shm_unix_connect_complete, state);
  state->connect_operation.socket = socket;
  state->connect_operation.address = remote_address;
  status =
      iree_async_proactor_submit_one(proactor, &state->connect_operation.base);
  if (!iree_status_is_ok(status)) {
    iree_async_socket_release(socket);
    iree_allocator_free(factory->host_allocator, state);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // !IREE_PLATFORM_WINDOWS
