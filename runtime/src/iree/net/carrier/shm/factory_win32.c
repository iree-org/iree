// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM factory operations for Windows.
//
// Implements listener and connect over Windows named pipes. Each accepted
// connection runs a synchronous handshake (handshake_win32.c) to exchange SHM
// handles via DuplicateHandle, then creates an independent SHM carrier pair.
//
// The named pipe is a temporary bootstrap channel: ConnectNamedPipe accepts a
// client, the handshake exchanges SHM region and notification handles, then the
// pipe is closed. The carrier uses the SHM ring buffers directly -- the pipe
// carries only the ~100-byte handshake messages.
//
// Async accept uses EVENT_WAIT: the pipe is NOT associated with IOCP. An
// iree_async_event_t signals when ConnectNamedPipe completes (the event's
// Win32 HANDLE is set as the hEvent in the OVERLAPPED structure). When a client
// connects, the kernel signals the event, and EVENT_WAIT delivers the
// completion to the proactor thread.

#include "iree/net/carrier/shm/factory_internal.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <string.h>
#include <windows.h>

#include "iree/async/event.h"
#include "iree/async/operations/scheduling.h"
#include "iree/net/carrier/shm/handshake.h"

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Prefix for Windows named pipe paths.
static const WCHAR iree_net_shm_pipe_prefix[] = L"\\\\.\\pipe\\";
#define IREE_NET_SHM_PIPE_PREFIX_LENGTH 9  // wcslen(L"\\\\.\\pipe\\")

// Maximum total pipe path length in wide characters (including null
// terminator). Windows named pipe paths must be <= MAX_PATH.
#define IREE_NET_SHM_MAX_PIPE_PATH_LENGTH (MAX_PATH + 1)

// Strips the "pipe:" prefix from an address and returns the pipe name portion.
// The caller already verified the prefix via starts_with in the factory
// dispatch.
static iree_string_view_t iree_net_shm_win32_strip_pipe_prefix(
    iree_string_view_t address) {
  return iree_make_string_view(address.data + 5, address.size - 5);
}

// Builds the wide-character pipe path (\\.\pipe\<name>) from a UTF-8 name.
// |out_path| must have room for IREE_NET_SHM_MAX_PIPE_PATH_LENGTH WCHARs.
// Returns the path length in WCHARs (excluding null terminator) via
// |out_path_length|.
static iree_status_t iree_net_shm_win32_build_pipe_path(iree_string_view_t name,
                                                        WCHAR* out_path,
                                                        int* out_path_length) {
  if (name.size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "pipe name is empty");
  }

  // Convert the name from UTF-8 to wide characters. First pass: compute the
  // required length.
  int wide_name_length = MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, name.data, (int)name.size, NULL, 0);
  if (wide_name_length <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipe name contains invalid UTF-8");
  }

  int total_length = IREE_NET_SHM_PIPE_PREFIX_LENGTH + wide_name_length;
  if (total_length >= IREE_NET_SHM_MAX_PIPE_PATH_LENGTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipe path too long (%d wide chars, max %d)",
                            total_length,
                            IREE_NET_SHM_MAX_PIPE_PATH_LENGTH - 1);
  }

  // Copy prefix and convert name into the output buffer.
  memcpy(out_path, iree_net_shm_pipe_prefix,
         IREE_NET_SHM_PIPE_PREFIX_LENGTH * sizeof(WCHAR));
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name.data, (int)name.size,
                      out_path + IREE_NET_SHM_PIPE_PREFIX_LENGTH,
                      wide_name_length);
  out_path[total_length] = L'\0';
  *out_path_length = total_length;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Cross-process listener (Windows named pipe)
//===----------------------------------------------------------------------===//

typedef enum iree_net_shm_win32_listener_state_e {
  IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING = 0,
  IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPING,
} iree_net_shm_win32_listener_state_t;

// Listener backed by a Windows named pipe. Each accepted connection runs a
// synchronous handshake to exchange SHM handles and notification primitives,
// then creates an independent SHM carrier pair.
//
// Accept loop: CreateNamedPipeW -> ConnectNamedPipe (overlapped, event-based)
// -> EVENT_WAIT -> callback -> handshake -> carrier -> deliver -> new pipe
// instance -> repeat.
typedef struct iree_net_shm_win32_listener_t {
  iree_net_listener_t base;
  iree_net_shm_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  // Current pipe instance awaiting ConnectNamedPipe. Set to
  // INVALID_HANDLE_VALUE after being consumed by the handshake or on cleanup.
  HANDLE pipe_handle;
  // Event signaled when ConnectNamedPipe completes. The event's Win32 HANDLE
  // is set as hEvent in the OVERLAPPED structure. The proactor monitors this
  // event via EVENT_WAIT. Retained for the listener's lifetime and reused
  // across accept cycles.
  iree_async_event_t* event;
  // OVERLAPPED for the pending ConnectNamedPipe call. hEvent points to the
  // event's primitive HANDLE.
  OVERLAPPED overlapped;
  iree_async_event_wait_operation_t event_wait_operation;
  struct {
    iree_net_listener_accept_callback_t fn;
    void* user_data;
  } accept;
  iree_net_shm_win32_listener_state_t state;
  iree_net_listener_stopped_callback_t stopped_callback;
  iree_allocator_t host_allocator;
  // Full address string (e.g., "pipe:my-service") for query_bound_address.
  // Null-terminated. The pipe: prefix is stripped and converted to a wide pipe
  // path on the stack each time a new pipe instance is created.
  iree_host_size_t address_length;
  char address[];
} iree_net_shm_win32_listener_t;

static const iree_net_listener_vtable_t iree_net_shm_win32_listener_vtable;

static void iree_net_shm_win32_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags);

// Creates a new named pipe instance. Stores the handle in
// listener->pipe_handle (which must be INVALID_HANDLE_VALUE on entry). The
// pipe is duplex, overlapped, byte-mode.
static iree_status_t iree_net_shm_win32_listener_create_pipe(
    iree_net_shm_win32_listener_t* listener) {
  iree_string_view_t name = iree_net_shm_win32_strip_pipe_prefix(
      iree_make_string_view(listener->address, listener->address_length));

  WCHAR pipe_path[IREE_NET_SHM_MAX_PIPE_PATH_LENGTH];
  int pipe_path_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_net_shm_win32_build_pipe_path(name, pipe_path, &pipe_path_length));

  listener->pipe_handle = CreateNamedPipeW(
      pipe_path, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
      PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, PIPE_UNLIMITED_INSTANCES,
      4096,   // Output buffer size (handshake messages are ~100 bytes).
      4096,   // Input buffer size.
      0,      // Default timeout (only affects WaitNamedPipe, not us).
      NULL);  // Default security attributes.
  if (listener->pipe_handle == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "CreateNamedPipeW failed for pipe '%.*s'",
                            (int)name.size, name.data);
  }

  return iree_ok_status();
}

// Starts an overlapped ConnectNamedPipe and submits an EVENT_WAIT to get
// notified when a client connects. The listener's pipe_handle must be valid
// and the event must be ready (not pending from a previous wait).
static iree_status_t iree_net_shm_win32_listener_start_accept(
    iree_net_shm_win32_listener_t* listener) {
  // Initialize the OVERLAPPED with the event's Win32 HANDLE. When the kernel
  // completes ConnectNamedPipe, it signals this event.
  memset(&listener->overlapped, 0, sizeof(listener->overlapped));
  listener->overlapped.hEvent =
      (HANDLE)listener->event->primitive.value.win32_handle;

  // Start listening for a client connection.
  BOOL connected =
      ConnectNamedPipe(listener->pipe_handle, &listener->overlapped);
  if (!connected) {
    DWORD error = GetLastError();
    if (error == ERROR_IO_PENDING) {
      // Normal case: waiting for a client to connect. The kernel will signal
      // the event when a client arrives.
    } else if (error == ERROR_PIPE_CONNECTED) {
      // A client connected between CreateNamedPipeW and ConnectNamedPipe.
      // Signal the event so the EVENT_WAIT fires immediately.
      IREE_RETURN_IF_ERROR(iree_async_event_set(listener->event));
    } else {
      return iree_make_status(iree_status_code_from_win32_error(error),
                              "ConnectNamedPipe failed");
    }
  } else {
    // ConnectNamedPipe returning TRUE with an overlapped structure is
    // documented as an error case by MSDN. Treat as unexpected success and
    // signal the event to proceed.
    IREE_RETURN_IF_ERROR(iree_async_event_set(listener->event));
  }

  // Submit EVENT_WAIT to deliver the completion to the proactor thread.
  memset(&listener->event_wait_operation, 0,
         sizeof(listener->event_wait_operation));
  iree_async_operation_initialize(
      &listener->event_wait_operation.base,
      IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT, IREE_ASYNC_OPERATION_FLAG_NONE,
      iree_net_shm_win32_listener_accept_complete, listener);
  listener->event_wait_operation.event = listener->event;
  iree_status_t status = iree_async_proactor_submit_one(
      listener->proactor, &listener->event_wait_operation.base);
  if (!iree_status_is_ok(status)) {
    // Submit failed: the pending ConnectNamedPipe has no completion path.
    // Close the pipe handle to cancel the overlapped IO and prevent leaking
    // the handle. This makes start_accept self-contained: callers don't need
    // to handle partial cleanup.
    CloseHandle(listener->pipe_handle);
    listener->pipe_handle = INVALID_HANDLE_VALUE;
  }
  return status;
}

// Re-arms the accept loop: creates a new pipe instance (the previous one was
// consumed by the handshake) and starts listening for the next connection.
static iree_status_t iree_net_shm_win32_listener_rearm(
    iree_net_shm_win32_listener_t* listener) {
  iree_status_t status = iree_net_shm_win32_listener_create_pipe(listener);
  if (!iree_status_is_ok(status)) return status;
  return iree_net_shm_win32_listener_start_accept(listener);
}

// Handles a successfully accepted pipe connection: takes ownership of the
// pipe, runs the server-side handshake, creates a carrier, wraps in a
// connection, and delivers to the consumer. On failure at any step, reports
// the error to the consumer.
static void iree_net_shm_win32_listener_handle_accepted(
    iree_net_shm_win32_listener_t* listener) {
  // Take ownership of the connected pipe. The handshake takes ownership of the
  // primitive and closes it on return (both success and error paths).
  HANDLE pipe = listener->pipe_handle;
  listener->pipe_handle = INVALID_HANDLE_VALUE;
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)pipe);

  // Get or create shared_wake for this proactor.
  iree_net_shm_shared_wake_t* shared_wake = NULL;
  iree_slim_mutex_lock(&listener->factory->mutex);
  iree_status_t status = iree_net_shm_factory_get_or_create_shared_wake(
      listener->factory, listener->proactor, &shared_wake);
  iree_slim_mutex_unlock(&listener->factory->mutex);
  if (!iree_status_is_ok(status)) {
    iree_async_primitive_close(&channel);
    listener->accept.fn(listener->accept.user_data, status, NULL);
    return;
  }

  // Run the server handshake. Synchronous -- completes in microseconds over a
  // local pipe. Closes the channel primitive on return.
  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  status = iree_net_shm_handshake_server(
      channel, shared_wake, listener->factory->options, listener->proactor,
      listener->host_allocator, &handshake_result);

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

// Accept completion callback. Fires on the proactor thread when the
// EVENT_WAIT for ConnectNamedPipe completes. Runs the server-side handshake,
// creates a carrier, delivers to the consumer, and re-arms for the next
// connection.
//
// The handshake is synchronous but completes in microseconds over a local
// named pipe. The 5s timeout in the handshake is a safety valve for
// pathological peers. During the handshake, the proactor thread is blocked --
// acceptable for local IPC bootstrapping.
static void iree_net_shm_win32_listener_accept_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Handle stopping: close the pipe and fire the stopped callback.
  if (listener->state == IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPING) {
    iree_status_ignore(status);
    if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(listener->pipe_handle);
      listener->pipe_handle = INVALID_HANDLE_VALUE;
    }
    listener->stopped_callback.fn(listener->stopped_callback.user_data);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  if (!iree_status_is_ok(status)) {
    // EVENT_WAIT failed (e.g., proactor shutdown). Report to consumer.
    if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(listener->pipe_handle);
      listener->pipe_handle = INVALID_HANDLE_VALUE;
    }
    listener->accept.fn(listener->accept.user_data, status, NULL);
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Confirm ConnectNamedPipe completed successfully via GetOverlappedResult.
  // bWait=FALSE because the I/O has already completed (the event was
  // signaled).
  DWORD bytes_transferred = 0;
  if (!GetOverlappedResult(listener->pipe_handle, &listener->overlapped,
                           &bytes_transferred, /*bWait=*/FALSE)) {
    DWORD error = GetLastError();
    CloseHandle(listener->pipe_handle);
    listener->pipe_handle = INVALID_HANDLE_VALUE;
    listener->accept.fn(
        listener->accept.user_data,
        iree_make_status(iree_status_code_from_win32_error(error),
                         "ConnectNamedPipe failed (overlapped result)"),
        NULL);
    // Re-arm with a fresh pipe instance.
    iree_status_t rearm_status = iree_net_shm_win32_listener_rearm(listener);
    if (!iree_status_is_ok(rearm_status)) {
      listener->accept.fn(listener->accept.user_data, rearm_status, NULL);
    }
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  // Connection accepted -- run the handshake and deliver the carrier.
  iree_net_shm_win32_listener_handle_accepted(listener);

  // Re-arm: create a new pipe instance and start accepting. Check state
  // first in case stop() was called during the (synchronous) handshake.
  if (listener->state == IREE_NET_SHM_WIN32_LISTENER_STATE_LISTENING) {
    iree_status_t rearm_status = iree_net_shm_win32_listener_rearm(listener);
    if (!iree_status_is_ok(rearm_status)) {
      listener->accept.fn(listener->accept.user_data, rearm_status, NULL);
    }
  } else {
    // stop() was called during the handshake. The pipe was consumed by the
    // handshake and no new I/O is pending, so fire stopped_callback directly.
    listener->stopped_callback.fn(listener->stopped_callback.user_data);
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_net_shm_win32_listener_free(
    iree_net_listener_t* base_listener) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
    CloseHandle(listener->pipe_handle);
  }
  if (listener->event) {
    iree_async_event_release(listener->event);
  }
  iree_allocator_t host_allocator = listener->host_allocator;
  iree_allocator_free(host_allocator, listener);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_net_shm_win32_listener_stop(
    iree_net_listener_t* base_listener,
    iree_net_listener_stopped_callback_t callback) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)base_listener;
  IREE_TRACE_ZONE_BEGIN(z0);
  listener->stopped_callback = callback;
  listener->state = IREE_NET_SHM_WIN32_LISTENER_STATE_STOPPING;
  // Cancel the pending ConnectNamedPipe. The kernel completes the I/O with
  // ERROR_OPERATION_ABORTED and signals the OVERLAPPED event. This triggers
  // the EVENT_WAIT, and the callback sees state=STOPPING for cleanup.
  //
  // If the pipe was already consumed (callback is running the handshake),
  // CancelIoEx has nothing to cancel. The callback will detect STOPPING after
  // the handshake and fire stopped_callback directly.
  if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
    CancelIoEx(listener->pipe_handle, &listener->overlapped);
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_net_shm_win32_listener_query_bound_address(
    iree_net_listener_t* base_listener, iree_host_size_t buffer_capacity,
    char* buffer, iree_string_view_t* out_address) {
  iree_net_shm_win32_listener_t* listener =
      (iree_net_shm_win32_listener_t*)base_listener;
  if (buffer_capacity < listener->address_length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "buffer too small for bound address");
  }
  memcpy(buffer, listener->address, listener->address_length);
  *out_address = iree_make_string_view(buffer, listener->address_length);
  return iree_ok_status();
}

static const iree_net_listener_vtable_t iree_net_shm_win32_listener_vtable = {
    .free = iree_net_shm_win32_listener_free,
    .stop = iree_net_shm_win32_listener_stop,
    .query_bound_address = iree_net_shm_win32_listener_query_bound_address,
};

// Creates a cross-process listener on a Windows named pipe. Validates the
// address, creates the pipe instance, and submits the initial accept operation.
iree_status_t iree_net_shm_factory_create_listener_win32(
    iree_net_shm_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_listener = NULL;

  // Validate the pipe name by building the wide path (catches empty names and
  // paths that exceed MAX_PATH).
  iree_string_view_t name = iree_net_shm_win32_strip_pipe_prefix(bind_address);
  WCHAR pipe_path[IREE_NET_SHM_MAX_PIPE_PATH_LENGTH];
  int pipe_path_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_net_shm_win32_build_pipe_path(name, pipe_path, &pipe_path_length));

  // Create the event for ConnectNamedPipe signaling. The event is reused
  // across accept cycles (one per listener, not per connection).
  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0,
                                    iree_async_event_create(proactor, &event));

  // Allocate the listener with trailing space for the address string.
  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      iree_sizeof_struct(iree_net_shm_win32_listener_t), &total_size,
      IREE_STRUCT_FIELD_FAM(bind_address.size + 1, char));
  iree_net_shm_win32_listener_t* listener = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&listener);
  }
  if (iree_status_is_ok(status)) {
    memset(listener, 0, total_size);
    listener->base.vtable = &iree_net_shm_win32_listener_vtable;
    listener->factory = factory;
    listener->proactor = proactor;
    listener->recv_pool = recv_pool;
    listener->pipe_handle = INVALID_HANDLE_VALUE;
    listener->event = event;
    listener->accept.fn = accept_callback;
    listener->accept.user_data = user_data;
    listener->host_allocator = host_allocator;
    listener->address_length = bind_address.size;
    memcpy(listener->address, bind_address.data, bind_address.size);
    listener->address[bind_address.size] = '\0';
  }

  // Create the first pipe instance.
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_win32_listener_create_pipe(listener);
  }

  // Start listening for the first connection.
  if (iree_status_is_ok(status)) {
    status = iree_net_shm_win32_listener_start_accept(listener);
  }

  if (iree_status_is_ok(status)) {
    *out_listener = &listener->base;
  } else {
    if (listener) {
      if (listener->pipe_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(listener->pipe_handle);
      }
      iree_allocator_free(host_allocator, listener);
    }
    iree_async_event_release(event);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Cross-process connect (Windows named pipe)
//===----------------------------------------------------------------------===//

// Heap-allocated state for async named pipe connect. Freed in the NOP
// completion callback after the handshake and carrier creation.
typedef struct iree_net_shm_win32_connect_state_t {
  iree_async_nop_operation_t nop_operation;
  struct {
    iree_net_transport_connect_callback_t fn;
    void* user_data;
  } callback;
  iree_net_shm_factory_t* factory;
  iree_async_proactor_t* proactor;
  iree_async_buffer_pool_t* recv_pool;
  // Connected pipe handle, passed to the handshake. Set to INVALID_HANDLE_VALUE
  // after the handshake takes ownership.
  HANDLE pipe_handle;
  iree_allocator_t host_allocator;
} iree_net_shm_win32_connect_state_t;

// Handles the connected pipe: runs the client-side handshake, creates a
// carrier, wraps in a connection, and delivers to the consumer. On failure at
// any step, reports the error to the consumer.
static void iree_net_shm_win32_connect_handle_connected(
    iree_net_shm_win32_connect_state_t* state) {
  // Take ownership of the pipe for the handshake. The handshake closes the
  // channel primitive on return (both success and error paths).
  iree_async_primitive_t channel =
      iree_async_primitive_from_win32_handle((uintptr_t)state->pipe_handle);
  state->pipe_handle = INVALID_HANDLE_VALUE;

  // Get or create shared_wake for this proactor.
  iree_net_shm_shared_wake_t* shared_wake = NULL;
  iree_slim_mutex_lock(&state->factory->mutex);
  iree_status_t status = iree_net_shm_factory_get_or_create_shared_wake(
      state->factory, state->proactor, &shared_wake);
  iree_slim_mutex_unlock(&state->factory->mutex);
  if (!iree_status_is_ok(status)) {
    iree_async_primitive_close(&channel);
    state->callback.fn(state->callback.user_data, status, NULL);
    return;
  }

  // Run the client handshake. Synchronous -- completes in microseconds over a
  // local pipe. Closes the channel primitive on return.
  iree_net_shm_handshake_result_t handshake_result;
  memset(&handshake_result, 0, sizeof(handshake_result));
  status =
      iree_net_shm_handshake_client(channel, shared_wake, state->proactor,
                                    state->host_allocator, &handshake_result);

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

// NOP completion callback for deferred delivery of the connect result. The
// pipe is already connected (CreateFile succeeded synchronously); we use a NOP
// to run the handshake on the proactor thread.
static void iree_net_shm_win32_connect_nop_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  iree_net_shm_win32_connect_state_t* state =
      (iree_net_shm_win32_connect_state_t*)user_data;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (iree_status_is_ok(status)) {
    iree_net_shm_win32_connect_handle_connected(state);
  } else {
    // NOP failed (cancelled or proactor shutdown). Close the pipe and report.
    if (state->pipe_handle != INVALID_HANDLE_VALUE) {
      CloseHandle(state->pipe_handle);
    }
    state->callback.fn(state->callback.user_data, status, NULL);
  }

  iree_allocator_free(state->host_allocator, state);
  IREE_TRACE_ZONE_END(z0);
}

// Initiates a cross-process connect to a Windows named pipe. Opens the pipe
// via CreateFile (which connects synchronously), then submits a NOP to defer
// the handshake to the proactor thread.
iree_status_t iree_net_shm_factory_connect_win32(
    iree_net_shm_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Build the wide pipe path from the address.
  iree_string_view_t name = iree_net_shm_win32_strip_pipe_prefix(address);
  WCHAR pipe_path[IREE_NET_SHM_MAX_PIPE_PATH_LENGTH];
  int pipe_path_length = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_net_shm_win32_build_pipe_path(name, pipe_path, &pipe_path_length));

  // Open the pipe. CreateFile on a named pipe connects to an existing server
  // instance synchronously -- either it succeeds immediately or fails.
  HANDLE pipe = CreateFileW(pipe_path, GENERIC_READ | GENERIC_WRITE,
                            0,     // No sharing.
                            NULL,  // Default security.
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED,
                            NULL);  // No template.
  if (pipe == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    if (error == ERROR_PIPE_BUSY) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "all pipe instances busy for '%.*s'; retry later",
                              (int)name.size, name.data);
    } else if (error == ERROR_FILE_NOT_FOUND) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "no server listening on pipe '%.*s'",
                              (int)name.size, name.data);
    }
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "CreateFileW failed for pipe '%.*s'",
                            (int)name.size, name.data);
  }

  // Allocate connect state for the deferred handshake.
  iree_net_shm_win32_connect_state_t* state = NULL;
  iree_status_t status = iree_allocator_malloc(factory->host_allocator,
                                               sizeof(*state), (void**)&state);
  if (!iree_status_is_ok(status)) {
    CloseHandle(pipe);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  memset(state, 0, sizeof(*state));
  state->callback.fn = callback;
  state->callback.user_data = user_data;
  state->factory = factory;
  state->proactor = proactor;
  state->recv_pool = recv_pool;
  state->pipe_handle = pipe;
  state->host_allocator = factory->host_allocator;

  // Submit a NOP for deferred delivery. The handshake runs on the proactor
  // thread in the NOP callback.
  iree_async_operation_initialize(
      &state->nop_operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_shm_win32_connect_nop_complete,
      state);
  status = iree_async_proactor_submit_one(proactor, &state->nop_operation.base);
  if (!iree_status_is_ok(status)) {
    CloseHandle(pipe);
    iree_allocator_free(factory->host_allocator, state);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

#endif  // IREE_PLATFORM_WINDOWS
