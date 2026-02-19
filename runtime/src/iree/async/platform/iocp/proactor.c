// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/iocp/proactor.h"

#include <string.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/iocp/socket.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/memory.h"

#if defined(IREE_PLATFORM_WINDOWS)

// Windows headers — winsock2.h must precede windows.h to avoid conflicts.
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <windows.h>
// clang-format on

// Verify scatter-gather constant matches across headers.
_Static_assert(IREE_ASYNC_IOCP_MAX_SCATTER_GATHER_BUFFERS ==
                   IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS,
               "IOCP carrier WSABUF count must match scatter-gather max");

// Verify non-Windows placeholder sizes in proactor.h are large enough for the
// real Windows structs. If these fire, update the placeholder arrays in the
// #else branch of iree_async_iocp_carrier_t.
_Static_assert(sizeof(OVERLAPPED) <= 32,
               "overlapped_placeholder[32] too small for OVERLAPPED");
_Static_assert(sizeof(((iree_async_iocp_carrier_t*)0)->data) <= 320,
               "data_placeholder[320] too small for carrier data union");

//===----------------------------------------------------------------------===//
// Console signal handling — global state
//===----------------------------------------------------------------------===//

// CompletionKey used by the console ctrl handler to post signal delivery
// completions to the IOCP port. Value 1 cannot be a valid operation pointer
// (operations are at least 4-byte aligned), so it is unambiguous.
#define IREE_ASYNC_IOCP_SIGNAL_COMPLETION_KEY ((ULONG_PTR)1)

// IOCP port handle for the proactor that owns signal handling. The console ctrl
// handler fires on an OS thread pool thread with no user_data parameter, so we
// store the completion port globally. Only one proactor may own signals
// (enforced by iree_async_signal_claim_ownership).
static HANDLE g_iocp_signal_completion_port = NULL;

// Atomic bitmask of pending console control events. The handler ORs in
// (1 << ctrl_type) for each event; the poll thread atomically exchanges the
// bitmask to 0 and dispatches all accumulated signals. Using a bitmask (not a
// single variable) prevents the race where a rapid CTRL_C then CTRL_CLOSE would
// lose the first event.
static iree_atomic_int32_t g_iocp_pending_ctrl_events = IREE_ATOMIC_VAR_INIT(0);

// Console ctrl handler registered via SetConsoleCtrlHandler. Fires on an OS
// thread pool thread — no proactor state access, only atomic bitmask update
// and IOCP post.
static BOOL WINAPI
iree_async_proactor_iocp_console_ctrl_handler(DWORD ctrl_type) {
  switch (ctrl_type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
      iree_atomic_fetch_or(&g_iocp_pending_ctrl_events,
                           (int32_t)(1u << ctrl_type),
                           iree_memory_order_release);
      // Wake the poll thread. The completion port handle is guaranteed non-NULL
      // while the handler is registered (set before SetConsoleCtrlHandler,
      // cleared after RemoveConsoleCtrlHandler).
      PostQueuedCompletionStatus(g_iocp_signal_completion_port, 0,
                                 IREE_ASYNC_IOCP_SIGNAL_COMPLETION_KEY, NULL);
      return TRUE;
    default:
      return FALSE;
  }
}

//===----------------------------------------------------------------------===//
// Wake
//===----------------------------------------------------------------------===//

void iree_async_proactor_iocp_wake(iree_async_proactor_t* base_proactor) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  // Post a sentinel completion with NULL overlapped to wake the poll thread.
  // CompletionKey = 0 and lpOverlapped = NULL distinguish this from real I/O.
  PostQueuedCompletionStatus((HANDLE)proactor->completion_port, 0, 0, NULL);
}

//===----------------------------------------------------------------------===//
// Pending queue helpers
//===----------------------------------------------------------------------===//

// Retains operation resources and pushes the operation to the pending_queue for
// the poll thread to process. Uses the operation's `next` pointer (offset 0) as
// the slist_entry, which is safe because the operation is exclusively owned by
// the queue until the poll thread pops it.
void iree_async_proactor_iocp_push_pending(iree_async_proactor_iocp_t* proactor,
                                           iree_async_operation_t* operation) {
  iree_async_operation_retain_resources(operation);
  iree_atomic_slist_push(&proactor->pending_queue,
                         (iree_atomic_slist_entry_t*)operation);
  iree_async_proactor_iocp_wake(&proactor->base);
}

// Returns the native HANDLE for a socket or file I/O operation, for use with
// CancelIoEx. The socket/file struct is ref-counted and retained during the
// operation's lifetime (via retain_resources at submit), so this is safe to
// call from any thread while the operation is in flight.
static HANDLE iree_async_proactor_iocp_handle_from_io_operation(
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return (HANDLE)((iree_async_socket_accept_operation_t*)operation)
          ->listen_socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return (HANDLE)((iree_async_socket_connect_operation_t*)operation)
          ->socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return (HANDLE)((iree_async_socket_recv_operation_t*)operation)
          ->socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return (HANDLE)((iree_async_socket_send_operation_t*)operation)
          ->socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return (HANDLE)((iree_async_socket_sendto_operation_t*)operation)
          ->socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return (HANDLE)((iree_async_socket_recvfrom_operation_t*)operation)
          ->socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return (HANDLE)((iree_async_socket_recv_pool_operation_t*)operation)
          ->socket->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
      return (HANDLE)((iree_async_file_read_operation_t*)operation)
          ->file->primitive.value.win32_handle;
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
      return (HANDLE)((iree_async_file_write_operation_t*)operation)
          ->file->primitive.value.win32_handle;
    default:
      return INVALID_HANDLE_VALUE;
  }
}

// Returns the socket associated with a socket I/O operation, or NULL for
// non-socket operations and operations where failure doesn't indicate a broken
// socket (accept errors reflect transient resource issues, not a broken
// listener; close errors are moot).
static iree_async_socket_t* iree_async_proactor_iocp_socket_from_io_operation(
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return ((iree_async_socket_connect_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return ((iree_async_socket_recv_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return ((iree_async_socket_send_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return ((iree_async_socket_recv_pool_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return ((iree_async_socket_recvfrom_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return ((iree_async_socket_sendto_operation_t*)operation)->socket;
    default:
      return NULL;
  }
}

// Dispatches completion for a single operation: sticky failure on socket,
// linked continuation dispatch, resource release, then user callback.
//
// For intermediate multishot completions (MORE flag set), resource release and
// linked continuation dispatch are skipped. The operation is still in flight:
// resources must remain retained until the final completion, and linked
// continuations should only fire once at the end of the multishot sequence.
static void iree_async_proactor_iocp_dispatch_completion(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation,
    iree_status_t status, iree_async_completion_flags_t flags,
    iree_host_size_t* completed_count) {
  // Set sticky failure on the socket when I/O completes with an error.
  if (!iree_status_is_ok(status)) {
    iree_async_socket_t* socket =
        iree_async_proactor_iocp_socket_from_io_operation(operation);
    if (socket) {
      iree_async_socket_set_failure(socket, status);
    }
  }
  if (!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
    *completed_count += iree_async_proactor_iocp_dispatch_linked_continuation(
        proactor, operation, status);
    iree_async_operation_release_resources(operation);
  }
  if (operation->completion_fn) {
    operation->completion_fn(operation->user_data, operation, status, flags);
  } else {
    iree_status_ignore(status);
  }
  ++(*completed_count);
}

//===----------------------------------------------------------------------===//
// Create
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_create_iocp(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  // Message pool capacity from options, with default fallback.
  iree_host_size_t message_pool_capacity = options.message_pool_capacity;
  if (message_pool_capacity == 0) {
    message_pool_capacity = IREE_ASYNC_MESSAGE_POOL_DEFAULT_CAPACITY;
  }

  // Calculate allocation layout with trailing message pool entries.
  iree_host_size_t total_size = 0;
  iree_host_size_t message_entries_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(sizeof(iree_async_proactor_iocp_t), &total_size,
                             IREE_STRUCT_FIELD(message_pool_capacity,
                                               iree_async_message_pool_entry_t,
                                               &message_entries_offset)));

  // Allocate the proactor structure with trailing data.
  iree_async_proactor_iocp_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&proactor));
  memset(proactor, 0, total_size);

  // Initialize base fields.
  iree_async_proactor_initialize(&iree_async_proactor_iocp_vtable,
                                 options.debug_name, allocator,
                                 &proactor->base);

  // Initialize sequence emulator for SEQUENCE operation support.
  iree_async_sequence_emulator_initialize(&proactor->sequence_emulator,
                                          &proactor->base,
                                          iree_async_proactor_submit_one);

  // Initialize message pool with trailing data entries.
  iree_async_message_pool_entry_t* message_entries =
      (iree_async_message_pool_entry_t*)((uint8_t*)proactor +
                                         message_entries_offset);
  iree_async_message_pool_initialize(message_pool_capacity, message_entries,
                                     &proactor->message_pool);

  // Initialize MPSC queues.
  iree_atomic_slist_initialize(&proactor->pending_queue);
  iree_atomic_slist_initialize(&proactor->pending_semaphore_waits);

  // Initialize timer list.
  iree_async_iocp_timer_list_initialize(&proactor->timers);

  // Initialize signal dispatch state.
  iree_async_signal_dispatch_state_initialize(&proactor->signal.dispatch_state);

  // Initialize Winsock. WSAStartup is ref-counted by Windows, so multiple
  // calls are safe (each proactor calls startup/cleanup independently).
  WSADATA wsa_data;
  int wsa_result = WSAStartup(MAKEWORD(2, 2), &wsa_data);
  if (wsa_result != 0) {
    iree_allocator_free(allocator, proactor);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "WSAStartup failed (error %d)", wsa_result);
  }

  // Create the IOCP completion port.
  // INVALID_HANDLE_VALUE creates a new port not associated with any file.
  // NumberOfConcurrentThreads = 0 means use the number of processors.
  HANDLE completion_port =
      CreateIoCompletionPort(INVALID_HANDLE_VALUE, NULL, 0, 0);
  if (completion_port == NULL) {
    DWORD error = GetLastError();
    WSACleanup();
    iree_allocator_free(allocator, proactor);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateIoCompletionPort failed (error %lu)",
                            (unsigned long)error);
  }
  proactor->completion_port = (uintptr_t)completion_port;

  // Detect and apply capabilities.
  // MULTISHOT is supported via re-arm emulation (no kernel support needed).
  proactor->capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_REGISTERED_BUFFERS |
                           IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT |
                           IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS |
                           IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT;
  proactor->capabilities &= options.allowed_capabilities;

  *out_proactor = &proactor->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//

// Dispatches a single IREE signal to all subscribers. Handles deferred
// unsubscribes triggered by callbacks (same pattern as POSIX backend).
static void iree_async_proactor_iocp_signal_dispatch(
    iree_async_proactor_iocp_t* proactor, iree_async_signal_t signal) {
  iree_async_signal_subscription_t* head =
      proactor->signal.subscriptions[signal];
  if (!head) return;

  iree_async_signal_subscription_t* to_free =
      iree_async_signal_subscription_dispatch(
          head, &proactor->signal.dispatch_state, signal);

  // Free any subscriptions that were deferred-unsubscribed during dispatch.
  while (to_free) {
    iree_async_signal_subscription_t* subscription = to_free;
    iree_async_signal_t sub_signal = subscription->signal;
    to_free = to_free->pending_next;
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[sub_signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);
    --proactor->signal.subscriber_count[sub_signal];
  }
}

// Drains the global pending ctrl events bitmask and dispatches to subscribers.
// Called from the poll thread when a SIGNAL_COMPLETION_KEY completion arrives.
static void iree_async_proactor_iocp_dispatch_pending_signals(
    iree_async_proactor_iocp_t* proactor) {
  if (!proactor->signal.initialized) return;

  int32_t events = iree_atomic_exchange(&g_iocp_pending_ctrl_events, 0,
                                        iree_memory_order_acquire);
  if (events == 0) return;

  if (events & ((1 << CTRL_C_EVENT) | (1 << CTRL_BREAK_EVENT))) {
    iree_async_proactor_iocp_signal_dispatch(proactor,
                                             IREE_ASYNC_SIGNAL_INTERRUPT);
  }
  if (events & ((1 << CTRL_CLOSE_EVENT) | (1 << CTRL_LOGOFF_EVENT) |
                (1 << CTRL_SHUTDOWN_EVENT))) {
    iree_async_proactor_iocp_signal_dispatch(proactor,
                                             IREE_ASYNC_SIGNAL_TERMINATE);
  }
}

// Lazy-initializes console signal handling on first subscribe. Sets up the
// global completion port pointer and registers the console ctrl handler.
static iree_status_t iree_async_proactor_iocp_signal_initialize(
    iree_async_proactor_iocp_t* proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Detect console presence. Services and GUI apps without a console cannot
  // receive console control events.
  HANDLE stdin_handle = GetStdHandle(STD_INPUT_HANDLE);
  if (stdin_handle == NULL || stdin_handle == INVALID_HANDLE_VALUE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "signal handling requires a console application on Windows");
  }

  // Store completion port for the global console ctrl handler.
  g_iocp_signal_completion_port = (HANDLE)proactor->completion_port;

  if (!SetConsoleCtrlHandler(iree_async_proactor_iocp_console_ctrl_handler,
                             TRUE)) {
    DWORD error = GetLastError();
    g_iocp_signal_completion_port = NULL;
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "SetConsoleCtrlHandler failed (error %lu)",
                            (unsigned long)error);
  }

  proactor->signal.initialized = true;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Tears down signal handling during proactor destroy. Removes the console ctrl
// handler, clears global state, frees all subscriptions, and releases global
// signal ownership.
static void iree_async_proactor_iocp_signal_deinitialize(
    iree_async_proactor_iocp_t* proactor) {
  iree_allocator_t allocator = proactor->base.allocator;

  // Remove console ctrl handler before clearing the completion port pointer.
  // SetConsoleCtrlHandler(handler, FALSE) synchronously removes the handler,
  // so after this returns no new completions will be posted.
  SetConsoleCtrlHandler(iree_async_proactor_iocp_console_ctrl_handler, FALSE);
  g_iocp_signal_completion_port = NULL;

  // Drain any pending events that arrived before the handler was removed.
  iree_atomic_exchange(&g_iocp_pending_ctrl_events, 0,
                       iree_memory_order_acquire);

  // Free all signal subscriptions.
  for (int i = 0; i < IREE_ASYNC_SIGNAL_COUNT; ++i) {
    while (proactor->signal.subscriptions[i]) {
      iree_async_signal_subscription_t* subscription =
          proactor->signal.subscriptions[i];
      proactor->signal.subscriptions[i] = subscription->next;
      iree_allocator_free(allocator, subscription);
    }
    proactor->signal.subscriber_count[i] = 0;
  }

  proactor->signal.initialized = false;
  iree_async_signal_release_ownership(&proactor->base);
}

static iree_status_t iree_async_proactor_iocp_subscribe_signal(
    iree_async_proactor_t* base_proactor, iree_async_signal_t signal,
    iree_async_signal_callback_t callback,
    iree_async_signal_subscription_t** out_subscription) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_subscription);
  *out_subscription = NULL;

  if (signal <= IREE_ASYNC_SIGNAL_NONE || signal >= IREE_ASYNC_SIGNAL_COUNT) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid signal type %d", (int)signal);
  }

  // Only INTERRUPT and TERMINATE are supported on Windows. Other signal types
  // (HANGUP, QUIT, USER1, USER2) have no Windows equivalent.
  if (signal != IREE_ASYNC_SIGNAL_INTERRUPT &&
      signal != IREE_ASYNC_SIGNAL_TERMINATE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "signal %.*s not supported on Windows "
                            "(only INTERRUPT and TERMINATE are available)",
                            (int)iree_async_signal_name(signal).size,
                            iree_async_signal_name(signal).data);
  }

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Claim global signal ownership (first proactor wins).
  iree_status_t status = iree_async_signal_claim_ownership(base_proactor);

  // Lazy-initialize console signal handling on first subscriber.
  if (iree_status_is_ok(status) && !proactor->signal.initialized) {
    status = iree_async_proactor_iocp_signal_initialize(proactor);
    if (!iree_status_is_ok(status)) {
      iree_async_signal_release_ownership(base_proactor);
    }
  }

  // Allocate and initialize the subscription.
  iree_async_signal_subscription_t* subscription = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(allocator, sizeof(*subscription),
                                   (void**)&subscription);
  }
  if (iree_status_is_ok(status)) {
    iree_async_signal_subscription_initialize(subscription, base_proactor,
                                              signal, callback);
    iree_async_signal_subscription_link(&proactor->signal.subscriptions[signal],
                                        subscription);
    ++proactor->signal.subscriber_count[signal];
    *out_subscription = subscription;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_async_proactor_iocp_unsubscribe_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_signal_subscription_t* subscription) {
  if (!subscription) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_async_signal_t signal = subscription->signal;

  // If we're currently dispatching signals, defer the unsubscribe to avoid
  // corrupting the subscription list during iteration.
  if (proactor->signal.dispatch_state.dispatching) {
    iree_async_signal_subscription_defer_unsubscribe(
        &proactor->signal.dispatch_state, subscription);
  } else {
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);
    --proactor->signal.subscriber_count[signal];
  }

  IREE_TRACE_ZONE_END(z0);
}

// Forward declarations for relay helpers (defined in the Relay section below).
// These are needed because the destroy and poll-loop dispatch functions call
// them before their definition point in the file.
static bool iree_async_proactor_iocp_relay_fire_sink(iree_async_relay_t* relay);
static void iree_async_proactor_iocp_relay_unlink(
    iree_async_proactor_iocp_t* proactor, iree_async_relay_t* relay);
static void iree_async_proactor_iocp_relay_release_resources(
    iree_async_relay_t* relay);
static void iree_async_proactor_iocp_relay_fault(
    iree_async_proactor_iocp_t* proactor, iree_async_relay_t* relay,
    iree_status_t status);
static bool iree_async_proactor_iocp_notification_has_consumers(
    iree_async_notification_t* notification);

//===----------------------------------------------------------------------===//
// Destroy
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_destroy(
    iree_async_proactor_t* base_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Unregister and free all active event wait carriers.
  while (proactor->active_carriers) {
    iree_async_iocp_carrier_t* carrier = proactor->active_carriers;
    proactor->active_carriers = carrier->next;
    // Synchronous unregister: INVALID_HANDLE_VALUE blocks until the callback
    // has completed or been cancelled, ensuring no dangling references.
    if (carrier->type == IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT &&
        carrier->data.event_wait.wait_handle != NULL) {
      UnregisterWaitEx(carrier->data.event_wait.wait_handle,
                       INVALID_HANDLE_VALUE);
    }
    iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                          iree_memory_order_relaxed);
    iree_allocator_free(allocator, carrier);
  }

  // All carriers (EVENT_WAIT, SOCKET_IO, ACCEPT, etc.) should have been freed
  // by completion dispatch or the active_carriers cleanup above. A non-zero
  // count means the caller destroyed the proactor with pending overlapped I/O.
  IREE_ASSERT(iree_atomic_load(&proactor->outstanding_carrier_count,
                               iree_memory_order_relaxed) == 0);

  // Tear down console signal handling if initialized.
  if (proactor->signal.initialized) {
    iree_async_proactor_iocp_signal_deinitialize(proactor);
  }

  // Free all event sources.
  while (proactor->event_sources) {
    iree_async_event_source_t* source = proactor->event_sources;
    proactor->event_sources = source->next;
    iree_allocator_free(source->allocator, source);
  }

  // Free all relays, releasing retained notifications.
  while (proactor->relays) {
    struct iree_async_relay_t* relay = proactor->relays;
    proactor->relays = relay->next;
    relay->next = NULL;
    relay->prev = NULL;
    iree_async_proactor_iocp_relay_release_resources(relay);
  }

  // Close the completion port.
  if (proactor->completion_port != 0) {
    CloseHandle((HANDLE)proactor->completion_port);
    proactor->completion_port = 0;
  }

  // Clean up Winsock (ref-counted, matches WSAStartup in create).
  WSACleanup();

  // Deinitialize MPSC queues.
  iree_atomic_slist_deinitialize(&proactor->pending_queue);
  iree_atomic_slist_deinitialize(&proactor->pending_semaphore_waits);

  // Deinitialize message pool.
  iree_async_message_pool_deinitialize(&proactor->message_pool);

  iree_allocator_free(allocator, proactor);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Query capabilities
//===----------------------------------------------------------------------===//

static iree_async_proactor_capabilities_t
iree_async_proactor_iocp_query_capabilities(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  return proactor->capabilities;
}

//===----------------------------------------------------------------------===//
// Event wait callback (RegisterWaitForSingleObject)
//===----------------------------------------------------------------------===//

// Callback invoked by the OS thread pool when a RegisterWaitForSingleObject
// wait is satisfied. Posts a completion to the IOCP port for the poll thread
// to dispatch. This is the ONLY work done on the thread pool thread — no
// proactor state mutation.
static VOID CALLBACK
iree_async_proactor_iocp_event_wait_callback(PVOID context, BOOLEAN timed_out) {
  iree_async_iocp_carrier_t* carrier = (iree_async_iocp_carrier_t*)context;
  // timed_out is always FALSE because we register with INFINITE timeout.
  (void)timed_out;
  PostQueuedCompletionStatus((HANDLE)carrier->completion_port, 0, 0,
                             &carrier->overlapped);
}

//===----------------------------------------------------------------------===//
// Poll: pending queue drain
//===----------------------------------------------------------------------===//

// Drains the pending_queue and registers operations with the appropriate
// subsystem (timer_list, RegisterWaitForSingleObject, etc.).
// Returns the number of operations that completed directly during drain
// (e.g., cancelled operations that fire callbacks inline).
static iree_host_size_t iree_async_proactor_iocp_drain_pending_queue(
    iree_async_proactor_iocp_t* proactor) {
  iree_host_size_t direct_completions = 0;

  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_queue,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;
  }
  iree_atomic_slist_entry_t* entry = head;
  while (entry) {
    iree_async_operation_t* operation = (iree_async_operation_t*)entry;
    entry = entry->next;
    operation->next = NULL;

    // Check for cancellation before registering. If cancel() was called
    // before the operation was drained from the pending queue, dispatch
    // CANCELLED immediately without registering. Decrement the type-specific
    // cancellation counter since the drain scan will never find this
    // operation (it was never registered).
    if (iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                         IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
      if (operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
        iree_atomic_fetch_sub(&proactor->pending_timer_cancellation_count, 1,
                              iree_memory_order_release);
      } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
        iree_atomic_fetch_sub(&proactor->pending_event_wait_cancellation_count,
                              1, iree_memory_order_release);
      }
      iree_async_proactor_iocp_dispatch_completion(
          proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
      continue;
    }

    switch (operation->type) {
      case IREE_ASYNC_OPERATION_TYPE_NOP:
        // NOP completes inline during drain. This enables same-poll-iteration
        // completion for NOPs submitted from callbacks (e.g., sequence
        // emulation step advancement from trampolines).
        iree_async_proactor_iocp_dispatch_completion(
            proactor, operation, iree_ok_status(),
            IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
        break;

      case IREE_ASYNC_OPERATION_TYPE_TIMER: {
        iree_async_iocp_timer_list_insert(
            &proactor->timers, (iree_async_timer_operation_t*)operation);
        break;
      }

      case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
        iree_async_event_wait_operation_t* event_wait =
            (iree_async_event_wait_operation_t*)operation;

        // Allocate a carrier for the RegisterWaitForSingleObject callback.
        iree_async_iocp_carrier_t* carrier = NULL;
        iree_status_t status = iree_allocator_malloc(
            proactor->base.allocator, sizeof(*carrier), (void**)&carrier);
        if (!iree_status_is_ok(status)) {
          // Allocation failed: complete with the error.
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation, status, IREE_ASYNC_COMPLETION_FLAG_NONE,
              &direct_completions);
          break;
        }
        memset(carrier, 0, sizeof(*carrier));
        carrier->type = IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT;
        carrier->completion_port = proactor->completion_port;
        carrier->operation = operation;
        iree_atomic_fetch_add(&proactor->outstanding_carrier_count, 1,
                              iree_memory_order_relaxed);

        // Register wait on the event's HANDLE. WT_EXECUTEONLYONCE means the
        // wait is automatically unregistered after the callback fires once.
        HANDLE event_handle =
            (HANDLE)event_wait->event->primitive.value.win32_handle;
        BOOL registered = RegisterWaitForSingleObject(
            &carrier->data.event_wait.wait_handle, event_handle,
            iree_async_proactor_iocp_event_wait_callback, carrier, INFINITE,
            WT_EXECUTEONLYONCE);
        if (!registered) {
          DWORD error = GetLastError();
          iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                iree_memory_order_relaxed);
          iree_allocator_free(proactor->base.allocator, carrier);
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation,
              iree_make_status(IREE_STATUS_INTERNAL,
                               "RegisterWaitForSingleObject failed (error %lu)",
                               (unsigned long)error),
              IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
          break;
        }

        // Link into active carrier list for cleanup during destroy.
        carrier->next = proactor->active_carriers;
        proactor->active_carriers = carrier;
        break;
      }

      case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
        iree_async_notification_wait_operation_t* notification_wait =
            (iree_async_notification_wait_operation_t*)operation;
        iree_async_notification_t* notification =
            notification_wait->notification;

        // Check if epoch already advanced past the captured token.
        uint32_t current_epoch = (uint32_t)iree_atomic_load(
            &notification->epoch, iree_memory_order_acquire);
        if (current_epoch != notification_wait->wait_token) {
          // Already satisfied: dispatch completion immediately.
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation, iree_ok_status(),
              IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
          break;
        }

        // Not yet satisfied: link into notification's pending_waits list.
        operation->next =
            (iree_async_operation_t*)notification->platform.iocp.pending_waits;
        notification->platform.iocp.pending_waits = notification_wait;

        // Add notification to proactor's tracking list if not already there.
        if (!notification->platform.iocp.in_wait_list) {
          notification->platform.iocp.next_with_waits =
              proactor->notifications_with_waits;
          proactor->notifications_with_waits = notification;
          notification->platform.iocp.in_wait_list = true;
        }
        break;
      }

      default:
        // Unsupported operation type in pending queue. This should not happen
        // because submit_operation only pushes supported types. Complete with
        // an error rather than silently dropping.
        iree_async_proactor_iocp_dispatch_completion(
            proactor, operation,
            iree_make_status(IREE_STATUS_INTERNAL,
                             "unexpected operation type %d in pending queue",
                             (int)operation->type),
            IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
        break;
    }
  }

  return direct_completions;
}

//===----------------------------------------------------------------------===//
// Poll: timer cancellation scan
//===----------------------------------------------------------------------===//

// Scans the timer list for cancelled timers and dispatches CANCELLED
// completions. Called when pending_timer_cancellation_count > 0.
static iree_host_size_t iree_async_proactor_iocp_drain_timer_cancellations(
    iree_async_proactor_iocp_t* proactor) {
  int32_t cancellation_count = iree_atomic_load(
      &proactor->pending_timer_cancellation_count, iree_memory_order_acquire);
  if (cancellation_count == 0) return 0;

  iree_host_size_t direct_completions = 0;

  // Walk the timer list looking for cancelled entries. We walk from head to
  // tail, collecting cancelled timers. We can't remove during iteration
  // because removal modifies the list; instead collect pointers first.
  // For simplicity with small lists, iterate and remove one at a time.
  iree_async_timer_operation_t* timer = proactor->timers.head;
  while (timer && cancellation_count > 0) {
    iree_async_timer_operation_t* next = timer->platform.iocp.next;
    if (iree_any_bit_set(iree_async_operation_load_internal_flags(&timer->base),
                         IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
      iree_async_iocp_timer_list_remove(&proactor->timers, timer);
      iree_atomic_fetch_sub(&proactor->pending_timer_cancellation_count, 1,
                            iree_memory_order_release);
      --cancellation_count;
      iree_async_proactor_iocp_dispatch_completion(
          proactor, &timer->base, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
    }
    timer = next;
  }

  return direct_completions;
}

//===----------------------------------------------------------------------===//
// Poll: event wait cancellation drain
//===----------------------------------------------------------------------===//

// Scans the active_carriers list for cancelled event wait carriers and
// dispatches CANCELLED completions. Called when
// pending_event_wait_cancellation_count > 0.
//
// For each cancelled carrier:
//   - UnregisterWaitEx(wait_handle, NULL) attempts a non-blocking unregister.
//     If it returns TRUE, the wait was successfully unregistered before the
//     callback fired. If FALSE with ERROR_IO_PENDING, the callback is currently
//     executing or already posted to IOCP — the carrier dispatch (Phase 6) will
//     check the CANCELLED flag and handle it.
static iree_host_size_t iree_async_proactor_iocp_drain_event_wait_cancellations(
    iree_async_proactor_iocp_t* proactor) {
  int32_t cancellation_count =
      iree_atomic_load(&proactor->pending_event_wait_cancellation_count,
                       iree_memory_order_acquire);
  if (cancellation_count == 0) return 0;

  iree_host_size_t direct_completions = 0;

  iree_async_iocp_carrier_t** prev_ptr = &proactor->active_carriers;
  iree_async_iocp_carrier_t* carrier = *prev_ptr;
  while (carrier && cancellation_count > 0) {
    iree_async_iocp_carrier_t* next = carrier->next;

    if (carrier->type != IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT ||
        !iree_any_bit_set(
            iree_async_operation_load_internal_flags(carrier->operation),
            IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
      prev_ptr = &carrier->next;
      carrier = next;
      continue;
    }

    // Try to unregister the wait. NULL means non-blocking: don't wait for
    // an in-progress callback to finish.
    BOOL unregistered =
        UnregisterWaitEx(carrier->data.event_wait.wait_handle, NULL);
    if (unregistered) {
      // Successfully unregistered before the callback fired.
      // Unlink from active list, free carrier, dispatch CANCELLED.
      *prev_ptr = next;
      iree_async_operation_t* operation = carrier->operation;
      operation->next = NULL;
      iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                            iree_memory_order_relaxed);
      iree_allocator_free(proactor->base.allocator, carrier);
      iree_atomic_fetch_sub(&proactor->pending_event_wait_cancellation_count, 1,
                            iree_memory_order_release);
      --cancellation_count;
      iree_async_proactor_iocp_dispatch_completion(
          proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE, &direct_completions);
    } else {
      // ERROR_IO_PENDING: the callback is executing or already posted to IOCP.
      // The carrier dispatch (Phase 6) will check the CANCELLED flag.
      // Decrement the counter since we've "handled" this cancellation request
      // — the carrier dispatch will do the actual CANCELLED delivery.
      iree_atomic_fetch_sub(&proactor->pending_event_wait_cancellation_count, 1,
                            iree_memory_order_release);
      --cancellation_count;
      prev_ptr = &carrier->next;
    }
    carrier = next;
  }

  return direct_completions;
}

//===----------------------------------------------------------------------===//
// Poll: timer expiration processing
//===----------------------------------------------------------------------===//

// Processes expired timers by dispatching completions.
static iree_host_size_t iree_async_proactor_iocp_process_expired_timers(
    iree_async_proactor_iocp_t* proactor) {
  iree_time_t now = iree_time_now();
  iree_host_size_t completed_count = 0;

  iree_async_timer_operation_t* timer = NULL;
  while ((timer = iree_async_iocp_timer_list_pop_expired(&proactor->timers,
                                                         now)) != NULL) {
    // Timer has been removed from list. Check cancelled flag — may have been
    // set by cancel() from another thread.
    iree_status_t status = iree_ok_status();
    if (iree_any_bit_set(iree_async_operation_load_internal_flags(&timer->base),
                         IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
      status = iree_status_from_code(IREE_STATUS_CANCELLED);
      // cancel() incremented the timer cancellation counter. Decrement it
      // since we're handling the timer via expiration rather than the
      // cancellation scan.
      iree_atomic_fetch_sub(&proactor->pending_timer_cancellation_count, 1,
                            iree_memory_order_release);
    }

    iree_async_proactor_iocp_dispatch_completion(
        proactor, &timer->base, status, IREE_ASYNC_COMPLETION_FLAG_NONE,
        &completed_count);
  }

  return completed_count;
}

//===----------------------------------------------------------------------===//
// Poll: drain pending semaphore waits
//===----------------------------------------------------------------------===//

// Drains pending semaphore wait completions and invokes callbacks.
// Semaphore timepoint callbacks fire from arbitrary threads and push trackers
// to the pending_semaphore_waits MPSC slist; this function processes those
// trackers on the poll thread.
static iree_host_size_t iree_async_proactor_iocp_drain_pending_semaphore_waits(
    iree_async_proactor_iocp_t* proactor) {
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_semaphore_waits,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;
  }

  iree_host_size_t drained_count = 0;
  iree_atomic_slist_entry_t* entry = head;
  while (entry != NULL) {
    iree_async_iocp_semaphore_wait_tracker_t* tracker =
        (iree_async_iocp_semaphore_wait_tracker_t*)entry;
    iree_atomic_slist_entry_t* next = entry->next;

    iree_status_t status = (iree_status_t)iree_atomic_load(
        &tracker->completion_status, iree_memory_order_acquire);

    // For ANY mode: record the satisfied index in the operation.
    iree_async_semaphore_wait_operation_t* wait_op = tracker->operation;
    if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ANY &&
        iree_status_is_ok(status)) {
      int32_t satisfied = iree_atomic_load(&tracker->remaining_or_satisfied,
                                           iree_memory_order_acquire);
      if (satisfied >= 0) {
        wait_op->satisfied_index = (iree_host_size_t)satisfied;
      }
    }

    // Cancel remaining timepoints. In ANY mode only one was satisfied; on
    // failure some may still be registered. cancel_timepoint is a no-op for
    // timepoints that already fired.
    for (iree_host_size_t i = 0; i < tracker->count; ++i) {
      iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                            &tracker->timepoints[i]);
    }

    // Dispatch LINKED continuation chain (if any) before invoking callback.
    if (tracker->continuation_head) {
      iree_async_operation_t* continuation = tracker->continuation_head;
      tracker->continuation_head = NULL;
      if (iree_status_is_ok(status)) {
        iree_async_proactor_iocp_submit_continuation_chain(proactor,
                                                           continuation);
      } else {
        drained_count +=
            iree_async_proactor_iocp_cancel_continuation_chain(continuation);
      }
    }

    // Invoke the operation's callback.
    if (wait_op->base.completion_fn) {
      wait_op->base.completion_fn(wait_op->base.user_data,
                                  (iree_async_operation_t*)wait_op, status,
                                  IREE_ASYNC_COMPLETION_FLAG_NONE);
      ++drained_count;
    } else {
      iree_status_ignore(status);
    }

    // Clear tracker reference and free.
    wait_op->base.next = NULL;
    iree_allocator_free(tracker->allocator, tracker);

    entry = next;
  }

  return drained_count;
}

//===----------------------------------------------------------------------===//
// Poll: drain incoming messages
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_drain_incoming_messages(
    iree_async_proactor_iocp_t* proactor) {
  iree_async_message_pool_entry_t* entry =
      iree_async_message_pool_flush(&proactor->message_pool);
  while (entry) {
    iree_async_message_pool_entry_t* next =
        iree_async_message_pool_entry_next(entry);
    if (proactor->message_callback.fn) {
      proactor->message_callback.fn(&proactor->base, entry->message_data,
                                    proactor->message_callback.user_data);
    }
    iree_async_message_pool_release(&proactor->message_pool, entry);
    entry = next;
  }
}

//===----------------------------------------------------------------------===//
// Poll: process pending notification waits and relays
//===----------------------------------------------------------------------===//

// Walks a notification's relay_list, firing sinks for relays whose wait_epoch
// has been surpassed by the current epoch. One-shot relays are cleaned up;
// persistent relays update their wait_epoch for the next dispatch cycle.
static void iree_async_proactor_iocp_dispatch_notification_relays(
    iree_async_proactor_iocp_t* proactor,
    iree_async_notification_t* notification, uint32_t current_epoch) {
  iree_async_relay_t** previous = &notification->platform.iocp.relay_list;
  iree_async_relay_t* relay = notification->platform.iocp.relay_list;
  while (relay) {
    iree_async_relay_t* next = relay->platform.iocp.notification_relay_next;

    if (current_epoch == relay->wait_epoch) {
      // No epoch advancement — skip this relay.
      previous = &relay->platform.iocp.notification_relay_next;
      relay = next;
      continue;
    }

    // Epoch advanced — fire the sink.
    if (!iree_async_proactor_iocp_relay_fire_sink(relay)) {
      // Sink fire failed. Unlink from notification relay list before calling
      // fault handler (which may re-enter unregister).
      *previous = next;
      relay->platform.iocp.notification_relay_next = NULL;
      iree_async_proactor_iocp_relay_unlink(proactor, relay);
      iree_async_proactor_iocp_relay_fault(
          proactor, relay,
          iree_make_status(IREE_STATUS_INTERNAL,
                           "relay sink fire failed (GetLastError=%lu)",
                           (unsigned long)GetLastError()));
      iree_async_proactor_iocp_relay_release_resources(relay);
      relay = next;
      continue;
    }

    bool is_persistent = (relay->flags & IREE_ASYNC_RELAY_FLAG_PERSISTENT) != 0;
    if (is_persistent) {
      // Update wait_epoch for next dispatch cycle.
      relay->wait_epoch = current_epoch;
      previous = &relay->platform.iocp.notification_relay_next;
    } else {
      // One-shot: remove from notification relay list and clean up.
      *previous = next;
      relay->platform.iocp.notification_relay_next = NULL;
      iree_async_proactor_iocp_relay_unlink(proactor, relay);
      iree_async_proactor_iocp_relay_release_resources(relay);
    }

    relay = next;
  }
}

// Walks notifications with pending async waits or relays and dispatches
// satisfied waits and fires relay sinks when the epoch advances.
static iree_host_size_t
iree_async_proactor_iocp_process_pending_notification_waits(
    iree_async_proactor_iocp_t* proactor) {
  iree_host_size_t completed_count = 0;
  iree_async_notification_t** prev_ptr = &proactor->notifications_with_waits;
  iree_async_notification_t* notification = *prev_ptr;

  while (notification) {
    iree_async_notification_t* next_notification =
        notification->platform.iocp.next_with_waits;
    uint32_t current_epoch = (uint32_t)iree_atomic_load(
        &notification->epoch, iree_memory_order_acquire);

    // Walk pending_waits list, dispatching satisfied or cancelled waits.
    iree_async_notification_wait_operation_t** wait_prev =
        &notification->platform.iocp.pending_waits;
    iree_async_notification_wait_operation_t* wait = *wait_prev;
    while (wait) {
      iree_async_notification_wait_operation_t* next_wait =
          (iree_async_notification_wait_operation_t*)wait->base.next;

      bool satisfied = (current_epoch != wait->wait_token);
      bool cancelled = iree_any_bit_set(
          iree_async_operation_load_internal_flags(&wait->base),
          IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);

      if (satisfied || cancelled) {
        // Unlink from pending_waits list.
        *wait_prev = next_wait;
        wait->base.next = NULL;

        iree_status_t status =
            satisfied ? iree_ok_status()
                      : iree_status_from_code(IREE_STATUS_CANCELLED);
        iree_async_proactor_iocp_dispatch_completion(
            proactor, &wait->base, status, IREE_ASYNC_COMPLETION_FLAG_NONE,
            &completed_count);
      } else {
        wait_prev =
            (iree_async_notification_wait_operation_t**)&wait->base.next;
      }

      wait = next_wait;
    }

    // Walk relay_list, firing sinks for relays that have seen epoch advance.
    iree_async_proactor_iocp_dispatch_notification_relays(
        proactor, notification, current_epoch);

    // If notification has no more consumers, remove from tracking list.
    if (!iree_async_proactor_iocp_notification_has_consumers(notification)) {
      *prev_ptr = next_notification;
      notification->platform.iocp.next_with_waits = NULL;
      notification->platform.iocp.in_wait_list = false;
    } else {
      prev_ptr = &notification->platform.iocp.next_with_waits;
    }

    notification = next_notification;
  }

  return completed_count;
}

//===----------------------------------------------------------------------===//
// Poll: timeout calculation
//===----------------------------------------------------------------------===//

// Calculates the GQCS timeout considering pending timers.
static DWORD iree_async_proactor_iocp_calculate_timeout_ms(
    iree_async_proactor_iocp_t* proactor, iree_timeout_t user_timeout) {
  // Start with user-requested timeout.
  DWORD timeout_ms = INFINITE;
  if (iree_timeout_is_immediate(user_timeout)) {
    timeout_ms = 0;
  } else if (!iree_timeout_is_infinite(user_timeout)) {
    iree_time_t deadline_ns = iree_timeout_as_deadline_ns(user_timeout);
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns <= now_ns) {
      timeout_ms = 0;
    } else {
      int64_t remaining_ns = deadline_ns - now_ns;
      timeout_ms = (DWORD)((remaining_ns + 999999) / 1000000);
    }
  }

  // Clamp to earliest timer deadline.
  iree_time_t earliest_deadline =
      iree_async_iocp_timer_list_next_deadline_ns(&proactor->timers);
  if (earliest_deadline != IREE_TIME_INFINITE_FUTURE) {
    iree_time_t now = iree_time_now();
    if (earliest_deadline <= now) {
      timeout_ms = 0;
    } else {
      int64_t delta_ns = earliest_deadline - now;
      DWORD delta_ms = (DWORD)((delta_ns + 999999) / 1000000);
      if (delta_ms < timeout_ms) {
        timeout_ms = delta_ms;
      }
    }
  }

  return timeout_ms;
}

//===----------------------------------------------------------------------===//
// Poll
//===----------------------------------------------------------------------===//

// Maximum number of completion entries to dequeue per poll call.
#define IREE_ASYNC_IOCP_MAX_COMPLETIONS_PER_POLL 64

// Maximum number of re-drain iterations in Phase 7 of the poll loop. Prevents
// unbounded looping when completion callbacks submit inline-completing
// operations (e.g., NOPs). If the cap is reached, the proactor wakes itself to
// continue draining on the next poll iteration.
#define IREE_ASYNC_IOCP_MAX_REDRAIN_ITERATIONS 16

static iree_status_t iree_async_proactor_iocp_poll(
    iree_async_proactor_t* base_proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_host_size_t completed_count = 0;

  // Phase 1: Drain pending queue (register new operations).
  completed_count += iree_async_proactor_iocp_drain_pending_queue(proactor);

  // Phase 1.5: Drain pending semaphore wait completions. Timepoint callbacks
  // may have fired between polls, pushing trackers to the MPSC slist.
  completed_count +=
      iree_async_proactor_iocp_drain_pending_semaphore_waits(proactor);

  // Phase 1.6: Drain incoming cross-proactor messages.
  iree_async_proactor_iocp_drain_incoming_messages(proactor);

  // Phase 2: Drain timer cancellations.
  completed_count +=
      iree_async_proactor_iocp_drain_timer_cancellations(proactor);

  // Phase 2.5: Drain event wait cancellations.
  completed_count +=
      iree_async_proactor_iocp_drain_event_wait_cancellations(proactor);

  // Phase 3: Calculate effective timeout considering timer deadlines.
  DWORD timeout_ms =
      iree_async_proactor_iocp_calculate_timeout_ms(proactor, timeout);

  // Phase 4: Dequeue completions from the IOCP port.
  OVERLAPPED_ENTRY entries[IREE_ASYNC_IOCP_MAX_COMPLETIONS_PER_POLL];
  ULONG entry_count = 0;
  iree_status_t gqcs_status = iree_ok_status();
  BOOL success =
      GetQueuedCompletionStatusEx((HANDLE)proactor->completion_port, entries,
                                  IREE_ASYNC_IOCP_MAX_COMPLETIONS_PER_POLL,
                                  &entry_count, timeout_ms, FALSE);

  if (!success) {
    DWORD error = GetLastError();
    if (error != WAIT_TIMEOUT) {
      // Stash the error but continue through remaining phases so that timer
      // expirations, notification waits, and re-drains still run. The error
      // is returned after all phases complete.
      gqcs_status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "GetQueuedCompletionStatusEx failed (error %lu)",
                           (unsigned long)error);
    }
    // WAIT_TIMEOUT: no completions, fall through to timer processing.
  }

  // Phase 5: Process expired timers (before GQCS completions for consistent
  // ordering — timer callbacks fire before I/O callbacks in the same poll).
  completed_count += iree_async_proactor_iocp_process_expired_timers(proactor);

  // Phase 6: Process GQCS completions.
  for (ULONG i = 0; i < entry_count; ++i) {
    OVERLAPPED_ENTRY* entry = &entries[i];

    // Wake sentinel: NULL overlapped with CompletionKey 0.
    if (entry->lpOverlapped == NULL && entry->lpCompletionKey == 0) {
      continue;
    }

    // Signal delivery: console ctrl handler posted a sentinel completion.
    // Drain the global bitmask and dispatch to subscribers.
    if (entry->lpOverlapped == NULL &&
        entry->lpCompletionKey == IREE_ASYNC_IOCP_SIGNAL_COMPLETION_KEY) {
      iree_async_proactor_iocp_dispatch_pending_signals(proactor);
      continue;
    }

    // Direct operation completion: NULL overlapped, CompletionKey is the
    // operation pointer. Used by:
    //   - Socket/file close (synchronous, dwNumberOfBytesTransferred=0)
    //   - File open (synchronous, dwNumberOfBytesTransferred=0)
    //   - Semaphore/notification signal (synchronous, bytes=0 or stashed)
    //   - Failed overlapped I/O submit (dwNumberOfBytesTransferred encodes
    //     the Win32 error code for poll-thread delivery)
    if (entry->lpOverlapped == NULL && entry->lpCompletionKey != 0) {
      iree_async_operation_t* operation =
          (iree_async_operation_t*)entry->lpCompletionKey;
      iree_status_t direct_status = iree_ok_status();
      if (entry->dwNumberOfBytesTransferred ==
          IREE_ASYNC_IOCP_STASHED_STATUS_SENTINEL) {
        // Pre-computed iree_status_t stashed in operation->next by
        // post_stashed_status. Retrieve and clear.
        direct_status = (iree_status_t)(uintptr_t)operation->next;
        operation->next = NULL;
      } else if (entry->dwNumberOfBytesTransferred != 0) {
        // Win32 error code encoded in bytes_transferred by the submit path.
        uint32_t error_code = entry->dwNumberOfBytesTransferred;
        direct_status = iree_make_status(
            iree_status_code_from_win32_error(error_code),
            "overlapped I/O failed (Win32 error %u)", error_code);
      }
      iree_async_proactor_iocp_dispatch_completion(
          proactor, operation, direct_status, IREE_ASYNC_COMPLETION_FLAG_NONE,
          &completed_count);
      continue;
    }

    // Carrier-based completion: non-NULL overlapped.
    if (entry->lpOverlapped != NULL) {
      iree_async_iocp_carrier_t* carrier = CONTAINING_RECORD(
          entry->lpOverlapped, iree_async_iocp_carrier_t, overlapped);
      iree_async_operation_t* operation = carrier->operation;

      switch (carrier->type) {
        case IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT: {
          // Unlink carrier from active list.
          iree_async_iocp_carrier_t** prev_ptr = &proactor->active_carriers;
          while (*prev_ptr && *prev_ptr != carrier) {
            prev_ptr = &(*prev_ptr)->next;
          }
          if (*prev_ptr == carrier) {
            *prev_ptr = carrier->next;
          }
          operation->next = NULL;
          iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                iree_memory_order_relaxed);
          iree_allocator_free(proactor->base.allocator, carrier);

          // Check CANCELLED flag — cancel() may have been called between the
          // RegisterWaitForSingleObject callback posting to IOCP and this
          // dispatch. The drain function decremented the cancellation counter
          // for this case, so we don't need to touch it here.
          iree_status_t status = iree_ok_status();
          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
            status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation, status, IREE_ASYNC_COMPLETION_FLAG_NONE,
              &completed_count);
          break;
        }

        case IREE_ASYNC_IOCP_CARRIER_SOCKET_IO: {
          // General socket I/O: WSARecv, WSASend, WSASendTo, WSARecvFrom.
          DWORD bytes_transferred = 0;
          DWORD flags = 0;
          iree_status_t io_status = iree_ok_status();

          BOOL overlapped_ok = WSAGetOverlappedResult(
              (SOCKET)carrier->io_handle, &carrier->overlapped,
              &bytes_transferred, FALSE, &flags);
          if (!overlapped_ok) {
            int wsa_error = WSAGetLastError();
            if (wsa_error == WSAENOTSOCK) {
              // Socket was closed while this I/O was pending. The kernel
              // posted ERROR_OPERATION_ABORTED but the handle is now invalid
              // so WSAGetOverlappedResult can't determine the provider.
              io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
            } else {
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "socket I/O failed (WSA error %d)", wsa_error);
            }
          }

          // Check for cancellation (may race with natural completion).
          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
            iree_status_ignore(io_status);
            io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }

          // Write results to the operation.
          switch (operation->type) {
            case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
              iree_async_socket_recv_operation_t* recv_op =
                  (iree_async_socket_recv_operation_t*)operation;
              recv_op->bytes_received = bytes_transferred;
              break;
            }
            case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
              iree_async_socket_send_operation_t* send_op =
                  (iree_async_socket_send_operation_t*)operation;
              send_op->bytes_sent = bytes_transferred;
              break;
            }
            case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
              iree_async_socket_sendto_operation_t* sendto_op =
                  (iree_async_socket_sendto_operation_t*)operation;
              sendto_op->bytes_sent = bytes_transferred;
              break;
            }
            case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
              iree_async_socket_recvfrom_operation_t* recvfrom_op =
                  (iree_async_socket_recvfrom_operation_t*)operation;
              recvfrom_op->bytes_received = bytes_transferred;
              recvfrom_op->sender.length =
                  (iree_host_size_t)
                      carrier->data.socket_io.sender_address_length;
              break;
            }
            default:
              break;
          }

          // Multishot re-arm for recv operations.
          bool multishot_rearm = false;
          if (iree_status_is_ok(io_status) && bytes_transferred > 0 &&
              iree_any_bit_set(operation->flags,
                               IREE_ASYNC_OPERATION_FLAG_MULTISHOT) &&
              (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV ||
               operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM)) {
            // Dispatch with MORE flag, then re-arm.
            iree_async_proactor_iocp_dispatch_completion(
                proactor, operation, iree_ok_status(),
                IREE_ASYNC_COMPLETION_FLAG_MORE, &completed_count);

            // Zero OVERLAPPED and re-issue WSARecv/WSARecvFrom.
            memset(&carrier->overlapped, 0, sizeof(carrier->overlapped));
            carrier->data.socket_io.flags = 0;
            int rearm_result = SOCKET_ERROR;
            if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV) {
              rearm_result = WSARecv(
                  (SOCKET)carrier->io_handle, carrier->data.socket_io.wsabuf,
                  carrier->data.socket_io.buffer_count, NULL,
                  &carrier->data.socket_io.flags, &carrier->overlapped, NULL);
            } else {
              iree_async_socket_recvfrom_operation_t* rf =
                  (iree_async_socket_recvfrom_operation_t*)operation;
              carrier->data.socket_io.sender_address_length =
                  (int)sizeof(rf->sender.storage);
              rearm_result = WSARecvFrom(
                  (SOCKET)carrier->io_handle, carrier->data.socket_io.wsabuf,
                  carrier->data.socket_io.buffer_count, NULL,
                  &carrier->data.socket_io.flags,
                  (struct sockaddr*)rf->sender.storage,
                  &carrier->data.socket_io.sender_address_length,
                  &carrier->overlapped, NULL);
            }

            int rearm_error =
                (rearm_result == SOCKET_ERROR) ? WSAGetLastError() : 0;
            if (rearm_result != SOCKET_ERROR || rearm_error == WSA_IO_PENDING) {
              multishot_rearm = true;
            } else {
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(rearm_error),
                  "multishot recv re-arm failed (WSA error %d)", rearm_error);
            }
          }

          if (!multishot_rearm) {
            operation->next = NULL;
            iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                  iree_memory_order_relaxed);
            iree_allocator_free(proactor->base.allocator, carrier);
            iree_async_proactor_iocp_dispatch_completion(
                proactor, operation, io_status, IREE_ASYNC_COMPLETION_FLAG_NONE,
                &completed_count);
          }
          break;
        }

        case IREE_ASYNC_IOCP_CARRIER_ACCEPT: {
          DWORD bytes_transferred = 0;
          DWORD flags = 0;
          iree_status_t io_status = iree_ok_status();

          SOCKET listen_sock = (SOCKET)carrier->io_handle;
          BOOL overlapped_ok =
              WSAGetOverlappedResult(listen_sock, &carrier->overlapped,
                                     &bytes_transferred, FALSE, &flags);
          if (!overlapped_ok) {
            int wsa_error = WSAGetLastError();
            if (wsa_error == WSAENOTSOCK) {
              io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
            } else {
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "AcceptEx completion failed (WSA error %d)", wsa_error);
            }
          }

          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
            iree_status_ignore(io_status);
            io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }

          iree_async_socket_accept_operation_t* accept_op =
              (iree_async_socket_accept_operation_t*)operation;

          if (iree_status_is_ok(io_status)) {
            SOCKET accept_sock = carrier->data.accept.accept_socket;

            // SO_UPDATE_ACCEPT_CONTEXT transfers the listen socket's properties
            // to the accepted socket. Required after AcceptEx for getpeername,
            // getsockname, shutdown, and inherited socket options to work.
            if (setsockopt(accept_sock, SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT,
                           (char*)&listen_sock,
                           sizeof(listen_sock)) == SOCKET_ERROR) {
              int wsa_error = WSAGetLastError();
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "SO_UPDATE_ACCEPT_CONTEXT failed (WSA error %d)", wsa_error);
              closesocket(accept_sock);
            }

            if (iree_status_is_ok(io_status)) {
              // Extract peer address from AcceptEx output buffer.
              struct sockaddr* local_addr = NULL;
              struct sockaddr* remote_addr = NULL;
              int local_addr_length = 0;
              int remote_addr_length = 0;
              proactor->wsa_extensions.GetAcceptExSockaddrs(
                  carrier->data.accept.address_buffer, 0,
                  carrier->data.accept.local_address_length,
                  carrier->data.accept.remote_address_length, &local_addr,
                  &local_addr_length, &remote_addr, &remote_addr_length);

              if (remote_addr && remote_addr_length > 0) {
                memcpy(accept_op->peer_address.storage, remote_addr,
                       (size_t)remote_addr_length);
                accept_op->peer_address.length =
                    (iree_host_size_t)remote_addr_length;
              }

              // Create an iree_async_socket_t for the accepted connection.
              iree_async_socket_t* accepted_socket = NULL;
              iree_status_t create_status = iree_allocator_malloc(
                  proactor->base.allocator, sizeof(*accepted_socket),
                  (void**)&accepted_socket);
              if (iree_status_is_ok(create_status)) {
                memset(accepted_socket, 0, sizeof(*accepted_socket));
                iree_atomic_ref_count_init(&accepted_socket->ref_count);
                accepted_socket->proactor = &proactor->base;
                accepted_socket->primitive =
                    iree_async_primitive_from_win32_handle(
                        (uintptr_t)accept_sock);
                accepted_socket->fixed_file_index = -1;
                accepted_socket->type = accept_op->listen_socket->type;
                accepted_socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
                accepted_socket->flags = accept_op->listen_socket->flags;
                iree_atomic_store(&accepted_socket->failure_status,
                                  (intptr_t)iree_ok_status(),
                                  iree_memory_order_release);
                IREE_TRACE({
                  snprintf(accepted_socket->debug_label,
                           sizeof(accepted_socket->debug_label),
                           "accepted:%llu", (unsigned long long)accept_sock);
                });
                accept_op->accepted_socket = accepted_socket;
              } else {
                closesocket(accept_sock);
                io_status = create_status;
              }
            }
          } else {
            // AcceptEx failed: close the pre-created accept socket.
            closesocket(carrier->data.accept.accept_socket);
          }

          // Multishot accept re-arm.
          bool multishot_rearm = false;
          if (iree_status_is_ok(io_status) &&
              iree_any_bit_set(operation->flags,
                               IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
            // Dispatch with MORE flag.
            iree_async_proactor_iocp_dispatch_completion(
                proactor, operation, iree_ok_status(),
                IREE_ASYNC_COMPLETION_FLAG_MORE, &completed_count);

            // Create a new accept socket for the next accept.
            int domain = AF_INET;
            int protocol = IPPROTO_TCP;
            switch (accept_op->listen_socket->type) {
              case IREE_ASYNC_SOCKET_TYPE_TCP6:
                domain = AF_INET6;
                break;
              case IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM:
                domain = AF_UNIX;
                protocol = 0;
                break;
              default:
                break;
            }

            SOCKET new_accept_sock = WSASocketW(domain, SOCK_STREAM, protocol,
                                                NULL, 0, WSA_FLAG_OVERLAPPED);
            if (new_accept_sock == INVALID_SOCKET) {
              int wsa_error = WSAGetLastError();
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "multishot accept re-arm: WSASocketW failed (WSA error %d)",
                  wsa_error);
            } else {
              HANDLE assoc_result = CreateIoCompletionPort(
                  (HANDLE)new_accept_sock, (HANDLE)proactor->completion_port, 0,
                  0);
              if (assoc_result == NULL) {
                DWORD error = GetLastError();
                closesocket(new_accept_sock);
                io_status = iree_make_status(
                    iree_status_code_from_win32_error(error),
                    "multishot accept re-arm: IOCP association failed "
                    "(error %lu)",
                    (unsigned long)error);
              } else {
                // Zero OVERLAPPED and re-arm AcceptEx.
                memset(&carrier->overlapped, 0, sizeof(carrier->overlapped));
                carrier->data.accept.accept_socket = new_accept_sock;

                DWORD rearm_bytes = 0;
                BOOL rearm_ok = proactor->wsa_extensions.AcceptEx(
                    listen_sock, new_accept_sock,
                    carrier->data.accept.address_buffer, 0,
                    carrier->data.accept.local_address_length,
                    carrier->data.accept.remote_address_length, &rearm_bytes,
                    &carrier->overlapped);
                int rearm_error = rearm_ok ? 0 : WSAGetLastError();
                if (rearm_ok || rearm_error == WSA_IO_PENDING) {
                  multishot_rearm = true;
                } else {
                  closesocket(new_accept_sock);
                  io_status = iree_make_status(
                      iree_status_code_from_win32_error(rearm_error),
                      "multishot accept re-arm: AcceptEx failed "
                      "(WSA error %d)",
                      rearm_error);
                }
              }
            }
          }

          if (!multishot_rearm) {
            operation->next = NULL;
            iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                  iree_memory_order_relaxed);
            iree_allocator_free(proactor->base.allocator, carrier);
            iree_async_proactor_iocp_dispatch_completion(
                proactor, operation, io_status, IREE_ASYNC_COMPLETION_FLAG_NONE,
                &completed_count);
          }
          break;
        }

        case IREE_ASYNC_IOCP_CARRIER_CONNECT: {
          DWORD bytes_transferred = 0;
          DWORD flags = 0;
          iree_status_t io_status = iree_ok_status();

          BOOL overlapped_ok = WSAGetOverlappedResult(
              (SOCKET)carrier->io_handle, &carrier->overlapped,
              &bytes_transferred, FALSE, &flags);
          if (!overlapped_ok) {
            int wsa_error = WSAGetLastError();
            if (wsa_error == WSAENOTSOCK) {
              io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
            } else {
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "ConnectEx completion failed (WSA error %d)", wsa_error);
            }
          }

          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
            iree_status_ignore(io_status);
            io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }

          if (iree_status_is_ok(io_status)) {
            // SO_UPDATE_CONNECT_CONTEXT transfers connection state to the
            // socket. Required after ConnectEx for getpeername, getsockname,
            // shutdown, and TransmitFile to work correctly.
            if (setsockopt((SOCKET)carrier->io_handle, SOL_SOCKET,
                           SO_UPDATE_CONNECT_CONTEXT, NULL,
                           0) == SOCKET_ERROR) {
              int wsa_error = WSAGetLastError();
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "SO_UPDATE_CONNECT_CONTEXT failed (WSA error %d)", wsa_error);
            } else {
              iree_async_socket_connect_operation_t* connect_op =
                  (iree_async_socket_connect_operation_t*)operation;
              connect_op->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
            }
          }

          operation->next = NULL;
          iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                iree_memory_order_relaxed);
          iree_allocator_free(proactor->base.allocator, carrier);
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation, io_status, IREE_ASYNC_COMPLETION_FLAG_NONE,
              &completed_count);
          break;
        }

        case IREE_ASYNC_IOCP_CARRIER_RECV_POOL: {
          DWORD bytes_transferred = 0;
          DWORD flags = 0;
          iree_status_t io_status = iree_ok_status();

          BOOL overlapped_ok = WSAGetOverlappedResult(
              (SOCKET)carrier->io_handle, &carrier->overlapped,
              &bytes_transferred, FALSE, &flags);
          if (!overlapped_ok) {
            int wsa_error = WSAGetLastError();
            if (wsa_error == WSAENOTSOCK) {
              io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
            } else {
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(wsa_error),
                  "WSARecv (pool) completion failed (WSA error %d)", wsa_error);
            }
          }

          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
            iree_status_ignore(io_status);
            io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }

          iree_async_socket_recv_pool_operation_t* recv_pool_op =
              (iree_async_socket_recv_pool_operation_t*)operation;
          recv_pool_op->bytes_received = bytes_transferred;

          // Multishot recv_pool re-arm.
          bool multishot_rearm = false;
          if (iree_status_is_ok(io_status) && bytes_transferred > 0 &&
              iree_any_bit_set(operation->flags,
                               IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
            // Dispatch with MORE flag.
            iree_async_proactor_iocp_dispatch_completion(
                proactor, operation, iree_ok_status(),
                IREE_ASYNC_COMPLETION_FLAG_MORE, &completed_count);

            // Acquire new buffer from pool for next receive.
            iree_status_t acquire_status = iree_async_buffer_pool_acquire(
                recv_pool_op->pool, &recv_pool_op->lease);
            if (iree_status_is_ok(acquire_status)) {
              carrier->data.recv_pool.wsabuf.buf =
                  (char*)iree_async_span_ptr(recv_pool_op->lease.span);
              carrier->data.recv_pool.wsabuf.len =
                  (ULONG)recv_pool_op->lease.span.length;
              carrier->data.recv_pool.flags = 0;
              memset(&carrier->overlapped, 0, sizeof(carrier->overlapped));

              int rearm_result = WSARecv((SOCKET)carrier->io_handle,
                                         &carrier->data.recv_pool.wsabuf, 1,
                                         NULL, &carrier->data.recv_pool.flags,
                                         &carrier->overlapped, NULL);
              int rearm_error =
                  (rearm_result == SOCKET_ERROR) ? WSAGetLastError() : 0;
              if (rearm_result != SOCKET_ERROR ||
                  rearm_error == WSA_IO_PENDING) {
                multishot_rearm = true;
              } else {
                iree_async_buffer_lease_release(&recv_pool_op->lease);
                io_status = iree_make_status(
                    iree_status_code_from_win32_error(rearm_error),
                    "multishot recv_pool re-arm failed (WSA error %d)",
                    rearm_error);
              }
            } else {
              io_status = acquire_status;
            }
          }

          if (!multishot_rearm) {
            operation->next = NULL;
            iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                  iree_memory_order_relaxed);
            iree_allocator_free(proactor->base.allocator, carrier);
            iree_async_proactor_iocp_dispatch_completion(
                proactor, operation, io_status, IREE_ASYNC_COMPLETION_FLAG_NONE,
                &completed_count);
          }
          break;
        }

        case IREE_ASYNC_IOCP_CARRIER_FILE_IO: {
          // File I/O: ReadFile/WriteFile completions.
          DWORD bytes_transferred = 0;
          iree_status_t io_status = iree_ok_status();

          BOOL overlapped_ok = GetOverlappedResult((HANDLE)carrier->io_handle,
                                                   &carrier->overlapped,
                                                   &bytes_transferred, FALSE);
          if (!overlapped_ok) {
            DWORD error = GetLastError();
            if (error == ERROR_HANDLE_EOF) {
              // Reading at or past EOF: not an error, just 0 bytes transferred.
              // Matches POSIX read() returning 0 at EOF.
              bytes_transferred = 0;
            } else if (error == ERROR_INVALID_HANDLE) {
              // File was closed while this I/O was pending.
              io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
            } else {
              io_status = iree_make_status(
                  iree_status_code_from_win32_error(error),
                  "file I/O failed (error %lu)", (unsigned long)error);
            }
          }

          if (iree_any_bit_set(
                  iree_async_operation_load_internal_flags(operation),
                  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
            iree_status_ignore(io_status);
            io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
          }

          // Write results to the operation.
          switch (operation->type) {
            case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
              iree_async_file_read_operation_t* read_op =
                  (iree_async_file_read_operation_t*)operation;
              read_op->bytes_read = bytes_transferred;
              break;
            }
            case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
              iree_async_file_write_operation_t* write_op =
                  (iree_async_file_write_operation_t*)operation;
              write_op->bytes_written = bytes_transferred;
              break;
            }
            default:
              break;
          }

          operation->next = NULL;
          iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                iree_memory_order_relaxed);
          iree_allocator_free(proactor->base.allocator, carrier);
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation, io_status, IREE_ASYNC_COMPLETION_FLAG_NONE,
              &completed_count);
          break;
        }

        default: {
          // Unknown carrier type. Free and dispatch with error.
          operation->next = NULL;
          iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                                iree_memory_order_relaxed);
          iree_allocator_free(proactor->base.allocator, carrier);
          iree_async_proactor_iocp_dispatch_completion(
              proactor, operation,
              iree_make_status(IREE_STATUS_INTERNAL,
                               "unknown IOCP carrier type %d",
                               (int)carrier->type),
              IREE_ASYNC_COMPLETION_FLAG_NONE, &completed_count);
          break;
        }
      }
      continue;
    }
  }

  // Phase 7: Re-drain pending queue until stable. Callbacks dispatched during
  // Phase 6 (and during this re-drain) may submit new operations. NOPs
  // complete inline during drain, so each iteration may produce more pending
  // operations (e.g., sequence emulation step advancement). Capped to prevent
  // unbounded looping from pathological callbacks; if the cap is reached the
  // proactor is woken to continue draining on the next poll iteration.
  {
    iree_host_size_t redrain_completions = 0;
    int redrain_iterations = 0;
    do {
      redrain_completions =
          iree_async_proactor_iocp_drain_pending_queue(proactor);
      completed_count += redrain_completions;
    } while (redrain_completions > 0 &&
             ++redrain_iterations < IREE_ASYNC_IOCP_MAX_REDRAIN_ITERATIONS);
    if (redrain_completions > 0) {
      iree_async_proactor_iocp_wake(&proactor->base);
    }
  }

  // Phase 8: Process pending notification waits. Check epoch advancement
  // for async waits registered during Phase 1 or earlier.
  completed_count +=
      iree_async_proactor_iocp_process_pending_notification_waits(proactor);

  // Phase 9: Re-drain semaphore waits. Timepoint callbacks may have fired
  // during GQCS completion processing.
  completed_count +=
      iree_async_proactor_iocp_drain_pending_semaphore_waits(proactor);

  // Phase 10: Re-drain incoming messages (messages sent during GQCS
  // processing).
  iree_async_proactor_iocp_drain_incoming_messages(proactor);

  // Phase 11: Final timer expiry check (timers submitted during Phase 6-10
  // callbacks might have already expired).
  completed_count += iree_async_proactor_iocp_process_expired_timers(proactor);

  if (out_completed_count) *out_completed_count = completed_count;
  IREE_TRACE_ZONE_END(z0);
  if (!iree_status_is_ok(gqcs_status)) return gqcs_status;
  return completed_count > 0
             ? iree_ok_status()
             : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}

//===----------------------------------------------------------------------===//
// Cancel
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_cancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_TIMER: {
      // Set the cancelled flag and increment the timer cancellation counter.
      // The poll thread scans the timer_list when the counter is non-zero,
      // removing cancelled timers and pushing CANCELLED completions.
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);
      iree_atomic_fetch_add(&proactor->pending_timer_cancellation_count, 1,
                            iree_memory_order_release);
      iree_async_proactor_iocp_wake(&proactor->base);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
      return iree_async_sequence_cancel(
          base_proactor, (iree_async_sequence_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);
      // CancelIoEx posts ERROR_OPERATION_ABORTED to the IOCP port for the
      // specific overlapped operation. The carrier address stored in
      // operation->next equals the OVERLAPPED address (OVERLAPPED is at
      // offset 0 in the carrier struct). We use it as the lpOverlapped
      // parameter — CancelIoEx uses this as a kernel-internal lookup key to
      // match against pending I/O requests, not as a user-space dereference.
      // The handle comes from the operation's socket struct (ref-counted and
      // retained, safe from any thread) rather than from the carrier, which
      // the poll thread may free concurrently after dispatching the
      // completion.
      LPOVERLAPPED overlapped = (LPOVERLAPPED)operation->next;
      if (overlapped != NULL) {
        HANDLE handle =
            iree_async_proactor_iocp_handle_from_io_operation(operation);
        CancelIoEx(handle, overlapped);
      }
      iree_async_proactor_iocp_wake(&proactor->base);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      // Event waits use RegisterWaitForSingleObject. Set the cancelled flag
      // and increment the cancellation counter so the poll thread scans
      // active_carriers to unregister the wait and dispatch CANCELLED.
      // Without the active scan, if the event is never signaled, the
      // RegisterWaitForSingleObject callback never fires and the operation
      // callback is never invoked — violating the "callback always fires"
      // invariant.
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);
      iree_atomic_fetch_add(&proactor->pending_event_wait_cancellation_count, 1,
                            iree_memory_order_release);
      iree_async_proactor_iocp_wake(&proactor->base);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);
      iree_async_proactor_iocp_wake(&proactor->base);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT: {
      iree_async_semaphore_wait_operation_t* wait_op =
          (iree_async_semaphore_wait_operation_t*)operation;
      iree_async_iocp_semaphore_wait_tracker_t* tracker =
          (iree_async_iocp_semaphore_wait_tracker_t*)wait_op->base.next;
      if (!tracker) {
        // No tracker: operation was never submitted or already completed.
        return iree_ok_status();
      }
      // Try to set CANCELLED status (first status wins via CAS). If another
      // callback already set a status, the tracker will be (or has been)
      // enqueued for completion by that callback.
      intptr_t expected = (intptr_t)iree_ok_status();
      iree_status_t cancelled_status =
          iree_make_status(IREE_STATUS_CANCELLED, "operation cancelled");
      if (!iree_atomic_compare_exchange_strong(
              &tracker->completion_status, &expected,
              (intptr_t)cancelled_status, iree_memory_order_acq_rel,
              iree_memory_order_acquire)) {
        iree_status_ignore(cancelled_status);
        return iree_ok_status();
      }
      // Cancel all registered timepoints. cancel_timepoint guarantees the
      // callback will not fire after it returns, so no double-enqueue risk.
      for (iree_host_size_t i = 0; i < tracker->count; ++i) {
        iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                              &tracker->timepoints[i]);
      }
      // Enqueue the tracker for completion on the poll thread.
      iree_async_proactor_iocp_semaphore_wait_enqueue_completion(tracker);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL:
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL:
    case IREE_ASYNC_OPERATION_TYPE_MESSAGE:
      // These complete synchronously during submit — nothing to cancel.
      return iree_ok_status();

    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);
      // Same pattern as socket I/O cancel: carrier address == OVERLAPPED
      // address (offset 0), handle from the operation's file struct.
      LPOVERLAPPED overlapped = (LPOVERLAPPED)operation->next;
      if (overlapped != NULL) {
        HANDLE handle =
            iree_async_proactor_iocp_handle_from_io_operation(operation);
        CancelIoEx(handle, overlapped);
      }
      iree_async_proactor_iocp_wake(&proactor->base);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      // These complete synchronously via PostQueuedCompletionStatus —
      // nothing to cancel.
      return iree_ok_status();

    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "IOCP proactor: cancel not implemented for operation type %d",
          (int)operation->type);
  }
}

//===----------------------------------------------------------------------===//
// Socket management
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_create_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  return iree_async_iocp_socket_create(proactor, type, options, out_socket);
}

static iree_status_t iree_async_proactor_iocp_import_socket(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  return iree_async_iocp_socket_import(proactor, primitive, type, flags,
                                       out_socket);
}

static void iree_async_proactor_iocp_destroy_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_t* socket) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_async_iocp_socket_destroy(proactor, socket);
}

//===----------------------------------------------------------------------===//
// File management
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_import_file(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Validate the handle.
  uintptr_t handle = primitive.value.win32_handle;
  if (handle == 0 || handle == (uintptr_t)INVALID_HANDLE_VALUE) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid file handle");
  }

  // Associate the file handle with the IOCP port. This is required for
  // overlapped ReadFile/WriteFile completions to flow to this port.
  HANDLE result = CreateIoCompletionPort(
      (HANDLE)handle, (HANDLE)proactor->completion_port, /*CompletionKey=*/0,
      /*NumberOfConcurrentThreads=*/0);
  if (result == NULL) {
    DWORD error = GetLastError();
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "CreateIoCompletionPort failed for file handle "
                            "(error %lu)",
                            (unsigned long)error);
  }

  // Allocate and initialize the file structure.
  iree_async_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*file), (void**)&file));
  memset(file, 0, sizeof(*file));

  iree_atomic_ref_count_init(&file->ref_count);
  file->proactor = base_proactor;
  file->primitive = primitive;
  file->fixed_file_index = -1;

  IREE_TRACE({
    snprintf(file->debug_path, sizeof(file->debug_path), "handle:%p",
             (void*)handle);
  });

  *out_file = file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_iocp_destroy_file(
    iree_async_proactor_t* base_proactor, iree_async_file_t* file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(file);

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Close the file handle. It may already be closed by a FILE_CLOSE operation,
  // in which case the handle was set to 0.
  uintptr_t handle = file->primitive.value.win32_handle;
  if (handle != 0 && handle != (uintptr_t)INVALID_HANDLE_VALUE) {
    CloseHandle((HANDLE)handle);
  }

  iree_allocator_free(allocator, file);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Event management
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_create_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t** out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Allocate and zero-initialize the event structure.
  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*event), (void**)&event));
  memset(event, 0, sizeof(*event));

  // Create an auto-reset Win32 event, initially non-signaled.
  // Auto-reset: RegisterWaitForSingleObject consumes the signal atomically,
  // matching the POSIX eventfd drain pattern. Only one waiter is woken per
  // SetEvent call.
  HANDLE event_handle = CreateEventW(NULL, FALSE, FALSE, NULL);
  if (event_handle == NULL) {
    DWORD error = GetLastError();
    iree_allocator_free(allocator, event);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CreateEvent failed (error %lu)",
                            (unsigned long)error);
  }

  iree_atomic_ref_count_init(&event->ref_count);
  event->proactor = base_proactor;
  event->primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)event_handle);
  // On Windows, signal_primitive is the same as primitive (same HANDLE used
  // for both monitoring and signaling, like Linux eventfd).
  event->signal_primitive = event->primitive;
  event->fixed_file_index = -1;
  event->drain_buffer = 0;
  event->pool = NULL;
  event->pool_next = NULL;
  event->pool_all_next = NULL;

  *out_event = event;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_iocp_destroy_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);

  // Close the Win32 event handle. Since primitive == signal_primitive on
  // Windows, we only close once.
  if (event->primitive.value.win32_handle != 0) {
    CloseHandle((HANDLE)event->primitive.value.win32_handle);
  }

  iree_allocator_free(proactor->base.allocator, event);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Event source registration (stubs)
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_register_event_source(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "IOCP proactor: register_event_source not yet implemented");
}

static void iree_async_proactor_iocp_unregister_event_source(
    iree_async_proactor_t* base_proactor,
    iree_async_event_source_t* event_source) {
  // Void return: nothing to do until event sources are implemented.
}

//===----------------------------------------------------------------------===//
// Notification management
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_create_notification(
    iree_async_proactor_t* base_proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  iree_async_notification_t* notification = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*notification),
                                (void**)&notification));
  memset(notification, 0, sizeof(*notification));

  iree_atomic_ref_count_init(&notification->ref_count);
  notification->proactor = &proactor->base;
  iree_atomic_store(&notification->epoch, 0, iree_memory_order_release);
  // IOCP uses WaitOnAddress for sync waits (functionally identical to futex).
  // No fd/primitive needed — the epoch atomic is the wait address.
  notification->mode = IREE_ASYNC_NOTIFICATION_MODE_FUTEX;
  notification->platform.iocp.pending_waits = NULL;
  notification->platform.iocp.next_with_waits = NULL;
  notification->platform.iocp.in_wait_list = false;
  notification->platform.iocp.relay_list = NULL;

  *out_notification = notification;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_iocp_destroy_notification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification) {
  if (!notification) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // No fds to close (unlike POSIX). Just free the allocation.
  iree_allocator_free(allocator, notification);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_async_proactor_iocp_notification_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, int32_t wake_count) {
  (void)base_proactor;
  // Epoch already incremented by the shared iree_async_notification_signal()
  // in notification.c before this vtable call. Wake sync waiters blocked in
  // WaitOnAddress on the epoch value.
  if (wake_count == 1) {
    WakeByAddressSingle((void*)&notification->epoch);
  } else {
    WakeByAddressAll((void*)&notification->epoch);
  }
  // Wake the poll thread so it checks async waits in the notification's
  // pending_waits list.
  iree_async_proactor_iocp_wake(notification->proactor);
}

static bool iree_async_proactor_iocp_notification_wait(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, iree_timeout_t timeout) {
  (void)base_proactor;
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  int32_t wait_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  while (iree_time_now() < deadline_ns) {
    int32_t current_epoch =
        iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
    if (current_epoch != wait_epoch) return true;
    // Calculate remaining time for WaitOnAddress timeout.
    iree_time_t now = iree_time_now();
    if (now >= deadline_ns) break;
    int64_t remaining_ns = deadline_ns - now;
    DWORD remaining_ms = (DWORD)((remaining_ns + 999999) / 1000000);
    if (remaining_ms == 0) remaining_ms = 1;
    BOOL waited = WaitOnAddress((volatile void*)&notification->epoch,
                                &wait_epoch, sizeof(int32_t), remaining_ms);
    (void)waited;
    // WaitOnAddress returns FALSE on timeout, TRUE on wake. Either way,
    // re-check the epoch.
  }
  int32_t final_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  return final_epoch != wait_epoch;
}

//===----------------------------------------------------------------------===//
// Relay
//===----------------------------------------------------------------------===//

// Fires the relay's sink action synchronously.
// Returns true on success, false on failure.
static bool iree_async_proactor_iocp_relay_fire_sink(
    iree_async_relay_t* relay) {
  switch (relay->sink.type) {
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE: {
      HANDLE handle =
          (HANDLE)relay->sink.signal_primitive.primitive.value.win32_handle;
      return SetEvent(handle) != 0;
    }
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION: {
      iree_async_notification_signal(
          relay->sink.signal_notification.notification,
          relay->sink.signal_notification.wake_count);
      return true;
    }
  }
  return false;
}

// Unlinks a relay from its source notification's relay_list.
static void iree_async_proactor_iocp_relay_remove_from_notification_list(
    iree_async_relay_t* relay) {
  iree_async_notification_t* notification = relay->source.notification;
  iree_async_relay_t** previous = &notification->platform.iocp.relay_list;
  iree_async_relay_t* current = notification->platform.iocp.relay_list;
  while (current) {
    if (current == relay) {
      *previous = current->platform.iocp.notification_relay_next;
      current->platform.iocp.notification_relay_next = NULL;
      return;
    }
    previous = &current->platform.iocp.notification_relay_next;
    current = current->platform.iocp.notification_relay_next;
  }
}

// Returns true if the notification has any consumers (pending waits or relays).
static bool iree_async_proactor_iocp_notification_has_consumers(
    iree_async_notification_t* notification) {
  return notification->platform.iocp.pending_waits != NULL ||
         notification->platform.iocp.relay_list != NULL;
}

// Removes a notification from the proactor's notifications_with_waits list.
static void iree_async_proactor_iocp_remove_from_wait_list(
    iree_async_proactor_iocp_t* proactor,
    iree_async_notification_t* notification) {
  iree_async_notification_t** previous = &proactor->notifications_with_waits;
  iree_async_notification_t* current = *previous;
  while (current) {
    if (current == notification) {
      *previous = current->platform.iocp.next_with_waits;
      current->platform.iocp.next_with_waits = NULL;
      current->platform.iocp.in_wait_list = false;
      return;
    }
    previous = &current->platform.iocp.next_with_waits;
    current = current->platform.iocp.next_with_waits;
  }
}

// Unlinks a relay from the proactor's doubly-linked relay list.
static void iree_async_proactor_iocp_relay_unlink(
    iree_async_proactor_iocp_t* proactor, iree_async_relay_t* relay) {
  if (relay->prev) {
    relay->prev->next = relay->next;
  } else {
    proactor->relays = relay->next;
  }
  if (relay->next) {
    relay->next->prev = relay->prev;
  }
  relay->next = NULL;
  relay->prev = NULL;
}

// Releases retained notifications and frees the relay struct.
static void iree_async_proactor_iocp_relay_release_resources(
    iree_async_relay_t* relay) {
  if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    iree_async_notification_release(relay->source.notification);
  }
  if (relay->sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
    iree_async_notification_release(
        relay->sink.signal_notification.notification);
  }
  iree_allocator_free(relay->allocator, relay);
}

// Invokes the error callback (if registered) and cleans up the relay.
// Takes ownership of |status|.
static void iree_async_proactor_iocp_relay_fault(
    iree_async_proactor_iocp_t* proactor, iree_async_relay_t* relay,
    iree_status_t status) {
  if (relay->error_callback.fn) {
    relay->error_callback.fn(relay->error_callback.user_data, relay, status);
  } else {
    iree_status_ignore(status);
  }
}

static iree_status_t iree_async_proactor_iocp_register_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);

  // Validate source. IOCP only supports notification sources — there is no
  // poll-style fd monitoring for arbitrary Windows HANDLEs.
  switch (source.type) {
    case IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION:
      if (!source.notification) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "relay source notification must not be NULL");
      }
      break;
    case IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "IOCP proactor does not support primitive-source relays");
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown relay source type %d", (int)source.type);
  }

  // Validate sink.
  switch (sink.type) {
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_PRIMITIVE:
      if (sink.signal_primitive.primitive.type !=
              IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE ||
          sink.signal_primitive.primitive.value.win32_handle == 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "relay sink signal_primitive must be a valid win32 handle");
      }
      break;
    case IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION:
      if (!sink.signal_notification.notification) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "relay sink signal_notification must not be NULL");
      }
      break;
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown relay sink type %d", (int)sink.type);
  }

  // Allocate and initialize the relay struct.
  iree_async_relay_t* relay = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->base.allocator, sizeof(*relay),
                                (void**)&relay));
  relay->next = NULL;
  relay->prev = NULL;
  relay->proactor = &proactor->base;
  relay->source = source;
  relay->sink = sink;
  relay->flags = flags;
  relay->error_callback = error_callback;
  relay->wait_epoch = 0;
  relay->allocator = proactor->base.allocator;
  memset(&relay->platform, 0, sizeof(relay->platform));

  // Retain notifications used in source/sink.
  iree_async_notification_retain(source.notification);
  if (sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
    iree_async_notification_retain(sink.signal_notification.notification);
  }

  // Notification source: capture epoch, link into notification's relay_list,
  // and ensure the notification is tracked in the proactor's
  // notifications_with_waits list for poll-loop dispatch.
  iree_async_notification_t* notification = source.notification;
  relay->wait_epoch = (uint32_t)iree_atomic_load(&notification->epoch,
                                                 iree_memory_order_acquire);
  relay->platform.iocp.notification_relay_next =
      notification->platform.iocp.relay_list;
  notification->platform.iocp.relay_list = relay;

  if (!notification->platform.iocp.in_wait_list) {
    notification->platform.iocp.next_with_waits =
        proactor->notifications_with_waits;
    proactor->notifications_with_waits = notification;
    notification->platform.iocp.in_wait_list = true;
  }

  // Link into proactor's doubly-linked relay list for cleanup.
  relay->next = proactor->relays;
  if (proactor->relays) {
    proactor->relays->prev = relay;
  }
  proactor->relays = relay;

  *out_relay = relay;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_iocp_unregister_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_t* relay) {
  if (!relay) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);

  // Remove from the source notification's relay list.
  if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
    iree_async_proactor_iocp_relay_remove_from_notification_list(relay);
    iree_async_notification_t* notification = relay->source.notification;
    if (!iree_async_proactor_iocp_notification_has_consumers(notification)) {
      iree_async_proactor_iocp_remove_from_wait_list(proactor, notification);
    }
  }

  // Unlink from proactor's doubly-linked relay list.
  iree_async_proactor_iocp_relay_unlink(proactor, relay);

  // Release retained notifications and free.
  iree_async_proactor_iocp_relay_release_resources(relay);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Buffer registration (metadata-only)
//===----------------------------------------------------------------------===//

// Combined allocation for registration entry + region.
// Keeps them together in memory and simplifies cleanup.
typedef struct iree_async_iocp_buffer_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
} iree_async_iocp_buffer_registration_t;

// Destroy callback for buffer registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_iocp_buffer_registration_destroy(
    iree_async_region_t* region) {
  iree_async_iocp_buffer_registration_t* registration =
      (iree_async_iocp_buffer_registration_t*)((char*)region -
                                               offsetof(
                                                   iree_async_iocp_buffer_registration_t,
                                                   region));
  iree_allocator_free(region->proactor->allocator, registration);
}

// Cleanup function for buffer registrations.
// Called when the registration state is cleaned up or explicitly unregistered.
static void iree_async_iocp_buffer_registration_cleanup(void* entry_ptr,
                                                        void* proactor_ptr) {
  iree_async_iocp_buffer_registration_t* registration =
      (iree_async_iocp_buffer_registration_t*)entry_ptr;
  (void)proactor_ptr;
  iree_async_region_release(&registration->region);
}

static iree_status_t iree_async_proactor_iocp_register_buffer(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_entry = NULL;

  // Allocate combined entry + region.
  iree_async_iocp_buffer_registration_t* registration = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(base_proactor->allocator, sizeof(*registration),
                                (void**)&registration));
  memset(registration, 0, sizeof(*registration));

  // Initialize the region (metadata-only, no kernel registration).
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->slab = NULL;
  region->destroy_fn = iree_async_iocp_buffer_registration_destroy;
  region->type = IREE_ASYNC_REGION_TYPE_NONE;
  region->access_flags = access_flags;
  region->base_ptr = (void*)buffer.data;
  region->length = buffer.data_length;
  region->recycle = iree_async_buffer_recycle_callback_null();
  region->buffer_size = 0;
  region->buffer_count = 0;

  // Initialize the entry.
  iree_async_buffer_registration_entry_t* entry = &registration->entry;
  entry->next = NULL;
  entry->proactor = base_proactor;
  entry->cleanup_fn = iree_async_iocp_buffer_registration_cleanup;
  entry->region = region;

  // Link into the caller's state.
  iree_async_buffer_registration_state_add(state, entry);

  *out_entry = entry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_iocp_register_dmabuf(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)base_proactor;
  (void)state;
  (void)dmabuf_fd;
  (void)offset;
  (void)length;
  (void)access_flags;
  *out_entry = NULL;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "IOCP backend does not support DMA-buf registration");
}

static void iree_async_proactor_iocp_unregister_buffer(
    iree_async_proactor_t* base_proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  iree_async_buffer_registration_state_remove(state, entry);
  entry->cleanup_fn(entry, base_proactor);
}

// Wrapper for slab-backed regions. Stores the allocator for self-deallocation.
typedef struct iree_async_iocp_slab_region_t {
  iree_async_region_t region;
  iree_allocator_t allocator;
} iree_async_iocp_slab_region_t;

// Destroy callback for slab registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_iocp_slab_region_destroy(iree_async_region_t* region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_iocp_slab_region_t* slab_region =
      (iree_async_iocp_slab_region_t*)((char*)region -
                                       offsetof(iree_async_iocp_slab_region_t,
                                                region));

  // Clear singleton tracking for READ-access registrations.
  if (iree_any_bit_set(region->access_flags,
                       IREE_ASYNC_BUFFER_ACCESS_FLAG_READ)) {
    iree_async_proactor_iocp_t* proactor =
        iree_async_proactor_iocp_cast(region->proactor);
    proactor->has_read_slab_registration = false;
  }

  // Release the retained slab reference.
  iree_async_slab_release(region->slab);

  iree_allocator_free(slab_region->allocator, slab_region);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_proactor_iocp_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_region);
  *out_region = NULL;

  iree_host_size_t buffer_size = iree_async_slab_buffer_size(slab);
  iree_host_size_t buffer_count = iree_async_slab_buffer_count(slab);
  void* base_ptr = iree_async_slab_base_ptr(slab);

  // Check singleton constraint for READ access (fixed buffer table equivalent).
  // io_uring allows only one buffer table registration at a time; enforced
  // identically here for API portability.
  bool needs_read = (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ) != 0;
  if (needs_read && proactor->has_read_slab_registration) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "READ-access slab already registered with this proactor; "
        "only one READ-access slab registration is allowed at a time");
  }

  // Validate buffer count fits in region handles.
  if (buffer_count > UINT16_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz
                            " exceeds maximum %u for indexed zero-copy",
                            buffer_count, (unsigned)UINT16_MAX);
  }

  // WRITE access (recv path) requires power-of-2 buffer count.
  // io_uring requires this for PBUF_RING; enforced identically here.
  bool needs_write = (access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE) != 0;
  if (needs_write && (buffer_count & (buffer_count - 1)) != 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_count %" PRIhsz
                            " must be power of 2 for recv registrations",
                            buffer_count);
  }

  // Validate buffer size fits in region handles.
  if (buffer_size > UINT32_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size %" PRIhsz
                            " exceeds maximum %u for indexed zero-copy",
                            buffer_size, (unsigned)UINT32_MAX);
  }

  // Reject zero buffer_size — would cause divide-by-zero in index derivation.
  if (buffer_size == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_size must be > 0 for slab registration");
  }

  // Allocate the slab region struct.
  iree_async_iocp_slab_region_t* slab_region = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(base_proactor->allocator, sizeof(*slab_region),
                                (void**)&slab_region));
  memset(slab_region, 0, sizeof(*slab_region));
  slab_region->allocator = base_proactor->allocator;

  // Initialize the region.
  iree_async_region_t* region = &slab_region->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = base_proactor;
  region->slab = slab;
  iree_async_slab_retain(slab);
  region->destroy_fn = iree_async_iocp_slab_region_destroy;
  // WRITE-access regions use EMULATED type to indicate proactor-managed buffer
  // selection (pool acquire in userspace). READ-only regions need no
  // backend-specific handles since the send path uses iree_async_span_ptr().
  region->type = needs_write ? IREE_ASYNC_REGION_TYPE_EMULATED
                             : IREE_ASYNC_REGION_TYPE_NONE;
  region->access_flags = access_flags;
  region->base_ptr = base_ptr;
  region->length = iree_async_slab_total_size(slab);
  // No recycle callback — IOCP uses pool freelist for buffer recycling.
  region->recycle = iree_async_buffer_recycle_callback_null();
  region->buffer_size = buffer_size;
  region->buffer_count = (uint32_t)buffer_count;

  // Update singleton tracking.
  if (needs_read) {
    proactor->has_read_slab_registration = true;
  }

  *out_region = region;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Fence import/export (stubs)
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_iocp_import_fence(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "IOCP proactor: import_fence not yet implemented");
}

static iree_status_t iree_async_proactor_iocp_export_fence(
    iree_async_proactor_t* base_proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "IOCP proactor: export_fence not yet implemented");
}

//===----------------------------------------------------------------------===//
// Cross-proactor messaging (stubs)
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_set_message_callback(
    iree_async_proactor_t* base_proactor,
    iree_async_proactor_message_callback_t callback) {
  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);
  proactor->message_callback = callback;
}

static iree_status_t iree_async_proactor_iocp_send_message(
    iree_async_proactor_t* base_target, uint64_t message_data) {
  iree_async_proactor_iocp_t* target =
      iree_async_proactor_iocp_cast(base_target);
  iree_status_t status =
      iree_async_message_pool_send(&target->message_pool, message_data);
  if (iree_status_is_ok(status)) {
    base_target->vtable->wake(base_target);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

const iree_async_proactor_vtable_t iree_async_proactor_iocp_vtable = {
    .destroy = iree_async_proactor_iocp_destroy,
    .query_capabilities = iree_async_proactor_iocp_query_capabilities,
    .submit = iree_async_proactor_iocp_submit,
    .poll = iree_async_proactor_iocp_poll,
    .wake = iree_async_proactor_iocp_wake,
    .cancel = iree_async_proactor_iocp_cancel,
    .create_socket = iree_async_proactor_iocp_create_socket,
    .import_socket = iree_async_proactor_iocp_import_socket,
    .destroy_socket = iree_async_proactor_iocp_destroy_socket,
    .import_file = iree_async_proactor_iocp_import_file,
    .destroy_file = iree_async_proactor_iocp_destroy_file,
    .create_event = iree_async_proactor_iocp_create_event,
    .destroy_event = iree_async_proactor_iocp_destroy_event,
    .register_event_source = iree_async_proactor_iocp_register_event_source,
    .unregister_event_source = iree_async_proactor_iocp_unregister_event_source,
    .create_notification = iree_async_proactor_iocp_create_notification,
    .destroy_notification = iree_async_proactor_iocp_destroy_notification,
    .notification_signal = iree_async_proactor_iocp_notification_signal,
    .notification_wait = iree_async_proactor_iocp_notification_wait,
    .register_relay = iree_async_proactor_iocp_register_relay,
    .unregister_relay = iree_async_proactor_iocp_unregister_relay,
    .register_buffer = iree_async_proactor_iocp_register_buffer,
    .register_dmabuf = iree_async_proactor_iocp_register_dmabuf,
    .unregister_buffer = iree_async_proactor_iocp_unregister_buffer,
    .register_slab = iree_async_proactor_iocp_register_slab,
    .import_fence = iree_async_proactor_iocp_import_fence,
    .export_fence = iree_async_proactor_iocp_export_fence,
    .set_message_callback = iree_async_proactor_iocp_set_message_callback,
    .send_message = iree_async_proactor_iocp_send_message,
    .subscribe_signal = iree_async_proactor_iocp_subscribe_signal,
    .unsubscribe_signal = iree_async_proactor_iocp_unsubscribe_signal,
};

#else  // !IREE_PLATFORM_WINDOWS

iree_status_t iree_async_proactor_create_iocp(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "IOCP proactor requires Windows");
}

#endif  // IREE_PLATFORM_WINDOWS
