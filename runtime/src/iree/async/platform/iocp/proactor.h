// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Internal header for IOCP proactor implementation.
//
// This header exposes the proactor struct and internal helpers for use by
// IOCP-specific modules. External users should include api.h instead.

#ifndef IREE_ASYNC_PLATFORM_IOCP_PROACTOR_H_
#define IREE_ASYNC_PLATFORM_IOCP_PROACTOR_H_

#include "iree/async/platform/iocp/api.h"
#include "iree/async/platform/iocp/timer_list.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/async/util/signal.h"
#include "iree/base/internal/atomic_slist.h"

// Windows headers for OVERLAPPED, WSABUF, SOCKET, sockaddr_in6.
// Only needed on Windows; non-Windows builds use placeholder storage.
#if defined(IREE_PLATFORM_WINDOWS)
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <windows.h>
// clang-format on
#endif  // IREE_PLATFORM_WINDOWS

// Maximum scatter-gather buffers in a carrier's WSABUF array.
// Mirrors IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS from operations/net.h;
// verified at compile time in proactor.c.
#define IREE_ASYNC_IOCP_MAX_SCATTER_GATHER_BUFFERS 8

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_semaphore_wait_operation_t
    iree_async_semaphore_wait_operation_t;

//===----------------------------------------------------------------------===//
// Internal flags for IOCP proactor operations
//===----------------------------------------------------------------------===//

// Internal flags used by the IOCP proactor for operation state management.
// These are written to iree_async_operation_t::internal_flags during execution.
enum iree_async_iocp_operation_internal_flags_e {
  // Set when an operation is cancelled via cancel(). The poll thread checks
  // this flag during pending_queue drain and timer expiration processing.
  IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED = (1u << 0),
};

//===----------------------------------------------------------------------===//
// Carrier struct for IOCP completions
//===----------------------------------------------------------------------===//

// Identifies the kind of operation a carrier wraps. Determines which member
// of the data union is active and how the poll thread dispatches completions.
enum iree_async_iocp_carrier_type_e {
  // Event wait: RegisterWaitForSingleObject callback posts to IOCP port.
  // Data: event_wait (wait_handle for UnregisterWaitEx).
  IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT = 0,

  // General socket I/O: WSARecv, WSASend, WSASendTo, WSARecvFrom.
  // Data: socket_io (WSABUF array for scatter-gather).
  IREE_ASYNC_IOCP_CARRIER_SOCKET_IO,

  // Accept: AcceptEx with pre-created accept socket and address buffer.
  // Data: accept (accept socket handle, address output buffer).
  IREE_ASYNC_IOCP_CARRIER_ACCEPT,

  // Pool-based recv: WSARecv with a buffer acquired from a pool.
  // Data: recv_pool (single WSABUF from pool buffer).
  IREE_ASYNC_IOCP_CARRIER_RECV_POOL,

  // Connect: ConnectEx. No extra data beyond OVERLAPPED.
  IREE_ASYNC_IOCP_CARRIER_CONNECT,

  // File I/O: ReadFile/WriteFile with OVERLAPPED.
  // The file offset is encoded in overlapped.Offset/OffsetHigh.
  // No extra data needed beyond the base carrier fields.
  IREE_ASYNC_IOCP_CARRIER_FILE_IO,
};
typedef uint8_t iree_async_iocp_carrier_type_t;

// Carrier wrapping an operation for delivery through the IOCP port.
//
// For event waits: the RegisterWaitForSingleObject callback fires on an OS
// thread pool thread and posts &carrier->overlapped via
// PostQueuedCompletionStatus. The poll thread recovers the carrier via
// CONTAINING_RECORD and dispatches the operation.
//
// For socket I/O: overlapped WSA functions (WSARecv, WSASend, AcceptEx,
// ConnectEx) take &carrier->overlapped directly. Completions arrive via
// GetQueuedCompletionStatusEx. The poll thread recovers the carrier via
// CONTAINING_RECORD, checks WSAGetOverlappedResult for error details, and
// dispatches the operation.
//
// Lifecycle: allocated at submit (or poll thread for event waits), freed on
// completion dispatch (poll thread) or during proactor destroy.
typedef struct iree_async_iocp_carrier_t {
  // Overlapped struct for IOCP delivery.
  // Must be first for CONTAINING_RECORD recovery from lpOverlapped.
  // NOTE: On non-Windows builds this is a placeholder for compilation.
#if defined(IREE_PLATFORM_WINDOWS)
  OVERLAPPED overlapped;
#else
  uint8_t overlapped_placeholder[32];
#endif

  // Discriminator for the data union.
  iree_async_iocp_carrier_type_t type;

  // Completion port handle for PostQueuedCompletionStatus in callbacks.
  // Stored here because callbacks have no access to the proactor struct.
  uintptr_t completion_port;

  // The operation this carrier delivers. Set during submit/registration,
  // consumed during poll dispatch.
  iree_async_operation_t* operation;

  // Native handle (SOCKET or HANDLE) at the time of submit. Stored here so
  // completion dispatch can call WSAGetOverlappedResult/GetOverlappedResult
  // without inspecting the operation type.
  uintptr_t io_handle;

  // Intrusive list linkage for proactor's active carrier tracking.
  // Used for event wait carriers (linked into proactor->active_carriers).
  // Socket I/O carriers are not tracked in the active list.
  struct iree_async_iocp_carrier_t* next;

  // Per-type auxiliary data. The active member is determined by |type|.
#if defined(IREE_PLATFORM_WINDOWS)
  union {
    // IREE_ASYNC_IOCP_CARRIER_EVENT_WAIT
    struct {
      // Handle returned by RegisterWaitForSingleObject. Required for
      // UnregisterWaitEx during cancellation or proactor destroy.
      HANDLE wait_handle;
    } event_wait;

    // IREE_ASYNC_IOCP_CARRIER_SOCKET_IO
    // Used for WSARecv, WSASend, WSASendTo, WSARecvFrom.
    struct {
      WSABUF wsabuf[IREE_ASYNC_IOCP_MAX_SCATTER_GATHER_BUFFERS];
      DWORD buffer_count;
      DWORD flags;
      // WSARecvFrom writes the actual sender address length here
      // asynchronously at completion time. Must be in the carrier (not a stack
      // local) because overlapped I/O completes after submit returns.
      // Only meaningful for SOCKET_RECVFROM operations.
      int sender_address_length;
    } socket_io;

    // IREE_ASYNC_IOCP_CARRIER_ACCEPT
    struct {
      // Pre-created accept socket. AcceptEx requires the accept socket to be
      // created before the call. On completion, SO_UPDATE_ACCEPT_CONTEXT
      // transfers the listen socket's properties.
      SOCKET accept_socket;
      // Address output buffer. AcceptEx writes both local and remote addresses
      // here. Each address slot needs the maximum address size plus 16 bytes
      // of padding (AcceptEx requirement). sockaddr_storage (128 bytes) covers
      // all address families including AF_UNIX (sockaddr_un is 110 bytes).
      uint8_t address_buffer[2 * (sizeof(struct sockaddr_storage) + 16)];
      // Address lengths passed to AcceptEx and GetAcceptExSockaddrs.
      DWORD local_address_length;
      DWORD remote_address_length;
    } accept;

    // IREE_ASYNC_IOCP_CARRIER_RECV_POOL
    struct {
      WSABUF wsabuf;
      DWORD flags;
    } recv_pool;

    // IREE_ASYNC_IOCP_CARRIER_CONNECT: no extra data needed.
  } data;
#else
  // Placeholder for non-Windows compilation. Sized to accommodate the largest
  // union member (accept: SOCKET + address_buffer[2*(128+16)] + 2 DWORDs =
  // ~304 bytes; socket_io: WSABUF[8] + DWORD + DWORD + int = ~140 bytes).
  // Verified by _Static_assert in proactor.c on Windows builds.
  uint8_t data_placeholder[320];
#endif
} iree_async_iocp_carrier_t;

//===----------------------------------------------------------------------===//
// Event source tracking
//===----------------------------------------------------------------------===//

// Tracks a registered event source for persistent monitoring of a HANDLE.
// On IOCP, event sources use RegisterWaitForSingleObject to receive callbacks
// when the HANDLE is signaled, then post a tagged completion to the IOCP port.
// Doubly-linked list node; proactor owns the list.
struct iree_async_event_source_t {
  // Intrusive doubly-linked list for efficient removal.
  struct iree_async_event_source_t* next;
  struct iree_async_event_source_t* prev;

  // Owning proactor (for vtable access in callbacks).
  iree_async_proactor_t* proactor;

  // The monitored fd/HANDLE (not owned by the event source).
  // On Windows this is stored as an iree_async_primitive_t for type safety.
  int fd;

  // User callback invoked when the handle is signaled.
  iree_async_event_source_callback_t callback;

  // Allocator used to allocate this struct (for deallocation).
  iree_allocator_t allocator;
};

//===----------------------------------------------------------------------===//
// Proactor implementation struct
//===----------------------------------------------------------------------===//

// IOCP proactor state.
//
// Unlike the POSIX backend, IOCP is completion-based: the kernel performs I/O
// and posts completion packets to the port. This means:
//   - No worker thread pool (I/O completes in kernel, not userspace)
//   - Socket/file operations issue overlapped I/O directly from submit()
//   - poll() calls GetQueuedCompletionStatusEx to dequeue completions
//   - wake() uses PostQueuedCompletionStatus with a sentinel
//   - MPSC pending_queue only for non-I/O operations (timers, events, etc.)
typedef struct iree_async_proactor_iocp_t {
  // Must be first for safe casting.
  iree_async_proactor_t base;
  iree_atomic_int32_t shutdown_requested;

  // IOCP completion port handle. All overlapped I/O completions and
  // cross-thread wakeups are delivered through this single handle.
  // Stored as uintptr_t to avoid requiring windows.h in headers.
  uintptr_t completion_port;

  // MPSC queue for operations that cannot be submitted directly as overlapped
  // I/O (timers, event waits, notification waits, relay management). Submit()
  // pushes here from arbitrary threads; poll() drains on the poll thread.
  iree_atomic_slist_t pending_queue;

  // Semaphore wait operations funneled to the poll thread.
  iree_atomic_slist_t pending_semaphore_waits;

  // Sorted timer list (poll thread only). Timers are inserted from
  // pending_queue drain and removed by expiration or cancellation.
  iree_async_iocp_timer_list_t timers;

  // Number of timers with CANCELLED flag set that haven't been removed from
  // the timer_list yet. cancel() increments from any thread; the poll thread
  // decrements after removing cancelled timers. When non-zero, the poll loop
  // scans the timer_list for cancelled entries.
  iree_atomic_int32_t pending_timer_cancellation_count;

  // Number of event wait carriers with CANCELLED flag set that haven't been
  // unregistered yet. cancel() increments from any thread; the poll thread
  // decrements after unregistering cancelled waits. When non-zero, the poll
  // loop scans the active_carriers list for cancelled entries.
  iree_atomic_int32_t pending_event_wait_cancellation_count;

  // Active event wait carriers (poll thread only). Singly-linked list of
  // carriers with outstanding RegisterWaitForSingleObject registrations.
  // Walked during proactor destroy to unregister outstanding waits.
  iree_async_iocp_carrier_t* active_carriers;

  // Notifications with pending async wait operations (poll thread only).
  // Singly-linked list via iree_async_notification_t::platform.iocp.
  // The poll loop walks this to check for epoch advancement.
  struct iree_async_notification_t* notifications_with_waits;

  // Cross-proactor messaging state.
  // IOCP has native kernel-mediated messaging via PostQueuedCompletionStatus
  // to the target proactor's completion port. The message pool is used for
  // consistent API with other backends.
  iree_async_message_pool_t message_pool;
  iree_async_proactor_message_callback_t message_callback;

  // Sequence emulator for IREE_ASYNC_OPERATION_TYPE_SEQUENCE operations.
  // Drives step-by-step execution when step_fn is set. When step_fn is NULL,
  // the LINK path is used instead (no emulator involvement).
  iree_async_sequence_emulator_t sequence_emulator;

  // Resource lists (poll thread only).
  // Event sources and relays are kept in linked lists for cleanup/destroy.
  iree_async_event_source_t* event_sources;
  struct iree_async_relay_t* relays;

  // Signal handling (lazy-initialized on first subscribe).
  // Windows uses SetConsoleCtrlHandler for INTERRUPT (Ctrl+C) and TERMINATE
  // (close). HANGUP/QUIT/USER1/USER2 are not supported and return
  // INVALID_ARGUMENT on subscribe. The console ctrl handler fires on an OS
  // thread pool thread and posts to the IOCP completion port; the poll thread
  // dispatches to subscribers. No event source or separate fd needed (unlike
  // POSIX signalfd).
  struct {
    bool initialized;
    iree_async_signal_dispatch_state_t dispatch_state;
    iree_async_signal_subscription_t* subscriptions[IREE_ASYNC_SIGNAL_COUNT];
    int subscriber_count[IREE_ASYNC_SIGNAL_COUNT];
  } signal;

  // WSA extension function pointers loaded lazily on first socket creation.
  // AcceptEx, ConnectEx, and GetAcceptExSockaddrs are not directly linked —
  // they must be loaded per-process via
  // WSAIoctl(SIO_GET_EXTENSION_FUNCTION_POINTER).
  struct {
    bool loaded;
#if defined(IREE_PLATFORM_WINDOWS)
    LPFN_ACCEPTEX AcceptEx;
    LPFN_CONNECTEX ConnectEx;
    LPFN_GETACCEPTEXSOCKADDRS GetAcceptExSockaddrs;
#endif  // IREE_PLATFORM_WINDOWS
  } wsa_extensions;

  // Singleton constraint: only one READ-access slab may be registered at a
  // time. Mirrors io_uring's fixed buffer table limitation, enforced as a
  // public API contract for portability.
  bool has_read_slab_registration;

  // Detected and allowed capabilities.
  iree_async_proactor_capabilities_t capabilities;

  // Outstanding carrier count for leak detection. Incremented at carrier
  // allocation, decremented at carrier free. Asserted zero at destroy to catch
  // callers that destroy the proactor with pending overlapped I/O.
  iree_atomic_int32_t outstanding_carrier_count;
} iree_async_proactor_iocp_t;

static inline iree_async_proactor_iocp_t* iree_async_proactor_iocp_cast(
    iree_async_proactor_t* proactor) {
  return (iree_async_proactor_iocp_t*)proactor;
}

//===----------------------------------------------------------------------===//
// Semaphore wait tracker (shared between submit and poll/cancel)
//===----------------------------------------------------------------------===//

// Tracks a pending SEMAPHORE_WAIT operation.
// Heap-allocated per wait operation, freed when the operation completes.
// Contains embedded timepoints for each semaphore being waited on.
typedef struct iree_async_iocp_semaphore_wait_tracker_t {
  // Intrusive MPSC list link for pending completion queue.
  iree_atomic_slist_entry_t slist_entry;

  // Back-pointer to the wait operation being tracked.
  iree_async_semaphore_wait_operation_t* operation;

  // Proactor to wake when a semaphore fires.
  iree_async_proactor_iocp_t* proactor;

  // Allocator used for this tracker.
  iree_allocator_t allocator;

  // Number of semaphores being waited on.
  iree_host_size_t count;

  // Number of successfully registered timepoints. Used during cleanup to
  // cancel only the timepoints that were actually registered.
  iree_host_size_t registered_count;

  // For ALL mode: remaining semaphores to satisfy (count down to 0).
  // For ANY mode: first satisfied index (starts at -1, CAS to winning index).
  iree_atomic_int32_t remaining_or_satisfied;

  // Completion status. Written by timepoint callback if failure occurs.
  // CAS: first non-OK status wins.
  iree_atomic_intptr_t completion_status;

  // Guards against double-enqueue to the pending_semaphore_waits slist.
  // Multiple paths can independently decide to enqueue (success callback via
  // remaining_or_satisfied, error callback unconditionally, cancel via
  // completion_status). Only the thread that CAS-es this from 0 to 1
  // actually pushes the tracker to the slist.
  iree_atomic_int32_t enqueued;

  // LINKED chain continuation head. When the wait has LINKED flag, the
  // chain is transferred here during submit and dispatched on completion.
  iree_async_operation_t* continuation_head;

  // Flexible array of timepoints (one per semaphore).
  iree_async_semaphore_timepoint_t timepoints[];
} iree_async_iocp_semaphore_wait_tracker_t;

//===----------------------------------------------------------------------===//
// Internal APIs (shared across proactor implementation files)
//===----------------------------------------------------------------------===//

// Vtable for same-backend validation in submit.
extern const iree_async_proactor_vtable_t iree_async_proactor_iocp_vtable;

// Sentinel value in dwNumberOfBytesTransferred indicating that a direct
// completion carries a pre-computed iree_status_t stashed in operation->next,
// rather than a WSA error code that needs conversion.
#define IREE_ASYNC_IOCP_STASHED_STATUS_SENTINEL 0xFFFFFFFFu

// Wakes the poll thread by posting a sentinel completion to the IOCP port.
// (proactor.c)
void iree_async_proactor_iocp_wake(iree_async_proactor_t* base_proactor);

// Enqueues an operation to the pending_queue for poll-thread processing.
// Retains operation resources and wakes the poll thread. (proactor.c)
void iree_async_proactor_iocp_push_pending(iree_async_proactor_iocp_t* proactor,
                                           iree_async_operation_t* operation);

// Cancels a continuation chain by directly invoking callbacks with CANCELLED.
// Returns the number of callbacks invoked. (proactor_submit.c)
iree_host_size_t iree_async_proactor_iocp_cancel_continuation_chain(
    iree_async_operation_t* chain_head);

// Submits a continuation chain head. On submit failure, fires the chain head's
// callback with the error and cancels remaining continuations.
// (proactor_submit.c)
void iree_async_proactor_iocp_submit_continuation_chain(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* chain_head);

// Dispatches a linked_next continuation chain based on the trigger's status.
// On success: submits the chain. On failure: cancels with CANCELLED callbacks.
// Returns the number of directly-invoked callbacks. (proactor_submit.c)
iree_host_size_t iree_async_proactor_iocp_dispatch_linked_continuation(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation,
    iree_status_t trigger_status);

// Enqueues a completed semaphore wait tracker for the poll thread to drain.
// Called from timepoint callbacks running on arbitrary threads.
// (proactor_submit.c)
void iree_async_proactor_iocp_semaphore_wait_enqueue_completion(
    iree_async_iocp_semaphore_wait_tracker_t* tracker);

// Submit vtable implementation (in proactor_submit.c).
iree_status_t iree_async_proactor_iocp_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IOCP_PROACTOR_H_
