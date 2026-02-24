// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// accept4() in compat.h requires _GNU_SOURCE for the glibc declaration.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif  // _GNU_SOURCE

#include "iree/async/platform/posix/proactor.h"

#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/posix/compat.h"
#include "iree/async/platform/posix/fence.h"
#include "iree/async/platform/posix/relay.h"
#include "iree/async/platform/posix/socket.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"
#include "iree/base/internal/math.h"

#if defined(IREE_PLATFORM_LINUX)
#include <sys/eventfd.h>
#endif  // IREE_PLATFORM_LINUX

// Result of attempting a non-blocking I/O operation. Separates the "did it
// complete?" question (control flow) from the "what happened?" question
// (status). EAGAIN is normal non-blocking behavior, not an error.
typedef enum {
  // The operation completed. The status out parameter carries the result.
  IREE_ASYNC_IO_COMPLETE = 0,
  // The I/O would block (EAGAIN/EWOULDBLOCK). Re-arm for poll.
  IREE_ASYNC_IO_WOULD_BLOCK,
} iree_async_io_result_t;

static const iree_async_proactor_vtable_t iree_async_proactor_posix_vtable;
static iree_host_size_t iree_async_proactor_posix_dispatch_linked_continuation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    iree_status_t trigger_status);
static void iree_async_proactor_posix_process_notification_waits(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_t* notification);
static void iree_async_proactor_posix_signal_deinitialize(
    iree_async_proactor_posix_t* proactor);

//===----------------------------------------------------------------------===//
// Internal flags for POSIX proactor operations
//===----------------------------------------------------------------------===//

// Internal flags used by the POSIX proactor for operation state management.
// These are written to iree_async_operation_t::internal_flags during execution.
enum iree_async_posix_operation_internal_flags_e {
  // Set when an operation is cancelled while being processed (not in the timer
  // list). The expiration loop checks this flag before invoking the callback.
  // See the cancellation algorithm in the timer processing code.
  IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED = 1u << 0,
};

//===----------------------------------------------------------------------===//
// Proactor creation and destruction
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_create_posix(
    iree_async_proactor_options_t options, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  return iree_async_proactor_create_posix_with_backend(
      options, IREE_ASYNC_POSIX_EVENT_BACKEND_DEFAULT, allocator, out_proactor);
}

iree_status_t iree_async_proactor_create_posix_with_backend(
    iree_async_proactor_options_t options,
    iree_async_posix_event_backend_t event_backend, iree_allocator_t allocator,
    iree_async_proactor_t** out_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_proactor);
  *out_proactor = NULL;

  // Worker count uses a fixed default for now. Could be extended to be
  // configurable via options if needed.
  iree_host_size_t worker_count = IREE_ASYNC_POSIX_DEFAULT_WORKER_COUNT;

  // Message pool capacity from options, with default fallback.
  iree_host_size_t message_pool_capacity = options.message_pool_capacity;
  if (message_pool_capacity == 0) {
    message_pool_capacity = IREE_ASYNC_MESSAGE_POOL_DEFAULT_CAPACITY;
  }

  // Pool capacity from options, with default fallback.
  iree_host_size_t ready_pool_capacity = options.max_concurrent_operations;
  if (ready_pool_capacity == 0) {
    ready_pool_capacity = IREE_ASYNC_POSIX_DEFAULT_POOL_CAPACITY;
  }
  // Completion pool is larger to avoid backpressure: workers may be holding
  // ready entries while needing completion entries.
  iree_host_size_t completion_pool_capacity =
      ready_pool_capacity + worker_count;

  // Calculate allocation layout with proper alignment.
  // Layout: [proactor | workers[] | ready_entries[] | completion_entries[]
  //          | message_entries[]]
  iree_host_size_t total_size = 0;
  iree_host_size_t workers_offset = 0;
  iree_host_size_t ready_entries_offset = 0;
  iree_host_size_t completion_entries_offset = 0;
  iree_host_size_t message_entries_offset = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_async_proactor_posix_t), &total_size,
              IREE_STRUCT_FIELD_ALIGNED(
                  worker_count, iree_async_posix_worker_t,
                  iree_hardware_destructive_interference_size, &workers_offset),
              IREE_STRUCT_FIELD_ALIGNED(
                  ready_pool_capacity, iree_async_posix_ready_op_t,
                  iree_hardware_destructive_interference_size,
                  &ready_entries_offset),
              IREE_STRUCT_FIELD(completion_pool_capacity,
                                iree_async_posix_completion_t,
                                &completion_entries_offset),
              IREE_STRUCT_FIELD(message_pool_capacity,
                                iree_async_message_pool_entry_t,
                                &message_entries_offset)))

  // Single allocation for everything.
  iree_async_proactor_posix_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&proactor));
  memset(proactor, 0, total_size);

  // Initialize base proactor.
  iree_async_proactor_initialize(&iree_async_proactor_posix_vtable,
                                 options.debug_name, allocator,
                                 &proactor->base);

  // Initialize sequence emulator for SEQUENCE operation support.
  // Uses vtable-dispatched submit_one which routes individual step operations
  // through this backend's normal submit path.
  iree_async_sequence_emulator_initialize(&proactor->sequence_emulator,
                                          &proactor->base,
                                          iree_async_proactor_submit_one);

  // Set up pointers into trailing data.
  proactor->workers =
      (iree_async_posix_worker_t*)((uint8_t*)proactor + workers_offset);
  proactor->worker_count = worker_count;

  // Store capabilities. LINKED_OPERATIONS is emulated in userspace via
  // linked_next chains — the proactor builds a linked list during submit and
  // dispatches continuations on completion.
  iree_async_proactor_capabilities_t supported_capabilities =
      IREE_ASYNC_PROACTOR_CAPABILITY_LINKED_OPERATIONS |
      IREE_ASYNC_PROACTOR_CAPABILITY_REGISTERED_BUFFERS |
      IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT |
      IREE_ASYNC_PROACTOR_CAPABILITY_DEVICE_FENCE;
  proactor->capabilities =
      options.allowed_capabilities & supported_capabilities;

  // Initialize atomic shutdown flag.
  iree_atomic_store(&proactor->shutdown_requested, 0,
                    iree_memory_order_release);

  // Initialize all MPSC queues.
  iree_atomic_slist_initialize(&proactor->pending_queue);
  iree_atomic_slist_initialize(&proactor->ready_queue);
  iree_notification_initialize(&proactor->ready_notification);
  iree_atomic_slist_initialize(&proactor->completion_queue);
  iree_atomic_slist_initialize(&proactor->pending_semaphore_waits);
  iree_atomic_slist_initialize(&proactor->pending_fence_imports);

  // Initialize pools with external storage.
  iree_async_posix_ready_op_t* ready_entries =
      (iree_async_posix_ready_op_t*)((uint8_t*)proactor + ready_entries_offset);
  iree_async_posix_ready_pool_initialize_with_storage(
      ready_pool_capacity, ready_entries, &proactor->ready_pool);

  iree_async_posix_completion_t* completion_entries =
      (iree_async_posix_completion_t*)((uint8_t*)proactor +
                                       completion_entries_offset);
  iree_async_posix_completion_pool_initialize_with_storage(
      completion_pool_capacity, completion_entries, &proactor->completion_pool);

  iree_async_message_pool_entry_t* message_entries =
      (iree_async_message_pool_entry_t*)((uint8_t*)proactor +
                                         message_entries_offset);
  iree_async_message_pool_initialize(message_pool_capacity, message_entries,
                                     &proactor->message_pool);

  // Suppress SIGPIPE process-wide. This backend calls writev() directly on
  // sockets, and a write to a socket whose peer has closed generates SIGPIPE
  // with default disposition = terminate. We need the EPIPE errno instead.
  // Idempotent and safe to call multiple times.
  iree_status_t status = iree_async_signal_ignore_broken_pipe();

  // Initialize components with chained error handling.
  if (iree_status_is_ok(status)) {
    status = iree_async_posix_event_set_allocate(event_backend, allocator,
                                                 &proactor->event_set);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_posix_fd_map_initialize(
        /*initial_capacity=*/16, allocator, &proactor->fd_map);
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_posix_wake_initialize(&proactor->wake);
  }

  // Add wake fd to event set so poll() wakes on wake signal.
  if (iree_status_is_ok(status)) {
    status = iree_async_posix_event_set_add(proactor->event_set,
                                            proactor->wake.read_fd, POLLIN);
  }

  // Initialize workers. Each worker creates its own thread.
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < worker_count; ++i) {
      status = iree_async_posix_worker_initialize(proactor, i, allocator,
                                                  &proactor->workers[i]);
      if (!iree_status_is_ok(status)) break;
      ++proactor->initialized_workers;
    }
  }

  // Single exit point: assign output on success, release on failure.
  if (iree_status_is_ok(status)) {
    *out_proactor = (iree_async_proactor_t*)proactor;
  } else {
    iree_async_proactor_release((iree_async_proactor_t*)proactor);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Proactor destruction
//===----------------------------------------------------------------------===//

static void iree_async_proactor_posix_destroy(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  if (!proactor) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Signal shutdown to prevent new work from being accepted.
  iree_atomic_store(&proactor->shutdown_requested, 1,
                    iree_memory_order_release);

  // Three-phase shutdown following iree/task/ patterns.
  // Only process workers that were actually initialized (handles partial init).

  // Phase 1: Request all workers to exit.
  // This is fast and non-blocking; just sets a flag and wakes each worker.
  for (iree_host_size_t i = 0; i < proactor->initialized_workers; ++i) {
    iree_async_posix_worker_request_exit(&proactor->workers[i]);
  }

  // Phase 2: Wait for all workers to reach ZOMBIE state.
  // Workers may take varying times to exit depending on their workload.
  for (iree_host_size_t i = 0; i < proactor->initialized_workers; ++i) {
    iree_async_posix_worker_await_exit(&proactor->workers[i]);
  }

  // Phase 3: Deinitialize all workers now that no threads are running.
  // Safe to access all worker data structures without synchronization.
  for (iree_host_size_t i = 0; i < proactor->initialized_workers; ++i) {
    iree_async_posix_worker_deinitialize(&proactor->workers[i]);
  }

  // Clean up signal handling if it was initialized.
  // Must happen before event_set cleanup since it unregisters the event source.
  if (proactor->signal.initialized) {
    iree_async_proactor_posix_signal_deinitialize(proactor);
  }

  // Clean up relays before event_set (relays may have fds registered there).
  iree_async_proactor_posix_destroy_all_relays(proactor);

  // Drain any pending fence imports that were never registered (import_fence
  // called but poll() never ran to drain them). Close fds, release semaphores.
  {
    iree_atomic_slist_entry_t* head = NULL;
    iree_atomic_slist_flush(&proactor->pending_fence_imports,
                            IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                            &head, /*out_tail=*/NULL);
    while (head) {
      iree_async_posix_fence_import_tracker_t* tracker =
          (iree_async_posix_fence_import_tracker_t*)head;
      head = head->next;
      close(tracker->fence_fd);
      iree_async_semaphore_release(tracker->semaphore);
      iree_allocator_free(tracker->allocator, tracker);
    }
  }

  // Clean up components in reverse initialization order.
  iree_async_posix_wake_deinitialize(&proactor->wake);
  iree_async_posix_fd_map_deinitialize(&proactor->fd_map);
  iree_async_posix_event_set_free(proactor->event_set);

  // Pools use external storage (in our allocation), just reset state.
  iree_async_posix_completion_pool_deinitialize(&proactor->completion_pool);
  iree_async_posix_ready_pool_deinitialize(&proactor->ready_pool);

  // Deinitialize notifications and queues.
  iree_notification_deinitialize(&proactor->ready_notification);
  iree_atomic_slist_deinitialize(&proactor->pending_semaphore_waits);
  iree_atomic_slist_deinitialize(&proactor->pending_fence_imports);
  iree_async_message_pool_deinitialize(&proactor->message_pool);
  iree_atomic_slist_deinitialize(&proactor->completion_queue);
  iree_atomic_slist_deinitialize(&proactor->ready_queue);
  iree_atomic_slist_deinitialize(&proactor->pending_queue);

  // Free the single allocation.
  iree_allocator_t allocator = proactor->base.allocator;
  iree_allocator_free(allocator, proactor);

  IREE_TRACE_ZONE_END(z0);
}

void iree_async_proactor_posix_wake_poll_thread(
    iree_async_proactor_posix_t* proactor) {
  iree_async_posix_wake_trigger(&proactor->wake);
}

//===----------------------------------------------------------------------===//
// Submit infrastructure
//===----------------------------------------------------------------------===//

// Tracks a pending SEMAPHORE_WAIT operation.
// Heap-allocated per wait operation, freed when the operation completes.
// Contains embedded timepoints for each semaphore being waited on.
typedef struct iree_async_posix_semaphore_wait_tracker_t {
  // Intrusive MPSC list link for pending completion queue.
  iree_atomic_slist_entry_t slist_entry;

  // Back-pointer to the wait operation being tracked.
  iree_async_semaphore_wait_operation_t* operation;

  // Proactor to wake when a semaphore fires.
  iree_async_proactor_posix_t* proactor;

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

  // Guard against double-enqueue: success callbacks (remaining_or_satisfied),
  // error callbacks, and cancel all independently decide to enqueue. Only the
  // first to CAS this from 0→1 actually pushes to the MPSC slist.
  iree_atomic_int32_t enqueued;

  // LINKED chain continuation head. When the wait has LINKED flag, the
  // chain is transferred here during submit and dispatched on completion.
  iree_async_operation_t* continuation_head;

  // Flexible array of timepoints (one per semaphore).
  iree_async_semaphore_timepoint_t timepoints[];
} iree_async_posix_semaphore_wait_tracker_t;

// Pushes a completion for delivery in poll(). Does not retain operation
// resources — callers in the drain/execute paths (where push_pending already
// retained) use this directly.
static iree_status_t iree_async_proactor_posix_complete_immediately(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    iree_status_t op_status, iree_async_completion_flags_t flags) {
  iree_async_posix_completion_t* completion =
      iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
  if (!completion) {
    iree_status_ignore(op_status);
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "completion pool exhausted; call poll() to drain completions");
  }
  completion->operation = operation;
  completion->status = op_status;
  completion->flags = flags;
  iree_atomic_slist_push(&proactor->completion_queue, &completion->slist_entry);
  return iree_ok_status();
}

// Retains operation resources and pushes a completion for delivery in poll().
// Used from submit paths where the operation is entering proactor management.
// On failure (pool exhausted), the retain is rolled back — the caller can
// safely return the error without any resource cleanup.
static iree_status_t iree_async_proactor_posix_complete_on_submit(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    iree_status_t op_status, iree_async_completion_flags_t flags) {
  iree_async_operation_retain_resources(operation);
  iree_status_t status = iree_async_proactor_posix_complete_immediately(
      proactor, operation, op_status, flags);
  if (!iree_status_is_ok(status)) {
    iree_async_operation_release_resources(operation);
  }
  return status;
}

// Pushes an operation directly to the ready queue for worker execution.
// Used for file operations that don't need to wait for fd readiness.
// Returns RESOURCE_EXHAUSTED if the ready pool is full.
static iree_status_t iree_async_proactor_posix_enqueue_for_execution(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    iree_async_poll_events_t poll_events) {
  // Acquire a ready entry from the pool.
  iree_async_posix_ready_op_t* ready_op =
      iree_async_posix_ready_pool_acquire(&proactor->ready_pool);
  if (!ready_op) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "ready pool exhausted; too many concurrent file operations");
  }

  // Fill in the ready entry.
  ready_op->operation = operation;
  ready_op->poll_events = poll_events;

  // Push to the ready queue for workers.
  iree_atomic_slist_push(&proactor->ready_queue, &ready_op->slist_entry);

  // Wake a worker to process it.
  iree_notification_post(&proactor->ready_notification, 1);

  return iree_ok_status();
}

// Returns the poll event mask for an operation type.
static short iree_async_operation_type_to_poll_events(
    iree_async_operation_type_t type) {
  switch (type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
      return POLLIN;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return POLLOUT;
    default:
      return 0;
  }
}

// Returns the fd to monitor for a pending operation.
// Requires the operation to be a type that monitors a file descriptor.
static int iree_async_proactor_posix_operation_fd(
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return ((iree_async_socket_accept_operation_t*)operation)
          ->listen_socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return ((iree_async_socket_connect_operation_t*)operation)
          ->socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return ((iree_async_socket_recv_operation_t*)operation)
          ->socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return ((iree_async_socket_send_operation_t*)operation)
          ->socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return ((iree_async_socket_recv_pool_operation_t*)operation)
          ->socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return ((iree_async_socket_recvfrom_operation_t*)operation)
          ->socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return ((iree_async_socket_sendto_operation_t*)operation)
          ->socket->primitive.value.fd;
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
      return ((iree_async_event_wait_operation_t*)operation)
          ->event->primitive.value.fd;
    default:
      return -1;
  }
}

// Returns true if the operation type is registered via the fd_map (socket ops,
// event wait). Used to match the set of types that cancel() increments the
// pending_fd_cancellation_count for.
static bool iree_async_proactor_posix_is_fd_operation(
    iree_async_operation_type_t type) {
  switch (type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
      return true;
    default:
      return false;
  }
}

// Retains operation resources and pushes the operation to the pending_queue for
// the poll thread to process. The pending_queue is an MPSC queue: any thread
// may push, only the poll thread pops. This ensures all fd_map, event_set, and
// timer_list mutations happen exclusively on the poll thread.
//
// Uses the operation's `next` pointer (offset 0) as the slist_entry, which is
// safe because the operation is exclusively owned by the queue until the poll
// thread pops it.
static void iree_async_proactor_posix_push_pending(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation) {
  iree_async_operation_retain_resources(operation);
  iree_atomic_slist_push(&proactor->pending_queue,
                         (iree_atomic_slist_entry_t*)operation);
}

// Computes the combined events mask for an operation chain.
// Walks the chain and ORs together all operation events.
static short iree_async_proactor_posix_compute_chain_events(
    iree_async_operation_t* chain_head) {
  short combined_events = 0;
  for (iree_async_operation_t* op = chain_head; op != NULL; op = op->next) {
    combined_events |= iree_async_operation_type_to_poll_events(op->type);
  }
  return combined_events;
}

// Registers an fd-based operation with the event_set and fd_map.
// Called exclusively from the poll thread during pending_queue drain.
//
// Supports multiple concurrent operations on the same fd (e.g., concurrent
// sends, or send + recv). When an fd already has pending operations, the new
// operation is chained and the event_set is updated if needed.
static iree_status_t iree_async_proactor_posix_register_fd_operation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    int fd) {
  short events = iree_async_operation_type_to_poll_events(operation->type);
  if (events == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid fd operation type %d",
                            (int)operation->type);
  }

  // Clear next pointer - will be set if chaining.
  operation->next = NULL;

  // Try to insert or chain into the fd_map.
  bool was_chained = false;
  IREE_RETURN_IF_ERROR(iree_async_posix_fd_map_insert_or_chain(
      &proactor->fd_map, fd, IREE_ASYNC_POSIX_FD_HANDLER_OPERATION, operation,
      &was_chained));

  iree_status_t status = iree_ok_status();
  if (was_chained) {
    // Operation was chained onto existing entry. May need to update event_set
    // if the new operation wants different events than currently registered.
    // Look up the chain head to compute combined events.
    iree_async_posix_fd_handler_type_t handler_type;
    void* handler;
    if (iree_async_posix_fd_map_lookup(&proactor->fd_map, fd, &handler_type,
                                       &handler)) {
      short combined_events =
          iree_async_proactor_posix_compute_chain_events(handler);
      // Modify to ensure all needed events are registered.
      status = iree_async_posix_event_set_modify(proactor->event_set, fd,
                                                 combined_events);
      // If modify fails, the operation is still chained - this is a partial
      // failure but the operation may still work if the existing events match.
      // Log and continue rather than trying to undo the chain.
      iree_status_ignore(status);
    }
  } else {
    // First operation for this fd - add to event_set.
    status = iree_async_posix_event_set_add(proactor->event_set, fd, events);
    if (!iree_status_is_ok(status)) {
      // Remove from fd_map on failure.
      iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
      return status;
    }
  }

  return iree_ok_status();
}

// Registers a NOTIFICATION_WAIT operation from the pending_queue.
// Checks for epoch advancement (immediate completion) and manages the
// notification's pending wait list and event_set registration.
// Called exclusively from the poll thread during pending_queue drain.
static iree_status_t iree_async_proactor_posix_register_notification_wait(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_wait_operation_t* wait) {
  iree_async_notification_t* notification = wait->notification;

  // Check if epoch already advanced (signal arrived between submit and
  // pending_queue drain). Complete immediately without registration.
  uint32_t current_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  if (wait->wait_token != current_epoch) {
    // Notification reference is released by release_operation_resources
    // when the completion is drained.
    return iree_async_proactor_posix_complete_immediately(
        proactor, &wait->base, iree_ok_status(),
        IREE_ASYNC_COMPLETION_FLAG_NONE);
  }

  // Activate the notification's fd in event_set + fd_map if this is the first
  // consumer (no prior async waits or relay subscribers).
  bool was_active = (notification->platform.posix.pending_waits != NULL ||
                     notification->platform.posix.relay_list != NULL);
  if (!was_active) {
    int fd = notification->platform.posix.primitive.value.fd;
    iree_status_t status =
        iree_async_posix_event_set_add(proactor->event_set, fd, POLLIN);
    if (iree_status_is_ok(status)) {
      status = iree_async_posix_fd_map_insert(
          &proactor->fd_map, fd, IREE_ASYNC_POSIX_FD_HANDLER_NOTIFICATION,
          notification);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(
            iree_async_posix_event_set_remove(proactor->event_set, fd));
      }
    }
    if (!iree_status_is_ok(status)) {
      // Notification reference is released by release_operation_resources
      // when the caller (drain_pending_queue) pushes the error completion.
      return status;
    }
  }

  // Add to notification's pending wait list (head insert).
  // Uses operation->next for intrusive linkage.
  wait->base.next =
      (iree_async_operation_t*)notification->platform.posix.pending_waits;
  notification->platform.posix.pending_waits = wait;
  return iree_ok_status();
}

// Drains the pending_queue and performs the actual fd/timer registration.
// Called exclusively from the poll thread at the top of each poll() iteration.
//
// For fd-based operations (sockets, EVENT_WAIT): registers with event_set
// and fd_map. For timers: inserts into the sorted timer list.
// Operations that were cancelled while in the queue (CANCELLED flag set)
// are completed immediately with IREE_STATUS_CANCELLED.
static void iree_async_proactor_posix_drain_pending_queue(
    iree_async_proactor_posix_t* proactor) {
  iree_atomic_slist_entry_t* entry = NULL;
  while ((entry = iree_atomic_slist_pop(&proactor->pending_queue)) != NULL) {
    iree_async_operation_t* operation = (iree_async_operation_t*)entry;

    // Check if the operation was cancelled while sitting in the queue.
    // Retained resources (sockets, events, notifications) are released via
    // release_operation_resources — either during drain_completion_queue
    // (normal path) or inline (fallback path).
    if (iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                         IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED)) {
      // cancel() incremented the type-specific cancellation counter. Decrement
      // it here since the operation was never registered in its data structure
      // (it was still in the pending queue when cancel was called).
      if (iree_async_proactor_posix_is_fd_operation(operation->type)) {
        iree_atomic_fetch_sub(&proactor->pending_fd_cancellation_count, 1,
                              iree_memory_order_release);
      } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
        iree_atomic_fetch_sub(&proactor->pending_timer_cancellation_count, 1,
                              iree_memory_order_release);
      }
      iree_status_t push_status =
          iree_async_proactor_posix_complete_immediately(
              proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED),
              IREE_ASYNC_COMPLETION_FLAG_NONE);
      if (!iree_status_is_ok(push_status)) {
        // Pool exhausted — release resources and dispatch directly.
        iree_status_ignore(push_status);
        iree_async_operation_release_resources(operation);
        iree_async_proactor_posix_dispatch_linked_continuation(
            proactor, operation, iree_status_from_code(IREE_STATUS_CANCELLED));
        if (operation->completion_fn) {
          operation->completion_fn(operation->user_data, operation,
                                   iree_status_from_code(IREE_STATUS_CANCELLED),
                                   IREE_ASYNC_COMPLETION_FLAG_NONE);
        }
      }
      continue;
    }

    iree_status_t status = iree_ok_status();

    switch (operation->type) {
      case IREE_ASYNC_OPERATION_TYPE_TIMER: {
        iree_async_posix_timer_list_insert(
            &proactor->timers, (iree_async_timer_operation_t*)operation);
        break;
      }

      case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
        int fd = iree_async_proactor_posix_operation_fd(operation);
        status = iree_async_proactor_posix_register_fd_operation(proactor,
                                                                 operation, fd);
        break;
      }

      case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
      case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
      case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
      case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE: {
        // File operations bypass fd polling and go directly to workers for
        // blocking I/O execution. Files don't poll reliably on all platforms.
        status = iree_async_proactor_posix_enqueue_for_execution(
            proactor, operation, IREE_ASYNC_POLL_EVENT_NONE);
        break;
      }

      case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
        iree_async_event_wait_operation_t* event_wait =
            (iree_async_event_wait_operation_t*)operation;
        int fd = event_wait->event->primitive.value.fd;
        status = iree_async_proactor_posix_register_fd_operation(proactor,
                                                                 operation, fd);
        break;
      }

      case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
        status = iree_async_proactor_posix_register_notification_wait(
            proactor, (iree_async_notification_wait_operation_t*)operation);
        break;
      }

      default:
        status = iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "POSIX proactor pending drain: unsupported operation type %d",
            (int)operation->type);
        break;
    }

    if (!iree_status_is_ok(status)) {
      // Registration failed. Complete the operation with the error.
      // Retained resources are released via release_operation_resources —
      // either during drain_completion_queue or inline in the fallback.
      iree_status_t push_status =
          iree_async_proactor_posix_complete_immediately(
              proactor, operation, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
      if (!iree_status_is_ok(push_status)) {
        // Pool exhausted — release resources and dispatch directly.
        iree_status_ignore(push_status);
        iree_async_operation_release_resources(operation);
        iree_async_proactor_posix_dispatch_linked_continuation(
            proactor, operation, status);
        if (operation->completion_fn) {
          operation->completion_fn(operation->user_data, operation, status,
                                   IREE_ASYNC_COMPLETION_FLAG_NONE);
        } else {
          iree_status_ignore(status);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Submit-time operation helpers
//===----------------------------------------------------------------------===//

// Eagerly tries accept4() on the submitting thread. If a connection is already
// pending, completes immediately. If EAGAIN (no pending connections — the
// common case for listening sockets), defers to the poll thread via the
// pending_queue for POLLIN notification. If any other error (e.g., EINVAL for
// a non-listening socket), completes immediately with the error.
//
// This matches io_uring's behavior where the kernel attempts accept4() when
// processing the SQE and returns a CQE immediately on both success and error.
static iree_status_t iree_async_proactor_posix_submit_socket_accept(
    iree_async_proactor_posix_t* proactor,
    iree_async_socket_accept_operation_t* accept_op) {
  struct sockaddr_storage addr;
  socklen_t addr_length = sizeof(addr);
  int accepted_fd =
      iree_posix_accept(accept_op->listen_socket->primitive.value.fd,
                        (struct sockaddr*)&addr, &addr_length);

  if (accepted_fd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    // No pending connections — defer to poll thread for POLLIN.
    // push_pending retains the listen_socket reference.
    iree_async_proactor_posix_push_pending(proactor, &accept_op->base);
    return iree_ok_status();
  }

  // All paths below go through complete_on_submit (which retains/releases the
  // listen_socket reference). Local resources (accepted_fd, accepted_socket)
  // must be cleaned up on complete_on_submit failure since no callback fires.

  iree_status_t op_status = iree_ok_status();
  iree_async_socket_t* accepted_socket = NULL;

  if (accepted_fd < 0) {
    // Immediate failure (e.g., EINVAL for a non-listening socket).
    op_status =
        iree_make_status(iree_status_code_from_errno(errno), "accept() failed");
  }

  if (iree_status_is_ok(op_status)) {
    op_status = iree_async_posix_socket_create_accepted(
        proactor, accepted_fd, accept_op->listen_socket->type,
        &accepted_socket);
    if (iree_status_is_ok(op_status)) {
      // Ownership of accepted_fd transferred to accepted_socket.
      accepted_fd = -1;
      accept_op->accepted_socket = accepted_socket;
      memcpy(accept_op->peer_address.storage, &addr, addr_length);
      accept_op->peer_address.length = addr_length;
    }
  }

  iree_async_completion_flags_t completion_flags =
      IREE_ASYNC_COMPLETION_FLAG_NONE;
  if (iree_status_is_ok(op_status) &&
      iree_any_bit_set(accept_op->base.flags,
                       IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    completion_flags = IREE_ASYNC_COMPLETION_FLAG_MORE;
  }

  iree_status_t status = iree_async_proactor_posix_complete_on_submit(
      proactor, &accept_op->base, op_status, completion_flags);
  if (!iree_status_is_ok(status)) {
    // complete_on_submit failed (pool exhausted). It already rolled back the
    // listen_socket retain. Clean up any local resources we created.
    if (accepted_socket) {
      accept_op->accepted_socket = NULL;
      iree_async_posix_socket_destroy(proactor, accepted_socket);
    }
    if (accepted_fd >= 0) close(accepted_fd);
  }
  return status;
}

// Starts a non-blocking connect(). The syscall itself is thread-safe and runs
// on the submitting thread. EINPROGRESS (the common case) defers completion to
// the poll thread via the pending_queue.
static iree_status_t iree_async_proactor_posix_submit_socket_connect(
    iree_async_proactor_posix_t* proactor,
    iree_async_socket_connect_operation_t* connect_op) {
  int fd = connect_op->socket->primitive.value.fd;
  int connect_result =
      connect(fd, (struct sockaddr*)connect_op->address.storage,
              (socklen_t)connect_op->address.length);
  if (connect_result == 0) {
    // Immediate success (rare, usually local connections).
    connect_op->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
    return iree_async_proactor_posix_complete_on_submit(
        proactor, &connect_op->base, iree_ok_status(),
        IREE_ASYNC_COMPLETION_FLAG_NONE);
  }
  if (errno == EINPROGRESS) {
    // Connection in progress — poll thread will register for POLLOUT.
    // push_pending retains the socket reference.
    connect_op->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTING;
    iree_async_proactor_posix_push_pending(proactor, &connect_op->base);
    return iree_ok_status();
  }
  // Immediate failure.
  return iree_async_proactor_posix_complete_on_submit(
      proactor, &connect_op->base,
      iree_make_status(iree_status_code_from_errno(errno), "connect() failed"),
      IREE_ASYNC_COMPLETION_FLAG_NONE);
}

// Closes a socket fd synchronously and completes the operation.
// close() doesn't touch fd_map/event_set, so it runs on the submitting thread.
// Uses complete_immediately (not complete_on_submit) because CLOSE consumes the
// caller's socket reference: release_operation_resources IS the consumption,
// with no matching retain. If the pool is exhausted, the close still happened
// (fd is closed) but the callback won't fire — the caller's reference is
// unaffected and they release it normally.
static iree_status_t iree_async_proactor_posix_submit_socket_close(
    iree_async_proactor_posix_t* proactor,
    iree_async_socket_close_operation_t* close_op) {
  int fd = close_op->socket->primitive.value.fd;
  int close_result = close(fd);
  close_op->socket->primitive.value.fd = -1;
  close_op->socket->state = IREE_ASYNC_SOCKET_STATE_CLOSED;
  iree_status_t close_status =
      close_result == 0 ? iree_ok_status()
                        : iree_make_status(iree_status_code_from_errno(errno),
                                           "close() failed");
  return iree_async_proactor_posix_complete_immediately(
      proactor, &close_op->base, close_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
}

// Validates and submits a SOCKET_RECV_POOL operation.
// Validates that the pool's region has WRITE access (required for receiving)
// and that the region was registered with this proactor. The actual recv is
// deferred to the poll thread via push_pending (which retains the socket).
static iree_status_t iree_async_proactor_posix_submit_recv_pool(
    iree_async_proactor_posix_t* proactor,
    iree_async_socket_recv_pool_operation_t* recv_pool) {
  iree_async_region_t* region = iree_async_buffer_pool_region(recv_pool->pool);
  if (IREE_UNLIKELY(!iree_any_bit_set(region->access_flags,
                                      IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SOCKET_RECV_POOL requires a buffer pool with WRITE access; "
        "this pool was registered with read-only access");
  }
  if (IREE_UNLIKELY(region->proactor != &proactor->base)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SOCKET_RECV_POOL buffer pool was registered with a different "
        "proactor");
  }

  recv_pool->bytes_received = 0;
  memset(&recv_pool->lease, 0, sizeof(recv_pool->lease));

  iree_async_proactor_posix_push_pending(proactor, &recv_pool->base);
  return iree_ok_status();
}

// Submits a timer operation. Timers with deadlines in the past complete
// immediately on the submitting thread; future timers go to the pending_queue
// for the poll thread to insert into the sorted timer list.
static iree_status_t iree_async_proactor_posix_submit_timer(
    iree_async_proactor_posix_t* proactor,
    iree_async_timer_operation_t* timer_op) {
  if (timer_op->deadline_ns <= iree_time_now()) {
    return iree_async_proactor_posix_complete_on_submit(
        proactor, &timer_op->base, iree_ok_status(),
        IREE_ASYNC_COMPLETION_FLAG_NONE);
  }
  iree_async_proactor_posix_push_pending(proactor, &timer_op->base);
  return iree_ok_status();
}

// Submits an EVENT_WAIT by deferring to the poll thread. push_pending retains
// the event reference; release_operation_resources releases it on completion.
static iree_status_t iree_async_proactor_posix_submit_event_wait(
    iree_async_proactor_posix_t* proactor,
    iree_async_event_wait_operation_t* event_wait) {
  iree_async_proactor_posix_push_pending(proactor, &event_wait->base);
  return iree_ok_status();
}

// Submits a NOTIFICATION_WAIT by capturing the current epoch as the wait token
// and deferring to the poll thread. push_pending retains the notification
// reference; release_operation_resources releases it on completion.
static iree_status_t iree_async_proactor_posix_submit_notification_wait(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_wait_operation_t* wait) {
  wait->wait_token =
      iree_atomic_load(&wait->notification->epoch, iree_memory_order_acquire);
  iree_async_proactor_posix_push_pending(proactor, &wait->base);
  return iree_ok_status();
}

// Submits a NOTIFICATION_SIGNAL by signaling synchronously and completing.
// The signal increments the epoch and writes to the eventfd/pipe, which also
// wakes the poll thread to check pending NOTIFICATION_WAITs.
static iree_status_t iree_async_proactor_posix_submit_notification_signal(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_signal_operation_t* signal_op) {
  signal_op->woken_count = 0;
  iree_async_notification_signal(signal_op->notification,
                                 signal_op->wake_count);
  return iree_async_proactor_posix_complete_on_submit(
      proactor, &signal_op->base, iree_ok_status(),
      IREE_ASYNC_COMPLETION_FLAG_NONE);
}

// Submits a SEMAPHORE_SIGNAL by signaling each semaphore synchronously and
// completing immediately. Errors are delivered via callback rather than
// returned from submit, maintaining standard async operation semantics.
static iree_status_t iree_async_proactor_posix_submit_semaphore_signal(
    iree_async_proactor_posix_t* proactor,
    iree_async_semaphore_signal_operation_t* signal_op) {
  iree_status_t op_status = iree_ok_status();
  for (iree_host_size_t i = 0; i < signal_op->count; ++i) {
    iree_status_t status = iree_async_semaphore_signal(
        signal_op->semaphores[i], signal_op->values[i], signal_op->frontier);
    if (!iree_status_is_ok(status)) {
      op_status = status;
      break;
    }
  }
  return iree_async_proactor_posix_complete_on_submit(
      proactor, &signal_op->base, op_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
}

// Enqueues a completed semaphore wait tracker for the poll thread to drain.
// Called from timepoint callbacks running on arbitrary threads.
static void iree_async_proactor_posix_semaphore_wait_enqueue_completion(
    iree_async_posix_semaphore_wait_tracker_t* tracker) {
  // Guard against double-enqueue: success callbacks (remaining_or_satisfied),
  // error callbacks, and cancel all independently decide to enqueue. Only the
  // first to set the flag actually pushes.
  int32_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(&tracker->enqueued, &expected, 1,
                                           iree_memory_order_acq_rel,
                                           iree_memory_order_relaxed)) {
    return;
  }
  iree_atomic_slist_push(&tracker->proactor->pending_semaphore_waits,
                         &tracker->slist_entry);
  iree_async_proactor_posix_wake_poll_thread(tracker->proactor);
}

// Timepoint callback for SEMAPHORE_WAIT operations.
// Called under the semaphore's internal lock — must be fast and non-blocking.
// Decodes the tracker and index from user_data, then tracks ALL/ANY
// satisfaction. When the wait is complete, pushes the tracker to the
// pending_semaphore_waits MPSC slist for the poll thread to drain.
static void iree_async_proactor_posix_semaphore_wait_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  uintptr_t encoded = (uintptr_t)user_data;
  iree_async_posix_semaphore_wait_tracker_t* tracker =
      (iree_async_posix_semaphore_wait_tracker_t*)(encoded &
                                                   0x00FFFFFFFFFFFFFFull);
  iree_host_size_t index = (iree_host_size_t)(encoded >> 56);

  if (!iree_status_is_ok(status)) {
    // Failure or cancellation. Store the error status (first one wins).
    intptr_t expected = (intptr_t)iree_ok_status();
    if (!iree_atomic_compare_exchange_strong(
            &tracker->completion_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      iree_status_ignore(status);
    }
    // Enqueue for completion regardless of whether we won the status race.
    iree_async_proactor_posix_semaphore_wait_enqueue_completion(tracker);
    return;
  }

  if (tracker->operation->mode == IREE_ASYNC_WAIT_MODE_ANY) {
    // ANY mode: first satisfied index wins via CAS.
    int32_t expected = -1;
    if (iree_atomic_compare_exchange_strong(
            &tracker->remaining_or_satisfied, &expected, (int32_t)index,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      iree_async_proactor_posix_semaphore_wait_enqueue_completion(tracker);
    }
  } else {
    // ALL mode: decrement remaining count.
    int32_t remaining = iree_atomic_fetch_sub(&tracker->remaining_or_satisfied,
                                              1, iree_memory_order_acq_rel) -
                        1;
    if (remaining == 0) {
      iree_async_proactor_posix_semaphore_wait_enqueue_completion(tracker);
    }
  }
}

// Submits a SEMAPHORE_WAIT by checking for immediate satisfaction, then
// allocating a tracker with embedded timepoints and registering callbacks
// on each semaphore. The timepoint callbacks fire from the signaling thread
// and push the tracker to the pending_semaphore_waits MPSC slist, which is
// drained by the poll thread.
static iree_status_t iree_async_proactor_posix_submit_semaphore_wait(
    iree_async_proactor_posix_t* proactor,
    iree_async_semaphore_wait_operation_t* wait_op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The timepoint callback encodes {tracker_pointer, semaphore_index} in a
  // single pointer using a 56-bit/8-bit split. The 8-bit index field limits
  // the maximum number of semaphores per wait to 255.
  if (IREE_UNLIKELY(wait_op->count > 255)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "semaphore wait count %" PRIhsz
        " exceeds the maximum of 255 (limited by timepoint user_data encoding)",
        wait_op->count);
  }

  // Check for immediate satisfaction before allocating a tracker.
  bool immediately_satisfied = false;
  bool all_satisfied = true;
  for (iree_host_size_t i = 0; i < wait_op->count; ++i) {
    uint64_t current = iree_async_semaphore_query(wait_op->semaphores[i]);
    if (current >= wait_op->values[i]) {
      if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ANY) {
        wait_op->satisfied_index = i;
        immediately_satisfied = true;
        break;
      }
    } else {
      all_satisfied = false;
    }
  }
  if (!immediately_satisfied && all_satisfied) {
    immediately_satisfied = true;
  }
  if (immediately_satisfied) {
    IREE_TRACE_ZONE_END(z0);
    return iree_async_proactor_posix_complete_on_submit(
        proactor, &wait_op->base, iree_ok_status(),
        IREE_ASYNC_COMPLETION_FLAG_NONE);
  }

  // Calculate allocation size with overflow checking.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              sizeof(iree_async_posix_semaphore_wait_tracker_t), &total_size,
              IREE_STRUCT_FIELD_FAM(wait_op->count,
                                    iree_async_semaphore_timepoint_t)));

  // Allocate tracker with embedded timepoints.
  iree_async_posix_semaphore_wait_tracker_t* tracker = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->base.allocator, total_size,
                                (void**)&tracker));
  memset(tracker, 0, total_size);

  tracker->operation = wait_op;
  tracker->proactor = proactor;
  tracker->allocator = proactor->base.allocator;
  tracker->count = wait_op->count;
  tracker->registered_count = 0;
  iree_atomic_store(&tracker->completion_status, (intptr_t)iree_ok_status(),
                    iree_memory_order_release);

  // Transfer LINKED continuation chain from operation to tracker.
  tracker->continuation_head = wait_op->base.linked_next;
  wait_op->base.linked_next = NULL;

  if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ALL) {
    iree_atomic_store(&tracker->remaining_or_satisfied, (int32_t)wait_op->count,
                      iree_memory_order_release);
  } else {
    // ANY mode: -1 indicates not yet satisfied.
    iree_atomic_store(&tracker->remaining_or_satisfied, -1,
                      iree_memory_order_release);
  }

  // Store tracker in operation for cancel to find it.
  wait_op->base.next = (iree_async_operation_t*)tracker;

  // Register timepoints for each semaphore. Use status chaining so that
  // on failure we cancel only the successfully registered timepoints.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < wait_op->count && iree_status_is_ok(status);
       ++i) {
    iree_async_semaphore_timepoint_t* timepoint = &tracker->timepoints[i];
    timepoint->callback =
        iree_async_proactor_posix_semaphore_wait_timepoint_callback;
    // Encode tracker pointer + index in user_data using a 56-bit/8-bit split
    // for LA57 (5-level paging) safety: userspace pointers use at most 56 bits
    // on x86-64, leaving 8 bits for the index (max 255).
    // The count <= 255 check above guarantees this shift is safe.
    timepoint->user_data = (void*)((uintptr_t)tracker | ((uintptr_t)i << 56));
    status = iree_async_semaphore_acquire_timepoint(
        wait_op->semaphores[i], wait_op->values[i], timepoint);
    if (iree_status_is_ok(status)) {
      tracker->registered_count = i + 1;
    }
  }

  // Single exit point: on failure, cancel registered timepoints and free.
  if (!iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < tracker->registered_count; ++i) {
      iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                            &tracker->timepoints[i]);
    }
    wait_op->base.next = NULL;
    iree_allocator_free(tracker->allocator, tracker);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Submits a MESSAGE operation: delivers payload to the target proactor's
// message pool and optionally generates a source completion.
// The target must be a POSIX proactor (same backend) since the message pool
// is on the backend-specific struct. Cross-backend messaging requires the
// simple send_message API instead.
static iree_status_t iree_async_proactor_posix_submit_message(
    iree_async_proactor_posix_t* proactor,
    iree_async_message_operation_t* message) {
  iree_async_proactor_t* target = message->target;
  if (!target) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "MESSAGE target proactor is NULL");
  }
  if (target->vtable != &iree_async_proactor_posix_vtable) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "MESSAGE target must be a POSIX proactor from the same backend; "
        "cross-backend messaging is not supported via operations");
  }
  iree_async_proactor_posix_t* target_posix =
      iree_async_proactor_posix_cast(target);

  // Deliver message to the target proactor's pool and wake its poll thread.
  iree_status_t send_status = iree_async_message_pool_send(
      &target_posix->message_pool, message->message_data);
  if (iree_status_is_ok(send_status)) {
    target->vtable->wake(target);
  }

  // Handle source-side completion.
  if (iree_any_bit_set(message->message_flags,
                       IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION)) {
    // Fire-and-forget: no source completion callback. Dispatch linked
    // continuation directly with the send status.
    iree_async_proactor_posix_dispatch_linked_continuation(
        proactor, &message->base, send_status);
    iree_status_ignore(send_status);
    return iree_ok_status();
  }

  // Push source completion through the completion queue so the caller's
  // callback fires from poll() (consistent with all other operation types).
  return iree_async_proactor_posix_complete_on_submit(
      proactor, &message->base, send_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
}

// Materializes iovecs from a span list at submit time so that the execute path
// does not need to dereference buffers.values (which may point to caller stack
// storage that is no longer valid by execution time).
static void iree_async_proactor_posix_materialize_iovecs(
    struct iovec* iovecs, const iree_async_span_list_t* buffers) {
  for (iree_host_size_t i = 0; i < buffers->count; ++i) {
    iovecs[i].iov_base = iree_async_span_ptr(buffers->values[i]);
    iovecs[i].iov_len = buffers->values[i].length;
  }
}

static iree_status_t iree_async_proactor_posix_execute_send(
    iree_async_socket_send_operation_t* send_op,
    iree_async_io_result_t* out_result);
static iree_status_t iree_async_proactor_posix_execute_sendto(
    iree_async_socket_sendto_operation_t* sendto_op,
    iree_async_io_result_t* out_result);

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

// Dispatches a single operation to the appropriate backend handler.
// Clears internal_flags for fresh submission. Does not build linked_next chains
// or wake the poll thread — callers handle those.
static iree_status_t iree_async_proactor_posix_submit_operation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation) {
  iree_async_operation_clear_internal_flags(operation);
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP:
      return iree_async_proactor_posix_complete_on_submit(
          proactor, operation, iree_ok_status(),
          IREE_ASYNC_COMPLETION_FLAG_NONE);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return iree_async_proactor_posix_submit_socket_accept(
          proactor, (iree_async_socket_accept_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv_op =
          (iree_async_socket_recv_operation_t*)operation;
      iree_async_proactor_posix_materialize_iovecs(
          (struct iovec*)recv_op->platform.posix.iovecs, &recv_op->buffers);
      iree_async_proactor_posix_push_pending(proactor, operation);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      // Eager send: attempt writev during submit as a latency optimization.
      // When the socket buffer has space, data is consumed immediately
      // without a poll-thread round-trip. On EAGAIN (buffer full, zero bytes
      // consumed), defer to the poll loop for POLLOUT-driven retry, matching
      // io_uring's internal async-poll behavior. Buffer data remains valid
      // until the completion callback (async contract).
      iree_async_socket_send_operation_t* send_op =
          (iree_async_socket_send_operation_t*)operation;
      // Fail immediately if the socket already has a sticky failure (e.g.,
      // ECONNRESET from a prior recv). Without this, eager writev() deposits
      // data in the kernel buffer and reports success even on broken
      // connections, bypassing error detection entirely.
      iree_status_t send_failure =
          iree_async_socket_query_failure(send_op->socket);
      if (!iree_status_is_ok(send_failure)) {
        return iree_async_proactor_posix_complete_on_submit(
            proactor, operation, iree_status_clone(send_failure),
            IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
      iree_async_proactor_posix_materialize_iovecs(
          (struct iovec*)send_op->platform.posix.iovecs, &send_op->buffers);
      iree_async_io_result_t send_result = IREE_ASYNC_IO_COMPLETE;
      iree_status_t send_status =
          iree_async_proactor_posix_execute_send(send_op, &send_result);
      if (send_result == IREE_ASYNC_IO_WOULD_BLOCK) {
        iree_async_proactor_posix_push_pending(proactor, operation);
        return iree_ok_status();
      }
      if (!iree_status_is_ok(send_status)) {
        iree_async_socket_set_failure(send_op->socket, send_status);
      }
      return iree_async_proactor_posix_complete_on_submit(
          proactor, operation, send_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      iree_async_socket_recvfrom_operation_t* recvfrom_op =
          (iree_async_socket_recvfrom_operation_t*)operation;
      iree_async_proactor_posix_materialize_iovecs(
          (struct iovec*)recvfrom_op->platform.posix.iovecs,
          &recvfrom_op->buffers);
      iree_async_proactor_posix_push_pending(proactor, operation);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      // Eager sendto: same latency optimization as SOCKET_SEND above.
      // On EAGAIN, defer to poll loop for POLLOUT-driven retry.
      iree_async_socket_sendto_operation_t* sendto_op =
          (iree_async_socket_sendto_operation_t*)operation;
      iree_status_t sendto_failure =
          iree_async_socket_query_failure(sendto_op->socket);
      if (!iree_status_is_ok(sendto_failure)) {
        return iree_async_proactor_posix_complete_on_submit(
            proactor, operation, iree_status_clone(sendto_failure),
            IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
      iree_async_proactor_posix_materialize_iovecs(
          (struct iovec*)sendto_op->platform.posix.iovecs, &sendto_op->buffers);
      iree_async_io_result_t sendto_result = IREE_ASYNC_IO_COMPLETE;
      iree_status_t sendto_status =
          iree_async_proactor_posix_execute_sendto(sendto_op, &sendto_result);
      if (sendto_result == IREE_ASYNC_IO_WOULD_BLOCK) {
        iree_async_proactor_posix_push_pending(proactor, operation);
        return iree_ok_status();
      }
      if (!iree_status_is_ok(sendto_status)) {
        iree_async_socket_set_failure(sendto_op->socket, sendto_status);
      }
      return iree_async_proactor_posix_complete_on_submit(
          proactor, operation, sendto_status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return iree_async_proactor_posix_submit_recv_pool(
          proactor, (iree_async_socket_recv_pool_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return iree_async_proactor_posix_submit_socket_connect(
          proactor, (iree_async_socket_connect_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
      return iree_async_proactor_posix_submit_socket_close(
          proactor, (iree_async_socket_close_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_TIMER:
      return iree_async_proactor_posix_submit_timer(
          proactor, (iree_async_timer_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
      return iree_async_proactor_posix_submit_event_wait(
          proactor, (iree_async_event_wait_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT:
      return iree_async_proactor_posix_submit_notification_wait(
          proactor, (iree_async_notification_wait_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL:
      return iree_async_proactor_posix_submit_notification_signal(
          proactor, (iree_async_notification_signal_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT:
      return iree_async_proactor_posix_submit_semaphore_wait(
          proactor, (iree_async_semaphore_wait_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL:
      return iree_async_proactor_posix_submit_semaphore_signal(
          proactor, (iree_async_semaphore_signal_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      iree_async_proactor_posix_push_pending(proactor, operation);
      return iree_ok_status();

    case IREE_ASYNC_OPERATION_TYPE_MESSAGE:
      return iree_async_proactor_posix_submit_message(
          proactor, (iree_async_message_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE: {
      iree_async_sequence_operation_t* sequence =
          (iree_async_sequence_operation_t*)operation;
      if (!sequence->step_fn) {
        return iree_async_sequence_submit_as_linked(&proactor->base, sequence);
      } else {
        return iree_async_sequence_emulation_begin(&proactor->sequence_emulator,
                                                   sequence);
      }
    }

    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "POSIX proactor does not yet support operation type %d",
          (int)operation->type);
  }
}

static iree_status_t iree_async_proactor_posix_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);

  if (iree_atomic_load(&proactor->shutdown_requested,
                       iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_ABORTED, "proactor is shutting down");
  }

  // Build linked_next chains from LINKED flags and validate.
  // Operations with LINKED flag point to the next operation in the batch.
  // Only "chain heads" (operations not preceded by a LINKED operation) are
  // submitted to their backends; continuations stay in linked_next and are
  // dispatched when the predecessor completes.
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    operation->linked_next = NULL;
    if (!iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }
    // LINKED on last operation is a contract violation.
    if (i + 1 >= operations.count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LINKED flag set on last operation in batch (no successor)");
    }
    operation->linked_next = operations.values[i + 1];
  }

  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];

    // Skip continuation operations — they are held in the predecessor's
    // linked_next and will be submitted when it completes.
    if (i > 0 && iree_any_bit_set(operations.values[i - 1]->flags,
                                  IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }

    IREE_RETURN_IF_ERROR(
        iree_async_proactor_posix_submit_operation(proactor, operation));
  }

  if (operations.count > 0) {
    iree_async_proactor_posix_wake_poll_thread(proactor);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// LINKED chain dispatch helpers
//===----------------------------------------------------------------------===//

// Cancels a continuation chain by directly invoking callbacks with CANCELLED.
// Cancelled continuations were never submitted, so no resources were retained —
// we bypass the completion queue (which would trigger
// release_operation_resources on unretained resources). Returns the number of
// callbacks invoked (for completion counting by callers).
static iree_host_size_t iree_async_proactor_posix_cancel_continuation_chain(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* chain_head) {
  if (!chain_head) return 0;
  (void)proactor;
  iree_host_size_t cancelled_count = 0;
  iree_async_operation_t* op = chain_head;
  while (op) {
    iree_async_operation_t* next = op->linked_next;
    op->linked_next = NULL;
    if (op->completion_fn) {
      op->completion_fn(op->user_data, op,
                        iree_status_from_code(IREE_STATUS_CANCELLED),
                        IREE_ASYNC_COMPLETION_FLAG_NONE);
      ++cancelled_count;
    }
    op = next;
  }
  return cancelled_count;
}

// Submits a continuation chain from a linked_next pointer.
// Only the chain head is dispatched here; when it completes,
// dispatch_linked_continuation cascades to the next operation.
// On submit failure, fires the chain head's callback with the error and
// cancels remaining continuations.
static void iree_async_proactor_posix_submit_continuation_chain(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* chain_head) {
  if (!chain_head) return;

  iree_status_t status =
      iree_async_proactor_posix_submit_operation(proactor, chain_head);
  if (iree_status_is_ok(status)) {
    // The operation was registered (pending_queue, completion_queue, etc.).
    // Wake the poll thread so it picks up the new work rather than blocking
    // in event_set_wait.
    iree_async_proactor_posix_wake_poll_thread(proactor);
    return;
  }

  // Chain head submit failed. Fire its callback with the error, then cancel
  // remaining continuations (they were never submitted).
  iree_async_operation_t* continuation = chain_head->linked_next;
  chain_head->linked_next = NULL;
  if (chain_head->completion_fn) {
    chain_head->completion_fn(chain_head->user_data, chain_head, status,
                              IREE_ASYNC_COMPLETION_FLAG_NONE);
  } else {
    iree_status_ignore(status);
  }
  iree_async_proactor_posix_cancel_continuation_chain(proactor, continuation);
}

// Dispatches a linked_next continuation chain based on the trigger's status.
// On success: submits the chain for execution (completions counted via queue).
// On failure: cancels the chain by directly invoking callbacks with CANCELLED.
// Returns the number of directly-invoked callbacks (for completion counting).
static iree_host_size_t iree_async_proactor_posix_dispatch_linked_continuation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    iree_status_t trigger_status) {
  iree_async_operation_t* continuation = operation->linked_next;
  if (!continuation) return 0;

  // Detach the chain before potentially recursive submit.
  operation->linked_next = NULL;

  if (iree_status_is_ok(trigger_status)) {
    iree_async_proactor_posix_submit_continuation_chain(proactor, continuation);
    return 0;  // Submitted ops produce completions counted by the drain loop.
  } else {
    return iree_async_proactor_posix_cancel_continuation_chain(proactor,
                                                               continuation);
  }
}

//===----------------------------------------------------------------------===//
// Timer helpers
//===----------------------------------------------------------------------===//

// Calculates the poll timeout considering pending timers.
// Returns the timeout in milliseconds, clamped to avoid overflow.
static int iree_async_proactor_posix_calculate_timeout_ms(
    iree_async_proactor_posix_t* proactor, iree_timeout_t user_timeout) {
  // Start with user-requested timeout.
  int timeout_ms = 0;
  if (iree_timeout_is_immediate(user_timeout)) {
    timeout_ms = 0;
  } else if (iree_timeout_is_infinite(user_timeout)) {
    timeout_ms = -1;
  } else {
    iree_duration_t remaining = iree_timeout_as_duration_ns(user_timeout);
    // Clamp to INT_MAX (24 days) to avoid overflow.
    int64_t remaining_ms = remaining / 1000000;
    if (remaining_ms > INT_MAX) remaining_ms = INT_MAX;
    timeout_ms = (int)remaining_ms;
    if (timeout_ms == 0 && remaining > 0) timeout_ms = 1;
  }

  // If we have pending timers, reduce timeout to fire them on time.
  iree_time_t earliest_deadline =
      iree_async_posix_timer_list_next_deadline_ns(&proactor->timers);
  if (earliest_deadline != IREE_TIME_INFINITE_FUTURE) {
    iree_time_t now = iree_time_now();
    if (earliest_deadline <= now) {
      // Timer already expired - immediate poll.
      timeout_ms = 0;
    } else {
      // Calculate time until next timer.
      int64_t delta_ns = earliest_deadline - now;
      int64_t delta_ms = delta_ns / 1000000;
      if (delta_ms > INT_MAX) delta_ms = INT_MAX;
      if (delta_ms < 0) delta_ms = 0;

      // Use the smaller of user timeout and timer timeout.
      if (timeout_ms < 0 || (int)delta_ms < timeout_ms) {
        timeout_ms = (int)delta_ms;
      }
    }
  }

  return timeout_ms;
}

// Processes expired timers by pushing completions to the queue.
// Uses remove-then-queue pattern with cancelled flag check for safety.
// Returns the number of timer completions pushed to the queue.
static iree_host_size_t iree_async_proactor_posix_process_expired_timers(
    iree_async_proactor_posix_t* proactor) {
  iree_time_t now = iree_time_now();
  iree_host_size_t timer_count = 0;

  iree_async_timer_operation_t* timer = NULL;
  while ((timer = iree_async_posix_timer_list_pop_expired(&proactor->timers,
                                                          now)) != NULL) {
    // Timer has been removed from list. Check cancelled flag — may have been
    // set by cancel() from another thread or by an earlier callback in this
    // loop.
    iree_status_t status = iree_ok_status();
    if (iree_any_bit_set(iree_async_operation_load_internal_flags(&timer->base),
                         IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED)) {
      status = iree_status_from_code(IREE_STATUS_CANCELLED);
      // cancel() incremented the timer cancellation counter. Decrement it
      // since we're handling the timer via expiration rather than the
      // cancellation scan.
      iree_atomic_fetch_sub(&proactor->pending_timer_cancellation_count, 1,
                            iree_memory_order_release);
    }

    // Push to completion queue (consistent with socket ops).
    iree_async_posix_completion_t* completion =
        iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
    if (completion) {
      completion->operation = &timer->base;
      completion->status = status;
      completion->flags = IREE_ASYNC_COMPLETION_FLAG_NONE;
      iree_atomic_slist_push(&proactor->completion_queue,
                             &completion->slist_entry);
      ++timer_count;
    } else {
      // Pool exhausted — dispatch directly (same as drain_completion_queue).
      iree_async_proactor_posix_dispatch_linked_continuation(
          proactor, &timer->base, status);
      iree_async_operation_release_resources(&timer->base);
      if (timer->base.completion_fn) {
        timer->base.completion_fn(timer->base.user_data, &timer->base, status,
                                  IREE_ASYNC_COMPLETION_FLAG_NONE);
      } else {
        iree_status_ignore(status);
      }
    }
  }

  return timer_count;
}

//===----------------------------------------------------------------------===//
// Completion queue helpers
//===----------------------------------------------------------------------===//

// Returns true if a successful multishot operation has reached a terminal state
// and should not re-arm. Non-terminal operations continue receiving with MORE.
//
// Terminal conditions by type:
//   SOCKET_ACCEPT: never terminal — each accept produces a new connection.
//   SOCKET_RECV: terminal on EOF (0 bytes received).
//   SOCKET_RECV_POOL: terminal on EOF (0 bytes received).
//   All others: terminal (unknown multishot type; be safe).
static bool iree_async_proactor_posix_is_multishot_terminal(
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return false;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      return recv->bytes_received == 0;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      iree_async_socket_recv_pool_operation_t* recv_pool =
          (iree_async_socket_recv_pool_operation_t*)operation;
      return recv_pool->bytes_received == 0;
    }
    default:
      return true;
  }
}

// Drains the completion queue, invoking callbacks for each entry.
// Returns the number of completions processed.
static iree_host_size_t iree_async_proactor_posix_drain_completion_queue(
    iree_async_proactor_posix_t* proactor) {
  // Flush the entire completion queue in approximate FIFO order. The slist is
  // a LIFO stack (push/pop), so without flushing in FIFO order completions
  // would be delivered in reverse submission order — breaking ordering
  // guarantees that callers (and io_uring) provide.
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->completion_queue,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;
  }

  iree_host_size_t count = 0;
  iree_atomic_slist_entry_t* entry = head;
  while (entry != NULL) {
    iree_atomic_slist_entry_t* next = entry->next;
    iree_async_posix_completion_t* completion =
        iree_containerof(entry, iree_async_posix_completion_t, slist_entry);

    iree_async_operation_t* operation = completion->operation;
    iree_status_t status = completion->status;
    iree_async_completion_flags_t flags = completion->flags;

    completion->status = iree_ok_status();
    iree_async_posix_completion_pool_release(&proactor->completion_pool,
                                             completion);

    // Dispatch LINKED continuation chain (if any) before invoking callback.
    // On success: submits the continuation chain for execution.
    // On failure: cancels the chain (directly invokes callbacks with
    // CANCELLED). Cancelled continuations bypass the completion queue, so their
    // count is added directly here.
    if (operation) {
      count += iree_async_proactor_posix_dispatch_linked_continuation(
          proactor, operation, status);
    }

    // Release resources retained during submission for final completions.
    // Must happen BEFORE the callback since the callback may free the
    // operation. Multishot operations (MORE flag set) keep their retained
    // resources across intermediate completions.
    if (operation &&
        !iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
      iree_async_operation_release_resources(operation);
    }

    if (operation && operation->completion_fn) {
      operation->completion_fn(operation->user_data, operation, status, flags);
    } else {
      iree_status_ignore(status);
    }

    ++count;
    entry = next;
  }
  return count;
}

// Drains incoming messages from this proactor's message pool and invokes the
// registered message callback for each. Other proactors (or threads using the
// send_message API) push entries to the pool; this function consumes them.
// Called from the poll thread only.
static void iree_async_proactor_posix_drain_incoming_messages(
    iree_async_proactor_posix_t* proactor) {
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

// Drains pending semaphore wait completions and invokes callbacks.
// Called from poll() after processing fds. Semaphore timepoint callbacks fire
// from arbitrary threads and push trackers to the pending_semaphore_waits MPSC
// slist; this function processes those trackers on the poll thread.
static iree_host_size_t iree_async_proactor_posix_drain_pending_semaphore_waits(
    iree_async_proactor_posix_t* proactor) {
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
    iree_async_posix_semaphore_wait_tracker_t* tracker =
        (iree_async_posix_semaphore_wait_tracker_t*)entry;
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
    // On success: submits the chain for execution (completions counted via
    //   the completion queue drain).
    // On failure: cancels the chain (directly invokes callbacks with
    // CANCELLED).
    //   Cancelled continuations bypass the completion queue, so their count is
    //   added directly here.
    if (tracker->continuation_head) {
      iree_async_operation_t* continuation = tracker->continuation_head;
      tracker->continuation_head = NULL;
      if (iree_status_is_ok(status)) {
        iree_async_proactor_posix_submit_continuation_chain(proactor,
                                                            continuation);
      } else {
        drained_count += iree_async_proactor_posix_cancel_continuation_chain(
            proactor, continuation);
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
// Vtable implementations
//===----------------------------------------------------------------------===//

static iree_async_proactor_capabilities_t
iree_async_proactor_posix_query_capabilities(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  return proactor->capabilities;
}

// Executes a socket accept operation when the listen fd is ready.
static iree_status_t iree_async_proactor_posix_execute_accept(
    iree_async_proactor_posix_t* proactor,
    iree_async_socket_accept_operation_t* accept_op,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  struct sockaddr_storage addr;
  socklen_t addr_len = sizeof(addr);
  int accepted_fd =
      iree_posix_accept(accept_op->listen_socket->primitive.value.fd,
                        (struct sockaddr*)&addr, &addr_len);
  if (accepted_fd < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      *out_result = IREE_ASYNC_IO_WOULD_BLOCK;
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "accept() failed");
  }

  // Create socket for the accepted connection.
  iree_async_socket_t* accepted_socket = NULL;
  iree_status_t status = iree_async_posix_socket_create_accepted(
      proactor, accepted_fd, accept_op->listen_socket->type, &accepted_socket);
  if (!iree_status_is_ok(status)) {
    close(accepted_fd);
    return status;
  }

  accept_op->accepted_socket = accepted_socket;
  memcpy(accept_op->peer_address.storage, &addr, addr_len);
  accept_op->peer_address.length = addr_len;

  return iree_ok_status();
}

// Executes a socket connect completion check when the fd is ready.
static iree_status_t iree_async_proactor_posix_execute_connect(
    iree_async_socket_connect_operation_t* connect_op,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  int error = 0;
  socklen_t len = sizeof(error);
  if (getsockopt(connect_op->socket->primitive.value.fd, SOL_SOCKET, SO_ERROR,
                 &error, &len) < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "getsockopt(SO_ERROR) failed");
  }
  if (error != 0) {
    return iree_make_status(iree_status_code_from_errno(error),
                            "connect() failed asynchronously");
  }
  connect_op->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
  return iree_ok_status();
}

// Executes a socket recv operation when the fd is ready.
static iree_status_t iree_async_proactor_posix_execute_recv(
    iree_async_socket_recv_operation_t* recv_op,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  struct iovec* iovecs = (struct iovec*)recv_op->platform.posix.iovecs;
  ssize_t result = readv(recv_op->socket->primitive.value.fd, iovecs,
                         recv_op->buffers.count);
  if (result < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      *out_result = IREE_ASYNC_IO_WOULD_BLOCK;
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "readv() failed");
  }

  recv_op->bytes_received = (iree_host_size_t)result;
  return iree_ok_status();
}

// Executes a socket send operation when the fd is ready.
static iree_status_t iree_async_proactor_posix_execute_send(
    iree_async_socket_send_operation_t* send_op,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  int fd = send_op->socket->primitive.value.fd;
  struct iovec* iovecs = (struct iovec*)send_op->platform.posix.iovecs;
  int iovec_count = (int)send_op->buffers.count;

  ssize_t result = writev(fd, iovecs, iovec_count);
  if (result < 0) {
    int error = errno;
    if (error == EAGAIN || error == EWOULDBLOCK) {
      *out_result = IREE_ASYNC_IO_WOULD_BLOCK;
      return iree_ok_status();
    }
    uint64_t total_requested = 0;
    for (int i = 0; i < iovec_count; ++i) {
      total_requested += iovecs[i].iov_len;
    }
    return iree_make_status(iree_status_code_from_errno(error),
                            "writev(fd=%d, count=%d, total=%" PRIu64
                            ") failed with errno=%d",
                            fd, iovec_count, total_requested, error);
  }

  send_op->bytes_sent = (iree_host_size_t)result;
  return iree_ok_status();
}

// Executes a SOCKET_RECV_POOL operation when the socket fd is readable.
// Acquires a buffer from the pool, performs readv into it, and populates the
// operation's lease and bytes_received on success. On EAGAIN the lease is
// released back to the pool and UNAVAILABLE is returned for re-arm. On pool
// exhaustion, returns the error which terminates multishot (safe because pool
// exhaustion cannot occur in normal multishot flow — callbacks drain between
// poll iterations, releasing buffers before the next execution).
static iree_status_t iree_async_proactor_posix_execute_recv_pool(
    iree_async_socket_recv_pool_operation_t* recv_pool,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;

  // Acquire a buffer from the pool.
  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  iree_status_t acquire_status =
      iree_async_buffer_pool_acquire(recv_pool->pool, &lease);
  if (!iree_status_is_ok(acquire_status)) {
    return acquire_status;
  }

  // Build iovec from the leased span and read into it.
  struct iovec iov = {
      .iov_base = iree_async_span_ptr(lease.span),
      .iov_len = lease.span.length,
  };
  ssize_t result = readv(recv_pool->socket->primitive.value.fd, &iov, 1);
  if (result < 0) {
    iree_async_buffer_lease_release(&lease);
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      *out_result = IREE_ASYNC_IO_WOULD_BLOCK;
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "readv() failed");
  }

  // Success: populate the operation's output fields.
  recv_pool->lease = lease;
  recv_pool->bytes_received = (iree_host_size_t)result;
  return iree_ok_status();
}

// Executes a socket recvfrom operation when the fd is readable.
// Uses recvfrom() to receive data and capture the sender's address.
static iree_status_t iree_async_proactor_posix_execute_recvfrom(
    iree_async_socket_recvfrom_operation_t* recvfrom_op,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  struct iovec* iovecs = (struct iovec*)recvfrom_op->platform.posix.iovecs;
  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_name = recvfrom_op->sender.storage;
  msg.msg_namelen = sizeof(recvfrom_op->sender.storage);
  msg.msg_iov = iovecs;
  msg.msg_iovlen = recvfrom_op->buffers.count;

  ssize_t result = recvmsg(recvfrom_op->socket->primitive.value.fd, &msg, 0);
  if (result < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      *out_result = IREE_ASYNC_IO_WOULD_BLOCK;
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "recvmsg() failed");
  }

  recvfrom_op->sender.length = (iree_host_size_t)msg.msg_namelen;
  recvfrom_op->bytes_received = (iree_host_size_t)result;
  return iree_ok_status();
}

// Executes a socket sendto operation when the fd is writable.
// Uses sendto() to send data to a specified destination address.
static iree_status_t iree_async_proactor_posix_execute_sendto(
    iree_async_socket_sendto_operation_t* sendto_op,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  struct iovec* iovecs = (struct iovec*)sendto_op->platform.posix.iovecs;
  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_name = (void*)sendto_op->destination.storage;
  msg.msg_namelen = (socklen_t)sendto_op->destination.length;
  msg.msg_iov = iovecs;
  msg.msg_iovlen = sendto_op->buffers.count;

  ssize_t result =
      sendmsg(sendto_op->socket->primitive.value.fd, &msg, MSG_NOSIGNAL);
  if (result < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      *out_result = IREE_ASYNC_IO_WOULD_BLOCK;
      return iree_ok_status();
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "sendmsg() failed");
  }

  sendto_op->bytes_sent = (iree_host_size_t)result;
  return iree_ok_status();
}

// Executes a socket operation when its fd is ready.
// POLLERR, POLLHUP, and POLLNVAL are not checked here because the individual
// I/O handlers produce better diagnostics from their system call errors:
//   POLLHUP + writev → EPIPE/ECONNRESET, readv → 0 (EOF) or error
//   POLLERR + any I/O → fails with the pending SO_ERROR
//   POLLNVAL + any I/O → EBADF
// All of these are non-EAGAIN results that complete the operation with a
// meaningful error rather than triggering an infinite re-registration loop.
static iree_status_t iree_async_proactor_posix_execute_socket_operation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    short revents, iree_async_io_result_t* out_result) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return iree_async_proactor_posix_execute_accept(
          proactor, (iree_async_socket_accept_operation_t*)operation,
          out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return iree_async_proactor_posix_execute_connect(
          (iree_async_socket_connect_operation_t*)operation, out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return iree_async_proactor_posix_execute_recv(
          (iree_async_socket_recv_operation_t*)operation, out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return iree_async_proactor_posix_execute_send(
          (iree_async_socket_send_operation_t*)operation, out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return iree_async_proactor_posix_execute_recv_pool(
          (iree_async_socket_recv_pool_operation_t*)operation, out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return iree_async_proactor_posix_execute_recvfrom(
          (iree_async_socket_recvfrom_operation_t*)operation, out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return iree_async_proactor_posix_execute_sendto(
          (iree_async_socket_sendto_operation_t*)operation, out_result);
    default:
      *out_result = IREE_ASYNC_IO_COMPLETE;
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported socket operation type %d",
                              (int)operation->type);
  }
}

// Executes an EVENT_WAIT operation when its fd is ready (POLLIN).
// Drains the eventfd/pipe to reset the event for reuse. The retained event
// reference is released later by release_operation_resources when the
// completion is drained (matching the io_uring pattern).
static iree_status_t iree_async_proactor_posix_execute_event_wait(
    iree_async_proactor_posix_t* proactor,
    iree_async_event_wait_operation_t* event_wait, short revents,
    iree_async_io_result_t* out_result) {
  *out_result = IREE_ASYNC_IO_COMPLETE;
  if (iree_any_bit_set(revents, POLLERR | POLLNVAL)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "event fd error (revents=0x%x)", (int)revents);
  }

  // Drain the eventfd/pipe to reset the event for future use.
  //   eventfd: single 8-byte read resets counter to 0.
  //   pipe: loop drains all accumulated bytes.
  // Both are non-blocking (eventfd has EFD_NONBLOCK, pipe has O_NONBLOCK).
  uint64_t drain_buffer = 0;
  while (read(event_wait->event->primitive.value.fd, &drain_buffer,
              sizeof(drain_buffer)) > 0) {
    // Keep draining.
  }
  // EAGAIN = fully drained. Other errors are non-fatal for drain.

  return iree_ok_status();
}

// Executes an fd-based operation when its fd is ready.
// Dispatches to the appropriate handler based on operation type.
static iree_status_t iree_async_proactor_posix_execute_fd_operation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation,
    short revents, iree_async_io_result_t* out_result) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
      return iree_async_proactor_posix_execute_event_wait(
          proactor, (iree_async_event_wait_operation_t*)operation, revents,
          out_result);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return iree_async_proactor_posix_execute_socket_operation(
          proactor, operation, revents, out_result);
    default:
      *out_result = IREE_ASYNC_IO_COMPLETE;
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unsupported fd operation type %d",
                              (int)operation->type);
  }
}

// Translates POSIX poll revents to the portable iree_async_poll_events_t.
static iree_async_poll_events_t iree_async_posix_translate_poll_events(
    short revents) {
  iree_async_poll_events_t events = IREE_ASYNC_POLL_EVENT_NONE;
  if (iree_any_bit_set(revents, POLLIN)) {
    events |= IREE_ASYNC_POLL_EVENT_IN;
  }
  if (iree_any_bit_set(revents, POLLERR)) {
    events |= IREE_ASYNC_POLL_EVENT_ERR;
  }
  if (iree_any_bit_set(revents, POLLHUP)) {
    events |= IREE_ASYNC_POLL_EVENT_HUP;
  }
  if (iree_any_bit_set(revents, POLLOUT)) {
    events |= IREE_ASYNC_POLL_EVENT_OUT;
  }
  return events;
}

// Returns the socket for a socket I/O operation that should propagate sticky
// failure on error. Returns NULL for non-socket operations and for types where
// errors don't indicate a broken socket (accept errors reflect transient
// resource issues, not a broken listener; close errors are moot).
static iree_async_socket_t* iree_async_proactor_posix_socket_from_io_operation(
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

// Processes an operation chain when its fd becomes ready.
//
// Walks the chain in FIFO order (head is oldest), executing each operation
// whose events match the ready events. Operations are processed until:
// - The chain is exhausted, or
// - An operation returns WOULD_BLOCK (socket buffer full/empty), in which case
//   remaining operations wanting the same events are skipped
//
// Completed operations are removed from the chain and pushed to the completion
// queue. Operations that would block stay in the chain for the next poll cycle.
//
// When the chain becomes empty, removes the fd from the event_set.
// When operations remain, updates the event_set if the combined events changed.
static void iree_async_proactor_posix_process_operation_chain(
    iree_async_proactor_posix_t* proactor, int fd, short revents) {
  // Look up the chain head.
  iree_async_posix_fd_handler_type_t handler_type;
  void* handler;
  if (!iree_async_posix_fd_map_lookup(&proactor->fd_map, fd, &handler_type,
                                      &handler)) {
    return;  // Already removed (stale event).
  }

  iree_async_operation_t* chain_head = (iree_async_operation_t*)handler;

  // Track which event types have gotten EAGAIN, so we skip remaining operations
  // wanting those events. For example, if a send gets EAGAIN (buffer full),
  // subsequent sends would also get EAGAIN.
  short blocked_events = 0;

  // Process operations in the chain. We maintain a "previous" pointer for
  // unlinking completed operations without searching the chain again.
  iree_async_operation_t* prev = NULL;
  iree_async_operation_t* current = chain_head;
  iree_async_operation_t* new_chain_head = chain_head;

  while (current != NULL) {
    iree_async_operation_t* next = current->next;
    short op_events = iree_async_operation_type_to_poll_events(current->type);

    // Skip operations whose events haven't fired.
    // POLLERR and POLLHUP are always delivered by poll() regardless of the
    // requested event mask. When either is set, all pending operations on the
    // fd must be attempted so their I/O syscalls can report the actual error
    // (e.g., getsockopt(SO_ERROR) for connect, EPIPE from writev for send).
    // Without this, macOS can deliver POLLERR alone for a connection-refused
    // and the connect handler would never fire.
    if (!(revents & (op_events | POLLERR | POLLHUP))) {
      prev = current;
      current = next;
      continue;
    }

    // Skip operations whose event type is blocked by a prior EAGAIN.
    if (blocked_events & op_events) {
      prev = current;
      current = next;
      continue;
    }

    // Check for cancellation.
    if (iree_any_bit_set(iree_async_operation_load_internal_flags(current),
                         IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED)) {
      // Unlink from chain.
      if (prev == NULL) {
        new_chain_head = next;
      } else {
        prev->next = next;
      }
      current->next = NULL;

      // Decrement the pending cancellation counter (incremented by cancel()).
      iree_atomic_fetch_sub(&proactor->pending_fd_cancellation_count, 1,
                            iree_memory_order_release);

      // Push cancellation completion.
      iree_status_t push_status =
          iree_async_proactor_posix_complete_immediately(
              proactor, current, iree_status_from_code(IREE_STATUS_CANCELLED),
              IREE_ASYNC_COMPLETION_FLAG_NONE);
      if (!iree_status_is_ok(push_status)) {
        iree_status_ignore(push_status);
        iree_async_operation_release_resources(current);
        iree_async_proactor_posix_dispatch_linked_continuation(
            proactor, current, iree_status_from_code(IREE_STATUS_CANCELLED));
        if (current->completion_fn) {
          current->completion_fn(current->user_data, current,
                                 iree_status_from_code(IREE_STATUS_CANCELLED),
                                 IREE_ASYNC_COMPLETION_FLAG_NONE);
        }
      }
      current = next;
      continue;
    }

    // Execute the operation.
    iree_async_io_result_t io_result = IREE_ASYNC_IO_COMPLETE;
    iree_status_t op_status = iree_async_proactor_posix_execute_fd_operation(
        proactor, current, revents, &io_result);

    // Handle would-block: operation stays in chain, block this event type so
    // we don't attempt further operations of the same kind until the next poll
    // cycle reports readiness again.
    if (io_result == IREE_ASYNC_IO_WOULD_BLOCK) {
      blocked_events |= op_events;
      prev = current;
      current = next;
      continue;
    }

    // Propagate error to socket's sticky failure status.
    if (!iree_status_is_ok(op_status)) {
      iree_async_socket_t* socket =
          iree_async_proactor_posix_socket_from_io_operation(current);
      if (socket) {
        iree_async_socket_set_failure(socket, op_status);
      }
    }

    // Operation completed (success or error). Unlink from chain.
    if (prev == NULL) {
      new_chain_head = next;
    } else {
      prev->next = next;
    }
    current->next = NULL;

    // Determine multishot continuation.
    iree_async_completion_flags_t completion_flags =
        IREE_ASYNC_COMPLETION_FLAG_NONE;
    iree_async_operation_t* completed_operation = current;
    if (iree_any_bit_set(current->flags, IREE_ASYNC_OPERATION_FLAG_MULTISHOT) &&
        iree_status_is_ok(op_status) &&
        !iree_async_proactor_posix_is_multishot_terminal(current)) {
      // Re-add to chain for next iteration.
      // Insert at the position we just removed from to maintain FIFO.
      current->next = next;
      if (prev == NULL) {
        new_chain_head = current;
      } else {
        prev->next = current;
      }
      completion_flags = IREE_ASYNC_COMPLETION_FLAG_MORE;
      prev = current;
      current = next;

      // Push completion with MORE flag.
      iree_async_posix_completion_t* completion =
          iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
      if (completion) {
        completion->operation = completed_operation;
        completion->status = op_status;
        completion->flags = completion_flags;
        iree_atomic_slist_push(&proactor->completion_queue,
                               &completion->slist_entry);
      } else {
        // Pool exhausted — dispatch directly. Multishot operations keep their
        // retained resources (no release_resources) and don't dispatch linked
        // continuations (those are for final completion only).
        if (completed_operation->completion_fn) {
          completed_operation->completion_fn(completed_operation->user_data,
                                             completed_operation, op_status,
                                             completion_flags);
        } else {
          iree_status_ignore(op_status);
        }
      }
      continue;
    }

    // Push completion.
    iree_async_posix_completion_t* completion =
        iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
    if (completion) {
      completion->operation = current;
      completion->status = op_status;
      completion->flags = completion_flags;
      iree_atomic_slist_push(&proactor->completion_queue,
                             &completion->slist_entry);
    } else {
      // Pool exhausted — dispatch directly (same pattern as cancel fallback).
      iree_async_proactor_posix_dispatch_linked_continuation(proactor, current,
                                                             op_status);
      iree_async_operation_release_resources(current);
      if (current->completion_fn) {
        current->completion_fn(current->user_data, current, op_status,
                               completion_flags);
      } else {
        iree_status_ignore(op_status);
      }
    }

    // Don't update prev - we removed current, so prev stays the same.
    current = next;
  }

  // Update fd_map with new chain head.
  if (new_chain_head == NULL) {
    // Chain is empty - remove fd from map and event_set.
    iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
    iree_status_ignore(
        iree_async_posix_event_set_remove(proactor->event_set, fd));
  } else if (new_chain_head != chain_head) {
    // Chain head changed - need to update fd_map.
    // The simplest way is remove and re-insert, but that's inefficient.
    // Instead, we directly update the handler pointer by looking up the entry.
    // Since we already did a lookup, we know the entry exists.
    // For now, do remove + insert (the common case is chain becoming empty).
    iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
    if (new_chain_head != NULL) {
      bool was_chained = false;
      iree_status_ignore(iree_async_posix_fd_map_insert_or_chain(
          &proactor->fd_map, fd, IREE_ASYNC_POSIX_FD_HANDLER_OPERATION,
          new_chain_head, &was_chained));
      // Update event_set to reflect remaining operations.
      short remaining_events =
          iree_async_proactor_posix_compute_chain_events(new_chain_head);
      iree_status_ignore(iree_async_posix_event_set_modify(
          proactor->event_set, fd, remaining_events));
    }
  } else {
    // Chain head unchanged but operations may have been removed from
    // middle/end. Update event_set if needed.
    short remaining_events =
        iree_async_proactor_posix_compute_chain_events(new_chain_head);
    iree_status_ignore(iree_async_posix_event_set_modify(proactor->event_set,
                                                         fd, remaining_events));
  }
}

// Processes pending notification waits, completing any whose epoch has advanced
// or whose cancellation flag is set. Drains the eventfd/pipe to reset for
// future signals. When the last waiter completes, unregisters the
// notification's fd from the event_set and removes it from the active list.
//
// Called from the poll thread when the notification's fd fires (POLLIN) and
// during the epoch scan after ready-fd processing.
static void iree_async_proactor_posix_process_notification_waits(
    iree_async_proactor_posix_t* proactor,
    iree_async_notification_t* notification) {
  // Drain the eventfd/pipe to reset for future signals.
  //   eventfd: single 8-byte read resets counter to 0.
  //   pipe: loop drains all accumulated bytes.
  // Returns EAGAIN immediately if already drained (e.g., by a sync waiter).
  uint64_t drain_buffer = 0;
  while (read(notification->platform.posix.primitive.value.fd, &drain_buffer,
              sizeof(drain_buffer)) > 0) {
  }

  uint32_t current_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);

  // Walk the pending wait list, completing waiters with advanced epoch or
  // cancellation flag set.
  iree_async_notification_wait_operation_t** previous =
      &notification->platform.posix.pending_waits;
  iree_async_notification_wait_operation_t* wait =
      notification->platform.posix.pending_waits;
  while (wait) {
    iree_async_notification_wait_operation_t* next =
        (iree_async_notification_wait_operation_t*)wait->base.next;

    bool cancelled =
        iree_any_bit_set(iree_async_operation_load_internal_flags(&wait->base),
                         IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED);
    bool epoch_advanced = (wait->wait_token != current_epoch);

    if (epoch_advanced || cancelled) {
      // Remove from the pending list.
      *previous = next;

      // Notification reference is released by release_operation_resources
      // when the completion is drained (or inline in the fallback path).

      iree_status_t status = cancelled
                                 ? iree_status_from_code(IREE_STATUS_CANCELLED)
                                 : iree_ok_status();

      // Push to completion queue.
      iree_async_posix_completion_t* completion =
          iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
      if (completion) {
        completion->operation = &wait->base;
        completion->status = status;
        completion->flags = IREE_ASYNC_COMPLETION_FLAG_NONE;
        iree_atomic_slist_push(&proactor->completion_queue,
                               &completion->slist_entry);
      } else {
        // Pool exhausted — release resources and dispatch directly.
        iree_async_operation_release_resources(&wait->base);
        iree_async_proactor_posix_dispatch_linked_continuation(
            proactor, &wait->base, status);
        if (wait->base.completion_fn) {
          wait->base.completion_fn(wait->base.user_data, &wait->base, status,
                                   IREE_ASYNC_COMPLETION_FLAG_NONE);
        } else {
          iree_status_ignore(status);
        }
      }
    } else {
      // Still waiting — advance the previous pointer.
      previous = (iree_async_notification_wait_operation_t**)&wait->base.next;
    }
    wait = next;
  }

  // If no consumers remain (no pending waits and no relay subscribers),
  // deactivate the notification's fd from event_set and fd_map.
  if (!notification->platform.posix.pending_waits &&
      !notification->platform.posix.relay_list) {
    int fd = notification->platform.posix.primitive.value.fd;
    iree_async_posix_fd_map_remove(&proactor->fd_map, fd);
    iree_status_ignore(
        iree_async_posix_event_set_remove(proactor->event_set, fd));
  }
}

// Scans the timer_list for timers with the CANCELLED flag set and processes
// them. Called from the poll thread when pending_timer_cancellation_count > 0.
//
// Without this scan, cancelled timers would linger in the timer_list until
// their original deadline expires — potentially minutes or hours. This mirrors
// the fd cancellation scan pattern for timely cancel completion.
static void iree_async_proactor_posix_drain_pending_timer_cancellations(
    iree_async_proactor_posix_t* proactor) {
  if (iree_atomic_load(&proactor->pending_timer_cancellation_count,
                       iree_memory_order_acquire) == 0) {
    return;
  }

  iree_async_timer_operation_t* timer = proactor->timers.head;
  while (timer != NULL) {
    // Capture next before potential removal.
    iree_async_timer_operation_t* next = timer->platform.posix.next;

    if (!iree_any_bit_set(
            iree_async_operation_load_internal_flags(&timer->base),
            IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED)) {
      timer = next;
      continue;
    }

    // Remove from the timer_list (O(1) with prev/next pointers).
    iree_async_posix_timer_list_remove(&proactor->timers, timer);

    // Push CANCELLED completion.
    iree_async_posix_completion_t* completion =
        iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
    if (completion) {
      completion->operation = &timer->base;
      completion->status = iree_status_from_code(IREE_STATUS_CANCELLED);
      completion->flags = IREE_ASYNC_COMPLETION_FLAG_NONE;
      iree_atomic_slist_push(&proactor->completion_queue,
                             &completion->slist_entry);
    } else {
      // Pool exhausted — dispatch directly (same as drain_completion_queue).
      iree_async_proactor_posix_dispatch_linked_continuation(
          proactor, &timer->base, iree_status_from_code(IREE_STATUS_CANCELLED));
      iree_async_operation_release_resources(&timer->base);
      if (timer->base.completion_fn) {
        timer->base.completion_fn(timer->base.user_data, &timer->base,
                                  iree_status_from_code(IREE_STATUS_CANCELLED),
                                  IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
    }

    iree_atomic_fetch_sub(&proactor->pending_timer_cancellation_count, 1,
                          iree_memory_order_release);

    if (iree_atomic_load(&proactor->pending_timer_cancellation_count,
                         iree_memory_order_acquire) == 0) {
      break;
    }

    timer = next;
  }
}

// Scans the fd_map for operations with the CANCELLED flag set and processes
// them. Called from the poll thread when pending_fd_cancellation_count > 0.
//
// This handles the case where cancel() is called on an fd-registered operation
// but the fd hasn't fired (e.g., a multishot accept on a listener with no
// pending connections). Without this scan, the cancellation would only be
// detected when the fd fires, which may never happen.
//
// For each cancelled operation found:
//   - Removes it from the fd_map chain
//   - Updates or removes the event_set registration
//   - Pushes a CANCELLED completion to the completion queue
//   - Decrements the pending cancellation counter
static void iree_async_proactor_posix_drain_pending_fd_cancellations(
    iree_async_proactor_posix_t* proactor) {
  if (iree_atomic_load(&proactor->pending_fd_cancellation_count,
                       iree_memory_order_acquire) == 0) {
    return;
  }

  iree_async_posix_fd_map_t* map = &proactor->fd_map;
  for (iree_host_size_t i = 0; i < map->bucket_count; ++i) {
    iree_async_posix_fd_map_entry_t* entry = &map->buckets[i];
    if (entry->fd == IREE_ASYNC_POSIX_FD_MAP_EMPTY ||
        entry->fd == IREE_ASYNC_POSIX_FD_MAP_TOMBSTONE) {
      continue;
    }
    if (entry->handler_type != IREE_ASYNC_POSIX_FD_HANDLER_OPERATION) {
      continue;
    }

    // Walk the operation chain for this fd, collecting cancelled operations.
    // We process cancelled ops by removing them from the chain one at a time.
    int fd = entry->fd;
    bool restart_chain;
    do {
      restart_chain = false;

      // Re-lookup the chain head (it may have changed after a removal).
      iree_async_posix_fd_handler_type_t handler_type;
      void* handler;
      if (!iree_async_posix_fd_map_lookup(map, fd, &handler_type, &handler)) {
        break;  // Entry was removed (chain became empty on a previous pass).
      }

      iree_async_operation_t* op = (iree_async_operation_t*)handler;
      while (op) {
        if (!iree_any_bit_set(iree_async_operation_load_internal_flags(op),
                              IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED)) {
          op = op->next;
          continue;
        }

        // Found a cancelled operation. Remove it from the chain.
        bool chain_empty =
            iree_async_posix_fd_map_remove_operation(map, fd, op);
        op->next = NULL;

        if (chain_empty) {
          iree_status_ignore(
              iree_async_posix_event_set_remove(proactor->event_set, fd));
        } else {
          // Recompute events for remaining operations.
          iree_async_posix_fd_handler_type_t remaining_type;
          void* remaining_handler;
          if (iree_async_posix_fd_map_lookup(map, fd, &remaining_type,
                                             &remaining_handler)) {
            short remaining_events =
                iree_async_proactor_posix_compute_chain_events(
                    (iree_async_operation_t*)remaining_handler);
            iree_status_ignore(iree_async_posix_event_set_modify(
                proactor->event_set, fd, remaining_events));
          }
        }

        // Push CANCELLED completion. Resources are released during the
        // completion drain (release_operation_resources).
        iree_status_t push_status =
            iree_async_proactor_posix_complete_immediately(
                proactor, op, iree_status_from_code(IREE_STATUS_CANCELLED),
                IREE_ASYNC_COMPLETION_FLAG_NONE);
        if (!iree_status_is_ok(push_status)) {
          iree_status_ignore(push_status);
          iree_async_operation_release_resources(op);
          iree_async_proactor_posix_dispatch_linked_continuation(
              proactor, op, iree_status_from_code(IREE_STATUS_CANCELLED));
          if (op->completion_fn) {
            op->completion_fn(op->user_data, op,
                              iree_status_from_code(IREE_STATUS_CANCELLED),
                              IREE_ASYNC_COMPLETION_FLAG_NONE);
          }
        }

        iree_atomic_fetch_sub(&proactor->pending_fd_cancellation_count, 1,
                              iree_memory_order_release);

        // Chain was modified — restart from the head to avoid stale pointers.
        restart_chain = !chain_empty;
        break;
      }
    } while (restart_chain);

    // Early exit if all pending cancellations have been processed.
    if (iree_atomic_load(&proactor->pending_fd_cancellation_count,
                         iree_memory_order_acquire) == 0) {
      break;
    }
  }
}

static iree_status_t iree_async_proactor_posix_poll(
    iree_async_proactor_t* base_proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);

  if (out_completed_count) *out_completed_count = 0;

  if (iree_atomic_load(&proactor->shutdown_requested,
                       iree_memory_order_acquire)) {
    return iree_make_status(IREE_STATUS_ABORTED, "proactor is shutting down");
  }

  // Drain pending_queue first: register newly submitted operations with
  // event_set/fd_map/timer_list. This is the only place where these
  // structures are mutated, ensuring thread-safety since poll() runs on
  // a single thread.
  iree_async_proactor_posix_drain_pending_queue(proactor);
  iree_async_proactor_posix_drain_pending_fence_imports(proactor);

  // Process pending cancellations — both timers (which would otherwise linger
  // until their deadline) and fd operations (whose fds may never fire).
  iree_async_proactor_posix_drain_pending_timer_cancellations(proactor);
  iree_async_proactor_posix_drain_pending_fd_cancellations(proactor);

  // Drain completions and invoke callbacks.
  iree_host_size_t completed_count =
      iree_async_proactor_posix_drain_completion_queue(proactor);
  completed_count +=
      iree_async_proactor_posix_drain_pending_semaphore_waits(proactor);
  // Re-drain: semaphore wait completions may dispatch LINKED continuations
  // that complete immediately (e.g., WAIT→NOP chains).
  completed_count += iree_async_proactor_posix_drain_completion_queue(proactor);
  iree_async_proactor_posix_drain_incoming_messages(proactor);

  // Calculate timeout considering both user request and pending timers.
  int timeout_ms =
      iree_async_proactor_posix_calculate_timeout_ms(proactor, timeout);

  // Poll for ready fds.
  iree_host_size_t ready_count = 0;
  bool timed_out = false;
  iree_status_t poll_status = iree_async_posix_event_set_wait(
      proactor->event_set, timeout_ms, &ready_count, &timed_out);
  if (!iree_status_is_ok(poll_status)) return poll_status;
  if (timed_out) {
    iree_async_posix_wake_drain(&proactor->wake);

    // Process any expired timers (we may have woken due to timer expiry).
    iree_async_proactor_posix_process_expired_timers(proactor);

    // Drain pending_queue again -- new operations may have arrived while
    // we were in event_set_wait.
    iree_async_proactor_posix_drain_pending_queue(proactor);
    iree_async_proactor_posix_drain_pending_fence_imports(proactor);

    // Process pending cancellations (cancel() called during wait).
    iree_async_proactor_posix_drain_pending_timer_cancellations(proactor);
    iree_async_proactor_posix_drain_pending_fd_cancellations(proactor);

    // Drain completion queue -- count all callbacks including LINKED
    // continuations that complete immediately during the drain.
    completed_count +=
        iree_async_proactor_posix_drain_completion_queue(proactor);
    completed_count +=
        iree_async_proactor_posix_drain_pending_semaphore_waits(proactor);
    completed_count +=
        iree_async_proactor_posix_drain_completion_queue(proactor);
    iree_async_proactor_posix_drain_incoming_messages(proactor);

    if (out_completed_count) *out_completed_count = completed_count;
    if (completed_count > 0) return iree_ok_status();

    // No completions - truly a deadline exceeded.
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  // Process expired timers BEFORE fd events (timers have priority).
  // Pushes to completion queue; counted when drained below.
  iree_async_proactor_posix_process_expired_timers(proactor);

  // Process ready fds using the unified fd_map dispatch.
  // Each fd is looked up once in the O(1) hash table, then dispatched by
  // handler type. This replaces the previous cascading O(n) linked-list
  // scans (find_event_source, find_relay, find_notification).
  iree_async_posix_event_set_reset_ready_iteration(proactor->event_set);
  int fd = 0;
  short revents = 0;
  while (iree_async_posix_event_set_next_ready(proactor->event_set, &fd,
                                               &revents)) {
    // Skip the wake fd (it's just for waking the poll thread).
    if (fd == proactor->wake.read_fd) {
      iree_async_posix_wake_drain(&proactor->wake);
      continue;
    }

    // Single O(1) lookup for this fd.
    iree_async_posix_fd_handler_type_t handler_type;
    void* handler;
    if (!iree_async_posix_fd_map_lookup(&proactor->fd_map, fd, &handler_type,
                                        &handler)) {
      continue;  // Stale fd — already removed.
    }

    switch (handler_type) {
      case IREE_ASYNC_POSIX_FD_HANDLER_EVENT_SOURCE: {
        iree_async_event_source_t* source = (iree_async_event_source_t*)handler;
        if (source->callback.fn) {
          iree_async_poll_events_t poll_events =
              iree_async_posix_translate_poll_events(revents);
          source->callback.fn(source->callback.user_data, source, poll_events);
        }
        break;
      }

      case IREE_ASYNC_POSIX_FD_HANDLER_RELAY: {
        iree_async_proactor_posix_dispatch_relay(
            proactor, (iree_async_relay_t*)handler, revents);
        break;
      }

      case IREE_ASYNC_POSIX_FD_HANDLER_NOTIFICATION: {
        iree_async_notification_t* notification =
            (iree_async_notification_t*)handler;
        // Process both pending async waits and notification-source relays.
        iree_async_proactor_posix_process_notification_waits(proactor,
                                                             notification);
        iree_async_proactor_posix_dispatch_notification_relays(proactor,
                                                               notification);
        break;
      }

      case IREE_ASYNC_POSIX_FD_HANDLER_FENCE_IMPORT: {
        iree_async_posix_fence_import_tracker_t* tracker =
            (iree_async_posix_fence_import_tracker_t*)handler;
        iree_async_proactor_posix_handle_fence_import(proactor, tracker,
                                                      revents);
        // Fence import completed (signaled semaphore). Count it.
        completed_count++;
        break;
      }

      case IREE_ASYNC_POSIX_FD_HANDLER_OPERATION: {
        // Process operation chain. Handles multiple concurrent operations on
        // the same fd (e.g., concurrent sends, or send + recv).
        iree_async_proactor_posix_process_operation_chain(proactor, fd,
                                                          revents);
        break;
      }
    }
  }

  // Drain pending_queue again — new operations may have arrived during fd
  // processing (e.g., LINKED continuations resubmitted from callbacks).
  iree_async_proactor_posix_drain_pending_queue(proactor);
  iree_async_proactor_posix_drain_pending_fence_imports(proactor);

  // Process pending cancellations not handled during ready-fd or timer dispatch
  // (e.g., cancelled operations on fds that didn't fire, or timers whose
  // deadlines haven't passed yet).
  iree_async_proactor_posix_drain_pending_timer_cancellations(proactor);
  iree_async_proactor_posix_drain_pending_fd_cancellations(proactor);

  // Drain completion queue — count all callbacks including LINKED
  // continuations that complete immediately during the drain.
  completed_count += iree_async_proactor_posix_drain_completion_queue(proactor);
  completed_count +=
      iree_async_proactor_posix_drain_pending_semaphore_waits(proactor);
  // Re-drain: semaphore wait completions may dispatch LINKED continuations
  // that complete immediately (e.g., WAIT→NOP chains).
  completed_count += iree_async_proactor_posix_drain_completion_queue(proactor);
  iree_async_proactor_posix_drain_incoming_messages(proactor);

  if (out_completed_count) *out_completed_count = completed_count;
  return completed_count > 0
             ? iree_ok_status()
             : iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
}

static void iree_async_proactor_posix_wake(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_async_proactor_posix_wake_poll_thread(proactor);
}

static iree_status_t iree_async_proactor_posix_cancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_TIMER: {
      // Set the cancelled flag and increment the timer cancellation counter.
      // The poll thread scans the timer_list when the counter is non-zero,
      // removing cancelled timers and pushing CANCELLED completions. Without
      // the scan, cancelled timers would linger until their original deadline.
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED);
      iree_atomic_fetch_add(&proactor->pending_timer_cancellation_count, 1,
                            iree_memory_order_release);
      iree_async_proactor_posix_wake_poll_thread(proactor);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      // Set the cancelled flag. The poll thread checks this during:
      //   - pending_queue drain (if not yet registered with fd_map)
      //   - fd_map cancellation scan (if registered but fd hasn't fired)
      //   - ready-fd processing (if registered and fd fires)
      // Increment the cancellation counter so the poll thread knows to scan
      // the fd_map even when no fds are ready.
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED);
      iree_atomic_fetch_add(&proactor->pending_fd_cancellation_count, 1,
                            iree_memory_order_release);
      iree_async_proactor_posix_wake_poll_thread(proactor);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      // Set the cancelled flag. The poll thread checks this during:
      //   - pending_queue drain (if not yet registered)
      //   - notification epoch scan (processes all pending notification waits)
      iree_async_operation_set_internal_flags(
          operation, IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED);
      iree_async_proactor_posix_wake_poll_thread(proactor);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL:
      // Signals complete synchronously during submit — nothing to cancel.
      return iree_ok_status();

    case IREE_ASYNC_OPERATION_TYPE_MESSAGE:
      // Messages complete synchronously during submit (delivery to the target
      // pool + source completion push both happen inline) — nothing to cancel.
      return iree_ok_status();

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT: {
      iree_async_semaphore_wait_operation_t* wait_op =
          (iree_async_semaphore_wait_operation_t*)operation;
      iree_async_posix_semaphore_wait_tracker_t* tracker =
          (iree_async_posix_semaphore_wait_tracker_t*)wait_op->base.next;
      if (!tracker) {
        // No tracker: operation was never submitted or already completed.
        return iree_ok_status();
      }
      // Try to set CANCELLED status (first status wins via CAS).
      intptr_t expected = (intptr_t)iree_ok_status();
      iree_status_t cancelled_status =
          iree_make_status(IREE_STATUS_CANCELLED, "operation cancelled");
      if (!iree_atomic_compare_exchange_strong(
              &tracker->completion_status, &expected,
              (intptr_t)cancelled_status, iree_memory_order_acq_rel,
              iree_memory_order_acquire)) {
        // Another callback already set a status. The tracker will be (or
        // has been) enqueued for completion by that callback.
        iree_status_ignore(cancelled_status);
        return iree_ok_status();
      }
      // Cancel all registered timepoints.
      for (iree_host_size_t i = 0; i < tracker->count; ++i) {
        iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                              &tracker->timepoints[i]);
      }
      // Enqueue for completion on the poll thread.
      iree_async_proactor_posix_semaphore_wait_enqueue_completion(tracker);
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
      return iree_async_sequence_cancel(
          base_proactor, (iree_async_sequence_operation_t*)operation);

    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "cancel not yet implemented for operation type %d",
          (int)operation->type);
  }
}

static iree_status_t iree_async_proactor_posix_create_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  return iree_async_posix_socket_create(proactor, type, options, out_socket);
}

static iree_status_t iree_async_proactor_posix_import_socket(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  return iree_async_posix_socket_import(proactor, primitive, type, flags,
                                        out_socket);
}

static void iree_async_proactor_posix_destroy_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_t* socket) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_async_posix_socket_destroy(proactor, socket);
}

//===----------------------------------------------------------------------===//
// File vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_posix_import_file(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Validate the file descriptor.
  int fd = primitive.value.fd;
  if (fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid file descriptor %d", fd);
  }

  // Allocate the file structure.
  iree_async_file_t* file = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*file), (void**)&file));
  memset(file, 0, sizeof(*file));

  // Initialize the file.
  iree_atomic_ref_count_init(&file->ref_count);
  file->proactor = base_proactor;
  file->primitive = primitive;
  file->fixed_file_index = -1;  // Not using io_uring fixed files in POSIX.

  IREE_TRACE({
    // For debugging, store a placeholder path since we don't know the actual
    // path for an imported fd.
    snprintf(file->debug_path, sizeof(file->debug_path), "fd:%d", fd);
  });

  *out_file = file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_posix_destroy_file(
    iree_async_proactor_t* base_proactor, iree_async_file_t* file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(file);

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Close the file descriptor. Ignore errors during destroy - the fd may
  // already be closed by a FILE_CLOSE operation.
  int fd = file->primitive.value.fd;
  if (fd >= 0) {
    close(fd);
  }

  // Free the file structure.
  iree_allocator_free(allocator, file);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Event vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_posix_create_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t** out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Allocate and zero-initialize the event structure.
  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*event), (void**)&event));
  memset(event, 0, sizeof(*event));

#if defined(IREE_PLATFORM_LINUX)
  // Linux: use eventfd (single fd, more efficient than pipe).
  // EFD_CLOEXEC prevents leakage to child processes.
  // EFD_NONBLOCK is required so reads during drain don't block.
  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd < 0) {
    iree_allocator_free(allocator, event);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed (%d)", errno);
  }
  iree_atomic_ref_count_init(&event->ref_count);
  event->proactor = base_proactor;
  event->primitive = iree_async_primitive_from_fd(efd);
  // On Linux, signal_primitive is the same as primitive (eventfd is
  // bidirectional: write to signal, read to drain).
  event->signal_primitive = event->primitive;
#else
  // macOS/BSD: use pipe (read end for monitoring, write end for signaling).
  int pipe_fds[2] = {-1, -1};
  if (pipe(pipe_fds) < 0) {
    iree_allocator_free(allocator, event);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pipe() failed for event creation");
  }
  // Set non-blocking and close-on-exec on both ends.
  for (int i = 0; i < 2; ++i) {
    int flags = fcntl(pipe_fds[i], F_GETFL);
    if (flags >= 0) {
      fcntl(pipe_fds[i], F_SETFL, flags | O_NONBLOCK);
    }
    fcntl(pipe_fds[i], F_SETFD, FD_CLOEXEC);
  }
  iree_atomic_ref_count_init(&event->ref_count);
  event->proactor = base_proactor;
  event->primitive = iree_async_primitive_from_fd(pipe_fds[0]);         // read
  event->signal_primitive = iree_async_primitive_from_fd(pipe_fds[1]);  // write
#endif  // IREE_PLATFORM_LINUX

  event->fixed_file_index = -1;
  event->pool = NULL;
  event->pool_next = NULL;
  event->pool_all_next = NULL;

  *out_event = event;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_posix_destroy_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t* event) {
  if (!event) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Close the fd(s).
  // On Linux (eventfd): primitive == signal_primitive, close once.
  // On macOS (pipe): primitive != signal_primitive, close both ends.
  if (event->primitive.value.fd >= 0) {
    close(event->primitive.value.fd);
  }
  if (event->signal_primitive.value.fd >= 0 &&
      event->signal_primitive.value.fd != event->primitive.value.fd) {
    close(event->signal_primitive.value.fd);
  }

  iree_allocator_free(allocator, event);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Notification vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_posix_create_notification(
    iree_async_proactor_t* base_proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_notification);
  *out_notification = NULL;

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  iree_async_notification_t* notification = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*notification),
                                (void**)&notification));
  memset(notification, 0, sizeof(*notification));

  iree_atomic_ref_count_init(&notification->ref_count);
  notification->proactor = &proactor->base;
  iree_atomic_store(&notification->epoch, 0, iree_memory_order_release);
  notification->mode = IREE_ASYNC_NOTIFICATION_MODE_EVENT;
  notification->platform.posix.pending_waits = NULL;
  notification->platform.posix.relay_list = NULL;
#if !defined(IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX)
  iree_notification_initialize(&notification->platform.posix.sync_notification);
#endif  // !IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX

#if defined(IREE_PLATFORM_LINUX)
  // Linux: eventfd provides a single fd for both monitoring and signaling.
  // EFD_NONBLOCK is required so reads during drain don't block.
  // EFD_CLOEXEC prevents leakage to child processes.
  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd < 0) {
    iree_allocator_free(allocator, notification);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed (%d)", errno);
  }
  notification->platform.posix.primitive = iree_async_primitive_from_fd(efd);
  notification->platform.posix.signal_primitive =
      notification->platform.posix.primitive;
#else
  // macOS/BSD: pipe with non-blocking and close-on-exec on both ends.
  // Read end for monitoring (POLLIN), write end for signaling.
  int pipe_fds[2] = {-1, -1};
  if (pipe(pipe_fds) < 0) {
    iree_allocator_free(allocator, notification);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pipe() failed for notification creation");
  }
  for (int i = 0; i < 2; ++i) {
    int current_flags = fcntl(pipe_fds[i], F_GETFL);
    if (current_flags >= 0) {
      fcntl(pipe_fds[i], F_SETFL, current_flags | O_NONBLOCK);
    }
    fcntl(pipe_fds[i], F_SETFD, FD_CLOEXEC);
  }
  notification->platform.posix.primitive =
      iree_async_primitive_from_fd(pipe_fds[0]);
  notification->platform.posix.signal_primitive =
      iree_async_primitive_from_fd(pipe_fds[1]);
#endif  // IREE_PLATFORM_LINUX

  *out_notification = notification;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_posix_destroy_notification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification) {
  if (!notification) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

#if !defined(IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX)
  iree_notification_deinitialize(
      &notification->platform.posix.sync_notification);
#endif  // !IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX

  // Close fds. On Linux (eventfd): primitive == signal_primitive, close once.
  // On macOS (pipe): close both ends independently.
  if (notification->platform.posix.primitive.value.fd >= 0) {
    close(notification->platform.posix.primitive.value.fd);
  }
  if (notification->platform.posix.signal_primitive.value.fd >= 0 &&
      notification->platform.posix.signal_primitive.value.fd !=
          notification->platform.posix.primitive.value.fd) {
    close(notification->platform.posix.signal_primitive.value.fd);
  }

  iree_allocator_free(allocator, notification);
  IREE_TRACE_ZONE_END(z0);
}

#if defined(IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX)

//===----------------------------------------------------------------------===//
// Futex-based sync notification signal/wait
//===----------------------------------------------------------------------===//
// Sync waiters use futex_wait() on the epoch atomic. The eventfd is written
// for async waiters and relays only; sync waiters never touch it, so there
// is no drain race with the poll loop.

static void iree_async_proactor_posix_notification_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, int32_t wake_count) {
  (void)base_proactor;
  // Write eventfd/pipe to wake the poll thread (async waits and relays).
  // EAGAIN means already signaled (redundant, benign). EBADF/EPIPE indicates
  // the notification's fd was closed while still in use — a lifecycle bug.
  int signal_fd = notification->platform.posix.signal_primitive.value.fd;
#if defined(IREE_PLATFORM_LINUX)
  uint64_t value = 1;
  ssize_t result = write(signal_fd, &value, sizeof(value));
#else
  uint8_t value = 1;
  ssize_t result = write(signal_fd, &value, sizeof(value));
#endif  // IREE_PLATFORM_LINUX
  IREE_ASSERT(result >= 0 || errno == EAGAIN);
  // Wake sync waiters directly via futex on the epoch atomic.
  iree_futex_wake(&notification->epoch, wake_count);
}

static bool iree_async_proactor_posix_notification_wait(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, iree_timeout_t timeout) {
  (void)base_proactor;
  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);
  uint32_t wait_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  while (iree_time_now() < deadline_ns) {
    uint32_t current_epoch =
        iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
    if (current_epoch != wait_epoch) return true;
    iree_status_code_t wait_result =
        iree_futex_wait(&notification->epoch, wait_epoch, deadline_ns);
    if (wait_result == IREE_STATUS_DEADLINE_EXCEEDED) break;
    // IREE_STATUS_OK or IREE_STATUS_UNAVAILABLE (spurious) — re-check epoch.
  }
  uint32_t final_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  return final_epoch != wait_epoch;
}

#else  // !IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX

//===----------------------------------------------------------------------===//
// Condvar-based sync notification signal/wait
//===----------------------------------------------------------------------===//
// Sync waiters use iree_notification_await() on sync_notification (condvar).
// The eventfd is written for async waiters and relays only; sync waiters never
// touch it, so there is no drain race with the poll loop.

static void iree_async_proactor_posix_notification_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, int32_t wake_count) {
  (void)base_proactor;
  // Write eventfd/pipe to wake the poll thread (async waits and relays).
  // EAGAIN means already signaled (redundant, benign). EBADF/EPIPE indicates
  // the notification's fd was closed while still in use — a lifecycle bug.
  int signal_fd = notification->platform.posix.signal_primitive.value.fd;
#if defined(IREE_PLATFORM_LINUX)
  uint64_t value = 1;
  ssize_t result = write(signal_fd, &value, sizeof(value));
#else
  uint8_t value = 1;
  ssize_t result = write(signal_fd, &value, sizeof(value));
#endif  // IREE_PLATFORM_LINUX
  IREE_ASSERT(result >= 0 || errno == EAGAIN);
  // Wake sync waiters via condvar.
  iree_notification_post(&notification->platform.posix.sync_notification,
                         wake_count);
}

// Predicate for iree_notification_await: true when epoch has advanced.
typedef struct iree_async_notification_epoch_predicate_t {
  iree_atomic_int32_t* epoch;
  uint32_t wait_epoch;
} iree_async_notification_epoch_predicate_t;

static bool iree_async_notification_epoch_advanced(void* arg) {
  iree_async_notification_epoch_predicate_t* predicate =
      (iree_async_notification_epoch_predicate_t*)arg;
  return iree_atomic_load(predicate->epoch, iree_memory_order_acquire) !=
         predicate->wait_epoch;
}

static bool iree_async_proactor_posix_notification_wait(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification, iree_timeout_t timeout) {
  (void)base_proactor;
  uint32_t wait_epoch =
      iree_atomic_load(&notification->epoch, iree_memory_order_acquire);
  // Fast path: epoch already advanced.
  if (iree_atomic_load(&notification->epoch, iree_memory_order_acquire) !=
      wait_epoch) {
    return true;
  }
  iree_async_notification_epoch_predicate_t predicate = {
      .epoch = &notification->epoch,
      .wait_epoch = wait_epoch,
  };
  return iree_notification_await(
      &notification->platform.posix.sync_notification,
      iree_async_notification_epoch_advanced, &predicate, timeout);
}

#endif  // IREE_ASYNC_POSIX_NOTIFICATION_USE_FUTEX

//===----------------------------------------------------------------------===//
// Event source vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_posix_register_event_source(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_event_source);
  *out_event_source = NULL;

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);

  // Validate arguments.
  if (handle.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "register_event_source requires an FD primitive (got type %d)",
        (int)handle.type);
  }
  if (handle.value.fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "register_event_source fd must be >= 0 (got %d)",
                            handle.value.fd);
  }
  if (!callback.fn) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "register_event_source requires a non-NULL "
                            "callback function");
  }

  // Allocate the event source struct.
  iree_async_event_source_t* source = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->base.allocator, sizeof(*source),
                                (void**)&source));

  // Initialize the event source.
  source->next = NULL;
  source->prev = NULL;
  source->proactor = base_proactor;
  source->fd = handle.value.fd;
  source->callback = callback;
  source->allocator = proactor->base.allocator;

  // Add fd to event set and fd_map for O(1) dispatch during poll.
  // Per the proactor.h API contract: "Registration and unregistration must be
  // serialized with respect to poll()." This means we are on the poll thread
  // (or externally serialized), so direct event_set/fd_map manipulation is
  // safe.
  iree_status_t status =
      iree_async_posix_event_set_add(proactor->event_set, source->fd, POLLIN);
  if (iree_status_is_ok(status)) {
    status = iree_async_posix_fd_map_insert(
        &proactor->fd_map, source->fd, IREE_ASYNC_POSIX_FD_HANDLER_EVENT_SOURCE,
        source);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(
          iree_async_posix_event_set_remove(proactor->event_set, source->fd));
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(proactor->base.allocator, source);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Link into the proactor's event source list (head insert).
  source->next = proactor->event_sources;
  if (proactor->event_sources) {
    proactor->event_sources->prev = source;
  }
  proactor->event_sources = source;

  *out_event_source = source;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_posix_unregister_event_source(
    iree_async_proactor_t* base_proactor,
    iree_async_event_source_t* event_source) {
  if (!event_source) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);

  // Remove from fd_map and event set (stops monitoring).
  iree_async_posix_fd_map_remove(&proactor->fd_map, event_source->fd);
  iree_status_ignore(
      iree_async_posix_event_set_remove(proactor->event_set, event_source->fd));

  // Unlink from the proactor's event source list.
  if (event_source->prev) {
    event_source->prev->next = event_source->next;
  } else {
    // Head of list.
    proactor->event_sources = event_source->next;
  }
  if (event_source->next) {
    event_source->next->prev = event_source->prev;
  }

  // Free the event source struct.
  iree_allocator_t allocator = event_source->allocator;
  iree_allocator_free(allocator, event_source);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Relay vtable wrappers
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_posix_vtable_register_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  return iree_async_proactor_posix_register_relay(proactor, source, sink, flags,
                                                  error_callback, out_relay);
}

static void iree_async_proactor_posix_vtable_unregister_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_t* relay) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_async_proactor_posix_unregister_relay(proactor, relay);
}

static void iree_async_proactor_posix_set_message_callback(
    iree_async_proactor_t* base_proactor,
    iree_async_proactor_message_callback_t callback) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  proactor->message_callback = callback;
}

static iree_status_t iree_async_proactor_posix_send_message(
    iree_async_proactor_t* base_target, uint64_t message_data) {
  iree_async_proactor_posix_t* target =
      iree_async_proactor_posix_cast(base_target);
  iree_status_t status =
      iree_async_message_pool_send(&target->message_pool, message_data);
  if (iree_status_is_ok(status)) {
    base_target->vtable->wake(base_target);
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Buffer registration (emulated)
//===----------------------------------------------------------------------===//

// Combined allocation for registration entry + region.
// Keeps them together in memory and simplifies cleanup.
typedef struct iree_async_posix_buffer_registration_t {
  iree_async_buffer_registration_entry_t entry;
  iree_async_region_t region;
} iree_async_posix_buffer_registration_t;

// Destroy callback for buffer registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_posix_buffer_registration_destroy(
    iree_async_region_t* region) {
  iree_async_posix_buffer_registration_t* registration =
      (iree_async_posix_buffer_registration_t*)((char*)region -
                                                offsetof(
                                                    iree_async_posix_buffer_registration_t,
                                                    region));
  iree_allocator_free(region->proactor->allocator, registration);
}

// Cleanup function for buffer registrations.
// Called when the registration state is cleaned up.
static void iree_async_posix_buffer_registration_cleanup(void* entry_ptr,
                                                         void* proactor_ptr) {
  iree_async_posix_buffer_registration_t* registration =
      (iree_async_posix_buffer_registration_t*)entry_ptr;
  (void)proactor_ptr;
  iree_async_region_release(&registration->region);
}

static iree_status_t iree_async_proactor_posix_register_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, iree_byte_span_t buffer,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_entry = NULL;

  // Allocate combined entry + region.
  iree_async_posix_buffer_registration_t* registration = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(proactor->allocator, sizeof(*registration),
                                (void**)&registration));
  memset(registration, 0, sizeof(*registration));

  // Initialize the region (metadata-only, no kernel registration).
  iree_async_region_t* region = &registration->region;
  iree_atomic_ref_count_init(&region->ref_count);
  region->proactor = proactor;
  region->slab = NULL;
  region->destroy_fn = iree_async_posix_buffer_registration_destroy;
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
  entry->proactor = proactor;
  entry->cleanup_fn = iree_async_posix_buffer_registration_cleanup;
  entry->region = region;

  // Link into the caller's state.
  iree_async_buffer_registration_state_add(state, entry);

  *out_entry = entry;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_posix_register_dmabuf(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_state_t* state, int dmabuf_fd,
    uint64_t offset, iree_host_size_t length,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_buffer_registration_entry_t** out_entry) {
  (void)proactor;
  (void)state;
  (void)dmabuf_fd;
  (void)offset;
  (void)length;
  (void)access_flags;
  *out_entry = NULL;
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "POSIX backend does not support DMA-buf registration");
}

static void iree_async_proactor_posix_unregister_buffer(
    iree_async_proactor_t* proactor,
    iree_async_buffer_registration_entry_t* entry,
    iree_async_buffer_registration_state_t* state) {
  iree_async_buffer_registration_state_remove(state, entry);
  entry->cleanup_fn(entry, proactor);
}

//===----------------------------------------------------------------------===//
// Slab registration (emulated)
//===----------------------------------------------------------------------===//

// Wrapper for slab-backed regions. Stores the allocator for self-deallocation.
typedef struct iree_async_posix_slab_region_t {
  iree_async_region_t region;
  iree_allocator_t allocator;
} iree_async_posix_slab_region_t;

// Destroy callback for slab registration regions.
// Called when the region's ref count reaches zero.
static void iree_async_posix_slab_region_destroy(iree_async_region_t* region) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_posix_slab_region_t* slab_region =
      (iree_async_posix_slab_region_t*)((char*)region -
                                        offsetof(iree_async_posix_slab_region_t,
                                                 region));

  // Clear singleton tracking for READ-access registrations.
  if (iree_any_bit_set(region->access_flags,
                       IREE_ASYNC_BUFFER_ACCESS_FLAG_READ)) {
    iree_async_proactor_posix_t* proactor =
        iree_async_proactor_posix_cast(region->proactor);
    proactor->has_read_slab_registration = false;
  }

  // Release the retained slab reference.
  iree_async_slab_release(region->slab);

  iree_allocator_free(slab_region->allocator, slab_region);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_proactor_posix_register_slab(
    iree_async_proactor_t* base_proactor, iree_async_slab_t* slab,
    iree_async_buffer_access_flags_t access_flags,
    iree_async_region_t** out_region) {
  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(slab);
  IREE_ASSERT_ARGUMENT(out_region);
  *out_region = NULL;

  iree_host_size_t buffer_size = iree_async_slab_buffer_size(slab);
  iree_host_size_t buffer_count = iree_async_slab_buffer_count(slab);
  void* base_ptr = iree_async_slab_base_ptr(slab);

  // Check singleton constraint for READ access (fixed buffer table equivalent).
  // io_uring allows only one buffer table registration at a time; enforced
  // identically here for API portability (see proactor.h:930-934).
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
  iree_async_posix_slab_region_t* slab_region = NULL;
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
  region->destroy_fn = iree_async_posix_slab_region_destroy;
  // WRITE-access regions use EMULATED type to indicate proactor-managed buffer
  // selection (pool acquire in userspace). READ-only regions need no
  // backend-specific handles since the send path uses iree_async_span_ptr().
  region->type = needs_write ? IREE_ASYNC_REGION_TYPE_EMULATED
                             : IREE_ASYNC_REGION_TYPE_NONE;
  region->access_flags = access_flags;
  region->base_ptr = base_ptr;
  region->length = iree_async_slab_total_size(slab);
  // No recycle callback — POSIX uses pool freelist for buffer recycling.
  // io_uring uses region->recycle to replenish PBUF_RING; POSIX has no ring.
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
// Signal handling
//===----------------------------------------------------------------------===//
// Uses signalfd on Linux (IREE_ASYNC_SIGNAL_USE_SIGNALFD) or the self-pipe
// trick on all other POSIX platforms (IREE_ASYNC_SIGNAL_USE_SELFPIPE).
// Both backends present the same interface: a readable fd that the proactor
// monitors via an event source. When the fd is readable, pending signals are
// drained and dispatched to subscribers.
//
// Normalized backend interface — single #if block, no ifdefs below.

#if defined(IREE_ASYNC_SIGNAL_USE_SIGNALFD)

#define iree_async_posix_signal_state_initialize \
  iree_async_linux_signal_state_initialize
#define iree_async_posix_signal_add iree_async_linux_signal_add_signal
#define iree_async_posix_signal_remove iree_async_linux_signal_remove_signal
#define iree_async_posix_signal_deinitialize \
  iree_async_linux_signal_deinitialize
#define iree_async_posix_signal_read iree_async_linux_signal_read_signalfd

#elif defined(IREE_ASYNC_SIGNAL_USE_SELFPIPE)

#define iree_async_posix_signal_state_initialize \
  iree_async_selfpipe_signal_state_initialize
#define iree_async_posix_signal_add iree_async_selfpipe_signal_add_signal
#define iree_async_posix_signal_remove iree_async_selfpipe_signal_remove_signal
#define iree_async_posix_signal_deinitialize \
  iree_async_selfpipe_signal_deinitialize
#define iree_async_posix_signal_read iree_async_selfpipe_signal_read_pipe

#endif  // IREE_ASYNC_SIGNAL_USE_*

// Callback invoked for each signal read from the signal fd during drain.
// Dispatches to all subscribers and processes deferred unsubscribes.
static void iree_async_proactor_posix_signal_dispatch_callback(
    void* user_data, iree_async_signal_t signal) {
  iree_async_proactor_posix_t* proactor =
      (iree_async_proactor_posix_t*)user_data;

  // Dispatch to all subscribers for this signal.
  iree_async_signal_subscription_t* head =
      proactor->signal.subscriptions[signal];
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

    if (--proactor->signal.subscriber_count[sub_signal] == 0) {
      iree_async_posix_signal_remove(&proactor->signal.backend_state,
                                     sub_signal);
    }
  }
}

// Event source callback: signal fd is readable, drain and dispatch.
static void iree_async_proactor_posix_signal_event_source_callback(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t events) {
  iree_async_proactor_posix_t* proactor =
      (iree_async_proactor_posix_t*)user_data;
  if (!proactor->signal.initialized) return;

  iree_status_t status = iree_async_posix_signal_read(
      &proactor->signal.backend_state,
      (iree_async_signal_dispatch_callback_t){
          .fn = iree_async_proactor_posix_signal_dispatch_callback,
          .user_data = proactor,
      });
  iree_status_ignore(status);
}

// Lazy-initializes signal handling: creates signal fd and registers as event
// source. Called on first subscribe.
static iree_status_t iree_async_proactor_posix_signal_initialize(
    iree_async_proactor_posix_t* proactor, iree_async_signal_t first_signal,
    int* out_signal_fd) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_posix_signal_state_initialize(&proactor->signal.backend_state);
  iree_status_t status = iree_async_posix_signal_add(
      &proactor->signal.backend_state, first_signal, out_signal_fd);

  // Register the signal fd as an event source for poll monitoring.
  if (iree_status_is_ok(status)) {
    iree_async_primitive_t handle;
    memset(&handle, 0, sizeof(handle));
    handle.type = IREE_ASYNC_PRIMITIVE_TYPE_FD;
    handle.value.fd = *out_signal_fd;
    iree_async_event_source_callback_t callback;
    callback.fn = iree_async_proactor_posix_signal_event_source_callback;
    callback.user_data = proactor;
    status = iree_async_proactor_posix_register_event_source(
        &proactor->base, handle, callback, &proactor->signal.event_source);
  }

  if (iree_status_is_ok(status)) {
    iree_async_signal_dispatch_state_initialize(
        &proactor->signal.dispatch_state);
    proactor->signal.initialized = true;
  } else {
    iree_async_posix_signal_deinitialize(&proactor->signal.backend_state);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Deinitializes all signal handling state. Called from proactor destroy.
static void iree_async_proactor_posix_signal_deinitialize(
    iree_async_proactor_posix_t* proactor) {
  iree_allocator_t allocator = proactor->base.allocator;

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

  // Unregister event source (removes from event_set and fd_map).
  if (proactor->signal.event_source) {
    iree_async_proactor_posix_unregister_event_source(
        &proactor->base, proactor->signal.event_source);
    proactor->signal.event_source = NULL;
  }

  // Deinitialize backend signal state (closes fd, restores sigmask/handlers).
  iree_async_posix_signal_deinitialize(&proactor->signal.backend_state);
  proactor->signal.initialized = false;

  // Release global signal ownership so another proactor can claim it.
  iree_async_signal_release_ownership(&proactor->base);
}

static iree_status_t iree_async_proactor_posix_subscribe_signal(
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

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Claim global signal ownership (first proactor wins).
  iree_status_t status = iree_async_signal_claim_ownership(base_proactor);

  // If first subscriber for this specific signal, add it to the signal backend.
  int signal_fd = -1;
  if (iree_status_is_ok(status) &&
      proactor->signal.subscriber_count[signal] == 0) {
    if (!proactor->signal.initialized) {
      // First signal overall: initialize backend and register event source.
      status = iree_async_proactor_posix_signal_initialize(proactor, signal,
                                                           &signal_fd);
      if (!iree_status_is_ok(status)) {
        iree_async_signal_release_ownership(base_proactor);
      }
    } else {
      // Add to existing signal backend.
      status = iree_async_posix_signal_add(&proactor->signal.backend_state,
                                           signal, &signal_fd);
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

static void iree_async_proactor_posix_unsubscribe_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_signal_subscription_t* subscription) {
  if (!subscription) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_posix_t* proactor =
      iree_async_proactor_posix_cast(base_proactor);
  iree_async_signal_t signal = subscription->signal;

  // If we're currently dispatching signals, defer the unsubscribe to avoid
  // corrupting the subscription list during iteration.
  if (proactor->signal.dispatch_state.dispatching) {
    iree_async_signal_subscription_defer_unsubscribe(
        &proactor->signal.dispatch_state, subscription);
  } else {
    // Immediate unsubscribe: unlink, free, and remove signal if last.
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);

    // If this was the last subscriber for this signal, remove it from the
    // backend so default behavior is restored.
    if (--proactor->signal.subscriber_count[signal] == 0) {
      iree_async_posix_signal_remove(&proactor->signal.backend_state, signal);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_async_proactor_vtable_t iree_async_proactor_posix_vtable = {
    .destroy = iree_async_proactor_posix_destroy,
    .query_capabilities = iree_async_proactor_posix_query_capabilities,
    .submit = iree_async_proactor_posix_submit,
    .poll = iree_async_proactor_posix_poll,
    .wake = iree_async_proactor_posix_wake,
    .cancel = iree_async_proactor_posix_cancel,
    .create_socket = iree_async_proactor_posix_create_socket,
    .import_socket = iree_async_proactor_posix_import_socket,
    .destroy_socket = iree_async_proactor_posix_destroy_socket,
    .import_file = iree_async_proactor_posix_import_file,
    .destroy_file = iree_async_proactor_posix_destroy_file,
    .create_event = iree_async_proactor_posix_create_event,
    .destroy_event = iree_async_proactor_posix_destroy_event,
    .register_event_source = iree_async_proactor_posix_register_event_source,
    .unregister_event_source =
        iree_async_proactor_posix_unregister_event_source,
    .create_notification = iree_async_proactor_posix_create_notification,
    .destroy_notification = iree_async_proactor_posix_destroy_notification,
    .notification_signal = iree_async_proactor_posix_notification_signal,
    .notification_wait = iree_async_proactor_posix_notification_wait,
    .register_relay = iree_async_proactor_posix_vtable_register_relay,
    .unregister_relay = iree_async_proactor_posix_vtable_unregister_relay,
    // Emulated buffer registration (metadata-only, no kernel registration).
    .register_buffer = iree_async_proactor_posix_register_buffer,
    .register_dmabuf = iree_async_proactor_posix_register_dmabuf,
    .unregister_buffer = iree_async_proactor_posix_unregister_buffer,
    .register_slab = iree_async_proactor_posix_register_slab,
    .import_fence = iree_async_proactor_posix_import_fence,
    .export_fence = iree_async_proactor_posix_export_fence,
    .set_message_callback = iree_async_proactor_posix_set_message_callback,
    .send_message = iree_async_proactor_posix_send_message,
    .subscribe_signal = iree_async_proactor_posix_subscribe_signal,
    .unsubscribe_signal = iree_async_proactor_posix_unsubscribe_signal,
};
