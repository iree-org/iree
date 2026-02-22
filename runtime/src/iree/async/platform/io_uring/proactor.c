// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Enable GNU extensions for POLLRDHUP (peer half-close detection).
// Must be defined before any includes.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "iree/async/platform/io_uring/proactor.h"

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/futex.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/io_uring/buffer_ring.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/relay.h"
#include "iree/async/platform/io_uring/socket.h"
#include "iree/async/semaphore.h"
#include "iree/async/types.h"
#include "iree/async/util/message_pool.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/memory.h"

static void iree_async_proactor_io_uring_destroy(
    iree_async_proactor_t* base_proactor);
static iree_status_t iree_async_proactor_io_uring_cancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation);

iree_status_t iree_async_proactor_create_io_uring(
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
      z0,
      IREE_STRUCT_LAYOUT(sizeof(iree_async_proactor_io_uring_t), &total_size,
                         IREE_STRUCT_FIELD(message_pool_capacity,
                                           iree_async_message_pool_entry_t,
                                           &message_entries_offset)));

  // Allocate the proactor structure with trailing data.
  iree_async_proactor_io_uring_t* proactor = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, total_size, (void**)&proactor));
  memset(proactor, 0, total_size);

  // Initialize base fields.
  iree_async_proactor_initialize(&iree_async_proactor_io_uring_vtable,
                                 options.debug_name, allocator,
                                 &proactor->base);

  // Initialize sequence emulator for SEQUENCE operation support.
  // Uses vtable-dispatched submit_one which routes individual step operations
  // through this backend's normal submit path.
  iree_async_sequence_emulator_initialize(&proactor->sequence_emulator,
                                          &proactor->base,
                                          iree_async_proactor_submit_one);

  // Initialize message pool with trailing data entries.
  iree_async_message_pool_entry_t* message_entries =
      (iree_async_message_pool_entry_t*)((uint8_t*)proactor +
                                         message_entries_offset);
  iree_async_message_pool_initialize(message_pool_capacity, message_entries,
                                     &proactor->message_pool);

  // Initialize our fields.
  proactor->wake_eventfd = -1;
  proactor->wake_poll_armed = false;
  proactor->defer_submissions = false;
  proactor->capabilities = IREE_ASYNC_PROACTOR_CAPABILITY_NONE;
  iree_atomic_slist_initialize(&proactor->pending_software_completions);
  iree_atomic_slist_initialize(&proactor->pending_semaphore_waits);
  proactor->event_sources = NULL;
  proactor->relays = NULL;

  // Initialize signal handling state (lazy-initialized on first subscribe).
  proactor->signal.initialized = false;
  iree_async_linux_signal_state_initialize(&proactor->signal.linux_state);
  iree_async_signal_dispatch_state_initialize(&proactor->signal.dispatch_state);
  for (int i = 0; i < IREE_ASYNC_SIGNAL_COUNT; ++i) {
    proactor->signal.subscriptions[i] = NULL;
    proactor->signal.subscriber_count[i] = 0;
  }
  proactor->signal.event_source = NULL;

  // Set up ring options based on proactor options.
  iree_io_uring_ring_options_t ring_options =
      iree_io_uring_ring_options_default();
  if (options.max_concurrent_operations > 0) {
    ring_options.sq_entries = (uint32_t)options.max_concurrent_operations;
  }

  // Initialize the io_uring ring.
  iree_status_t status =
      iree_io_uring_ring_initialize(ring_options, &proactor->ring);

  // Detect capabilities from kernel feature flags and opcode probing.
  // This verifies we have a new enough kernel and determines what features
  // are available.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_io_uring_detect_capabilities(
        proactor->ring.ring_fd, proactor->ring.features,
        &proactor->capabilities);
  }

  // Apply the allowed_capabilities mask from options.
  if (iree_status_is_ok(status)) {
    proactor->capabilities &= options.allowed_capabilities;
  }

  // Create sparse buffer table on 5.19+ kernels for dynamic buffer
  // registration. This pre-allocates an empty table in the kernel so
  // individual slots can be populated later via IORING_REGISTER_BUFFERS_UPDATE.
  // MULTISHOT capability implies 5.19+ (the probe checks SOCKET opcode 45).
  if (iree_status_is_ok(status) &&
      iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    uint16_t table_capacity = IREE_IO_URING_SPARSE_TABLE_DEFAULT_CAPACITY;
    status = iree_io_uring_sparse_table_allocate(table_capacity, allocator,
                                                 &proactor->buffer_table);
    if (iree_status_is_ok(status)) {
      iree_io_uring_rsrc_register_t reg = {
          .nr = table_capacity,
          .flags = IREE_IORING_RSRC_REGISTER_SPARSE,
          .resv2 = 0,
          .data = 0,
          .tags = 0,
      };
      long ret = 0;
      do {
        ret = syscall(IREE_IO_URING_SYSCALL_REGISTER, proactor->ring.ring_fd,
                      IREE_IORING_REGISTER_BUFFERS2, &reg, sizeof(reg));
      } while (ret < 0 && errno == EINTR);
      if (ret < 0) {
        int saved_errno = errno;
        iree_io_uring_sparse_table_free(proactor->buffer_table, allocator);
        proactor->buffer_table = NULL;
        status = iree_make_status(
            iree_status_code_from_errno(saved_errno),
            "IORING_REGISTER_BUFFERS2 (sparse, capacity=%u) failed (%d)",
            (unsigned)table_capacity, saved_errno);
      }
    }
  }

  // Create the wake eventfd.
  if (iree_status_is_ok(status)) {
    proactor->wake_eventfd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (proactor->wake_eventfd < 0) {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "eventfd creation failed (%d)", errno);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_proactor = &proactor->base;
  } else {
    iree_async_proactor_io_uring_destroy(&proactor->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_async_proactor_io_uring_destroy(
    iree_async_proactor_t* base_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Clean up signal handling state.
  if (proactor->signal.initialized) {
    // Free all signal subscriptions.
    for (int i = 0; i < IREE_ASYNC_SIGNAL_COUNT; ++i) {
      while (proactor->signal.subscriptions[i]) {
        iree_async_signal_subscription_t* subscription =
            proactor->signal.subscriptions[i];
        proactor->signal.subscriptions[i] = subscription->next;
        iree_allocator_free(allocator, subscription);
      }
    }
    // Free the signal event source.
    iree_allocator_free(allocator, proactor->signal.event_source);
    proactor->signal.event_source = NULL;
    // Deinitialize Linux signalfd state.
    iree_async_linux_signal_deinitialize(&proactor->signal.linux_state);
    proactor->signal.initialized = false;
    // Release global signal ownership so another proactor can claim it.
    iree_async_signal_release_ownership(&proactor->base);
  }

  // Free any remaining event sources. In normal use, callers should unregister
  // all event sources before destroying the proactor, but we clean up here to
  // avoid leaks.
  while (proactor->event_sources) {
    iree_async_event_source_t* source = proactor->event_sources;
    proactor->event_sources = source->next;
    iree_allocator_free(source->allocator, source);
  }

  // Free any remaining relays. In normal use, callers should unregister all
  // relays before destroying the proactor, but we clean up here to avoid leaks.
  while (proactor->relays) {
    iree_async_relay_t* relay = proactor->relays;
    proactor->relays = relay->next;
    // Close source fd if owned.
    if (iree_any_bit_set(relay->flags,
                         IREE_ASYNC_RELAY_FLAG_OWN_SOURCE_PRIMITIVE) &&
        relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_PRIMITIVE) {
      close(relay->source.primitive.value.fd);
    }
    // Release retained notifications.
    if (relay->source.type == IREE_ASYNC_RELAY_SOURCE_TYPE_NOTIFICATION) {
      iree_async_notification_release(relay->source.notification);
    }
    if (relay->sink.type == IREE_ASYNC_RELAY_SINK_TYPE_SIGNAL_NOTIFICATION) {
      iree_async_notification_release(
          relay->sink.signal_notification.notification);
    }
    iree_allocator_free(relay->allocator, relay);
  }

  // Deinitialize the message pool (all entries returned to free list by now).
  iree_async_message_pool_deinitialize(&proactor->message_pool);

  // Free sparse buffer table. The kernel table is automatically destroyed
  // when the ring fd is closed; this frees the userspace slot allocator.
  iree_io_uring_sparse_table_free(proactor->buffer_table, allocator);

  // Close the wake eventfd.
  if (proactor->wake_eventfd >= 0) {
    close(proactor->wake_eventfd);
    proactor->wake_eventfd = -1;
  }

  // Deinitialize the ring.
  iree_io_uring_ring_deinitialize(&proactor->ring);

  // Free the proactor structure.
  iree_allocator_free(allocator, proactor);

  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

static iree_async_proactor_capabilities_t
iree_async_proactor_io_uring_query_capabilities(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return proactor->capabilities;
}

//===----------------------------------------------------------------------===//
// Message delivery
//===----------------------------------------------------------------------===//

// Drains pending messages from the message pool and invokes the callback.
// Called from poll() after processing CQEs.
static void iree_async_proactor_io_uring_drain_pending_messages(
    iree_async_proactor_io_uring_t* proactor) {
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
// LINKED chain dispatch helpers
//===----------------------------------------------------------------------===//

// Submits a continuation chain from an operation's linked_next pointer.
// Builds a temporary array from the linked list and submits as a batch.
// On submit failure, invokes callbacks directly with the error.
void iree_async_proactor_io_uring_submit_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head) {
  if (!chain_head) return;

  // Count operations in the chain (must happen before submit, which rebuilds
  // linked_next from LINKED flags and destroys the incoming chain).
  iree_host_size_t chain_count = 0;
  for (iree_async_operation_t* op = chain_head; op; op = op->linked_next) {
    ++chain_count;
  }

  // Build array from linked_next chain. Use stack for the common case (chains
  // are typically 1-5 operations), heap-allocate for longer chains.
  iree_async_operation_t* stack_array[16];
  iree_async_operation_t** chain_array = stack_array;
  if (chain_count > IREE_ARRAYSIZE(stack_array)) {
    iree_status_t alloc_status = iree_allocator_malloc(
        proactor->base.allocator, chain_count * sizeof(iree_async_operation_t*),
        (void**)&chain_array);
    if (!iree_status_is_ok(alloc_status)) {
      // Allocation failed — complete all operations with the error.
      for (iree_async_operation_t* op = chain_head; op;) {
        iree_async_operation_t* next = op->linked_next;
        if (op->completion_fn) {
          op->completion_fn(op->user_data, op, iree_status_clone(alloc_status),
                            IREE_ASYNC_COMPLETION_FLAG_NONE);
        }
        op = next;
      }
      iree_status_ignore(alloc_status);
      return;
    }
  }

  iree_host_size_t index = 0;
  for (iree_async_operation_t* op = chain_head; op; op = op->linked_next) {
    chain_array[index++] = op;
  }

  iree_async_operation_list_t chain_list = {chain_array, chain_count};
  iree_status_t submit_status =
      iree_async_proactor_io_uring_submit(&proactor->base, chain_list);
  if (!iree_status_is_ok(submit_status)) {
    // Submit failed — invoke callbacks directly with the error.
    for (iree_host_size_t i = 0; i < chain_count; ++i) {
      if (chain_array[i]->completion_fn) {
        chain_array[i]->completion_fn(chain_array[i]->user_data, chain_array[i],
                                      iree_status_clone(submit_status),
                                      IREE_ASYNC_COMPLETION_FLAG_NONE);
      }
    }
    iree_status_ignore(submit_status);
  }

  if (chain_array != stack_array) {
    iree_allocator_free(proactor->base.allocator, chain_array);
  }
}

//===----------------------------------------------------------------------===//
// Software operation completion delivery
//===----------------------------------------------------------------------===//

// Pushes a completed software operation to the MPSC queue for callback delivery
// on the poll thread. The operation's side effects (e.g., semaphore signal)
// have already been executed; this only defers the callback.
//
// The completion status is carried in base.linked_next (repurposed after the
// continuation chain is consumed). The operation's base.next (offset 0) is used
// as the iree_atomic_slist_entry_t for the MPSC push.
void iree_async_proactor_io_uring_push_software_completion(
    iree_async_proactor_io_uring_t* proactor, iree_async_operation_t* operation,
    iree_status_t status) {
  // Stash the completion status in linked_next (reinterpreted as
  // iree_status_t).
  operation->linked_next = (iree_async_operation_t*)(uintptr_t)status;
  iree_atomic_slist_push(&proactor->pending_software_completions,
                         (iree_atomic_slist_entry_t*)operation);
}

// Drains pending software operation completions and invokes callbacks.
// Called from poll() BEFORE CQE processing to preserve callback ordering for
// chains where a software operation precedes a kernel operation (e.g.,
// SIGNAL(LINKED) → RECV: SIGNAL callback must fire before RECV CQE callback).
//
// Returns the number of completions drained (for inclusion in poll's
// completion count).
static iree_host_size_t
iree_async_proactor_io_uring_drain_pending_software_completions(
    iree_async_proactor_io_uring_t* proactor) {
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_software_completions,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;
  }

  iree_host_size_t drained_count = 0;
  iree_atomic_slist_entry_t* entry = head;
  while (entry != NULL) {
    iree_async_operation_t* operation = (iree_async_operation_t*)entry;
    iree_atomic_slist_entry_t* next = entry->next;

    // Extract the completion status from linked_next.
    iree_status_t status = (iree_status_t)(uintptr_t)operation->linked_next;
    operation->linked_next = NULL;

    // Release resources retained at submit time.
    iree_async_operation_release_resources(operation);

    // Invoke the user callback.
    if (operation->completion_fn) {
      operation->completion_fn(operation->user_data, operation, status,
                               IREE_ASYNC_COMPLETION_FLAG_NONE);
      ++drained_count;
    } else {
      iree_status_ignore(status);
    }

    entry = next;
  }

  return drained_count;
}

//===----------------------------------------------------------------------===//
// Semaphore wait drain
//===----------------------------------------------------------------------===//

// Drains pending semaphore wait completions and invokes callbacks.
// Called from poll() after processing CQEs. Returns the number of completions
// drained (for inclusion in poll's completion count).
static iree_host_size_t
iree_async_proactor_io_uring_drain_pending_semaphore_waits(
    iree_async_proactor_io_uring_t* proactor) {
  // Flush all pending completions with approximate FIFO ordering.
  iree_atomic_slist_entry_t* head = NULL;
  iree_atomic_slist_entry_t* tail = NULL;
  if (!iree_atomic_slist_flush(&proactor->pending_semaphore_waits,
                               IREE_ATOMIC_SLIST_FLUSH_ORDER_APPROXIMATE_FIFO,
                               &head, &tail)) {
    return 0;  // No completions pending.
  }

  // Process each completed semaphore wait.
  iree_host_size_t drained_count = 0;
  iree_atomic_slist_entry_t* entry = head;
  while (entry != NULL) {
    iree_async_io_uring_semaphore_wait_tracker_t* tracker =
        (iree_async_io_uring_semaphore_wait_tracker_t*)entry;
    iree_atomic_slist_entry_t* next = entry->next;

    // Extract completion status.
    iree_status_t status = (iree_status_t)iree_atomic_load(
        &tracker->completion_status, iree_memory_order_acquire);

    // For ANY mode, record which semaphore was satisfied.
    iree_async_semaphore_wait_operation_t* wait_op = tracker->operation;
    if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ANY &&
        iree_status_is_ok(status)) {
      int32_t satisfied = iree_atomic_load(&tracker->remaining_or_satisfied,
                                           iree_memory_order_acquire);
      if (satisfied >= 0) {
        wait_op->satisfied_index = (iree_host_size_t)satisfied;
      }
    }

    // Cancel any remaining timepoints before freeing the tracker.
    // In ANY mode, only one timepoint was satisfied; in ALL mode, all were
    // satisfied. Either way, some timepoints may still be linked to their
    // semaphores' lists. We must unlink them before freeing to avoid
    // use-after-free when those semaphores are later destroyed/failed.
    for (iree_host_size_t i = 0; i < tracker->count; ++i) {
      iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                            &tracker->timepoints[i]);
    }

    // Detach continuation chain before callback (callback may free operation).
    // Continuations are dispatched AFTER the trigger's callback.
    iree_async_operation_t* continuation = tracker->continuation_head;
    tracker->continuation_head = NULL;

    // Invoke the operation's callback.
    if (wait_op->base.completion_fn) {
      wait_op->base.completion_fn(wait_op->base.user_data,
                                  (iree_async_operation_t*)wait_op, status,
                                  IREE_ASYNC_COMPLETION_FLAG_NONE);
      ++drained_count;
    }

    // Dispatch continuation chain after the trigger's callback. Software
    // completions are pushed to MPSC (counted by the post-CQE drain).
    if (continuation) {
      if (iree_status_is_ok(status)) {
        iree_async_proactor_io_uring_dispatch_continuation_chain(proactor,
                                                                 continuation);
      } else {
        iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
            proactor, continuation);
      }
    }

    // Clear the tracker reference from the operation.
    wait_op->base.next = NULL;

    // Free the tracker.
    iree_allocator_free(tracker->allocator, tracker);

    entry = next;
  }

  return drained_count;
}

//===----------------------------------------------------------------------===//
// Wake mechanism
//===----------------------------------------------------------------------===//

// Arms a POLL_ADD on the wake eventfd. This allows wake() from another thread
// to interrupt a blocking poll().
static iree_status_t iree_async_proactor_io_uring_arm_wake(
    iree_async_proactor_io_uring_t* proactor) {
  if (proactor->wake_poll_armed) return iree_ok_status();
  if (proactor->wake_eventfd < 0) return iree_ok_status();

  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full when arming wake");
  }

  sqe->opcode = IREE_IORING_OP_POLL_ADD;
  sqe->fd = proactor->wake_eventfd;
  sqe->poll32_events = POLLIN;
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_WAKE, 0);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  proactor->wake_poll_armed = true;
  return iree_ok_status();
}

// Handles completion of the wake POLL_ADD. Drains the eventfd and re-arms.
static void iree_async_proactor_io_uring_handle_wake_completion(
    iree_async_proactor_io_uring_t* proactor) {
  proactor->wake_poll_armed = false;

  // Drain the eventfd (read returns the count of wake() calls).
  uint64_t value;
  ssize_t ret = read(proactor->wake_eventfd, &value, sizeof(value));
  (void)ret;  // Ignore errors; the poll was the signal.

  // Re-arm for next wake.
  iree_status_t status = iree_async_proactor_io_uring_arm_wake(proactor);
  iree_status_ignore(status);  // Best effort.
}

//===----------------------------------------------------------------------===//
// Fence import completion
//===----------------------------------------------------------------------===//

// Handles completion of a fence import POLL_ADD.
// On success (POLLIN set), signals the semaphore. On error, fails it.
// Always closes the fd, releases the semaphore, and frees the tracker.
static void iree_async_proactor_io_uring_handle_fence_import_completion(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  (void)proactor;

  // Extract the tracker pointer from user_data payload.
  iree_async_io_uring_fence_import_tracker_t* tracker =
      (iree_async_io_uring_fence_import_tracker_t*)(uintptr_t)
          iree_io_uring_internal_payload(cqe->user_data);

  if (cqe->res < 0) {
    // Kernel error (EBADF, ECANCELED, etc.). Fail the semaphore.
    iree_async_semaphore_fail(
        tracker->semaphore,
        iree_make_status(iree_status_code_from_errno(-cqe->res),
                         "fence import POLL_ADD failed (%d)", -cqe->res));
  } else if (!iree_any_bit_set((uint32_t)cqe->res, POLLIN)) {
    // No POLLIN: pure POLLHUP or POLLERR (fd error condition).
    iree_async_semaphore_fail(
        tracker->semaphore,
        iree_make_status(IREE_STATUS_DATA_LOSS,
                         "fence fd signaled error (poll events=0x%04x)",
                         cqe->res));
  } else {
    // POLLIN set: fd became readable, fence signaled.
    iree_status_t signal_status = iree_async_semaphore_signal(
        tracker->semaphore, tracker->signal_value, /*frontier=*/NULL);
    if (!iree_status_is_ok(signal_status)) {
      // Signal failed (e.g., non-monotonic value). Fail the semaphore.
      iree_async_semaphore_fail(tracker->semaphore, signal_status);
    }
  }

  // Cleanup: close fd, release semaphore, free tracker.
  close(tracker->fence_fd);
  iree_async_semaphore_release(tracker->semaphore);
  iree_allocator_free(tracker->allocator, tracker);
}

//===----------------------------------------------------------------------===//
// Fence export timepoint callback
//===----------------------------------------------------------------------===//

// Timepoint callback that signals the exported eventfd when the semaphore
// reaches the target value. On success, writes to the eventfd to make it
// readable. On failure (semaphore failed or cancelled), leaves the fd
// unreadable. Always releases the semaphore and frees the tracker.
//
// This callback fires under the semaphore's internal lock. write() to a
// non-blocking eventfd is ~100-200ns and never blocks, which is within the
// "must be fast" callback contract.
static void iree_async_io_uring_fence_export_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  iree_async_io_uring_fence_export_tracker_t* tracker =
      (iree_async_io_uring_fence_export_tracker_t*)user_data;

  if (iree_status_is_ok(status)) {
    // Semaphore reached the target value. Write to eventfd to make it readable.
    uint64_t value = 1;
    ssize_t result = write(tracker->eventfd, &value, sizeof(value));
    // EAGAIN means counter is already nonzero (redundant signal, benign).
    // Any other failure (EBADF) indicates a lifecycle bug — the eventfd was
    // closed while a timepoint callback was still pending.
    IREE_ASSERT(result >= 0 || errno == EAGAIN);
  } else {
    // Semaphore failed or was cancelled. Leave fd unreadable so consumers
    // observe a timeout or check the semaphore for failure status.
    iree_status_ignore(status);
  }

  // Cleanup: release semaphore, free tracker. The eventfd is caller-owned and
  // must not be closed here.
  iree_async_semaphore_release(tracker->semaphore);
  iree_allocator_free(tracker->allocator, tracker);
}

//===----------------------------------------------------------------------===//
// CQE Processing Helpers
//===----------------------------------------------------------------------===//

// Handles MESSAGE_RECEIVE internal CQE.
// Incoming message from another proactor via MSG_RING.
static inline void iree_async_proactor_io_uring_handle_message_receive_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  // The 64-bit message_data was split across two CQE fields:
  //   - Lower 32 bits: sqe->len → cqe->res
  //   - Upper 32 bits: encoded in sqe->off → cqe->user_data payload
  uint64_t upper_32 = iree_io_uring_internal_payload(cqe->user_data);
  uint64_t message_data = (upper_32 << 32) | (uint32_t)cqe->res;
  if (proactor->message_callback.fn) {
    proactor->message_callback.fn(&proactor->base, message_data,
                                  proactor->message_callback.user_data);
  }
}

// Handles EVENT_SOURCE internal CQE.
// Multishot poll completion for an event source.
static void iree_async_proactor_io_uring_handle_event_source_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  // Extract the event source pointer from the payload.
  iree_async_event_source_t* source =
      (iree_async_event_source_t*)(uintptr_t)iree_io_uring_internal_payload(
          cqe->user_data);
  if (!source) return;

  // Check if this source is being removed (callback.fn set to NULL).
  // If so, we're in "zombie" mode waiting for the final CQE.
  if (source->callback.fn) {
    // Translate native poll events to cross-platform enum.
    // cqe->res contains the poll mask for POLL_ADD completions.
    iree_async_poll_events_t events = IREE_ASYNC_POLL_EVENT_NONE;
    if (cqe->res >= 0) {
      if (cqe->res & POLLIN) events |= IREE_ASYNC_POLL_EVENT_IN;
      if (cqe->res & POLLOUT) events |= IREE_ASYNC_POLL_EVENT_OUT;
      if (cqe->res & POLLERR) events |= IREE_ASYNC_POLL_EVENT_ERR;
      if (cqe->res & POLLHUP) events |= IREE_ASYNC_POLL_EVENT_HUP;
      // POLLRDHUP: peer closed write half (common during clean shutdown).
      // POLLNVAL: fd was revoked or invalid.
      // Both indicate the fd is no longer usable for normal I/O.
      if (cqe->res & POLLRDHUP) events |= IREE_ASYNC_POLL_EVENT_HUP;
      if (cqe->res & POLLNVAL) events |= IREE_ASYNC_POLL_EVENT_ERR;
    } else {
      // Kernel error (EBADF, ECANCELED, etc.) - treat as error event.
      events = IREE_ASYNC_POLL_EVENT_ERR;
    }

    // Invoke the user callback with poll events. The source remains
    // armed (multishot). CQE_F_MORE should be set for multishot poll;
    // if not, the poll was cancelled or the fd has a terminal error.
    source->callback.fn(source->callback.user_data, source, events);
  }

  // Multishot poll produces CQEs until cancelled. CQE_F_MORE indicates
  // more CQEs will follow. When MORE is clear, this is the final CQE
  // (poll cancelled or error). If the source is a zombie, unlink and
  // free it now.
  if (!(cqe->flags & IREE_IORING_CQE_F_MORE) && !source->callback.fn) {
    // Unlink from the proactor's event source list.
    if (source->prev) {
      source->prev->next = source->next;
    } else {
      proactor->event_sources = source->next;
    }
    if (source->next) {
      source->next->prev = source->prev;
    }
    iree_allocator_free(source->allocator, source);
  }
}

// Callback for signalfd dispatch - invoked for each signal read.
static void iree_async_proactor_io_uring_signal_dispatch_callback(
    void* user_data, iree_async_signal_t signal) {
  iree_async_proactor_io_uring_t* proactor =
      (iree_async_proactor_io_uring_t*)user_data;

  // Dispatch to all subscribers for this signal.
  iree_async_signal_subscription_t* head =
      proactor->signal.subscriptions[signal];
  iree_async_signal_subscription_t* to_free =
      iree_async_signal_subscription_dispatch(
          head, &proactor->signal.dispatch_state, signal);

  // Free any subscriptions that were unsubscribed during dispatch.
  while (to_free) {
    iree_async_signal_subscription_t* subscription = to_free;
    iree_async_signal_t signal = subscription->signal;
    to_free = to_free->pending_next;
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);

    // Decrement count and remove signal from mask if last subscriber.
    if (--proactor->signal.subscriber_count[signal] == 0) {
      iree_async_linux_signal_remove_signal(&proactor->signal.linux_state,
                                            signal);
    }
  }
}

// Handles SIGNAL internal CQE.
// Multishot poll completion for the signalfd.
static void iree_async_proactor_io_uring_handle_signal_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  // Only process if signal handling is still active.
  if (!proactor->signal.initialized) return;

  // Read and dispatch all pending signals from signalfd.
  if (cqe->res >= 0 && (cqe->res & POLLIN)) {
    iree_status_t status = iree_async_linux_signal_read_signalfd(
        &proactor->signal.linux_state,
        (iree_async_signal_dispatch_callback_t){
            .fn = iree_async_proactor_io_uring_signal_dispatch_callback,
            .user_data = proactor,
        });
    // Exception: ignore here because failure requires the signalfd to be
    // invalid (EBADF) or a kernel bug - neither of which we can recover from
    // in a CQE handler, and both are effectively impossible in practice.
    iree_status_ignore(status);
  }

  // Multishot poll remains armed until cancelled. CQE_F_MORE indicates more
  // CQEs will follow. If MORE is clear, the poll was cancelled (proactor
  // teardown). No cleanup needed here - destroy() handles it.
}

// Converts a CQE result to an iree_status_t.
// Handles special cases like timer expiration, futex value mismatch, and
// zero-copy send notification CQEs that are not errors.
static iree_status_t iree_async_proactor_io_uring_cqe_to_status(
    const iree_io_uring_cqe_t* cqe, iree_async_operation_t* operation) {
  if (cqe->res >= 0) {
    return iree_ok_status();
  }

  // Timer: -ETIME is the expected success case (timeout expired).
  if (cqe->res == -ETIME &&
      operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) {
    return iree_ok_status();
  }

  // FUTEX_WAIT: -EAGAIN means value mismatch (already changed), not error.
  if (cqe->res == -EAGAIN &&
      operation->type == IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT) {
    return iree_ok_status();
  }

  // NOTIFICATION_WAIT (event mode): -EAGAIN means another waiter drained the
  // eventfd before us (the linked READ got nothing). Treat as success: the
  // epoch may or may not have advanced, and the caller will re-check or
  // re-wait as appropriate.
  if (cqe->res == -EAGAIN &&
      operation->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
    return iree_ok_status();
  }

  // NOTIF CQE for ZC send: res indicates whether ZC was achieved, not error.
  // res=0 means true zero-copy; res=IORING_NOTIF_USAGE_ZC_COPIED (0x80000000)
  // means the kernel fell back to copying. Both are success.
  if (iree_any_bit_set(cqe->flags, IREE_IORING_CQE_F_NOTIF) &&
      (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND ||
       operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO)) {
    return iree_ok_status();
  }

  return iree_make_status(iree_status_code_from_errno(-cqe->res),
                          "io_uring operation failed (%d)", -cqe->res);
}

// Handles SOCKET_ACCEPT completion: imports the accepted fd as a socket.
static iree_status_t iree_async_proactor_io_uring_complete_socket_accept(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_async_socket_accept_operation_t* accept) {
  int accepted_fd = cqe->res;

  // Propagate ZERO_COPY flag from listener.
  // SO_ZEROCOPY is NOT inherited on accept(), so we must set it
  // explicitly on each accepted socket if the listener has it enabled.
  iree_async_socket_flags_t inherited_flags = IREE_ASYNC_SOCKET_FLAG_NONE;
  bool listener_wants_zc = iree_any_bit_set(accept->listen_socket->flags,
                                            IREE_ASYNC_SOCKET_FLAG_ZERO_COPY);
#if defined(SO_ZEROCOPY)
  if (listener_wants_zc) {
    int optval = 1;
    if (setsockopt(accepted_fd, SOL_SOCKET, SO_ZEROCOPY, &optval,
                   sizeof(optval)) == 0) {
      inherited_flags |= IREE_ASYNC_SOCKET_FLAG_ZERO_COPY;
    }
  }
#else
  (void)listener_wants_zc;
#endif  // SO_ZEROCOPY

  iree_async_primitive_t accepted_primitive =
      iree_async_primitive_from_fd(accepted_fd);
  iree_status_t status = iree_async_io_uring_socket_import(
      proactor, accepted_primitive, accept->listen_socket->type,
      inherited_flags, &accept->accepted_socket);
  if (iree_status_is_ok(status)) {
    accept->accepted_socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
  } else {
    // Import failed (e.g., allocation failure). Close the accepted fd.
    close(accepted_fd);
  }
  return status;
}

// Handles SOCKET_RECV_POOL completion: builds the buffer lease from CQE.
static inline void iree_async_proactor_io_uring_complete_socket_recv_pool(
    const iree_io_uring_cqe_t* cqe,
    iree_async_socket_recv_pool_operation_t* recv_pool) {
  if (cqe->res >= 0 && (cqe->flags & IREE_IORING_CQE_F_BUFFER)) {
    uint16_t buffer_index =
        (uint16_t)(cqe->flags >> IREE_IORING_CQE_BUFFER_SHIFT);

    iree_async_region_t* region =
        iree_async_buffer_pool_region(recv_pool->pool);
    IREE_ASSERT(buffer_index < region->buffer_count);
    recv_pool->lease.span = iree_async_span_make(
        region, (iree_host_size_t)buffer_index * region->buffer_size,
        region->buffer_size);
    recv_pool->lease.release = region->recycle;
    recv_pool->lease.buffer_index = (iree_async_buffer_index_t)buffer_index;
    recv_pool->bytes_received = (iree_host_size_t)cqe->res;
  }
}

// Handles SOCKET_RECVFROM completion: extracts sender address.
static inline void iree_async_proactor_io_uring_complete_socket_recvfrom(
    const iree_io_uring_cqe_t* cqe,
    iree_async_socket_recvfrom_operation_t* recvfrom) {
  recvfrom->bytes_received = (iree_host_size_t)cqe->res;
  struct msghdr* msg = (struct msghdr*)recvfrom->platform.posix.msg_header;
  recvfrom->sender.length = (iree_host_size_t)msg->msg_namelen;
}

// Handles NOTIFICATION_SIGNAL completion: populates woken_count.
static inline void iree_async_proactor_io_uring_complete_notification_signal(
    const iree_io_uring_cqe_t* cqe,
    iree_async_notification_signal_operation_t* signal) {
  if (signal->notification->mode == IREE_ASYNC_NOTIFICATION_MODE_FUTEX) {
    signal->woken_count = cqe->res;
  } else {
    // Event mode: write succeeded, but we can't know how many waiters
    // were actually woken (eventfd semantics differ from futex).
    signal->woken_count = -1;
  }
}

// Handles FILE_OPEN completion: imports the opened fd as a file handle.
static iree_status_t iree_async_proactor_io_uring_complete_file_open(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_async_file_open_operation_t* open_op) {
  int fd = cqe->res;
  iree_async_primitive_t primitive = iree_async_primitive_from_fd(fd);
  iree_status_t status =
      iree_async_file_import(&proactor->base, primitive, &open_op->opened_file);
  if (!iree_status_is_ok(status)) {
    // Import failed (e.g., allocation failure). Close the opened fd.
    close(fd);
  }
  return status;
}

// Populates operation result fields from a successful CQE.
// Returns updated status (may fail for SOCKET_ACCEPT or FILE_OPEN if import
// fails).
static iree_status_t iree_async_proactor_io_uring_populate_result(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      iree_async_socket_connect_operation_t* connect =
          (iree_async_socket_connect_operation_t*)operation;
      connect->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      return iree_async_proactor_io_uring_complete_socket_accept(
          proactor, cqe, (iree_async_socket_accept_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      recv->bytes_received = (iree_host_size_t)cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      iree_async_proactor_io_uring_complete_socket_recv_pool(
          cqe, (iree_async_socket_recv_pool_operation_t*)operation);
      break;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      iree_async_socket_send_operation_t* send =
          (iree_async_socket_send_operation_t*)operation;
      // Only update bytes_sent on first CQE (not on NOTIF).
      if (!(cqe->flags & IREE_IORING_CQE_F_NOTIF)) {
        send->bytes_sent = (iree_host_size_t)cqe->res;
      }
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      iree_async_socket_sendto_operation_t* sendto =
          (iree_async_socket_sendto_operation_t*)operation;
      if (!(cqe->flags & IREE_IORING_CQE_F_NOTIF)) {
        sendto->bytes_sent = (iree_host_size_t)cqe->res;
      }
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      iree_async_proactor_io_uring_complete_socket_recvfrom(
          cqe, (iree_async_socket_recvfrom_operation_t*)operation);
      break;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE: {
      iree_async_socket_close_operation_t* close_op =
          (iree_async_socket_close_operation_t*)operation;
      close_op->socket->state = IREE_ASYNC_SOCKET_STATE_CLOSED;
      close_op->socket->primitive.value.fd = -1;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE: {
      iree_async_futex_wake_operation_t* futex_wake =
          (iree_async_futex_wake_operation_t*)operation;
      futex_wake->woken_count = cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL:
      iree_async_proactor_io_uring_complete_notification_signal(
          cqe, (iree_async_notification_signal_operation_t*)operation);
      break;
    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
      return iree_async_proactor_io_uring_complete_file_open(
          proactor, cqe, (iree_async_file_open_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      read_op->bytes_read = (iree_host_size_t)cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      write_op->bytes_written = (iree_host_size_t)cqe->res;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE: {
      iree_async_file_close_operation_t* close_op =
          (iree_async_file_close_operation_t*)operation;
      close_op->file->primitive.value.fd = -1;
      break;
    }
    default:
      break;
  }
  return iree_ok_status();
}

// Checks if a zero-copy send is waiting for its NOTIF CQE.
// Returns true if the callback should be deferred until NOTIF arrives.
static inline bool iree_async_proactor_io_uring_is_zc_send_deferred(
    const iree_io_uring_cqe_t* cqe, iree_async_operation_t* operation) {
  if (operation->type != IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND &&
      operation->type != IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO) {
    return false;
  }

  iree_async_socket_t* socket = NULL;
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND) {
    socket = ((iree_async_socket_send_operation_t*)operation)->socket;
  } else {
    socket = ((iree_async_socket_sendto_operation_t*)operation)->socket;
  }

  bool is_zc =
      iree_any_bit_set(socket->flags, IREE_ASYNC_SOCKET_FLAG_ZERO_COPY);
  bool has_more = (cqe->flags & IREE_IORING_CQE_F_MORE) != 0;
  bool is_notif = (cqe->flags & IREE_IORING_CQE_F_NOTIF) != 0;

  // ZC send with CQE_F_MORE but not yet NOTIF: defer callback.
  return is_zc && has_more && !is_notif;
}

// Computes completion flags from CQE and operation type.
static inline iree_async_completion_flags_t
iree_async_proactor_io_uring_completion_flags(
    const iree_io_uring_cqe_t* cqe, iree_async_operation_t* operation) {
  iree_async_completion_flags_t flags = IREE_ASYNC_COMPLETION_FLAG_NONE;

  if (iree_any_bit_set(cqe->flags, IREE_IORING_CQE_F_MORE)) {
    flags |= IREE_ASYNC_COMPLETION_FLAG_MORE;
  }

  // For ZC send NOTIF CQEs, check if zero-copy was actually achieved.
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND ||
      operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO) {
    bool is_notif = iree_any_bit_set(cqe->flags, IREE_IORING_CQE_F_NOTIF);
    if (is_notif && cqe->res == 0) {
      flags |= IREE_ASYNC_COMPLETION_FLAG_ZERO_COPY_ACHIEVED;
    }
  }

  return flags;
}

// Handles internal CQEs (wake poll, cancel, fence import, message, event).
// Returns false since internal ops don't count as user completions.
static inline bool iree_async_proactor_io_uring_process_internal_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  switch (iree_io_uring_internal_tag(cqe->user_data)) {
    case IREE_IO_URING_TAG_WAKE:
      iree_async_proactor_io_uring_handle_wake_completion(proactor);
      break;
    case IREE_IO_URING_TAG_CANCEL:
      // Cancel operation completed. The result indicates whether the cancel
      // succeeded (0), the operation was not found (-ENOENT), or was already
      // completing (-EALREADY). We don't need to do anything here since the
      // target operation's CQE will deliver the final status to its callback.
      break;
    case IREE_IO_URING_TAG_FENCE_IMPORT:
      iree_async_proactor_io_uring_handle_fence_import_completion(proactor,
                                                                  cqe);
      break;
    case IREE_IO_URING_TAG_MESSAGE_RECEIVE:
      iree_async_proactor_io_uring_handle_message_receive_cqe(proactor, cqe);
      break;
    case IREE_IO_URING_TAG_MESSAGE_SOURCE:
      // Source-side completion of a message send where the user requested
      // SKIP_SOURCE_COMPLETION (fire-and-forget semantics).
      //
      // We get CQEs here when:
      // - FEAT_CQE_SKIP is not available (kernel always generates source CQE)
      // - The operation failed (error CQEs bypass CQE_SKIP)
      //
      // Fire-and-forget means the user explicitly chose to not be notified
      // of either success or failure, so we discard the result.
      break;
    case IREE_IO_URING_TAG_EVENT_SOURCE:
      iree_async_proactor_io_uring_handle_event_source_cqe(proactor, cqe);
      break;
    case IREE_IO_URING_TAG_RELAY: {
      // Extract the relay pointer from the payload.
      iree_async_relay_t* relay =
          (iree_async_relay_t*)(uintptr_t)iree_io_uring_internal_payload(
              cqe->user_data);
      iree_async_io_uring_handle_relay_cqe(proactor, relay, cqe->res,
                                           cqe->flags);
      break;
    }
    case IREE_IO_URING_TAG_SIGNAL:
      iree_async_proactor_io_uring_handle_signal_cqe(proactor, cqe);
      break;
    default:
      IREE_ASSERT(false,
                  "unrecognized internal tag %d in CQE (user_data=0x%" PRIx64
                  ", res=%d, flags=0x%x)",
                  (int)iree_io_uring_internal_tag(cqe->user_data),
                  cqe->user_data, cqe->res, cqe->flags);
      break;
  }
  return false;  // Internal ops don't count as user completions.
}

// Returns the socket for a socket I/O operation that should propagate sticky
// failure on error. Returns NULL for non-socket operations and for types where
// errors don't indicate a broken socket (accept errors reflect transient
// resource issues, not a broken listener; close errors are moot).
static iree_async_socket_t*
iree_async_proactor_io_uring_socket_from_io_operation(
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

//===----------------------------------------------------------------------===//
// Poll
//===----------------------------------------------------------------------===//

// Processes a single CQE, invoking the operation's callback.
// Returns the number of user completions (typically 1, but can be more if
// continuation callbacks were invoked directly; 0 if suppressed, e.g., first
// CQE of a zero-copy send waiting for NOTIF).
static iree_host_size_t iree_async_proactor_io_uring_process_cqe(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  // Linked POLL_ADD head CQE for EVENT_WAIT or NOTIFICATION_WAIT (event mode).
  // Always ignored — the linked READ CQE handles resource release and user
  // callback dispatch for both success (READ returns data) and failure
  // (READ gets -ECANCELED when POLL_ADD fails).
  if (iree_io_uring_is_internal_cqe(cqe) &&
      iree_io_uring_internal_tag(cqe->user_data) ==
          IREE_IO_URING_TAG_LINKED_POLL) {
    return 0;
  }

  // Handle internal operations (wake poll, cancel, fence import, message).
  if (iree_io_uring_is_internal_cqe(cqe)) {
    iree_async_proactor_io_uring_process_internal_cqe(proactor, cqe);
    return 0;
  }

  // User operation: extract the operation pointer.
  iree_async_operation_t* operation =
      (iree_async_operation_t*)(uintptr_t)cqe->user_data;

  // Convert kernel result to status.
  iree_status_t status =
      iree_async_proactor_io_uring_cqe_to_status(cqe, operation);

  // Populate result fields on success.
  if (iree_status_is_ok(status)) {
    status =
        iree_async_proactor_io_uring_populate_result(proactor, cqe, operation);
  }

  // Propagate error to socket's sticky failure status.
  if (!iree_status_is_ok(status)) {
    iree_async_socket_t* socket =
        iree_async_proactor_io_uring_socket_from_io_operation(operation);
    if (socket) {
      iree_async_socket_set_failure(socket, status);
    }
  }

  // Zero-copy send handling: first CQE has MORE set, defer until NOTIF.
  if (iree_async_proactor_io_uring_is_zc_send_deferred(cqe, operation)) {
    iree_status_ignore(status);
    return 0;
  }

  // Detach LINKED continuation chain before callback (callback may free the
  // operation). Continuations are dispatched AFTER the trigger's callback to
  // preserve callback ordering: trigger first, then its continuations.
  iree_async_operation_t* continuation = operation->linked_next;
  operation->linked_next = NULL;

  // Compute completion flags.
  iree_async_completion_flags_t flags =
      iree_async_proactor_io_uring_completion_flags(cqe, operation);

  // Release resources retained during submission (not for multishot).
  // Must happen BEFORE callback since callback may free the operation.
  if (!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
    iree_async_operation_release_resources(operation);
  }

  // Invoke callback. Note: callback may free the operation, so no access to
  // operation is allowed after this point.
  operation->completion_fn(operation->user_data, operation, status, flags);

  // Dispatch continuation chain after the trigger's callback. Software
  // completions in the chain are pushed to the MPSC queue (counted by the
  // post-CQE MPSC drain in the poll loop). Kernel ops are submitted and
  // produce their own CQEs.
  if (continuation) {
    if (iree_status_is_ok(status)) {
      iree_async_proactor_io_uring_dispatch_continuation_chain(proactor,
                                                               continuation);
    } else {
      iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
          proactor, continuation);
    }
  }

  return 1;
}

static iree_status_t iree_async_proactor_io_uring_poll(
    iree_async_proactor_t* base_proactor, iree_timeout_t timeout,
    iree_host_size_t* out_completed_count) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (out_completed_count) *out_completed_count = 0;

  bool is_immediate = iree_timeout_is_immediate(timeout);

  // Arm the wake poll before potentially blocking.
  IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_arm_wake(proactor));

  // Flush pending SQEs and deferred completions. With DEFER_TASKRUN, the kernel
  // defers CQE generation for async notifications (e.g., recv completions
  // triggered by completed sends on the same ring) until we explicitly request
  // processing via GETEVENTS. This flush must happen unconditionally — even
  // when CQEs are already in the CQ ring from inline SQE processing — because
  // those inline completions may have triggered deferred work that produced
  // additional CQEs we haven't seen yet.
  IREE_RETURN_IF_ERROR(
      iree_io_uring_ring_submit(&proactor->ring,
                                /*min_complete=*/0,
                                /*flags=*/IREE_IORING_ENTER_GETEVENTS));

  // If no CQEs are available after flushing and timeout allows blocking,
  // wait for completions.
  if (!iree_io_uring_ring_cq_ready(&proactor->ring) && !is_immediate) {
    iree_duration_t timeout_ns = iree_timeout_as_duration_ns(timeout);
    iree_status_t wait_status =
        iree_io_uring_ring_wait_cqe(&proactor->ring, /*min_complete=*/1,
                                    /*flush_pending=*/true, timeout_ns);
    if (iree_status_is_deadline_exceeded(wait_status)) {
      iree_status_ignore(wait_status);
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    IREE_RETURN_IF_ERROR(wait_status);
  }

  // First MPSC drain: software completions from submit threads.
  // Software ops pushed to the MPSC by submit threads have their side effects
  // already done; only callback delivery is deferred. Draining these first
  // preserves callback ordering for chains like SIGNAL(LINKED) → RECV: the
  // SIGNAL callback fires before the RECV CQE callback.
  // A second drain occurs after CQE processing and semaphore wait dispatch
  // to catch software completions pushed during those phases.
  iree_host_size_t completed =
      iree_async_proactor_io_uring_drain_pending_software_completions(proactor);

  // Defer kernel entry during CQE processing. Operations submitted by CQE
  // callbacks (e.g., multishot recv re-arm on ENOBUFS) fill SQEs in the
  // submission ring but skip io_uring_enter. Without this, a synchronous
  // completion (kernel processes the SQE inline and posts a CQE immediately)
  // creates an infinite loop: the CQE loop processes the new CQE, which
  // submits another SQE, which generates another CQE, etc.
  proactor->defer_submissions = true;

  // Process all available CQEs. peek_cqe returns NULL when the CQ is empty,
  // combining the readiness check and CQE retrieval in a single atomic load.
  iree_io_uring_cqe_t* cqe;
  while ((cqe = iree_io_uring_ring_peek_cqe(&proactor->ring)) != NULL) {
    // Process the CQE and add user completions to count.
    // Includes this operation's callback plus any directly-invoked continuation
    // callbacks. Returns 0 for suppressed CQEs (e.g., ZC send waiting for
    // NOTIF).
    completed += iree_async_proactor_io_uring_process_cqe(proactor, cqe);
    iree_io_uring_ring_cq_advance(&proactor->ring, 1);
  }

  proactor->defer_submissions = false;

  // Second MPSC drain: software completions pushed during the main CQE loop.
  // CQE processing dispatches continuation chains that push software
  // completions to the MPSC. These must fire BEFORE the CQE drain loop flushes
  // deferred SQEs, because the deferred SQEs (submitted by continuation
  // dispatch) are later in the chain. Example: [RECV → SIGNAL → SEND] —
  // SIGNAL's callback must fire before SEND's CQE is processed.
  completed +=
      iree_async_proactor_io_uring_drain_pending_software_completions(proactor);

  // Flush SQEs deferred during CQE processing and process any resulting CQEs.
  // Each flush may generate CQEs (inline from SQE processing, or deferred
  // task_work flushed by GETEVENTS), and processing those CQEs may defer more
  // SQEs. Loop until a flush produces no new CQEs.
  //
  // GETEVENTS is required here because DEFER_TASKRUN defers async completions
  // (e.g., multishot recv triggered by a send on the same ring) as task_work.
  // Without GETEVENTS, io_uring_enter only processes inline completions, and
  // the deferred CQEs survive past poll(). If the operations they reference
  // are freed between polls (carrier destroyed after deactivation), this
  // produces a use-after-free.
  //
  // The loop is bounded: each pass processes at least one CQE, and the total
  // number of in-flight operations is finite. The 32-pass limit is a safety
  // net against pathological cascading (e.g., persistent ENOBUFS on an empty
  // buffer ring).
  for (int drain_pass = 0; drain_pass < 32; ++drain_pass) {
    IREE_RETURN_IF_ERROR(
        iree_io_uring_ring_submit(&proactor->ring,
                                  /*min_complete=*/0,
                                  /*flags=*/IREE_IORING_ENTER_GETEVENTS));
    cqe = iree_io_uring_ring_peek_cqe(&proactor->ring);
    if (!cqe) break;
    // Defer submissions during processing to prevent io_uring_enter from
    // generating synchronous CQEs that the loop would pick up immediately
    // (the same infinite-loop prevention as the primary CQE loop above).
    proactor->defer_submissions = true;
    do {
      completed += iree_async_proactor_io_uring_process_cqe(proactor, cqe);
      iree_io_uring_ring_cq_advance(&proactor->ring, 1);
    } while ((cqe = iree_io_uring_ring_peek_cqe(&proactor->ring)) != NULL);
    proactor->defer_submissions = false;
  }

  // Drain pending messages from the fallback MPSC queue.
  // This handles messages that arrived via eventfd wake rather than MSG_RING.
  iree_async_proactor_io_uring_drain_pending_messages(proactor);

  // Drain pending semaphore wait completions.
  // These are pushed by timepoint callbacks when semaphores reach target
  // values. Count them as completions since they invoke user callbacks.
  completed +=
      iree_async_proactor_io_uring_drain_pending_semaphore_waits(proactor);

  // Third MPSC drain: software completions pushed during the CQE drain loop
  // and semaphore wait dispatch. Each drain pass in the CQE drain loop may
  // dispatch continuations that push software completions. Semaphore wait
  // dispatch similarly pushes continuation completions.
  completed +=
      iree_async_proactor_io_uring_drain_pending_software_completions(proactor);

  // Retry re-arming relays that failed due to SQ backpressure.
  // Now that we've processed CQEs, there may be SQ space available.
  iree_async_io_uring_retry_pending_relays(proactor);

  if (out_completed_count) *out_completed_count = completed;

  // Return DEADLINE_EXCEEDED for immediate poll with no completions.
  if (completed == 0 && is_immediate) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Wake
//===----------------------------------------------------------------------===//

static void iree_async_proactor_io_uring_wake(
    iree_async_proactor_t* base_proactor) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (proactor->wake_eventfd < 0) return;

  // Write to eventfd to wake a blocked poll. This is thread-safe and
  // signal-safe. The value accumulates if wake() is called multiple times
  // before the eventfd is read.
  uint64_t value = 1;
  ssize_t ret = write(proactor->wake_eventfd, &value, sizeof(value));
  (void)ret;  // Ignore errors; eventfd writes only fail if the counter would
              // overflow (extremely unlikely with uint64_t).
}

//===----------------------------------------------------------------------===//
// Cancel
//===----------------------------------------------------------------------===//

// Cancels a SEMAPHORE_WAIT operation by cancelling its timepoints.
// SEMAPHORE_WAIT uses software-only timepoints, not kernel operations, so we
// can't use ASYNC_CANCEL. Instead, we cancel the timepoints directly and
// enqueue the tracker for completion with CANCELLED status.
static iree_status_t iree_async_proactor_io_uring_cancel_semaphore_wait(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_semaphore_wait_operation_t* wait_op) {
  // Get the tracker from the operation's platform storage.
  iree_async_io_uring_semaphore_wait_tracker_t* tracker =
      (iree_async_io_uring_semaphore_wait_tracker_t*)wait_op->base.next;
  if (!tracker) {
    // No tracker means the operation was never submitted or already completed.
    return iree_ok_status();
  }

  // Try to set CANCELLED status. If another callback already set a status,
  // this is a no-op (first status wins).
  intptr_t expected = (intptr_t)iree_ok_status();
  iree_status_t cancelled_status =
      iree_make_status(IREE_STATUS_CANCELLED, "operation cancelled");
  if (!iree_atomic_compare_exchange_strong(
          &tracker->completion_status, &expected, (intptr_t)cancelled_status,
          iree_memory_order_acq_rel, iree_memory_order_acquire)) {
    // Another callback already set a status. The tracker will be (or has been)
    // enqueued for completion by that callback.
    iree_status_ignore(cancelled_status);
    return iree_ok_status();
  }

  // Cancel all registered timepoints. This prevents them from firing later.
  for (iree_host_size_t i = 0; i < tracker->count; ++i) {
    iree_async_semaphore_cancel_timepoint(wait_op->semaphores[i],
                                          &tracker->timepoints[i]);
  }

  // Enqueue the tracker for completion.
  iree_atomic_slist_push(&proactor->pending_semaphore_waits,
                         &tracker->slist_entry);

  // Wake the proactor to process the cancellation.
  uint64_t wake_value = 1;
  ssize_t result =
      write(proactor->wake_eventfd, &wake_value, sizeof(wake_value));
  IREE_ASSERT(result >= 0 || errno == EAGAIN);

  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_cancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  // SEMAPHORE_WAIT uses software-only timepoints, not kernel operations.
  // Handle cancellation specially.
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT) {
    return iree_async_proactor_io_uring_cancel_semaphore_wait(
        proactor, (iree_async_semaphore_wait_operation_t*)operation);
  }

  // SEQUENCE manages its own step tracking. Cancel propagates through the
  // shared sequence_emulation cancel function, which cancels the current
  // in-flight step via the proactor's normal cancel path (re-entering this
  // function for the step's operation type).
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    return iree_async_sequence_cancel(
        base_proactor, (iree_async_sequence_operation_t*)operation);
  }

  // Get an SQE for the cancel request.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full, cannot submit cancel");
  }

  // Fill the cancel SQE.
  // ASYNC_CANCEL targets operations by their user_data. The cancel SQE uses an
  // internal token so its CQE is handled silently (the target operation's CQE
  // delivers the actual result).
  //
  // EVENT_WAIT and NOTIFICATION_WAIT use linked POLL_ADD+READ pairs where:
  //   POLL_ADD user_data = TAG_LINKED_POLL encoded with operation pointer
  //   READ user_data = operation pointer
  //
  // Cancel must target the POLL_ADD because the linked READ hasn't been started
  // by the kernel (it waits behind the POLL_ADD link). The kernel does not
  // generate a CQE for the unstarted linked READ, so the TAG_LINKED_POLL
  // handler dispatches the user callback from the POLL_ADD's error CQE.
  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_ASYNC_CANCEL;
  sqe->fd = -1;
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT ||
      operation->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
    sqe->addr = iree_io_uring_internal_encode(IREE_IO_URING_TAG_LINKED_POLL,
                                              (uintptr_t)operation);
  } else {
    sqe->addr = (uint64_t)(uintptr_t)operation;
  }
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL, 0);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // Wake the poll thread to submit the pending ASYNC_CANCEL SQE.
  // Only the poll thread may call io_uring_enter (SINGLE_ISSUER constraint),
  // and cancel() can be called from any thread. The SQE is fully prepared in
  // the ring; the next io_uring_enter (from poll or from the same thread's
  // PollUntil) will submit it.
  iree_async_proactor_wake(&proactor->base);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Socket
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_create_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_socket_create(proactor, type, options, out_socket);
}

static iree_status_t iree_async_proactor_io_uring_import_socket(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_socket_import(proactor, primitive, type, flags,
                                           out_socket);
}

static void iree_async_proactor_io_uring_destroy_socket(
    iree_async_proactor_t* base_proactor, iree_async_socket_t* socket) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_io_uring_socket_destroy(proactor, socket);
}

//===----------------------------------------------------------------------===//
// File vtable implementations
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_import_file(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t primitive,
    iree_async_file_t** out_file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_file);
  *out_file = NULL;

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
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
  file->fixed_file_index = -1;  // Not using io_uring fixed files yet.

  IREE_TRACE(
      { snprintf(file->debug_path, sizeof(file->debug_path), "fd:%d", fd); });

  *out_file = file;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_io_uring_destroy_file(
    iree_async_proactor_t* base_proactor, iree_async_file_t* file) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(file);

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Close the file descriptor if still open. The fd may already be -1 if
  // a FILE_CLOSE operation was submitted before the final release.
  int fd = file->primitive.value.fd;
  if (fd >= 0) {
    close(fd);
  }

  iree_allocator_free(allocator, file);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_async_proactor_io_uring_create_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t** out_event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = NULL;

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Allocate and zero-initialize the event structure.
  iree_async_event_t* event = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(allocator, sizeof(*event), (void**)&event));
  memset(event, 0, sizeof(*event));

  // Create the eventfd. EFD_CLOEXEC prevents leakage to child processes.
  // EFD_NONBLOCK is required for io_uring POLL_ADD (avoids blocking on read).
  int eventfd_result = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (eventfd_result < 0) {
    iree_allocator_free(allocator, event);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed (%d)", errno);
  }

  // Initialize the event structure.
  iree_atomic_ref_count_init(&event->ref_count);
  event->proactor = base_proactor;
  event->primitive = iree_async_primitive_from_fd(eventfd_result);
  // On Linux, signal_primitive is the same as primitive (eventfd is
  // bidirectional: write to signal, read to drain).
  event->signal_primitive = event->primitive;
  event->fixed_file_index = -1;
  event->pool = NULL;
  event->pool_next = NULL;
  event->pool_all_next = NULL;

  *out_event = event;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_io_uring_destroy_event(
    iree_async_proactor_t* base_proactor, iree_async_event_t* event) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (!event) {
    IREE_TRACE_ZONE_END(z0);
    return;
  }

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Close the eventfd. On Linux, primitive == signal_primitive, so close once.
  if (event->primitive.value.fd >= 0) {
    close(event->primitive.value.fd);
  }

  iree_allocator_free(allocator, event);
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Notification vtable implementation
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_create_notification(
    iree_async_proactor_t* base_proactor, iree_async_notification_flags_t flags,
    iree_async_notification_t** out_notification) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_notification_create(proactor, flags,
                                                 out_notification);
}

static void iree_async_proactor_io_uring_destroy_notification(
    iree_async_proactor_t* base_proactor,
    iree_async_notification_t* notification) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_io_uring_notification_destroy(proactor, notification);
}

static iree_status_t iree_async_proactor_io_uring_import_fence(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t fence,
    iree_async_semaphore_t* semaphore, uint64_t signal_value) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  // Validate: must be a POSIX fd with a valid descriptor.
  // On validation error the caller retains fd ownership.
  if (fence.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "import_fence requires an FD primitive (got type %d)", (int)fence.type);
  }
  if (fence.value.fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "import_fence fd must be >= 0 (got %d)",
                            fence.value.fd);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  // Get an SQE for POLL_ADD on the fence fd.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full, cannot submit fence import POLL_ADD");
  }

  // Allocate a tracker to associate the fd with the semaphore.
  iree_async_io_uring_fence_import_tracker_t* tracker = NULL;
  iree_status_t alloc_status = iree_allocator_malloc(
      proactor->base.allocator, sizeof(*tracker), (void**)&tracker);
  if (!iree_status_is_ok(alloc_status)) {
    iree_io_uring_ring_sq_rollback(&proactor->ring, 1);
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    IREE_TRACE_ZONE_END(z0);
    return alloc_status;
  }

  // Retain the semaphore so it survives until the CQE fires.
  iree_async_semaphore_retain(semaphore);
  tracker->semaphore = semaphore;
  tracker->signal_value = signal_value;
  tracker->fence_fd = fence.value.fd;
  tracker->allocator = proactor->base.allocator;

  // Fill the POLL_ADD SQE.
  sqe->opcode = IREE_IORING_OP_POLL_ADD;
  sqe->fd = fence.value.fd;
  sqe->poll32_events = POLLIN;
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_FENCE_IMPORT,
                                                 (uintptr_t)tracker);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // Wake the poll thread to flush the SQE. Do NOT call ring_submit()
  // directly — only the poll thread may call io_uring_enter (SINGLE_ISSUER).
  // The SQE is already committed to sq_local_tail and will be included in
  // the next ring_submit() call during poll().
  iree_async_proactor_wake(&proactor->base);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_export_fence(
    iree_async_proactor_t* base_proactor, iree_async_semaphore_t* semaphore,
    uint64_t wait_value, iree_async_primitive_t* out_fence) {
  IREE_ASSERT_ARGUMENT(out_fence);
  *out_fence = iree_async_primitive_none();

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  IREE_TRACE_ZONE_BEGIN(z0);

  // Create a non-blocking eventfd. The caller will own this fd and can poll it,
  // pass it to Vulkan (VK_KHR_external_semaphore_fd), HIP, DRM/KMS, or use it
  // in io_uring LINK chains.
  int efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "eventfd creation failed for export_fence");
  }

  // Allocate a tracker to bridge the semaphore timepoint to the eventfd.
  iree_async_io_uring_fence_export_tracker_t* tracker = NULL;
  iree_status_t status = iree_allocator_malloc(
      proactor->base.allocator, sizeof(*tracker), (void**)&tracker);
  if (!iree_status_is_ok(status)) {
    close(efd);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Retain the semaphore so it survives until the timepoint callback fires.
  // The timepoint API does not hold a reference.
  iree_async_semaphore_retain(semaphore);
  tracker->semaphore = semaphore;
  tracker->eventfd = efd;
  tracker->allocator = proactor->base.allocator;

  // Set callback on the embedded timepoint before registering.
  tracker->timepoint.callback = iree_async_io_uring_fence_export_callback;
  tracker->timepoint.user_data = tracker;

  // Register the timepoint. If the semaphore has already reached wait_value,
  // the callback fires synchronously (writing to eventfd before we return).
  // If the semaphore is already failed, the callback fires with the failure
  // status (eventfd stays unreadable). acquire_timepoint always returns OK.
  status = iree_async_semaphore_acquire_timepoint(semaphore, wait_value,
                                                  &tracker->timepoint);

  if (iree_status_is_ok(status)) {
    // Return the eventfd as the exported fence. Caller owns the fd.
    *out_fence = iree_async_primitive_from_fd(efd);
  } else {
    // Should not happen (acquire_timepoint only fails on NULL semaphore), but
    // handle defensively.
    close(efd);
    iree_async_semaphore_release(semaphore);
    iree_allocator_free(proactor->base.allocator, tracker);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Event source registration
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_register_event_source(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_event_source);
  *out_event_source = NULL;

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

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

  // Get an SQE for multishot POLL_ADD.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    iree_allocator_free(proactor->base.allocator, source);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full, cannot submit event source POLL_ADD");
  }

  // Fill multishot POLL_ADD SQE.
  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_POLL_ADD;
  sqe->fd = source->fd;
  sqe->poll32_events = POLLIN;
  // Use multishot mode (POLL_ADD_MULTI) so the poll stays armed after each
  // completion. This requires kernel 5.19+.
  sqe->len = IREE_IORING_POLL_ADD_MULTI;
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_EVENT_SOURCE,
                                                 (uintptr_t)source);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // Flush the SQE to the kernel so the POLL_ADD begins monitoring and the SQ
  // slot is reclaimed for subsequent submissions. If ring_submit fails, the
  // SQE is already committed to the kernel via *sq_tail (which ring_submit
  // advances before calling io_uring_enter) and will be processed on the next
  // successful io_uring_enter. We must NOT attempt to rollback the SQE after
  // ring_submit because *sq_tail has already been advanced — rollback would
  // desync sq_local_tail from *sq_tail, corrupting the ring.
  iree_status_ignore(iree_io_uring_ring_submit(&proactor->ring,
                                               /*min_complete=*/0,
                                               /*flags=*/0));

  // Link into the proactor's event source list.
  source->next = proactor->event_sources;
  if (proactor->event_sources) {
    proactor->event_sources->prev = source;
  }
  proactor->event_sources = source;

  *out_event_source = source;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_async_proactor_io_uring_unregister_event_source(
    iree_async_proactor_t* base_proactor,
    iree_async_event_source_t* event_source) {
  if (!event_source) return;

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  // Mark as "zombie" - callback won't be invoked, but struct stays alive
  // until the final CQE arrives (when POLL_REMOVE completes).
  // The source stays in the list so destroy() can clean up any stragglers.
  event_source->callback.fn = NULL;

  // Submit POLL_REMOVE to cancel the multishot poll.
  // The poll was submitted with user_data = internal_encode(TAG, source).
  // POLL_REMOVE identifies polls by their user_data.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (sqe) {
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IREE_IORING_OP_POLL_REMOVE;
    sqe->fd = -1;
    // Target the poll by its user_data value.
    sqe->addr = iree_io_uring_internal_encode(IREE_IO_URING_TAG_EVENT_SOURCE,
                                              (uintptr_t)event_source);
    // Mark the cancel CQE as internal so we don't confuse it with user ops.
    sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL, 0);
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  if (sqe) {
    // Submit. Best-effort - if this fails, the poll will continue until the
    // proactor is destroyed.
    iree_status_t status = iree_io_uring_ring_submit(&proactor->ring,
                                                     /*min_complete=*/0,
                                                     /*flags=*/0);
    iree_status_ignore(status);
  }

  // Note: We do NOT free or unlink the event_source here. The CQE handler will
  // unlink and free it when the final CQE arrives (POLL_REMOVE completion with
  // !CQE_F_MORE). This prevents use-after-free when there are pending CQEs in
  // the queue. If the proactor is destroyed first, destroy() cleans up the
  // list.
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Relay vtable wrappers
//===----------------------------------------------------------------------===//

static iree_status_t iree_async_proactor_io_uring_register_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_source_t source,
    iree_async_relay_sink_t sink, iree_async_relay_flags_t flags,
    iree_async_relay_error_callback_t error_callback,
    iree_async_relay_t** out_relay) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  return iree_async_io_uring_register_relay(proactor, source, sink, flags,
                                            error_callback, out_relay);
}

static void iree_async_proactor_io_uring_unregister_relay(
    iree_async_proactor_t* base_proactor, iree_async_relay_t* relay) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_io_uring_unregister_relay(proactor, relay);
}

//===----------------------------------------------------------------------===//
// Signal handling
//===----------------------------------------------------------------------===//

// Submits a multishot POLL_ADD for the signalfd. Called when the first signal
// subscription is added and we have a signalfd to monitor.
static iree_status_t iree_async_proactor_io_uring_submit_signal_poll(
    iree_async_proactor_io_uring_t* proactor, int signal_fd) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Get an SQE for multishot POLL_ADD on the signalfd.
  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  iree_status_t status =
      sqe ? iree_ok_status()
          : iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                             "SQ full, cannot submit signal poll");

  // Fill POLL_ADD SQE for multishot monitoring.
  if (iree_status_is_ok(status)) {
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IREE_IORING_OP_POLL_ADD;
    sqe->fd = signal_fd;
    sqe->poll32_events = POLLIN;
    sqe->len = IREE_IORING_POLL_ADD_MULTI;
    sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_SIGNAL, 0);
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // Flush the SQE to the kernel (see register_event_source for rationale).
  // ring_submit errors are ignored because the SQE is already committed via
  // *sq_tail and will be processed on the next successful io_uring_enter.
  if (iree_status_is_ok(status)) {
    iree_status_ignore(iree_io_uring_ring_submit(&proactor->ring,
                                                 /*min_complete=*/0,
                                                 /*flags=*/0));
    proactor->signal.initialized = true;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_async_proactor_io_uring_subscribe_signal(
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

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Claim global signal ownership (first proactor wins).
  iree_status_t status = iree_async_signal_claim_ownership(base_proactor);

  // If this is the first subscriber for this specific signal, add it to the
  // signalfd mask. This blocks only the signals that actually have subscribers.
  int signal_fd = -1;
  if (iree_status_is_ok(status) &&
      proactor->signal.subscriber_count[signal] == 0) {
    status = iree_async_linux_signal_add_signal(&proactor->signal.linux_state,
                                                signal, &signal_fd);
    if (!iree_status_is_ok(status)) {
      // Failed to add signal - release ownership if we were the first.
      iree_async_signal_release_ownership(base_proactor);
    }
  }

  // If this is the first signal subscription overall, submit POLL_ADD.
  if (iree_status_is_ok(status) && !proactor->signal.initialized &&
      signal_fd >= 0) {
    status =
        iree_async_proactor_io_uring_submit_signal_poll(proactor, signal_fd);
    if (!iree_status_is_ok(status)) {
      // Failed to submit poll - remove the signal we just added.
      iree_async_linux_signal_remove_signal(&proactor->signal.linux_state,
                                            signal);
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

static void iree_async_proactor_io_uring_unsubscribe_signal(
    iree_async_proactor_t* base_proactor,
    iree_async_signal_subscription_t* subscription) {
  if (!subscription) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_signal_t signal = subscription->signal;

  // If we're currently dispatching signals, defer the unsubscribe to avoid
  // corrupting the iteration. The subscription will be freed after dispatch.
  if (proactor->signal.dispatch_state.dispatching) {
    iree_async_signal_subscription_defer_unsubscribe(
        &proactor->signal.dispatch_state, subscription);
    // Note: count decrement and remove_signal happen in dispatch callback
    // when the deferred subscription is actually processed.
  } else {
    // Immediate unsubscribe: unlink, decrement count, and free.
    iree_async_signal_subscription_unlink(
        &proactor->signal.subscriptions[signal], subscription);
    iree_allocator_free(proactor->base.allocator, subscription);

    // If this was the last subscriber for this signal, remove it from the
    // signalfd mask so default behavior is restored.
    if (--proactor->signal.subscriber_count[signal] == 0) {
      iree_async_linux_signal_remove_signal(&proactor->signal.linux_state,
                                            signal);
    }
  }

  IREE_TRACE_ZONE_END(z0);
}

static void iree_async_proactor_io_uring_set_message_callback(
    iree_async_proactor_t* base_proactor,
    iree_async_proactor_message_callback_t callback) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  proactor->message_callback = callback;
}

static iree_status_t iree_async_proactor_io_uring_send_message(
    iree_async_proactor_t* base_target, uint64_t message_data) {
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(base_target);
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

const iree_async_proactor_vtable_t iree_async_proactor_io_uring_vtable = {
    .destroy = iree_async_proactor_io_uring_destroy,
    .query_capabilities = iree_async_proactor_io_uring_query_capabilities,
    .submit = iree_async_proactor_io_uring_submit,
    .poll = iree_async_proactor_io_uring_poll,
    .wake = iree_async_proactor_io_uring_wake,
    .cancel = iree_async_proactor_io_uring_cancel,
    .create_socket = iree_async_proactor_io_uring_create_socket,
    .import_socket = iree_async_proactor_io_uring_import_socket,
    .destroy_socket = iree_async_proactor_io_uring_destroy_socket,
    .import_file = iree_async_proactor_io_uring_import_file,
    .destroy_file = iree_async_proactor_io_uring_destroy_file,
    .create_event = iree_async_proactor_io_uring_create_event,
    .destroy_event = iree_async_proactor_io_uring_destroy_event,
    .register_event_source = iree_async_proactor_io_uring_register_event_source,
    .unregister_event_source =
        iree_async_proactor_io_uring_unregister_event_source,
    .create_notification = iree_async_proactor_io_uring_create_notification,
    .destroy_notification = iree_async_proactor_io_uring_destroy_notification,
    .notification_signal = iree_async_io_uring_notification_signal,
    .notification_wait = iree_async_io_uring_notification_wait,
    .register_relay = iree_async_proactor_io_uring_register_relay,
    .unregister_relay = iree_async_proactor_io_uring_unregister_relay,
    .register_buffer = iree_async_proactor_io_uring_register_buffer,
    .register_dmabuf = iree_async_proactor_io_uring_register_dmabuf,
    .unregister_buffer = iree_async_proactor_io_uring_unregister_buffer,
    .register_slab = iree_async_proactor_io_uring_register_slab,
    .import_fence = iree_async_proactor_io_uring_import_fence,
    .export_fence = iree_async_proactor_io_uring_export_fence,
    .set_message_callback = iree_async_proactor_io_uring_set_message_callback,
    .send_message = iree_async_proactor_io_uring_send_message,
    .subscribe_signal = iree_async_proactor_io_uring_subscribe_signal,
    .unsubscribe_signal = iree_async_proactor_io_uring_unsubscribe_signal,
};
