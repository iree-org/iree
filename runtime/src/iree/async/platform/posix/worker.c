// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/worker.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <unistd.h>

#include "iree/async/file.h"
#include "iree/async/operations/file.h"
#include "iree/async/platform/posix/proactor.h"
#include "iree/async/platform/posix/wake.h"
#include "iree/async/util/completion_pool.h"
#include "iree/async/util/ready_pool.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

//===----------------------------------------------------------------------===//
// File operation handlers (blocking I/O)
//===----------------------------------------------------------------------===//

// Executes a FILE_OPEN operation using the open() syscall.
static iree_status_t iree_async_posix_execute_file_open(
    iree_async_proactor_posix_t* proactor,
    iree_async_file_open_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Translate open flags to POSIX flags.
  int flags = 0;
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_READ)) {
    flags |= iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)
                 ? O_RDWR
                 : O_RDONLY;
  } else if (iree_any_bit_set(op->open_flags,
                              IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
    flags |= O_WRONLY;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_CREATE)) {
    flags |= O_CREAT;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE)) {
    flags |= O_TRUNC;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_APPEND)) {
    flags |= O_APPEND;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)) {
#if defined(O_DIRECT)
    flags |= O_DIRECT;
#else
    // O_DIRECT not available on this platform - fall back to buffered I/O.
#endif  // O_DIRECT
  }
  flags |= O_CLOEXEC;  // Always set close-on-exec for safety.

  // Default mode for newly created files (rw-rw-r--).
  mode_t mode = 0664;

  // Execute the open syscall.
  int fd = open(op->path, flags, mode);
  if (fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s': %s", op->path,
                            strerror(errno));
  }

  // Import the fd as a file handle.
  iree_async_primitive_t primitive = {.type = IREE_ASYNC_PRIMITIVE_TYPE_FD,
                                      .value = {.fd = fd}};
  iree_status_t status =
      iree_async_file_import(&proactor->base, primitive, &op->opened_file);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Executes a FILE_READ operation using the pread() syscall.
static iree_status_t iree_async_posix_execute_file_read(
    iree_async_file_read_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int fd = op->file->primitive.value.fd;
  void* buffer = iree_async_span_ptr(op->buffer);
  size_t length = op->buffer.length;
  off_t offset = (off_t)op->offset;

  // Execute the pread syscall (blocking).
  ssize_t result = pread(fd, buffer, length, offset);
  if (result < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pread failed at offset %" PRIu64 ": %s",
                            op->offset, strerror(errno));
  }

  // Success - record bytes read.
  op->bytes_read = (iree_host_size_t)result;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Executes a FILE_WRITE operation using the pwrite() syscall.
static iree_status_t iree_async_posix_execute_file_write(
    iree_async_file_write_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int fd = op->file->primitive.value.fd;
  const void* buffer = iree_async_span_ptr(op->buffer);
  size_t length = op->buffer.length;
  off_t offset = (off_t)op->offset;

  // Execute the pwrite syscall (blocking).
  ssize_t result = pwrite(fd, buffer, length, offset);
  if (result < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pwrite failed at offset %" PRIu64 ": %s",
                            op->offset, strerror(errno));
  }

  // Success - record bytes written.
  op->bytes_written = (iree_host_size_t)result;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Executes a FILE_CLOSE operation using the close() syscall.
static iree_status_t iree_async_posix_execute_file_close(
    iree_async_file_close_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int fd = op->file->primitive.value.fd;

  // Execute the close syscall.
  int result = close(fd);
  if (result < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "close failed for fd %d: %s", fd, strerror(errno));
  }

  // Mark the fd as invalid to prevent double-close in file destroy.
  // The caller's file reference is consumed by release_operation_resources
  // during completion dispatch (no prior retain to balance it).
  op->file->primitive.value.fd = -1;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Worker thread entry point
//===----------------------------------------------------------------------===//

// Returns true if the worker is in ZOMBIE state.
static bool iree_async_posix_worker_is_zombie(
    iree_async_posix_worker_t* worker) {
  return iree_atomic_load(&worker->state, iree_memory_order_acquire) ==
         IREE_ASYNC_POSIX_WORKER_STATE_ZOMBIE;
}

static int iree_async_posix_worker_main(iree_async_posix_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Enter running state. If we were already requested to exit, restore that
  // state so the while loop condition naturally fails.
  iree_async_posix_worker_state_t prev_state =
      (iree_async_posix_worker_state_t)iree_atomic_exchange(
          &worker->state, IREE_ASYNC_POSIX_WORKER_STATE_RUNNING,
          iree_memory_order_acq_rel);
  if (prev_state == IREE_ASYNC_POSIX_WORKER_STATE_EXITING) {
    iree_atomic_store(&worker->state, IREE_ASYNC_POSIX_WORKER_STATE_EXITING,
                      iree_memory_order_release);
  }

  iree_async_proactor_posix_t* proactor = worker->proactor;
  while (iree_atomic_load(&worker->state, iree_memory_order_acquire) ==
         IREE_ASYNC_POSIX_WORKER_STATE_RUNNING) {
    // Event count pattern: prepare_wait captures the current epoch BEFORE
    // checking any conditions that would prevent sleeping. This ensures we
    // don't miss notifications that arrive between checking and sleeping.
    //
    // Both conditions that prevent sleeping (new work AND exit request) must
    // be checked AFTER prepare_wait and BEFORE commit_wait. The corresponding
    // posters (enqueue_for_execution and request_exit) increment the epoch
    // via post(), so if they fire between prepare_wait and commit_wait, the
    // epoch will differ from our token and commit_wait returns immediately.
    iree_wait_token_t wait_token =
        iree_notification_prepare_wait(&proactor->ready_notification);

    // Try to pop work from ready queue.
    iree_atomic_slist_entry_t* entry =
        iree_atomic_slist_pop(&proactor->ready_queue);

    if (entry) {
      // Got work - cancel the wait (we won't be sleeping).
      iree_notification_cancel_wait(&proactor->ready_notification);

      iree_async_posix_ready_op_t* ready_op =
          iree_containerof(entry, iree_async_posix_ready_op_t, slist_entry);

      IREE_TRACE_ZONE_BEGIN_NAMED(z1, "worker_execute_operation");

      // Execute the operation based on its type.
      iree_status_t status = iree_ok_status();
      iree_async_completion_flags_t completion_flags =
          IREE_ASYNC_COMPLETION_FLAG_NONE;

      switch (ready_op->operation->type) {
        case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
          status = iree_async_posix_execute_file_open(
              proactor, (iree_async_file_open_operation_t*)ready_op->operation);
          break;

        case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
          status = iree_async_posix_execute_file_read(
              (iree_async_file_read_operation_t*)ready_op->operation);
          break;

        case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
          status = iree_async_posix_execute_file_write(
              (iree_async_file_write_operation_t*)ready_op->operation);
          break;

        case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
          status = iree_async_posix_execute_file_close(
              (iree_async_file_close_operation_t*)ready_op->operation);
          break;

        default:
          // Socket operations are executed directly on the poll thread via
          // process_operation_chain (non-blocking I/O after poll readiness).
          // Workers are exclusively for file operations that require blocking
          // I/O. Any operation reaching here without a handler is an
          // implementation bug.
          status =
              iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                               "worker received unsupported operation type %d",
                               (int)ready_op->operation->type);
          break;
      }

      IREE_TRACE_ZONE_END(z1);

      // Acquire a completion entry from the pool.
      // If pool is exhausted, spin with yield until one becomes available.
      // This provides backpressure without deadlock (poll thread will drain
      // completions and free entries).
      iree_async_posix_completion_t* completion = NULL;
      while (!(completion = iree_async_posix_completion_pool_acquire(
                   &proactor->completion_pool))) {
        iree_thread_yield();
      }

      // Fill in the completion entry.
      completion->operation = ready_op->operation;
      completion->status = status;
      completion->flags = completion_flags;

      // Release ready entry back to pool.
      iree_async_posix_ready_pool_release(&proactor->ready_pool, ready_op);

      // Push completion to queue for poll thread.
      iree_atomic_slist_push(&proactor->completion_queue,
                             &completion->slist_entry);

      // Wake the poll thread to process the completion.
      iree_async_posix_wake_trigger(&proactor->wake);

    } else if (iree_atomic_load(&worker->state, iree_memory_order_acquire) !=
               IREE_ASYNC_POSIX_WORKER_STATE_RUNNING) {
      // Exit requested between the loop-top state check and here. The
      // request_exit post may have already incremented the epoch past our
      // wait_token, so commit_wait would return immediately — but if the post
      // happened before prepare_wait, our token matches the current epoch and
      // commit_wait would block forever (classic event-count lost wakeup).
      // Cancel the wait and let the loop condition handle the exit.
      iree_notification_cancel_wait(&proactor->ready_notification);
    } else {
      // No work and no exit request — commit to waiting on the shared
      // notification. The event count guarantees: if request_exit posts between
      // our prepare_wait and this commit_wait, the epoch will have changed,
      // causing commit_wait to return immediately.
      iree_notification_commit_wait(&proactor->ready_notification, wait_token,
                                    /*spin_ns=*/0,
                                    /*deadline_ns=*/IREE_TIME_INFINITE_FUTURE);
    }
  }

  // Single exit point: transition to zombie state and notify waiters.
  iree_atomic_store(&worker->state, IREE_ASYNC_POSIX_WORKER_STATE_ZOMBIE,
                    iree_memory_order_release);
  iree_notification_post(&worker->state_notification, IREE_ALL_WAITERS);

  IREE_TRACE_ZONE_END(z0);
  return 0;
}

//===----------------------------------------------------------------------===//
// Worker lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_async_posix_worker_initialize(
    iree_async_proactor_posix_t* proactor, iree_host_size_t worker_index,
    iree_allocator_t allocator, iree_async_posix_worker_t* out_worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Zero-init the worker struct. This ensures cleanup in deinitialize works
  // even if we fail partway through initialization.
  memset(out_worker, 0, sizeof(*out_worker));

  out_worker->proactor = proactor;
  out_worker->worker_index = worker_index;

  // Initialize state notification (used for exit synchronization).
  // Workers share proactor->ready_notification for work-stealing.
  iree_notification_initialize(&out_worker->state_notification);

  // Set initial state to RUNNING (thread will confirm this on entry).
  iree_atomic_store(&out_worker->state, IREE_ASYNC_POSIX_WORKER_STATE_RUNNING,
                    iree_memory_order_release);

  // Create thread parameters.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_make_cstring_view("async_worker");
  params.create_suspended = false;

  // Set thread affinity to the proactor's NUMA node if specified.
  if (proactor->numa_node != 0) {
    iree_thread_affinity_set_group_any(proactor->numa_node,
                                       &params.initial_affinity);
  }

  // Create and start the worker thread.
  iree_status_t status =
      iree_thread_create((iree_thread_entry_t)iree_async_posix_worker_main,
                         out_worker, params, allocator, &out_worker->thread);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_posix_worker_request_exit(iree_async_posix_worker_t* worker) {
  if (!worker->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Set state to EXITING. If already ZOMBIE, restore ZOMBIE state.
  iree_async_posix_worker_state_t prev_state =
      (iree_async_posix_worker_state_t)iree_atomic_exchange(
          &worker->state, IREE_ASYNC_POSIX_WORKER_STATE_EXITING,
          iree_memory_order_acq_rel);
  if (prev_state == IREE_ASYNC_POSIX_WORKER_STATE_ZOMBIE) {
    // Worker already exited; reset to ZOMBIE.
    iree_atomic_store(&worker->state, IREE_ASYNC_POSIX_WORKER_STATE_ZOMBIE,
                      iree_memory_order_release);
  }

  // Wake workers in case the target is waiting for work. Workers share the
  // proactor's ready_notification, so we must wake all of them -- only the
  // targeted worker will see EXITING state and exit its loop.
  iree_notification_post(&worker->proactor->ready_notification,
                         IREE_ALL_WAITERS);

  IREE_TRACE_ZONE_END(z0);
}

void iree_async_posix_worker_await_exit(iree_async_posix_worker_t* worker) {
  if (!worker->thread) return;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Ensure exit was requested.
  iree_async_posix_worker_request_exit(worker);

  // Wait for the worker to enter ZOMBIE state.
  iree_notification_await(
      &worker->state_notification,
      (iree_condition_fn_t)iree_async_posix_worker_is_zombie, worker,
      iree_infinite_timeout());

  IREE_TRACE_ZONE_END(z0);
}

void iree_async_posix_worker_deinitialize(iree_async_posix_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Worker must be in ZOMBIE state (or thread creation failed).
  IREE_ASSERT(!worker->thread || iree_async_posix_worker_is_zombie(worker),
              "worker must be in ZOMBIE state before deinitialize");

  // Release thread handle.
  iree_thread_release(worker->thread);
  worker->thread = NULL;

  // Deinitialize state notification.
  // Workers share proactor->ready_notification (not deinit'd here).
  iree_notification_deinitialize(&worker->state_notification);

  IREE_TRACE_ZONE_END(z0);
}
