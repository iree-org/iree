// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Submit path for io_uring proactor.
//
// This module handles SQE preparation and submission for all operation types.
// Fill functions prepare SQEs with operation-specific parameters. The submit
// function handles batching, linked operations, and timer chain emulation.

// Enable GNU extensions for O_DIRECT (used in file open flag translation).
// Must be defined before any includes.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
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
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/platform/io_uring/socket.h"
#include "iree/async/semaphore.h"

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

// Fills an SQE for a NOP operation.
static void iree_async_proactor_io_uring_fill_nop(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* operation) {
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;
  sqe->user_data = (uint64_t)(uintptr_t)operation;
}

// Fills two linked SQEs for an EVENT_WAIT operation.
// Uses POLL_ADD linked to READ to wait for the event's eventfd and auto-drain
// it in the kernel when it becomes readable. This eliminates the need for an
// explicit reset syscall on acquire.
//
// The linked pair:
//   SQE 1 (poll_sqe): POLL_ADD on eventfd, IOSQE_IO_LINK | CQE_SKIP_SUCCESS
//   SQE 2 (read_sqe): READ to drain the eventfd counter into drain_buffer
//
// On success, POLL_ADD produces no CQE (CQE_SKIP_SUCCESS). The READ CQE is
// the user-visible completion that fires the callback. This ensures one
// logical request (EVENT_WAIT) produces one logical response (READ CQE).
//
// On POLL_ADD failure, it produces an error CQE (marked internal) and READ
// is cancelled. The READ's -ECANCELED CQE still fires the user callback.
static void iree_async_proactor_io_uring_fill_event_wait(
    iree_io_uring_sqe_t* poll_sqe, iree_io_uring_sqe_t* read_sqe,
    iree_async_operation_t* base_operation) {
  iree_async_event_wait_operation_t* event_wait =
      (iree_async_event_wait_operation_t*)base_operation;

  int fd = event_wait->event->primitive.value.fd;

  // SQE 1: POLL_ADD with link to next SQE.
  // CQE_SKIP_SUCCESS suppresses the CQE on success — only the READ CQE fires
  // the user callback. On error (bad fd, cancellation), the linked READ is
  // never started by the kernel and produces no CQE. The TAG_LINKED_POLL tag
  // in user_data routes the error CQE to a handler that dispatches the user
  // callback directly.
  memset(poll_sqe, 0, sizeof(*poll_sqe));
  poll_sqe->opcode = IREE_IORING_OP_POLL_ADD;
  poll_sqe->flags = IREE_IOSQE_IO_LINK | IREE_IOSQE_CQE_SKIP_SUCCESS;
  poll_sqe->fd = fd;
  poll_sqe->poll32_events = POLLIN;
  poll_sqe->user_data = iree_io_uring_internal_encode(
      IREE_IO_URING_TAG_LINKED_POLL, (uintptr_t)base_operation);

  // SQE 2: READ to drain the eventfd counter.
  // This is the user-visible operation — its CQE fires the callback.
  // The eventfd stores an 8-byte counter; reading resets it to 0.
  memset(read_sqe, 0, sizeof(*read_sqe));
  read_sqe->opcode = IREE_IORING_OP_READ;
  read_sqe->fd = fd;
  read_sqe->addr = (uint64_t)(uintptr_t)&event_wait->event->drain_buffer;
  read_sqe->len = sizeof(event_wait->event->drain_buffer);
  read_sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a TIMER operation.
// Uses absolute timeout when CAPABILITY_ABSOLUTE_TIMEOUT is set (kernel 5.4+),
// otherwise converts the deadline to a relative duration at submission time.
static void iree_async_proactor_io_uring_fill_timer(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_timer_operation_t* timer =
      (iree_async_timer_operation_t*)base_operation;

  if (iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT)) {
    // Preferred: absolute timeout with CLOCK_MONOTONIC (no drift).
    // iree/async/ uses iree_time_now() which matches CLOCK_MONOTONIC.
    timer->platform.timespec.tv_sec = timer->deadline_ns / 1000000000LL;
    timer->platform.timespec.tv_nsec = timer->deadline_ns % 1000000000LL;
    sqe->timeout_flags = IREE_IORING_TIMEOUT_ABS;
  } else {
    // Fallback: relative timeout. There's a small window for drift between
    // computing this and the kernel processing the SQE.
    iree_duration_t remaining = timer->deadline_ns - iree_time_now();
    if (remaining < 0) remaining = 0;
    timer->platform.timespec.tv_sec = remaining / 1000000000LL;
    timer->platform.timespec.tv_nsec = remaining % 1000000000LL;
    sqe->timeout_flags = 0;
  }

  // Note: We do NOT set ETIME_SUCCESS here even though it's available on 5.16+.
  // Linked timers use userspace emulation (split at timer, submit continuation
  // on completion) rather than kernel LINK chains. Without kernel LINK, there's
  // no chain to break when -ETIME is returned - we convert -ETIME to OK in
  // cqe_to_status anyway.

  sqe->opcode = IREE_IORING_OP_TIMEOUT;
  sqe->fd = -1;
  sqe->addr = (uint64_t)(uintptr_t)&timer->platform.timespec;
  sqe->len = 1;  // One timespec structure. Event count is in sqe->off (0 = pure
                 // timer).
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a SOCKET_CONNECT operation.
// Uses IORING_OP_CONNECT to initiate an outbound connection.
static void iree_async_proactor_io_uring_fill_socket_connect(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_connect_operation_t* connect =
      (iree_async_socket_connect_operation_t*)base_operation;

  sqe->opcode = IREE_IORING_OP_CONNECT;
  sqe->fd = connect->socket->primitive.value.fd;
  sqe->addr = (uint64_t)(uintptr_t)connect->address.storage;
  sqe->off = connect->address.length;  // connect uses off for addr_len.
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a SOCKET_ACCEPT operation.
// Uses IORING_OP_ACCEPT to accept an incoming connection on a listening socket.
static void iree_async_proactor_io_uring_fill_socket_accept(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_accept_operation_t* accept =
      (iree_async_socket_accept_operation_t*)base_operation;

  // Clear output fields.
  accept->accepted_socket = NULL;
  accept->peer_address.length = sizeof(accept->peer_address.storage);

  sqe->opcode = IREE_IORING_OP_ACCEPT;
  sqe->fd = accept->listen_socket->primitive.value.fd;
  sqe->addr = (uint64_t)(uintptr_t)accept->peer_address.storage;
  sqe->off = (uint64_t)(uintptr_t)&accept->peer_address.length;
  sqe->accept_flags = SOCK_NONBLOCK | SOCK_CLOEXEC;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  // Enable multishot mode if requested (kernel 5.19+).
  // In multishot mode, a single SQE produces multiple CQEs - one for each
  // accepted connection - with CQE_F_MORE set on all but the final CQE.
  if (base_operation->flags & IREE_ASYNC_OPERATION_FLAG_MULTISHOT) {
    sqe->ioprio = IREE_IORING_ACCEPT_MULTISHOT;
  }
}

// Fills an SQE for a SOCKET_RECV operation.
// Uses IORING_OP_RECV for single-buffer receives, IORING_OP_RECVMSG for
// scatter- gather (multiple buffers). Retains the socket to ensure it outlives
// the in-flight operation.
static iree_status_t iree_async_proactor_io_uring_fill_socket_recv(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_recv_operation_t* recv =
      (iree_async_socket_recv_operation_t*)base_operation;

  // Validate buffer count.
  if (recv->buffers.count > IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recv buffer count %" PRIhsz
                            " exceeds maximum %d; "
                            "split into multiple operations or reduce buffers",
                            recv->buffers.count,
                            IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS);
  }

  // Clear output fields.
  recv->bytes_received = 0;

  sqe->fd = recv->socket->primitive.value.fd;
  sqe->msg_flags = 0;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  if (recv->buffers.count == 1) {
    // Single buffer: use simple RECV for efficiency.
    iree_async_span_t first_buffer = recv->buffers.values[0];
    sqe->opcode = IREE_IORING_OP_RECV;
    sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(first_buffer);
    sqe->len = (uint32_t)first_buffer.length;
  } else {
    // Multiple buffers: use RECVMSG with scatter-gather.
    // Convert spans to iovecs. The iovec array is stored in the operation's
    // platform storage to ensure it lives until completion.
    struct iovec* iovecs = (struct iovec*)recv->platform.posix.iovecs;
    iree_host_size_t count = recv->buffers.count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      iovecs[i].iov_base = iree_async_span_ptr(recv->buffers.values[i]);
      iovecs[i].iov_len = recv->buffers.values[i].length;
    }

    struct msghdr* msg = (struct msghdr*)recv->platform.posix.msg_header;
    memset(msg, 0, sizeof(*msg));
    msg->msg_iov = iovecs;
    msg->msg_iovlen = count;

    sqe->opcode = IREE_IORING_OP_RECVMSG;
    sqe->addr = (uint64_t)(uintptr_t)msg;
    sqe->len = 1;  // Number of messages (always 1 for RECVMSG).
  }

  // Enable multishot mode if requested (kernel 5.19+).
  // In multishot mode, a single SQE produces multiple CQEs - one for each
  // received message - with CQE_F_MORE set on all but the final CQE.
  // NOTE: Multishot recv with a fixed buffer reuses the same buffer for each
  // completion. The caller must process data before the next completion or
  // risk data being overwritten. For zero-copy multishot recv, use RECV_POOL
  // with a provided buffer ring (kernel 6.0+).
  if (base_operation->flags & IREE_ASYNC_OPERATION_FLAG_MULTISHOT) {
    sqe->ioprio |= IREE_IORING_RECV_MULTISHOT;
  }

  return iree_ok_status();
}

// Fills an SQE for a SOCKET_RECV_POOL operation.
// Uses IORING_OP_RECV with IOSQE_BUFFER_SELECT to let the kernel select a
// buffer from a provided buffer ring (PBUF_RING). On completion, the CQE
// indicates which buffer was used. The operation's lease is populated with
// the buffer index and span.
static iree_status_t iree_async_proactor_io_uring_fill_socket_recv_pool(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_socket_recv_pool_operation_t* recv_pool =
      (iree_async_socket_recv_pool_operation_t*)base_operation;

  // Validate the buffer pool region.
  // RECV_POOL requires a pool with a provided buffer ring (PBUF_RING), which
  // is set up during slab registration when the pool has write access and the
  // kernel supports multishot (5.19+). If the pool lacks a buffer ring, callers
  // should use SOCKET_RECV with manual buffer management instead.
  iree_async_region_t* region = iree_async_buffer_pool_region(recv_pool->pool);
  if (IREE_UNLIKELY(region->type != IREE_ASYNC_REGION_TYPE_IOURING)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SOCKET_RECV_POOL requires a buffer pool registered with an io_uring "
        "proactor; got region type %d",
        (int)region->type);
  }
  // Region must belong to the submitting proactor.
  // Buffer group IDs are ring-local; using a group ID from a different
  // proactor's registration would select the wrong buffer pool.
  if (IREE_UNLIKELY(region->proactor != &proactor->base)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SOCKET_RECV_POOL buffer pool was registered with a different "
        "proactor; buffer group IDs are ring-local and cannot be used "
        "across proactors");
  }
  if (IREE_UNLIKELY(region->handles.iouring.buffer_group_id < 0)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SOCKET_RECV_POOL requires a buffer pool with a provided buffer ring "
        "(buffer_group_id >= 0); this pool was registered without "
        "IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE or the kernel lacks multishot "
        "support (requires 5.19+). Use SOCKET_RECV with manual buffer "
        "management instead.");
  }

  // Clear output fields.
  recv_pool->bytes_received = 0;
  memset(&recv_pool->lease, 0, sizeof(recv_pool->lease));

  sqe->opcode = IREE_IORING_OP_RECV;
  sqe->fd = recv_pool->socket->primitive.value.fd;
  sqe->addr = 0;  // Kernel provides buffer from the ring.
  sqe->msg_flags = 0;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
  sqe->flags |= IREE_IOSQE_BUFFER_SELECT;
  sqe->buf_group = (uint16_t)region->handles.iouring.buffer_group_id;

  // Enable multishot mode if requested (kernel 5.19+).
  // Multishot recv requires len=0; the kernel gets buffer sizes from the ring.
  // Single-shot can use buffer_size as a cap on receive length.
  if (iree_any_bit_set(base_operation->flags,
                       IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    sqe->ioprio |= IREE_IORING_RECV_MULTISHOT;
    sqe->len = 0;
  } else {
    sqe->len = (uint32_t)region->buffer_size;
  }

  return iree_ok_status();
}

// Result of checking if a span can use fixed-buffer zero-copy send.
typedef enum {
  // Span is not eligible for fixed-buffer path. Use standard zero-copy.
  // This is NOT an error - just means we fall back to page-pinning per send.
  IREE_ASYNC_FIXED_BUFFER_INELIGIBLE = 0,
  // Span is eligible. out_buffer_index contains the kernel buffer table index.
  IREE_ASYNC_FIXED_BUFFER_ELIGIBLE = 1,
} iree_async_fixed_buffer_eligibility_t;

// Checks if a span can use the fixed-buffer path for zero-copy send.
//
// Returns OK with ELIGIBLE if the span comes from a registered io_uring region
// with indexed buffers and fits within a single buffer. The buffer index is
// written to out_buffer_index.
//
// Returns OK with INELIGIBLE for legitimate fallback cases (no region,
// non-io_uring region, unregistered, span crosses buffer boundary, or
// region belongs to a different proactor).
//
// Returns error status for provably corrupt regions (buffer_size == 0 with
// buffer_count > 0, or index overflow).
static iree_status_t iree_async_span_check_fixed_buffer_send(
    iree_async_proactor_io_uring_t* proactor, iree_async_span_t span,
    iree_async_fixed_buffer_eligibility_t* out_eligibility,
    uint16_t* out_buffer_index) {
  *out_eligibility = IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  *out_buffer_index = 0;

  // No region = heap buffer, not registered.
  if (!span.region) return iree_ok_status();
  iree_async_region_t* region = span.region;

  // Region must belong to the submitting proactor.
  // Fixed buffer indices are ring-local; using indices from a different
  // proactor's registration would access wrong memory.
  if (region->proactor != &proactor->base) return iree_ok_status();

  // Non-io_uring region = different backend (RDMA, dmabuf, etc.).
  if (region->type != IREE_ASYNC_REGION_TYPE_IOURING) return iree_ok_status();

  // buffer_count == 0 = region not registered with indexed buffers.
  uint32_t buffer_count = region->buffer_count;
  if (buffer_count == 0) return iree_ok_status();

  // From here on, the region claims to have indexed buffers.
  // Validate invariants that should have been established at registration.

  iree_host_size_t buffer_size = region->buffer_size;

  // Corrupt region: has indexed buffers but zero buffer_size.
  // This would cause divide-by-zero and indicates registration bug or memory
  // corruption.
  if (IREE_UNLIKELY(buffer_size == 0)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "io_uring region has buffer_count %" PRIu32
                            " but buffer_size == 0; "
                            "region was not properly registered",
                            buffer_count);
  }

  // Must have read access for send operations.
  if (!(region->access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ)) {
    return iree_ok_status();  // Not eligible, but not an error.
  }

  // Calculate which buffer this span starts in.
  // Use uint64_t to avoid truncation before bounds check.
  uint64_t buffer_index_offset = span.offset / buffer_size;

  // Span starts beyond registered buffer range.
  if (buffer_index_offset >= buffer_count) return iree_ok_status();

  // Check if span fits entirely within one buffer.
  uint32_t offset_in_buffer = (uint32_t)(span.offset % buffer_size);
  if (offset_in_buffer + span.length > buffer_size) return iree_ok_status();

  // Calculate final buffer index with overflow check.
  uint16_t base_buffer_index = region->handles.iouring.base_buffer_index;
  uint64_t final_index = (uint64_t)base_buffer_index + buffer_index_offset;

  // Index overflow: registration allowed too many buffers or base_buffer_index
  // is corrupt.
  if (IREE_UNLIKELY(final_index > UINT16_MAX)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "fixed buffer index %" PRIu64
                            " exceeds uint16 maximum; "
                            "base_buffer_index=%" PRIu16 " + offset=%" PRIu64,
                            final_index, base_buffer_index,
                            buffer_index_offset);
  }

  *out_eligibility = IREE_ASYNC_FIXED_BUFFER_ELIGIBLE;
  *out_buffer_index = (uint16_t)final_index;
  return iree_ok_status();
}

// Fills an SQE for a SOCKET_SEND operation.
// Uses IORING_OP_SEND[_ZC] for single-buffer sends, IORING_OP_SENDMSG[_ZC] for
// scatter-gather (multiple buffers). Zero-copy variants are used when the
// socket has IREE_ASYNC_SOCKET_OPTION_ZERO_COPY AND the capability is
// available. ZERO_COPY is a socket-level hint ("use ZC if you can"), not a
// requirement - sends fall back to regular SEND/SENDMSG on kernels < 6.0 that
// lack ZC support. Retains the socket to ensure it outlives the in-flight
// operation.
static iree_status_t iree_async_proactor_io_uring_fill_socket_send(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_socket_send_operation_t* send =
      (iree_async_socket_send_operation_t*)base_operation;

  // Validate buffer count.
  if (send->buffers.count > IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "send buffer count %" PRIhsz
                            " exceeds maximum %d; "
                            "split into multiple operations or reduce buffers",
                            send->buffers.count,
                            IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS);
  }

  // Zero-copy is determined by socket flags, not per-send flags.
  // The socket flag is set at creation (from option) or import time.
  // The kernel capability must also be available.
  bool zero_copy_requested =
      iree_any_bit_set(send->socket->flags, IREE_ASYNC_SOCKET_FLAG_ZERO_COPY);
  bool zero_copy_available = iree_any_bit_set(
      proactor->capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND);
  bool use_zero_copy = zero_copy_requested && zero_copy_available;

  // Check fixed-buffer eligibility.
  // The span check is pure validation and can fail on corrupt regions.
  iree_async_fixed_buffer_eligibility_t eligibility =
      IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  uint16_t fixed_buffer_index = 0;
  if (use_zero_copy && send->buffers.count == 1) {
    IREE_RETURN_IF_ERROR(iree_async_span_check_fixed_buffer_send(
        proactor, send->buffers.values[0], &eligibility, &fixed_buffer_index));
  }

  // Clear output fields.
  send->bytes_sent = 0;

  sqe->fd = send->socket->primitive.value.fd;
  sqe->msg_flags = 0;
  if (iree_any_bit_set(send->send_flags, IREE_ASYNC_SOCKET_SEND_FLAG_MORE)) {
    sqe->msg_flags |= IREE_MSG_MORE;
  }
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  if (send->buffers.count == 1) {
    // Single buffer: use simple SEND[_ZC] for efficiency.
    iree_async_span_t first_buffer = send->buffers.values[0];

    sqe->opcode = use_zero_copy ? IREE_IORING_OP_SEND_ZC : IREE_IORING_OP_SEND;

    // CRITICAL: sqe->addr is ALWAYS the full pointer, even with FIXED_BUF.
    // The kernel validates that addr falls within the registered buffer.
    sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(first_buffer);
    sqe->len = (uint32_t)first_buffer.length;

    if (use_zero_copy) {
      // Always request usage reporting so we can tell callers if ZC succeeded.
      sqe->ioprio |= IREE_IORING_SEND_ZC_REPORT_USAGE;
      if (eligibility == IREE_ASYNC_FIXED_BUFFER_ELIGIBLE) {
        sqe->ioprio |= IREE_IORING_RECVSEND_FIXED_BUF;
        sqe->buf_index = fixed_buffer_index;
      }
    }
  } else {
    // Multiple buffers: use SENDMSG[_ZC] with scatter-gather.
    // Convert spans to iovecs (spans have region+offset+length, iovecs have
    // base+length). The iovec array is stored in the operation's platform
    // storage to ensure it lives until completion.
    struct iovec* iovecs = (struct iovec*)send->platform.posix.iovecs;
    iree_host_size_t count = send->buffers.count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      iovecs[i].iov_base = iree_async_span_ptr(send->buffers.values[i]);
      iovecs[i].iov_len = send->buffers.values[i].length;
    }

    struct msghdr* msg = (struct msghdr*)send->platform.posix.msg_header;
    memset(msg, 0, sizeof(*msg));
    msg->msg_iov = iovecs;
    msg->msg_iovlen = count;

    sqe->opcode =
        use_zero_copy ? IREE_IORING_OP_SENDMSG_ZC : IREE_IORING_OP_SENDMSG;
    sqe->addr = (uint64_t)(uintptr_t)msg;
    sqe->len = 1;  // Number of messages (always 1 for SENDMSG[_ZC]).

    if (use_zero_copy) {
      // Always request usage reporting so we can tell callers if ZC succeeded.
      sqe->ioprio |= IREE_IORING_SEND_ZC_REPORT_USAGE;
    }
  }

  return iree_ok_status();
}

// Fills an SQE for a SOCKET_SENDTO operation.
// Always uses IORING_OP_SENDMSG[_ZC] since we need msg_name for the destination
// address. This works for both single-buffer and scatter-gather sends.
// Retains the socket to ensure it outlives the in-flight operation.
static iree_status_t iree_async_proactor_io_uring_fill_socket_sendto(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation,
    iree_async_proactor_capabilities_t capabilities) {
  iree_async_socket_sendto_operation_t* sendto =
      (iree_async_socket_sendto_operation_t*)base_operation;

  // Validate buffer count.
  if (sendto->buffers.count > IREE_ASYNC_SOCKET_SENDTO_MAX_BUFFERS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "sendto buffer count %" PRIhsz
                            " exceeds maximum %d; "
                            "split into multiple operations or reduce buffers",
                            sendto->buffers.count,
                            IREE_ASYNC_SOCKET_SENDTO_MAX_BUFFERS);
  }

  // Clear output fields.
  sendto->bytes_sent = 0;

  // Zero-copy is determined by socket flags, not per-send flags.
  bool zero_copy_requested =
      iree_any_bit_set(sendto->socket->flags, IREE_ASYNC_SOCKET_FLAG_ZERO_COPY);
  bool zero_copy_available = iree_any_bit_set(
      capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND);
  bool use_zero_copy = zero_copy_requested && zero_copy_available;

  // Convert spans to iovecs. The iovec array is stored in the operation's
  // platform storage to ensure it lives until completion.
  struct iovec* iovecs = (struct iovec*)sendto->platform.posix.iovecs;
  iree_host_size_t count = sendto->buffers.count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iovecs[i].iov_base = iree_async_span_ptr(sendto->buffers.values[i]);
    iovecs[i].iov_len = sendto->buffers.values[i].length;
  }

  // Build msghdr with destination address.
  struct msghdr* msg = (struct msghdr*)sendto->platform.posix.msg_header;
  memset(msg, 0, sizeof(*msg));
  msg->msg_name = sendto->destination.storage;
  msg->msg_namelen = (socklen_t)sendto->destination.length;
  msg->msg_iov = iovecs;
  msg->msg_iovlen = count;

  sqe->fd = sendto->socket->primitive.value.fd;
  sqe->opcode =
      use_zero_copy ? IREE_IORING_OP_SENDMSG_ZC : IREE_IORING_OP_SENDMSG;
  sqe->addr = (uint64_t)(uintptr_t)msg;
  sqe->len = 1;  // Number of messages (always 1 for SENDMSG).
  sqe->msg_flags = 0;
  if (iree_any_bit_set(sendto->send_flags, IREE_ASYNC_SOCKET_SEND_FLAG_MORE)) {
    sqe->msg_flags |= IREE_MSG_MORE;
  }
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  if (use_zero_copy) {
    sqe->ioprio |= IREE_IORING_SEND_ZC_REPORT_USAGE;
  }

  return iree_ok_status();
}

// Fills an SQE for a SOCKET_RECVFROM operation.
// Always uses IORING_OP_RECVMSG since we need msg_name for the sender address.
// This works for both single-buffer and scatter-gather receives.
// Retains the socket to ensure it outlives the in-flight operation.
static iree_status_t iree_async_proactor_io_uring_fill_socket_recvfrom(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_recvfrom_operation_t* recvfrom =
      (iree_async_socket_recvfrom_operation_t*)base_operation;

  // Validate buffer count.
  if (recvfrom->buffers.count > IREE_ASYNC_SOCKET_RECVFROM_MAX_BUFFERS) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recvfrom buffer count %" PRIhsz
                            " exceeds maximum %d; "
                            "split into multiple operations or reduce buffers",
                            recvfrom->buffers.count,
                            IREE_ASYNC_SOCKET_RECVFROM_MAX_BUFFERS);
  }

  // Clear output fields. The sender address length is set to the storage size;
  // the kernel will update it to the actual address length on completion.
  recvfrom->bytes_received = 0;
  recvfrom->sender.length = sizeof(recvfrom->sender.storage);

  // Convert spans to iovecs. The iovec array is stored in the operation's
  // platform storage to ensure it lives until completion.
  struct iovec* iovecs = (struct iovec*)recvfrom->platform.posix.iovecs;
  iree_host_size_t count = recvfrom->buffers.count;
  for (iree_host_size_t i = 0; i < count; ++i) {
    iovecs[i].iov_base = iree_async_span_ptr(recvfrom->buffers.values[i]);
    iovecs[i].iov_len = recvfrom->buffers.values[i].length;
  }

  // Build msghdr with sender address buffer.
  struct msghdr* msg = (struct msghdr*)recvfrom->platform.posix.msg_header;
  memset(msg, 0, sizeof(*msg));
  msg->msg_name = recvfrom->sender.storage;
  msg->msg_namelen = sizeof(recvfrom->sender.storage);
  msg->msg_iov = iovecs;
  msg->msg_iovlen = count;

  sqe->fd = recvfrom->socket->primitive.value.fd;
  sqe->opcode = IREE_IORING_OP_RECVMSG;
  sqe->addr = (uint64_t)(uintptr_t)msg;
  sqe->len = 1;  // Number of messages (always 1 for RECVMSG).
  sqe->msg_flags = 0;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  return iree_ok_status();
}

// Fills an SQE for a SOCKET_CLOSE operation.
// Uses IORING_OP_CLOSE to close the socket asynchronously.
// The socket reference is consumed (not retained) - it will be destroyed
// after completion.
static void iree_async_proactor_io_uring_fill_socket_close(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_close_operation_t* close_op =
      (iree_async_socket_close_operation_t*)base_operation;

  // Note: We do NOT retain the socket here. The close operation consumes
  // the caller's reference. The socket will be destroyed on completion.

  sqe->opcode = IREE_IORING_OP_CLOSE;
  sqe->fd = close_op->socket->primitive.value.fd;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

//===----------------------------------------------------------------------===//
// File operation SQE fills
//===----------------------------------------------------------------------===//

// Translates iree_async_file_open_flags_t to POSIX open flags for OPENAT.
static int iree_async_proactor_io_uring_translate_open_flags(
    iree_async_file_open_flags_t open_flags) {
  int flags = 0;
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_READ)) {
    flags |= iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)
                 ? O_RDWR
                 : O_RDONLY;
  } else if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
    flags |= O_WRONLY;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_CREATE)) {
    flags |= O_CREAT;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE)) {
    flags |= O_TRUNC;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_APPEND)) {
    flags |= O_APPEND;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)) {
    flags |= O_DIRECT;
  }
  flags |= O_CLOEXEC;  // Always set close-on-exec for safety.
  return flags;
}

// Fills an SQE for a FILE_OPEN operation.
// Uses IORING_OP_OPENAT with AT_FDCWD to open a file relative to the
// current working directory. The kernel performs the open asynchronously.
//
// SQE layout (io_uring OPENAT):
//   fd         = directory fd (AT_FDCWD for cwd)
//   addr       = pathname
//   len        = mode (permissions for creation)
//   open_flags = POSIX O_* flags
static void iree_async_proactor_io_uring_fill_file_open(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_open_operation_t* open_op =
      (iree_async_file_open_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_OPENAT;
  sqe->fd = AT_FDCWD;
  sqe->addr = (uint64_t)(uintptr_t)open_op->path;
  sqe->len = 0664;  // Default mode for newly created files (rw-rw-r--).
  sqe->open_flags = (uint32_t)iree_async_proactor_io_uring_translate_open_flags(
      open_op->open_flags);
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FILE_READ operation.
// Uses IORING_OP_READ for positioned file I/O (pread semantics).
//
// SQE layout (io_uring READ):
//   fd   = file descriptor
//   off  = file offset
//   addr = buffer address
//   len  = buffer length
static void iree_async_proactor_io_uring_fill_file_read(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_read_operation_t* read_op =
      (iree_async_file_read_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_READ;
  sqe->fd = read_op->file->primitive.value.fd;
  sqe->off = read_op->offset;
  sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(read_op->buffer);
  sqe->len = (uint32_t)read_op->buffer.length;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FILE_WRITE operation.
// Uses IORING_OP_WRITE for positioned file I/O (pwrite semantics).
//
// SQE layout (io_uring WRITE):
//   fd   = file descriptor
//   off  = file offset
//   addr = buffer address
//   len  = buffer length
static void iree_async_proactor_io_uring_fill_file_write(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_write_operation_t* write_op =
      (iree_async_file_write_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_WRITE;
  sqe->fd = write_op->file->primitive.value.fd;
  sqe->off = write_op->offset;
  sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(write_op->buffer);
  sqe->len = (uint32_t)write_op->buffer.length;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FILE_CLOSE operation.
// Uses IORING_OP_CLOSE to close the file descriptor asynchronously.
// The file reference is consumed (not retained) — it will be released
// during completion resource cleanup.
static void iree_async_proactor_io_uring_fill_file_close(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_close_operation_t* close_op =
      (iree_async_file_close_operation_t*)base_operation;

  sqe->opcode = IREE_IORING_OP_CLOSE;
  sqe->fd = close_op->file->primitive.value.fd;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FUTEX_WAIT operation.
// Uses IORING_OP_FUTEX_WAIT to wait on a futex address in the kernel.
// Requires IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS (kernel 6.7+).
//
// SQE layout (from liburing io_uring_prep_futex_wait):
//   fd    = futex2 flags (FUTEX2_SIZE_* | FUTEX2_PRIVATE)
//   addr  = futex address
//   off   = expected value
//   len   = 0
//   futex_flags = io_uring flags (0)
//   addr3 = bitset mask (~0 to match any)
static void iree_async_proactor_io_uring_fill_futex_wait(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_futex_wait_operation_t* futex_wait =
      (iree_async_futex_wait_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_FUTEX_WAIT;
  sqe->fd = (int32_t)futex_wait->futex_flags;  // FUTEX2_SIZE_* | FUTEX2_PRIVATE
  sqe->addr = (uint64_t)(uintptr_t)futex_wait->futex_address;
  sqe->off = futex_wait->expected_value;
  sqe->len = 0;
  sqe->futex_flags = 0;      // io_uring flags, not futex2 flags.
  sqe->addr3 = 0xffffffffU;  // FUTEX_BITSET_MATCH_ANY (32-bit mask)
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FUTEX_WAKE operation.
// Uses IORING_OP_FUTEX_WAKE to wake waiters on a futex address.
// Requires IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS (kernel 6.7+).
//
// SQE layout (from liburing io_uring_prep_futex_wake):
//   fd    = futex2 flags (FUTEX2_SIZE_* | FUTEX2_PRIVATE)
//   addr  = futex address
//   off   = number of waiters to wake
//   len   = 0
//   futex_flags = io_uring flags (0)
//   addr3 = bitset mask (~0 to match any)
static void iree_async_proactor_io_uring_fill_futex_wake(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_futex_wake_operation_t* futex_wake =
      (iree_async_futex_wake_operation_t*)base_operation;

  // Clear output field.
  futex_wake->woken_count = 0;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_FUTEX_WAKE;
  sqe->fd = (int32_t)futex_wake->futex_flags;  // FUTEX2_SIZE_* | FUTEX2_PRIVATE
  sqe->addr = (uint64_t)(uintptr_t)futex_wake->futex_address;
  sqe->off = (uint64_t)futex_wake->wake_count;  // Number of waiters to wake.
  sqe->len = 0;
  sqe->futex_flags = 0;      // io_uring flags, not futex2 flags.
  sqe->addr3 = 0xffffffffU;  // FUTEX_BITSET_MATCH_ANY (32-bit mask)
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

//===----------------------------------------------------------------------===//
// Notification wait helpers
//===----------------------------------------------------------------------===//

// Fills SQE for a NOTIFICATION_WAIT operation in futex mode.
static void iree_async_proactor_io_uring_fill_notification_wait_futex(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_notification_wait_operation_t* wait =
      (iree_async_notification_wait_operation_t*)base_operation;

  wait->wait_token =
      iree_atomic_load(&wait->notification->epoch, iree_memory_order_acquire);

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_FUTEX_WAIT;
  sqe->fd = IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  sqe->addr = (uint64_t)(uintptr_t)&wait->notification->epoch;
  sqe->off = wait->wait_token;
  sqe->len = 0;
  sqe->futex_flags = 0;
  sqe->addr3 = 0xffffffffU;  // FUTEX_BITSET_MATCH_ANY
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills two linked SQEs for a NOTIFICATION_WAIT operation in event mode.
// Uses POLL_ADD linked to READ, same pattern as EVENT_WAIT.
static void iree_async_proactor_io_uring_fill_notification_wait_event(
    iree_io_uring_sqe_t* poll_sqe, iree_io_uring_sqe_t* read_sqe,
    iree_async_operation_t* base_operation) {
  iree_async_notification_wait_operation_t* wait =
      (iree_async_notification_wait_operation_t*)base_operation;

  wait->wait_token =
      iree_atomic_load(&wait->notification->epoch, iree_memory_order_acquire);

  int fd = wait->notification->platform.io_uring.primitive.value.fd;

  // SQE 1: POLL_ADD with link to next SQE.
  // Same TAG_LINKED_POLL pattern as EVENT_WAIT — see fill_event_wait comments.
  memset(poll_sqe, 0, sizeof(*poll_sqe));
  poll_sqe->opcode = IREE_IORING_OP_POLL_ADD;
  poll_sqe->flags = IREE_IOSQE_IO_LINK | IREE_IOSQE_CQE_SKIP_SUCCESS;
  poll_sqe->fd = fd;
  poll_sqe->poll32_events = POLLIN;
  poll_sqe->user_data = iree_io_uring_internal_encode(
      IREE_IO_URING_TAG_LINKED_POLL, (uintptr_t)base_operation);

  // SQE 2: READ to drain the eventfd counter.
  memset(read_sqe, 0, sizeof(*read_sqe));
  read_sqe->opcode = IREE_IORING_OP_READ;
  read_sqe->fd = fd;
  read_sqe->addr =
      (uint64_t)(uintptr_t)&wait->notification->platform.io_uring.drain_buffer;
  read_sqe->len = sizeof(wait->notification->platform.io_uring.drain_buffer);
  read_sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills SQE for a NOTIFICATION_SIGNAL operation in futex mode.
static void iree_async_proactor_io_uring_fill_notification_signal_futex(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_notification_signal_operation_t* signal =
      (iree_async_notification_signal_operation_t*)base_operation;

  signal->woken_count = 0;

  // Increment epoch before kernel FUTEX_WAKE so waiters see the new value.
  iree_atomic_fetch_add(&signal->notification->epoch, 1,
                        iree_memory_order_release);

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_FUTEX_WAKE;
  sqe->fd = IREE_ASYNC_FUTEX_SIZE_U32 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  sqe->addr = (uint64_t)(uintptr_t)&signal->notification->epoch;
  sqe->off = (uint64_t)signal->wake_count;
  sqe->len = 0;
  sqe->futex_flags = 0;
  sqe->addr3 = 0xffffffffU;  // FUTEX_BITSET_MATCH_ANY
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills SQE for a NOTIFICATION_SIGNAL operation in event mode.
static void iree_async_proactor_io_uring_fill_notification_signal_event(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_notification_signal_operation_t* signal =
      (iree_async_notification_signal_operation_t*)base_operation;

  signal->woken_count = 0;

  // Increment epoch before kernel writes to eventfd.
  iree_atomic_fetch_add(&signal->notification->epoch, 1,
                        iree_memory_order_release);

  int fd = signal->notification->platform.io_uring.primitive.value.fd;

  // With EFD_SEMAPHORE, write(N) allows N read()s to succeed.
  signal->write_value =
      (signal->wake_count > 0) ? (uint64_t)signal->wake_count : UINT32_MAX;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_WRITE;
  sqe->fd = fd;
  sqe->addr = (uint64_t)(uintptr_t)&signal->write_value;
  sqe->len = sizeof(signal->write_value);
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

//===----------------------------------------------------------------------===//
// Semaphore operation helpers
//===----------------------------------------------------------------------===//

// Executes a SEMAPHORE_SIGNAL operation synchronously.
// Signal operations execute inline during submit - no actual kernel operation
// is needed. We use a NOP to trigger the completion callback.
//
// Errors are stored in base.next (repurposed as status storage) rather than
// returned, so that submit succeeds and errors are delivered via callback.
// This maintains standard async operation semantics.
static void iree_async_proactor_io_uring_execute_semaphore_signal(
    iree_async_semaphore_signal_operation_t* signal_op) {
  // Signal each semaphore in turn.
  for (iree_host_size_t i = 0; i < signal_op->count; ++i) {
    iree_status_t status = iree_async_semaphore_signal(
        signal_op->semaphores[i], signal_op->values[i], signal_op->frontier);
    if (!iree_status_is_ok(status)) {
      // Store error in base.next for retrieval when CQE is processed.
      signal_op->base.next = (iree_async_operation_t*)(uintptr_t)status;
      return;
    }
  }
  // Success - clear any previous error storage.
  signal_op->base.next = NULL;
}

// Fills an SQE for a SEMAPHORE_SIGNAL operation.
// The actual signal has already been executed; this NOP triggers the callback.
static void iree_async_proactor_io_uring_fill_semaphore_signal(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* operation) {
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;
  sqe->user_data = (uint64_t)(uintptr_t)operation;
}

// Computes the allocation size for a semaphore wait tracker with |count|
// semaphores using overflow-checked arithmetic.
static inline iree_status_t iree_async_io_uring_semaphore_wait_tracker_size(
    iree_host_size_t count, iree_host_size_t* out_size) {
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_async_io_uring_semaphore_wait_tracker_t), out_size,
      IREE_STRUCT_FIELD_FAM(count, iree_async_semaphore_timepoint_t));
}

static void iree_async_io_uring_semaphore_wait_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status);

// Submits a SEMAPHORE_WAIT operation by registering timepoints.
// For single-semaphore waits, registers one timepoint.
// For multi-semaphore waits, registers one per semaphore and tracks progress.
static iree_status_t iree_async_proactor_io_uring_submit_semaphore_wait(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_semaphore_wait_operation_t* wait_op =
      (iree_async_semaphore_wait_operation_t*)base_operation;

  // Check for immediate satisfaction before allocating a tracker.
  bool all_satisfied = true;
  iree_host_size_t satisfied_index = 0;
  for (iree_host_size_t i = 0; i < wait_op->count; ++i) {
    uint64_t current = iree_async_semaphore_query(wait_op->semaphores[i]);
    if (current >= wait_op->values[i]) {
      if (wait_op->mode == IREE_ASYNC_WAIT_MODE_ANY) {
        // ANY mode: any satisfied semaphore completes immediately.
        wait_op->satisfied_index = i;
        sqe->opcode = IREE_IORING_OP_NOP;
        sqe->fd = -1;
        sqe->user_data = (uint64_t)(uintptr_t)base_operation;
        return iree_ok_status();
      }
      satisfied_index = i;
    } else {
      all_satisfied = false;
    }
  }
  (void)satisfied_index;  // May be used in future for tracking partial state.
  if (all_satisfied) {
    // ALL mode: all satisfied, complete immediately.
    sqe->opcode = IREE_IORING_OP_NOP;
    sqe->fd = -1;
    sqe->user_data = (uint64_t)(uintptr_t)base_operation;
    return iree_ok_status();
  }

  // Allocate tracker with embedded timepoints.
  iree_host_size_t tracker_size = 0;
  IREE_RETURN_IF_ERROR(iree_async_io_uring_semaphore_wait_tracker_size(
      wait_op->count, &tracker_size));
  iree_async_io_uring_semaphore_wait_tracker_t* tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(proactor->base.allocator,
                                             tracker_size, (void**)&tracker));
  memset(tracker, 0, tracker_size);

  tracker->operation = wait_op;
  tracker->proactor = proactor;
  tracker->allocator = proactor->base.allocator;
  tracker->count = wait_op->count;
  iree_atomic_store(&tracker->completion_status, (intptr_t)iree_ok_status(),
                    iree_memory_order_release);

  // Transfer linked_next chain from operation to tracker.
  // The pre-scan built the chain via linked_next pointers; we move the head
  // to the tracker so it survives until the wait completes.
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

  // Store the tracker in the operation's platform-specific storage.
  // This allows cancel to find it.
  wait_op->base.next = (iree_async_operation_t*)tracker;

  // Register timepoints for each semaphore.
  for (iree_host_size_t i = 0; i < wait_op->count; ++i) {
    iree_async_semaphore_timepoint_t* timepoint = &tracker->timepoints[i];
    timepoint->callback = iree_async_io_uring_semaphore_wait_timepoint_callback;
    // Encode the index in user_data so the callback knows which semaphore
    // fired.
    timepoint->user_data = (void*)((uintptr_t)tracker | ((uintptr_t)i << 48));

    iree_status_t status = iree_async_semaphore_acquire_timepoint(
        wait_op->semaphores[i], wait_op->values[i], timepoint);
    if (!iree_status_is_ok(status)) {
      // Registration failed. Cancel already-registered timepoints.
      for (iree_host_size_t j = 0; j < i; ++j) {
        iree_async_semaphore_cancel_timepoint(wait_op->semaphores[j],
                                              &tracker->timepoints[j]);
      }
      iree_allocator_free(tracker->allocator, tracker);
      wait_op->base.next = NULL;
      return status;
    }
  }

  // Use a NOP SQE to create a slot in the ring for this operation.
  // The operation will actually be completed when the timepoints fire.
  // We mark the SQE as linked-hardlink so that if this is part of a chain,
  // the chain will wait for this operation's completion.
  sqe->opcode = IREE_IORING_OP_NOP;
  sqe->fd = -1;
  // Tag with internal marker so poll knows to handle this specially.
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL,
                                                 (uintptr_t)tracker);

  return iree_ok_status();
}

// Helper to enqueue a tracker for completion and wake the proactor.
// Multiple callbacks may race to enqueue (error callbacks in ALL mode, success
// callbacks, cancel). The enqueued CAS ensures exactly one push to the MPSC
// slist — a duplicate push would create a self-loop and hang the drain.
static void iree_async_io_uring_semaphore_wait_enqueue_completion(
    iree_async_io_uring_semaphore_wait_tracker_t* tracker) {
  int32_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(&tracker->enqueued, &expected, 1,
                                           iree_memory_order_acq_rel,
                                           iree_memory_order_relaxed)) {
    return;
  }
  iree_atomic_slist_push(&tracker->proactor->pending_semaphore_waits,
                         &tracker->slist_entry);
  uint64_t wake_value = 1;
  (void)write(tracker->proactor->wake_eventfd, &wake_value, sizeof(wake_value));
}

// Timepoint callback for SEMAPHORE_WAIT operations.
// Called under the semaphore's lock - MUST be fast and non-blocking.
static void iree_async_io_uring_semaphore_wait_timepoint_callback(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  // Decode tracker and index from user_data.
  uintptr_t encoded = (uintptr_t)user_data;
  iree_async_io_uring_semaphore_wait_tracker_t* tracker =
      (iree_async_io_uring_semaphore_wait_tracker_t*)(encoded &
                                                      0x0000FFFFFFFFFFFFull);
  iree_host_size_t index = (iree_host_size_t)(encoded >> 48);

  if (!iree_status_is_ok(status)) {
    // Failure or cancellation. Store the error status (first one wins).
    intptr_t expected = (intptr_t)iree_ok_status();
    if (!iree_atomic_compare_exchange_strong(
            &tracker->completion_status, &expected, (intptr_t)status,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      // Another callback already stored a status; ignore this one.
      iree_status_ignore(status);
    }
    // Enqueue for completion regardless of whether we won the status race.
    // The enqueued CAS guard inside enqueue_completion ensures exactly one
    // push.
    iree_async_io_uring_semaphore_wait_enqueue_completion(tracker);
    return;
  }

  // Success case.
  if (tracker->operation->mode == IREE_ASYNC_WAIT_MODE_ANY) {
    // ANY mode: first satisfied index wins.
    int32_t expected = -1;
    if (iree_atomic_compare_exchange_strong(
            &tracker->remaining_or_satisfied, &expected, (int32_t)index,
            iree_memory_order_acq_rel, iree_memory_order_acquire)) {
      // We won the race - enqueue for completion.
      iree_async_io_uring_semaphore_wait_enqueue_completion(tracker);
    }
    // Otherwise another callback already completed it; nothing to do.
  } else {
    // ALL mode: decrement remaining count.
    int32_t remaining = iree_atomic_fetch_sub(&tracker->remaining_or_satisfied,
                                              1, iree_memory_order_acq_rel) -
                        1;
    if (remaining == 0) {
      // All semaphores satisfied - enqueue for completion.
      iree_async_io_uring_semaphore_wait_enqueue_completion(tracker);
    }
  }
}

//===----------------------------------------------------------------------===//
// Message operation helpers
//===----------------------------------------------------------------------===//

// Fills an SQE for a MESSAGE operation (cross-proactor messaging via MSG_RING).
//
// MSG_RING posts a CQE directly to the target ring without any userspace
// involvement on the target. The message_data is delivered via:
//   - sqe->len (32 bits) -> target cqe->res
//   - sqe->off (64 bits) -> target cqe->user_data
//
// We encode the message_data in the target's user_data along with an internal
// marker so the target proactor can identify it as an incoming message rather
// than a normal operation completion.
static void iree_async_proactor_io_uring_fill_message(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_message_operation_t* message =
      (iree_async_message_operation_t*)base_operation;

  // Get target ring fd from the target proactor.
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(message->target);

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_MSG_RING;
  sqe->fd = target->ring.ring_fd;

  // The target CQE will receive:
  //   cqe->res = sqe->len (lower 32 bits of message_data)
  //   cqe->user_data = sqe->off (internal encoding with upper 32 bits)
  //
  // We split the 64-bit message_data across both fields to preserve all bits:
  //   - Lower 32 bits go in sqe->len → cqe->res
  //   - Upper 32 bits go in the payload of sqe->off → cqe->user_data
  // The decode reconstructs the full 64-bit value from both fields.
  sqe->len = (uint32_t)(message->message_data & 0xFFFFFFFFULL);
  sqe->off = iree_io_uring_internal_encode(IREE_IO_URING_TAG_MESSAGE_RECEIVE,
                                           message->message_data >> 32);

  // Handle source completion suppression.
  if (message->message_flags & IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION) {
    // User doesn't want a source callback. We mark this as internal with
    // MESSAGE_SOURCE tag so we can discard the CQE when it arrives.
    //
    // NOTE: The kernel has IORING_MSG_RING_CQE_SKIP to suppress the source CQE
    // entirely, but there's no reliable way to detect support for it (it's
    // separate from FEAT_CQE_SKIP which is about IOSQE_CQE_SKIP_SUCCESS). We
    // could try setting it and handle EINVAL, but that would require complex
    // retry logic. For now, we always receive and discard the source CQE.
    sqe->user_data =
        iree_io_uring_internal_encode(IREE_IO_URING_TAG_MESSAGE_SOURCE, 0);
  } else {
    // Normal case: source gets a completion via user callback.
    sqe->user_data = (uint64_t)(uintptr_t)base_operation;
  }
}

// Fills an SQE for a MESSAGE operation using the fallback path.
//
// When MSG_RING is not available (kernel < 5.18), we use:
//   1. Message pool send to target's pre-allocated pool
//   2. eventfd WRITE to wake target's poll()
//
// The eventfd WRITE is linkable on io_uring, so LINK chains like
// RECV -> MESSAGE still work correctly. The message data is delivered
// via the pool (no per-message heap allocation).
static iree_status_t iree_async_proactor_io_uring_fill_message_fallback(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_message_operation_t* message =
      (iree_async_message_operation_t*)base_operation;

  // Get target proactor.
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(message->target);

  // Send message data via the target's pre-allocated pool.
  IREE_RETURN_IF_ERROR(iree_async_message_pool_send(&target->message_pool,
                                                    message->message_data));

  // Fill SQE: WRITE to target's wake eventfd (LINKABLE!).
  // We use a static constant for write_value rather than storing it in the
  // operation struct. This is critical for SKIP_SOURCE_COMPLETION: the user
  // may free the operation struct immediately after submit, but the kernel
  // still needs to read the value when processing the WRITE SQE.
  static const uint64_t kWakeValue = 1;
  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_WRITE;
  sqe->fd = target->wake_eventfd;
  sqe->addr = (uint64_t)(uintptr_t)&kWakeValue;
  sqe->len = sizeof(kWakeValue);

  if (message->message_flags & IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION) {
    // Use MESSAGE_SOURCE tag for explicit handling, matching MSG_RING fast
    // path.
    sqe->user_data =
        iree_io_uring_internal_encode(IREE_IO_URING_TAG_MESSAGE_SOURCE, 0);
  } else {
    sqe->user_data = (uint64_t)(uintptr_t)base_operation;
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Resource cleanup
//===----------------------------------------------------------------------===//

// Releases resources retained during submission for a batch of operations.
// Called on submit rollback to avoid leaking retained references.
static void iree_async_proactor_io_uring_release_prepared(
    iree_async_operation_list_t operations, iree_host_size_t prepared_count) {
  for (iree_host_size_t i = 0; i < prepared_count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    iree_async_operation_release_resources(operation);
  }
}

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

// Dispatches standalone SEQUENCE operations in the batch. Must be called before
// acquiring the SQ lock because submit_as_linked re-enters the submit function
// through the vtable for the expanded steps, which acquires the lock itself.
//
// SEQUENCE continuations (preceded by a LINKED predecessor) are not handled
// here — they're held in linked_next chains and dispatched on predecessor
// completion, at which point the SQ lock is not held.
static iree_status_t iree_async_proactor_io_uring_submit_sequences(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (operations.values[i]->type != IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
      continue;
    }
    // Skip continuations — dispatched via linked_next on predecessor
    // completion.
    if (i > 0 && iree_any_bit_set(operations.values[i - 1]->flags,
                                  IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }
    iree_async_sequence_operation_t* sequence =
        (iree_async_sequence_operation_t*)operations.values[i];
    iree_status_t status;
    if (!sequence->step_fn) {
      status = iree_async_sequence_submit_as_linked(base_proactor, sequence);
    } else {
      status = iree_async_sequence_emulation_begin(&proactor->sequence_emulator,
                                                   sequence);
    }
    if (!iree_status_is_ok(status)) return status;
  }
  return iree_ok_status();
}

iree_status_t iree_async_proactor_io_uring_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (operations.count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_submit_sequences(
      proactor, base_proactor, operations));

  // Calculate total SQEs needed.
  // EVENT_WAIT: always 2 SQEs (linked POLL_ADD+READ).
  // NOTIFICATION_WAIT: 2 SQEs in event mode, 1 SQE in futex mode.
  iree_host_size_t sqes_needed = 0;
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_type_t type = operations.values[i]->type;
    // Standalone SEQUENCE operations were handled in the pre-scan above.
    // SEQUENCE continuations (in linked_next chains past the split point)
    // don't consume SQEs either — they're dispatched on predecessor completion.
    if (type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) continue;
    if (type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
      sqes_needed += 2;
    } else if (type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
      // Check the notification's mode to determine SQE count.
      iree_async_notification_wait_operation_t* wait =
          (iree_async_notification_wait_operation_t*)operations.values[i];
      sqes_needed +=
          (wait->notification->mode == IREE_ASYNC_NOTIFICATION_MODE_FUTEX) ? 1
                                                                           : 2;
    } else {
      sqes_needed += 1;
    }
  }

  // All operations were SEQUENCE (handled in pre-scan). Nothing left to submit.
  if (sqes_needed == 0) return iree_ok_status();

  // Acquire the SQ lock for the duration of SQE preparation and capacity check.
  // All sq_local_tail reads/writes must happen under this lock to prevent a
  // concurrent thread from seeing a partially-filled SQE during flush.
  iree_io_uring_ring_sq_lock(&proactor->ring);

  // Check SQ capacity.
  uint32_t available = iree_io_uring_ring_sq_space_left(&proactor->ring);
  if (available < sqes_needed) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ has %u slots but %zu SQEs needed for %zu "
                            "operations",
                            available, sqes_needed, operations.count);
  }

  // Validate LINKED flag usage: the last operation must not have LINKED set.
  // LINKED means "link to next operation" - there is no next for the last one.
  // io_uring would treat this as linking to the next submit batch, which could
  // cause unrelated future operations to be spuriously cancelled.
  iree_async_operation_t* last_operation =
      operations.values[operations.count - 1];
  if (iree_any_bit_set(last_operation->flags,
                       IREE_ASYNC_OPERATION_FLAG_LINKED)) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LINKED flag set on last operation in batch; LINKED means 'link to "
        "next' but there is no next operation");
  }

  // Build linked_next chain for LINKED operations and find split points.
  //
  // LINKED operations form intrusive chains via the base linked_next pointer.
  // When the head of a chain completes, the proactor submits its linked_next
  // continuation (on success) or cancels the chain (on failure).
  //
  // Most operation types support kernel LINK (IOSQE_IO_LINK) for zero-overhead
  // chaining. However, some operations require userspace chain emulation:
  //
  // Timers (ETIME_SUCCESS, 5.16+):
  //   ASYNC_CANCEL on a mid-chain operation has problematic semantics:
  //   the kernel doesn't reliably post CQEs for not-yet-issued linked ops,
  //   so C in "A->B->C" may never complete if B is cancelled.
  //
  // Semaphore WAIT:
  //   Uses software timepoints, not kernel operations. The kernel NOP completes
  //   immediately, but the actual wait may take longer.
  //
  // Semaphore SIGNAL:
  //   Executes inline during submit with errors stored in base.next. The kernel
  //   NOP always succeeds, so LINK would continue the chain on signal failure.
  //
  // For these operations, we split the batch: only submit SQEs up to the
  // emulated operation, and the rest are held in the linked_next chain for
  // submission on completion.
  iree_host_size_t effective_count = operations.count;

  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    operation->linked_next = NULL;

    // Standalone SEQUENCE operations were handled in the pre-scan.
    // Skip them in chain building (they don't participate in LINKED chains).
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE &&
        (i == 0 || !iree_any_bit_set(operations.values[i - 1]->flags,
                                     IREE_ASYNC_OPERATION_FLAG_LINKED))) {
      continue;
    }

    if (!iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_LINKED)) {
      continue;
    }

    // Build the intrusive linked list for the chain.
    operation->linked_next = operations.values[i + 1];

    // Split the batch at the first operation requiring userspace emulation.
    // Also split when the successor is a SEQUENCE operation — SEQUENCE can't
    // be represented as a kernel SQE. The SEQUENCE continuation is held in
    // linked_next and dispatched on predecessor completion (without SQ lock).
    if (effective_count == operations.count) {
      bool needs_userspace_emulation =
          (operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER) ||
          (operation->type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT) ||
          (operation->type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL) ||
          (operations.values[i + 1]->type ==
           IREE_ASYNC_OPERATION_TYPE_SEQUENCE);
      if (needs_userspace_emulation) {
        effective_count = i + 1;
        // Continue building linked_next for remaining operations in the chain.
      }
    }
  }

  // Fill SQEs for each operation. Track count for rollback on failure.
  // Use effective_count (may be less than operations.count if timer emulation
  // split the batch).
  iree_host_size_t sqes_prepared = 0;
  for (iree_host_size_t i = 0; i < effective_count; ++i) {
    iree_async_operation_t* operation = operations.values[i];

    // Standalone SEQUENCE operations were handled in the pre-scan (before the
    // SQ lock). Skip them here — they don't consume SQEs.
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) continue;

    // Retain resources referenced by this operation to prevent premature
    // destruction while the SQE is in flight. On rollback, release_prepared
    // will undo these retains for all operations [0, i+1).
    iree_async_operation_retain_resources(operation);

    iree_status_t status = iree_ok_status();

    // EVENT_WAIT always uses linked POLL_ADD+READ (2 SQEs).
    // NOTIFICATION_WAIT uses 2 SQEs in event mode, 1 SQE in futex mode.
    bool needs_two_sqes = false;
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
      needs_two_sqes = true;
    } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
      iree_async_notification_wait_operation_t* wait =
          (iree_async_notification_wait_operation_t*)operation;
      needs_two_sqes =
          (wait->notification->mode == IREE_ASYNC_NOTIFICATION_MODE_EVENT);
    }

    if (needs_two_sqes) {
      iree_io_uring_sqe_t* poll_sqe =
          iree_io_uring_ring_get_sqe(&proactor->ring);
      iree_io_uring_sqe_t* read_sqe =
          iree_io_uring_ring_get_sqe(&proactor->ring);
      if (!poll_sqe || !read_sqe) {
        // Should not happen since we pre-checked capacity.
        iree_io_uring_ring_sq_rollback(&proactor->ring,
                                       (uint32_t)sqes_prepared);
        iree_io_uring_ring_sq_unlock(&proactor->ring);
        iree_async_proactor_io_uring_release_prepared(operations, i + 1);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to get SQEs for 2-SQE op %zu", i);
      }
      sqes_prepared += 2;

      if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
        iree_async_proactor_io_uring_fill_event_wait(poll_sqe, read_sqe,
                                                     operation);
      } else {
        iree_async_proactor_io_uring_fill_notification_wait_event(
            poll_sqe, read_sqe, operation);
      }

      // Apply kernel LINK to the terminal SQE of the internal chain.
      // Only if the next operation will also be submitted in this batch
      // (otherwise the chain continues in userspace via linked_next).
      if (iree_any_bit_set(operation->flags,
                           IREE_ASYNC_OPERATION_FLAG_LINKED) &&
          (i + 1 < effective_count)) {
        read_sqe->flags |= IREE_IOSQE_IO_LINK;
        // Kernel handles chaining — clear userspace continuation pointer
        // to prevent double-dispatch on CQE processing.
        operation->linked_next = NULL;
      }
    } else {
      // All other operations use 1 SQE.
      iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
      if (!sqe) {
        iree_io_uring_ring_sq_rollback(&proactor->ring,
                                       (uint32_t)sqes_prepared);
        iree_io_uring_ring_sq_unlock(&proactor->ring);
        iree_async_proactor_io_uring_release_prepared(operations, i + 1);
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "failed to get SQE for operation %zu", i);
      }
      ++sqes_prepared;

      switch (operation->type) {
        case IREE_ASYNC_OPERATION_TYPE_NOP:
          iree_async_proactor_io_uring_fill_nop(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_TIMER:
          iree_async_proactor_io_uring_fill_timer(proactor, sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT:
          status = iree_async_proactor_io_uring_submit_semaphore_wait(
              proactor, sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL: {
          // Execute the signal synchronously, then use a NOP for the callback.
          // Any errors are stored in base.next and delivered via callback.
          iree_async_semaphore_signal_operation_t* signal_op =
              (iree_async_semaphore_signal_operation_t*)operation;
          iree_async_proactor_io_uring_execute_semaphore_signal(signal_op);
          iree_async_proactor_io_uring_fill_semaphore_signal(sqe, operation);
          break;
        }
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
          iree_async_proactor_io_uring_fill_socket_connect(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
          iree_async_proactor_io_uring_fill_socket_accept(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
          status =
              iree_async_proactor_io_uring_fill_socket_recv(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
          status = iree_async_proactor_io_uring_fill_socket_recv_pool(
              proactor, sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
          status = iree_async_proactor_io_uring_fill_socket_send(proactor, sqe,
                                                                 operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
          status = iree_async_proactor_io_uring_fill_socket_sendto(
              sqe, operation, proactor->capabilities);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
          status =
              iree_async_proactor_io_uring_fill_socket_recvfrom(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
          iree_async_proactor_io_uring_fill_socket_close(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT:
          if (!iree_any_bit_set(
                  proactor->capabilities,
                  IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
            status = iree_make_status(
                IREE_STATUS_UNAVAILABLE,
                "FUTEX_WAIT requires kernel 6.7+ with io_uring futex support");
          } else {
            iree_async_proactor_io_uring_fill_futex_wait(sqe, operation);
          }
          break;
        case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE:
          if (!iree_any_bit_set(
                  proactor->capabilities,
                  IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
            status = iree_make_status(
                IREE_STATUS_UNAVAILABLE,
                "FUTEX_WAKE requires kernel 6.7+ with io_uring futex support");
          } else {
            iree_async_proactor_io_uring_fill_futex_wake(sqe, operation);
          }
          break;
        case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT:
          // Futex mode only here - event mode uses 2 SQEs and is handled above.
          iree_async_proactor_io_uring_fill_notification_wait_futex(sqe,
                                                                    operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL: {
          iree_async_notification_signal_operation_t* signal_op =
              (iree_async_notification_signal_operation_t*)operation;
          if (signal_op->notification->mode ==
              IREE_ASYNC_NOTIFICATION_MODE_FUTEX) {
            iree_async_proactor_io_uring_fill_notification_signal_futex(
                sqe, operation);
          } else {
            iree_async_proactor_io_uring_fill_notification_signal_event(
                sqe, operation);
          }
          break;
        }
        case IREE_ASYNC_OPERATION_TYPE_MESSAGE: {
          // Validate that target is an io_uring proactor. Cross-backend
          // messaging is not supported since each backend has different
          // internal structures for message delivery.
          iree_async_message_operation_t* message_op =
              (iree_async_message_operation_t*)operation;
          if (!message_op->target || message_op->target->vtable !=
                                         &iree_async_proactor_io_uring_vtable) {
            status = iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "MESSAGE target must be an io_uring proactor from the same "
                "backend; cross-backend messaging is not supported");
            break;
          }

          if (iree_any_bit_set(
                  proactor->capabilities,
                  IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING)) {
            // Fast path: kernel-mediated MSG_RING (5.18+).
            iree_async_proactor_io_uring_fill_message(proactor, sqe, operation);
          } else {
            // Fallback path: MPSC queue + eventfd WRITE.
            status = iree_async_proactor_io_uring_fill_message_fallback(
                proactor, sqe, operation);
          }
          break;
        }
        case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
          iree_async_proactor_io_uring_fill_file_open(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
          iree_async_proactor_io_uring_fill_file_read(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
          iree_async_proactor_io_uring_fill_file_write(sqe, operation);
          break;
        case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
          iree_async_proactor_io_uring_fill_file_close(sqe, operation);
          break;
        default:
          status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                    "operation type %d not yet implemented",
                                    (int)operation->type);
          break;
      }

      // Apply kernel LINK to create kernel-enforced operation chains.
      // Only if the next operation will also be submitted in this batch
      // (otherwise the chain continues in userspace via linked_next).
      if (iree_any_bit_set(operation->flags,
                           IREE_ASYNC_OPERATION_FLAG_LINKED) &&
          (i + 1 < effective_count)) {
        sqe->flags |= IREE_IOSQE_IO_LINK;
        // Kernel handles chaining — clear userspace continuation pointer
        // to prevent double-dispatch on CQE processing.
        operation->linked_next = NULL;
      }
    }

    if (!iree_status_is_ok(status)) {
      // Rollback all SQEs and release retained resources for all operations
      // that were retained, including the current one (i) whose fill failed.
      iree_io_uring_ring_sq_rollback(&proactor->ring, (uint32_t)sqes_prepared);
      iree_io_uring_ring_sq_unlock(&proactor->ring);
      iree_async_proactor_io_uring_release_prepared(operations, i + 1);
      return status;
    }

    IREE_TRACE({ operation->submit_time_ns = iree_time_now(); });
  }

  // Release the SQ lock. All SQEs are fully filled; sq_local_tail is advanced.
  // The SQEs are not yet visible to the kernel (*sq_tail is unchanged).
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // During CQE processing, the poll function sets defer_submissions to prevent
  // io_uring_enter from generating synchronous CQEs that the CQE loop would
  // pick up (creating infinite re-submission loops). The SQEs remain in the
  // submission ring and are flushed after CQE processing completes.
  if (proactor->defer_submissions) {
    return iree_ok_status();
  }

  // Only the poll thread may call io_uring_enter (SINGLE_ISSUER constraint).
  // Wake the poll thread so its next ring_submit flushes *sq_tail and enters
  // the kernel. The eventfd write causes the poll thread's blocking
  // io_uring_enter to return via the wake POLL_ADD completion.
  iree_async_proactor_wake(&proactor->base);
  return iree_ok_status();
}
