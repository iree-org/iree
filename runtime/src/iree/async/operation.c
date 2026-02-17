// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/operation.h"

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/socket.h"

void iree_async_operation_retain_resources(iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT: {
      iree_async_socket_accept_operation_t* accept =
          (iree_async_socket_accept_operation_t*)operation;
      iree_async_socket_retain(accept->listen_socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      iree_async_socket_connect_operation_t* connect_op =
          (iree_async_socket_connect_operation_t*)operation;
      iree_async_socket_retain(connect_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      iree_async_socket_retain(recv->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      iree_async_socket_recv_pool_operation_t* recv_pool =
          (iree_async_socket_recv_pool_operation_t*)operation;
      iree_async_socket_retain(recv_pool->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      iree_async_socket_send_operation_t* send =
          (iree_async_socket_send_operation_t*)operation;
      iree_async_socket_retain(send->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      iree_async_socket_recvfrom_operation_t* recvfrom =
          (iree_async_socket_recvfrom_operation_t*)operation;
      iree_async_socket_retain(recvfrom->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      iree_async_socket_sendto_operation_t* sendto_op =
          (iree_async_socket_sendto_operation_t*)operation;
      iree_async_socket_retain(sendto_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      iree_async_event_wait_operation_t* event_wait =
          (iree_async_event_wait_operation_t*)operation;
      iree_async_event_retain(event_wait->event);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      iree_async_notification_wait_operation_t* notification_wait =
          (iree_async_notification_wait_operation_t*)operation;
      iree_async_notification_retain(notification_wait->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL: {
      iree_async_notification_signal_operation_t* notification_signal =
          (iree_async_notification_signal_operation_t*)operation;
      iree_async_notification_retain(notification_signal->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      iree_async_file_retain(read_op->file);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      iree_async_file_retain(write_op->file);
      break;
    }
    default:
      break;
  }
}

void iree_async_operation_release_resources(iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT: {
      iree_async_socket_accept_operation_t* accept =
          (iree_async_socket_accept_operation_t*)operation;
      iree_async_socket_release(accept->listen_socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      iree_async_socket_connect_operation_t* connect_op =
          (iree_async_socket_connect_operation_t*)operation;
      iree_async_socket_release(connect_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      iree_async_socket_release(recv->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      iree_async_socket_recv_pool_operation_t* recv_pool =
          (iree_async_socket_recv_pool_operation_t*)operation;
      iree_async_socket_release(recv_pool->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      iree_async_socket_send_operation_t* send =
          (iree_async_socket_send_operation_t*)operation;
      iree_async_socket_release(send->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      iree_async_socket_recvfrom_operation_t* recvfrom =
          (iree_async_socket_recvfrom_operation_t*)operation;
      iree_async_socket_release(recvfrom->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      iree_async_socket_sendto_operation_t* sendto_op =
          (iree_async_socket_sendto_operation_t*)operation;
      iree_async_socket_release(sendto_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE: {
      // Close consumes the caller's reference. This release IS the consumption,
      // with no prior retain to balance it.
      iree_async_socket_close_operation_t* close_op =
          (iree_async_socket_close_operation_t*)operation;
      iree_async_socket_release(close_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      iree_async_event_wait_operation_t* event_wait =
          (iree_async_event_wait_operation_t*)operation;
      iree_async_event_release(event_wait->event);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      iree_async_notification_wait_operation_t* notification_wait =
          (iree_async_notification_wait_operation_t*)operation;
      iree_async_notification_release(notification_wait->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL: {
      iree_async_notification_signal_operation_t* signal =
          (iree_async_notification_signal_operation_t*)operation;
      iree_async_notification_release(signal->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      iree_async_file_release(read_op->file);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      iree_async_file_release(write_op->file);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE: {
      // Close consumes the caller's reference. This release IS the consumption,
      // with no prior retain to balance it.
      iree_async_file_close_operation_t* file_close_op =
          (iree_async_file_close_operation_t*)operation;
      iree_async_file_release(file_close_op->file);
      break;
    }
    default:
      break;
  }
}
