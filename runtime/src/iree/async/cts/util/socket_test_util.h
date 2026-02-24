// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Initialization helpers for socket CTS test operations.
//
// Each Init* function prepares a caller-owned stack operation struct for
// submission. The caller retains full access to result fields (bytes_sent,
// accepted_socket, etc.) and is responsible for calling
// iree_async_proactor_submit_one() after initialization.
//
// All functions take C callback types (iree_async_completion_fn_t + void*)
// to avoid coupling to CompletionTracker/CompletionLog and to prevent
// circular includes. This header only includes C headers.

#ifndef IREE_ASYNC_CTS_UTIL_SOCKET_TEST_UTIL_H_
#define IREE_ASYNC_CTS_UTIL_SOCKET_TEST_UTIL_H_

#include <string.h>

#include "iree/async/operations/net.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares an accept operation on |listen_socket|.
static inline void InitAcceptOperation(
    iree_async_socket_accept_operation_t* operation,
    iree_async_socket_t* listen_socket,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT;
  operation->listen_socket = listen_socket;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a multishot accept operation on |listen_socket|.
// Delivers repeated connections until cancelled.
static inline void InitMultishotAcceptOperation(
    iree_async_socket_accept_operation_t* operation,
    iree_async_socket_t* listen_socket,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT;
  operation->base.flags = IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
  operation->listen_socket = listen_socket;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a connect operation to |address| on |socket|.
static inline void InitConnectOperation(
    iree_async_socket_connect_operation_t* operation,
    iree_async_socket_t* socket, iree_async_address_t address,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT;
  operation->socket = socket;
  operation->address = address;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a send operation on |socket| with the given buffers and flags.
static inline void InitSendOperation(
    iree_async_socket_send_operation_t* operation, iree_async_socket_t* socket,
    iree_async_span_t* buffers, iree_host_size_t buffer_count,
    iree_async_socket_send_flags_t send_flags,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND;
  operation->socket = socket;
  operation->buffers.values = buffers;
  operation->buffers.count = buffer_count;
  operation->send_flags = send_flags;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a recv operation on |socket| with the given buffers.
static inline void InitRecvOperation(
    iree_async_socket_recv_operation_t* operation, iree_async_socket_t* socket,
    iree_async_span_t* buffers, iree_host_size_t buffer_count,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  operation->socket = socket;
  operation->buffers.values = buffers;
  operation->buffers.count = buffer_count;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a multishot recv operation on |socket| with the given buffers.
// Delivers repeated completions until cancelled.
static inline void InitMultishotRecvOperation(
    iree_async_socket_recv_operation_t* operation, iree_async_socket_t* socket,
    iree_async_span_t* buffers, iree_host_size_t buffer_count,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV;
  operation->base.flags = IREE_ASYNC_OPERATION_FLAG_MULTISHOT;
  operation->socket = socket;
  operation->buffers.values = buffers;
  operation->buffers.count = buffer_count;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a sendto operation on |socket| with the given buffers and
// destination address. For unconnected UDP or when sending to a specific
// address regardless of connection state.
static inline void InitSendtoOperation(
    iree_async_socket_sendto_operation_t* operation,
    iree_async_socket_t* socket, iree_async_span_t* buffers,
    iree_host_size_t buffer_count, iree_async_socket_send_flags_t send_flags,
    const iree_async_address_t* destination,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO;
  operation->socket = socket;
  operation->buffers.values = buffers;
  operation->buffers.count = buffer_count;
  operation->send_flags = send_flags;
  operation->destination = *destination;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a recvfrom operation on |socket| with the given buffers.
// On completion, |operation->sender| contains the source address of the
// received datagram.
static inline void InitRecvfromOperation(
    iree_async_socket_recvfrom_operation_t* operation,
    iree_async_socket_t* socket, iree_async_span_t* buffers,
    iree_host_size_t buffer_count, iree_async_completion_fn_t completion_fn,
    void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM;
  operation->socket = socket;
  operation->buffers.values = buffers;
  operation->buffers.count = buffer_count;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

// Prepares a close operation for |socket|.
// The socket reference is consumed on submit.
static inline void InitCloseOperation(
    iree_async_socket_close_operation_t* operation, iree_async_socket_t* socket,
    iree_async_completion_fn_t completion_fn, void* user_data) {
  memset(operation, 0, sizeof(*operation));
  operation->base.type = IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE;
  operation->socket = socket;
  operation->base.completion_fn = completion_fn;
  operation->base.user_data = user_data;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_ASYNC_CTS_UTIL_SOCKET_TEST_UTIL_H_
