// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IOCP proactor socket lifecycle implementation.
//
// Internal header for the IOCP proactor. Functions here are called by
// the proactor's vtable methods to implement socket create/import/destroy.

#ifndef IREE_ASYNC_PLATFORM_IOCP_SOCKET_H_
#define IREE_ASYNC_PLATFORM_IOCP_SOCKET_H_

#include "iree/async/primitive.h"
#include "iree/async/socket.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_iocp_t iree_async_proactor_iocp_t;

// Creates a new socket bound to the IOCP proactor.
// Allocates the socket struct, creates the underlying WSASocket with
// WSA_FLAG_OVERLAPPED, applies options via setsockopt, associates the socket
// with the IOCP completion port, and initializes WSA extension function
// pointers on first use.
iree_status_t iree_async_iocp_socket_create(
    iree_async_proactor_iocp_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket);

// Imports an existing platform socket handle as a proactor-managed socket.
// The proactor takes ownership of the handle (will close it on destroy).
// The socket is associated with the IOCP completion port. Socket options are
// not applied — caller is responsible for pre-configuration including
// WSA_FLAG_OVERLAPPED at socket creation time. The |flags| parameter declares
// runtime behavior (e.g., ZERO_COPY for sockets where the caller already set
// SO_ZEROCOPY).
iree_status_t iree_async_iocp_socket_import(
    iree_async_proactor_iocp_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket);

// Destroys a socket created or imported by this proactor.
// Closes the underlying socket handle via closesocket() and frees the socket
// struct.
void iree_async_iocp_socket_destroy(iree_async_proactor_iocp_t* proactor,
                                    iree_async_socket_t* socket);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IOCP_SOCKET_H_
