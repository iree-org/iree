// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generic proactor socket lifecycle implementation.
//
// Internal header for the POSIX proactor. Functions here are called by
// the proactor's vtable methods to implement socket create/import/destroy.

#ifndef IREE_ASYNC_PLATFORM_POSIX_SOCKET_H_
#define IREE_ASYNC_PLATFORM_POSIX_SOCKET_H_

#include "iree/async/primitive.h"
#include "iree/async/socket.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_posix_t iree_async_proactor_posix_t;

// Creates a new socket bound to the POSIX proactor.
// Allocates the socket struct, creates the underlying platform socket,
// applies options via setsockopt, and sets up the initial state.
iree_status_t iree_async_posix_socket_create(
    iree_async_proactor_posix_t* proactor, iree_async_socket_type_t type,
    iree_async_socket_options_t options, iree_async_socket_t** out_socket);

// Imports an existing platform socket handle as a proactor-managed socket.
// The proactor takes ownership of the handle (will close it on destroy).
// Socket options are not applied - caller is responsible for pre-configuration.
// The |flags| parameter declares runtime behavior (e.g., ZERO_COPY for sockets
// where the caller already set SO_ZEROCOPY).
iree_status_t iree_async_posix_socket_import(
    iree_async_proactor_posix_t* proactor, iree_async_primitive_t primitive,
    iree_async_socket_type_t type, iree_async_socket_flags_t flags,
    iree_async_socket_t** out_socket);

// Allocates and initializes a socket struct for an accepted connection.
// The socket is set to CONNECTED state with the given |type| (inherited from
// the listening socket). On success, the socket takes ownership of
// |accepted_fd|. On failure, |accepted_fd| is NOT closed — the caller is
// responsible for cleanup.
iree_status_t iree_async_posix_socket_create_accepted(
    iree_async_proactor_posix_t* proactor, int accepted_fd,
    iree_async_socket_type_t type, iree_async_socket_t** out_socket);

// Destroys a socket created or imported by this proactor.
// Closes the underlying platform handle and frees the socket struct.
void iree_async_posix_socket_destroy(iree_async_proactor_posix_t* proactor,
                                     iree_async_socket_t* socket);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_POSIX_SOCKET_H_
