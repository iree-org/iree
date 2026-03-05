// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Private header shared by factory.c, factory_unix.c, and factory_win32.c.
//
// Contains the factory struct definition and declarations for shared utility
// functions that the platform-specific cross-process implementations need.

#ifndef IREE_NET_CARRIER_SHM_FACTORY_INTERNAL_H_
#define IREE_NET_CARRIER_SHM_FACTORY_INTERNAL_H_

#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/net/connection.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Internal types
//===----------------------------------------------------------------------===//

typedef struct iree_net_shm_listener_t iree_net_shm_listener_t;

// Entry in the proactor->shared_wake lookup table.
typedef struct iree_net_shm_proactor_wake_t {
  iree_async_proactor_t* proactor;
  iree_net_shm_shared_wake_t* shared_wake;
} iree_net_shm_proactor_wake_t;

typedef struct iree_net_shm_factory_t {
  iree_net_transport_factory_t base;
  iree_slim_mutex_t mutex;
  // Dynamic array of active listener pointers. Listeners add/remove themselves
  // during create_listener and stop.
  iree_net_shm_listener_t** listeners;
  iree_host_size_t listener_count;
  iree_host_size_t listener_capacity;
  iree_net_shm_carrier_options_t options;
  // Proactor -> shared_wake lookup table. Each proactor gets at most one
  // shared_wake, created lazily on first use and released when the factory
  // is destroyed. Grows dynamically (one entry per NUMA node in practice).
  iree_net_shm_proactor_wake_t* proactor_wakes;
  iree_host_size_t proactor_wake_count;
  iree_host_size_t proactor_wake_capacity;
  iree_allocator_t host_allocator;
} iree_net_shm_factory_t;

//===----------------------------------------------------------------------===//
// Shared utilities (defined in factory.c)
//===----------------------------------------------------------------------===//

// Gets or creates a shared_wake for the given proactor. Caller must hold
// factory->mutex. On success, the returned shared_wake is owned by the factory
// (caller does not need to release it).
iree_status_t iree_net_shm_factory_get_or_create_shared_wake(
    iree_net_shm_factory_t* factory, iree_async_proactor_t* proactor,
    iree_net_shm_shared_wake_t** out_shared_wake);

// Creates an SHM connection with the given initial carrier. The connection
// takes ownership of |initial_carrier| -- the caller must not release it on
// success. On failure, the caller retains ownership.
iree_status_t iree_net_shm_connection_create(
    iree_async_proactor_t* proactor, iree_net_carrier_t* initial_carrier,
    iree_async_buffer_pool_t* recv_pool, iree_allocator_t host_allocator,
    iree_net_connection_t** out_connection);

//===----------------------------------------------------------------------===//
// Platform-specific cross-process operations
//===----------------------------------------------------------------------===//

#if !defined(IREE_PLATFORM_WINDOWS)

// Creates a cross-process listener bound to a Unix domain socket path.
// Parses the address, creates the socket, binds, listens, and submits the
// initial accept operation.
iree_status_t iree_net_shm_factory_create_listener_unix(
    iree_net_shm_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener);

// Initiates a cross-process connect to a Unix domain socket path. Parses the
// address, creates a socket, and submits an async connect operation. On
// completion, the handshake runs and the carrier is created.
iree_status_t iree_net_shm_factory_connect_unix(
    iree_net_shm_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data);

#else  // IREE_PLATFORM_WINDOWS

// Creates a cross-process listener on a Windows named pipe. Creates the pipe
// instance, submits ConnectNamedPipe via EVENT_WAIT, and accepts incoming
// connections.
iree_status_t iree_net_shm_factory_create_listener_win32(
    iree_net_shm_factory_t* factory, iree_string_view_t bind_address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_listener_accept_callback_t accept_callback, void* user_data,
    iree_allocator_t host_allocator, iree_net_listener_t** out_listener);

// Initiates a cross-process connect to a Windows named pipe. Opens the pipe
// via CreateFile, runs the handshake, and delivers the connection via callback.
iree_status_t iree_net_shm_factory_connect_win32(
    iree_net_shm_factory_t* factory, iree_string_view_t address,
    iree_async_proactor_t* proactor, iree_async_buffer_pool_t* recv_pool,
    iree_net_transport_connect_callback_t callback, void* user_data);

#endif  // IREE_PLATFORM_WINDOWS

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_FACTORY_INTERNAL_H_
