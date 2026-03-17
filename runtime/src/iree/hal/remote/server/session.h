// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_SESSION_H_
#define IREE_HAL_REMOTE_SERVER_SESSION_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/resource_table.h"
#include "iree/net/channel/queue/queue_channel.h"
#include "iree/net/session.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_remote_server_t iree_hal_remote_server_t;

// Per-client session tracking entry.
// Stored in the server's sessions array (indexed by slot).
typedef struct iree_hal_remote_server_session_t {
  // Back-pointer to the owning server. Used by queue channel callbacks to
  // access server->devices without a global search.
  iree_hal_remote_server_t* server;

  // The net-layer session handling bootstrap and control channel.
  // NULL when the slot is free.
  iree_net_session_t* session;

  // Server-assigned session ID (unique, monotonically increasing).
  uint64_t session_id;

  // Queue channel for HAL command dispatch (NULL until queue endpoint opens).
  // The channel owns the header pool for its frame_sender (freed on channel
  // destroy). This ensures the pool remains valid as long as any reference
  // to the channel exists (e.g., command completion contexts).
  iree_net_queue_channel_t* queue_channel;

  // Resource table mapping resource_ids to retained HAL resources (buffers,
  // semaphores, etc.). Initialized when the session is accepted, deinitialized
  // when the session is removed.
  iree_hal_remote_resource_table_t resource_table;

  // Epoch→local semaphore mapping for wait frontier resolution. Each COMMAND
  // creates a local semaphore for completion tracking; subsequent commands
  // with wait frontiers look up earlier epochs to build local wait semaphore
  // lists. The mapping retains each semaphore and releases them all on session
  // removal.
  struct {
    uint64_t* epochs;
    iree_hal_semaphore_t** semaphores;  // retained
    iree_host_size_t count;
    iree_host_size_t capacity;
  } epoch_semaphore_map;

  // Provisional→resolved resource ID mapping. Populated during BUFFER_ALLOCA
  // processing (the server assigns a canonical ID and records the mapping).
  // Queried during subsequent commands that reference the provisional ID
  // (fill, copy, dealloca, etc.). Entries persist for the session lifetime
  // to handle out-of-order or late-arriving commands.
  struct {
    iree_hal_remote_resource_id_t* provisional_ids;
    iree_hal_remote_resource_id_t* resolved_ids;
    iree_host_size_t count;
    iree_host_size_t capacity;
  } provisional_map;
} iree_hal_remote_server_session_t;

// Called when session bootstrap completes and the session is ready for use.
void iree_hal_remote_server_on_session_ready(
    void* user_data, iree_net_session_t* session,
    const iree_net_session_topology_t* remote_topology);

// Called when the peer initiates a graceful disconnect.
void iree_hal_remote_server_on_session_goaway(void* user_data,
                                              iree_net_session_t* session,
                                              uint32_t reason_code,
                                              iree_string_view_t message);

// Called when the session encounters an unrecoverable error. Consumes |status|.
void iree_hal_remote_server_on_session_error(void* user_data,
                                             iree_net_session_t* session,
                                             iree_status_t status);

// Dispatches an incoming control channel frame to the appropriate handler
// (buffer alloc, query heaps, resource release, etc.).
iree_status_t iree_hal_remote_server_on_control_data(
    void* user_data, iree_net_control_frame_flags_t flags,
    iree_const_byte_span_t payload, iree_async_buffer_lease_t* lease);

// Removes a session from the server's tracking. Called when a session reaches
// a terminal state (CLOSED or ERROR). Safe to call multiple times for the
// same session (second call is a no-op).
void iree_hal_remote_server_remove_session(iree_hal_remote_server_t* server,
                                           iree_net_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_SESSION_H_
