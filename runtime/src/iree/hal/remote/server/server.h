// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_SERVER_SERVER_H_
#define IREE_HAL_REMOTE_SERVER_SERVER_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/server/session.h"
#include "iree/net/transport_factory.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Server lifecycle states.
//
// State transitions are monotonic (forward-only):
//   STOPPED → RUNNING → STOPPING → STOPPED
//   Any state → ERROR (terminal)
typedef enum iree_hal_remote_server_state_e {
  IREE_HAL_REMOTE_SERVER_STATE_STOPPED = 0,
  IREE_HAL_REMOTE_SERVER_STATE_RUNNING,
  IREE_HAL_REMOTE_SERVER_STATE_STOPPING,
  IREE_HAL_REMOTE_SERVER_STATE_ERROR,
} iree_hal_remote_server_state_t;

struct iree_hal_remote_server_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;

  // Protects mutable server state accessed from both the application thread
  // (start/stop/query) and the proactor thread (accept/remove/endpoint
  // callbacks). Protected fields: state, sessions[], active_session_count,
  // next_session_id, stopped_callback, listener pointer.
  iree_slim_mutex_t session_mutex;

  // Configuration (bind_address stored in trailing allocation).
  iree_hal_remote_server_options_t options;

  // Wrapped local devices (retained, array of device_count pointers).
  iree_hal_device_t** devices;
  iree_host_size_t device_count;

  // Borrowed infrastructure (must outlive the server).
  iree_async_proactor_t* proactor;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_buffer_pool_t* recv_pool;

  // Listener created during start(), freed during stop().
  iree_net_listener_t* listener;

  // Active sessions (fixed-capacity array, max_connections slots).
  iree_hal_remote_server_session_t* sessions;
  uint32_t active_session_count;

  // Monotonically increasing session ID counter (starts at 1).
  uint64_t next_session_id;

  // Current server state.
  iree_hal_remote_server_state_t state;

  // Callback to fire when stop() completes (all sessions closed, listener
  // freed). Only valid during STOPPING state.
  iree_hal_remote_server_stopped_callback_t stopped_callback;

  // Copied local topology (axes and epochs stored in trailing allocation).
  // Used to populate session options for each accepted connection.
  iree_net_session_topology_t local_topology;

  // Trailing storage layout:
  //   char bind_address_storage[options.bind_address.size]
  //   iree_async_axis_t local_axes[local_topology.axis_count]
  //   uint64_t local_epochs[local_topology.axis_count]
  //   iree_hal_device_t* device_ptrs[device_count]
  //   iree_hal_remote_server_session_t session_slots[max_connections]
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_SERVER_SERVER_H_
