// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_DEVICE_H_
#define IREE_HAL_REMOTE_CLIENT_DEVICE_H_

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/protocol/common.h"
#include "iree/net/session.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;
typedef struct iree_async_frontier_tracker_t iree_async_frontier_tracker_t;
typedef struct iree_net_queue_channel_t iree_net_queue_channel_t;
typedef struct iree_hal_remote_pending_rpc_t iree_hal_remote_pending_rpc_t;

typedef struct iree_hal_remote_client_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  // Optional provider used for creating/configuring collective channels.
  iree_hal_channel_provider_t* channel_provider;

  // Device configuration options.
  iree_hal_remote_client_device_options_t options;

  // Async infrastructure. Either borrowed (from CTS/test harness) or owned
  // (when created via the driver from a proactor_pool). The owned_* fields
  // are non-NULL only when the device created its own infrastructure.
  iree_async_proactor_t* proactor;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_buffer_pool_t* recv_pool;
  iree_async_slab_t* owned_slab;
  iree_async_region_t* owned_region;
  iree_async_buffer_pool_t* owned_recv_pool;
  iree_async_frontier_tracker_t owned_tracker;
  iree_async_axis_table_entry_t owned_axis_entries[16];
  bool owns_infra;

  // Active session (NULL when disconnected).
  iree_net_session_t* session;

  // Queue channel for HAL command dispatch (0 until queue endpoint opens).
  // The channel owns the header pool for its frame_sender (freed on channel
  // destroy).
  //
  // Accessed lock-free using the Dekker pattern: the hot path (queue_execute)
  // increments channel_users then reads queue_channel; the teardown path
  // (goaway/error) zeroes queue_channel then drains channel_users. The seq_cst
  // total order guarantees at least one side sees the other's update: either
  // the hot path sees the zeroed channel and bails, or teardown sees in-flight
  // users and waits for them to finish.
  iree_atomic_intptr_t queue_channel;

  // Count of in-flight queue_execute calls currently using queue_channel.
  // Incremented before reading queue_channel, decremented after the send
  // completes. Teardown spins until this reaches zero before releasing the
  // channel.
  iree_atomic_int32_t channel_users;

  // Remote queue axis from the server's topology. Used to build signal
  // frontiers for queue submissions. Valid after on_session_ready.
  iree_async_axis_t remote_queue_axis;

  // Monotonically increasing epoch counter for signal frontiers.
  // Each queue submission assigns the next epoch on remote_queue_axis.
  // Atomic because immediate sends (app thread) and deferred sends
  // (proactor thread, via gate timepoint callbacks) can race.
  iree_atomic_int64_t next_submission_epoch;

  // Monotonically increasing generation counter for provisional resource IDs.
  // Each queue_alloca assigns the next generation to create a unique
  // provisional_id. Atomic for the same reason as next_submission_epoch
  // (deferred callbacks on proactor thread can race with app thread).
  iree_atomic_int32_t next_provisional_generation;

  // Provisional buffer tracking. Buffers created by queue_alloca have
  // provisional resource_ids until the server resolves them via ADVANCE
  // resolution entries. This table maps provisional_id → buffer proxy so
  // on_advance can update the proxy's resource_id when the resolution
  // arrives. Protected by provisional_mutex.
  iree_slim_mutex_t provisional_mutex;
  struct {
    iree_hal_remote_resource_id_t* provisional_ids;
    iree_hal_buffer_t** buffers;  // retained
    iree_host_size_t count;
    iree_host_size_t capacity;
  } provisional_buffers;

  // Monotonically increasing request ID for control channel RPCs.
  iree_atomic_int32_t next_request_id;

  // Pending control RPCs (stack-allocated entries linked during blocking
  // calls). Protected by rpc_mutex.
  iree_slim_mutex_t rpc_mutex;
  iree_hal_remote_pending_rpc_t* pending_rpcs;

  // Current connection state. Atomic because it's written by the proactor
  // thread (session callbacks) and read by the app thread (queue operations).
  // The release-store on state transitions publishes all prior field writes
  // (session, remote_queue_axis, next_submission_epoch); the acquire-load on
  // reads ensures those fields are visible.
  iree_atomic_int32_t state;

  // Pending connect callback (valid during CONNECTING/CONNECTED state until
  // on_queue_endpoint_ready fires).
  iree_hal_remote_client_device_connected_callback_t connect_callback;

  // Trailing storage layout:
  //   char identifier_storage[identifier.size]
  //   char server_address_storage[options.server_address.size]
} iree_hal_remote_client_device_t;

// Loads the device state with acquire ordering. Pairs with the release-store
// in state transitions to ensure field writes are visible to readers.
static inline iree_hal_remote_client_device_state_t
iree_hal_remote_client_device_load_state(
    iree_hal_remote_client_device_t* device) {
  return (iree_hal_remote_client_device_state_t)iree_atomic_load(
      &device->state, iree_memory_order_acquire);
}

// Stores the device state with release ordering. Must be called AFTER all
// associated field writes (session, remote_queue_axis, etc.) so that readers
// who acquire-load the new state see the prior writes.
static inline void iree_hal_remote_client_device_store_state(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_client_device_state_t new_state) {
  iree_atomic_store(&device->state, (int32_t)new_state,
                    iree_memory_order_release);
}

iree_hal_remote_client_device_t* iree_hal_remote_client_device_cast(
    iree_hal_device_t* base_value);

// All queue operations require the device to be connected.
#define IREE_HAL_REMOTE_REQUIRE_CONNECTED(device)            \
  if (iree_hal_remote_client_device_load_state(device) !=    \
      IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED) {       \
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION, \
                            "device is not connected");      \
  }

// Sends a control channel request and blocks until the response arrives.
// |request| is [envelope + body] built by the caller. The envelope's
// request_id field is patched by this function.
// On success, |out_response_payload| points into the retained lease and
// |out_response_lease| holds the backing buffer. The caller must release
// the lease after processing the response.
iree_status_t iree_hal_remote_client_device_control_rpc(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t request,
    iree_const_byte_span_t* out_response_payload,
    iree_async_buffer_lease_t* out_response_lease);

// Sends a fire-and-forget control message (no response expected).
iree_status_t iree_hal_remote_client_device_send_fire_and_forget(
    iree_hal_remote_client_device_t* device, iree_const_byte_span_t message);

// Returns the device's active session (for sending control messages).
iree_net_session_t* iree_hal_remote_client_device_session(
    iree_hal_remote_client_device_t* device);

// Registers a buffer with a provisional resource_id. The buffer is retained
// until resolve_provisional removes it (or the device is destroyed).
iree_status_t iree_hal_remote_client_device_register_provisional(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t provisional_id, iree_hal_buffer_t* buffer);

// Looks up and removes a provisional buffer by its provisional_id. Returns
// the buffer (borrowed — the caller's reference from queue_alloca keeps it
// alive) and releases the provisional tracking reference. Returns NULL if
// the provisional_id is not found.
iree_hal_buffer_t* iree_hal_remote_client_device_resolve_provisional(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t provisional_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_DEVICE_H_
