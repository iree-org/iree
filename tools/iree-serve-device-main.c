// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// iree-serve-device: Exposes local HAL devices to remote clients.
//
// Usage:
//   iree-serve-device --device=local-task --bind=tcp://0.0.0.0:5000
//   iree-serve-device --device=hip://0 --bind=tcp://[::]:5000
//
// Clients connect using the remote HAL driver:
//   iree-run-module --device=remote-tcp://server:5000 --module=model.vmfb

#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/remote/util/recv_pool.h"
#include "iree/net/carrier/shm/factory.h"
#include "iree/net/carrier/tcp/factory.h"
#include "iree/net/session.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(string, bind, "tcp://0.0.0.0:5000",
          "Address to bind the server to.\n"
          "Transport prefixes:\n"
          "  tcp://host:port       TCP sockets (default)\n"
          "  shm:///path           Shared memory (local IPC)");

IREE_FLAG(int32_t, max_connections, 16,
          "Maximum number of concurrent client connections.");

IREE_FLAG(bool, rdma, false, "Enable RDMA for bulk transfers when available.");

typedef struct iree_serve_device_state_t {
  iree_allocator_t host_allocator;
  iree_hal_device_t* device;
  iree_async_proactor_pool_t* device_proactor_pool;
  iree_async_proactor_pool_t* server_proactor_pool;
  iree_hal_remote_recv_pool_t* recv_pool;
  iree_async_axis_table_entry_t axis_entries[16];
  iree_async_frontier_tracker_t tracker;
  iree_net_transport_factory_t* factory;
  iree_hal_remote_server_t* server;
  iree_async_signal_subscription_t* interrupt_subscription;
  iree_async_signal_subscription_t* terminate_subscription;
  iree_notification_t shutdown_notification;
  iree_atomic_int32_t shutdown_requested;
} iree_serve_device_state_t;

static iree_status_t iree_serve_device_parse_bind_uri(
    iree_string_view_t bind_uri, iree_string_view_t* out_transport,
    iree_string_view_t* out_address) {
  iree_string_view_t remainder = bind_uri;
  if (iree_string_view_consume_prefix(&remainder, IREE_SV("tcp://"))) {
    *out_transport = IREE_SV("tcp");
    *out_address = remainder;
    return iree_ok_status();
  }
  if (iree_string_view_consume_prefix(&remainder, IREE_SV("shm://"))) {
    *out_transport = IREE_SV("shm");
    *out_address = remainder;
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "bind URI must have a transport prefix (tcp://, shm://), got: '%.*s'",
      (int)bind_uri.size, bind_uri.data);
}

static iree_status_t iree_serve_device_create_transport(
    iree_string_view_t transport_name, iree_allocator_t host_allocator,
    iree_net_transport_factory_t** out_factory) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (iree_string_view_equal(transport_name, IREE_SV("tcp"))) {
    iree_net_tcp_carrier_options_t tcp_options =
        iree_net_tcp_carrier_options_default();
    // HAL remote uses multiple endpoints per connection (control channel +
    // queue channels).
    tcp_options.max_endpoint_count = 4;
    iree_status_t status =
        iree_net_tcp_factory_create(tcp_options, host_allocator, out_factory);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  if (iree_string_view_equal(transport_name, IREE_SV("shm"))) {
    iree_status_t status = iree_net_shm_factory_create(
        iree_net_shm_carrier_options_default(), host_allocator, out_factory);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }
  IREE_TRACE_ZONE_END(z0);
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported transport: %.*s",
                          (int)transport_name.size, transport_name.data);
}

static void iree_serve_device_on_signal(void* user_data,
                                        iree_async_signal_t signal) {
  iree_serve_device_state_t* state = (iree_serve_device_state_t*)user_data;
  fprintf(stdout, "\nReceived %.*s, shutting down...\n",
          (int)iree_async_signal_name(signal).size,
          iree_async_signal_name(signal).data);
  fflush(stdout);
  iree_atomic_store(&state->shutdown_requested, 1, iree_memory_order_release);
  iree_notification_post(&state->shutdown_notification, IREE_ALL_WAITERS);
}

static bool iree_serve_device_is_shutdown_requested(void* user_data) {
  iree_serve_device_state_t* state = (iree_serve_device_state_t*)user_data;
  return iree_atomic_load(&state->shutdown_requested,
                          iree_memory_order_acquire) != 0;
}

// Creates the server, subscribes to signals, and blocks until SIGINT/SIGTERM.
static iree_status_t iree_serve_device_create_and_run_server(
    iree_serve_device_state_t* state, iree_string_view_t bind_address) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_t* proactor =
      iree_hal_remote_recv_pool_proactor(state->recv_pool);

  iree_async_axis_t server_axes[] = {0x0200};
  uint64_t server_epochs[] = {0};
  iree_net_session_topology_t server_topology = {
      .axes = server_axes,
      .current_epochs = server_epochs,
      .axis_count = 1,
      .machine_index = 1,
      .session_epoch = 1,
  };

  iree_hal_remote_server_options_t options;
  iree_hal_remote_server_options_initialize(&options);
  options.transport_factory = state->factory;
  options.bind_address = bind_address;
  options.local_topology = &server_topology;
  options.max_connections = (uint32_t)FLAG_max_connections;
  if (FLAG_rdma) {
    options.flags |= IREE_HAL_REMOTE_SERVER_FLAG_ENABLE_RDMA;
  }

  iree_hal_device_t* devices[] = {state->device};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_server_create(
              &options, devices, 1, proactor, &state->tracker,
              iree_hal_remote_recv_pool_buffer_pool(state->recv_pool),
              state->host_allocator, &state->server));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_remote_server_start(state->server));

  // Subscribe to signals for graceful shutdown. The proactor thread delivers
  // signal callbacks — the main thread blocks on a notification.
  iree_async_signal_callback_t signal_callback = {
      .fn = iree_serve_device_on_signal,
      .user_data = state,
  };
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_subscribe_signal(
              proactor, IREE_ASYNC_SIGNAL_INTERRUPT, signal_callback,
              &state->interrupt_subscription));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_subscribe_signal(
              proactor, IREE_ASYNC_SIGNAL_TERMINATE, signal_callback,
              &state->terminate_subscription));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_async_signal_block_default());
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_async_signal_ignore_broken_pipe());

  // Block until shutdown. The proactor thread drives all async I/O.
  iree_notification_await(&state->shutdown_notification,
                          iree_serve_device_is_shutdown_requested, state,
                          iree_infinite_timeout());

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_serve_device_teardown(
    iree_serve_device_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_t* proactor =
      state->recv_pool ? iree_hal_remote_recv_pool_proactor(state->recv_pool)
                       : NULL;
  if (proactor) {
    iree_async_proactor_unsubscribe_signal(proactor,
                                           state->interrupt_subscription);
    iree_async_proactor_unsubscribe_signal(proactor,
                                           state->terminate_subscription);
  }

  iree_hal_remote_server_release(state->server);
  iree_notification_deinitialize(&state->shutdown_notification);
  iree_async_frontier_tracker_deinitialize(&state->tracker);
  iree_net_transport_factory_release(state->factory);
  iree_hal_remote_recv_pool_release(state->recv_pool);
  iree_async_proactor_pool_release(state->server_proactor_pool);
  iree_hal_device_release(state->device);
  iree_async_proactor_pool_release(state->device_proactor_pool);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_serve_device_run(void) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_serve_device_state_t state;
  memset(&state, 0, sizeof(state));
  state.host_allocator = iree_allocator_system();
  iree_notification_initialize(&state.shutdown_notification);
  iree_atomic_store(&state.shutdown_requested, 0, iree_memory_order_relaxed);

  // Create a proactor pool for the local device. Kept alive for the server's
  // lifetime — the device needs its proactor thread for async queue operations.
  iree_status_t status = iree_async_proactor_pool_create(
      iree_numa_node_count(), /*node_ids=*/NULL,
      iree_async_proactor_pool_options_default(), state.host_allocator,
      &state.device_proactor_pool);

  // Create the local device to serve (uses --device flag).
  if (iree_status_is_ok(status)) {
    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = state.device_proactor_pool;
    status = iree_hal_create_device_from_flags(
        iree_hal_available_driver_registry(),
        /*default_device=*/iree_string_view_empty(), &create_params,
        state.host_allocator, &state.device);
  }

  iree_string_view_t transport_name = iree_string_view_empty();
  iree_string_view_t bind_address = iree_string_view_empty();
  if (iree_status_is_ok(status)) {
    iree_string_view_t device_id = iree_hal_device_id(state.device);
    fprintf(stdout, "Device: %.*s\n", (int)device_id.size, device_id.data);
    status = iree_serve_device_parse_bind_uri(iree_make_cstring_view(FLAG_bind),
                                              &transport_name, &bind_address);
  }

  // Create a proactor pool for the server's networking.
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_create(
        1, /*node_ids=*/NULL, iree_async_proactor_pool_options_default(),
        state.host_allocator, &state.server_proactor_pool);
  }

  // Create the recv_pool from the server proactor pool.
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_recv_pool_create(
        state.server_proactor_pool, IREE_ASYNC_AFFINITY_NUMA_NODE_ANY,
        state.host_allocator, &state.recv_pool);
  }

  // Create the frontier tracker.
  if (iree_status_is_ok(status)) {
    status = iree_async_frontier_tracker_initialize(
        &state.tracker, state.axis_entries, 16, state.host_allocator);
  }

  if (iree_status_is_ok(status)) {
    status = iree_serve_device_create_transport(
        transport_name, state.host_allocator, &state.factory);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Serving on %.*s://%.*s (Ctrl+C to stop)\n",
            (int)transport_name.size, transport_name.data,
            (int)bind_address.size, bind_address.data);
    fflush(stdout);
    status = iree_serve_device_create_and_run_server(&state, bind_address);
  }

  iree_status_t teardown_status = iree_serve_device_teardown(&state);
  if (iree_status_is_ok(status)) {
    status = teardown_status;
  } else {
    iree_status_free(teardown_status);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_flags_set_usage(
      "iree-serve-device",
      "Exposes local HAL devices to remote clients over the network.\n"
      "\n"
      "Examples:\n"
      "  # Serve a local-task device on port 5000 over TCP\n"
      "  iree-serve-device --device=local-task --bind=tcp://0.0.0.0:5000\n"
      "\n"
      "  # Serve a HIP GPU over TCP\n"
      "  iree-serve-device --device=hip://0 --bind=tcp://0.0.0.0:5000\n"
      "\n"
      "  # Serve over shared memory (local IPC)\n"
      "  iree-serve-device --device=hip://0 --bind=shm:///dev/shm/iree-gpu\n"
      "\n"
      "  # Connect from another machine\n"
      "  iree-run-module --device=remote-tcp://server:5000 "
      "--module=model.vmfb\n"
      "\n"
      "  # Connect via shared memory\n"
      "  iree-run-module --device=remote-shm:///dev/shm/iree-gpu "
      "--module=model.vmfb\n");
  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_status_t status = iree_serve_device_run();

  int exit_code = EXIT_SUCCESS;
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_free(status);
    exit_code = EXIT_FAILURE;
  }

  IREE_TRACE_ZONE_END(z0);
  IREE_TRACE_APP_EXIT(exit_code);
  return exit_code;
}
