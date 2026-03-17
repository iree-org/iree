// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// End-to-end integration tests for the HAL remote client ↔ server lifecycle.
//
// Tests the full path: loopback transport → session bootstrap → HAL remote
// server accepting connections → HAL remote client device connecting → both
// sides reaching operational/connected state → graceful shutdown.
//
// Uses the loopback carrier factory (in-memory, no network) and the mock HAL
// device (no GPU required). This validates the session-layer integration
// without hardware dependencies.

#include <atomic>
#include <cstring>
#include <thread>

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/operation.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/async/util/proactor_thread.h"
#include "iree/base/api.h"
#include "iree/base/threading/numa.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "iree/hal/remote/client/api.h"
#include "iree/hal/remote/server/api.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/net/carrier/loopback/factory.h"
#include "iree/net/session.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class RemoteSessionTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    // Create proactor.
    iree_async_proactor_options_t proactor_options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        proactor_options, iree_allocator_system(), &proactor_));

    // Create slab/region/recv_pool for buffer management.
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
    IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &recv_pool_));

    // Create frontier trackers for client and server.
    memset(client_axis_entries_, 0, sizeof(client_axis_entries_));
    memset(server_axis_entries_, 0, sizeof(server_axis_entries_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_initialize(
        &client_tracker_, client_axis_entries_, kAxisTableCapacity,
        iree_allocator_system()));
    IREE_ASSERT_OK(iree_async_frontier_tracker_initialize(
        &server_tracker_, server_axis_entries_, kAxisTableCapacity,
        iree_allocator_system()));

    // Create loopback transport factory.
    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));
  }

  void TearDown() override {
    // Drain proactor to complete any accumulated work before teardown.
    DrainProactor();

    if (client_device_) {
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }

    // Drain after client device release. This completes the session's
    // two-phase teardown: session_begin_teardown deactivates the connection's
    // carriers, and the deactivation callbacks fire here on the proactor.
    DrainProactor();

    if (server_) {
      iree_hal_remote_server_release(server_);
      server_ = nullptr;
    }

    // Drain after server release — server-side session teardowns.
    DrainProactor();

    if (mock_device_) {
      iree_hal_device_release(mock_device_);
      mock_device_ = nullptr;
    }
    if (factory_) {
      iree_net_transport_factory_release(factory_);
      factory_ = nullptr;
    }

    iree_async_frontier_tracker_deinitialize(&server_tracker_);
    iree_async_frontier_tracker_deinitialize(&client_tracker_);

    if (recv_pool_) {
      iree_async_buffer_pool_free(recv_pool_);
      recv_pool_ = nullptr;
    }
    if (region_) {
      iree_async_region_release(region_);
      region_ = nullptr;
    }
    if (slab_) {
      iree_async_slab_release(slab_);
      slab_ = nullptr;
    }
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  //===--------------------------------------------------------------------===//
  // Polling helpers
  //===--------------------------------------------------------------------===//

  // Polls the proactor until it has no more completions to process. Handles
  // cascading work (one completion triggering another) by looping until a
  // poll returns zero completions.
  void DrainProactor() {
    for (;;) {
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(0), &completed);
      iree_status_ignore(status);
      if (completed == 0) break;
    }
  }

  // Polls the proactor until |condition| returns true or the time budget
  // expires. Returns true if the condition was met, false on timeout.
  bool PollUntil(std::function<bool()> condition,
                 iree_duration_t budget = iree_make_duration_ms(5000)) {
    iree_time_t deadline = iree_time_now() + budget;
    while (!condition()) {
      if (iree_time_now() >= deadline) return false;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_deadline(deadline), &completed);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
      } else if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        return false;
      }
    }
    return true;
  }

  // Polls the proactor for the given duration (non-conditional drain).
  void PollFor(iree_duration_t duration) {
    PollUntil([&]() { return false; }, duration);
  }

  //===--------------------------------------------------------------------===//
  // Setup helpers
  //===--------------------------------------------------------------------===//

  // Creates a mock HAL device for the server to wrap.
  void CreateMockDevice() {
    iree_hal_mock_device_options_t mock_options;
    iree_hal_mock_device_options_initialize(&mock_options);
    mock_options.identifier = IREE_SV("mock");
    IREE_ASSERT_OK(iree_hal_mock_device_create(
        &mock_options, iree_allocator_system(), &mock_device_));
  }

  // Creates and starts the server with a single-axis topology.
  void CreateAndStartServer() {
    CreateMockDevice();

    // Build server topology: one axis representing the mock device's queue.
    iree_async_axis_t server_axes[] = {0x0200};
    uint64_t server_epochs[] = {0};
    iree_net_session_topology_t server_topology = {};
    server_topology.axes = server_axes;
    server_topology.current_epochs = server_epochs;
    server_topology.axis_count = 1;
    server_topology.machine_index = 1;
    server_topology.session_epoch = 1;

    iree_hal_remote_server_options_t server_options;
    iree_hal_remote_server_options_initialize(&server_options);
    server_options.transport_factory = factory_;
    server_options.bind_address = IREE_SV("test-server");
    server_options.local_topology = &server_topology;
    server_options.max_connections = 4;

    iree_hal_device_t* devices[] = {mock_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, 1, proactor_, &server_tracker_, recv_pool_,
        iree_allocator_system(), &server_));

    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  // Creates the client device configured to connect to the server.
  void CreateClientDevice() {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = factory_;
    client_options.server_address = IREE_SV("test-server");

    // Wire the error callback to track post-connect errors.
    client_options.error_callback.fn = OnClientError;
    client_options.error_callback.user_data = this;

    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, /*create_params=*/nullptr,
        proactor_, &client_tracker_, recv_pool_, iree_allocator_system(),
        &client_device_));
  }

  // Connects the client device and polls until the connect callback fires.
  // Returns the status delivered to the connect callback.
  iree_status_code_t ConnectAndWait() {
    client_connect_fired_ = false;
    client_connect_status_ = IREE_STATUS_OK;

    iree_hal_remote_client_device_connected_callback_t callback;
    callback.fn = OnClientConnected;
    callback.user_data = this;

    iree_status_t connect_status =
        iree_hal_remote_client_device_connect(client_device_, callback);
    if (!iree_status_is_ok(connect_status)) {
      iree_status_code_t code = iree_status_code(connect_status);
      iree_status_ignore(connect_status);
      return code;
    }

    EXPECT_TRUE(PollUntil([&]() { return client_connect_fired_; }))
        << "Client connect callback timed out";
    return client_connect_status_;
  }

  // Stops the server and polls until the stopped callback fires.
  void StopServerAndWait() {
    server_stopped_ = false;
    iree_hal_remote_server_stopped_callback_t callback;
    callback.fn = OnServerStopped;
    callback.user_data = this;
    IREE_ASSERT_OK(iree_hal_remote_server_stop(server_, callback));
    ASSERT_TRUE(PollUntil([&]() { return server_stopped_; }))
        << "Server stop timed out";
  }

  //===--------------------------------------------------------------------===//
  // Callback implementations
  //===--------------------------------------------------------------------===//

  static void OnClientConnected(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteSessionTest*>(user_data);
    self->client_connect_fired_ = true;
    self->client_connect_status_ = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void OnClientError(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteSessionTest*>(user_data);
    self->client_error_fired_ = true;
    self->client_error_status_ = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void OnServerStopped(void* user_data) {
    auto* self = static_cast<RemoteSessionTest*>(user_data);
    self->server_stopped_ = true;
  }

  //===--------------------------------------------------------------------===//
  // Test state
  //===--------------------------------------------------------------------===//

  // Shared infrastructure.
  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;

  // Frontier trackers (separate, as they would be on different machines).
  iree_async_axis_table_entry_t client_axis_entries_[kAxisTableCapacity];
  iree_async_frontier_tracker_t client_tracker_;
  iree_async_axis_table_entry_t server_axis_entries_[kAxisTableCapacity];
  iree_async_frontier_tracker_t server_tracker_;

  // Server side.
  iree_hal_device_t* mock_device_ = nullptr;
  iree_hal_remote_server_t* server_ = nullptr;
  bool server_stopped_ = false;

  // Client side.
  iree_hal_device_t* client_device_ = nullptr;
  bool client_connect_fired_ = false;
  iree_status_code_t client_connect_status_ = IREE_STATUS_OK;
  bool client_error_fired_ = false;
  iree_status_code_t client_error_status_ = IREE_STATUS_OK;
};

//===----------------------------------------------------------------------===//
// Connection lifecycle tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteSessionTest, ConnectSucceeds) {
  CreateAndStartServer();
  CreateClientDevice();

  // Client should start disconnected.
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED);

  // Connect and verify success.
  EXPECT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
}

TEST_F(RemoteSessionTest, ConnectThenGracefulShutdown) {
  CreateAndStartServer();
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  // Stop the server. This sends GOAWAY to the client session.
  StopServerAndWait();

  // The client should receive the GOAWAY and transition to DISCONNECTED or
  // ERROR.
  EXPECT_TRUE(PollUntil([&]() {
    iree_hal_remote_client_device_state_t state =
        iree_hal_remote_client_device_state(client_device_);
    return state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED ||
           state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR;
  })) << "Client did not transition after server GOAWAY";
}

TEST_F(RemoteSessionTest, DoubleConnectFails) {
  CreateAndStartServer();
  CreateClientDevice();
  ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);

  // Second connect should fail with ALREADY_EXISTS.
  iree_hal_remote_client_device_connected_callback_t callback;
  callback.fn = OnClientConnected;
  callback.user_data = this;
  iree_status_t status =
      iree_hal_remote_client_device_connect(client_device_, callback);
  EXPECT_TRUE(iree_status_is_already_exists(status))
      << "Expected ALREADY_EXISTS, got " << iree_status_code(status);
  iree_status_ignore(status);
}

TEST_F(RemoteSessionTest, ConnectBeforeDisconnected) {
  CreateAndStartServer();
  CreateClientDevice();

  // Start a connect (transitions to CONNECTING).
  client_connect_fired_ = false;
  iree_hal_remote_client_device_connected_callback_t callback;
  callback.fn = OnClientConnected;
  callback.user_data = this;
  IREE_ASSERT_OK(
      iree_hal_remote_client_device_connect(client_device_, callback));

  EXPECT_EQ(iree_hal_remote_client_device_state(client_device_),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTING);

  // Attempting another connect while CONNECTING should fail.
  iree_status_t status =
      iree_hal_remote_client_device_connect(client_device_, callback);
  EXPECT_TRUE(iree_status_is_failed_precondition(status))
      << "Expected FAILED_PRECONDITION, got " << iree_status_code(status);
  iree_status_ignore(status);

  // Let the original connect complete.
  ASSERT_TRUE(PollUntil([&]() { return client_connect_fired_; }));
}

TEST_F(RemoteSessionTest, MultipleClientsConnect) {
  CreateAndStartServer();

  // Create and connect two separate client devices.
  iree_hal_device_t* client_a = nullptr;
  iree_hal_device_t* client_b = nullptr;

  iree_hal_remote_client_device_options_t options;
  iree_hal_remote_client_device_options_initialize(&options);
  options.transport_factory = factory_;
  options.server_address = IREE_SV("test-server");

  IREE_ASSERT_OK(iree_hal_remote_client_device_create(
      IREE_SV("remote-a"), &options, /*create_params=*/nullptr, proactor_,
      &client_tracker_, recv_pool_, iree_allocator_system(), &client_a));
  IREE_ASSERT_OK(iree_hal_remote_client_device_create(
      IREE_SV("remote-b"), &options, /*create_params=*/nullptr, proactor_,
      &client_tracker_, recv_pool_, iree_allocator_system(), &client_b));

  // Connect both.
  struct ConnectCtx {
    bool fired = false;
    iree_status_code_t status = IREE_STATUS_OK;
  };
  auto connect_callback = [](void* user_data, iree_status_t status) {
    auto* ctx = static_cast<ConnectCtx*>(user_data);
    ctx->fired = true;
    ctx->status = iree_status_code(status);
    iree_status_ignore(status);
  };
  ConnectCtx ctx_a, ctx_b;

  iree_hal_remote_client_device_connected_callback_t cb_a;
  cb_a.fn = connect_callback;
  cb_a.user_data = &ctx_a;
  IREE_ASSERT_OK(iree_hal_remote_client_device_connect(client_a, cb_a));

  iree_hal_remote_client_device_connected_callback_t cb_b;
  cb_b.fn = connect_callback;
  cb_b.user_data = &ctx_b;
  IREE_ASSERT_OK(iree_hal_remote_client_device_connect(client_b, cb_b));

  ASSERT_TRUE(PollUntil([&]() { return ctx_a.fired && ctx_b.fired; }))
      << "Multi-client connect timed out";

  EXPECT_EQ(ctx_a.status, IREE_STATUS_OK);
  EXPECT_EQ(ctx_b.status, IREE_STATUS_OK);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_a),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);
  EXPECT_EQ(iree_hal_remote_client_device_state(client_b),
            IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_CONNECTED);

  // Clean shutdown.
  StopServerAndWait();
  DrainProactor();

  iree_hal_device_release(client_b);
  DrainProactor();
  iree_hal_device_release(client_a);
  DrainProactor();
}

//===----------------------------------------------------------------------===//
// Device API tests (connected vs disconnected behavior)
//===----------------------------------------------------------------------===//

TEST_F(RemoteSessionTest, QueueOpsFailWhenDisconnected) {
  CreateAndStartServer();
  CreateClientDevice();

  // Queue operations should fail with FAILED_PRECONDITION before connecting.
  iree_hal_semaphore_list_t empty_list = {0, nullptr, nullptr};
  iree_hal_buffer_t* buffer = nullptr;
  iree_hal_buffer_params_t buffer_params = {0};
  buffer_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  iree_status_t status = iree_hal_device_queue_alloca(
      client_device_, /*queue_affinity=*/0, empty_list, empty_list,
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, buffer_params,
      /*allocation_size=*/1024, IREE_HAL_ALLOCA_FLAG_NONE, &buffer);
  EXPECT_TRUE(iree_status_is_failed_precondition(status))
      << "Expected FAILED_PRECONDITION for queue op while disconnected";
  iree_status_ignore(status);
}

TEST_F(RemoteSessionTest, DeviceIdQueryWorksWithoutConnection) {
  CreateAndStartServer();
  CreateClientDevice();

  // Device ID queries should work even without connection.
  int64_t value = -1;
  IREE_ASSERT_OK(iree_hal_device_query_i64(
      client_device_, IREE_SV("hal.device.id"), IREE_SV("remote"), &value));
  EXPECT_EQ(value, 1);

  // Non-matching pattern should return 0.
  IREE_ASSERT_OK(iree_hal_device_query_i64(
      client_device_, IREE_SV("hal.device.id"), IREE_SV("local"), &value));
  EXPECT_EQ(value, 0);
}

//===----------------------------------------------------------------------===//
// Buffer operations fixture
//===----------------------------------------------------------------------===//

// Heavier fixture with a background poll thread and a real local-task device.
// The background thread drives the proactor so the test thread can make
// blocking control_rpc calls (buffer allocation, map, unmap).
class RemoteBufferTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kAxisTableCapacity = 16;

  void SetUp() override {
    // Create proactor for network I/O. Slab registration must happen before
    // the poll thread starts (proactor.h: "must be serialized with poll()").
    iree_async_proactor_options_t proactor_options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        proactor_options, iree_allocator_system(), &proactor_));

    // Create slab/region/recv_pool for buffer management.
    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
    IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &recv_pool_));

    // Start a dedicated poll thread. This frees the test thread to make
    // blocking RPC calls. Must be after slab registration.
    IREE_ASSERT_OK(iree_async_proactor_thread_create(
        proactor_, iree_async_proactor_thread_options_default(),
        iree_allocator_system(), &proactor_thread_));

    // Create frontier trackers.
    memset(client_axis_entries_, 0, sizeof(client_axis_entries_));
    memset(server_axis_entries_, 0, sizeof(server_axis_entries_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_initialize(
        &client_tracker_, client_axis_entries_, kAxisTableCapacity,
        iree_allocator_system()));
    IREE_ASSERT_OK(iree_async_frontier_tracker_initialize(
        &server_tracker_, server_axis_entries_, kAxisTableCapacity,
        iree_allocator_system()));

    // Create loopback transport factory.
    iree_net_loopback_factory_options_t factory_options =
        iree_net_loopback_factory_options_default();
    IREE_ASSERT_OK(iree_net_loopback_factory_create(
        factory_options, iree_allocator_system(), &factory_));

    // Create local-task device (real allocator, real async completion).
    CreateLocalTaskDevice();

    // Create server + client and connect.
    CreateAndStartServer();
    CreateClientDevice();
    ASSERT_EQ(ConnectAndWait(), IREE_STATUS_OK);
  }

  void TearDown() override {
    // Release client device and server ON THE PROACTOR THREAD by sending a
    // message. This is critical: carrier_deactivate and its cascading work
    // (peer disconnect notifications, session error handling) must run on the
    // same thread that processes carrier completions. Releasing from the main
    // thread while the proactor thread is running causes data races on carrier
    // state. The message fires during poll(), which also processes all the
    // cascading completions (disconnect NOPs, deactivation callbacks) in the
    // same or subsequent iterations.
    teardown_phase_.store(0, std::memory_order_relaxed);
    teardown_client_device_ = client_device_;
    client_device_ = nullptr;
    teardown_server_ = server_;
    server_ = nullptr;

    iree_async_proactor_message_callback_t msg_callback;
    msg_callback.fn = OnTeardownMessage;
    msg_callback.user_data = this;
    iree_async_proactor_set_message_callback(proactor_, msg_callback);

    // Phase 1: release client device and server on proactor thread.
    IREE_ASSERT_OK(iree_async_proactor_send_message(proactor_, 1));
    ASSERT_TRUE(WaitUntil([&]() { return teardown_phase_.load() >= 1; }))
        << "Teardown phase 1 timed out";

    // Phase 2: flush cascading work (disconnect notifications, session errors,
    // server-side teardown). The proactor's poll drain loop handles most of
    // this within a single poll() call, but a second message guarantees any
    // work deferred to the next iteration is also processed.
    IREE_ASSERT_OK(iree_async_proactor_send_message(proactor_, 2));
    ASSERT_TRUE(WaitUntil([&]() { return teardown_phase_.load() >= 2; }))
        << "Teardown phase 2 timed out";

    // Stop and join the proactor thread. All session/carrier teardown is done.
    if (proactor_thread_) {
      iree_async_proactor_thread_request_stop(proactor_thread_);
      IREE_ASSERT_OK(iree_async_proactor_thread_join(proactor_thread_,
                                                     IREE_DURATION_INFINITE));
      iree_status_ignore(
          iree_async_proactor_thread_consume_status(proactor_thread_));
      iree_async_proactor_thread_release(proactor_thread_);
      proactor_thread_ = nullptr;
    }

    // All async operations are complete. Release remaining infrastructure.
    if (local_task_device_) {
      iree_hal_device_release(local_task_device_);
      local_task_device_ = nullptr;
    }
    if (local_task_driver_) {
      iree_hal_driver_release(local_task_driver_);
      local_task_driver_ = nullptr;
    }
    if (factory_) {
      iree_net_transport_factory_release(factory_);
      factory_ = nullptr;
    }

    iree_async_frontier_tracker_deinitialize(&server_tracker_);
    iree_async_frontier_tracker_deinitialize(&client_tracker_);

    if (recv_pool_) {
      iree_async_buffer_pool_free(recv_pool_);
      recv_pool_ = nullptr;
    }
    if (region_) {
      iree_async_region_release(region_);
      region_ = nullptr;
    }
    if (slab_) {
      iree_async_slab_release(slab_);
      slab_ = nullptr;
    }
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  bool WaitUntil(std::function<bool()> condition,
                 iree_duration_t budget = iree_make_duration_ms(5000)) {
    iree_time_t deadline = iree_time_now() + budget;
    while (!condition()) {
      if (iree_time_now() >= deadline) return false;
      std::this_thread::yield();
    }
    return true;
  }

  void CreateLocalTaskDevice() {
    iree_status_t status = iree_hal_local_task_driver_module_register(
        iree_hal_driver_registry_default());
    if (iree_status_is_already_exists(status)) {
      iree_status_ignore(status);
      status = iree_ok_status();
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view("local-task"), iree_allocator_system(),
        &local_task_driver_));

    iree_async_proactor_pool_t* proactor_pool = NULL;
    IREE_ASSERT_OK(iree_async_proactor_pool_create(
        iree_numa_node_count(), /*node_ids=*/NULL,
        iree_async_proactor_pool_options_default(), iree_allocator_system(),
        &proactor_pool));

    iree_hal_device_create_params_t create_params =
        iree_hal_device_create_params_default();
    create_params.proactor_pool = proactor_pool;
    IREE_ASSERT_OK(iree_hal_driver_create_default_device(
        local_task_driver_, &create_params, iree_allocator_system(),
        &local_task_device_));

    iree_async_proactor_pool_release(proactor_pool);
  }

  void CreateAndStartServer() {
    iree_async_axis_t server_axes[] = {0x0200};
    uint64_t server_epochs[] = {0};
    iree_net_session_topology_t server_topology = {};
    server_topology.axes = server_axes;
    server_topology.current_epochs = server_epochs;
    server_topology.axis_count = 1;
    server_topology.machine_index = 1;
    server_topology.session_epoch = 1;

    iree_hal_remote_server_options_t server_options;
    iree_hal_remote_server_options_initialize(&server_options);
    server_options.transport_factory = factory_;
    server_options.bind_address = IREE_SV("test-server");
    server_options.local_topology = &server_topology;
    server_options.max_connections = 4;

    iree_hal_device_t* devices[] = {local_task_device_};
    IREE_ASSERT_OK(iree_hal_remote_server_create(
        &server_options, devices, 1, proactor_, &server_tracker_, recv_pool_,
        iree_allocator_system(), &server_));

    IREE_ASSERT_OK(iree_hal_remote_server_start(server_));
  }

  void CreateClientDevice() {
    iree_hal_remote_client_device_options_t client_options;
    iree_hal_remote_client_device_options_initialize(&client_options);
    client_options.transport_factory = factory_;
    client_options.server_address = IREE_SV("test-server");
    client_options.error_callback.fn = OnClientError;
    client_options.error_callback.user_data = this;

    IREE_ASSERT_OK(iree_hal_remote_client_device_create(
        IREE_SV("remote"), &client_options, /*create_params=*/nullptr,
        proactor_, &client_tracker_, recv_pool_, iree_allocator_system(),
        &client_device_));
  }

  iree_status_code_t ConnectAndWait() {
    client_connect_fired_ = false;
    client_connect_status_ = IREE_STATUS_OK;

    iree_hal_remote_client_device_connected_callback_t callback;
    callback.fn = OnClientConnected;
    callback.user_data = this;

    iree_status_t connect_status =
        iree_hal_remote_client_device_connect(client_device_, callback);
    if (!iree_status_is_ok(connect_status)) {
      iree_status_code_t code = iree_status_code(connect_status);
      iree_status_ignore(connect_status);
      return code;
    }

    EXPECT_TRUE(WaitUntil([&]() { return client_connect_fired_.load(); }))
        << "Client connect callback timed out";
    return client_connect_status_.load();
  }

  static void OnClientConnected(void* user_data, iree_status_t status) {
    auto* self = static_cast<RemoteBufferTest*>(user_data);
    self->client_connect_fired_ = true;
    self->client_connect_status_ = iree_status_code(status);
    iree_status_ignore(status);
  }

  static void OnClientError(void* user_data, iree_status_t status) {
    iree_status_ignore(status);
  }

  // Proactor message callback for phased teardown. Runs on the proactor
  // thread so carrier_deactivate and all cascading work execute without
  // races against completion processing.
  static void OnTeardownMessage(iree_async_proactor_t* proactor,
                                uint64_t message_data, void* user_data) {
    (void)proactor;
    auto* self = static_cast<RemoteBufferTest*>(user_data);
    if (message_data == 1) {
      // Phase 1: release client device and server.
      if (self->teardown_client_device_) {
        iree_hal_device_release(self->teardown_client_device_);
        self->teardown_client_device_ = nullptr;
      }
      if (self->teardown_server_) {
        iree_hal_remote_server_release(self->teardown_server_);
        self->teardown_server_ = nullptr;
      }
    }
    // Each message completion advances the phase counter.
    self->teardown_phase_.store(static_cast<int32_t>(message_data),
                                std::memory_order_release);
  }

  // Shared infrastructure.
  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_proactor_thread_t* proactor_thread_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;

  // Frontier trackers.
  iree_async_axis_table_entry_t client_axis_entries_[kAxisTableCapacity];
  iree_async_frontier_tracker_t client_tracker_;
  iree_async_axis_table_entry_t server_axis_entries_[kAxisTableCapacity];
  iree_async_frontier_tracker_t server_tracker_;

  // Server side.
  iree_hal_driver_t* local_task_driver_ = nullptr;
  iree_hal_device_t* local_task_device_ = nullptr;
  iree_hal_remote_server_t* server_ = nullptr;

  // Client side.
  iree_hal_device_t* client_device_ = nullptr;
  std::atomic<bool> client_connect_fired_{false};
  std::atomic<iree_status_code_t> client_connect_status_{IREE_STATUS_OK};

  // Teardown state: objects moved here before message-based release.
  iree_hal_device_t* teardown_client_device_ = nullptr;
  iree_hal_remote_server_t* teardown_server_ = nullptr;
  std::atomic<int32_t> teardown_phase_{0};
};

//===----------------------------------------------------------------------===//
// Buffer allocation and map/unmap tests
//===----------------------------------------------------------------------===//

TEST_F(RemoteBufferTest, AllocateAndDeallocate) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/256, &buffer));
  ASSERT_NE(buffer, nullptr);

  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 256);

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, WriteDiscardThenReadBack) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/64, &buffer));

  // Map WRITE|DISCARD: fill with a known pattern.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/0,
                                           /*byte_length=*/64, &mapping));
  ASSERT_NE(mapping.contents.data, nullptr);
  ASSERT_EQ(mapping.contents.data_length, 64);

  // Fill with 0xAB pattern.
  memset(mapping.contents.data, 0xAB, 64);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Map READ: verify the pattern persisted through the round-trip.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/64, &mapping));
  ASSERT_NE(mapping.contents.data, nullptr);

  // Every byte should be 0xAB.
  const uint8_t* data = mapping.contents.data;
  for (iree_host_size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(data[i], 0xAB) << "Mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, PartialMapRange) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/256, &buffer));

  // Write 0xCC to the first 128 bytes.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/0,
                                           /*byte_length=*/128, &mapping));
  memset(mapping.contents.data, 0xCC, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Write 0xDD to bytes 128-255.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/128,
                                           /*byte_length=*/128, &mapping));
  memset(mapping.contents.data, 0xDD, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Read the full buffer and verify both halves.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/256, &mapping));

  const uint8_t* data = mapping.contents.data;
  for (iree_host_size_t i = 0; i < 128; ++i) {
    EXPECT_EQ(data[i], 0xCC) << "First half mismatch at byte " << i;
  }
  for (iree_host_size_t i = 128; i < 256; ++i) {
    EXPECT_EQ(data[i], 0xDD) << "Second half mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, ReadWriteModifiesInPlace) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/16, &buffer));

  // Write initial data.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE,
                                           /*byte_offset=*/0,
                                           /*byte_length=*/16, &mapping));
  for (iree_host_size_t i = 0; i < 16; ++i) {
    mapping.contents.data[i] = (uint8_t)i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Map READ|WRITE: read current data, modify, write back.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      /*byte_offset=*/0, /*byte_length=*/16, &mapping));

  // Verify initial data was pulled.
  for (iree_host_size_t i = 0; i < 16; ++i) {
    ASSERT_EQ(mapping.contents.data[i], (uint8_t)i)
        << "READ|WRITE initial data mismatch at byte " << i;
  }

  // Increment each byte.
  for (iree_host_size_t i = 0; i < 16; ++i) {
    mapping.contents.data[i] += 100;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  // Read back and verify modification.
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/16, &mapping));
  for (iree_host_size_t i = 0; i < 16; ++i) {
    EXPECT_EQ(mapping.contents.data[i], (uint8_t)(i + 100))
        << "Modified data mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_buffer_release(buffer);
}

//===----------------------------------------------------------------------===//
// Queue fill, copy, and update tests
//===----------------------------------------------------------------------===//

// Helper to build a semaphore list with a single semaphore and value.
// Named storage avoids C++ compound literal lifetime issues.
struct SemaphoreListHelper {
  iree_hal_semaphore_t* semaphore;
  uint64_t value;
  iree_hal_semaphore_list_t list;
  SemaphoreListHelper(iree_hal_semaphore_t* sem, uint64_t val)
      : semaphore(sem), value(val) {
    list.count = 1;
    list.semaphores = &semaphore;
    list.payload_values = &value;
  }
};

TEST_F(RemoteBufferTest, QueueFillAndReadBack) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator, params, /*allocation_size=*/1024, &buffer));

  // Create a semaphore to track fill completion.
  iree_hal_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_NONE, &sem));

  // Queue fill with 4-byte pattern.
  uint32_t pattern = 0xDEADBEEF;
  SemaphoreListHelper signal(sem, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, buffer,
      /*target_offset=*/0, /*length=*/1024, &pattern,
      /*pattern_length=*/sizeof(pattern), IREE_HAL_FILL_FLAG_NONE));

  // Wait for fill to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(sem, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back and verify.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      /*byte_offset=*/0, /*byte_length=*/1024, &mapping));
  const uint32_t* data = (const uint32_t*)mapping.contents.data;
  for (iree_host_size_t i = 0; i < 256; ++i) {
    ASSERT_EQ(data[i], 0xDEADBEEF) << "Fill pattern mismatch at uint32 " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(sem);
  iree_hal_buffer_release(buffer);
}

TEST_F(RemoteBufferTest, QueueCopyChained) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buf_a = nullptr;
  iree_hal_buffer_t* buf_b = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 1024, &buf_a));
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 1024, &buf_b));

  // Create two semaphores for chaining: fill → copy.
  iree_hal_semaphore_t* sem_a = nullptr;
  iree_hal_semaphore_t* sem_b = nullptr;
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_a));
  IREE_ASSERT_OK(
      iree_hal_semaphore_create(client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                IREE_HAL_SEMAPHORE_FLAG_NONE, &sem_b));

  // Fill buf_a with 0xAA, signal sem_a.
  uint8_t fill_pattern = 0xAA;
  SemaphoreListHelper fill_signal(sem_a, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), fill_signal.list, buf_a, 0, 1024,
      &fill_pattern, sizeof(fill_pattern), IREE_HAL_FILL_FLAG_NONE));

  // Copy buf_a → buf_b, wait on sem_a, signal sem_b.
  // This is the critical frontier ordering test: the copy depends on the
  // fill via sem_a. If the wait frontier doesn't correctly chain them
  // through local semaphores on the server, the copy reads stale data.
  SemaphoreListHelper copy_wait(sem_a, 1);
  SemaphoreListHelper copy_signal(sem_b, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY, copy_wait.list,
      copy_signal.list, buf_a, 0, buf_b, 0, 1024, IREE_HAL_COPY_FLAG_NONE));

  // Wait for copy to complete.
  IREE_ASSERT_OK(iree_hal_semaphore_wait(sem_b, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read buf_b and verify it has the fill pattern from buf_a.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buf_b, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0, 1024,
                                           &mapping));
  for (iree_host_size_t i = 0; i < 1024; ++i) {
    ASSERT_EQ(mapping.contents.data[i], 0xAA)
        << "Copy data mismatch at byte " << i;
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(sem_b);
  iree_hal_semaphore_release(sem_a);
  iree_hal_buffer_release(buf_b);
  iree_hal_buffer_release(buf_a);
}

TEST_F(RemoteBufferTest, QueueUpdateAndReadBack) {
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(client_device_);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;

  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      iree_hal_allocator_allocate_buffer(allocator, params, 64, &buffer));

  iree_hal_semaphore_t* sem = nullptr;
  IREE_ASSERT_OK(iree_hal_semaphore_create(client_device_,
                                           IREE_HAL_QUEUE_AFFINITY_ANY, 0,
                                           IREE_HAL_SEMAPHORE_FLAG_NONE, &sem));

  // Queue update with inline host data.
  const char host_data[] = "Hello, remote HAL!";
  iree_host_size_t data_length = sizeof(host_data) - 1;  // exclude NUL
  SemaphoreListHelper signal(sem, 1);
  IREE_ASSERT_OK(iree_hal_device_queue_update(
      client_device_, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), signal.list, host_data,
      /*source_offset=*/0, buffer, /*target_offset=*/0, data_length,
      IREE_HAL_UPDATE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_wait(sem, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Read back and verify.
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0,
                                           data_length, &mapping));
  ASSERT_EQ(memcmp(mapping.contents.data, host_data, data_length), 0)
      << "Update data mismatch";
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_semaphore_release(sem);
  iree_hal_buffer_release(buffer);
}

}  // namespace
