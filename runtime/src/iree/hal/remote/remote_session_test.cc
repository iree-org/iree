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

#include <cstring>

#include "iree/async/buffer_pool.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
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
    // Drain proactor to let any pending operations complete.
    PollFor(iree_make_duration_ms(100));

    if (client_device_) {
      iree_hal_device_release(client_device_);
      client_device_ = nullptr;
    }

    // Drain again after client device release (session teardown).
    PollFor(iree_make_duration_ms(100));

    if (server_) {
      iree_hal_remote_server_release(server_);
      server_ = nullptr;
    }

    // Drain after server release.
    PollFor(iree_make_duration_ms(100));

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
  // ERROR. Poll to let the GOAWAY propagate.
  PollFor(iree_make_duration_ms(500));

  iree_hal_remote_client_device_state_t client_state =
      iree_hal_remote_client_device_state(client_device_);
  EXPECT_TRUE(client_state ==
                  IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_DISCONNECTED ||
              client_state == IREE_HAL_REMOTE_CLIENT_DEVICE_STATE_ERROR)
      << "Expected DISCONNECTED or ERROR after server GOAWAY, got "
      << (int)client_state;
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
  PollFor(iree_make_duration_ms(500));

  iree_hal_device_release(client_b);
  PollFor(iree_make_duration_ms(100));
  iree_hal_device_release(client_a);
  PollFor(iree_make_duration_ms(100));
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
  iree_status_t status = iree_hal_device_queue_alloca(
      client_device_, /*queue_affinity=*/0, empty_list, empty_list,
      IREE_HAL_ALLOCATOR_POOL_DEFAULT,
      (iree_hal_buffer_params_t){
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
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

}  // namespace
