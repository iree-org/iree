// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS session tests: bootstrap, topology exchange, control data, shutdown.
//
// These tests exercise the session lifecycle across all transport backends
// that provide factory support. The session manages connection bootstrap
// (HELLO/HELLO_ACK), proxy semaphore creation, and control channel forwarding
// — all of which must work identically regardless of the underlying transport.
//
// Registered with the "factory" tag — only instantiated for backends that
// provide factory-level fields in their BackendInfo.

#include <cstring>
#include <string>
#include <vector>

#include "iree/async/semaphore.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/session_test_base.h"

namespace iree::net::carrier::cts {
namespace {

using SessionTest = SessionTestBase;

//===----------------------------------------------------------------------===//
// Bootstrap
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, BootstrapSucceeds) {
  EstablishDefaultSessionPair();

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
}

TEST_P(SessionTest, BootstrapWithZeroAxes) {
  iree_net_session_topology_t empty_topo = {};
  EstablishSessionPair(empty_topo, empty_topo);

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);

  EXPECT_EQ(client_callbacks_.remote_axis_count, 0u);
  EXPECT_EQ(server_callbacks_.remote_axis_count, 0u);
}

TEST_P(SessionTest, TopologyExchange) {
  // Client: 2 axes with non-zero epochs.
  iree_async_axis_t client_axes[] = {0x0100, 0x0101};
  uint64_t client_epochs[] = {10, 20};
  iree_net_session_topology_t client_topo = {};
  client_topo.axes = client_axes;
  client_topo.current_epochs = client_epochs;
  client_topo.axis_count = 2;
  client_topo.machine_index = 5;
  client_topo.session_epoch = 3;

  // Server: 3 axes with non-zero epochs.
  iree_async_axis_t server_axes[] = {0x0200, 0x0201, 0x0202};
  uint64_t server_epochs[] = {100, 200, 300};
  iree_net_session_topology_t server_topo = {};
  server_topo.axes = server_axes;
  server_topo.current_epochs = server_epochs;
  server_topo.axis_count = 3;
  server_topo.machine_index = 7;
  server_topo.session_epoch = 4;

  EstablishSessionPair(client_topo, server_topo);

  // Client should see server's topology (3 axes).
  ASSERT_EQ(client_callbacks_.remote_axis_count, 3u);
  EXPECT_EQ(client_callbacks_.remote_axes[0], 0x0200u);
  EXPECT_EQ(client_callbacks_.remote_axes[1], 0x0201u);
  EXPECT_EQ(client_callbacks_.remote_axes[2], 0x0202u);
  EXPECT_EQ(client_callbacks_.remote_epochs[0], 100u);
  EXPECT_EQ(client_callbacks_.remote_epochs[1], 200u);
  EXPECT_EQ(client_callbacks_.remote_epochs[2], 300u);
  EXPECT_EQ(client_callbacks_.remote_machine_index, 7);
  EXPECT_EQ(client_callbacks_.remote_session_epoch, 4);

  // Server should see client's topology (2 axes).
  ASSERT_EQ(server_callbacks_.remote_axis_count, 2u);
  EXPECT_EQ(server_callbacks_.remote_axes[0], 0x0100u);
  EXPECT_EQ(server_callbacks_.remote_axes[1], 0x0101u);
  EXPECT_EQ(server_callbacks_.remote_epochs[0], 10u);
  EXPECT_EQ(server_callbacks_.remote_epochs[1], 20u);
  EXPECT_EQ(server_callbacks_.remote_machine_index, 5);
  EXPECT_EQ(server_callbacks_.remote_session_epoch, 3);
}

TEST_P(SessionTest, SessionIdAssignment) {
  EstablishSessionPair(iree_net_session_topology_t{},
                       iree_net_session_topology_t{},
                       /*server_session_id=*/99);

  // Both sides should agree on the server-assigned session ID.
  EXPECT_EQ(iree_net_session_id(client_session_), 99u);
  EXPECT_EQ(iree_net_session_id(server_session_), 99u);
}

//===----------------------------------------------------------------------===//
// Control data forwarding
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, ControlDataClientToServer) {
  EstablishDefaultSessionPair();

  const char* message = "hello server";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(
      iree_net_session_send_control_data(client_session_, 0, span_list, 0));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.control_data_fired; }))
      << "Server never received control data";

  EXPECT_EQ(std::string(server_callbacks_.control_data.begin(),
                        server_callbacks_.control_data.end()),
            "hello server");
}

TEST_P(SessionTest, ControlDataServerToClient) {
  EstablishDefaultSessionPair();

  const char* message = "hello client";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)message, strlen(message));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(
      iree_net_session_send_control_data(server_session_, 0, span_list, 0));

  ASSERT_TRUE(PollUntil([&]() { return client_callbacks_.control_data_fired; }))
      << "Client never received control data";

  EXPECT_EQ(std::string(client_callbacks_.control_data.begin(),
                        client_callbacks_.control_data.end()),
            "hello client");
}

TEST_P(SessionTest, ControlDataBidirectional) {
  EstablishDefaultSessionPair();

  // Send in both directions simultaneously.
  const char* to_server = "ping";
  const char* to_client = "pong";
  iree_async_span_t span_to_server =
      iree_async_span_from_ptr((void*)to_server, strlen(to_server));
  iree_async_span_list_t list_to_server =
      iree_async_span_list_make(&span_to_server, 1);
  IREE_ASSERT_OK(iree_net_session_send_control_data(client_session_, 0,
                                                    list_to_server, 0));

  iree_async_span_t span_to_client =
      iree_async_span_from_ptr((void*)to_client, strlen(to_client));
  iree_async_span_list_t list_to_client =
      iree_async_span_list_make(&span_to_client, 1);
  IREE_ASSERT_OK(iree_net_session_send_control_data(server_session_, 0,
                                                    list_to_client, 0));

  ASSERT_TRUE(PollUntil([&]() {
    return server_callbacks_.control_data_fired &&
           client_callbacks_.control_data_fired;
  })) << "Bidirectional control data timed out";

  EXPECT_EQ(std::string(server_callbacks_.control_data.begin(),
                        server_callbacks_.control_data.end()),
            "ping");
  EXPECT_EQ(std::string(client_callbacks_.control_data.begin(),
                        client_callbacks_.control_data.end()),
            "pong");
}

//===----------------------------------------------------------------------===//
// Graceful shutdown
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, GracefulShutdownFromClient) {
  EstablishDefaultSessionPair();

  IREE_ASSERT_OK(iree_net_session_shutdown(client_session_, 0,
                                           iree_make_cstring_view("bye")));

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_DRAINING);

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.goaway_fired; }))
      << "Server never received GOAWAY";

  EXPECT_EQ(server_callbacks_.goaway_reason_code, 0u);
  EXPECT_EQ(server_callbacks_.goaway_message, "bye");
}

TEST_P(SessionTest, GracefulShutdownFromServer) {
  EstablishDefaultSessionPair();

  IREE_ASSERT_OK(iree_net_session_shutdown(server_session_, 42,
                                           iree_make_cstring_view("done")));

  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_DRAINING);

  ASSERT_TRUE(PollUntil([&]() { return client_callbacks_.goaway_fired; }))
      << "Client never received GOAWAY";

  EXPECT_EQ(client_callbacks_.goaway_reason_code, 42u);
  EXPECT_EQ(client_callbacks_.goaway_message, "done");
}

TEST_P(SessionTest, OperationsFailAfterShutdown) {
  EstablishDefaultSessionPair();

  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  // send_control_data should fail in DRAINING state.
  const char* data = "nope";
  iree_async_span_t span = iree_async_span_from_ptr((void*)data, strlen(data));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_send_control_data(client_session_, 0, span_list, 0));

  // A second shutdown should also fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_shutdown(client_session_, 0, IREE_SV("again")));
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, ServerSessionRequiresNonzeroId) {
  // session_accept must reject session_id=0.
  auto pair = EstablishConnection();
  ASSERT_NE(pair.server, nullptr);

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 0;

  SessionCallbackTracker callbacks;
  iree_net_session_t* session = nullptr;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_session_accept(pair.server, &server_tracker_, &options,
                              callbacks.MakeCallbacks(),
                              iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(SessionTest, OnReadyCallbackRequired) {
  auto pair = EstablishConnection();
  ASSERT_NE(pair.server, nullptr);

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 1;

  // Missing on_ready (on_control_data provided).
  iree_net_session_callbacks_t bad_callbacks;
  memset(&bad_callbacks, 0, sizeof(bad_callbacks));
  bad_callbacks.on_control_data = SessionCallbackTracker::OnControlData;

  iree_net_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_session_accept(
                            pair.server, &server_tracker_, &options,
                            bad_callbacks, iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(SessionTest, OnControlDataCallbackRequired) {
  auto pair = EstablishConnection();
  ASSERT_NE(pair.server, nullptr);

  iree_net_session_options_t options = iree_net_session_options_default();
  options.session_id = 1;

  // Missing on_control_data (on_ready provided).
  iree_net_session_callbacks_t bad_callbacks;
  memset(&bad_callbacks, 0, sizeof(bad_callbacks));
  bad_callbacks.on_ready = SessionCallbackTracker::OnReady;

  iree_net_session_t* session = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_session_accept(
                            pair.server, &server_tracker_, &options,
                            bad_callbacks, iree_allocator_system(), &session));
  EXPECT_EQ(session, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

//===----------------------------------------------------------------------===//
// Proxy semaphore signaling
//===----------------------------------------------------------------------===//

// After bootstrap, the client's frontier tracker should contain the server's
// axes with proxy semaphores initialized to the exchanged epoch values.
TEST_P(SessionTest, ProxySemaphoreRegisteredAfterBootstrap) {
  iree_async_axis_t server_axes[] = {0x0200, 0x0201};
  uint64_t server_epochs[] = {10, 20};
  iree_net_session_topology_t server_topo = {};
  server_topo.axes = server_axes;
  server_topo.current_epochs = server_epochs;
  server_topo.axis_count = 2;
  server_topo.machine_index = 1;
  server_topo.session_epoch = 1;

  iree_net_session_topology_t client_topo = {};
  client_topo.machine_index = 0;
  client_topo.session_epoch = 1;

  EstablishSessionPair(client_topo, server_topo);

  // The client's tracker should have the server's axes registered.
  for (uint32_t i = 0; i < 2; ++i) {
    int32_t index =
        iree_async_axis_table_find(&client_tracker_.axis_table, server_axes[i]);
    ASSERT_GE(index, 0) << "Server axis 0x" << std::hex << server_axes[i]
                        << " not found in client's axis table";

    // The proxy semaphore should be non-NULL and initialized to the exchanged
    // epoch.
    iree_async_semaphore_t* semaphore =
        client_tracker_.axis_table.entries[index].semaphore;
    ASSERT_NE(semaphore, nullptr) << "Proxy semaphore for axis 0x" << std::hex
                                  << server_axes[i] << " is NULL";
    EXPECT_EQ(iree_async_semaphore_query(semaphore), server_epochs[i])
        << "Proxy semaphore for axis 0x" << std::hex << server_axes[i]
        << " should be initialized to epoch " << std::dec << server_epochs[i];
  }
}

// Advancing a remote axis via frontier_tracker_advance() should signal the
// proxy semaphore to the new epoch.
TEST_P(SessionTest, ProxySemaphoreSignaledOnAdvance) {
  EstablishDefaultSessionPair();

  // The client's tracker has server axis 0x0200 at epoch 0. Advance it.
  iree_async_axis_t server_axis = 0x0200;
  iree_host_size_t dispatched =
      iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 42);
  (void)dispatched;

  // The proxy semaphore should now report epoch 42.
  int32_t index =
      iree_async_axis_table_find(&client_tracker_.axis_table, server_axis);
  ASSERT_GE(index, 0);
  iree_async_semaphore_t* semaphore =
      client_tracker_.axis_table.entries[index].semaphore;
  ASSERT_NE(semaphore, nullptr);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 42u);
}

// Advancing a remote axis should satisfy frontier tracker waiters that
// reference that axis.
TEST_P(SessionTest, FrontierWaiterSatisfiedByRemoteAxisAdvance) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Build a single-entry frontier waiting for server_axis to reach epoch 5.
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 5};

  // Register a waiter.
  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      &client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_ignore(status);
      },
      &result, &waiter));

  // Waiter should not fire yet (epoch is 0, target is 5).
  EXPECT_FALSE(result.fired);

  // Advance to epoch 3 — still below target.
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 3);
  EXPECT_FALSE(result.fired);

  // Advance to epoch 5 — should satisfy the waiter.
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 5);
  EXPECT_TRUE(result.fired) << "Waiter should have fired when axis reached "
                               "target epoch";
  EXPECT_EQ(result.status_code, IREE_STATUS_OK);
}

// A semaphore timepoint on a proxy semaphore should fire when the remote axis
// is advanced via frontier_tracker_advance().
TEST_P(SessionTest, SemaphoreTimepointSatisfiedByRemoteAxisAdvance) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;
  int32_t index =
      iree_async_axis_table_find(&client_tracker_.axis_table, server_axis);
  ASSERT_GE(index, 0);
  iree_async_semaphore_t* semaphore =
      client_tracker_.axis_table.entries[index].semaphore;
  ASSERT_NE(semaphore, nullptr);

  // Acquire a timepoint waiting for the semaphore to reach epoch 10.
  struct TimepointResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_semaphore_timepoint_t timepoint;
  memset(&timepoint, 0, sizeof(timepoint));
  timepoint.callback = [](void* user_data, iree_async_semaphore_timepoint_t* tp,
                          iree_status_t status) {
    auto* r = static_cast<TimepointResult*>(user_data);
    r->fired = true;
    r->status_code = iree_status_code(status);
    iree_status_ignore(status);
  };
  timepoint.user_data = &result;

  IREE_ASSERT_OK(
      iree_async_semaphore_acquire_timepoint(semaphore, 10, &timepoint));

  // Should not fire yet.
  EXPECT_FALSE(result.fired);

  // Advance the axis to 10 via the frontier tracker — this should signal the
  // proxy semaphore and dispatch the timepoint.
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 10);
  EXPECT_TRUE(result.fired) << "Timepoint should have fired when proxy "
                               "semaphore was signaled via advance";
  EXPECT_EQ(result.status_code, IREE_STATUS_OK);
}

// Monotonic advancement: advancing to a lower epoch should be a no-op.
TEST_P(SessionTest, ProxySemaphoreMonotonicAdvance) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Advance to 100.
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 100);

  int32_t index =
      iree_async_axis_table_find(&client_tracker_.axis_table, server_axis);
  ASSERT_GE(index, 0);
  iree_async_semaphore_t* semaphore =
      client_tracker_.axis_table.entries[index].semaphore;
  ASSERT_NE(semaphore, nullptr);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 100u);

  // Advance to 50 — should be a no-op.
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 50);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 100u);

  // Advance to 100 again — also a no-op (not strictly greater).
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 100);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 100u);

  // Advance to 101 — should succeed.
  iree_async_frontier_tracker_advance(&client_tracker_, server_axis, 101);
  EXPECT_EQ(iree_async_semaphore_query(semaphore), 101u);
}

// Failing a remote axis should propagate the error to both frontier tracker
// waiters and semaphore timepoints.
TEST_P(SessionTest, AxisFailurePropagatesToWaitersAndTimepoints) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;
  int32_t index =
      iree_async_axis_table_find(&client_tracker_.axis_table, server_axis);
  ASSERT_GE(index, 0);
  iree_async_semaphore_t* semaphore =
      client_tracker_.axis_table.entries[index].semaphore;
  ASSERT_NE(semaphore, nullptr);

  // Register a frontier waiter on the remote axis.
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 999};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } waiter_result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      &client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_ignore(status);
      },
      &waiter_result, &waiter));

  // Register a semaphore timepoint on the proxy semaphore.
  struct TimepointResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } timepoint_result;
  iree_async_semaphore_timepoint_t timepoint;
  memset(&timepoint, 0, sizeof(timepoint));
  timepoint.callback = [](void* user_data, iree_async_semaphore_timepoint_t* tp,
                          iree_status_t status) {
    auto* r = static_cast<TimepointResult*>(user_data);
    r->fired = true;
    r->status_code = iree_status_code(status);
    iree_status_ignore(status);
  };
  timepoint.user_data = &timepoint_result;
  IREE_ASSERT_OK(
      iree_async_semaphore_acquire_timepoint(semaphore, 999, &timepoint));

  // Neither should have fired yet.
  EXPECT_FALSE(waiter_result.fired);
  EXPECT_FALSE(timepoint_result.fired);

  // Fail the axis — simulates remote disconnect.
  iree_async_frontier_tracker_fail_axis(
      &client_tracker_, server_axis,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "connection lost"));

  // Both the frontier waiter and the semaphore timepoint should have fired
  // with an error status.
  EXPECT_TRUE(waiter_result.fired) << "Frontier waiter should fire on axis "
                                      "failure";
  EXPECT_EQ(waiter_result.status_code, IREE_STATUS_UNAVAILABLE);

  EXPECT_TRUE(timepoint_result.fired) << "Semaphore timepoint should fire on "
                                         "axis failure";
  EXPECT_EQ(timepoint_result.status_code, IREE_STATUS_UNAVAILABLE);
}

// After axis failure, new waits on the failed axis should fail immediately.
TEST_P(SessionTest, NewWaitsFailAfterAxisFailure) {
  EstablishDefaultSessionPair();

  iree_async_axis_t server_axis = 0x0200;

  // Fail the axis.
  iree_async_frontier_tracker_fail_axis(
      &client_tracker_, server_axis,
      iree_make_status(IREE_STATUS_UNAVAILABLE, "gone"));

  // A new frontier waiter on the failed axis should fire immediately with
  // the failure status.
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {server_axis, 1};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      &client_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_ignore(status);
      },
      &result, &waiter));

  // Should have fired immediately since the axis is already failed.
  EXPECT_TRUE(result.fired)
      << "Wait on failed axis should dispatch immediately";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, RetainRelease) {
  EstablishDefaultSessionPair();

  // Extra retain/release cycle should not crash or affect state.
  iree_net_session_retain(client_session_);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  iree_net_session_release(client_session_);
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
}

TEST_P(SessionTest, RetainReleaseNullSafe) {
  // Both should be no-ops on NULL.
  iree_net_session_retain(nullptr);
  iree_net_session_release(nullptr);
}

//===----------------------------------------------------------------------===//
// Endpoint provisioning
//===----------------------------------------------------------------------===//

TEST_P(SessionTest, OpenEndpointRequiresOperational) {
  EstablishDefaultSessionPair();

  // Shut down first, then verify endpoint opening fails.
  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  EndpointReadyResult endpoint_result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_open_endpoint(
          client_session_, EndpointReadyResult::Callback, &endpoint_result));
}

}  // namespace

CTS_REGISTER_TEST_SUITE_WITH_TAGS(SessionTest, {"factory"}, {});

// SessionTest requires the "factory" tag — backends without factory support
// legitimately have zero instantiations.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SessionTest);

}  // namespace iree::net::carrier::cts
