// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS session tests: bootstrap, topology exchange, control data, shutdown,
// and queue channel round-trips over application endpoints.
//
// These tests exercise the session lifecycle across all transport backends
// that provide factory support. The session manages connection bootstrap
// (HELLO/HELLO_ACK), proxy semaphore creation, and control channel forwarding
// — all of which must work identically regardless of the underlying transport.
//
// The endpoint provisioning tests open application endpoints via the session
// and run queue COMMAND frames end-to-end over those endpoints, validating the
// full path: application → queue frame → message_endpoint → carrier →
// message_endpoint → application.
//
// Registered with the "factory" tag — only instantiated for backends that
// provide factory-level fields in their BackendInfo.

#include <cstring>
#include <string>
#include <vector>

#include "iree/async/semaphore.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/session_test_base.h"
#include "iree/net/channel/queue/frame.h"
#include "iree/net/channel/queue/queue_channel.h"

namespace iree::net::carrier::cts {
namespace {

// Creates a buffer pool for queue channel header encoding, with self-contained
// ownership: the region's destroy callback frees the buffer memory. Ownership
// of the returned pool is transferred to the caller (typically a queue
// channel).
static iree_async_buffer_pool_t* CreateHeaderPool() {
  static constexpr iree_host_size_t kBufferCount = 16;
  static constexpr iree_host_size_t kBufferSize = 256;
  iree_host_size_t total_size = kBufferCount * kBufferSize;

  void* memory = malloc(total_size);
  memset(memory, 0, total_size);

  iree_async_region_t* region =
      static_cast<iree_async_region_t*>(malloc(sizeof(iree_async_region_t)));
  memset(region, 0, sizeof(*region));
  iree_atomic_ref_count_init(&region->ref_count);
  region->destroy_fn = [](iree_async_region_t* r) {
    free(r->base_ptr);
    free(r);
  };
  region->base_ptr = memory;
  region->length = total_size;
  region->buffer_size = kBufferSize;
  region->buffer_count = kBufferCount;

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_CHECK_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));
  iree_async_region_release(region);  // Pool retains it.
  return pool;
}

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
      iree_net_session_accept(pair.server, proactor_, &server_tracker_,
                              &options, callbacks.MakeCallbacks(),
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
                            pair.server, proactor_, &server_tracker_, &options,
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
                            pair.server, proactor_, &server_tracker_, &options,
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
          client_session_, {EndpointReadyResult::Callback, &endpoint_result}));
}

TEST_P(SessionTest, OpenEndpointSucceeds) {
  EstablishDefaultSessionPair();

  // Open application endpoints on both sides (slot 1; slot 0 is the control
  // channel consumed during bootstrap).
  EndpointReadyResult client_endpoint;
  EndpointReadyResult server_endpoint;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_endpoint}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_endpoint}));

  ASSERT_TRUE(PollUntil([&]() {
    return client_endpoint.fired && server_endpoint.fired;
  })) << "Endpoint open timed out";

  EXPECT_EQ(client_endpoint.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_endpoint.status_code, IREE_STATUS_OK);
  EXPECT_NE(client_endpoint.endpoint.self, nullptr);
  EXPECT_NE(server_endpoint.endpoint.self, nullptr);
}

TEST_P(SessionTest, MultipleEndpointsSucceed) {
  EstablishDefaultSessionPair();

  // Open two application endpoints on each side (slots 1 and 2).
  EndpointReadyResult client_ep1, client_ep2;
  EndpointReadyResult server_ep1, server_ep2;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_ep1}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_ep2}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_ep1}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_ep2}));

  ASSERT_TRUE(PollUntil([&]() {
    return client_ep1.fired && client_ep2.fired && server_ep1.fired &&
           server_ep2.fired;
  })) << "Multiple endpoint open timed out";

  EXPECT_EQ(client_ep1.status_code, IREE_STATUS_OK);
  EXPECT_EQ(client_ep2.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_ep1.status_code, IREE_STATUS_OK);
  EXPECT_EQ(server_ep2.status_code, IREE_STATUS_OK);

  // Each endpoint should be distinct.
  EXPECT_NE(client_ep1.endpoint.self, client_ep2.endpoint.self);
  EXPECT_NE(server_ep1.endpoint.self, server_ep2.endpoint.self);
}

//===----------------------------------------------------------------------===//
// Queue channel round-trip over application endpoints
//===----------------------------------------------------------------------===//

// Tracks a received queue COMMAND for test assertions.
struct ReceivedQueueCommand {
  bool fired = false;
  uint32_t stream_id = 0;
  std::vector<uint8_t> data;

  static iree_status_t Callback(void* user_data, uint32_t stream_id,
                                const iree_async_frontier_t* wait_frontier,
                                const iree_async_frontier_t* signal_frontier,
                                iree_const_byte_span_t command_data,
                                iree_async_buffer_lease_t* lease) {
    auto* result = static_cast<ReceivedQueueCommand*>(user_data);
    result->fired = true;
    result->stream_id = stream_id;
    result->data.insert(result->data.end(), command_data.data,
                        command_data.data + command_data.data_length);
    return iree_ok_status();
  }
};

TEST_P(SessionTest, QueueChannelCommandRoundTrip) {
  EstablishDefaultSessionPair();

  // Open application endpoints on both sides.
  EndpointReadyResult client_ep_result;
  EndpointReadyResult server_ep_result;
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      client_session_, {EndpointReadyResult::Callback, &client_ep_result}));
  IREE_ASSERT_OK(iree_net_session_open_endpoint(
      server_session_, {EndpointReadyResult::Callback, &server_ep_result}));

  ASSERT_TRUE(PollUntil([&]() {
    return client_ep_result.fired && server_ep_result.fired;
  })) << "Endpoint open timed out";
  ASSERT_EQ(client_ep_result.status_code, IREE_STATUS_OK);
  ASSERT_EQ(server_ep_result.status_code, IREE_STATUS_OK);

  // Create queue channels on the application endpoints.
  ReceivedQueueCommand server_command;
  ReceivedQueueCommand client_command;

  iree_net_queue_channel_callbacks_t server_qcb = {};
  server_qcb.on_command = ReceivedQueueCommand::Callback;
  server_qcb.user_data = &server_command;

  iree_net_queue_channel_callbacks_t client_qcb = {};
  client_qcb.on_command = ReceivedQueueCommand::Callback;
  client_qcb.user_data = &client_command;

  iree_net_queue_channel_t* client_channel = nullptr;
  iree_net_queue_channel_t* server_channel = nullptr;

  IREE_ASSERT_OK(iree_net_queue_channel_create(
      client_ep_result.endpoint, /*max_send_spans=*/8, CreateHeaderPool(),
      client_qcb, iree_allocator_system(), &client_channel));
  IREE_ASSERT_OK(iree_net_queue_channel_create(
      server_ep_result.endpoint, /*max_send_spans=*/8, CreateHeaderPool(),
      server_qcb, iree_allocator_system(), &server_channel));

  // Activate both channels (installs recv handlers and activates endpoints).
  IREE_ASSERT_OK(iree_net_queue_channel_activate(client_channel));
  IREE_ASSERT_OK(iree_net_queue_channel_activate(server_channel));

  // Send a COMMAND from client → server.
  const char* payload = "hello queue";
  iree_async_span_t span =
      iree_async_span_from_ptr((void*)payload, strlen(payload));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      client_channel, /*stream_id=*/7, /*wait_frontier=*/NULL,
      /*signal_frontier=*/NULL, span_list, /*operation_user_data=*/0));

  ASSERT_TRUE(PollUntil([&]() { return server_command.fired; }))
      << "Server never received queue command";
  EXPECT_EQ(server_command.stream_id, 7u);
  EXPECT_EQ(std::string(server_command.data.begin(), server_command.data.end()),
            "hello queue");

  // Send a COMMAND from server → client.
  const char* reply = "queue reply";
  iree_async_span_t reply_span =
      iree_async_span_from_ptr((void*)reply, strlen(reply));
  iree_async_span_list_t reply_list = iree_async_span_list_make(&reply_span, 1);
  IREE_ASSERT_OK(iree_net_queue_channel_send_command(
      server_channel, /*stream_id=*/42, /*wait_frontier=*/NULL,
      /*signal_frontier=*/NULL, reply_list, /*operation_user_data=*/0));

  ASSERT_TRUE(PollUntil([&]() { return client_command.fired; }))
      << "Client never received queue reply";
  EXPECT_EQ(client_command.stream_id, 42u);
  EXPECT_EQ(std::string(client_command.data.begin(), client_command.data.end()),
            "queue reply");

  // Drain pending send completions before releasing channels. SHM carriers
  // fire send completions asynchronously when the peer's SPSC ring consumer
  // advances; the completion may lag one poll cycle behind data receipt.
  ASSERT_TRUE(PollUntil([&]() {
    return !iree_net_queue_channel_has_pending_sends(client_channel) &&
           !iree_net_queue_channel_has_pending_sends(server_channel);
  })) << "Send completions did not drain";

  iree_net_queue_channel_release(server_channel);
  iree_net_queue_channel_release(client_channel);
}

//===----------------------------------------------------------------------===//
// Error state transitions
//===----------------------------------------------------------------------===//

// Protocol version mismatch during bootstrap causes the server session to
// transition to ERROR state and fire on_error with the validation failure.
//
// This exercises the bootstrap error routing fix: handle_hello() returns an
// error, on_data catches it, and routes it through fail() so the session
// transitions directly to ERROR with the specific diagnostic. Without the fix,
// the error would bubble through the carrier's transport error path, losing the
// original message.
TEST_P(SessionTest, ProtocolVersionMismatchCausesServerError) {
  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  // AcceptCtx holds direct pointers to avoid C++ protected member access
  // restrictions (lambdas in TEST_P can't access protected base members
  // through a base class pointer).
  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    bool fired = false;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = &server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_CHECK_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        IREE_CHECK_OK(status);

        iree_net_session_options_t server_options =
            iree_net_session_options_default();
        server_options.session_id = 1;

        IREE_CHECK_OK(iree_net_session_accept(
            connection, ctx->proactor, ctx->tracker, &server_options,
            ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
            ctx->out_session));

        iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  std::string connect_str = ResolveConnectAddress(bind_str, listener_);

  // Client uses a wrong protocol version — the server will reject the HELLO.
  // Short bootstrap timeout so the client's timer expires during TearDown
  // rather than leaking (the client never receives a HELLO_ACK).
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.protocol_version = 999;
  client_options.bootstrap_timeout_ns = iree_make_duration_ms(500);

  IREE_CHECK_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, &client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  // Wait for the server to receive the bad HELLO and fire on_error.
  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.error_fired; }))
      << "Server on_error never fired after protocol version mismatch";

  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_ERROR);
  // The server should see INVALID_ARGUMENT from the protocol version check.
  EXPECT_EQ(server_callbacks_.error_code, IREE_STATUS_INVALID_ARGUMENT);

  // The server should NOT have reached OPERATIONAL.
  EXPECT_FALSE(server_callbacks_.ready_fired);

  // Drain carrier operations before TearDown releases sessions. SHM carriers
  // process send completions asynchronously via wake events — the client's
  // HELLO send completion may still be in-flight when the error fires.
  PollUntil([&]() { return false; }, iree_make_duration_ms(200));
}

// After a session enters ERROR state, all operations return
// FAILED_PRECONDITION.
TEST_P(SessionTest, OperationsFailInErrorState) {
  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    bool fired = false;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = &server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;

  IREE_CHECK_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        IREE_CHECK_OK(status);

        iree_net_session_options_t server_options =
            iree_net_session_options_default();
        server_options.session_id = 1;

        IREE_CHECK_OK(iree_net_session_accept(
            connection, ctx->proactor, ctx->tracker, &server_options,
            ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
            ctx->out_session));

        iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  std::string connect_str = ResolveConnectAddress(bind_str, listener_);

  // Client with wrong protocol version forces server into ERROR state.
  // Short bootstrap timeout so the client's timer expires during TearDown
  // rather than leaking (the client never receives a HELLO_ACK).
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.protocol_version = 999;
  client_options.bootstrap_timeout_ns = iree_make_duration_ms(500);

  IREE_CHECK_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, &client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.error_fired; }))
      << "Server on_error never fired";

  ASSERT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_ERROR);

  // send_control_data should fail.
  const char* data = "nope";
  iree_async_span_t span = iree_async_span_from_ptr((void*)data, strlen(data));
  iree_async_span_list_t span_list = iree_async_span_list_make(&span, 1);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_send_control_data(server_session_, 0, span_list, 0));

  // shutdown should fail.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_shutdown(server_session_, 0, IREE_SV("late")));

  // open_endpoint should fail.
  EndpointReadyResult endpoint_result;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_open_endpoint(
          server_session_, {EndpointReadyResult::Callback, &endpoint_result}));

  // Drain carrier operations before TearDown releases sessions.
  PollUntil([&]() { return false; }, iree_make_duration_ms(200));
}

//===----------------------------------------------------------------------===//
// Proxy semaphore cleanup on shutdown/goaway
//===----------------------------------------------------------------------===//

// When a session shuts down (sends GOAWAY), it synchronously fails all remote
// axes in the frontier tracker. Pending frontier waiters on those axes should
// fire with UNAVAILABLE.
TEST_P(SessionTest, ShutdownCleanupFailsRemoteAxisWaiters) {
  EstablishDefaultSessionPair();

  // The client has server axis 0x0200 in its tracker. Register a waiter on
  // that axis waiting for epoch 999 (will never arrive normally).
  iree_async_axis_t server_axis = 0x0200;
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

  EXPECT_FALSE(result.fired);

  // Client shuts down — this synchronously fails remote axes in the client's
  // tracker, which should wake the waiter.
  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  EXPECT_TRUE(result.fired)
      << "Frontier waiter on remote axis should fire when session shuts down";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

// When a session receives GOAWAY from the peer, it fails all remote axes in
// the frontier tracker. Pending frontier waiters on those axes should fire
// with UNAVAILABLE.
TEST_P(SessionTest, GoawayReceivedCleanupFailsRemoteAxisWaiters) {
  EstablishDefaultSessionPair();

  // The server has client axis 0x0100 in its tracker. Register a waiter on
  // that axis waiting for epoch 999 (will never arrive normally).
  iree_async_axis_t client_axis = 0x0100;
  iree_host_size_t frontier_size = 0;
  IREE_ASSERT_OK(iree_async_frontier_size(1, &frontier_size));
  std::vector<uint8_t> frontier_storage(frontier_size);
  auto* frontier =
      reinterpret_cast<iree_async_frontier_t*>(frontier_storage.data());
  iree_async_frontier_initialize(frontier, 1);
  frontier->entries[0] = {client_axis, 999};

  struct WaiterResult {
    bool fired = false;
    iree_status_code_t status_code = IREE_STATUS_OK;
  } result;
  iree_async_frontier_waiter_t waiter;
  IREE_ASSERT_OK(iree_async_frontier_tracker_wait(
      &server_tracker_, frontier,
      [](void* user_data, iree_status_t status) {
        auto* r = static_cast<WaiterResult*>(user_data);
        r->fired = true;
        r->status_code = iree_status_code(status);
        iree_status_ignore(status);
      },
      &result, &waiter));

  EXPECT_FALSE(result.fired);

  // Client sends GOAWAY → server receives it → server's cleanup_remote_axes
  // fires → client axis failed in server's tracker.
  IREE_ASSERT_OK(iree_net_session_shutdown(client_session_, 0, IREE_SV("bye")));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.goaway_fired; }))
      << "Server never received GOAWAY";

  EXPECT_TRUE(result.fired)
      << "Frontier waiter on remote axis should fire when GOAWAY is received";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

// Semaphore timepoints on proxy semaphores should fire with UNAVAILABLE when
// the peer sends GOAWAY. This tests the full chain: GOAWAY → session cleanup →
// frontier_tracker_fail_axis → semaphore failure → timepoint dispatch.
TEST_P(SessionTest, GoawayReceivedCleanupFailsSemaphoreTimepoints) {
  EstablishDefaultSessionPair();

  // Find the proxy semaphore for client axis 0x0100 in the server's tracker.
  iree_async_axis_t client_axis = 0x0100;
  int32_t index =
      iree_async_axis_table_find(&server_tracker_.axis_table, client_axis);
  ASSERT_GE(index, 0) << "Client axis not found in server tracker";
  iree_async_semaphore_t* semaphore =
      server_tracker_.axis_table.entries[index].semaphore;
  ASSERT_NE(semaphore, nullptr);

  // Acquire a timepoint waiting for epoch 999 on the proxy semaphore.
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
      iree_async_semaphore_acquire_timepoint(semaphore, 999, &timepoint));

  EXPECT_FALSE(result.fired);

  // Client sends GOAWAY → server cleanup fails the client axis.
  IREE_ASSERT_OK(
      iree_net_session_shutdown(client_session_, 0, IREE_SV("done")));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.goaway_fired; }))
      << "Server never received GOAWAY";

  EXPECT_TRUE(result.fired)
      << "Semaphore timepoint on proxy semaphore should fire on GOAWAY";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

//===----------------------------------------------------------------------===//
// Bootstrap timeout
//===----------------------------------------------------------------------===//

// When the server accepts a connection but never creates a session, the client
// session should eventually enter ERROR state (either from a transport error
// or from the bootstrap timeout).
//
// Transport behavior varies:
// - Loopback: the server's carrier is never activated, so the client's HELLO
//   send fails immediately with UNAVAILABLE (the timer is cancelled by this
//   transport error, not by expiry).
// - TCP/SHM: the client's HELLO reaches the kernel buffer / shared memory
//   ring and succeeds. The server never responds, so the client waits until
//   the bootstrap timer fires with DEADLINE_EXCEEDED.
//
// Both paths exercise the bootstrap timer lifecycle: start at session creation,
// cancel on error (loopback) or fire on expiry (TCP/SHM).
TEST_P(SessionTest, ClientErrorsWhenServerNeverResponds) {
  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  // Server accepts connections but does NOT create a session. The connection
  // is held alive (retained by the accept callback) so the transport layer
  // doesn't report a disconnect.
  iree_net_connection_t* held_connection = nullptr;
  IREE_CHECK_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        IREE_CHECK_OK(status);
        auto** held = static_cast<iree_net_connection_t**>(user_data);
        *held = connection;  // Hold the connection; don't create a session.
      },
      &held_connection, iree_allocator_system(), &listener_));

  std::string connect_str = ResolveConnectAddress(bind_str, listener_);

  // Short bootstrap timeout so the test doesn't wait 10 seconds.
  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.bootstrap_timeout_ns = iree_make_duration_ms(200);

  IREE_CHECK_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, &client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  // Wait for the client to enter ERROR state.
  ASSERT_TRUE(PollUntil([&]() { return client_callbacks_.error_fired; }))
      << "Client on_error never fired (expected UNAVAILABLE or "
         "DEADLINE_EXCEEDED)";

  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_ERROR);
  EXPECT_FALSE(client_callbacks_.ready_fired)
      << "on_ready should not fire when server never responds";

  // The error code depends on transport: UNAVAILABLE for loopback (immediate
  // transport error), DEADLINE_EXCEEDED for TCP/SHM (bootstrap timeout).
  EXPECT_TRUE(client_callbacks_.error_code == IREE_STATUS_UNAVAILABLE ||
              client_callbacks_.error_code == IREE_STATUS_DEADLINE_EXCEEDED)
      << "Expected UNAVAILABLE or DEADLINE_EXCEEDED, got "
      << client_callbacks_.error_code;

  // Release the held server connection.
  if (held_connection) {
    iree_net_connection_release(held_connection);
    held_connection = nullptr;
  }

  // Drain any pending operations before TearDown.
  PollUntil([&]() { return false; }, iree_make_duration_ms(100));
}

// Normal bootstrap cancels the timer — verify no DEADLINE_EXCEEDED error fires
// after a successful session establishment.
TEST_P(SessionTest, BootstrapTimeoutCancelledOnSuccess) {
  // Use a very short timeout (50ms). If the cancel path were broken, this
  // timer would fire during the test and transition the session to ERROR.
  iree_async_axis_t client_axes[] = {0x0100};
  uint64_t client_epochs[] = {0};
  iree_net_session_topology_t client_topo = {};
  client_topo.axes = client_axes;
  client_topo.current_epochs = client_epochs;
  client_topo.axis_count = 1;
  client_topo.machine_index = 0;
  client_topo.session_epoch = 1;

  iree_async_axis_t server_axes[] = {0x0200};
  uint64_t server_epochs[] = {0};
  iree_net_session_topology_t server_topo = {};
  server_topo.axes = server_axes;
  server_topo.current_epochs = server_epochs;
  server_topo.axis_count = 1;
  server_topo.machine_index = 1;
  server_topo.session_epoch = 1;

  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct AcceptCtx {
    iree_async_proactor_t* proactor = nullptr;
    iree_async_frontier_tracker_t* tracker = nullptr;
    SessionCallbackTracker* callbacks = nullptr;
    iree_net_session_t** out_session = nullptr;
    iree_net_session_topology_t server_topology = {};
    bool fired = false;
  } accept_ctx;
  accept_ctx.proactor = proactor_;
  accept_ctx.tracker = &server_tracker_;
  accept_ctx.callbacks = &server_callbacks_;
  accept_ctx.out_session = &server_session_;
  accept_ctx.server_topology = server_topo;

  IREE_CHECK_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        IREE_CHECK_OK(status);

        iree_net_session_options_t server_options =
            iree_net_session_options_default();
        server_options.local_topology = ctx->server_topology;
        server_options.session_id = 42;
        server_options.bootstrap_timeout_ns = iree_make_duration_ms(50);

        IREE_CHECK_OK(iree_net_session_accept(
            connection, ctx->proactor, ctx->tracker, &server_options,
            ctx->callbacks->MakeCallbacks(), iree_allocator_system(),
            ctx->out_session));

        iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx, iree_allocator_system(), &listener_));

  std::string connect_str = ResolveConnectAddress(bind_str, listener_);

  iree_net_session_options_t client_options =
      iree_net_session_options_default();
  client_options.local_topology = client_topo;
  client_options.bootstrap_timeout_ns = iree_make_duration_ms(50);

  IREE_CHECK_OK(iree_net_session_connect(
      factory_, iree_make_string_view(connect_str.c_str(), connect_str.size()),
      proactor_, recv_pool_, &client_tracker_, &client_options,
      client_callbacks_.MakeCallbacks(), iree_allocator_system(),
      &client_session_));

  // Bootstrap should complete successfully.
  ASSERT_TRUE(PollUntil([&]() {
    return client_callbacks_.ready_fired && server_callbacks_.ready_fired;
  })) << "Bootstrap timed out";

  // Wait an extra 200ms — well past the 50ms bootstrap timeout. If the
  // timer cancel is broken, it would fire here and transition to ERROR.
  PollUntil([&]() { return false; }, iree_make_duration_ms(200));

  // Both sessions should still be OPERATIONAL (no spurious timeout).
  EXPECT_EQ(iree_net_session_state(client_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  EXPECT_EQ(iree_net_session_state(server_session_),
            IREE_NET_SESSION_STATE_OPERATIONAL);
  EXPECT_FALSE(client_callbacks_.error_fired)
      << "Client should not get DEADLINE_EXCEEDED after successful bootstrap";
  EXPECT_FALSE(server_callbacks_.error_fired)
      << "Server should not get DEADLINE_EXCEEDED after successful bootstrap";
}

}  // namespace

CTS_REGISTER_TEST_SUITE_WITH_TAGS(SessionTest, {"factory"}, {});

// SessionTest requires the "factory" tag — backends without factory support
// legitimately have zero instantiations.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SessionTest);

}  // namespace iree::net::carrier::cts
