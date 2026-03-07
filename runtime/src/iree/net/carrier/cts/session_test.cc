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
  IREE_ASSERT_OK(iree_net_session_send_control_data(
      client_session_, 0, iree_make_const_byte_span(message, strlen(message))));

  ASSERT_TRUE(PollUntil([&]() { return server_callbacks_.control_data_fired; }))
      << "Server never received control data";

  EXPECT_EQ(std::string(server_callbacks_.control_data.begin(),
                        server_callbacks_.control_data.end()),
            "hello server");
}

TEST_P(SessionTest, ControlDataServerToClient) {
  EstablishDefaultSessionPair();

  const char* message = "hello client";
  IREE_ASSERT_OK(iree_net_session_send_control_data(
      server_session_, 0, iree_make_const_byte_span(message, strlen(message))));

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
  IREE_ASSERT_OK(iree_net_session_send_control_data(
      client_session_, 0,
      iree_make_const_byte_span(to_server, strlen(to_server))));
  IREE_ASSERT_OK(iree_net_session_send_control_data(
      server_session_, 0,
      iree_make_const_byte_span(to_client, strlen(to_client))));

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
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_session_send_control_data(
          client_session_, 0, iree_make_const_byte_span(data, strlen(data))));

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
