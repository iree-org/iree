// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS factory tests: transport factory → listener → connection → endpoint.
//
// These tests exercise the factory-level pipeline that is common across all
// transport backends. Transport-specific tests (address parsing, ordering
// guarantees, capability queries) remain in per-transport factory_test.cc
// files.
//
// Registered with the "factory" tag — only instantiated for backends that
// provide factory-level fields in their BackendInfo.

#include <cstring>
#include <string>
#include <vector>

#include "iree/net/carrier/cts/util/factory_test_base.h"
#include "iree/net/carrier/cts/util/registry.h"

namespace iree::net::carrier::cts {
namespace {

using FactoryTest = FactoryTestBase;

//===----------------------------------------------------------------------===//
// Factory lifecycle
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, AllocateAndFree) { EXPECT_NE(factory_, nullptr); }

//===----------------------------------------------------------------------===//
// Connect and listener management
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, ConnectAsyncUnreachable) {
  std::string unreachable = MakeUnreachableAddress();
  iree_string_view_t address = iree_make_cstring_view(unreachable.c_str());

  ConnectResult result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, address, proactor_, recv_pool_, ConnectResult::Callback,
      &result));

  // Callback must NOT fire synchronously.
  EXPECT_FALSE(result.fired);

  ASSERT_TRUE(PollUntil([&]() { return result.fired; }))
      << "Connect callback never fired";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
  EXPECT_EQ(result.connection, nullptr);
}

TEST_P(FactoryTest, ConnectAsyncSuccess) {
  auto pair = EstablishConnection();
  ASSERT_NE(pair.client, nullptr);
  ASSERT_NE(pair.server, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(FactoryTest, CallbackNotFiredBeforePoll) {
  // Verify that connect returns with the callback still pending.
  // This is the fundamental async contract for all transports.
  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  iree_net_listener_t* listener = nullptr;
  bool accept_fired = false;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        *static_cast<bool*>(user_data) = true;
        iree_status_ignore(status);
        if (connection) iree_net_connection_release(connection);
      },
      &accept_fired, iree_allocator_system(), &listener));

  std::string connect_str = ResolveConnectAddress(bind_str, listener);
  iree_string_view_t connect_addr = iree_make_cstring_view(connect_str.c_str());

  ConnectResult result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, connect_addr, proactor_, recv_pool_, ConnectResult::Callback,
      &result));

  // No poll — neither callback should have fired.
  EXPECT_FALSE(accept_fired);
  EXPECT_FALSE(result.fired);

  // Now poll and verify they fire.
  ASSERT_TRUE(PollUntil([&]() { return result.fired; }));
  EXPECT_TRUE(result.fired);
  EXPECT_TRUE(accept_fired);

  if (result.connection) iree_net_connection_release(result.connection);
  StopAndWait(listener);
  iree_net_listener_free(listener);
}

TEST_P(FactoryTest, MultipleConnections) {
  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  struct MultiAcceptCtx {
    std::vector<iree_net_connection_t*> connections;
  } accept_ctx;

  iree_net_listener_t* listener = nullptr;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<MultiAcceptCtx*>(user_data);
        IREE_CHECK_OK(status);
        ctx->connections.push_back(connection);
      },
      &accept_ctx, iree_allocator_system(), &listener));

  std::string connect_str = ResolveConnectAddress(bind_str, listener);
  iree_string_view_t connect_addr = iree_make_cstring_view(connect_str.c_str());

  ConnectResult connect_results[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_net_transport_factory_connect(
        factory_, connect_addr, proactor_, recv_pool_, ConnectResult::Callback,
        &connect_results[i]));
  }

  ASSERT_TRUE(PollUntil([&]() {
    return accept_ctx.connections.size() >= 3 && connect_results[0].fired &&
           connect_results[1].fired && connect_results[2].fired;
  })) << "Not all connections completed";
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(connect_results[i].status_code, IREE_STATUS_OK);
    ASSERT_NE(connect_results[i].connection, nullptr);
    iree_net_connection_release(connect_results[i].connection);
  }
  for (auto* connection : accept_ctx.connections) {
    iree_net_connection_release(connection);
  }

  StopAndWait(listener);
  iree_net_listener_free(listener);
}

TEST_P(FactoryTest, ListenerStop) {
  std::string bind_str = MakeBindAddress();
  iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

  iree_net_listener_t* listener = nullptr;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, bind_addr, proactor_, recv_pool_,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        iree_status_ignore(status);
        if (connection) iree_net_connection_release(connection);
      },
      nullptr, iree_allocator_system(), &listener));

  std::string connect_str = ResolveConnectAddress(bind_str, listener);

  // Stopped callback must be delivered asynchronously.
  bool stopped = false;
  IREE_ASSERT_OK(iree_net_listener_stop(
      listener, {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
                 &stopped}));
  EXPECT_FALSE(stopped);

  ASSERT_TRUE(PollUntil([&]() { return stopped; }));

  // Free the listener so the transport can clean up (TCP releases the socket,
  // which causes the kernel to RST new SYNs).
  iree_net_listener_free(listener);

  // After stop+free, connecting should fail asynchronously.
  iree_string_view_t connect_addr = iree_make_cstring_view(connect_str.c_str());
  ConnectResult result;
  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, connect_addr, proactor_, recv_pool_, ConnectResult::Callback,
      &result));
  ASSERT_TRUE(PollUntil([&]() { return result.fired; }))
      << "Connect callback never fired after listener stop";
  EXPECT_EQ(result.status_code, IREE_STATUS_UNAVAILABLE);
}

//===----------------------------------------------------------------------===//
// Endpoint opening
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, OpenEndpointFirst) {
  auto pair = EstablishConnection();

  // First open_endpoint returns a borrowed endpoint view.
  EndpointReadyResult endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.client, {EndpointReadyResult::Callback, &endpoint_result}));

  // Must be async.
  EXPECT_FALSE(endpoint_result.fired);

  ASSERT_TRUE(PollUntil([&]() { return endpoint_result.fired; }))
      << "Endpoint ready callback never fired";
  EXPECT_EQ(endpoint_result.status_code, IREE_STATUS_OK);
  ASSERT_NE(endpoint_result.endpoint.self, nullptr);

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

TEST_P(FactoryTest, ConnectionReleaseWithoutEndpoint) {
  // Create a connection but never open_endpoint — the transport stack should be
  // released when the connection is destroyed. ASan/LSan catch leaks.
  auto pair = EstablishConnection();

  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

//===----------------------------------------------------------------------===//
// End-to-end data transfer
//===----------------------------------------------------------------------===//

TEST_P(FactoryTest, BidirectionalSendRecv) {
  auto pair = EstablishConnection();

  // Open endpoints on both sides.
  EndpointReadyResult client_endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.client, {EndpointReadyResult::Callback, &client_endpoint_result}));
  EndpointReadyResult server_endpoint_result;
  IREE_ASSERT_OK(iree_net_connection_open_endpoint(
      pair.server, {EndpointReadyResult::Callback, &server_endpoint_result}));
  ASSERT_TRUE(PollUntil([&]() {
    return client_endpoint_result.fired && server_endpoint_result.fired;
  })) << "Endpoint ready callbacks never fired";
  ASSERT_NE(client_endpoint_result.endpoint.self, nullptr);
  ASSERT_NE(server_endpoint_result.endpoint.self, nullptr);

  iree_net_message_endpoint_t client_endpoint = client_endpoint_result.endpoint;
  iree_net_message_endpoint_t server_endpoint = server_endpoint_result.endpoint;

  // Set recv handlers on both sides.
  struct RecvContext {
    std::vector<uint8_t> data;
    bool received = false;
  };
  RecvContext client_recv;
  RecvContext server_recv;

  auto make_callbacks = [](RecvContext* ctx) {
    return iree_net_message_endpoint_callbacks_t{
        [](void* user_data, iree_const_byte_span_t message,
           iree_async_buffer_lease_t* lease) -> iree_status_t {
          auto* recv_ctx = static_cast<RecvContext*>(user_data);
          recv_ctx->data.insert(recv_ctx->data.end(), message.data,
                                message.data + message.data_length);
          recv_ctx->received = true;
          if (lease) iree_async_buffer_lease_release(lease);
          return iree_ok_status();
        },
        nullptr,  // on_error
        ctx};
  };

  iree_net_message_endpoint_set_callbacks(client_endpoint,
                                          make_callbacks(&client_recv));
  iree_net_message_endpoint_set_callbacks(server_endpoint,
                                          make_callbacks(&server_recv));

  IREE_ASSERT_OK(iree_net_message_endpoint_activate(client_endpoint));
  IREE_ASSERT_OK(iree_net_message_endpoint_activate(server_endpoint));

  // Client → server.
  char client_msg[] = "hello from client";
  iree_async_span_t client_span =
      iree_async_span_from_ptr(client_msg, strlen(client_msg));
  iree_net_message_endpoint_send_params_t params;
  memset(&params, 0, sizeof(params));
  params.data = iree_async_span_list_make(&client_span, 1);
  IREE_ASSERT_OK(iree_net_message_endpoint_send(client_endpoint, &params));

  // Poll until server receives.
  ASSERT_TRUE(PollUntil([&]() { return server_recv.received; }))
      << "Server never received data";
  EXPECT_EQ(std::string(server_recv.data.begin(), server_recv.data.end()),
            "hello from client");

  // Server → client.
  char server_msg[] = "hello from server";
  iree_async_span_t server_span =
      iree_async_span_from_ptr(server_msg, strlen(server_msg));
  params.data = iree_async_span_list_make(&server_span, 1);
  IREE_ASSERT_OK(iree_net_message_endpoint_send(server_endpoint, &params));

  ASSERT_TRUE(PollUntil([&]() { return client_recv.received; }))
      << "Client never received data";
  EXPECT_EQ(std::string(client_recv.data.begin(), client_recv.data.end()),
            "hello from server");

  // Cleanup.
  DeactivateEndpointAndWait(client_endpoint);
  DeactivateEndpointAndWait(server_endpoint);
  iree_net_connection_release(pair.client);
  iree_net_connection_release(pair.server);
  StopAndWait(pair.listener);
  iree_net_listener_free(pair.listener);
}

}  // namespace

CTS_REGISTER_TEST_SUITE_WITH_TAGS(FactoryTest, {"factory"}, {});

}  // namespace iree::net::carrier::cts
