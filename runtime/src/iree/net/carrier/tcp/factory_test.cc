// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TCP-specific transport factory tests.
//
// Tests that exercise TCP-specific behavior which cannot be generalized across
// transports: address parsing, capability queries, ephemeral port assignment.
// Common factory tests live in cts/factory_test.cc.

#include "iree/net/carrier/tcp/factory.h"

#include <string>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/net/connection.h"
#include "iree/net/transport_factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture and helpers
//===----------------------------------------------------------------------===//

class TcpFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));

    iree_async_slab_options_t slab_options = {0};
    slab_options.buffer_size = 4096;
    slab_options.buffer_count = 16;
    IREE_ASSERT_OK(
        iree_async_slab_create(slab_options, iree_allocator_system(), &slab_));
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
    IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
        region_, iree_allocator_system(), &recv_pool_));

    IREE_ASSERT_OK(
        iree_net_tcp_factory_create(iree_net_tcp_carrier_options_default(),
                                    iree_allocator_system(), &factory_));
  }

  void TearDown() override {
    if (factory_) {
      iree_net_transport_factory_release(factory_);
      factory_ = nullptr;
    }
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

  template <typename Fn>
  bool PollUntil(Fn condition, int max_polls = 200) {
    for (int i = 0; i < max_polls && !condition(); ++i) {
      iree_host_size_t count = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor_, iree_make_timeout_ms(50), &count);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
      } else {
        IREE_CHECK_OK(status);
      }
    }
    return condition();
  }

  struct ListenerInfo {
    iree_net_listener_t* listener = nullptr;
    std::string connect_address;
  };
  ListenerInfo CreateListener(
      iree_net_listener_accept_callback_t accept_callback, void* user_data) {
    ListenerInfo info;
    IREE_CHECK_OK(iree_net_transport_factory_create_listener(
        factory_, IREE_SV("127.0.0.1:0"), proactor_, recv_pool_,
        accept_callback, user_data, iree_allocator_system(), &info.listener));

    char address_buffer[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
    iree_string_view_t bound_address;
    IREE_CHECK_OK(iree_net_listener_query_bound_address(
        info.listener, sizeof(address_buffer), address_buffer, &bound_address));
    info.connect_address = std::string(bound_address.data, bound_address.size);
    return info;
  }

  void StopAndWait(iree_net_listener_t* listener) {
    bool stopped = false;
    IREE_ASSERT_OK(iree_net_listener_stop(
        listener,
        {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
         &stopped}));
    ASSERT_TRUE(PollUntil([&]() { return stopped; }))
        << "Listener stop timed out";
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
};

// Tracks the result of a connect callback.
struct ConnectResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_connection_t* connection = nullptr;
};

static void TrackConnectCallback(void* user_data, iree_status_t status,
                                 iree_net_connection_t* connection) {
  auto* result = static_cast<ConnectResult*>(user_data);
  result->fired = true;
  result->status_code = iree_status_code(status);
  result->connection = connection;
  iree_status_ignore(status);
}

//===----------------------------------------------------------------------===//
// TCP-specific tests
//===----------------------------------------------------------------------===//

TEST_F(TcpFactoryTest, QueryCapabilities) {
  iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory_);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
}

TEST_F(TcpFactoryTest, ConnectAsyncBadAddress) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("localhost"), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ConnectAsyncEmptyAddress) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, iree_string_view_empty(), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ConnectAsyncBadPort) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("127.0.0.1:abc"), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ConnectAsyncPortOutOfRange) {
  ConnectResult result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_transport_factory_connect(
                            factory_, IREE_SV("127.0.0.1:99999"), proactor_,
                            recv_pool_, TrackConnectCallback, &result));
  EXPECT_FALSE(result.fired);
}

TEST_F(TcpFactoryTest, ListenerEphemeralPort) {
  struct AcceptCtx {
    bool fired = false;
  } accept_ctx;

  auto info = CreateListener(
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<AcceptCtx*>(user_data);
        iree_status_ignore(status);
        if (connection) iree_net_connection_release(connection);
        ctx->fired = true;
      },
      &accept_ctx);

  // The bound address should have a non-zero port.
  EXPECT_FALSE(info.connect_address.empty());
  auto colon_position = info.connect_address.rfind(':');
  ASSERT_NE(colon_position, std::string::npos);
  std::string port_string = info.connect_address.substr(colon_position + 1);
  int port = std::stoi(port_string);
  EXPECT_GT(port, 0);
  EXPECT_LE(port, 65535);

  StopAndWait(info.listener);
  iree_net_listener_free(info.listener);
}

}  // namespace
}  // namespace iree
