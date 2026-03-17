// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loopback-specific transport factory tests.
//
// Tests that exercise loopback-specific behavior which cannot be generalized
// across transports. Common factory tests live in cts/factory_test.cc.

#include "iree/net/carrier/loopback/factory.h"

#include <string>

#include "iree/async/proactor_platform.h"
#include "iree/net/connection.h"
#include "iree/net/transport_factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture and helpers
//===----------------------------------------------------------------------===//

class LoopbackFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));
    IREE_ASSERT_OK(
        iree_net_loopback_factory_allocate(iree_allocator_system(), &factory_));
  }

  void TearDown() override {
    if (factory_) {
      iree_net_transport_factory_free(factory_);
      factory_ = nullptr;
    }
    if (proactor_) {
      iree_async_proactor_release(proactor_);
      proactor_ = nullptr;
    }
  }

  template <typename Fn>
  void PollUntil(Fn condition, int max_polls = 100) {
    for (int i = 0; i < max_polls && !condition(); ++i) {
      iree_host_size_t count = 0;
      iree_status_t status =
          iree_async_proactor_poll(proactor_, iree_make_timeout_ms(10), &count);
      if (iree_status_is_deadline_exceeded(status)) {
        iree_status_ignore(status);
      } else {
        IREE_ASSERT_OK(status);
      }
    }
  }

  void StopAndWait(iree_net_listener_t* listener) {
    bool stopped = false;
    IREE_ASSERT_OK(iree_net_listener_stop(
        listener,
        {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
         &stopped}));
    PollUntil([&]() { return stopped; });
    ASSERT_TRUE(stopped) << "Listener stop timed out";
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Loopback-specific tests
//===----------------------------------------------------------------------===//

TEST_F(LoopbackFactoryTest, DuplicateListenerName) {
  bool accept_fired = false;
  iree_net_listener_t* listener1 = nullptr;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, IREE_SV("dup"), proactor_, /*recv_pool=*/nullptr,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        *static_cast<bool*>(user_data) = true;
        iree_status_ignore(status);
        if (connection) iree_net_connection_release(connection);
      },
      &accept_fired, iree_allocator_system(), &listener1));

  iree_net_listener_t* listener2 = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_net_transport_factory_create_listener(
          factory_, IREE_SV("dup"), proactor_, /*recv_pool=*/nullptr,
          [](void*, iree_status_t, iree_net_connection_t*) {}, nullptr,
          iree_allocator_system(), &listener2));
  EXPECT_EQ(listener2, nullptr);

  StopAndWait(listener1);
  iree_net_listener_free(listener1);
}

TEST_F(LoopbackFactoryTest, AcceptCallbackFiresBeforeConnect) {
  // Verify ordering: accept callback fires before connect callback.
  // This is a loopback-specific invariant (NOP completion fires accept first).
  struct OrderContext {
    int counter = 0;
    int accept_order = -1;
    int connect_order = -1;
    iree_net_connection_t* accept_connection = nullptr;
    iree_net_connection_t* connect_connection = nullptr;
    bool done = false;
  } order_ctx;

  iree_net_listener_t* listener = nullptr;
  IREE_ASSERT_OK(iree_net_transport_factory_create_listener(
      factory_, IREE_SV("order"), proactor_, /*recv_pool=*/nullptr,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<OrderContext*>(user_data);
        IREE_CHECK_OK(status);
        ctx->accept_order = ctx->counter++;
        ctx->accept_connection = connection;
      },
      &order_ctx, iree_allocator_system(), &listener));

  IREE_ASSERT_OK(iree_net_transport_factory_connect(
      factory_, IREE_SV("order"), proactor_, /*recv_pool=*/nullptr,
      [](void* user_data, iree_status_t status,
         iree_net_connection_t* connection) {
        auto* ctx = static_cast<OrderContext*>(user_data);
        IREE_CHECK_OK(status);
        ctx->connect_order = ctx->counter++;
        ctx->connect_connection = connection;
        ctx->done = true;
      },
      &order_ctx));

  PollUntil([&]() { return order_ctx.done; });
  ASSERT_TRUE(order_ctx.done);

  // Accept must fire before connect.
  EXPECT_EQ(order_ctx.accept_order, 0);
  EXPECT_EQ(order_ctx.connect_order, 1);

  iree_net_connection_release(order_ctx.accept_connection);
  iree_net_connection_release(order_ctx.connect_connection);
  StopAndWait(listener);
  iree_net_listener_free(listener);
}

}  // namespace
}  // namespace iree
