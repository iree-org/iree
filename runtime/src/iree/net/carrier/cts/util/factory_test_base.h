// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base class for transport factory CTS tests.
//
// Provides the fixture and helpers shared across all transport factory tests.
// Parameterized on BackendInfo — the create_factory, make_bind_address,
// resolve_connect_address, and make_unreachable_address fields in BackendInfo
// drive transport-specific behavior.
//
// SetUp creates a proactor, slab/region/recv_pool (for transports that need
// registered buffers), and the transport factory. TearDown releases everything
// in reverse order.
//
// This base class is also usable by non-CTS tests that need a working factory
// for setup — any test that creates connections or carriers via the factory
// interface can derive from this and set up BackendInfo appropriately.

#ifndef IREE_NET_CARRIER_CTS_UTIL_FACTORY_TEST_BASE_H_
#define IREE_NET_CARRIER_CTS_UTIL_FACTORY_TEST_BASE_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/slab.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/connection.h"
#include "iree/net/message_endpoint.h"
#include "iree/net/transport_factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::carrier::cts {

//===----------------------------------------------------------------------===//
// Callback tracking utilities
//===----------------------------------------------------------------------===//

// Tracks the result of an async connect callback.
struct ConnectResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_connection_t* connection = nullptr;

  static void Callback(void* user_data, iree_status_t status,
                       iree_net_connection_t* connection) {
    auto* result = static_cast<ConnectResult*>(user_data);
    result->fired = true;
    result->status_code = iree_status_code(status);
    result->connection = connection;
    iree_status_ignore(status);
  }
};

// Tracks the result of an async open_endpoint callback.
struct EndpointReadyResult {
  bool fired = false;
  iree_status_code_t status_code = IREE_STATUS_OK;
  iree_net_message_endpoint_t endpoint = {nullptr, nullptr};

  static void Callback(void* user_data, iree_status_t status,
                       iree_net_message_endpoint_t endpoint) {
    auto* result = static_cast<EndpointReadyResult*>(user_data);
    result->fired = true;
    result->status_code = iree_status_code(status);
    result->endpoint = endpoint;
    iree_status_ignore(status);
  }
};

//===----------------------------------------------------------------------===//
// Factory test base fixture
//===----------------------------------------------------------------------===//

// Base class for all transport factory CTS tests. Parameterized on BackendInfo.
// The factory-level fields in GetParam() drive transport-specific behavior
// (factory creation, address generation, unreachable address creation).
class FactoryTestBase : public ::testing::TestWithParam<BackendInfo> {
 protected:
  void SetUp() override {
    iree_async_proactor_options_t options =
        iree_async_proactor_options_default();
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        options, iree_allocator_system(), &proactor_));

    // Create slab/region/recv_pool. Transports that need registered buffers
    // (e.g., TCP with io_uring) use the pool; transports that don't (e.g.,
    // loopback) pass NULL and the pool is harmlessly unused.
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
        GetParam().create_factory(iree_allocator_system(), &factory_));
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

  //===--------------------------------------------------------------------===//
  // Address delegates
  //===--------------------------------------------------------------------===//

  std::string MakeBindAddress() { return GetParam().make_bind_address(); }

  std::string ResolveConnectAddress(const std::string& bind_address,
                                    iree_net_listener_t* listener) {
    return GetParam().resolve_connect_address(bind_address, listener);
  }

  std::string MakeUnreachableAddress() {
    return GetParam().make_unreachable_address(proactor_);
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

  //===--------------------------------------------------------------------===//
  // Endpoint lifecycle helpers
  //===--------------------------------------------------------------------===//

  // Deactivates a message endpoint and polls until the callback fires.
  void DeactivateEndpointAndWait(iree_net_message_endpoint_t endpoint) {
    bool deactivated = false;
    iree_status_t status = iree_net_message_endpoint_deactivate(
        endpoint,
        [](void* user_data) { *static_cast<bool*>(user_data) = true; },
        &deactivated);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return;
    }
    PollUntil([&]() { return deactivated; });
  }

  //===--------------------------------------------------------------------===//
  // Listener lifecycle helpers
  //===--------------------------------------------------------------------===//

  // Stops a listener and polls until the stopped callback fires.
  void StopAndWait(iree_net_listener_t* listener) {
    bool stopped = false;
    IREE_ASSERT_OK(iree_net_listener_stop(
        listener,
        {[](void* user_data) { *static_cast<bool*>(user_data) = true; },
         &stopped}));
    ASSERT_TRUE(PollUntil([&]() { return stopped; }))
        << "Listener stop timed out";
  }

  //===--------------------------------------------------------------------===//
  // Connection establishment helpers
  //===--------------------------------------------------------------------===//

  // A connected client/server pair with the listener that produced them.
  struct ConnectPair {
    iree_net_connection_t* client = nullptr;
    iree_net_connection_t* server = nullptr;
    iree_net_listener_t* listener = nullptr;
    std::string connect_address;
  };

  // Creates a listener, connects to it, and returns the resulting pair.
  // Polls until both accept and connect callbacks fire.
  ConnectPair EstablishConnection() {
    ConnectPair pair;
    std::string bind_str = MakeBindAddress();
    iree_string_view_t bind_addr = iree_make_cstring_view(bind_str.c_str());

    struct AcceptCtx {
      iree_net_connection_t* connection = nullptr;
      bool fired = false;
    } accept_ctx;

    IREE_CHECK_OK(iree_net_transport_factory_create_listener(
        factory_, bind_addr, proactor_, recv_pool_,
        [](void* user_data, iree_status_t status,
           iree_net_connection_t* connection) {
          auto* ctx = static_cast<AcceptCtx*>(user_data);
          IREE_CHECK_OK(status);
          ctx->connection = connection;
          ctx->fired = true;
        },
        &accept_ctx, iree_allocator_system(), &pair.listener));

    pair.connect_address = ResolveConnectAddress(bind_str, pair.listener);
    iree_string_view_t connect_addr =
        iree_make_cstring_view(pair.connect_address.c_str());

    struct ConnectCtx {
      iree_net_connection_t* connection = nullptr;
      bool fired = false;
    } connect_ctx;

    IREE_CHECK_OK(iree_net_transport_factory_connect(
        factory_, connect_addr, proactor_, recv_pool_,
        [](void* user_data, iree_status_t status,
           iree_net_connection_t* connection) {
          auto* ctx = static_cast<ConnectCtx*>(user_data);
          IREE_CHECK_OK(status);
          ctx->connection = connection;
          ctx->fired = true;
        },
        &connect_ctx));

    bool ok =
        PollUntil([&]() { return accept_ctx.fired && connect_ctx.fired; });
    EXPECT_TRUE(ok) << "Connection establishment timed out";
    if (ok) {
      pair.client = connect_ctx.connection;
      pair.server = accept_ctx.connection;
    }
    return pair;
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_async_slab_t* slab_ = nullptr;
  iree_async_region_t* region_ = nullptr;
  iree_async_buffer_pool_t* recv_pool_ = nullptr;
  iree_net_transport_factory_t* factory_ = nullptr;
};

}  // namespace iree::net::carrier::cts

#endif  // IREE_NET_CARRIER_CTS_UTIL_FACTORY_TEST_BASE_H_
