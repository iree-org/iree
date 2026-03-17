// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Carrier lifecycle tests: state machine transitions from create to destroy.
//
// These tests verify the fundamental lifecycle contract that all carriers must
// implement: create -> set_recv_handler -> activate -> deactivate -> destroy.

#include <atomic>
#include <thread>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class LifecycleTest : public CarrierTestBase<> {};

// Carrier starts in CREATED state after factory returns.
TEST_P(LifecycleTest, InitialState) {
  EXPECT_EQ(iree_net_carrier_state(client_), IREE_NET_CARRIER_STATE_CREATED);
  EXPECT_EQ(iree_net_carrier_state(server_), IREE_NET_CARRIER_STATE_CREATED);
}

// Activating without setting recv handler should fail.
TEST_P(LifecycleTest, ActivateWithoutHandlerFails) {
  // Don't set recv handler, just try to activate.
  iree_status_t status = iree_net_carrier_activate(client_);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);

  // State should still be CREATED.
  EXPECT_EQ(iree_net_carrier_state(client_), IREE_NET_CARRIER_STATE_CREATED);
}

// Setting recv handler and activating transitions to ACTIVE.
TEST_P(LifecycleTest, ActivateTransitionsToActive) {
  iree_net_carrier_set_recv_handler(client_, MakeNullRecvHandler());
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));
  EXPECT_EQ(iree_net_carrier_state(client_), IREE_NET_CARRIER_STATE_ACTIVE);
}

// Deactivate fires callback and transitions to DEACTIVATED.
TEST_P(LifecycleTest, DeactivateCallbackFires) {
  // Activate first.
  ActivateBothWithNullHandlers();

  // Request deactivation with callback tracking.
  std::atomic<bool> callback_fired{false};
  auto callback = [](void* ud) { *static_cast<std::atomic<bool>*>(ud) = true; };

  IREE_ASSERT_OK(
      iree_net_carrier_deactivate(client_, callback, &callback_fired));

  // Poll until the deactivation callback fires.
  ASSERT_TRUE(PollUntil([&] { return callback_fired.load(); }));
  EXPECT_EQ(iree_net_carrier_state(client_),
            IREE_NET_CARRIER_STATE_DEACTIVATED);
}

// Can deactivate from CREATED state (never activated).
TEST_P(LifecycleTest, DeactivateFromCreatedState) {
  std::atomic<bool> callback_fired{false};
  auto callback = [](void* ud) { *static_cast<std::atomic<bool>*>(ud) = true; };

  // Should succeed even without activation.
  IREE_ASSERT_OK(
      iree_net_carrier_deactivate(client_, callback, &callback_fired));
  EXPECT_TRUE(callback_fired.load());
  EXPECT_EQ(iree_net_carrier_state(client_),
            IREE_NET_CARRIER_STATE_DEACTIVATED);
}

// Double activation fails with FAILED_PRECONDITION.
TEST_P(LifecycleTest, DoubleActivateFails) {
  ActivateBothWithNullHandlers();

  // Second activation on already-active carrier should fail.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_carrier_activate(client_));

  // State should remain ACTIVE (not corrupted by failed second activate).
  EXPECT_EQ(iree_net_carrier_state(client_), IREE_NET_CARRIER_STATE_ACTIVE);
}

// Capabilities are non-zero (carrier reports something).
TEST_P(LifecycleTest, CapabilitiesReported) {
  iree_net_carrier_capabilities_t caps = iree_net_carrier_capabilities(client_);

  // Both carriers in a pair should report the same capabilities.
  EXPECT_EQ(caps, iree_net_carrier_capabilities(server_));

  // Capabilities should be consistent between queries.
  EXPECT_EQ(caps, iree_net_carrier_capabilities(client_));
}

// Multiple carrier lifecycles on a shared proactor. Exercises the drain path
// that caused heap-use-after-free when stale io_uring CQEs from a destroyed
// carrier fired during a subsequent carrier's lifetime.
TEST_P(LifecycleTest, ProactorReuse) {
  // The fixture already created a carrier pair with its own proactor.
  // We'll use the factory to create additional pairs on the same proactor,
  // cycling through create -> activate -> send -> deactivate -> destroy
  // multiple times.
  BackendInfo backend = this->GetParam();

  // Save the proactor from the fixture's pair for reuse.
  iree_async_proactor_t* shared_proactor = proactor_;

  // The fixture pair is still alive. Activate it, send some data to generate
  // io_uring operations, then tear it down via the fixture's normal path.
  std::vector<uint8_t> received;
  RecvCapture server_capture(&received);
  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  uint8_t message[] = {0xDE, 0xAD, 0xBE, 0xEF};
  iree_async_span_t span;
  auto params = MakeSendParams(message, sizeof(message), &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  ASSERT_TRUE(
      PollUntil([&] { return server_capture.total_bytes.load() >= 4; }));

  // Now deactivate the fixture's carriers, replacing handlers first.
  iree_net_carrier_set_recv_handler(client_, MakeNullRecvHandler());
  iree_net_carrier_set_recv_handler(server_, MakeNullRecvHandler());
  DeactivateAndDrain(client_, shared_proactor);
  DeactivateAndDrain(server_, shared_proactor);

  // Release the fixture's carriers.
  iree_net_carrier_release(client_);
  iree_net_carrier_release(server_);
  client_ = nullptr;
  server_ = nullptr;

  // Run pair cleanup if needed (releases TCP listener, buffer pools, etc).
  if (pair_.cleanup) {
    pair_.cleanup(pair_.context);
    pair_.cleanup = nullptr;
    pair_.context = nullptr;
  }

  // Now create a second carrier pair on the SAME proactor. If the first pair's
  // io_uring operations weren't fully drained, stale CQEs will fire here and
  // dereference freed memory (the original bd-hajy bug).
  auto result = backend.factory(shared_proactor);
  if (!result.ok()) {
    // Some backends may not support proactor reuse; skip gracefully.
    GTEST_SKIP() << "Backend does not support proactor reuse: "
                 << result.status().ToString();
  }
  CarrierPair pair2 = std::move(result).value();

  // Run the second pair through a full lifecycle.
  std::vector<uint8_t> received2;
  RecvCapture server_capture2(&received2);
  iree_net_carrier_set_recv_handler(pair2.client, MakeNullRecvHandler());
  iree_net_carrier_set_recv_handler(pair2.server, server_capture2.AsHandler());
  IREE_ASSERT_OK(iree_net_carrier_activate(pair2.client));
  IREE_ASSERT_OK(iree_net_carrier_activate(pair2.server));

  uint8_t message2[] = {0xCA, 0xFE};
  iree_async_span_t span2;
  auto params2 = MakeSendParams(message2, sizeof(message2), &span2);
  IREE_ASSERT_OK(iree_net_carrier_send(pair2.client, &params2));

  // Poll on the shared proactor for the second pair's recv.
  iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
  while (server_capture2.total_bytes.load() < 2) {
    ASSERT_LT(iree_time_now(), deadline) << "Timed out waiting for recv";
    iree_host_size_t completed = 0;
    iree_status_t status = iree_async_proactor_poll(
        shared_proactor, iree_make_timeout_ms(100), &completed);
    if (iree_status_is_deadline_exceeded(status)) {
      iree_status_ignore(status);
    } else {
      IREE_ASSERT_OK(status);
    }
  }
  EXPECT_GE(received2.size(), 2u);

  // Clean up the second pair.
  iree_net_carrier_set_recv_handler(pair2.client, MakeNullRecvHandler());
  iree_net_carrier_set_recv_handler(pair2.server, MakeNullRecvHandler());
  DeactivateAndDrain(pair2.client, shared_proactor);
  DeactivateAndDrain(pair2.server, shared_proactor);
  iree_net_carrier_release(pair2.client);
  iree_net_carrier_release(pair2.server);
  if (pair2.cleanup) {
    pair2.cleanup(pair2.context);
  }

  // Don't release the proactor here — TearDown handles it via proactor_.
  // (We kept proactor_ pointing to it.)
}

// Rapid create-send-teardown cycles on a shared proactor.
// Exercises proactor reuse across multiple carrier lifecycles, verifying that
// deactivation properly drains all pending operations before the carrier is
// destroyed.
TEST_P(LifecycleTest, RapidProactorReuseCycles) {
  BackendInfo backend = this->GetParam();
  iree_async_proactor_t* shared_proactor = proactor_;

  // Tear down the fixture's pair first.
  iree_net_carrier_set_recv_handler(client_, MakeNullRecvHandler());
  iree_net_carrier_set_recv_handler(server_, MakeNullRecvHandler());
  DeactivateAndDrain(client_, shared_proactor);
  DeactivateAndDrain(server_, shared_proactor);
  iree_net_carrier_release(client_);
  iree_net_carrier_release(server_);
  client_ = nullptr;
  server_ = nullptr;
  if (pair_.cleanup) {
    pair_.cleanup(pair_.context);
    pair_.cleanup = nullptr;
    pair_.context = nullptr;
  }

  for (int iteration = 0; iteration < 20; ++iteration) {
    auto result = backend.factory(shared_proactor);
    if (!result.ok()) {
      GTEST_SKIP() << "Backend does not support proactor reuse: "
                   << result.status().ToString();
    }
    CarrierPair pair = std::move(result).value();

    // Activate with null handlers.
    iree_net_carrier_set_recv_handler(pair.client, MakeNullRecvHandler());
    iree_net_carrier_set_recv_handler(pair.server, MakeNullRecvHandler());
    IREE_ASSERT_OK(iree_net_carrier_activate(pair.client));
    IREE_ASSERT_OK(iree_net_carrier_activate(pair.server));

    // Send data from both sides.
    uint8_t payload[] = {0xAA};
    for (int send_round = 0; send_round < 5; ++send_round) {
      iree_async_span_t span_c, span_s;
      auto params_c = MakeSendParams(payload, 1, &span_c);
      auto params_s = MakeSendParams(payload, 1, &span_s);
      IREE_ASSERT_OK(iree_net_carrier_send(pair.client, &params_c));
      IREE_ASSERT_OK(iree_net_carrier_send(pair.server, &params_s));
    }

    // Deactivate and destroy.
    iree_net_carrier_set_recv_handler(pair.client, MakeNullRecvHandler());
    iree_net_carrier_set_recv_handler(pair.server, MakeNullRecvHandler());
    DeactivateAndDrain(pair.client, shared_proactor);
    DeactivateAndDrain(pair.server, shared_proactor);
    ASSERT_EQ(iree_net_carrier_state(pair.client),
              IREE_NET_CARRIER_STATE_DEACTIVATED)
        << "Client drain timed out on iteration " << iteration << " (pending="
        << iree_atomic_load(&pair.client->pending_operations,
                            iree_memory_order_acquire)
        << ")";
    ASSERT_EQ(iree_net_carrier_state(pair.server),
              IREE_NET_CARRIER_STATE_DEACTIVATED)
        << "Server drain timed out on iteration " << iteration << " (pending="
        << iree_atomic_load(&pair.server->pending_operations,
                            iree_memory_order_acquire)
        << ")";
    iree_net_carrier_release(pair.client);
    iree_net_carrier_release(pair.server);
    if (pair.cleanup) {
      pair.cleanup(pair.context);
    }
  }
}

CTS_REGISTER_TEST_SUITE(LifecycleTest);

}  // namespace
}  // namespace iree::net::carrier::cts
