// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Loopback-specific carrier tests.
//
// Tests loopback-specific behaviors that cannot be generalized across
// transports. Common carrier tests live in cts/.

#include "iree/net/carrier/loopback/carrier.h"

#include <atomic>
#include <cstring>

#include "iree/async/proactor_platform.h"
#include "iree/net/carrier.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

//===----------------------------------------------------------------------===//
// Test fixture
//===----------------------------------------------------------------------===//

class LoopbackCarrierTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
  }

  void TearDown() override {
    if (client_) {
      DeactivateAndDrain(client_);
      iree_net_carrier_release(client_);
    }
    if (server_) {
      DeactivateAndDrain(server_);
      iree_net_carrier_release(server_);
    }
    if (proactor_) {
      iree_async_proactor_release(proactor_);
    }
  }

  void DeactivateAndDrain(iree_net_carrier_t* carrier) {
    iree_net_carrier_state_t state = iree_net_carrier_state(carrier);
    if (state == IREE_NET_CARRIER_STATE_CREATED ||
        state == IREE_NET_CARRIER_STATE_DEACTIVATED) {
      return;
    }
    if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
      iree_status_t status =
          iree_net_carrier_deactivate(carrier, nullptr, nullptr);
      iree_status_ignore(status);
    }
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(1000);
    while (iree_net_carrier_state(carrier) !=
           IREE_NET_CARRIER_STATE_DEACTIVATED) {
      if (iree_time_now() >= deadline) break;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(10), &completed);
      iree_status_ignore(status);
    }
  }

  void PollUntil(std::function<bool()> condition,
                 iree_duration_t budget = iree_make_duration_ms(5000)) {
    iree_time_t deadline = iree_time_now() + budget;
    while (!condition()) {
      if (iree_time_now() >= deadline) break;
      iree_host_size_t completed = 0;
      iree_status_t status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(10), &completed);
      iree_status_ignore(status);
    }
  }

  static iree_status_t NullRecvHandler(void* /*user_data*/,
                                       iree_async_span_t /*data*/,
                                       iree_async_buffer_lease_t* /*lease*/) {
    return iree_ok_status();
  }

  iree_async_proactor_t* proactor_ = nullptr;
  iree_net_carrier_t* client_ = nullptr;
  iree_net_carrier_t* server_ = nullptr;
};

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

// Loopback delivery is synchronous within the NOP completion callback: the
// sender's NOP fires, checks if the peer is alive, and delivers data inline.
// When the peer departs between send() and the NOP completion, delivery is
// skipped and the send completion callback must report an error.
//
// This is loopback-specific because TCP and SHM have decoupled send/delivery:
// data reaches the transport layer (socket buffer / shared memory ring) before
// the peer link is cleared, so the send genuinely succeeds. The OS handles
// peer departure detection separately for those transports.
TEST_F(LoopbackCarrierTest, InFlightSendReportsErrorOnPeerDeparture) {
  // Track send completions.
  struct CompletionState {
    std::atomic<int> count{0};
    std::atomic<bool> had_error{false};
  } completion;

  iree_net_carrier_callback_t callback = {
      [](void* user_data, uint64_t /*operation_user_data*/,
         iree_status_t status, iree_host_size_t /*bytes_transferred*/,
         iree_async_buffer_lease_t* /*recv_lease*/) {
        auto* state = static_cast<CompletionState*>(user_data);
        if (!iree_status_is_ok(status)) {
          state->had_error.store(true, std::memory_order_release);
        }
        iree_status_ignore(status);
        state->count.fetch_add(1, std::memory_order_release);
      },
      &completion};

  IREE_ASSERT_OK(iree_net_loopback_carrier_create_pair(
      proactor_, callback, iree_allocator_system(), &client_, &server_));

  iree_net_carrier_recv_handler_t null_handler = {NullRecvHandler, nullptr};
  iree_net_carrier_set_recv_handler(client_, null_handler);
  iree_net_carrier_set_recv_handler(server_, null_handler);
  IREE_ASSERT_OK(iree_net_carrier_activate(client_));
  IREE_ASSERT_OK(iree_net_carrier_activate(server_));

  // Send a message (queues a NOP, returns immediately).
  const char* message = "in flight when peer departs";
  iree_async_span_t span =
      iree_async_span_from_ptr(const_cast<char*>(message), strlen(message));
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  // Deactivate the server (peer) BEFORE polling. The NOP is queued but
  // hasn't fired yet. Deactivation clears client->peer, so the NOP
  // completion will see peer==NULL.
  std::atomic<bool> deactivated{false};
  IREE_ASSERT_OK(iree_net_carrier_deactivate(
      server_, [](void* ud) { *static_cast<std::atomic<bool>*>(ud) = true; },
      &deactivated));

  // Poll until the send completion fires.
  PollUntil([&] { return completion.count.load() > 0; });
  ASSERT_GT(completion.count.load(), 0)
      << "Send completion callback never fired";

  // The completion must report an error: the data was never delivered.
  EXPECT_TRUE(completion.had_error.load())
      << "In-flight send should report error when peer departs before "
         "delivery";

  // Ensure deactivation completes before TearDown.
  PollUntil([&] { return deactivated.load(); });
}

}  // namespace
}  // namespace iree
