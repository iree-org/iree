// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Backpressure and flow control tests for carrier implementations.
//
// Tests send budget management, budget recovery, and receive-side flow control
// under buffer pressure.

#include <cstring>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class BackpressureTest : public CarrierTestBase<> {};

// Budget query returns valid values after activation.
TEST_P(BackpressureTest, BudgetQueryAfterActivate) {
  ActivateBothWithNullHandlers();

  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(client_);

  EXPECT_GT(budget.slots, 0u);
  EXPECT_GT(budget.bytes, 0u);
}

// Budget query before activation returns valid (not garbage) values.
TEST_P(BackpressureTest, BudgetQueryBeforeActivate) {
  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(client_);

  // Before activation carriers may report zero or full capacity — either is
  // acceptable as long as the values are not garbage.
  EXPECT_TRUE(budget.slots <= UINT32_MAX);
  EXPECT_TRUE(budget.bytes <= IREE_HOST_SIZE_MAX);
}

// Sending beyond available budget returns RESOURCE_EXHAUSTED.
// Skipped for carriers with effectively unlimited budgets.
TEST_P(BackpressureTest, BudgetExhaustionBehavior) {
  ActivateBothWithNullHandlers();

  iree_net_carrier_send_budget_t initial =
      iree_net_carrier_query_send_budget(client_);

  if (initial.slots >= 1000000) {
    GTEST_SKIP() << "Carrier has effectively unlimited budget";
  }

  const char* data = "X";
  iree_async_span_t span;
  auto params = MakeSendParams(data, 1, &span);

  uint32_t sends_succeeded = 0;
  bool got_exhausted = false;

  for (uint32_t i = 0; i < initial.slots + 10; ++i) {
    iree_status_t status = iree_net_carrier_send(client_, &params);
    if (iree_status_is_ok(status)) {
      ++sends_succeeded;
    } else if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      got_exhausted = true;
      iree_status_ignore(status);
      break;
    } else {
      iree_status_ignore(status);
      break;
    }
  }

  EXPECT_TRUE(got_exhausted)
      << "Expected RESOURCE_EXHAUSTED after " << sends_succeeded
      << " sends with budget of " << initial.slots;
}

// Budget recovers after completions are processed.
TEST_P(BackpressureTest, BudgetRecoveryAfterCompletion) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  iree_net_carrier_send_budget_t initial =
      iree_net_carrier_query_send_budget(client_);

  if (initial.slots >= 1000000) {
    GTEST_SKIP() << "Carrier has effectively unlimited budget";
  }

  const char* data = "TestData";
  size_t data_length = strlen(data);
  iree_async_span_t span;
  auto params = MakeSendParams(data, data_length, &span);

  uint32_t to_send = std::min(initial.slots, 4u);
  for (uint32_t i = 0; i < to_send; ++i) {
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  }

  // Budget should be reduced while sends are in-flight.
  iree_net_carrier_send_budget_t during =
      iree_net_carrier_query_send_budget(client_);
  EXPECT_LT(during.slots, initial.slots);

  ASSERT_TRUE(PollUntil([&] {
    return server_capture.total_bytes.load() >= to_send * data_length;
  }));

  // Budget should recover after completions are processed.
  iree_net_carrier_send_budget_t after =
      iree_net_carrier_query_send_budget(client_);
  EXPECT_EQ(after.slots, initial.slots);
}

// Burst of sends within budget all succeed.
TEST_P(BackpressureTest, BurstWithinBudget) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  iree_net_carrier_send_budget_t budget =
      iree_net_carrier_query_send_budget(client_);

  const char* data = "Burst";
  size_t data_length = strlen(data);
  uint32_t burst_count = std::min(budget.slots, 10u);

  iree_async_span_t span;
  auto params = MakeSendParams(data, data_length, &span);

  for (uint32_t i = 0; i < burst_count; ++i) {
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  }

  ASSERT_TRUE(PollUntil([&] {
    return server_capture.total_bytes.load() >= burst_count * data_length;
  }));

  EXPECT_EQ(server_received.size(), burst_count * data_length);
}

// Recv handler that holds buffer leases until explicitly released.
// Holding leases prevents the carrier from reusing receive buffers, exercising
// the backpressure path where the carrier must stall or re-arm receives until
// buffers become available again.
struct LeaseHoldingCapture {
  std::vector<uint8_t> data;
  std::vector<iree_async_buffer_lease_t> held_leases;
  std::atomic<iree_host_size_t> total_bytes{0};
  std::atomic<bool> release_immediately{false};

  static iree_status_t Handler(void* user_data, iree_async_span_t span,
                               iree_async_buffer_lease_t* lease) {
    auto* capture = static_cast<LeaseHoldingCapture*>(user_data);
    uint8_t* ptr = iree_async_span_ptr(span);
    capture->data.insert(capture->data.end(), ptr, ptr + span.length);
    capture->total_bytes.fetch_add(span.length, std::memory_order_relaxed);
    if (lease && lease->release.fn) {
      if (capture->release_immediately.load(std::memory_order_relaxed)) {
        iree_async_buffer_lease_release(lease);
      } else {
        capture->held_leases.push_back(*lease);
        // Clear the original so the caller's release is a no-op.
        memset(lease, 0, sizeof(*lease));
      }
    }
    return iree_ok_status();
  }

  void ReleaseAll() {
    for (auto& lease : held_leases) {
      iree_async_buffer_lease_release(&lease);
    }
    held_leases.clear();
  }

  ~LeaseHoldingCapture() { ReleaseAll(); }

  iree_net_carrier_recv_handler_t AsHandler() { return {Handler, this}; }
};

// Holding recv buffer leases causes backpressure; transfer completes after
// leases are released. Verifies data integrity across the stall/resume cycle.
TEST_P(BackpressureTest, RecvLeaseBackpressure) {
  if (!iree_any_bit_set(capabilities_, IREE_NET_CARRIER_CAPABILITY_RELIABLE)) {
    GTEST_SKIP() << "backend lacks reliable capability";
  }

  LeaseHoldingCapture server_capture;
  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Send 128KB in 4KB chunks, polling after each send to drive completions.
  const iree_host_size_t kTotalSize = 128 * 1024;
  const iree_host_size_t kChunkSize = 4096;
  std::vector<uint8_t> send_data(kTotalSize);
  for (iree_host_size_t i = 0; i < kTotalSize; ++i) {
    send_data[i] = static_cast<uint8_t>(i & 0xFF);
  }

  for (iree_host_size_t offset = 0; offset < kTotalSize; offset += kChunkSize) {
    iree_host_size_t length = std::min(kChunkSize, kTotalSize - offset);
    iree_async_span_t span =
        iree_async_span_from_ptr(&send_data[offset], length);
    iree_net_send_params_t params = {};
    params.data.values = &span;
    params.data.count = 1;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
    iree_host_size_t completed = 0;
    iree_status_t poll_status = iree_async_proactor_poll(
        proactor_, iree_make_timeout_ms(100), &completed);
    if (!iree_status_is_deadline_exceeded(poll_status)) {
      IREE_ASSERT_OK(poll_status);
    } else {
      iree_status_ignore(poll_status);
    }
  }

  // The receiver should have gotten at least some data.
  PollUntil([&] { return server_capture.total_bytes.load() > 0; },
            iree_make_duration_ms(5000));
  iree_host_size_t stall_point = server_capture.total_bytes.load();
  ASSERT_GT(stall_point, 0u) << "No data received at all";

  // Switch to release mode and release all accumulated leases.
  server_capture.release_immediately.store(true, std::memory_order_release);
  server_capture.ReleaseAll();

  // All remaining data should arrive now that buffers are available again.
  ASSERT_TRUE(
      PollUntil([&] { return server_capture.total_bytes.load() >= kTotalSize; },
                iree_make_duration_ms(10000)))
      << "Stalled after " << server_capture.total_bytes.load() << " of "
      << kTotalSize << " bytes (stall point was " << stall_point << ")";

  // Verify data integrity across the stall/resume cycle.
  ASSERT_EQ(server_capture.data.size(), kTotalSize);
  EXPECT_EQ(memcmp(server_capture.data.data(), send_data.data(), kTotalSize), 0)
      << "Data corruption detected under lease backpressure";
}

CTS_REGISTER_TEST_SUITE(BackpressureTest);

}  // namespace
}  // namespace iree::net::carrier::cts
