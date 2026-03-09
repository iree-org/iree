// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for the begin_send/commit_send/abort_send carrier API.
//
// begin_send returns a pointer into a carrier-managed buffer where the caller
// writes directly, then commits (publishes) or aborts (discards). This is the
// zero-allocation path for data being generated (protocol headers, serialized
// frontiers). For pre-existing data at multi-MB scale, send() with
// scatter-gather remains the primary path.

#include <cstring>
#include <thread>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class BeginSendTest : public CarrierTestBase<> {};

// Basic: begin_send, write a known pattern, commit_send. Verify receiver gets
// exact bytes.
TEST_P(BeginSendTest, BasicBeginSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const char* msg = "hello begin_send";
  iree_host_size_t size = strlen(msg);

  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, size, &ptr, &handle));
  ASSERT_NE(ptr, nullptr);

  memcpy(ptr, msg, size);
  IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));

  ASSERT_TRUE(
      PollUntil([&] { return server_capture.total_bytes.load() >= size; }));

  ASSERT_EQ(server_received.size(), size);
  EXPECT_EQ(memcmp(server_received.data(), msg, size), 0);
}

// Multiple sequential begin+write+commit cycles. Verify all received in order.
TEST_P(BeginSendTest, MultipleSequential) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const int kCount = 20;
  iree_host_size_t total_expected = 0;

  for (int i = 0; i < kCount; ++i) {
    uint32_t value = (uint32_t)i;
    iree_host_size_t size = sizeof(value);

    void* ptr = nullptr;
    iree_net_carrier_send_handle_t handle = 0;
    IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, size, &ptr, &handle));
    ASSERT_NE(ptr, nullptr);

    memcpy(ptr, &value, size);
    IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));

    total_expected += size;
  }

  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= total_expected; }));

  ASSERT_EQ(server_received.size(), total_expected);

  // Verify all values arrived in order.
  for (int i = 0; i < kCount; ++i) {
    uint32_t value = 0;
    memcpy(&value, server_received.data() + i * sizeof(uint32_t),
           sizeof(uint32_t));
    EXPECT_EQ(value, (uint32_t)i) << "at index " << i;
  }
}

// begin_send, abort_send. Verify no data received. Then begin+commit succeeds
// normally (abort didn't corrupt state).
TEST_P(BeginSendTest, AbortDoesNotSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Abort a send.
  iree_host_size_t abort_size = 64;
  void* abort_ptr = nullptr;
  iree_net_carrier_send_handle_t abort_handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, abort_size, &abort_ptr,
                                             &abort_handle));
  ASSERT_NE(abort_ptr, nullptr);
  memset(abort_ptr, 0xAA, abort_size);
  iree_net_carrier_abort_send(client_, abort_handle);

  // Now send real data.
  const char* msg = "after abort";
  iree_host_size_t size = strlen(msg);
  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, size, &ptr, &handle));
  ASSERT_NE(ptr, nullptr);
  memcpy(ptr, msg, size);
  IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));

  ASSERT_TRUE(
      PollUntil([&] { return server_capture.total_bytes.load() >= size; }));

  // Should have received only the committed data, not the aborted data.
  ASSERT_EQ(server_received.size(), size);
  EXPECT_EQ(memcmp(server_received.data(), msg, size), 0);
}

// Fill transport until begin_send returns RESOURCE_EXHAUSTED. Drain, verify
// begin_send succeeds again.
TEST_P(BeginSendTest, Backpressure) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Fill the transport with committed sends until exhausted.
  const iree_host_size_t kChunkSize = 256;
  int committed_count = 0;
  iree_host_size_t total_committed = 0;

  for (int i = 0; i < 10000; ++i) {
    void* ptr = nullptr;
    iree_net_carrier_send_handle_t handle = 0;
    iree_status_t status =
        iree_net_carrier_begin_send(client_, kChunkSize, &ptr, &handle);
    if (!iree_status_is_ok(status)) {
      // Should be RESOURCE_EXHAUSTED.
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      break;
    }
    memset(ptr, (uint8_t)(i & 0xFF), kChunkSize);
    IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));
    ++committed_count;
    total_committed += kChunkSize;
  }

  // We should have committed at least one.
  ASSERT_GT(committed_count, 0);

  // Drain the transport so the receiver processes everything.
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= total_committed; }));

  // Now begin_send should succeed again.
  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, 16, &ptr, &handle));
  ASSERT_NE(ptr, nullptr);
  memset(ptr, 0x42, 16);
  IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));
}

// begin_send before activation returns FAILED_PRECONDITION.
TEST_P(BeginSendTest, BeforeActivate) {
  // Don't activate — the carriers are in CREATED state from SetUp().
  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      iree_net_carrier_begin_send(client_, 16, &ptr, &handle));
}

// After shutdown, begin_send must not deliver data to the peer. The carrier can
// reject synchronously (FAILED_PRECONDITION) or defer the error — either way,
// the server must not see the data.
TEST_P(BeginSendTest, AfterShutdown) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  IREE_ASSERT_OK(iree_net_carrier_shutdown(client_));

  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  iree_status_t status =
      iree_net_carrier_begin_send(client_, 16, &ptr, &handle);
  if (!iree_status_is_ok(status)) {
    // Synchronous rejection (e.g. SHM checks shutdown under tx_lock).
    EXPECT_TRUE(iree_status_is_failed_precondition(status));
    iree_status_ignore(status);
  } else {
    // Async carriers (TCP) may accept the reservation but fail at send time.
    // Commit the data and let the proactor handle the failure.
    memset(ptr, 0x42, 16);
    iree_status_t commit_status = iree_net_carrier_commit_send(client_, handle);
    if (!iree_status_is_ok(commit_status)) {
      iree_status_ignore(commit_status);
    } else {
      PollOnce();
    }
  }

  // Drain any pending events.
  PollOnce();

  // The server must not have received the post-shutdown data.
  EXPECT_EQ(server_capture.total_bytes.load(), 0u);
}

// Send same data via begin_send and send(), verify identical bytes received.
// This confirms the two paths produce the same wire format (SHM prepends type
// tags, so the application data must match regardless of send path).
TEST_P(BeginSendTest, EquivalenceWithSend) {
  // Send via begin_send.
  std::vector<uint8_t> begin_send_received;
  RecvCapture begin_send_capture(&begin_send_received);

  ActivateBoth(MakeNullRecvHandler(), begin_send_capture.AsHandler());

  const char* msg = "equivalence test data";
  iree_host_size_t size = strlen(msg);

  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, size, &ptr, &handle));
  memcpy(ptr, msg, size);
  IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));

  ASSERT_TRUE(
      PollUntil([&] { return begin_send_capture.total_bytes.load() >= size; }));

  // Send via regular send().
  std::vector<uint8_t> send_received;
  RecvCapture send_capture(&send_received);

  // Re-create the carrier pair for a clean test.
  // Instead, just send the same data from server to client via send().
  // (Both directions are symmetric for data equivalence.)
  iree_async_span_t span;
  auto params = MakeSendParams(msg, size, &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  ASSERT_TRUE(PollUntil(
      [&] { return begin_send_capture.total_bytes.load() >= size * 2; }));

  // Both sends should produce identical application data on the receiver.
  ASSERT_EQ(begin_send_received.size(), size * 2);
  EXPECT_EQ(memcmp(begin_send_received.data(), msg, size), 0)
      << "begin_send data mismatch";
  EXPECT_EQ(memcmp(begin_send_received.data() + size, msg, size), 0)
      << "send() data mismatch";
}

// begin_send with a large payload (64KB+). Verifies the transport handles
// large contiguous writes.
TEST_P(BeginSendTest, LargePayload) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // 64KB payload with a recognizable pattern.
  const iree_host_size_t kPayloadSize = 65536;
  std::vector<uint8_t> expected(kPayloadSize);
  for (iree_host_size_t i = 0; i < kPayloadSize; ++i) {
    expected[i] = (uint8_t)(i & 0xFF);
  }

  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(
      iree_net_carrier_begin_send(client_, kPayloadSize, &ptr, &handle));
  ASSERT_NE(ptr, nullptr);

  memcpy(ptr, expected.data(), kPayloadSize);
  IREE_ASSERT_OK(iree_net_carrier_commit_send(client_, handle));

  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= kPayloadSize; }));

  ASSERT_EQ(server_received.size(), kPayloadSize);
  EXPECT_EQ(memcmp(server_received.data(), expected.data(), kPayloadSize), 0);
}

// Zero-size begin_send is rejected.
TEST_P(BeginSendTest, ZeroSizeRejected) {
  ActivateBothWithNullHandlers();

  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_begin_send(client_, 0, &ptr, &handle));
}

// Two threads each do N begin+write+commit cycles on the same carrier
// concurrently. Verify all 2N messages are received with correct data.
// Exercises the CAS-based slot ring (loopback/TCP) and MPSC queue (SHM) under
// contention.
TEST_P(BeginSendTest, ConcurrentBeginSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const int kMessagesPerThread = 50;
  const iree_host_size_t kPayloadSize = sizeof(uint32_t);
  const iree_host_size_t kTotalExpected = kPayloadSize * kMessagesPerThread * 2;

  // Each thread writes its thread_id (0 or 1) in the upper 16 bits and the
  // message index in the lower 16 bits, allowing us to verify all messages
  // arrived with correct data.
  std::atomic<bool> stop{false};

  auto sender = [&](uint32_t thread_id) {
    for (int i = 0; i < kMessagesPerThread && !stop.load(); ++i) {
      uint32_t payload = (thread_id << 16) | (uint32_t)i;
      void* ptr = nullptr;
      iree_net_carrier_send_handle_t handle = 0;
      iree_status_t status =
          iree_net_carrier_begin_send(client_, kPayloadSize, &ptr, &handle);
      if (!iree_status_is_ok(status)) {
        // RESOURCE_EXHAUSTED under contention is acceptable — retry.
        iree_status_ignore(status);
        --i;
        continue;
      }
      memcpy(ptr, &payload, sizeof(payload));

      // After commit_send, the buffer is owned by the carrier and must not be
      // accessed. The carrier may complete the send and free the buffer on
      // another thread before commit_send even returns to the caller.
      iree_status_t commit_status =
          iree_net_carrier_commit_send(client_, handle);
      if (!iree_status_is_ok(commit_status)) {
        iree_status_ignore(commit_status);
        --i;
        continue;
      }
    }
  };

  std::thread t0(sender, 0);
  std::thread t1(sender, 1);

  // Poll for delivery while sender threads run.
  bool all_received = PollUntil(
      [&] { return server_capture.total_bytes.load() >= kTotalExpected; });

  stop.store(true);
  t0.join();
  t1.join();
  ASSERT_TRUE(all_received);

  // Verify all messages from both threads are present with correct data.
  ASSERT_EQ(server_received.size(), kTotalExpected);
  bool seen_t0[kMessagesPerThread] = {};
  bool seen_t1[kMessagesPerThread] = {};
  for (size_t offset = 0; offset + kPayloadSize <= server_received.size();
       offset += kPayloadSize) {
    uint32_t payload = 0;
    memcpy(&payload, server_received.data() + offset, sizeof(payload));
    uint32_t thread_id = payload >> 16;
    uint32_t index = payload & 0xFFFF;
    ASSERT_LT(thread_id, 2u) << "invalid thread_id at offset " << offset
                             << ": payload=0x" << std::hex << payload;
    ASSERT_LT(index, (uint32_t)kMessagesPerThread)
        << "invalid index at offset " << offset << ": payload=0x" << std::hex
        << payload;
    if (thread_id == 0) {
      EXPECT_FALSE(seen_t0[index])
          << "duplicate message from t0 index " << index;
      seen_t0[index] = true;
    } else {
      EXPECT_FALSE(seen_t1[index])
          << "duplicate message from t1 index " << index;
      seen_t1[index] = true;
    }
  }
  for (int i = 0; i < kMessagesPerThread; ++i) {
    EXPECT_TRUE(seen_t0[i]) << "missing message from t0 index " << i;
    EXPECT_TRUE(seen_t1[i]) << "missing message from t1 index " << i;
  }
}

// One thread does begin_send+commit_send, another does regular send(),
// concurrently. Verify all messages received correctly.
TEST_P(BeginSendTest, InterleavedWithSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const int kCount = 50;
  const iree_host_size_t kPayloadSize = sizeof(uint32_t);
  const iree_host_size_t kTotalExpected = kPayloadSize * kCount * 2;

  // Thread 0: begin_send path.
  std::atomic<bool> stop{false};
  std::atomic<int> t0_sent{0};
  std::atomic<int> t0_begin_fail{0};
  std::atomic<int> t0_commit_fail{0};
  std::atomic<int> t1_sent{0};
  std::atomic<int> t1_send_fail{0};
  auto begin_send_thread = [&]() {
    for (int i = 0; i < kCount && !stop.load(); ++i) {
      uint32_t payload = (0u << 16) | (uint32_t)i;
      void* ptr = nullptr;
      iree_net_carrier_send_handle_t handle = 0;
      iree_status_t status =
          iree_net_carrier_begin_send(client_, kPayloadSize, &ptr, &handle);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        t0_begin_fail.fetch_add(1);
        --i;
        continue;
      }
      memcpy(ptr, &payload, sizeof(payload));
      iree_status_t commit_status =
          iree_net_carrier_commit_send(client_, handle);
      if (!iree_status_is_ok(commit_status)) {
        iree_status_ignore(commit_status);
        t0_commit_fail.fetch_add(1);
        --i;
        continue;
      }
      t0_sent.fetch_add(1);
    }
  };

  // Thread 1: regular send() path.
  // Pre-allocate payloads in stable storage: send() requires buffer data to
  // survive until the completion callback fires (the proactor may defer SQE
  // processing for cross-thread submitters). Stack-local buffers would be
  // overwritten by the next loop iteration before the kernel reads them.
  std::vector<uint32_t> t1_payloads(kCount);
  for (int i = 0; i < kCount; ++i) {
    t1_payloads[i] = (1u << 16) | (uint32_t)i;
  }
  auto regular_send_thread = [&]() {
    for (int i = 0; i < kCount && !stop.load(); ++i) {
      iree_async_span_t span;
      auto params =
          MakeSendParams(&t1_payloads[i], sizeof(t1_payloads[i]), &span);
      iree_status_t status = iree_net_carrier_send(client_, &params);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        t1_send_fail.fetch_add(1);
        --i;
        continue;
      }
      t1_sent.fetch_add(1);
    }
  };

  std::thread t0(begin_send_thread);
  std::thread t1(regular_send_thread);

  bool all_received = PollUntil(
      [&] { return server_capture.total_bytes.load() >= kTotalExpected; });

  stop.store(true);
  t0.join();
  t1.join();

  if (!all_received) {
    int32_t pending = iree_atomic_load(&client_->pending_operations,
                                       iree_memory_order_seq_cst);
    int64_t bytes_sent =
        iree_atomic_load(&client_->bytes_sent, iree_memory_order_relaxed);
    fprintf(stderr,
            "DIAG: received %" PRIhsz " of %" PRIhsz
            " bytes; t0_sent=%d(begin_fail=%d commit_fail=%d)"
            " t1_sent=%d(send_fail=%d)"
            " pending_ops=%d bytes_sent=%" PRId64 "\n",
            (iree_host_size_t)server_capture.total_bytes.load(), kTotalExpected,
            t0_sent.load(), t0_begin_fail.load(), t0_commit_fail.load(),
            t1_sent.load(), t1_send_fail.load(), pending, bytes_sent);
  }
  ASSERT_TRUE(all_received);

  // Verify all messages from both threads are present.
  ASSERT_EQ(server_received.size(), kTotalExpected);
}

// begin_send on the server side (reverse direction). Verify symmetry.
TEST_P(BeginSendTest, ServerSideBeginSend) {
  std::vector<uint8_t> client_received;
  RecvCapture client_capture(&client_received);

  ActivateBoth(client_capture.AsHandler(), MakeNullRecvHandler());

  const char* msg = "hello from server";
  iree_host_size_t size = strlen(msg);

  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(server_, size, &ptr, &handle));
  ASSERT_NE(ptr, nullptr);

  memcpy(ptr, msg, size);
  IREE_ASSERT_OK(iree_net_carrier_commit_send(server_, handle));

  ASSERT_TRUE(
      PollUntil([&] { return client_capture.total_bytes.load() >= size; }));

  ASSERT_EQ(client_received.size(), size);
  EXPECT_EQ(memcmp(client_received.data(), msg, size), 0);
}

// begin_send succeeds, then deactivate() is called before commit/abort.
// Deactivation must not complete until the reservation is resolved.
TEST_P(BeginSendTest, DeactivationDuringReservation) {
  ActivateBothWithNullHandlers();

  // Reserve a send buffer.
  void* ptr = nullptr;
  iree_net_carrier_send_handle_t handle = 0;
  IREE_ASSERT_OK(iree_net_carrier_begin_send(client_, 16, &ptr, &handle));
  ASSERT_NE(ptr, nullptr);

  // Begin deactivation. The pending_operations count from begin_send should
  // prevent deactivation from completing immediately.
  std::atomic<bool> deactivation_completed{false};
  IREE_ASSERT_OK(iree_net_carrier_deactivate(
      client_,
      [](void* user_data) {
        static_cast<std::atomic<bool>*>(user_data)->store(true);
      },
      &deactivation_completed));

  // Poll a few times — deactivation should NOT complete yet.
  for (int i = 0; i < 5; ++i) {
    PollOnce();
    EXPECT_FALSE(deactivation_completed.load())
        << "Deactivation completed while reservation was outstanding";
  }

  // Abort the reservation.
  iree_net_carrier_abort_send(client_, handle);

  // Now deactivation should complete.
  ASSERT_TRUE(PollUntil([&] { return deactivation_completed.load(); }));

  // Prevent TearDown from trying to deactivate again (already deactivated).
  // Wait for DEACTIVATED state.
  ASSERT_TRUE(PollUntil([&] {
    return iree_net_carrier_state(client_) ==
           IREE_NET_CARRIER_STATE_DEACTIVATED;
  }));
}

CTS_REGISTER_TEST_SUITE(BeginSendTest);

}  // namespace
}  // namespace iree::net::carrier::cts
