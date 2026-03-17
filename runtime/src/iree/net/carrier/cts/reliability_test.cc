// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reliability and data integrity tests for carrier implementations.
//
// Tests that reliable carriers deliver data correctly without corruption,
// and that ordered carriers preserve message ordering. These tests are
// capability-gated - carriers without the required capabilities will skip.

#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <vector>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class ReliabilityTest : public CarrierTestBase<> {};

// Large transfer maintains data integrity (reliable carriers only).
TEST_P(ReliabilityTest, LargeTransferIntegrity) {
  if (!iree_any_bit_set(capabilities_, IREE_NET_CARRIER_CAPABILITY_RELIABLE)) {
    GTEST_SKIP() << "backend lacks reliable capability";
  }

  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Generate test pattern: 1MB of sequentially numbered bytes.
  // This is large enough to exercise buffer recycling across multiple receive
  // completions.
  const size_t kSize = 1024 * 1024;
  std::vector<uint8_t> send_data(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    send_data[i] = static_cast<uint8_t>(i & 0xFF);
  }

  // Send in 4KB chunks to exercise multiple sends.
  const size_t kChunkSize = 4096;
  for (size_t offset = 0; offset < kSize; offset += kChunkSize) {
    size_t len = std::min(kChunkSize, kSize - offset);
    iree_async_span_t span = iree_async_span_from_ptr(&send_data[offset], len);
    iree_net_send_params_t params = {};
    params.data.values = &span;
    params.data.count = 1;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
    PollOnce();
  }

  // Wait for all data to arrive.
  ASSERT_TRUE(
      PollUntil([&] { return server_capture.total_bytes.load() >= kSize; },
                iree_make_duration_ms(10000)))
      << "Stalled after " << server_capture.total_bytes.load() << " of "
      << kSize << " bytes (" << (server_capture.total_bytes.load() / 1024)
      << "KB / " << (kSize / 1024) << "KB)";

  // Verify data integrity.
  ASSERT_EQ(server_received.size(), kSize);
  EXPECT_EQ(memcmp(server_received.data(), send_data.data(), kSize), 0)
      << "Data corruption detected in large transfer";
}

// Ordered carriers preserve message ordering.
TEST_P(ReliabilityTest, MessageOrdering) {
  if (!iree_any_bit_set(capabilities_, IREE_NET_CARRIER_CAPABILITY_RELIABLE)) {
    GTEST_SKIP() << "backend lacks reliable capability";
  }
  if (!iree_any_bit_set(capabilities_, IREE_NET_CARRIER_CAPABILITY_ORDERED)) {
    GTEST_SKIP() << "backend lacks ordered capability";
  }

  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Send numbered messages with recognizable patterns.
  // Each message is: [sequence_high, sequence_low, marker_byte].
  //
  // Pre-allocate all message data so each send's span points to stable storage.
  // Send buffers must remain valid until the async send completes — the
  // proactor stores a pointer (sqe->addr) and the kernel reads from it during
  // io_uring_enter, which is deferred until the next poll.
  const int kMessageCount = 100;
  std::vector<std::array<uint8_t, 3>> messages(kMessageCount);
  std::vector<uint8_t> expected;

  for (int i = 0; i < kMessageCount; ++i) {
    messages[i] = {static_cast<uint8_t>((i >> 8) & 0xFF),
                   static_cast<uint8_t>(i & 0xFF), static_cast<uint8_t>(0xAA)};
    expected.insert(expected.end(), messages[i].begin(), messages[i].end());

    iree_async_span_t span =
        iree_async_span_from_ptr(messages[i].data(), messages[i].size());
    iree_net_send_params_t params = {};
    params.data.values = &span;
    params.data.count = 1;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

    // Poll occasionally to avoid exhausting send budget.
    if (i % 10 == 9) {
      PollOnce();
    }
  }

  // Wait for all messages.
  const size_t expected_size = kMessageCount * 3;
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= expected_size; },
      iree_make_duration_ms(10000)));

  ASSERT_EQ(server_received.size(), expected_size);

  // Verify ordering - each message should appear in sequence.
  for (int i = 0; i < kMessageCount; ++i) {
    size_t offset = i * 3;
    int received_seq =
        (server_received[offset] << 8) | server_received[offset + 1];
    EXPECT_EQ(received_seq, i) << "Message " << i << " received out of order";
    EXPECT_EQ(server_received[offset + 2], 0xAA)
        << "Marker corrupted at message " << i;
  }
}

// Bidirectional large transfer (reliable carriers only).
TEST_P(ReliabilityTest, BidirectionalLargeTransfer) {
  if (!iree_any_bit_set(capabilities_, IREE_NET_CARRIER_CAPABILITY_RELIABLE)) {
    GTEST_SKIP() << "backend lacks reliable capability";
  }

  std::vector<uint8_t> client_received, server_received;
  RecvCapture client_capture(&client_received);
  RecvCapture server_capture(&server_received);

  ActivateBoth(client_capture.AsHandler(), server_capture.AsHandler());

  // Generate different patterns for each direction.
  const size_t kSize = 16 * 1024;
  std::vector<uint8_t> to_server(kSize), to_client(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    to_server[i] = static_cast<uint8_t>(i & 0xFF);
    to_client[i] = static_cast<uint8_t>((i + 0x55) & 0xFF);
  }

  // Send from both sides simultaneously.
  const size_t kChunkSize = 4096;
  for (size_t offset = 0; offset < kSize; offset += kChunkSize) {
    size_t len = std::min(kChunkSize, kSize - offset);

    iree_async_span_t span_to_server =
        iree_async_span_from_ptr(&to_server[offset], len);
    iree_net_send_params_t params_to_server = {};
    params_to_server.data.values = &span_to_server;
    params_to_server.data.count = 1;
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params_to_server));

    iree_async_span_t span_to_client =
        iree_async_span_from_ptr(&to_client[offset], len);
    iree_net_send_params_t params_to_client = {};
    params_to_client.data.values = &span_to_client;
    params_to_client.data.count = 1;
    IREE_ASSERT_OK(iree_net_carrier_send(server_, &params_to_client));

    PollOnce();
  }

  // Wait for both sides.
  ASSERT_TRUE(PollUntil(
      [&] {
        return server_capture.total_bytes.load() >= kSize &&
               client_capture.total_bytes.load() >= kSize;
      },
      iree_make_duration_ms(10000)));

  // Verify both directions.
  ASSERT_EQ(server_received.size(), kSize);
  ASSERT_EQ(client_received.size(), kSize);
  EXPECT_EQ(memcmp(server_received.data(), to_server.data(), kSize), 0)
      << "Client-to-server data corrupted";
  EXPECT_EQ(memcmp(client_received.data(), to_client.data(), kSize), 0)
      << "Server-to-client data corrupted";
}

// Small message stress test - many tiny messages (reliable carriers only).
TEST_P(ReliabilityTest, SmallMessageStress) {
  if (!iree_any_bit_set(capabilities_, IREE_NET_CARRIER_CAPABILITY_RELIABLE)) {
    GTEST_SKIP() << "backend lacks reliable capability";
  }

  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Send 1000 single-byte messages. Pre-allocate all data so each send span
  // points to stable storage (send buffers must outlive the async operation).
  const int kMessageCount = 1000;
  std::vector<uint8_t> send_data(kMessageCount);
  for (int i = 0; i < kMessageCount; ++i) {
    send_data[i] = static_cast<uint8_t>(i & 0xFF);
  }
  for (int i = 0; i < kMessageCount; ++i) {
    iree_async_span_t span = iree_async_span_from_ptr(&send_data[i], 1);
    iree_net_send_params_t params = {};
    params.data.values = &span;
    params.data.count = 1;
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

    // Poll frequently to avoid exhausting send budget.
    if (i % 20 == 19) {
      PollOnce();
    }
  }

  // Wait for all bytes.
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= kMessageCount; },
      iree_make_duration_ms(10000)));

  EXPECT_EQ(server_received.size(), static_cast<size_t>(kMessageCount));
}

CTS_REGISTER_TEST_SUITE(ReliabilityTest);

}  // namespace
}  // namespace iree::net::carrier::cts
