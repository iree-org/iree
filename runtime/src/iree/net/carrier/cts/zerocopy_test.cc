// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Zero-copy send/receive tests for carrier implementations.
//
// Tests carriers that advertise ZERO_COPY_TX and ZERO_COPY_RX capabilities.
// Zero-copy eliminates kernel copies for large transfers, improving performance
// at the cost of stricter buffer lifetime requirements. These tests verify:
//   - Basic functionality: zero-copy sends deliver data correctly
//   - Buffer lifetime: buffers remain valid until completion
//   - Large transfers: zero-copy optimizations work for large data
//   - Scatter-gather: zero-copy works with multiple buffers

#include <cstring>
#include <vector>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class ZeroCopyTest : public CarrierTestBase<> {
 protected:
  void SetUp() override {
    CarrierTestBase::SetUp();
    // Skip tests if carrier lacks zero-copy TX capability.
    if (!HasCapability(IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX)) {
      GTEST_SKIP() << "Carrier lacks ZERO_COPY_TX capability";
    }
  }
};

// Basic zero-copy send delivers data correctly to receiver.
// Verifies that carriers advertising ZERO_COPY_TX capability can send data
// successfully. The underlying implementation may use kernel zero-copy
// mechanisms (io_uring SEND_ZC, sendfile, splice) or fall back to regular
// send if zero-copy is unavailable.
TEST_P(ZeroCopyTest, BasicSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Use a 16KB buffer to ensure zero-copy path is likely (if available).
  // Small sends may use copy path even with zero-copy capability.
  const iree_host_size_t send_length = 16 * 1024;
  std::vector<uint8_t> send_buffer(send_length);
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    send_buffer[i] = static_cast<uint8_t>((i * 7) & 0xFF);
  }

  iree_async_span_t span;
  auto params = MakeSendParams(send_buffer.data(), send_length, &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= send_length; }));

  ASSERT_EQ(server_received.size(), send_length);
  EXPECT_EQ(memcmp(server_received.data(), send_buffer.data(), send_length), 0);
}

// Buffer must remain valid until send completion.
// Zero-copy sends reference the original buffer rather than copying it, so
// the buffer must remain valid and unmodified until the carrier releases it.
// This test uses a persistent buffer that outlives the send operation,
// verifying correct data delivery with large zero-copy eligible transfers.
TEST_P(ZeroCopyTest, BufferLifetimeWithLargeSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Use a large buffer where zero-copy is likely to be used.
  const iree_host_size_t send_length = 32 * 1024;
  std::vector<uint8_t> send_buffer(send_length);
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    send_buffer[i] = static_cast<uint8_t>((i * 13) & 0xFF);
  }

  iree_async_span_t span;
  auto params = MakeSendParams(send_buffer.data(), send_length, &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  // Poll until receive completes.
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= send_length; }));

  // Verify data integrity.
  ASSERT_EQ(server_received.size(), send_length);
  EXPECT_EQ(memcmp(server_received.data(), send_buffer.data(), send_length), 0);
}

// Large send (128KB) works correctly with zero-copy.
// Zero-copy optimizations provide the most benefit for large transfers where
// kernel copy overhead is significant. This test verifies large sends work
// correctly, with proper data integrity across the full transfer.
TEST_P(ZeroCopyTest, LargeSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // 128KB buffer - well above typical zero-copy thresholds.
  const iree_host_size_t send_length = 128 * 1024;
  std::vector<uint8_t> send_buffer(send_length);
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    send_buffer[i] = static_cast<uint8_t>((i * 17) & 0xFF);
  }

  iree_async_span_t span;
  auto params = MakeSendParams(send_buffer.data(), send_length, &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= send_length; }));

  ASSERT_EQ(server_received.size(), send_length);
  EXPECT_EQ(memcmp(server_received.data(), send_buffer.data(), send_length), 0);
}

// Scatter-gather send with multiple buffers works with zero-copy.
// Zero-copy carriers should support scatter-gather operations (sending
// multiple non-contiguous buffers in a single call). This is critical for RPC
// systems that construct messages from headers + payloads without copying
// into a single buffer.
TEST_P(ZeroCopyTest, ScatterGatherSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Three buffers of different sizes: header, payload, trailer.
  std::vector<uint8_t> header(256);
  std::vector<uint8_t> payload(16 * 1024);
  std::vector<uint8_t> trailer(128);

  // Fill with distinct patterns.
  for (size_t i = 0; i < header.size(); ++i) {
    header[i] = static_cast<uint8_t>(0xAA);
  }
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<uint8_t>((i * 23) & 0xFF);
  }
  for (size_t i = 0; i < trailer.size(); ++i) {
    trailer[i] = static_cast<uint8_t>(0xBB);
  }

  // Construct scatter-gather send.
  iree_async_span_t spans[3] = {
      iree_async_span_from_ptr(header.data(), header.size()),
      iree_async_span_from_ptr(payload.data(), payload.size()),
      iree_async_span_from_ptr(trailer.data(), trailer.size()),
  };
  auto params = MakeSendParamsFromSpans(spans, 3);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  const size_t expected_size = header.size() + payload.size() + trailer.size();
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= expected_size; }));

  ASSERT_EQ(server_received.size(), expected_size);

  // Verify each segment arrived correctly.
  size_t offset = 0;
  EXPECT_EQ(memcmp(&server_received[offset], header.data(), header.size()), 0)
      << "Header mismatch";
  offset += header.size();
  EXPECT_EQ(memcmp(&server_received[offset], payload.data(), payload.size()), 0)
      << "Payload mismatch";
  offset += payload.size();
  EXPECT_EQ(memcmp(&server_received[offset], trailer.data(), trailer.size()), 0)
      << "Trailer mismatch";
}

// Multiple sequential sends work correctly with zero-copy.
// Verifies that zero-copy path handles multiple sends without buffer lifetime
// violations or corruption. Each send must complete before its buffer can be
// reused (though sends may overlap if using different buffers).
TEST_P(ZeroCopyTest, MultipleSequentialSends) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const int kSendCount = 5;
  const iree_host_size_t kSendSize = 8 * 1024;
  std::vector<std::vector<uint8_t>> send_buffers(kSendCount);

  // Prepare all buffers with unique patterns.
  for (int i = 0; i < kSendCount; ++i) {
    send_buffers[i].resize(kSendSize);
    for (iree_host_size_t j = 0; j < kSendSize; ++j) {
      send_buffers[i][j] = static_cast<uint8_t>((i * 37 + j * 11) & 0xFF);
    }
  }

  // Send all buffers sequentially.
  for (int i = 0; i < kSendCount; ++i) {
    iree_async_span_t span;
    auto params = MakeSendParams(send_buffers[i].data(), kSendSize, &span);
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  }

  const size_t expected_total = kSendCount * kSendSize;
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= expected_total; }));

  ASSERT_EQ(server_received.size(), expected_total);

  // Verify each send's data arrived in order.
  for (int i = 0; i < kSendCount; ++i) {
    size_t offset = i * kSendSize;
    EXPECT_EQ(
        memcmp(&server_received[offset], send_buffers[i].data(), kSendSize), 0)
        << "Send " << i << " data mismatch";
  }
}

// Bidirectional zero-copy: both client and server send large data.
// Verifies that zero-copy works correctly in both directions simultaneously,
// which is the typical pattern for RPC systems (request + response).
TEST_P(ZeroCopyTest, BidirectionalLargeSends) {
  std::vector<uint8_t> client_received, server_received;
  RecvCapture client_capture(&client_received);
  RecvCapture server_capture(&server_received);

  ActivateBoth(client_capture.AsHandler(), server_capture.AsHandler());

  // Each side sends 32KB with distinct patterns.
  const iree_host_size_t send_length = 32 * 1024;
  std::vector<uint8_t> to_server(send_length);
  std::vector<uint8_t> to_client(send_length);

  for (iree_host_size_t i = 0; i < send_length; ++i) {
    to_server[i] = static_cast<uint8_t>((i * 19) & 0xFF);
    to_client[i] = static_cast<uint8_t>((i * 23) & 0xFF);
  }

  iree_async_span_t span1, span2;
  auto params1 = MakeSendParams(to_server.data(), send_length, &span1);
  auto params2 = MakeSendParams(to_client.data(), send_length, &span2);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params1));
  IREE_ASSERT_OK(iree_net_carrier_send(server_, &params2));

  ASSERT_TRUE(PollUntil([&] {
    return server_capture.total_bytes.load() >= send_length &&
           client_capture.total_bytes.load() >= send_length;
  }));

  // Verify data integrity in both directions.
  ASSERT_EQ(server_received.size(), send_length);
  EXPECT_EQ(memcmp(server_received.data(), to_server.data(), send_length), 0);

  ASSERT_EQ(client_received.size(), send_length);
  EXPECT_EQ(memcmp(client_received.data(), to_client.data(), send_length), 0);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(ZeroCopyTest, {"zerocopy_tx"}, {});

}  // namespace
}  // namespace iree::net::carrier::cts
