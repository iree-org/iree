// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Basic send/receive tests for carrier implementations.
//
// Tests the fundamental data transfer operations that all carriers must
// support: sending data, receiving data, and bidirectional transfer.

#include <cstring>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class SendRecvTest : public CarrierTestBase<> {};

// Client sends to server, server receives the data correctly.
TEST_P(SendRecvTest, BasicSend) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const char* msg = "Hello from client!";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= strlen(msg); }));

  ASSERT_EQ(server_received.size(), strlen(msg));
  EXPECT_EQ(memcmp(server_received.data(), msg, strlen(msg)), 0);
}

// Server sends to client, client receives the data correctly.
TEST_P(SendRecvTest, ReverseDirection) {
  std::vector<uint8_t> client_received;
  RecvCapture client_capture(&client_received);

  ActivateBoth(client_capture.AsHandler(), MakeNullRecvHandler());

  const char* msg = "Hello from server!";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);
  IREE_ASSERT_OK(iree_net_carrier_send(server_, &params));

  ASSERT_TRUE(PollUntil(
      [&] { return client_capture.total_bytes.load() >= strlen(msg); }));

  ASSERT_EQ(client_received.size(), strlen(msg));
  EXPECT_EQ(memcmp(client_received.data(), msg, strlen(msg)), 0);
}

// Both sides send and receive simultaneously.
TEST_P(SendRecvTest, Bidirectional) {
  std::vector<uint8_t> client_received, server_received;
  RecvCapture client_capture(&client_received);
  RecvCapture server_capture(&server_received);

  ActivateBoth(client_capture.AsHandler(), server_capture.AsHandler());

  const char* to_server = "From client";
  const char* to_client = "From server";
  iree_async_span_t span1, span2;

  auto params1 = MakeSendParams(to_server, strlen(to_server), &span1);
  auto params2 = MakeSendParams(to_client, strlen(to_client), &span2);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params1));
  IREE_ASSERT_OK(iree_net_carrier_send(server_, &params2));

  ASSERT_TRUE(PollUntil([&] {
    return server_capture.total_bytes.load() >= strlen(to_server) &&
           client_capture.total_bytes.load() >= strlen(to_client);
  }));

  ASSERT_EQ(server_received.size(), strlen(to_server));
  EXPECT_EQ(memcmp(server_received.data(), to_server, strlen(to_server)), 0);

  ASSERT_EQ(client_received.size(), strlen(to_client));
  EXPECT_EQ(memcmp(client_received.data(), to_client, strlen(to_client)), 0);
}

// Multiple sequential sends accumulate correctly on the receiver.
TEST_P(SendRecvTest, MultipleMessages) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Use a single persistent buffer with all messages concatenated.
  const char* messages = "Msg0Msg1Msg2Msg3Msg4Msg5Msg6Msg7Msg8Msg9";
  const size_t kMsgLen = 4;  // Each "MsgN" is 4 bytes.
  const int kMessageCount = 10;
  const size_t total_bytes = kMsgLen * kMessageCount;

  // Send each message segment.
  for (int i = 0; i < kMessageCount; ++i) {
    iree_async_span_t span = iree_async_span_from_ptr(
        const_cast<char*>(messages + i * kMsgLen), kMsgLen);
    iree_net_send_params_t params = {};
    params.data.values = &span;
    params.data.count = 1;
    params.flags = IREE_NET_SEND_FLAG_NONE;
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));
  }

  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= total_bytes; }));

  EXPECT_EQ(server_received.size(), total_bytes);
  EXPECT_EQ(memcmp(server_received.data(), messages, total_bytes), 0);
}

// Sending before activation fails with FAILED_PRECONDITION.
TEST_P(SendRecvTest, SendBeforeActivateFails) {
  const char* msg = "Should fail";
  iree_async_span_t span;
  auto params = MakeSendParams(msg, strlen(msg), &span);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_carrier_send(client_, &params));
}

// Empty send (zero total bytes) is an error - prevents callback chain hangs.
TEST_P(SendRecvTest, EmptySendFails) {
  ActivateBothWithNullHandlers();

  iree_async_span_t span = iree_async_span_from_ptr(nullptr, 0);
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_carrier_send(client_, &params));
}

// Context for echo recv handler that sends response from inside callback.
struct EchoContext {
  iree_net_carrier_t* carrier;
  std::atomic<int> recv_count{0};
  std::atomic<int> echo_count{0};
  iree_status_t last_send_status = iree_ok_status();

  ~EchoContext() { iree_status_ignore(last_send_status); }
};

static iree_status_t EchoRecvHandler(void* user_data, iree_async_span_t data,
                                     iree_async_buffer_lease_t* lease) {
  auto* ctx = static_cast<EchoContext*>(user_data);
  ctx->recv_count.fetch_add(1, std::memory_order_release);

  // Send response from inside recv handler (the critical RPC pattern).
  // Data must remain valid until the send completion callback fires (standard
  // async I/O contract). Using a string literal (static storage duration).
  iree_async_span_t span = iree_async_span_from_ptr((void*)"ECHO", 4);
  iree_net_send_params_t params = {};
  params.data.values = &span;
  params.data.count = 1;
  params.flags = IREE_NET_SEND_FLAG_NONE;

  ctx->last_send_status = iree_status_ignore(ctx->last_send_status);
  ctx->last_send_status = iree_net_carrier_send(ctx->carrier, &params);
  if (iree_status_is_ok(ctx->last_send_status)) {
    ctx->echo_count.fetch_add(1, std::memory_order_release);
  }
  return iree_ok_status();
}

// Critical RPC pattern: recv handler sends response immediately.
TEST_P(SendRecvTest, SendFromRecvHandler) {
  std::vector<uint8_t> client_received;
  RecvCapture client_capture(&client_received);

  EchoContext server_echo;
  server_echo.carrier = server_;
  iree_net_carrier_recv_handler_t echo_handler = {EchoRecvHandler,
                                                  &server_echo};

  ActivateBoth(client_capture.AsHandler(), echo_handler);

  // Send request from client.
  const char* request = "REQUEST";
  iree_async_span_t span;
  auto params = MakeSendParams(request, strlen(request), &span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  // Wait for echo response.
  ASSERT_TRUE(PollUntil([&] { return server_echo.echo_count.load() >= 1; }));
  ASSERT_TRUE(
      PollUntil([&] { return client_capture.total_bytes.load() >= 4; }));

  // Verify response received.
  ASSERT_GE(client_received.size(), 4u);
  EXPECT_EQ(memcmp(client_received.data(), "ECHO", 4), 0);

  // Verify no send errors occurred.
  IREE_EXPECT_OK(server_echo.last_send_status);
}

// Multiple request-response exchanges (RPC pattern).
// Sends requests one at a time, waiting for each response, to avoid
// overwhelming the send budget when echo responses are generated.
TEST_P(SendRecvTest, MultipleRequestResponse) {
  std::vector<uint8_t> client_received;
  RecvCapture client_capture(&client_received);

  EchoContext server_echo;
  server_echo.carrier = server_;
  iree_net_carrier_recv_handler_t echo_handler = {EchoRecvHandler,
                                                  &server_echo};

  ActivateBoth(client_capture.AsHandler(), echo_handler);

  // Send requests sequentially, waiting for each response.
  // This avoids overwhelming the send budget with echo responses.
  const int kRequestCount = 5;
  for (int i = 0; i < kRequestCount; ++i) {
    size_t bytes_before = client_capture.total_bytes.load();

    const char* request = "REQ";
    iree_async_span_t span;
    auto params = MakeSendParams(request, strlen(request), &span);
    IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

    // Wait for this request's echo response before sending next.
    ASSERT_TRUE(PollUntil([&] {
      return client_capture.total_bytes.load() >= bytes_before + 4;
    })) << "Timeout waiting for echo response "
        << i;
  }

  // Verify all echoes received.
  EXPECT_EQ(server_echo.echo_count.load(), kRequestCount);
  EXPECT_GE(client_received.size(), static_cast<size_t>(kRequestCount * 4));
}

// Scatter-gather send with two spans delivers concatenated data.
TEST_P(SendRecvTest, ScatterGatherTwoSpans) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  const char part_a[] = "Hello, ";
  const char part_b[] = "World!";
  iree_async_span_t spans[2] = {
      iree_async_span_from_ptr(const_cast<char*>(part_a), strlen(part_a)),
      iree_async_span_from_ptr(const_cast<char*>(part_b), strlen(part_b)),
  };
  auto params = MakeSendParamsFromSpans(spans, 2);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  const size_t expected_size = strlen(part_a) + strlen(part_b);
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= expected_size; }));

  ASSERT_EQ(server_received.size(), expected_size);
  EXPECT_EQ(memcmp(server_received.data(), "Hello, World!", expected_size), 0);
}

// Scatter-gather send with maximum supported span count.
TEST_P(SendRecvTest, ScatterGatherMaxSpans) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Cap at 8 spans regardless of the carrier's reported maximum.
  iree_host_size_t max_iov = iree_net_carrier_max_iov(client_);
  iree_host_size_t span_count = (max_iov > 8) ? 8 : max_iov;
  ASSERT_GT(span_count, 0u);

  // Each segment is a 4-byte label: "S00\0", "S01\0", etc.
  char segments[8][4];
  iree_async_span_t spans[8];
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    snprintf(segments[i], sizeof(segments[i]), "S%02d", (int)i);
    spans[i] = iree_async_span_from_ptr(segments[i], 3);
  }
  auto params = MakeSendParamsFromSpans(spans, span_count);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  const size_t expected_size = span_count * 3;
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= expected_size; }));

  // Verify each segment arrived in order.
  ASSERT_EQ(server_received.size(), expected_size);
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    EXPECT_EQ(memcmp(&server_received[i * 3], segments[i], 3), 0)
        << "Segment " << i << " mismatch";
  }
}

// Scatter-gather with mixed span sizes (1 byte, 4KB, 100 bytes).
TEST_P(SendRecvTest, ScatterGatherMixedSizes) {
  std::vector<uint8_t> server_received;
  RecvCapture server_capture(&server_received);

  ActivateBoth(MakeNullRecvHandler(), server_capture.AsHandler());

  // Three spans of very different sizes.
  uint8_t tiny = 0xAA;
  std::vector<uint8_t> medium(4096);
  for (size_t i = 0; i < medium.size(); ++i) {
    medium[i] = static_cast<uint8_t>(i & 0xFF);
  }
  uint8_t tail[100];
  memset(tail, 0xBB, sizeof(tail));

  iree_async_span_t spans[3] = {
      iree_async_span_from_ptr(&tiny, 1),
      iree_async_span_from_ptr(medium.data(), medium.size()),
      iree_async_span_from_ptr(tail, sizeof(tail)),
  };
  auto params = MakeSendParamsFromSpans(spans, 3);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &params));

  const size_t expected_size = 1 + medium.size() + sizeof(tail);
  ASSERT_TRUE(PollUntil(
      [&] { return server_capture.total_bytes.load() >= expected_size; }));

  ASSERT_EQ(server_received.size(), expected_size);

  // Verify each segment.
  size_t offset = 0;
  EXPECT_EQ(server_received[offset], 0xAA) << "Tiny span mismatch";
  offset += 1;
  EXPECT_EQ(memcmp(&server_received[offset], medium.data(), medium.size()), 0)
      << "Medium span mismatch";
  offset += medium.size();
  EXPECT_EQ(memcmp(&server_received[offset], tail, sizeof(tail)), 0)
      << "Tail span mismatch";
}

CTS_REGISTER_TEST_SUITE(SendRecvTest);

}  // namespace
}  // namespace iree::net::carrier::cts
