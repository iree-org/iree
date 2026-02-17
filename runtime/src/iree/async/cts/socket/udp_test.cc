// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for UDP socket operations.
//
// Tests verify both connected UDP (connect + send/recv) and unconnected UDP
// (sendto/recvfrom) patterns for point-to-point and server-style communication.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// UDP tests (connected UDP pattern)
//===----------------------------------------------------------------------===//

// Tests for UDP sockets using the "connected UDP" pattern:
// Both endpoints bind, then connect to each other, enabling regular SEND/RECV.
// This is the recommended pattern for point-to-point UDP channels (e.g., DMA).
class UdpTest : public SocketTestBase<> {
 protected:
  // Creates a pair of connected UDP sockets for bidirectional communication.
  // Both sockets are bound to loopback and connected to each other.
  void CreateConnectedUdpPair(iree_async_socket_t** out_socket_a,
                              iree_async_socket_t** out_socket_b) {
    // Create two UDP sockets.
    IREE_ASSERT_OK(
        iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                 IREE_ASYNC_SOCKET_OPTION_NONE, out_socket_a));
    IREE_ASSERT_OK(
        iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                 IREE_ASYNC_SOCKET_OPTION_NONE, out_socket_b));

    // Bind both to loopback with ephemeral ports.
    iree_async_address_t bind_address;
    IREE_ASSERT_OK(iree_async_address_from_ipv4(
        iree_make_cstring_view("127.0.0.1"), 0, &bind_address));

    IREE_ASSERT_OK(iree_async_socket_bind(*out_socket_a, &bind_address));
    IREE_ASSERT_OK(iree_async_socket_bind(*out_socket_b, &bind_address));

    // Query assigned addresses.
    iree_async_address_t addr_a, addr_b;
    IREE_ASSERT_OK(
        iree_async_socket_query_local_address(*out_socket_a, &addr_a));
    IREE_ASSERT_OK(
        iree_async_socket_query_local_address(*out_socket_b, &addr_b));

    // Connect each socket to the other's address.
    // For UDP, connect() sets the default destination for send().
    iree_async_socket_connect_operation_t connect_a;
    CompletionTracker tracker_a;
    InitConnectOperation(&connect_a, *out_socket_a, addr_b,
                         CompletionTracker::Callback, &tracker_a);

    iree_async_socket_connect_operation_t connect_b;
    CompletionTracker tracker_b;
    InitConnectOperation(&connect_b, *out_socket_b, addr_a,
                         CompletionTracker::Callback, &tracker_b);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_a.base));
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_b.base));

    // UDP connect is typically synchronous, but poll to be safe.
    PollUntil(/*min_completions=*/2,
              /*total_budget=*/iree_make_duration_ms(1000));

    ASSERT_EQ(tracker_a.call_count, 1);
    IREE_ASSERT_OK(tracker_a.ConsumeStatus());
    ASSERT_EQ(tracker_b.call_count, 1);
    IREE_ASSERT_OK(tracker_b.ConsumeStatus());
  }
};

// Basic UDP send/recv using connected sockets.
TEST_P(UdpTest, UDP_ConnectedLoopback) {
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  CreateConnectedUdpPair(&socket_a, &socket_b);

  // Send from A to B.
  const char* message = "UDP datagram test";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, socket_a, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  // Recv on B.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, socket_b, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, message_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, message_length);
  EXPECT_EQ(memcmp(recv_buffer, message, message_length), 0);

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

// UDP preserves message boundaries (unlike TCP stream).
TEST_P(UdpTest, UDP_MessageBoundary) {
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  CreateConnectedUdpPair(&socket_a, &socket_b);

  // Send two separate datagrams.
  const char* msg1 = "First";
  const char* msg2 = "Second";

  iree_async_span_t span1 = iree_async_span_from_ptr((void*)msg1, strlen(msg1));
  iree_async_span_t span2 = iree_async_span_from_ptr((void*)msg2, strlen(msg2));

  iree_async_socket_send_operation_t send1, send2;
  CompletionTracker send1_tracker, send2_tracker;
  InitSendOperation(&send1, socket_a, &span1, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send1_tracker);
  InitSendOperation(&send2, socket_a, &span2, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send2_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send1.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send2.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Each recv should get exactly one datagram, not a merged stream.
  char recv1_buffer[256], recv2_buffer[256];
  memset(recv1_buffer, 0, sizeof(recv1_buffer));
  memset(recv2_buffer, 0, sizeof(recv2_buffer));

  iree_async_span_t recv1_span =
      iree_async_span_from_ptr(recv1_buffer, sizeof(recv1_buffer));
  iree_async_span_t recv2_span =
      iree_async_span_from_ptr(recv2_buffer, sizeof(recv2_buffer));

  iree_async_socket_recv_operation_t recv1, recv2;
  CompletionTracker recv1_tracker, recv2_tracker;
  InitRecvOperation(&recv1, socket_b, &recv1_span, 1,
                    CompletionTracker::Callback, &recv1_tracker);
  InitRecvOperation(&recv2, socket_b, &recv2_span, 1,
                    CompletionTracker::Callback, &recv2_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv1.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv2.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Each recv should get exactly one message (order may vary).
  IREE_EXPECT_OK(recv1_tracker.ConsumeStatus());
  IREE_EXPECT_OK(recv2_tracker.ConsumeStatus());

  // One recv got "First", other got "Second" (order not guaranteed).
  bool recv1_is_first = (recv1.bytes_received == strlen(msg1) &&
                         memcmp(recv1_buffer, msg1, strlen(msg1)) == 0);
  bool recv1_is_second = (recv1.bytes_received == strlen(msg2) &&
                          memcmp(recv1_buffer, msg2, strlen(msg2)) == 0);
  bool recv2_is_first = (recv2.bytes_received == strlen(msg1) &&
                         memcmp(recv2_buffer, msg1, strlen(msg1)) == 0);
  bool recv2_is_second = (recv2.bytes_received == strlen(msg2) &&
                          memcmp(recv2_buffer, msg2, strlen(msg2)) == 0);

  EXPECT_TRUE((recv1_is_first && recv2_is_second) ||
              (recv1_is_second && recv2_is_first))
      << "Expected each recv to get exactly one datagram";

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

// Bidirectional UDP communication.
TEST_P(UdpTest, UDP_Bidirectional) {
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  CreateConnectedUdpPair(&socket_a, &socket_b);

  const char* msg_a_to_b = "Hello from A";
  const char* msg_b_to_a = "Hello from B";

  // Send from both sides simultaneously.
  iree_async_span_t span_a =
      iree_async_span_from_ptr((void*)msg_a_to_b, strlen(msg_a_to_b));
  iree_async_span_t span_b =
      iree_async_span_from_ptr((void*)msg_b_to_a, strlen(msg_b_to_a));

  iree_async_socket_send_operation_t send_a, send_b;
  CompletionTracker send_a_tracker, send_b_tracker;
  InitSendOperation(&send_a, socket_a, &span_a, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_a_tracker);
  InitSendOperation(&send_b, socket_b, &span_b, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_b_tracker);

  char recv_a_buffer[256], recv_b_buffer[256];
  memset(recv_a_buffer, 0, sizeof(recv_a_buffer));
  memset(recv_b_buffer, 0, sizeof(recv_b_buffer));

  iree_async_span_t recv_a_span =
      iree_async_span_from_ptr(recv_a_buffer, sizeof(recv_a_buffer));
  iree_async_span_t recv_b_span =
      iree_async_span_from_ptr(recv_b_buffer, sizeof(recv_b_buffer));

  iree_async_socket_recv_operation_t recv_a, recv_b;
  CompletionTracker recv_a_tracker, recv_b_tracker;
  InitRecvOperation(&recv_a, socket_a, &recv_a_span, 1,
                    CompletionTracker::Callback, &recv_a_tracker);
  InitRecvOperation(&recv_b, socket_b, &recv_b_span, 1,
                    CompletionTracker::Callback, &recv_b_tracker);

  // Submit all four operations.
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_a.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_b.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_a.base));
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_b.base));

  PollUntil(/*min_completions=*/4,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify both directions.
  IREE_EXPECT_OK(send_a_tracker.ConsumeStatus());
  IREE_EXPECT_OK(send_b_tracker.ConsumeStatus());
  IREE_EXPECT_OK(recv_a_tracker.ConsumeStatus());
  IREE_EXPECT_OK(recv_b_tracker.ConsumeStatus());

  EXPECT_EQ(recv_b.bytes_received, strlen(msg_a_to_b));
  EXPECT_EQ(memcmp(recv_b_buffer, msg_a_to_b, strlen(msg_a_to_b)), 0);

  EXPECT_EQ(recv_a.bytes_received, strlen(msg_b_to_a));
  EXPECT_EQ(memcmp(recv_a_buffer, msg_b_to_a, strlen(msg_b_to_a)), 0);

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

// Multiple datagrams in rapid succession.
TEST_P(UdpTest, UDP_BurstTransfer) {
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  CreateConnectedUdpPair(&socket_a, &socket_b);

  static constexpr int kNumDatagrams = 10;
  static constexpr iree_host_size_t kDatagramSize = 512;

  // Send burst of datagrams.
  std::vector<std::vector<uint8_t>> send_buffers(kNumDatagrams);
  std::vector<iree_async_span_t> send_spans(kNumDatagrams);
  std::vector<iree_async_socket_send_operation_t> send_ops(kNumDatagrams);
  std::vector<CompletionTracker> send_trackers(kNumDatagrams);

  for (int i = 0; i < kNumDatagrams; ++i) {
    send_buffers[i].resize(kDatagramSize);
    // Fill with pattern including datagram index.
    for (iree_host_size_t j = 0; j < kDatagramSize; ++j) {
      send_buffers[i][j] = static_cast<uint8_t>((i + j) & 0xFF);
    }
    send_spans[i] =
        iree_async_span_from_ptr(send_buffers[i].data(), kDatagramSize);

    InitSendOperation(&send_ops[i], socket_a, &send_spans[i], 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_trackers[i]);

    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &send_ops[i].base));
  }

  // Receive all datagrams.
  std::vector<std::vector<uint8_t>> recv_buffers(kNumDatagrams);
  std::vector<iree_async_span_t> recv_spans(kNumDatagrams);
  std::vector<iree_async_socket_recv_operation_t> recv_ops(kNumDatagrams);
  std::vector<CompletionTracker> recv_trackers(kNumDatagrams);

  for (int i = 0; i < kNumDatagrams; ++i) {
    recv_buffers[i].resize(kDatagramSize + 64);  // Extra space.
    recv_spans[i] = iree_async_span_from_ptr(recv_buffers[i].data(),
                                             recv_buffers[i].size());

    InitRecvOperation(&recv_ops[i], socket_b, &recv_spans[i], 1,
                      CompletionTracker::Callback, &recv_trackers[i]);

    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &recv_ops[i].base));
  }

  PollUntil(/*min_completions=*/kNumDatagrams * 2,
            /*total_budget=*/iree_make_duration_ms(10000));

  // Verify all sends completed.
  for (int i = 0; i < kNumDatagrams; ++i) {
    IREE_EXPECT_OK(send_trackers[i].ConsumeStatus())
        << "Send " << i << " failed";
    EXPECT_EQ(send_ops[i].bytes_sent, kDatagramSize);
  }

  // Verify all receives completed with correct size.
  // Note: Order may not match send order, but each should be kDatagramSize.
  for (int i = 0; i < kNumDatagrams; ++i) {
    IREE_EXPECT_OK(recv_trackers[i].ConsumeStatus())
        << "Recv " << i << " failed";
    EXPECT_EQ(recv_ops[i].bytes_received, kDatagramSize)
        << "Recv " << i << " got wrong size";
  }

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unconnected UDP tests (sendto/recvfrom pattern)
//===----------------------------------------------------------------------===//

// Unlike connected UDP (which uses connect + send/recv), these tests use
// explicit destination addresses per-send and capture sender addresses on recv.
// This pattern is used for servers handling multiple clients, discovery
// protocols, and multicast/broadcast.
class UnconnectedUdpTest : public SocketTestBase<> {};

// Basic sendto/recvfrom between two unconnected UDP sockets.
TEST_P(UnconnectedUdpTest, SendtoRecvfrom_Basic) {
  // Create two UDP sockets.
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket_a));
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket_b));

  // Bind both to loopback with ephemeral ports.
  iree_async_address_t bind_address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket_a, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket_b, &bind_address));

  // Query assigned addresses.
  iree_async_address_t addr_a, addr_b;
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket_a, &addr_a));
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket_b, &addr_b));

  // Prepare recv on B before send from A (no connection, so we need recvfrom).
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recvfrom_operation_t recvfrom_op;
  CompletionTracker recvfrom_tracker;
  InitRecvfromOperation(&recvfrom_op, socket_b, &recv_span, 1,
                        CompletionTracker::Callback, &recvfrom_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recvfrom_op.base));

  // Send from A to B using sendto with B's address.
  const char* message = "Hello via sendto!";
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_sendto_operation_t sendto_op;
  CompletionTracker sendto_tracker;
  InitSendtoOperation(&sendto_op, socket_a, &send_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE, &addr_b,
                      CompletionTracker::Callback, &sendto_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sendto_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  // Verify send completed.
  IREE_EXPECT_OK(sendto_tracker.ConsumeStatus());
  EXPECT_EQ(sendto_op.bytes_sent, message_length);

  // Verify recv completed with correct data.
  IREE_EXPECT_OK(recvfrom_tracker.ConsumeStatus());
  EXPECT_EQ(recvfrom_op.bytes_received, message_length);
  EXPECT_EQ(memcmp(recv_buffer, message, message_length), 0);

  // Verify sender address matches socket_a's address.
  char addr_a_buf[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  char sender_buf[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t addr_a_str, sender_str;
  IREE_ASSERT_OK(iree_async_address_format(&addr_a, sizeof(addr_a_buf),
                                           addr_a_buf, &addr_a_str));
  IREE_ASSERT_OK(iree_async_address_format(
      &recvfrom_op.sender, sizeof(sender_buf), sender_buf, &sender_str));
  EXPECT_TRUE(iree_string_view_equal(addr_a_str, sender_str))
      << "Expected sender " << std::string(addr_a_str.data, addr_a_str.size)
      << " but got " << std::string(sender_str.data, sender_str.size);

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

// Reply-to-sender pattern: receive a datagram, send response to the sender.
TEST_P(UnconnectedUdpTest, SendtoRecvfrom_ReplyToSender) {
  // Create server and client sockets.
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &server));
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &client));

  // Bind both.
  iree_async_address_t bind_address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(server, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(client, &bind_address));

  iree_async_address_t server_addr, client_addr;
  IREE_ASSERT_OK(iree_async_socket_query_local_address(server, &server_addr));
  IREE_ASSERT_OK(iree_async_socket_query_local_address(client, &client_addr));

  // Server waits for request.
  char server_recv_buffer[256];
  memset(server_recv_buffer, 0, sizeof(server_recv_buffer));
  iree_async_span_t server_recv_span =
      iree_async_span_from_ptr(server_recv_buffer, sizeof(server_recv_buffer));

  iree_async_socket_recvfrom_operation_t server_recvfrom_op;
  CompletionTracker server_recv_tracker;
  InitRecvfromOperation(&server_recvfrom_op, server, &server_recv_span, 1,
                        CompletionTracker::Callback, &server_recv_tracker);

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &server_recvfrom_op.base));

  // Client sends request to server.
  const char* request = "PING";
  iree_async_span_t request_span =
      iree_async_span_from_ptr((void*)request, strlen(request));

  iree_async_socket_sendto_operation_t client_sendto_op;
  CompletionTracker client_send_tracker;
  InitSendtoOperation(&client_sendto_op, client, &request_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE, &server_addr,
                      CompletionTracker::Callback, &client_send_tracker);

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &client_sendto_op.base));

  // Wait for server to receive.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_ASSERT_OK(client_send_tracker.ConsumeStatus());
  IREE_ASSERT_OK(server_recv_tracker.ConsumeStatus());
  EXPECT_EQ(memcmp(server_recv_buffer, request, strlen(request)), 0);

  // Server sends reply to the sender address (client).
  const char* response = "PONG";
  iree_async_span_t response_span =
      iree_async_span_from_ptr((void*)response, strlen(response));

  iree_async_socket_sendto_operation_t server_sendto_op;
  CompletionTracker server_send_tracker;
  InitSendtoOperation(&server_sendto_op, server, &response_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      &server_recvfrom_op.sender,  // Reply to sender!
                      CompletionTracker::Callback, &server_send_tracker);

  // Client waits for reply.
  char client_recv_buffer[256];
  memset(client_recv_buffer, 0, sizeof(client_recv_buffer));
  iree_async_span_t client_recv_span =
      iree_async_span_from_ptr(client_recv_buffer, sizeof(client_recv_buffer));

  iree_async_socket_recvfrom_operation_t client_recvfrom_op;
  CompletionTracker client_recv_tracker;
  InitRecvfromOperation(&client_recvfrom_op, client, &client_recv_span, 1,
                        CompletionTracker::Callback, &client_recv_tracker);

  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &server_sendto_op.base));
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &client_recvfrom_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(server_send_tracker.ConsumeStatus());
  IREE_EXPECT_OK(client_recv_tracker.ConsumeStatus());
  EXPECT_EQ(memcmp(client_recv_buffer, response, strlen(response)), 0);

  // Verify reply came from server.
  char server_buf[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  char reply_sender_buf[IREE_ASYNC_ADDRESS_MAX_FORMAT_LENGTH];
  iree_string_view_t server_str, reply_sender_str;
  IREE_ASSERT_OK(iree_async_address_format(&server_addr, sizeof(server_buf),
                                           server_buf, &server_str));
  IREE_ASSERT_OK(iree_async_address_format(
      &client_recvfrom_op.sender, sizeof(reply_sender_buf), reply_sender_buf,
      &reply_sender_str));
  EXPECT_TRUE(iree_string_view_equal(server_str, reply_sender_str));

  iree_async_socket_release(server);
  iree_async_socket_release(client);
}

// Scatter-gather sendto with multiple buffers.
TEST_P(UnconnectedUdpTest, SendtoRecvfrom_ScatterGather) {
  iree_async_socket_t* socket_a = nullptr;
  iree_async_socket_t* socket_b = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket_a));
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_UDP,
                                          IREE_ASYNC_SOCKET_OPTION_NONE,
                                          &socket_b));

  iree_async_address_t bind_address;
  IREE_ASSERT_OK(iree_async_address_from_ipv4(
      iree_make_cstring_view("127.0.0.1"), 0, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket_a, &bind_address));
  IREE_ASSERT_OK(iree_async_socket_bind(socket_b, &bind_address));

  iree_async_address_t addr_b;
  IREE_ASSERT_OK(iree_async_socket_query_local_address(socket_b, &addr_b));

  // Start recv on B.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recvfrom_operation_t recvfrom_op;
  CompletionTracker recvfrom_tracker;
  InitRecvfromOperation(&recvfrom_op, socket_b, &recv_span, 1,
                        CompletionTracker::Callback, &recvfrom_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recvfrom_op.base));

  // Send from A using scatter-gather (3 buffers).
  const char* part1 = "Hello";
  const char* part2 = " ";
  const char* part3 = "World!";
  iree_async_span_t send_spans[3] = {
      iree_async_span_from_ptr((void*)part1, strlen(part1)),
      iree_async_span_from_ptr((void*)part2, strlen(part2)),
      iree_async_span_from_ptr((void*)part3, strlen(part3)),
  };
  iree_host_size_t total_length = strlen(part1) + strlen(part2) + strlen(part3);

  iree_async_socket_sendto_operation_t sendto_op;
  CompletionTracker sendto_tracker;
  InitSendtoOperation(&sendto_op, socket_a, send_spans, 3,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE, &addr_b,
                      CompletionTracker::Callback, &sendto_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &sendto_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(sendto_tracker.ConsumeStatus());
  EXPECT_EQ(sendto_op.bytes_sent, total_length);

  IREE_EXPECT_OK(recvfrom_tracker.ConsumeStatus());
  EXPECT_EQ(recvfrom_op.bytes_received, total_length);
  EXPECT_EQ(memcmp(recv_buffer, "Hello World!", total_length), 0);

  iree_async_socket_release(socket_a);
  iree_async_socket_release(socket_b);
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(UdpTest);
CTS_REGISTER_TEST_SUITE(UnconnectedUdpTest);

}  // namespace iree::async::cts
