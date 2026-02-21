// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for large TCP transfers, concurrent connections, scatter-gather.

#include <cstring>

#if defined(_WIN32)
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

//===----------------------------------------------------------------------===//
// Large transfer tests
//===----------------------------------------------------------------------===//

// Helper to establish a connected client/server pair for data transfer tests.
// Returns true on success; sets |client| and |server| to connected sockets.
class LargeTransferTest : public SocketTestBase<> {
 protected:
  // Fills a buffer with a predictable pattern based on position.
  static void FillPattern(uint8_t* buffer, iree_host_size_t size,
                          iree_host_size_t offset = 0) {
    for (iree_host_size_t i = 0; i < size; ++i) {
      // Use position-dependent pattern: makes corruption easy to locate.
      buffer[i] = static_cast<uint8_t>((offset + i) & 0xFF);
    }
  }

  // Verifies a buffer matches the expected pattern.
  static bool VerifyPattern(const uint8_t* buffer, iree_host_size_t size,
                            iree_host_size_t offset = 0) {
    for (iree_host_size_t i = 0; i < size; ++i) {
      uint8_t expected = static_cast<uint8_t>((offset + i) & 0xFF);
      if (buffer[i] != expected) {
        return false;
      }
    }
    return true;
  }

  // Sends all data, handling partial sends. Polls the proactor as needed.
  // Returns the total bytes sent, or 0 on error.
  iree_host_size_t SendAll(iree_async_socket_t* socket, const uint8_t* data,
                           iree_host_size_t total_size) {
    iree_host_size_t total_sent = 0;
    while (total_sent < total_size) {
      iree_host_size_t remaining = total_size - total_sent;
      iree_async_span_t send_span =
          iree_async_span_from_ptr((void*)(data + total_sent), remaining);

      iree_async_socket_send_operation_t send_op;
      CompletionTracker tracker;
      InitSendOperation(&send_op, socket, &send_span, 1,
                        IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                        CompletionTracker::Callback, &tracker);

      iree_status_t status =
          iree_async_proactor_submit_one(proactor_, &send_op.base);
      if (!iree_status_is_ok(status)) {
        ADD_FAILURE() << "SendAll: submit_one failed at offset " << total_sent
                      << "/" << total_size << ": "
                      << iree::Status(std::move(status)).ToString();
        return 0;
      }

      PollUntil(/*min_completions=*/1,
                /*total_budget=*/iree_make_duration_ms(30000));

      if (!iree_status_is_ok(tracker.last_status)) {
        ADD_FAILURE() << "SendAll: completion error at offset " << total_sent
                      << "/" << total_size << ": "
                      << iree::Status(tracker.ConsumeStatus()).ToString();
        return 0;
      }
      if (send_op.bytes_sent == 0) {
        ADD_FAILURE() << "SendAll: writev returned 0 bytes at offset "
                      << total_sent << "/" << total_size
                      << " (tracker.call_count=" << tracker.call_count << ")";
        return 0;
      }
      total_sent += send_op.bytes_sent;
    }
    return total_sent;
  }

  // Receives all data, handling partial receives. Polls the proactor as needed.
  // Returns the total bytes received, or 0 on error.
  iree_host_size_t RecvAll(iree_async_socket_t* socket, uint8_t* buffer,
                           iree_host_size_t total_size) {
    iree_host_size_t total_received = 0;
    while (total_received < total_size) {
      iree_host_size_t remaining = total_size - total_received;
      iree_async_span_t recv_span =
          iree_async_span_from_ptr(buffer + total_received, remaining);

      iree_async_socket_recv_operation_t recv_op;
      CompletionTracker tracker;
      InitRecvOperation(&recv_op, socket, &recv_span, 1,
                        CompletionTracker::Callback, &tracker);

      iree_status_t status =
          iree_async_proactor_submit_one(proactor_, &recv_op.base);
      if (!iree_status_is_ok(status)) {
        ADD_FAILURE() << "RecvAll: submit_one failed at offset "
                      << total_received << "/" << total_size << ": "
                      << iree::Status(std::move(status)).ToString();
        return 0;
      }

      PollUntil(/*min_completions=*/1,
                /*total_budget=*/iree_make_duration_ms(30000));

      if (!iree_status_is_ok(tracker.last_status)) {
        ADD_FAILURE() << "RecvAll: completion error at offset "
                      << total_received << "/" << total_size << ": "
                      << iree::Status(tracker.ConsumeStatus()).ToString();
        return 0;
      }
      if (recv_op.bytes_received == 0) {
        ADD_FAILURE() << "RecvAll: readv returned 0 bytes (EOF) at offset "
                      << total_received << "/" << total_size;
        return 0;
      }
      total_received += recv_op.bytes_received;
    }
    return total_received;
  }
};

// Send 1MB of data with concurrent recv to prevent TCP flow control deadlock.
//
// Sends and recvs are interleaved: each iteration submits a send for remaining
// data and a recv (if none is in flight), then polls until at least one
// completes. This prevents deadlock when the send buffer fills and sends defer
// to the poll loop for POLLOUT — without a concurrent recv draining the
// receiver, the TCP window closes and POLLOUT never fires.
TEST_P(LargeTransferTest, LargeTransfer_1MB) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
  ASSERT_EQ(connect_tracker.call_count, 1);
  IREE_ASSERT_OK(connect_tracker.ConsumeStatus());
  ASSERT_NE(accept_op.accepted_socket, nullptr);

  iree_async_socket_t* server = accept_op.accepted_socket;

  IREE_ASSERT_OK(iree_async_socket_query_failure(client));
  IREE_ASSERT_OK(iree_async_socket_query_failure(server));

  static constexpr iree_host_size_t kTransferSize = 1024 * 1024;  // 1MB
  std::vector<uint8_t> send_buffer(kTransferSize);
  std::vector<uint8_t> recv_buffer(kTransferSize, 0);

  FillPattern(send_buffer.data(), kTransferSize);

  iree_host_size_t total_sent = 0;
  iree_host_size_t total_received = 0;

  // Operations and trackers are declared outside the loop because deferred
  // operations (EAGAIN → push_pending) remain registered in the proactor's
  // fd_map until their completion callback fires. Stack-local operations that
  // go out of scope while still registered produce dangling pointers and
  // corrupt the fd_map's operation chain.
  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  bool recv_in_flight = false;
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  bool send_in_flight = false;

  while (total_sent < kTransferSize || total_received < kTransferSize) {
    // Submit a recv if none is in flight and we haven't received everything.
    if (!recv_in_flight && total_received < kTransferSize) {
      iree_async_span_t recv_span = iree_async_span_from_ptr(
          recv_buffer.data() + total_received, kTransferSize - total_received);
      recv_tracker.Reset();
      InitRecvOperation(&recv_op, server, &recv_span, 1,
                        CompletionTracker::Callback, &recv_tracker);
      IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));
      recv_in_flight = true;
    }

    // Submit a send if none is in flight and we haven't sent everything.
    if (!send_in_flight && total_sent < kTransferSize) {
      iree_async_span_t send_span = iree_async_span_from_ptr(
          (void*)(send_buffer.data() + total_sent), kTransferSize - total_sent);
      send_tracker.Reset();
      InitSendOperation(&send_op, client, &send_span, 1,
                        IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                        CompletionTracker::Callback, &send_tracker);
      IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
      send_in_flight = true;
    }

    // Poll until at least one operation completes.
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(30000));

    // Process send completion.
    if (send_in_flight && send_tracker.call_count > 0) {
      send_in_flight = false;
      IREE_ASSERT_OK(send_tracker.ConsumeStatus());
      ASSERT_GT(send_op.bytes_sent, static_cast<iree_host_size_t>(0))
          << "writev returned 0 bytes at offset " << total_sent;
      total_sent += send_op.bytes_sent;
    }

    // Process recv completion.
    if (recv_in_flight && recv_tracker.call_count > 0) {
      recv_in_flight = false;
      IREE_ASSERT_OK(recv_tracker.ConsumeStatus());
      ASSERT_GT(recv_op.bytes_received, static_cast<iree_host_size_t>(0))
          << "readv returned 0 bytes (EOF) at offset " << total_received;
      total_received += recv_op.bytes_received;
    }
  }

  ASSERT_EQ(total_sent, kTransferSize) << "Failed to send all data";
  ASSERT_EQ(total_received, kTransferSize) << "Failed to receive all data";

  EXPECT_TRUE(VerifyPattern(recv_buffer.data(), kTransferSize))
      << "Data corruption detected in received buffer";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Verify that sends complete when the socket send buffer is artificially small.
// Sets SO_SNDBUF to the minimum (kernel rounds up to its floor, typically ~2KB
// on Linux) and sends 64KB. On POSIX, the eager writev during submit hits
// EAGAIN when the small buffer fills; the proactor defers to the poll loop for
// POLLOUT-driven retry. On io_uring, the kernel handles this internally. On
// IOCP, WSASend is inherently async. All backends must deliver the data.
TEST_P(LargeTransferTest, LargeTransfer_SmallSendBuffer) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  // Shrink the send buffer to the platform minimum. The kernel rounds up: on
  // Linux the minimum is ~2304 bytes (SOCK_MIN_SNDBUF), on macOS it varies.
  // This ensures that a 64KB send cannot complete in a single writev.
  int send_buffer_size = 1;
#if defined(_WIN32)
  ASSERT_EQ(
      setsockopt(static_cast<SOCKET>(client->primitive.value.win32_handle),
                 SOL_SOCKET, SO_SNDBUF,
                 reinterpret_cast<const char*>(&send_buffer_size),
                 sizeof(send_buffer_size)),
      0)
      << "setsockopt SO_SNDBUF failed";
#else
  ASSERT_EQ(setsockopt(client->primitive.value.fd, SOL_SOCKET, SO_SNDBUF,
                       &send_buffer_size, sizeof(send_buffer_size)),
            0)
      << "setsockopt SO_SNDBUF failed: errno=" << errno;
#endif

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
  ASSERT_EQ(connect_tracker.call_count, 1);
  IREE_ASSERT_OK(connect_tracker.ConsumeStatus());
  ASSERT_NE(accept_op.accepted_socket, nullptr);

  iree_async_socket_t* server = accept_op.accepted_socket;

  // 64KB is large enough to exceed any platform's minimum SO_SNDBUF.
  static constexpr iree_host_size_t kTransferSize = 64 * 1024;
  std::vector<uint8_t> send_buffer(kTransferSize);
  std::vector<uint8_t> recv_buffer(kTransferSize);

  FillPattern(send_buffer.data(), kTransferSize);

  iree_host_size_t bytes_sent =
      SendAll(client, send_buffer.data(), kTransferSize);
  ASSERT_EQ(bytes_sent, kTransferSize) << "Failed to send all data";

  iree_host_size_t bytes_received =
      RecvAll(server, recv_buffer.data(), kTransferSize);
  ASSERT_EQ(bytes_received, kTransferSize) << "Failed to receive all data";

  EXPECT_TRUE(VerifyPattern(recv_buffer.data(), kTransferSize))
      << "Data corruption detected in received buffer";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Send data in multiple smaller chunks, receive all in one stream.
//
// Sends and recvs are interleaved with in-flight tracking. Each send is limited
// to kChunkSize bytes, exercising the proactor's ability to handle many smaller
// operations. Concurrent recvs prevent TCP flow control deadlock when the send
// buffer fills and sends defer to the poll loop for POLLOUT-driven retry.
TEST_P(LargeTransferTest, LargeTransfer_Chunked) {
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  ASSERT_EQ(connect_tracker.call_count, 1);
  ASSERT_NE(accept_op.accepted_socket, nullptr);

  iree_async_socket_t* server = accept_op.accepted_socket;

  static constexpr iree_host_size_t kTotalSize = 256 * 1024;
  static constexpr iree_host_size_t kChunkSize = 16 * 1024;

  std::vector<uint8_t> send_buffer(kTotalSize);
  std::vector<uint8_t> recv_buffer(kTotalSize, 0);

  FillPattern(send_buffer.data(), kTotalSize);

  iree_host_size_t total_sent = 0;
  iree_host_size_t total_received = 0;

  // Operations and trackers are declared outside the loop because deferred
  // operations (EAGAIN -> push_pending) remain registered in the proactor's
  // fd_map until their completion callback fires. Stack-local operations that
  // go out of scope while still registered produce dangling pointers and
  // corrupt the fd_map's operation chain.
  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  bool recv_in_flight = false;
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  bool send_in_flight = false;

  while (total_sent < kTotalSize || total_received < kTotalSize) {
    // Submit a recv if none is in flight and we haven't received everything.
    if (!recv_in_flight && total_received < kTotalSize) {
      iree_async_span_t recv_span = iree_async_span_from_ptr(
          recv_buffer.data() + total_received, kTotalSize - total_received);
      recv_tracker.Reset();
      InitRecvOperation(&recv_op, server, &recv_span, 1,
                        CompletionTracker::Callback, &recv_tracker);
      IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));
      recv_in_flight = true;
    }

    // Submit a send if none is in flight, limiting each operation to
    // kChunkSize bytes to exercise many smaller send operations.
    if (!send_in_flight && total_sent < kTotalSize) {
      iree_host_size_t send_size =
          std::min(kChunkSize, kTotalSize - total_sent);
      iree_async_span_t send_span = iree_async_span_from_ptr(
          (void*)(send_buffer.data() + total_sent), send_size);
      send_tracker.Reset();
      InitSendOperation(&send_op, client, &send_span, 1,
                        IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                        CompletionTracker::Callback, &send_tracker);
      IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));
      send_in_flight = true;
    }

    // Poll until at least one operation completes.
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(30000));

    // Process send completion.
    if (send_in_flight && send_tracker.call_count > 0) {
      send_in_flight = false;
      IREE_ASSERT_OK(send_tracker.ConsumeStatus());
      ASSERT_GT(send_op.bytes_sent, static_cast<iree_host_size_t>(0))
          << "writev returned 0 bytes at offset " << total_sent;
      total_sent += send_op.bytes_sent;
    }

    // Process recv completion.
    if (recv_in_flight && recv_tracker.call_count > 0) {
      recv_in_flight = false;
      IREE_ASSERT_OK(recv_tracker.ConsumeStatus());
      ASSERT_GT(recv_op.bytes_received, static_cast<iree_host_size_t>(0))
          << "readv returned 0 bytes (EOF) at offset " << total_received;
      total_received += recv_op.bytes_received;
    }
  }

  ASSERT_EQ(total_sent, kTotalSize) << "Failed to send all data";
  ASSERT_EQ(total_received, kTotalSize) << "Failed to receive all data";

  // Verify pattern matches.
  EXPECT_TRUE(VerifyPattern(recv_buffer.data(), kTotalSize))
      << "Data corruption detected";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Simultaneous large sends in both directions on the same connection.
TEST_P(LargeTransferTest, LargeTransfer_Bidirectional) {
  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit accept operation.
  iree_async_socket_accept_operation_t accept_op;
  CompletionTracker accept_tracker;
  InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                      &accept_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

  // Create client with TCP_NODELAY.
  iree_async_socket_t* client = nullptr;
  IREE_ASSERT_OK(iree_async_socket_create(proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
                                          IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                          &client));

  iree_async_socket_connect_operation_t connect_op;
  CompletionTracker connect_tracker;
  InitConnectOperation(&connect_op, client, listen_address,
                       CompletionTracker::Callback, &connect_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

  // Poll until connect and accept complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  ASSERT_EQ(accept_tracker.call_count, 1);
  ASSERT_EQ(connect_tracker.call_count, 1);
  ASSERT_NE(accept_op.accepted_socket, nullptr);

  iree_async_socket_t* server = accept_op.accepted_socket;

  // Prepare 128KB transfers in each direction.
  static constexpr iree_host_size_t kTransferSize = 128 * 1024;

  // Client->Server data (pattern starts at 0).
  std::vector<uint8_t> client_send_buffer(kTransferSize);
  std::vector<uint8_t> server_recv_buffer(kTransferSize);
  FillPattern(client_send_buffer.data(), kTransferSize, 0);

  // Server->Client data (pattern starts at 0x80 to be distinct).
  std::vector<uint8_t> server_send_buffer(kTransferSize);
  std::vector<uint8_t> client_recv_buffer(kTransferSize);
  FillPattern(server_send_buffer.data(), kTransferSize, 0x80);

  // Submit sends from both sides concurrently.
  iree_async_span_t client_send_span =
      iree_async_span_from_ptr(client_send_buffer.data(), kTransferSize);
  iree_async_socket_send_operation_t client_send_op;
  CompletionTracker client_send_tracker;
  InitSendOperation(&client_send_op, client, &client_send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &client_send_tracker);

  iree_async_span_t server_send_span =
      iree_async_span_from_ptr(server_send_buffer.data(), kTransferSize);
  iree_async_socket_send_operation_t server_send_op;
  CompletionTracker server_send_tracker;
  InitSendOperation(&server_send_op, server, &server_send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &server_send_tracker);

  // Submit receives on both sides.
  iree_async_span_t client_recv_span =
      iree_async_span_from_ptr(client_recv_buffer.data(), kTransferSize);
  iree_async_socket_recv_operation_t client_recv_op;
  CompletionTracker client_recv_tracker;
  InitRecvOperation(&client_recv_op, client, &client_recv_span, 1,
                    CompletionTracker::Callback, &client_recv_tracker);

  iree_async_span_t server_recv_span =
      iree_async_span_from_ptr(server_recv_buffer.data(), kTransferSize);
  iree_async_socket_recv_operation_t server_recv_op;
  CompletionTracker server_recv_tracker;
  InitRecvOperation(&server_recv_op, server, &server_recv_span, 1,
                    CompletionTracker::Callback, &server_recv_tracker);

  // Submit all four operations.
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &client_send_op.base));
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &server_send_op.base));
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &client_recv_op.base));
  IREE_ASSERT_OK(
      iree_async_proactor_submit_one(proactor_, &server_recv_op.base));

  // This test demonstrates concurrent bidirectional I/O. Due to TCP's
  // streaming nature and potential partial transfers, the first round of
  // completions may not transfer all data. Continue polling and submitting
  // until all data is transferred.
  iree_host_size_t client_total_sent = 0;
  iree_host_size_t server_total_sent = 0;
  iree_host_size_t client_total_recv = 0;
  iree_host_size_t server_total_recv = 0;

  // Track which operations are currently pending to avoid double-counting.
  bool client_send_pending = true;
  bool server_send_pending = true;
  bool client_recv_pending = true;
  bool server_recv_pending = true;

  // Poll until all data transferred.
  while (
      client_total_sent < kTransferSize || server_total_sent < kTransferSize ||
      client_total_recv < kTransferSize || server_total_recv < kTransferSize) {
    int pending_count = 0;
    if (client_send_pending) ++pending_count;
    if (server_send_pending) ++pending_count;
    if (client_recv_pending) ++pending_count;
    if (server_recv_pending) ++pending_count;

    if (pending_count == 0) break;

    PollUntil(/*min_completions=*/pending_count,
              /*total_budget=*/iree_make_duration_ms(30000));

    // Process completed operations and resubmit if needed.
    if (client_send_pending && client_send_tracker.call_count > 0) {
      client_send_pending = false;
      if (iree_status_is_ok(client_send_tracker.last_status)) {
        client_total_sent += client_send_op.bytes_sent;
      }
      // Resubmit if more data to send.
      if (client_total_sent < kTransferSize) {
        client_send_span = iree_async_span_from_ptr(
            client_send_buffer.data() + client_total_sent,
            kTransferSize - client_total_sent);
        client_send_tracker.Reset();
        InitSendOperation(&client_send_op, client, &client_send_span, 1,
                          IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                          CompletionTracker::Callback, &client_send_tracker);
        IREE_ASSERT_OK(
            iree_async_proactor_submit_one(proactor_, &client_send_op.base));
        client_send_pending = true;
      }
    }

    if (server_send_pending && server_send_tracker.call_count > 0) {
      server_send_pending = false;
      if (iree_status_is_ok(server_send_tracker.last_status)) {
        server_total_sent += server_send_op.bytes_sent;
      }
      // Resubmit if more data to send.
      if (server_total_sent < kTransferSize) {
        server_send_span = iree_async_span_from_ptr(
            server_send_buffer.data() + server_total_sent,
            kTransferSize - server_total_sent);
        server_send_tracker.Reset();
        InitSendOperation(&server_send_op, server, &server_send_span, 1,
                          IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                          CompletionTracker::Callback, &server_send_tracker);
        IREE_ASSERT_OK(
            iree_async_proactor_submit_one(proactor_, &server_send_op.base));
        server_send_pending = true;
      }
    }

    if (client_recv_pending && client_recv_tracker.call_count > 0) {
      client_recv_pending = false;
      if (iree_status_is_ok(client_recv_tracker.last_status)) {
        client_total_recv += client_recv_op.bytes_received;
      }
      // Resubmit if more data to receive.
      if (client_total_recv < kTransferSize) {
        client_recv_span = iree_async_span_from_ptr(
            client_recv_buffer.data() + client_total_recv,
            kTransferSize - client_total_recv);
        client_recv_tracker.Reset();
        InitRecvOperation(&client_recv_op, client, &client_recv_span, 1,
                          CompletionTracker::Callback, &client_recv_tracker);
        IREE_ASSERT_OK(
            iree_async_proactor_submit_one(proactor_, &client_recv_op.base));
        client_recv_pending = true;
      }
    }

    if (server_recv_pending && server_recv_tracker.call_count > 0) {
      server_recv_pending = false;
      if (iree_status_is_ok(server_recv_tracker.last_status)) {
        server_total_recv += server_recv_op.bytes_received;
      }
      // Resubmit if more data to receive.
      if (server_total_recv < kTransferSize) {
        server_recv_span = iree_async_span_from_ptr(
            server_recv_buffer.data() + server_total_recv,
            kTransferSize - server_total_recv);
        server_recv_tracker.Reset();
        InitRecvOperation(&server_recv_op, server, &server_recv_span, 1,
                          CompletionTracker::Callback, &server_recv_tracker);
        IREE_ASSERT_OK(
            iree_async_proactor_submit_one(proactor_, &server_recv_op.base));
        server_recv_pending = true;
      }
    }
  }

  // Verify all data transferred.
  EXPECT_EQ(client_total_sent, kTransferSize);
  EXPECT_EQ(server_total_sent, kTransferSize);
  EXPECT_EQ(client_total_recv, kTransferSize);
  EXPECT_EQ(server_total_recv, kTransferSize);

  // Verify patterns match.
  EXPECT_TRUE(VerifyPattern(server_recv_buffer.data(), kTransferSize, 0))
      << "Client->Server data corruption detected";
  EXPECT_TRUE(VerifyPattern(client_recv_buffer.data(), kTransferSize, 0x80))
      << "Server->Client data corruption detected";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Concurrent connections tests
//===----------------------------------------------------------------------===//

// Tests for handling multiple simultaneous connections to the same listener.
class ConcurrentConnectionsTest : public SocketTestBase<> {};

// Accept multiple connections sequentially on the same listener.
TEST_P(ConcurrentConnectionsTest, ConcurrentConnections_Sequential) {
  static constexpr int kNumConnections = 4;

  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Store clients and accepted sockets.
  iree_async_socket_t* clients[kNumConnections] = {};
  iree_async_socket_t* servers[kNumConnections] = {};

  // Accept each connection sequentially.
  for (int i = 0; i < kNumConnections; ++i) {
    // Submit accept operation.
    iree_async_socket_accept_operation_t accept_op;
    CompletionTracker accept_tracker;
    InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                        &accept_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

    // Create client and connect.
    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_NO_DELAY, &clients[i]));

    iree_async_socket_connect_operation_t connect_op;
    CompletionTracker connect_tracker;
    InitConnectOperation(&connect_op, clients[i], listen_address,
                         CompletionTracker::Callback, &connect_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

    // Poll until both complete.
    PollUntil(/*min_completions=*/2,
              /*total_budget=*/iree_make_duration_ms(5000));

    ASSERT_EQ(accept_tracker.call_count, 1);
    IREE_ASSERT_OK(accept_tracker.ConsumeStatus());
    ASSERT_NE(accept_op.accepted_socket, nullptr);

    ASSERT_EQ(connect_tracker.call_count, 1);
    IREE_ASSERT_OK(connect_tracker.ConsumeStatus());

    servers[i] = accept_op.accepted_socket;
  }

  // Verify each connection is independent by sending unique data.
  for (int i = 0; i < kNumConnections; ++i) {
    char send_data[32];
    snprintf(send_data, sizeof(send_data), "Connection %d", i);
    iree_host_size_t send_length = strlen(send_data);

    iree_async_span_t send_span =
        iree_async_span_from_ptr((void*)send_data, send_length);

    iree_async_socket_send_operation_t send_op;
    CompletionTracker send_tracker;
    InitSendOperation(&send_op, clients[i], &send_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

    char recv_buffer[64];
    memset(recv_buffer, 0, sizeof(recv_buffer));
    iree_async_span_t recv_span =
        iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

    iree_async_socket_recv_operation_t recv_op;
    CompletionTracker recv_tracker;
    InitRecvOperation(&recv_op, servers[i], &recv_span, 1,
                      CompletionTracker::Callback, &recv_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

    PollUntil(/*min_completions=*/2,
              /*total_budget=*/iree_make_duration_ms(5000));

    IREE_EXPECT_OK(send_tracker.ConsumeStatus());
    IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
    EXPECT_EQ(recv_op.bytes_received, send_length);
    EXPECT_EQ(memcmp(recv_buffer, send_data, send_length), 0)
        << "Data mismatch on connection " << i;
  }

  // Cleanup.
  for (int i = 0; i < kNumConnections; ++i) {
    iree_async_socket_release(servers[i]);
    iree_async_socket_release(clients[i]);
  }
  iree_async_socket_release(listener);
}

// Submit multiple accept operations and connect multiple clients in parallel.
TEST_P(ConcurrentConnectionsTest, ConcurrentConnections_Parallel) {
  static constexpr int kNumConnections = 4;

  // Create a listener with sufficient backlog.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Submit multiple accept operations upfront.
  iree_async_socket_accept_operation_t accept_ops[kNumConnections];
  CompletionTracker accept_trackers[kNumConnections];

  for (int i = 0; i < kNumConnections; ++i) {
    InitAcceptOperation(&accept_ops[i], listener, CompletionTracker::Callback,
                        &accept_trackers[i]);

    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &accept_ops[i].base));
  }

  // Create and connect all clients.
  iree_async_socket_t* clients[kNumConnections] = {};
  iree_async_socket_connect_operation_t connect_ops[kNumConnections];
  CompletionTracker connect_trackers[kNumConnections];

  for (int i = 0; i < kNumConnections; ++i) {
    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_NO_DELAY, &clients[i]));

    InitConnectOperation(&connect_ops[i], clients[i], listen_address,
                         CompletionTracker::Callback, &connect_trackers[i]);

    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &connect_ops[i].base));
  }

  // Poll until all connections established (accepts + connects).
  PollUntil(/*min_completions=*/kNumConnections * 2,
            /*total_budget=*/iree_make_duration_ms(10000));

  // Verify all accepts succeeded.
  for (int i = 0; i < kNumConnections; ++i) {
    EXPECT_EQ(accept_trackers[i].call_count, 1);
    IREE_EXPECT_OK(accept_trackers[i].ConsumeStatus());
    ASSERT_NE(accept_ops[i].accepted_socket, nullptr)
        << "Accept " << i << " did not produce socket";
  }

  // Verify all connects succeeded.
  for (int i = 0; i < kNumConnections; ++i) {
    EXPECT_EQ(connect_trackers[i].call_count, 1);
    IREE_EXPECT_OK(connect_trackers[i].ConsumeStatus());
  }

  // Send unique data on each connection in parallel.
  char send_data[kNumConnections][32];
  iree_async_span_t send_spans[kNumConnections];
  iree_async_socket_send_operation_t send_ops[kNumConnections];
  CompletionTracker send_trackers[kNumConnections];

  char recv_buffers[kNumConnections][64];
  iree_async_span_t recv_spans[kNumConnections];
  iree_async_socket_recv_operation_t recv_ops[kNumConnections];
  CompletionTracker recv_trackers[kNumConnections];

  for (int i = 0; i < kNumConnections; ++i) {
    snprintf(send_data[i], sizeof(send_data[i]), "Parallel %d", i);
    send_spans[i] =
        iree_async_span_from_ptr((void*)send_data[i], strlen(send_data[i]));

    InitSendOperation(&send_ops[i], clients[i], &send_spans[i], 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &send_trackers[i]);

    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &send_ops[i].base));

    memset(recv_buffers[i], 0, sizeof(recv_buffers[i]));
    recv_spans[i] =
        iree_async_span_from_ptr(recv_buffers[i], sizeof(recv_buffers[i]));

    InitRecvOperation(&recv_ops[i], accept_ops[i].accepted_socket,
                      &recv_spans[i], 1, CompletionTracker::Callback,
                      &recv_trackers[i]);

    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &recv_ops[i].base));
  }

  // Poll until all sends and recvs complete.
  PollUntil(/*min_completions=*/kNumConnections * 2,
            /*total_budget=*/iree_make_duration_ms(10000));

  // Verify all sends and recvs completed.
  for (int i = 0; i < kNumConnections; ++i) {
    IREE_EXPECT_OK(send_trackers[i].ConsumeStatus());
    IREE_EXPECT_OK(recv_trackers[i].ConsumeStatus());
  }

  // Verify data integrity. With parallel accepts, the kernel may complete
  // AcceptEx calls in any order — accept_ops[i].accepted_socket is not
  // necessarily connected to clients[i]. Check that the set of received
  // messages is a permutation of the set of sent messages.
  bool matched[kNumConnections] = {};
  for (int i = 0; i < kNumConnections; ++i) {
    bool found = false;
    for (int j = 0; j < kNumConnections; ++j) {
      if (!matched[j] && recv_ops[i].bytes_received == strlen(send_data[j]) &&
          memcmp(recv_buffers[i], send_data[j], strlen(send_data[j])) == 0) {
        matched[j] = true;
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found) << "Recv " << i << " did not match any expected message";
  }
  for (int j = 0; j < kNumConnections; ++j) {
    EXPECT_TRUE(matched[j]) << "Message '" << send_data[j]
                            << "' was not received by any connection";
  }

  // Cleanup.
  for (int i = 0; i < kNumConnections; ++i) {
    iree_async_socket_release(accept_ops[i].accepted_socket);
    iree_async_socket_release(clients[i]);
  }
  iree_async_socket_release(listener);
}

// Multiple connections with interleaved send/recv across connections.
TEST_P(ConcurrentConnectionsTest, ConcurrentConnections_Interleaved) {
  static constexpr int kNumConnections = 3;
  static constexpr int kNumRounds = 3;

  // Create a listener.
  iree_async_address_t listen_address;
  iree_async_socket_t* listener = CreateListener(&listen_address);

  // Establish all connections.
  iree_async_socket_t* clients[kNumConnections] = {};
  iree_async_socket_t* servers[kNumConnections] = {};

  for (int i = 0; i < kNumConnections; ++i) {
    iree_async_socket_accept_operation_t accept_op;
    CompletionTracker accept_tracker;
    InitAcceptOperation(&accept_op, listener, CompletionTracker::Callback,
                        &accept_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &accept_op.base));

    IREE_ASSERT_OK(iree_async_socket_create(
        proactor_, IREE_ASYNC_SOCKET_TYPE_TCP,
        IREE_ASYNC_SOCKET_OPTION_NO_DELAY, &clients[i]));

    iree_async_socket_connect_operation_t connect_op;
    CompletionTracker connect_tracker;
    InitConnectOperation(&connect_op, clients[i], listen_address,
                         CompletionTracker::Callback, &connect_tracker);

    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &connect_op.base));

    PollUntil(/*min_completions=*/2,
              /*total_budget=*/iree_make_duration_ms(5000));

    ASSERT_NE(accept_op.accepted_socket, nullptr);
    servers[i] = accept_op.accepted_socket;
  }

  // Interleaved communication: each round, all connections send/recv.
  // Different connections are at different stages (some sending, some
  // receiving) to stress the proactor's ability to handle mixed operations.
  for (int round = 0; round < kNumRounds; ++round) {
    // Prepare data buffers for this round.
    char send_data[kNumConnections][64];
    iree_async_span_t send_spans[kNumConnections];
    iree_async_socket_send_operation_t send_ops[kNumConnections];
    CompletionTracker send_trackers[kNumConnections];

    char recv_buffers[kNumConnections][128];
    iree_async_span_t recv_spans[kNumConnections];
    iree_async_socket_recv_operation_t recv_ops[kNumConnections];
    CompletionTracker recv_trackers[kNumConnections];

    // Submit all sends and recvs for this round.
    for (int i = 0; i < kNumConnections; ++i) {
      snprintf(send_data[i], sizeof(send_data[i]), "Round %d Conn %d", round,
               i);
      send_spans[i] =
          iree_async_span_from_ptr((void*)send_data[i], strlen(send_data[i]));

      InitSendOperation(&send_ops[i], clients[i], &send_spans[i], 1,
                        IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                        CompletionTracker::Callback, &send_trackers[i]);

      IREE_ASSERT_OK(
          iree_async_proactor_submit_one(proactor_, &send_ops[i].base));

      memset(recv_buffers[i], 0, sizeof(recv_buffers[i]));
      recv_spans[i] =
          iree_async_span_from_ptr(recv_buffers[i], sizeof(recv_buffers[i]));

      InitRecvOperation(&recv_ops[i], servers[i], &recv_spans[i], 1,
                        CompletionTracker::Callback, &recv_trackers[i]);

      IREE_ASSERT_OK(
          iree_async_proactor_submit_one(proactor_, &recv_ops[i].base));
    }

    // Poll until all complete.
    PollUntil(/*min_completions=*/kNumConnections * 2,
              /*total_budget=*/iree_make_duration_ms(10000));

    // Verify all data.
    for (int i = 0; i < kNumConnections; ++i) {
      IREE_EXPECT_OK(send_trackers[i].ConsumeStatus())
          << "Send failed round=" << round << " conn=" << i;
      IREE_EXPECT_OK(recv_trackers[i].ConsumeStatus())
          << "Recv failed round=" << round << " conn=" << i;

      iree_host_size_t expected_length = strlen(send_data[i]);
      EXPECT_EQ(recv_ops[i].bytes_received, expected_length)
          << "Length mismatch round=" << round << " conn=" << i;
      EXPECT_EQ(memcmp(recv_buffers[i], send_data[i], expected_length), 0)
          << "Data mismatch round=" << round << " conn=" << i;
    }
  }

  // Cleanup.
  for (int i = 0; i < kNumConnections; ++i) {
    iree_async_socket_release(servers[i]);
    iree_async_socket_release(clients[i]);
  }
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Scatter-gather I/O tests
//===----------------------------------------------------------------------===//

// Tests for vectored I/O (scatter-gather) send operations.
class ScatterGatherTest : public SocketTestBase<> {};

// Send from multiple separate buffers in a single operation.
TEST_P(ScatterGatherTest, SendScatter_MultipleBuffers) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Prepare 3 separate send buffers with different content.
  const char* buffer1 = "First buffer. ";
  const char* buffer2 = "Second buffer. ";
  const char* buffer3 = "Third buffer.";
  iree_host_size_t len1 = strlen(buffer1);
  iree_host_size_t len2 = strlen(buffer2);
  iree_host_size_t len3 = strlen(buffer3);
  iree_host_size_t total_length = len1 + len2 + len3;

  // Build scatter-gather list.
  iree_async_span_t spans[3];
  spans[0] = iree_async_span_from_ptr((void*)buffer1, len1);
  spans[1] = iree_async_span_from_ptr((void*)buffer2, len2);
  spans[2] = iree_async_span_from_ptr((void*)buffer3, len3);

  // Send all buffers in one operation.
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, spans, 3,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive into a single buffer.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, total_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, total_length);

  // Verify the received data is the concatenation of all buffers.
  std::string expected = std::string(buffer1) + buffer2 + buffer3;
  EXPECT_EQ(memcmp(recv_buffer, expected.c_str(), total_length), 0)
      << "Expected: " << expected << "\nReceived: " << recv_buffer;

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Include a zero-length buffer in the scatter list.
TEST_P(ScatterGatherTest, ScatterGather_ZeroLengthBuffer) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Three buffers: first has data, second is empty, third has data.
  const char* buffer1 = "Before.";
  const char* buffer3 = "After.";
  iree_host_size_t len1 = strlen(buffer1);
  iree_host_size_t len3 = strlen(buffer3);
  iree_host_size_t total_length = len1 + len3;

  iree_async_span_t spans[3];
  spans[0] = iree_async_span_from_ptr((void*)buffer1, len1);
  spans[1] = iree_async_span_from_ptr(nullptr, 0);  // Zero-length buffer.
  spans[2] = iree_async_span_from_ptr((void*)buffer3, len3);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, spans, 3,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, total_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, total_length);

  // Empty buffer should be skipped, data concatenated.
  std::string expected = std::string(buffer1) + buffer3;
  EXPECT_EQ(memcmp(recv_buffer, expected.c_str(), total_length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Degenerate case: scatter list with count=1.
TEST_P(ScatterGatherTest, ScatterGather_SingleBuffer) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  const char* buffer = "Single buffer test.";
  iree_host_size_t length = strlen(buffer);

  iree_async_span_t span = iree_async_span_from_ptr((void*)buffer, length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  char recv_buffer[128];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, length);

  EXPECT_EQ(memcmp(recv_buffer, buffer, length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Large scatter-gather with many small buffers.
TEST_P(ScatterGatherTest, ScatterGather_ManyBuffers) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Create 8 small buffers.
  static constexpr int kNumBuffers = 8;
  char buffers[kNumBuffers][32];
  iree_async_span_t spans[kNumBuffers];
  iree_host_size_t total_length = 0;

  for (int i = 0; i < kNumBuffers; ++i) {
    snprintf(buffers[i], sizeof(buffers[i]), "[Chunk %d]", i);
    iree_host_size_t len = strlen(buffers[i]);
    spans[i] = iree_async_span_from_ptr(buffers[i], len);
    total_length += len;
  }

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, spans, kNumBuffers,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  char recv_buffer[512];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, total_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, total_length);

  // Build expected string by concatenating all buffers.
  std::string expected;
  for (int i = 0; i < kNumBuffers; ++i) {
    expected += buffers[i];
  }
  EXPECT_EQ(memcmp(recv_buffer, expected.c_str(), total_length), 0)
      << "Data mismatch with many buffers";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Receive into multiple separate buffers in a single operation (scatter recv).
TEST_P(ScatterGatherTest, RecvScatter_MultipleBuffers) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Send a single contiguous message.
  const char* message = "AAAABBBBCCCC";  // 12 bytes: 4 + 4 + 4
  iree_host_size_t message_length = strlen(message);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)message, message_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive into multiple separate buffers (scatter).
  char recv1[4], recv2[4], recv3[4];
  memset(recv1, 0, sizeof(recv1));
  memset(recv2, 0, sizeof(recv2));
  memset(recv3, 0, sizeof(recv3));

  iree_async_span_t recv_spans[3];
  recv_spans[0] = iree_async_span_from_ptr(recv1, sizeof(recv1));
  recv_spans[1] = iree_async_span_from_ptr(recv2, sizeof(recv2));
  recv_spans[2] = iree_async_span_from_ptr(recv3, sizeof(recv3));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, recv_spans, 3,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, message_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, message_length);

  // Verify data was distributed correctly across buffers.
  // Vectored I/O fills buffers in order until each is full.
  EXPECT_EQ(memcmp(recv1, "AAAA", 4), 0) << "First buffer should have 'AAAA'";
  EXPECT_EQ(memcmp(recv2, "BBBB", 4), 0) << "Second buffer should have 'BBBB'";
  EXPECT_EQ(memcmp(recv3, "CCCC", 4), 0) << "Third buffer should have 'CCCC'";

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Bidirectional scatter-gather: scatter send to scatter recv.
TEST_P(ScatterGatherTest, ScatterGather_Bidirectional) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Send with scatter: 3 separate buffers.
  const char* send1 = "PART1";
  const char* send2 = "PART2";
  const char* send3 = "PART3";
  iree_host_size_t total_length = strlen(send1) + strlen(send2) + strlen(send3);

  iree_async_span_t send_spans[3];
  send_spans[0] = iree_async_span_from_ptr((void*)send1, strlen(send1));
  send_spans[1] = iree_async_span_from_ptr((void*)send2, strlen(send2));
  send_spans[2] = iree_async_span_from_ptr((void*)send3, strlen(send3));

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, send_spans, 3,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive with scatter: 3 separate buffers of different sizes.
  char recv1[6], recv2[4], recv3[5];  // 6 + 4 + 5 = 15 total capacity
  memset(recv1, 0, sizeof(recv1));
  memset(recv2, 0, sizeof(recv2));
  memset(recv3, 0, sizeof(recv3));

  iree_async_span_t recv_spans[3];
  recv_spans[0] = iree_async_span_from_ptr(recv1, sizeof(recv1));
  recv_spans[1] = iree_async_span_from_ptr(recv2, sizeof(recv2));
  recv_spans[2] = iree_async_span_from_ptr(recv3, sizeof(recv3));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, recv_spans, 3,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, total_length);

  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, total_length);

  // Data fills recv buffers in order: "PART1PART2PART3" (15 bytes)
  // recv1[6] gets "PART1P"
  // recv2[4] gets "ART2"
  // recv3[5] gets "PART3"
  EXPECT_EQ(memcmp(recv1, "PART1P", 6), 0);
  EXPECT_EQ(memcmp(recv2, "ART2", 4), 0);
  EXPECT_EQ(memcmp(recv3, "PART3", 5), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

CTS_REGISTER_TEST_SUITE(LargeTransferTest);
CTS_REGISTER_TEST_SUITE(ConcurrentConnectionsTest);
CTS_REGISTER_TEST_SUITE(ScatterGatherTest);

}  // namespace iree::async::cts
