// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for send data lifetime during CQE callback processing.
//
// Validates that data referenced by send SQEs is read by the kernel before
// the submitting function's stack frame unwinds. This catches a class of bugs
// where SQE submission is deferred past the caller's data lifetime.
//
// The test pattern:
//   - recv callback fires on the poll thread (CQE processing)
//   - Callback calls a helper function that creates stack-local send data
//   - Helper submits a send referencing that stack-local data, then returns
//   - The stack frame with the send data is now dead
//   - The receiver verifies the data matches the expected pattern
//
// Without immediate SQE flushing in submit's Phase 4, the kernel reads from
// the dead stack frame, producing corrupted data. With the fix, the SQE is
// flushed during submit (before the helper returns), and the kernel reads the
// data while it's still alive.

#include <cstring>

#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/socket.h"
#include "iree/async/span.h"

namespace iree::async::cts {

// Context passed to the recv callback for the echo-back test.
// The callback uses this to submit a send operation referencing stack-local
// data in a nested function call.
struct EchoBackContext {
  iree_async_proactor_t* proactor;
  iree_async_socket_t* socket;
  CompletionTracker send_tracker;
  bool recv_callback_fired = false;
  iree_status_t recv_status = iree_ok_status();

  // The send operation must outlive the submit call (completion references it).
  // Stored here so it's valid until the send completes.
  iree_async_socket_send_operation_t send_op;

  // Pattern byte for the response. The helper function creates stack-local
  // buffers filled with this byte and submits a send referencing them.
  uint8_t pattern;

  // Size of the response payload.
  iree_host_size_t response_size;
};

// Helper function called from the recv callback. Creates stack-local buffers,
// fills them with the pattern, submits a send, and returns — destroying the
// stack frame. This is the critical scenario: the send SQE references data
// on this function's stack, which dies when the function returns.
//
// NOINLINE prevents the compiler from inlining this into the callback, which
// would keep the stack data alive for the callback's lifetime. We need the
// stack frame to actually die on return to exercise the bug.
static IREE_ATTRIBUTE_NOINLINE void submit_echo_response(
    EchoBackContext* context) {
  // Stack-local header (mimics the mux frame header in TCP stream endpoint).
  uint8_t header[16];
  memset(header, context->pattern, sizeof(header));

  // Stack-local body (mimics the control frame payload).
  uint8_t body[128];
  memset(body, context->pattern, sizeof(body));

  // Build scatter-gather spans referencing the stack-local buffers.
  iree_async_span_t spans[2];
  spans[0] = iree_async_span_from_ptr(header, sizeof(header));
  spans[1] = iree_async_span_from_ptr(body, sizeof(body));

  InitSendOperation(&context->send_op, context->socket, spans, 2,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &context->send_tracker);

  iree_status_t status =
      iree_async_proactor_submit_one(context->proactor, &context->send_op.base);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }

  // NOTE: spans[], header[], and body[] die here when this function returns.
  // The send SQE still references them. If the proactor doesn't flush the
  // SQE before this return, the kernel will read garbage from the dead stack.
}

// Recv completion callback. Fires on the poll thread during CQE processing.
// Calls submit_echo_response() which creates stack-local data and submits a
// send referencing it.
static void echo_back_recv_callback(void* user_data,
                                    iree_async_operation_t* operation,
                                    iree_status_t status,
                                    iree_async_completion_flags_t flags) {
  auto* context = static_cast<EchoBackContext*>(user_data);
  context->recv_callback_fired = true;
  context->recv_status = status;

  if (iree_status_is_ok(status)) {
    submit_echo_response(context);
  }

  // submit_echo_response has returned. Its stack-local header[] and body[]
  // are dead. If the send SQE wasn't flushed during submit, the kernel will
  // read garbage when the drain loop eventually flushes it.
}

// Tests the inline send path (no backpressure): when the socket buffer has
// room, inline sends complete during io_uring_enter (or eager writev on POSIX),
// so stack-local data is consumed before the function returns. See the
// DataLifetimeBackpressureTest below for the deferred send path (socket buffer
// full, sends deferred to the poll loop).
class DataLifetimeTest : public SocketTestBase<> {
 protected:
  static constexpr iree_host_size_t kResponseSize = 16 + 128;  // header + body

  // Runs one iteration of the echo-back test:
  //   1. Client sends a trigger byte to the server.
  //   2. Server recv callback fires, submits send-back with stack-local data.
  //   3. Client receives the response and verifies the pattern.
  void RunEchoBack(iree_async_socket_t* client, iree_async_socket_t* server,
                   uint8_t pattern) {
    // Set up the echo-back context for the server's recv callback.
    EchoBackContext echo_context;
    echo_context.proactor = proactor_;
    echo_context.socket = server;
    echo_context.pattern = pattern;
    echo_context.response_size = kResponseSize;

    // Submit server recv with the echo-back callback.
    uint8_t server_recv_buffer[1];
    iree_async_span_t server_recv_span = iree_async_span_from_ptr(
        server_recv_buffer, sizeof(server_recv_buffer));
    iree_async_socket_recv_operation_t server_recv_op;
    InitRecvOperation(&server_recv_op, server, &server_recv_span, 1,
                      echo_back_recv_callback, &echo_context);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &server_recv_op.base));

    // Submit client recv to capture the echo response.
    uint8_t response_buffer[kResponseSize];
    memset(response_buffer, 0, sizeof(response_buffer));
    iree_async_span_t client_recv_span =
        iree_async_span_from_ptr(response_buffer, sizeof(response_buffer));
    iree_async_socket_recv_operation_t client_recv_op;
    CompletionTracker client_recv_tracker;
    InitRecvOperation(&client_recv_op, client, &client_recv_span, 1,
                      CompletionTracker::Callback, &client_recv_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &client_recv_op.base));

    // Send a trigger byte from client to server. This generates a recv CQE
    // on the server side, whose callback submits the echo response.
    uint8_t trigger = pattern;
    iree_async_span_t trigger_span =
        iree_async_span_from_ptr(&trigger, sizeof(trigger));
    iree_async_socket_send_operation_t trigger_send_op;
    CompletionTracker trigger_tracker;
    InitSendOperation(&trigger_send_op, client, &trigger_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &trigger_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &trigger_send_op.base));

    // Poll until the client receives the echo response.
    // Need at minimum: trigger send completion + server recv completion +
    // echo send completion + client recv completion = 4 completions.
    // Use a generous budget since we need the full echo round-trip.
    iree_host_size_t total_received = 0;
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(5000);
    while (total_received < kResponseSize && iree_time_now() < deadline) {
      iree_host_size_t completed = 0;
      iree_status_t poll_status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(100), &completed);
      if (iree_status_is_deadline_exceeded(poll_status)) {
        iree_status_ignore(poll_status);
        continue;
      }
      IREE_ASSERT_OK(poll_status);

      // Check if we've received data on the client side.
      if (client_recv_tracker.call_count > 0) {
        total_received += client_recv_op.bytes_received;

        // TCP may deliver partial data. If we haven't received everything,
        // submit another recv for the remainder.
        if (total_received < kResponseSize) {
          client_recv_tracker.Reset();
          iree_async_span_t remaining_span = iree_async_span_from_ptr(
              response_buffer + total_received, kResponseSize - total_received);
          InitRecvOperation(&client_recv_op, client, &remaining_span, 1,
                            CompletionTracker::Callback, &client_recv_tracker);
          IREE_ASSERT_OK(
              iree_async_proactor_submit_one(proactor_, &client_recv_op.base));
        }
      }
    }

    // Verify the echo-back callback fired.
    ASSERT_TRUE(echo_context.recv_callback_fired)
        << "Server recv callback did not fire";
    IREE_ASSERT_OK(echo_context.recv_status);

    // Verify we received the full response.
    ASSERT_EQ(total_received, kResponseSize)
        << "Expected " << kResponseSize << " bytes but received "
        << total_received;

    // Verify EVERY byte matches the expected pattern. This is the core
    // assertion: if the send SQE referenced dead stack data, the kernel
    // read garbage and the received bytes won't match.
    for (iree_host_size_t i = 0; i < kResponseSize; ++i) {
      ASSERT_EQ(response_buffer[i], pattern)
          << "Data corruption at byte " << i << ": expected 0x" << std::hex
          << (int)pattern << " but got 0x" << (int)response_buffer[i]
          << std::dec
          << ". This indicates the send SQE referenced data that was already "
             "freed (stack-local buffer in submit_echo_response).";
    }
  }
};

// Single echo-back: verifies the basic mechanism works.
TEST_P(DataLifetimeTest, SendFromRecvCallback_StackLocalData) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  RunEchoBack(client, server, 0xAA);

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Repeated echo-backs with different patterns. Each iteration uses a different
// fill byte, making it increasingly unlikely that stale stack data accidentally
// matches the expected pattern.
TEST_P(DataLifetimeTest, RepeatedSendFromRecvCallback_VaryingPatterns) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // 50 iterations with varying patterns. The stack region used by
  // submit_echo_response gets overwritten on each iteration (by the next
  // iteration's recv callback processing), making corruption detectable.
  for (int i = 0; i < 50; ++i) {
    uint8_t pattern = static_cast<uint8_t>(0x01 + i);
    SCOPED_TRACE(::testing::Message() << "iteration " << i << " pattern=0x"
                                      << std::hex << (int)pattern);
    RunEchoBack(client, server, pattern);
  }

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

CTS_REGISTER_TEST_SUITE(DataLifetimeTest);

//===----------------------------------------------------------------------===//
// Backpressure data lifetime tests
//===----------------------------------------------------------------------===//
// The inline-path tests above validate that stack-local data survives the
// submit call. These tests validate the *deferred* send path: when the socket
// buffer is full, the proactor cannot transmit data inline and must defer to
// the poll loop (POLLOUT-driven retry on POSIX, kernel-managed on io_uring,
// inherently async on IOCP). The correct contract is that the caller's buffer
// must remain valid until the completion callback fires.
//
// To exercise this, we shrink SO_SNDBUF to the platform minimum (~2KB on
// Linux) and send 8KB responses — well above what fits in a single writev.
// The response data is stored in the context struct (not on the stack), so it
// remains valid for the full duration of the deferred send. If the proactor
// reads from the wrong address or the caller freed the buffer too early, the
// received data would be corrupted.

// Context for backpressure echo-back tests. Unlike EchoBackContext, the
// response data lives IN the context (not on the stack of a NOINLINE helper),
// because the whole point is that the data must survive until the send
// completion callback fires — which may be arbitrarily delayed under
// backpressure.
struct BackpressureEchoBackContext {
  iree_async_proactor_t* proactor;
  iree_async_socket_t* socket;
  CompletionTracker send_tracker;
  bool recv_callback_fired = false;
  iree_status_t recv_status = iree_ok_status();

  iree_async_socket_send_operation_t send_op;

  uint8_t pattern;
  iree_host_size_t response_size;

  // Response data stored in the context — lives until send completion.
  // This is the critical difference from the stack-local test: the data
  // survives the callback return because the context outlives the send.
  static constexpr iree_host_size_t kMaxResponseSize = 8192;
  uint8_t response_data[kMaxResponseSize];
};

// Fills the context's response_data with the pattern and submits the send.
// Unlike submit_echo_response() above, this does NOT need NOINLINE — the data
// is in the context, not on the stack. The test validates that the proactor
// correctly reads from the context buffer even when the send is deferred.
static void submit_backpressure_echo_response(
    BackpressureEchoBackContext* context) {
  memset(context->response_data, context->pattern, context->response_size);

  iree_async_span_t span =
      iree_async_span_from_ptr(context->response_data, context->response_size);

  InitSendOperation(&context->send_op, context->socket, &span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &context->send_tracker);

  iree_status_t status =
      iree_async_proactor_submit_one(context->proactor, &context->send_op.base);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
  }
}

// Recv completion callback for backpressure tests. Fires on the poll thread
// during CQE processing and submits a large response through a constrained
// send buffer.
static void backpressure_echo_recv_callback(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  auto* context = static_cast<BackpressureEchoBackContext*>(user_data);
  context->recv_callback_fired = true;
  context->recv_status = status;
  if (iree_status_is_ok(status)) {
    submit_backpressure_echo_response(context);
  }
}

class DataLifetimeBackpressureTest : public SocketTestBase<> {
 protected:
  // 8KB response — well above the ~2KB minimum SO_SNDBUF on Linux. This
  // guarantees the send cannot complete in a single writev and must be
  // deferred to the poll loop for POLLOUT-driven retry (on POSIX).
  static constexpr iree_host_size_t kResponseSize = 8192;

  // Shrinks SO_SNDBUF on |socket| to the platform minimum. The kernel rounds
  // the requested value of 1 up to its floor (SOCK_MIN_SNDBUF = ~2304 bytes
  // on Linux, varies on other platforms). This ensures that an 8KB send
  // overflows the buffer and exercises the deferred send path.
  void ShrinkSendBuffer(iree_async_socket_t* socket) {
    int send_buffer_size = 1;
#if defined(IREE_PLATFORM_WINDOWS)
    ASSERT_EQ(
        setsockopt(static_cast<SOCKET>(socket->primitive.value.win32_handle),
                   SOL_SOCKET, SO_SNDBUF,
                   reinterpret_cast<const char*>(&send_buffer_size),
                   sizeof(send_buffer_size)),
        0)
        << "setsockopt SO_SNDBUF failed";
#else
    ASSERT_EQ(setsockopt(socket->primitive.value.fd, SOL_SOCKET, SO_SNDBUF,
                         &send_buffer_size, sizeof(send_buffer_size)),
              0)
        << "setsockopt SO_SNDBUF failed: errno=" << errno;
#endif  // IREE_PLATFORM_WINDOWS
  }

  // Runs one iteration of the backpressure echo-back test:
  //   1. Shrinks server SO_SNDBUF to force backpressure.
  //   2. Client sends a trigger byte to the server.
  //   3. Server recv callback fires, submits 8KB send-back from context data.
  //   4. Client receives the full response (handling partial recvs) and
  //      verifies every byte matches the pattern.
  void RunBackpressureEchoBack(iree_async_socket_t* client,
                               iree_async_socket_t* server, uint8_t pattern) {
    ShrinkSendBuffer(server);

    BackpressureEchoBackContext echo_context;
    echo_context.proactor = proactor_;
    echo_context.socket = server;
    echo_context.pattern = pattern;
    echo_context.response_size = kResponseSize;

    // Submit server recv with the backpressure echo callback.
    uint8_t server_recv_buffer[1];
    iree_async_span_t server_recv_span = iree_async_span_from_ptr(
        server_recv_buffer, sizeof(server_recv_buffer));
    iree_async_socket_recv_operation_t server_recv_op;
    InitRecvOperation(&server_recv_op, server, &server_recv_span, 1,
                      backpressure_echo_recv_callback, &echo_context);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &server_recv_op.base));

    // Submit client recv for the large response.
    uint8_t response_buffer[kResponseSize];
    memset(response_buffer, 0, sizeof(response_buffer));
    iree_async_span_t client_recv_span =
        iree_async_span_from_ptr(response_buffer, kResponseSize);
    iree_async_socket_recv_operation_t client_recv_op;
    CompletionTracker client_recv_tracker;
    InitRecvOperation(&client_recv_op, client, &client_recv_span, 1,
                      CompletionTracker::Callback, &client_recv_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &client_recv_op.base));

    // Send a trigger byte from client to server. This generates a recv CQE
    // on the server side, whose callback submits the backpressure response.
    uint8_t trigger = pattern;
    iree_async_span_t trigger_span =
        iree_async_span_from_ptr(&trigger, sizeof(trigger));
    iree_async_socket_send_operation_t trigger_send_op;
    CompletionTracker trigger_tracker;
    InitSendOperation(&trigger_send_op, client, &trigger_span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &trigger_tracker);
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &trigger_send_op.base));

    // Poll until the client receives the full 8KB response. Backpressure means
    // the server send is split across multiple POLLOUT-driven retries, so this
    // takes more poll cycles than the inline-path test. Use 10s timeout.
    iree_host_size_t total_received = 0;
    iree_time_t deadline = iree_time_now() + iree_make_duration_ms(10000);
    while (total_received < kResponseSize && iree_time_now() < deadline) {
      iree_host_size_t completed = 0;
      iree_status_t poll_status = iree_async_proactor_poll(
          proactor_, iree_make_timeout_ms(100), &completed);
      if (iree_status_is_deadline_exceeded(poll_status)) {
        iree_status_ignore(poll_status);
        continue;
      }
      IREE_ASSERT_OK(poll_status);

      // Check if we've received data on the client side.
      if (client_recv_tracker.call_count > 0) {
        total_received += client_recv_op.bytes_received;

        // TCP may deliver partial data. If we haven't received everything,
        // submit another recv for the remainder.
        if (total_received < kResponseSize) {
          client_recv_tracker.Reset();
          iree_async_span_t remaining_span = iree_async_span_from_ptr(
              response_buffer + total_received, kResponseSize - total_received);
          InitRecvOperation(&client_recv_op, client, &remaining_span, 1,
                            CompletionTracker::Callback, &client_recv_tracker);
          IREE_ASSERT_OK(
              iree_async_proactor_submit_one(proactor_, &client_recv_op.base));
        }
      }
    }

    // Verify the echo-back callback fired.
    ASSERT_TRUE(echo_context.recv_callback_fired)
        << "Server recv callback did not fire";
    IREE_ASSERT_OK(echo_context.recv_status);

    // Verify we received the full response.
    ASSERT_EQ(total_received, kResponseSize)
        << "Expected " << kResponseSize << " bytes but received "
        << total_received;

    // Verify EVERY byte matches the expected pattern. Under backpressure the
    // send is deferred to the poll loop; if the proactor reads from a stale
    // or freed buffer, the received data will be corrupted.
    for (iree_host_size_t i = 0; i < kResponseSize; ++i) {
      ASSERT_EQ(response_buffer[i], pattern)
          << "Data corruption at byte " << i << ": expected 0x" << std::hex
          << (int)pattern << " but got 0x" << (int)response_buffer[i]
          << std::dec
          << ". This indicates the send deferred under backpressure and "
             "read from stale buffer data.";
    }
  }
};

// Single backpressure echo-back: verifies the deferred send path delivers
// correct data when the send buffer is saturated.
TEST_P(DataLifetimeBackpressureTest, BackpressureSend_ContextOwnedData) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  RunBackpressureEchoBack(client, server, 0xBB);

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

// Repeated backpressure echo-backs with different patterns. Each iteration
// fills the response with a distinct byte, making it progressively less likely
// that stale buffer contents accidentally match the expected pattern. Fewer
// iterations than the inline-path test (10 vs 50) because each iteration
// transfers 8KB through a ~2KB send buffer.
TEST_P(DataLifetimeBackpressureTest, RepeatedBackpressureSend_VaryingPatterns) {
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  for (int i = 0; i < 10; ++i) {
    uint8_t pattern = static_cast<uint8_t>(0x10 + i);
    SCOPED_TRACE(::testing::Message() << "iteration " << i << " pattern=0x"
                                      << std::hex << (int)pattern);
    RunBackpressureEchoBack(client, server, pattern);
  }

  iree_async_socket_release(client);
  iree_async_socket_release(server);
  iree_async_socket_release(listener);
}

CTS_REGISTER_TEST_SUITE(DataLifetimeBackpressureTest);

}  // namespace iree::async::cts
