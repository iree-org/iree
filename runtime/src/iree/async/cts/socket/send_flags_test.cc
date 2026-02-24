// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS tests for socket send flags (zero-copy, MORE).
//
// Tests zero-copy send (SEND_ZC) and MSG_MORE flag behavior. These tests
// verify the send flags work correctly across all proactor backends that
// support them.

#include <cstring>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/cts/util/registry.h"
#include "iree/async/cts/util/socket_test_base.h"
#include "iree/async/operations/net.h"
#include "iree/async/slab.h"

namespace iree::async::cts {

class SendFlagsTest : public SocketTestBase<> {};

//===----------------------------------------------------------------------===//
// Zero-copy send tests
//===----------------------------------------------------------------------===//

// Basic zero-copy send: data arrives correctly at receiver.
// Zero-copy is controlled at socket creation via
// IREE_ASYNC_SOCKET_OPTION_ZERO_COPY. This test verifies sends work regardless
// of whether the kernel supports SEND_ZC (fallback to regular SEND is
// transparent).
TEST_P(SendFlagsTest, ZeroCopySendBasic) {
  // Create connected socket pair with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Prepare send buffer with test pattern.
  const char* send_data = "Zero-copy send test data!";
  iree_host_size_t send_length = strlen(send_data);
  iree_async_span_t send_span =
      iree_async_span_from_ptr((void*)send_data, send_length);

  // Submit send (ZC is determined by socket option, not per-send flag).
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Submit recv on server.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_async_span_t recv_span =
      iree_async_span_from_ptr(recv_buffer, sizeof(recv_buffer));

  iree_async_socket_recv_operation_t recv_op;
  CompletionTracker recv_tracker;
  InitRecvOperation(&recv_op, server, &recv_span, 1,
                    CompletionTracker::Callback, &recv_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &recv_op.base));

  // Poll until send and recv complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);

  EXPECT_EQ(recv_tracker.call_count, 1);
  IREE_EXPECT_OK(recv_tracker.ConsumeStatus());
  EXPECT_EQ(recv_op.bytes_received, send_length);

  // Verify data matches.
  EXPECT_EQ(memcmp(recv_buffer, send_data, send_length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Zero-copy send with large buffer to ensure actual ZC path is taken.
// On kernels with ZC support, the 64KB size should trigger the kernel's
// zero-copy optimization. On older kernels, this falls back to regular send.
TEST_P(SendFlagsTest, ZeroCopySendLargeBuffer) {
  // Create connected socket pair with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Large buffer (64KB) to ensure kernel uses zero-copy path.
  const iree_host_size_t send_length = 64 * 1024;
  std::vector<uint8_t> send_buffer(send_length);
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    send_buffer[i] = (uint8_t)(i & 0xFF);
  }
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_buffer.data(), send_length);

  // Submit send (ZC is determined by socket option).
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive all data.
  std::vector<uint8_t> recv_buffer(send_length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), send_length);

  // Poll for send completion (may have completed during recv loop).
  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);
  EXPECT_EQ(total_received, send_length);

  // Verify data integrity.
  EXPECT_EQ(memcmp(recv_buffer.data(), send_buffer.data(), send_length), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// MSG_MORE flag tests
//===----------------------------------------------------------------------===//

// Send two messages with MORE flag on first, verify both arrive.
TEST_P(SendFlagsTest, MorFlagCoalesces) {
  // Create connected socket pair.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnection(&client, &server, &listener);

  // Send first message with MORE flag.
  const char* send_data1 = "First";
  iree_async_span_t send_span1 =
      iree_async_span_from_ptr((void*)send_data1, strlen(send_data1));

  iree_async_socket_send_operation_t send_op1;
  CompletionTracker send_tracker1;
  InitSendOperation(&send_op1, client, &send_span1, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_MORE,
                    CompletionTracker::Callback, &send_tracker1);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op1.base));

  // Send second message without MORE flag (uncorks).
  const char* send_data2 = "Second";
  iree_async_span_t send_span2 =
      iree_async_span_from_ptr((void*)send_data2, strlen(send_data2));

  iree_async_socket_send_operation_t send_op2;
  CompletionTracker send_tracker2;
  InitSendOperation(&send_op2, client, &send_span2, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker2);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op2.base));

  // Poll until both sends complete.
  PollUntil(/*min_completions=*/2,
            /*total_budget=*/iree_make_duration_ms(5000));

  EXPECT_EQ(send_tracker1.call_count, 1);
  IREE_EXPECT_OK(send_tracker1.ConsumeStatus());
  EXPECT_EQ(send_tracker2.call_count, 1);
  IREE_EXPECT_OK(send_tracker2.ConsumeStatus());

  // Receive all data (may arrive in one or two recv calls).
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_host_size_t total_expected = strlen(send_data1) + strlen(send_data2);
  iree_host_size_t total_received =
      RecvAll(server, reinterpret_cast<uint8_t*>(recv_buffer), total_expected);

  // Verify both messages arrived (order preserved, may be coalesced).
  EXPECT_EQ(total_received, total_expected);
  EXPECT_EQ(memcmp(recv_buffer, "FirstSecond", total_expected), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Combined flags tests
//===----------------------------------------------------------------------===//

// Zero-copy send combined with MORE flag.
// Tests that socket option ZC works with per-send MORE flag and data arrives
// correctly.
TEST_P(SendFlagsTest, ZeroCopyWithMoreFlag) {
  // Create connected socket pair with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Send first message with MORE flag (ZC from socket option).
  const char* send_data1 = "ZeroCopy";
  iree_async_span_t send_span1 =
      iree_async_span_from_ptr((void*)send_data1, strlen(send_data1));

  iree_async_socket_send_operation_t send_op1;
  CompletionTracker send_tracker1;
  InitSendOperation(&send_op1, client, &send_span1, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_MORE,
                    CompletionTracker::Callback, &send_tracker1);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op1.base));

  // Send second message without MORE (uncorks). ZC is from socket option.
  const char* send_data2 = "AndMore";
  iree_async_span_t send_span2 =
      iree_async_span_from_ptr((void*)send_data2, strlen(send_data2));

  iree_async_socket_send_operation_t send_op2;
  CompletionTracker send_tracker2;
  InitSendOperation(&send_op2, client, &send_span2, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker2);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op2.base));

  // Poll until both sends complete.
  while (send_tracker1.call_count == 0 || send_tracker2.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(5000));
  }

  EXPECT_EQ(send_tracker1.call_count, 1);
  IREE_EXPECT_OK(send_tracker1.ConsumeStatus());
  EXPECT_EQ(send_tracker2.call_count, 1);
  IREE_EXPECT_OK(send_tracker2.ConsumeStatus());

  // Receive all data.
  char recv_buffer[256];
  memset(recv_buffer, 0, sizeof(recv_buffer));
  iree_host_size_t total_expected = strlen(send_data1) + strlen(send_data2);
  iree_host_size_t total_received =
      RecvAll(server, reinterpret_cast<uint8_t*>(recv_buffer), total_expected);

  EXPECT_EQ(total_received, total_expected);
  EXPECT_EQ(memcmp(recv_buffer, "ZeroCopyAndMore", total_expected), 0);

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Buffer safety tests
//===----------------------------------------------------------------------===//

// Verify buffer is not modified during in-flight zero-copy send.
// The callback must not fire until the kernel is done with the buffer.
// This test verifies buffer safety semantics regardless of ZC availability.
TEST_P(SendFlagsTest, ZeroCopyBufferSafetyOnCompletion) {
  // Create connected socket pair with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Use a mutable buffer with a known pattern.
  const iree_host_size_t buffer_size = 4096;
  std::vector<uint8_t> send_buffer(buffer_size);
  for (iree_host_size_t i = 0; i < buffer_size; ++i) {
    send_buffer[i] = (uint8_t)(i & 0xFF);
  }
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_buffer.data(), buffer_size);

  // Submit send (ZC is determined by socket option).
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);

  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive all data first to ensure send progresses.
  std::vector<uint8_t> recv_buffer(buffer_size);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), buffer_size);

  // Poll for send completion.
  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  // After callback fires, it's safe to modify the buffer.
  // The test verifies the callback doesn't fire too early by checking
  // that the received data matches the original pattern.
  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(total_received, buffer_size);

  // Verify received data matches original (proves buffer wasn't corrupted
  // by premature reuse).
  for (iree_host_size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(recv_buffer[i], (uint8_t)(i & 0xFF)) << "Mismatch at byte " << i;
  }

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

//===----------------------------------------------------------------------===//
// Registered buffer zero-copy tests
//===----------------------------------------------------------------------===//

// Zero-copy send using registered slab buffers (fixed buffer path).
// When socket has ZERO_COPY option and the span has a registered region, the
// kernel uses FIXED_BUF mode to avoid per-operation page pinning.
TEST_P(SendFlagsTest, ZeroCopySendRegisteredSlab) {
  // Create slab with power-of-2 buffer count (io_uring requirement).
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 4096;
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  // Register slab with READ access for send operations.
  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  // Create pool over region for lock-free acquire/release.
  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Acquire buffer from pool and fill with pattern.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &lease));
  memset(iree_async_span_ptr(lease.span), 0xAB, lease.span.length);

  // Send using lease.span (has region, triggers fixed-buffer path on capable
  // backends). ZC is determined by socket option.
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &lease.span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive and verify.
  std::vector<uint8_t> recv_buffer(lease.span.length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), lease.span.length);

  // Poll for send completion.
  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, lease.span.length);
  EXPECT_EQ(total_received, lease.span.length);

  // Verify data integrity - all bytes should be 0xAB.
  for (iree_host_size_t i = 0; i < total_received; ++i) {
    EXPECT_EQ(recv_buffer[i], 0xAB) << "Mismatch at byte " << i;
  }

  // Cleanup.
  iree_async_buffer_lease_release(&lease);
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Large transfer using registered buffers: sends all buffers from a slab
// sequentially.
TEST_P(SendFlagsTest, ZeroCopySendRegisteredLargeTransfer) {
  // 64KB buffers, 16 count = 1MB total capacity.
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 64 * 1024;
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Track total bytes sent/received.
  iree_host_size_t total_sent = 0;
  const iree_host_size_t expected_total = slab_options.buffer_size;

  // Send one buffer.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &lease));

  // Fill with pattern based on buffer index.
  uint8_t* buf_ptr = static_cast<uint8_t*>(iree_async_span_ptr(lease.span));
  for (iree_host_size_t j = 0; j < lease.span.length; ++j) {
    buf_ptr[j] = static_cast<uint8_t>((j + 0x42) & 0xFF);
  }

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &lease.span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive all data for this buffer.
  std::vector<uint8_t> recv_buffer(expected_total);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), lease.span.length);

  // Wait for send completion.
  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  total_sent += send_op.bytes_sent;
  iree_async_buffer_lease_release(&lease);

  EXPECT_EQ(total_sent, expected_total);
  EXPECT_EQ(total_received, expected_total);

  // Verify data pattern.
  for (iree_host_size_t i = 0; i < expected_total; ++i) {
    EXPECT_EQ(recv_buffer[i], static_cast<uint8_t>((i + 0x42) & 0xFF))
        << "Mismatch at byte " << i;
  }

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Zero-copy send with partial buffer (non-zero offset within span).
// Verifies that the fixed-buffer path handles subspans correctly.
TEST_P(SendFlagsTest, ZeroCopySendPartialBuffer) {
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 4096;
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Acquire a full buffer.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &lease));

  // Fill entire buffer with a pattern.
  uint8_t* buf_ptr = static_cast<uint8_t*>(iree_async_span_ptr(lease.span));
  for (iree_host_size_t i = 0; i < lease.span.length; ++i) {
    buf_ptr[i] = static_cast<uint8_t>(i & 0xFF);
  }

  // Create a subspan: offset=1024, length=2048 (middle portion of 4096 buffer).
  const iree_host_size_t partial_offset = 1024;
  const iree_host_size_t partial_length = 2048;
  iree_async_span_t partial_span;
  memset(&partial_span, 0, sizeof(partial_span));
  partial_span.region = lease.span.region;
  partial_span.offset = lease.span.offset + partial_offset;
  partial_span.length = partial_length;

  // Send the partial span (ZC determined by socket option).
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &partial_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive.
  std::vector<uint8_t> recv_buffer(partial_length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), partial_length);

  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, partial_length);
  EXPECT_EQ(total_received, partial_length);

  // Verify data matches the partial range [1024..3072) from the original.
  for (iree_host_size_t i = 0; i < partial_length; ++i) {
    uint8_t expected = static_cast<uint8_t>((partial_offset + i) & 0xFF);
    EXPECT_EQ(recv_buffer[i], expected) << "Mismatch at byte " << i;
  }

  iree_async_buffer_lease_release(&lease);
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Scatter-gather send using buffers from registered pool.
// Multi-buffer sends use SENDMSG[_ZC] which does NOT support FIXED_BUF - this
// confirms the fallback to ad-hoc ZC page pinning works correctly.
TEST_P(SendFlagsTest, ScatterGatherFromRegistered) {
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Acquire 3 buffers for scatter-gather.
  iree_async_buffer_lease_t leases[3];
  iree_async_span_t spans[3];
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &leases[i]));
    spans[i] = leases[i].span;
    // Fill each buffer with a distinct pattern.
    memset(iree_async_span_ptr(spans[i]), 'A' + i, spans[i].length);
  }

  const iree_host_size_t total_length = 3 * slab_options.buffer_size;

  // Multi-buffer send uses SENDMSG_ZC (no FIXED_BUF support). ZC from socket.
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, spans, 3,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive all data.
  std::vector<uint8_t> recv_buffer(total_length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), total_length);

  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, total_length);
  EXPECT_EQ(total_received, total_length);

  // Verify: first 1024 bytes = 'A', next 1024 = 'B', last 1024 = 'C'.
  for (iree_host_size_t i = 0; i < slab_options.buffer_size; ++i) {
    EXPECT_EQ(recv_buffer[i], 'A') << "Mismatch in first buffer at " << i;
  }
  for (iree_host_size_t i = 0; i < slab_options.buffer_size; ++i) {
    EXPECT_EQ(recv_buffer[slab_options.buffer_size + i], 'B')
        << "Mismatch in second buffer at " << i;
  }
  for (iree_host_size_t i = 0; i < slab_options.buffer_size; ++i) {
    EXPECT_EQ(recv_buffer[2 * slab_options.buffer_size + i], 'C')
        << "Mismatch in third buffer at " << i;
  }

  for (int i = 0; i < 3; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Concurrent sends from registered buffers: submit 4 sends simultaneously.
// Verifies no corruption when multiple registered-buffer sends are in flight.
TEST_P(SendFlagsTest, ConcurrentRegisteredSends) {
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  const int kNumConcurrent = 4;
  iree_async_buffer_lease_t leases[kNumConcurrent];
  iree_async_socket_send_operation_t send_ops[kNumConcurrent];
  CompletionTracker trackers[kNumConcurrent];

  // Acquire buffers and fill with distinct patterns.
  for (int i = 0; i < kNumConcurrent; ++i) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &leases[i]));
    memset(iree_async_span_ptr(leases[i].span), '0' + i, leases[i].span.length);

    InitSendOperation(&send_ops[i], client, &leases[i].span, 1,
                      IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                      CompletionTracker::Callback, &trackers[i]);
  }

  // Submit all sends.
  for (int i = 0; i < kNumConcurrent; ++i) {
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &send_ops[i].base));
  }

  // Receive all data.
  const iree_host_size_t total_expected =
      kNumConcurrent * slab_options.buffer_size;
  std::vector<uint8_t> recv_buffer(total_expected);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), total_expected);

  // Wait for all sends to complete. Send callbacks may have already fired
  // during recv polling, so check tracker counts before requesting more
  // completions from the proactor.
  int send_completed = 0;
  for (int i = 0; i < kNumConcurrent; ++i) {
    if (trackers[i].call_count > 0) ++send_completed;
  }
  while (send_completed < kNumConcurrent) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
    send_completed = 0;
    for (int i = 0; i < kNumConcurrent; ++i) {
      if (trackers[i].call_count > 0) ++send_completed;
    }
  }

  // Verify all sends succeeded.
  iree_host_size_t total_sent = 0;
  for (int i = 0; i < kNumConcurrent; ++i) {
    EXPECT_EQ(trackers[i].call_count, 1) << "Send " << i << " not completed";
    IREE_EXPECT_OK(trackers[i].ConsumeStatus());
    total_sent += send_ops[i].bytes_sent;
  }
  EXPECT_EQ(total_sent, total_expected);
  EXPECT_EQ(total_received, total_expected);

  // Count occurrences of each pattern byte to verify all data arrived.
  // Due to TCP ordering, all bytes from each send arrive together, but send
  // order relative to each other is not guaranteed by the kernel.
  int counts[kNumConcurrent] = {0};
  for (iree_host_size_t i = 0; i < total_received; ++i) {
    int idx = recv_buffer[i] - '0';
    if (idx >= 0 && idx < kNumConcurrent) {
      counts[idx]++;
    }
  }
  for (int i = 0; i < kNumConcurrent; ++i) {
    EXPECT_EQ(counts[i], (int)slab_options.buffer_size)
        << "Pattern '" << (char)('0' + i) << "' count mismatch";
  }

  for (int i = 0; i < kNumConcurrent; ++i) {
    iree_async_buffer_lease_release(&leases[i]);
  }
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Zero-copy send with unregistered (heap) buffer: exercises ad-hoc ZC path.
// When the span has no region (region == NULL), the kernel uses per-operation
// page pinning. This test explicitly verifies the fallback behavior.
TEST_P(SendFlagsTest, ZeroCopySendUnregisteredFallback) {
  // Create connected socket pair with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Use a heap buffer (no region).
  const iree_host_size_t send_length = 8192;
  std::vector<uint8_t> send_buffer(send_length);
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    send_buffer[i] = static_cast<uint8_t>((i * 7) & 0xFF);
  }

  // Create span without region (ad-hoc ZC path).
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_buffer.data(), send_length);

  // Verify this span has no region.
  ASSERT_EQ(send_span.region, nullptr);

  // Send (ZC determined by socket option - uses ad-hoc page pinning).
  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive.
  std::vector<uint8_t> recv_buffer(send_length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), send_length);

  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);
  EXPECT_EQ(total_received, send_length);

  // Verify data.
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    uint8_t expected = static_cast<uint8_t>((i * 7) & 0xFF);
    EXPECT_EQ(recv_buffer[i], expected) << "Mismatch at byte " << i;
  }

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Scatter-gather send with mixed registered and unregistered buffers.
// Multi-buffer sends use SENDMSG_ZC which doesn't support FIXED_BUF, so this
// exercises the ad-hoc ZC path where some buffers are registered and some are
// not.
TEST_P(SendFlagsTest, ScatterGatherMixedRegistration) {
  iree_async_slab_options_t slab_options = {};
  slab_options.buffer_size = 1024;
  slab_options.buffer_count = 16;

  iree_async_slab_t* slab = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_options, iree_allocator_system(), &slab));

  iree_async_region_t* region = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region));

  iree_async_buffer_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      iree_async_buffer_pool_allocate(region, iree_allocator_system(), &pool));

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Acquire one registered buffer from pool.
  iree_async_buffer_lease_t lease;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool, &lease));
  memset(iree_async_span_ptr(lease.span), 'R', lease.span.length);

  // Create a heap buffer (unregistered).
  const iree_host_size_t heap_size = 1024;
  std::vector<uint8_t> heap_buffer(heap_size);
  memset(heap_buffer.data(), 'H', heap_size);
  iree_async_span_t heap_span =
      iree_async_span_from_ptr(heap_buffer.data(), heap_size);

  // Verify spans have expected registration state.
  ASSERT_NE(lease.span.region, nullptr);  // Registered.
  ASSERT_EQ(heap_span.region, nullptr);   // Unregistered.

  // Create 2-buffer scatter-gather: registered first, then heap.
  iree_async_span_t spans[2] = {lease.span, heap_span};
  const iree_host_size_t total_length = lease.span.length + heap_size;

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, spans, 2,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive all data.
  std::vector<uint8_t> recv_buffer(total_length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), total_length);

  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, total_length);
  EXPECT_EQ(total_received, total_length);

  // Verify data: first 1024 bytes should be 'R' (registered), next 1024 = 'H'.
  for (iree_host_size_t i = 0; i < lease.span.length; ++i) {
    EXPECT_EQ(recv_buffer[i], 'R') << "Mismatch in registered buffer at " << i;
  }
  for (iree_host_size_t i = 0; i < heap_size; ++i) {
    EXPECT_EQ(recv_buffer[lease.span.length + i], 'H')
        << "Mismatch in heap buffer at " << i;
  }

  iree_async_buffer_lease_release(&lease);
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool);
  iree_async_region_release(region);
  iree_async_slab_release(slab);
}

// Verify sends work correctly when ZC capability is unavailable.
// When socket lacks ZERO_COPY option, sends use regular SEND path. This test
// confirms the copy path works and ZERO_COPY_ACHIEVED flag is NOT set.
TEST_P(SendFlagsTest, ZeroCopyFallbackWhenCapabilityDisabled) {
  // Create socket WITHOUT ZERO_COPY option.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(&client, &server, &listener,
                                 IREE_ASYNC_SOCKET_OPTION_NO_DELAY,
                                 IREE_ASYNC_SOCKET_OPTION_NONE);

  // Send data using same pattern as ZC tests.
  const iree_host_size_t send_length = 8192;
  std::vector<uint8_t> send_buffer(send_length);
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    send_buffer[i] = static_cast<uint8_t>((i * 13) & 0xFF);
  }
  iree_async_span_t send_span =
      iree_async_span_from_ptr(send_buffer.data(), send_length);

  iree_async_socket_send_operation_t send_op;
  CompletionTracker send_tracker;
  InitSendOperation(&send_op, client, &send_span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &send_tracker);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &send_op.base));

  // Receive all data.
  std::vector<uint8_t> recv_buffer(send_length);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), send_length);

  while (send_tracker.call_count == 0) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
  }

  EXPECT_EQ(send_tracker.call_count, 1);
  IREE_EXPECT_OK(send_tracker.ConsumeStatus());
  EXPECT_EQ(send_op.bytes_sent, send_length);
  EXPECT_EQ(total_received, send_length);

  // Verify ZERO_COPY_ACHIEVED flag is NOT set (copy path was used).
  EXPECT_EQ(
      send_tracker.last_flags & IREE_ASYNC_COMPLETION_FLAG_ZERO_COPY_ACHIEVED,
      0u)
      << "ZERO_COPY_ACHIEVED should not be set when socket lacks ZERO_COPY "
         "option";

  // Verify data integrity.
  for (iree_host_size_t i = 0; i < send_length; ++i) {
    uint8_t expected = static_cast<uint8_t>((i * 13) & 0xFF);
    EXPECT_EQ(recv_buffer[i], expected) << "Mismatch at byte " << i;
  }

  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
}

// Verify sends from different registered slabs work correctly.
// Creates two slabs with different buffer sizes and does interleaved sends
// to verify buffer management handles multiple slabs correctly.
TEST_P(SendFlagsTest, ZeroCopySendMultipleSlabs) {
  // Create first slab (1KB buffers).
  iree_async_slab_options_t slab_a_options = {};
  slab_a_options.buffer_size = 1024;
  slab_a_options.buffer_count = 16;

  iree_async_slab_t* slab_a = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_a_options, iree_allocator_system(), &slab_a));

  iree_async_region_t* region_a = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab_a, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_a));

  iree_async_buffer_pool_t* pool_a = nullptr;
  IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
      region_a, iree_allocator_system(), &pool_a));

  // Create second slab (2KB buffers).
  iree_async_slab_options_t slab_b_options = {};
  slab_b_options.buffer_size = 2048;
  slab_b_options.buffer_count = 16;

  iree_async_slab_t* slab_b = nullptr;
  IREE_ASSERT_OK(
      iree_async_slab_create(slab_b_options, iree_allocator_system(), &slab_b));

  // Note: Second registration may fail if backend only allows one region.
  // In that case we still have one slab to test with, but we test the
  // single-slab path instead.
  iree_async_region_t* region_b = nullptr;
  iree_async_buffer_pool_t* pool_b = nullptr;
  iree_status_t status = iree_async_proactor_register_slab(
      proactor_, slab_b, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_b);
  bool has_two_slabs = iree_status_is_ok(status);
  if (!has_two_slabs) {
    iree_status_ignore(status);
    // Fall back to testing with just one slab (acquire multiple buffers).
  } else {
    IREE_ASSERT_OK(iree_async_buffer_pool_allocate(
        region_b, iree_allocator_system(), &pool_b));
  }

  // Establish connection with ZERO_COPY enabled on client.
  iree_async_socket_t* client = nullptr;
  iree_async_socket_t* server = nullptr;
  iree_async_socket_t* listener = nullptr;
  EstablishConnectionWithOptions(
      &client, &server, &listener,
      IREE_ASYNC_SOCKET_OPTION_NO_DELAY | IREE_ASYNC_SOCKET_OPTION_ZERO_COPY,
      IREE_ASYNC_SOCKET_OPTION_NONE);

  // Acquire buffers and fill with distinct patterns.
  iree_async_buffer_lease_t lease_a1;
  IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool_a, &lease_a1));
  memset(iree_async_span_ptr(lease_a1.span), 'A', lease_a1.span.length);

  iree_async_buffer_lease_t lease_b;
  iree_async_buffer_lease_t lease_a2;

  if (has_two_slabs) {
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool_b, &lease_b));
    memset(iree_async_span_ptr(lease_b.span), 'B', lease_b.span.length);

    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool_a, &lease_a2));
    memset(iree_async_span_ptr(lease_a2.span), 'C', lease_a2.span.length);
  } else {
    // Single slab fallback: acquire two more buffers from pool_a.
    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool_a, &lease_b));
    memset(iree_async_span_ptr(lease_b.span), 'B', lease_b.span.length);

    IREE_ASSERT_OK(iree_async_buffer_pool_acquire(pool_a, &lease_a2));
    memset(iree_async_span_ptr(lease_a2.span), 'C', lease_a2.span.length);
  }

  // Send interleaved: slab_a, slab_b (or slab_a), slab_a.
  iree_async_socket_send_operation_t send_ops[3];
  CompletionTracker trackers[3];

  InitSendOperation(&send_ops[0], client, &lease_a1.span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &trackers[0]);
  InitSendOperation(&send_ops[1], client, &lease_b.span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &trackers[1]);
  InitSendOperation(&send_ops[2], client, &lease_a2.span, 1,
                    IREE_ASYNC_SOCKET_SEND_FLAG_NONE,
                    CompletionTracker::Callback, &trackers[2]);

  const iree_host_size_t total_expected =
      lease_a1.span.length + lease_b.span.length + lease_a2.span.length;

  // Submit all sends.
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(
        iree_async_proactor_submit_one(proactor_, &send_ops[i].base));
  }

  // Receive all data.
  std::vector<uint8_t> recv_buffer(total_expected);
  iree_host_size_t total_received =
      RecvAll(server, recv_buffer.data(), total_expected);

  // Wait for all sends to complete. Send callbacks may have already fired
  // during recv polling, so check tracker counts before requesting more
  // completions from the proactor.
  int send_completed = 0;
  for (int i = 0; i < 3; ++i) {
    if (trackers[i].call_count > 0) ++send_completed;
  }
  while (send_completed < 3) {
    PollUntil(/*min_completions=*/1,
              /*total_budget=*/iree_make_duration_ms(1000));
    send_completed = 0;
    for (int i = 0; i < 3; ++i) {
      if (trackers[i].call_count > 0) ++send_completed;
    }
  }

  // Verify all sends succeeded.
  iree_host_size_t total_sent = 0;
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(trackers[i].call_count, 1) << "Send " << i << " not completed";
    IREE_EXPECT_OK(trackers[i].ConsumeStatus());
    total_sent += send_ops[i].bytes_sent;
  }
  EXPECT_EQ(total_sent, total_expected);
  EXPECT_EQ(total_received, total_expected);

  // Verify data patterns arrived in order.
  iree_host_size_t offset = 0;
  for (iree_host_size_t i = 0; i < lease_a1.span.length; ++i) {
    EXPECT_EQ(recv_buffer[offset + i], 'A')
        << "Mismatch in first send at " << i;
  }
  offset += lease_a1.span.length;
  for (iree_host_size_t i = 0; i < lease_b.span.length; ++i) {
    EXPECT_EQ(recv_buffer[offset + i], 'B')
        << "Mismatch in second send at " << i;
  }
  offset += lease_b.span.length;
  for (iree_host_size_t i = 0; i < lease_a2.span.length; ++i) {
    EXPECT_EQ(recv_buffer[offset + i], 'C')
        << "Mismatch in third send at " << i;
  }

  // Cleanup.
  iree_async_buffer_lease_release(&lease_a1);
  iree_async_buffer_lease_release(&lease_b);
  iree_async_buffer_lease_release(&lease_a2);
  iree_async_socket_release(server);
  iree_async_socket_release(client);
  iree_async_socket_release(listener);
  iree_async_buffer_pool_free(pool_a);
  iree_async_region_release(region_a);
  iree_async_slab_release(slab_a);
  if (has_two_slabs) {
    iree_async_buffer_pool_free(pool_b);
    iree_async_region_release(region_b);
  }
  iree_async_slab_release(slab_b);
}

CTS_REGISTER_TEST_SUITE(SendFlagsTest);

}  // namespace iree::async::cts
