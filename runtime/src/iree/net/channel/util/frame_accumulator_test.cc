// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/util/frame_accumulator.h"

#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace net {
namespace {

//===----------------------------------------------------------------------===//
// Test protocol: 4-byte little-endian length prefix
//===----------------------------------------------------------------------===//

// Header size for our test protocol.
static constexpr iree_host_size_t kHeaderSize = 4;

// Minimum frame size (header only, no payload).
static constexpr iree_host_size_t kMinFrameSize = kHeaderSize;

// Determines frame length from partial data using 4-byte LE length prefix.
// Returns 0 if not enough data to determine length.
static iree_host_size_t TestFrameLength(void* user_data,
                                        iree_const_byte_span_t available) {
  if (available.data_length < kHeaderSize) {
    return 0;  // Need more data to read length.
  }
  // 4-byte little-endian length prefix (total frame size including header).
  uint32_t frame_size = available.data[0] | (available.data[1] << 8) |
                        (available.data[2] << 16) | (available.data[3] << 24);
  return static_cast<iree_host_size_t>(frame_size);
}

// Creates a test frame with 4-byte LE length prefix.
static std::vector<uint8_t> MakeFrame(const std::string& payload) {
  uint32_t frame_size = static_cast<uint32_t>(kHeaderSize + payload.size());
  std::vector<uint8_t> frame(frame_size);
  frame[0] = frame_size & 0xFF;
  frame[1] = (frame_size >> 8) & 0xFF;
  frame[2] = (frame_size >> 16) & 0xFF;
  frame[3] = (frame_size >> 24) & 0xFF;
  memcpy(frame.data() + kHeaderSize, payload.data(), payload.size());
  return frame;
}

//===----------------------------------------------------------------------===//
// Test context for tracking received frames
//===----------------------------------------------------------------------===//

struct ReceivedFrame {
  std::vector<uint8_t> data;
  bool zero_copy;  // True if lease was non-NULL (zero-copy path).
};

struct TestContext {
  std::vector<ReceivedFrame> frames;
  iree_status_code_t next_error = IREE_STATUS_OK;
  bool should_retain_lease = false;
};

// Callback that records received frames.
static iree_status_t TestOnFrameComplete(void* user_data,
                                         iree_const_byte_span_t frame,
                                         iree_async_buffer_lease_t* lease) {
  TestContext* ctx = static_cast<TestContext*>(user_data);

  if (ctx->next_error != IREE_STATUS_OK) {
    iree_status_code_t error = ctx->next_error;
    ctx->next_error = IREE_STATUS_OK;  // Reset for next call.
    return iree_make_status(error, "injected error");
  }

  ReceivedFrame received;
  received.data.assign(frame.data, frame.data + frame.data_length);
  received.zero_copy = (lease != nullptr);
  ctx->frames.push_back(std::move(received));

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Mock lease for testing
//===----------------------------------------------------------------------===//

struct MockLease {
  std::vector<uint8_t> data;
  iree_async_buffer_lease_t lease;
  int release_count = 0;

  static void ReleaseCallback(void* user_data,
                              iree_async_buffer_index_t buffer_index) {
    MockLease* self = static_cast<MockLease*>(user_data);
    self->release_count++;
  }

  MockLease(const std::vector<uint8_t>& buffer_data)
      : data(buffer_data), release_count(0) {
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(data.data(), data.size());
    lease.release.fn = ReleaseCallback;
    lease.release.user_data = this;
    lease.buffer_index = 0;
  }

  // Create a lease from multiple frames concatenated.
  MockLease(std::initializer_list<std::vector<uint8_t>> frame_list)
      : release_count(0) {
    for (const auto& frame : frame_list) {
      data.insert(data.end(), frame.begin(), frame.end());
    }
    memset(&lease, 0, sizeof(lease));
    lease.span = iree_async_span_from_ptr(data.data(), data.size());
    lease.release.fn = ReleaseCallback;
    lease.release.user_data = this;
    lease.buffer_index = 0;
  }
};

//===----------------------------------------------------------------------===//
// RAII wrapper for accumulator
//===----------------------------------------------------------------------===//

class AccumulatorWrapper {
 public:
  AccumulatorWrapper(iree_host_size_t max_frame_size, TestContext* ctx)
      : storage_(iree_net_frame_accumulator_storage_size(max_frame_size)) {
    accumulator_ =
        reinterpret_cast<iree_net_frame_accumulator_t*>(storage_.data());
    iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
    iree_net_frame_complete_callback_t on_frame_complete = {TestOnFrameComplete,
                                                            ctx};
    iree_status_t status = iree_net_frame_accumulator_initialize(
        accumulator_, max_frame_size, frame_length, on_frame_complete);
    IREE_CHECK_OK(status);
  }

  ~AccumulatorWrapper() {
    iree_net_frame_accumulator_deinitialize(accumulator_);
  }

  iree_net_frame_accumulator_t* get() { return accumulator_; }

 private:
  std::vector<uint8_t> storage_;
  iree_net_frame_accumulator_t* accumulator_;
};

//===----------------------------------------------------------------------===//
// Lifecycle Tests
//===----------------------------------------------------------------------===//

TEST(FrameAccumulatorTest, InitializeSetsFields) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);
  auto* acc = wrapper.get();

  EXPECT_EQ(acc->buffer_capacity, kMaxFrameSize);
  EXPECT_EQ(acc->buffer_used, 0u);
  EXPECT_EQ(acc->frame_length.fn, TestFrameLength);
  EXPECT_EQ(acc->on_frame_complete.fn, TestOnFrameComplete);
  EXPECT_EQ(acc->on_frame_complete.user_data, &ctx);
}

TEST(FrameAccumulatorTest, DeinitializeClearsFields) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;

  std::vector<uint8_t> storage(
      iree_net_frame_accumulator_storage_size(kMaxFrameSize));
  auto* acc = reinterpret_cast<iree_net_frame_accumulator_t*>(storage.data());

  iree_net_frame_length_callback_t frame_length = {TestFrameLength, nullptr};
  iree_net_frame_complete_callback_t on_frame_complete = {TestOnFrameComplete,
                                                          &ctx};
  IREE_ASSERT_OK(iree_net_frame_accumulator_initialize(
      acc, kMaxFrameSize, frame_length, on_frame_complete));

  // Buffer some data.
  auto frame = MakeFrame("test");
  MockLease lease({frame[0]});  // Just first byte.
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(acc, &lease.lease, 1));
  EXPECT_GT(acc->buffer_used, 0u);

  iree_net_frame_accumulator_deinitialize(acc);

  // After deinit, fields should be zeroed.
  EXPECT_EQ(acc->buffer_capacity, 0u);
  EXPECT_EQ(acc->buffer_used, 0u);
  EXPECT_EQ(acc->frame_length.fn, nullptr);
}

TEST(FrameAccumulatorTest, ResetClearsBuffer) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);
  auto* acc = wrapper.get();

  // Buffer some data (partial frame).
  auto frame = MakeFrame("test");
  MockLease lease({frame[0], frame[1]});  // Just first 2 bytes.
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(acc, &lease.lease, 2));
  EXPECT_EQ(acc->buffer_used, 2u);
  EXPECT_TRUE(iree_net_frame_accumulator_has_partial_frame(acc));

  iree_net_frame_accumulator_reset(acc);

  EXPECT_EQ(acc->buffer_used, 0u);
  EXPECT_FALSE(iree_net_frame_accumulator_has_partial_frame(acc));
}

//===----------------------------------------------------------------------===//
// Zero-Copy Path Tests
//===----------------------------------------------------------------------===//

TEST(FrameAccumulatorTest, SingleCompleteFrameIsZeroCopy) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("Hello, World!");
  MockLease lease(frame);

  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease.lease, frame.size()));

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame);
  EXPECT_TRUE(ctx.frames[0].zero_copy);
  EXPECT_EQ(lease.release_count, 1);
}

TEST(FrameAccumulatorTest, MultipleFramesInOneBufferAllZeroCopy) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame1 = MakeFrame("First");
  auto frame2 = MakeFrame("Second");
  auto frame3 = MakeFrame("Third");
  MockLease lease({frame1, frame2, frame3});

  iree_host_size_t total_size = frame1.size() + frame2.size() + frame3.size();
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease.lease, total_size));

  ASSERT_EQ(ctx.frames.size(), 3u);
  EXPECT_EQ(ctx.frames[0].data, frame1);
  EXPECT_EQ(ctx.frames[1].data, frame2);
  EXPECT_EQ(ctx.frames[2].data, frame3);
  EXPECT_TRUE(ctx.frames[0].zero_copy);
  EXPECT_TRUE(ctx.frames[1].zero_copy);
  EXPECT_TRUE(ctx.frames[2].zero_copy);
  EXPECT_EQ(lease.release_count, 1);
}

TEST(FrameAccumulatorTest, FrameExactlyFillsBuffer) {
  TestContext ctx;
  auto frame = MakeFrame("Exact");
  AccumulatorWrapper wrapper(frame.size(), &ctx);

  MockLease lease(frame);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease.lease, frame.size()));

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame);
  EXPECT_TRUE(ctx.frames[0].zero_copy);
}

//===----------------------------------------------------------------------===//
// Copy Path Tests (Frame spans multiple buffers)
//===----------------------------------------------------------------------===//

TEST(FrameAccumulatorTest, FrameSpansTwoBuffers) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("Split across buffers");
  iree_host_size_t split_point = frame.size() / 2;

  // First half.
  std::vector<uint8_t> first_half(frame.begin(), frame.begin() + split_point);
  MockLease lease1(first_half);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease1.lease, first_half.size()));
  EXPECT_EQ(ctx.frames.size(), 0u);  // Not complete yet.
  EXPECT_EQ(lease1.release_count, 1);

  // Second half.
  std::vector<uint8_t> second_half(frame.begin() + split_point, frame.end());
  MockLease lease2(second_half);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease2.lease, second_half.size()));

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame);
  EXPECT_FALSE(ctx.frames[0].zero_copy);  // From internal buffer.
  EXPECT_EQ(lease2.release_count, 1);
}

TEST(FrameAccumulatorTest, PartialHeaderThenRest) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("test");

  // First, just 2 bytes of the 4-byte header.
  std::vector<uint8_t> partial_header(frame.begin(), frame.begin() + 2);
  MockLease lease1(partial_header);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease1.lease, partial_header.size()));
  EXPECT_EQ(ctx.frames.size(), 0u);

  // Then the rest.
  std::vector<uint8_t> rest(frame.begin() + 2, frame.end());
  MockLease lease2(rest);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease2.lease, rest.size()));

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame);
  EXPECT_FALSE(ctx.frames[0].zero_copy);
}

TEST(FrameAccumulatorTest, SingleByteAtATime) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("byte by byte");

  for (size_t i = 0; i < frame.size(); ++i) {
    std::vector<uint8_t> single_byte = {frame[i]};
    MockLease lease(single_byte);
    IREE_ASSERT_OK(
        iree_net_frame_accumulator_push_lease(wrapper.get(), &lease.lease, 1));
  }

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame);
  EXPECT_FALSE(ctx.frames[0].zero_copy);
}

//===----------------------------------------------------------------------===//
// Mixed Path Tests
//===----------------------------------------------------------------------===//

TEST(FrameAccumulatorTest, CompleteAndPartialInOneBuffer) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame1 = MakeFrame("Complete");
  auto frame2 = MakeFrame("Partial frame");

  // Buffer contains complete frame1 + partial frame2.
  iree_host_size_t partial_bytes = frame2.size() / 2;
  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin(), frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.begin() + partial_bytes);

  MockLease lease1(buffer);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease1.lease, buffer.size()));

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame1);
  EXPECT_TRUE(ctx.frames[0].zero_copy);
  EXPECT_TRUE(iree_net_frame_accumulator_has_partial_frame(wrapper.get()));

  // Send rest of frame2.
  std::vector<uint8_t> rest(frame2.begin() + partial_bytes, frame2.end());
  MockLease lease2(rest);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease2.lease, rest.size()));

  ASSERT_EQ(ctx.frames.size(), 2u);
  EXPECT_EQ(ctx.frames[1].data, frame2);
  EXPECT_FALSE(ctx.frames[1].zero_copy);  // From internal buffer.
}

TEST(FrameAccumulatorTest, BufferedCompletionThenZeroCopyInSamePush) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame1 = MakeFrame("First");
  auto frame2 = MakeFrame("Second");

  // First push: partial frame1.
  iree_host_size_t split_point = frame1.size() / 2;
  std::vector<uint8_t> partial(frame1.begin(), frame1.begin() + split_point);
  MockLease lease1(partial);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease1.lease, partial.size()));
  EXPECT_EQ(ctx.frames.size(), 0u);

  // Second push: rest of frame1 + complete frame2.
  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin() + split_point, frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.end());
  MockLease lease2(buffer);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease2.lease, buffer.size()));

  ASSERT_EQ(ctx.frames.size(), 2u);
  EXPECT_EQ(ctx.frames[0].data, frame1);
  EXPECT_FALSE(ctx.frames[0].zero_copy);  // From internal buffer.
  EXPECT_EQ(ctx.frames[1].data, frame2);
  EXPECT_TRUE(ctx.frames[1].zero_copy);  // Direct from lease.
}

//===----------------------------------------------------------------------===//
// Error Tests
//===----------------------------------------------------------------------===//

TEST(FrameAccumulatorTest, FrameExceedsMaxSize) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 16;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  // Create a frame larger than max size.
  auto frame = MakeFrame("This payload is way too long for the buffer!");
  ASSERT_GT(frame.size(), kMaxFrameSize);

  MockLease lease(frame);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_frame_accumulator_push_lease(
                            wrapper.get(), &lease.lease, frame.size()));
  EXPECT_EQ(ctx.frames.size(), 0u);
  EXPECT_EQ(lease.release_count, 1);  // Lease released on error.
}

TEST(FrameAccumulatorTest, FrameExceedsMaxSizeDiscoveredDuringBuffering) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 16;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  // Create a frame larger than max size.
  auto frame = MakeFrame("This payload is way too long!");
  ASSERT_GT(frame.size(), kMaxFrameSize);

  // Send just the header first (which encodes the too-large size).
  std::vector<uint8_t> header(frame.begin(), frame.begin() + kHeaderSize);
  MockLease lease1(header);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_net_frame_accumulator_push_lease(
                            wrapper.get(), &lease1.lease, header.size()));
}

// A frame_length_fn that always returns 0, simulating a protocol where the
// header size exceeds max_frame_size. This tests the stuck condition fix.
static iree_host_size_t NeverKnowsFrameLength(
    void* user_data, iree_const_byte_span_t available) {
  (void)user_data;
  (void)available;
  return 0;  // Always claims to need more data.
}

TEST(FrameAccumulatorTest, BufferFullButCannotDetermineFrameSize) {
  // This tests the fix for the infinite loop DoS vulnerability.
  // If frame_length_fn can't determine frame size within max_frame_size bytes,
  // we should get RESOURCE_EXHAUSTED, not an infinite loop.
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 8;

  std::vector<uint8_t> storage(
      iree_net_frame_accumulator_storage_size(kMaxFrameSize));
  auto* acc = reinterpret_cast<iree_net_frame_accumulator_t*>(storage.data());

  // Use a frame_length callback that never determines the frame size.
  iree_net_frame_length_callback_t frame_length = {NeverKnowsFrameLength,
                                                   nullptr};
  iree_net_frame_complete_callback_t on_frame_complete = {TestOnFrameComplete,
                                                          &ctx};
  IREE_ASSERT_OK(iree_net_frame_accumulator_initialize(
      acc, kMaxFrameSize, frame_length, on_frame_complete));

  // Push more data than the buffer can hold.
  std::vector<uint8_t> data(16, 0x42);
  MockLease lease(data);

  // Should fail with RESOURCE_EXHAUSTED, not hang forever.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_net_frame_accumulator_push_lease(acc, &lease.lease, data.size()));

  // Lease should still be released on error.
  EXPECT_EQ(lease.release_count, 1);

  // No frames should have been delivered.
  EXPECT_EQ(ctx.frames.size(), 0u);

  iree_net_frame_accumulator_deinitialize(acc);
}

TEST(FrameAccumulatorTest, CallbackErrorPropagates) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("test");
  MockLease lease(frame);

  ctx.next_error = IREE_STATUS_INTERNAL;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_net_frame_accumulator_push_lease(
                            wrapper.get(), &lease.lease, frame.size()));
  EXPECT_EQ(ctx.frames.size(), 0u);   // Frame not recorded due to error.
  EXPECT_EQ(lease.release_count, 1);  // Lease released even on callback error.
}

TEST(FrameAccumulatorTest, CallbackErrorStopsProcessing) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame1 = MakeFrame("First");
  auto frame2 = MakeFrame("Second");
  MockLease lease({frame1, frame2});

  ctx.next_error = IREE_STATUS_CANCELLED;  // Error on first frame.

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_CANCELLED,
      iree_net_frame_accumulator_push_lease(wrapper.get(), &lease.lease,
                                            frame1.size() + frame2.size()));

  // Only first frame attempted, second frame not processed.
  EXPECT_EQ(ctx.frames.size(), 0u);
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST(FrameAccumulatorTest, EmptyPayloadFrame) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("");  // Empty payload, just header.
  ASSERT_EQ(frame.size(), kHeaderSize);

  MockLease lease(frame);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease.lease, frame.size()));

  ASSERT_EQ(ctx.frames.size(), 1u);
  EXPECT_EQ(ctx.frames[0].data, frame);
  EXPECT_TRUE(ctx.frames[0].zero_copy);
}

TEST(FrameAccumulatorTest, ZeroValidBytesIsNoOp) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame = MakeFrame("test");
  MockLease lease(frame);

  IREE_ASSERT_OK(
      iree_net_frame_accumulator_push_lease(wrapper.get(), &lease.lease, 0));

  EXPECT_EQ(ctx.frames.size(), 0u);
  EXPECT_EQ(lease.release_count, 1);  // Lease still released.
}

TEST(FrameAccumulatorTest, StorageSizeCalculation) {
  constexpr iree_host_size_t kMaxFrameSize = 4096;
  iree_host_size_t expected_size =
      sizeof(iree_net_frame_accumulator_t) + kMaxFrameSize;
  EXPECT_EQ(iree_net_frame_accumulator_storage_size(kMaxFrameSize),
            expected_size);
}

TEST(FrameAccumulatorTest, BufferedBytesQuery) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);
  auto* acc = wrapper.get();

  EXPECT_EQ(iree_net_frame_accumulator_buffered_bytes(acc), 0u);

  auto frame = MakeFrame("test data here");
  // Send partial data.
  iree_host_size_t partial = 5;
  std::vector<uint8_t> partial_data(frame.begin(), frame.begin() + partial);
  MockLease lease(partial_data);
  IREE_ASSERT_OK(
      iree_net_frame_accumulator_push_lease(acc, &lease.lease, partial));

  EXPECT_EQ(iree_net_frame_accumulator_buffered_bytes(acc), partial);
}

TEST(FrameAccumulatorTest, MultipleFramesWithLeftover) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame1 = MakeFrame("A");
  auto frame2 = MakeFrame("BB");
  auto frame3 = MakeFrame("CCC");

  // Send frame1 + frame2 + half of frame3.
  std::vector<uint8_t> buffer;
  buffer.insert(buffer.end(), frame1.begin(), frame1.end());
  buffer.insert(buffer.end(), frame2.begin(), frame2.end());
  iree_host_size_t frame3_partial = frame3.size() / 2;
  buffer.insert(buffer.end(), frame3.begin(), frame3.begin() + frame3_partial);

  MockLease lease1(buffer);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease1.lease, buffer.size()));

  EXPECT_EQ(ctx.frames.size(), 2u);
  EXPECT_EQ(ctx.frames[0].data, frame1);
  EXPECT_EQ(ctx.frames[1].data, frame2);

  // Finish frame3.
  std::vector<uint8_t> rest(frame3.begin() + frame3_partial, frame3.end());
  MockLease lease2(rest);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease2.lease, rest.size()));

  EXPECT_EQ(ctx.frames.size(), 3u);
  EXPECT_EQ(ctx.frames[2].data, frame3);
}

// Verifies that completing a buffered frame transitions to zero-copy for
// subsequent frames in the same push.
TEST(FrameAccumulatorTest, BufferedToZeroCopyTransition) {
  TestContext ctx;
  constexpr iree_host_size_t kMaxFrameSize = 1024;
  AccumulatorWrapper wrapper(kMaxFrameSize, &ctx);

  auto frame1 = MakeFrame("FIRST");
  auto frame2 = MakeFrame("SECOND");
  auto frame3 = MakeFrame("THIRD");

  // Push partial frame1 (just the header).
  std::vector<uint8_t> header_only(frame1.begin(), frame1.begin() + 4);
  MockLease lease1(header_only);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease1.lease, header_only.size()));
  EXPECT_EQ(ctx.frames.size(), 0u);
  EXPECT_TRUE(iree_net_frame_accumulator_has_partial_frame(wrapper.get()));

  // Push rest of frame1 + frame2 + frame3.
  // After frame1 completes from buffer, frames 2 and 3 should be zero-copy.
  std::vector<uint8_t> rest_and_more;
  rest_and_more.insert(rest_and_more.end(), frame1.begin() + 4, frame1.end());
  rest_and_more.insert(rest_and_more.end(), frame2.begin(), frame2.end());
  rest_and_more.insert(rest_and_more.end(), frame3.begin(), frame3.end());
  MockLease lease2(rest_and_more);
  IREE_ASSERT_OK(iree_net_frame_accumulator_push_lease(
      wrapper.get(), &lease2.lease, rest_and_more.size()));

  ASSERT_EQ(ctx.frames.size(), 3u);

  // Frame 1 was completed from buffer (not zero-copy).
  EXPECT_EQ(ctx.frames[0].data, frame1);
  EXPECT_FALSE(ctx.frames[0].zero_copy);

  // Frames 2 and 3 should be zero-copy (not buffered).
  EXPECT_EQ(ctx.frames[1].data, frame2);
  EXPECT_TRUE(ctx.frames[1].zero_copy);
  EXPECT_EQ(ctx.frames[2].data, frame3);
  EXPECT_TRUE(ctx.frames[2].zero_copy);

  // No partial frame remaining.
  EXPECT_FALSE(iree_net_frame_accumulator_has_partial_frame(wrapper.get()));
}

}  // namespace
}  // namespace net
}  // namespace iree
