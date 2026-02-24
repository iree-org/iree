// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/passthrough.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedSegmenter;
using testing::ScopedSegmenterState;
using testing::TestLimitedOutputCapacity;
using testing::TestZeroCapacityOutput;

//===----------------------------------------------------------------------===//
// Test fixture providing a passthrough segmenter for each test.
//===----------------------------------------------------------------------===//

class PassthroughSegmenterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_passthrough_allocate(
        iree_allocator_system(), &raw_segmenter));
    segmenter_ = ScopedSegmenter(raw_segmenter);
  }

  iree_tokenizer_segmenter_t* segmenter() { return segmenter_.get(); }

 private:
  ScopedSegmenter segmenter_;
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, CreateAndDestroy) {
  // Segmenter was created in SetUp, will be destroyed in TearDown.
  // This test verifies the allocation/free path doesn't crash.
  EXPECT_NE(segmenter(), nullptr);
}

TEST_F(PassthroughSegmenterTest, StateSizeIsReasonable) {
  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter());
  // State should exist and be minimal (passthrough tracks almost nothing).
  EXPECT_GT(state_size, 0u);
  EXPECT_LE(state_size, 64u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, EmptyInput) {
  // Empty input should produce no segments and no pending data.
  auto segments = ProcessAndFinalize(segmenter(), iree_make_cstring_view(""),
                                     /*expect_pending_after_process=*/false);
  EXPECT_TRUE(segments.empty());
}

TEST_F(PassthroughSegmenterTest, ZeroCapacityOutput) {
  // Zero capacity output should consume nothing and produce nothing.
  TestZeroCapacityOutput(segmenter(), iree_make_cstring_view("Hello, World!"));
}

//===----------------------------------------------------------------------===//
// Edge Cases
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, SingleChar) {
  // Single character input becomes a single segment.
  auto segments = ProcessAndFinalize(segmenter(), iree_make_cstring_view("a"),
                                     /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "a");
}

//===----------------------------------------------------------------------===//
// Basic Functionality
// Passthrough emits entire input as one segment when fed all at once.
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, EntireInputAsOneSegment) {
  // Passthrough treats entire input as a single segment.
  auto segments =
      ProcessAndFinalize(segmenter(), iree_make_cstring_view("Hello, World!"),
                         /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "Hello, World!");
}

TEST_F(PassthroughSegmenterTest, MultiWordInputAsOneSegment) {
  // Whitespace does not split — entire string is one segment.
  auto segments = ProcessAndFinalize(
      segmenter(), iree_make_cstring_view("hello beautiful world"),
      /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "hello beautiful world");
}

TEST_F(PassthroughSegmenterTest, SpecialCharactersPreserved) {
  // Special characters are preserved in the single segment.
  auto segments =
      ProcessAndFinalize(segmenter(), iree_make_cstring_view("!@#$%^&*()"),
                         /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "!@#$%^&*()");
}

TEST_F(PassthroughSegmenterTest, WhitespaceOnlyInput) {
  // Even whitespace-only input becomes a segment (passthrough doesn't filter).
  auto segments =
      ProcessAndFinalize(segmenter(), iree_make_cstring_view("   \t\n   "),
                         /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "   \t\n   ");
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
// The has_pending() and finalize() calls are internally verified by test utils.
// Passthrough is unique: has_pending() is always false.
//
// Unlike other segmenters, passthrough produces one segment per process() call.
// Chunked input produces multiple segments matching chunks.
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, HasPendingAlwaysFalse) {
  // Passthrough never buffers data, so has_pending is always false.
  ScopedSegmenterState state(segmenter());

  // Before any processing.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Process some data.
  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("Hello"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  // Still no pending data.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

TEST_F(PassthroughSegmenterTest, FinalizeProducesNothing) {
  // Passthrough's finalize should never produce segments (nothing pending).
  ScopedSegmenterState state(segmenter());

  // Process some data first.
  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("Hello"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(segment_count, 1u);  // One segment from process.

  // Finalize should produce nothing (passthrough consumes all input).
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_string_view_empty(),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  EXPECT_EQ(finalize_count, 0u);
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

TEST_F(PassthroughSegmenterTest, ChunkedInputProducesMultipleSegments) {
  // Passthrough produces one segment per process() call.
  // This is intentional - it tests the vtable machinery, not segmentation.
  ScopedSegmenterState state(segmenter());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // First chunk: "Hello"
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view("Hello"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 5u);
  EXPECT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 5u);

  // Second chunk: " World"
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_cstring_view(" World"),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));
  EXPECT_EQ(consumed, 6u);
  EXPECT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 6u);

  // Finalize produces nothing (passthrough consumes all input).
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_string_view_empty(),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));
  EXPECT_EQ(finalize_count, 0u);
}

//===----------------------------------------------------------------------===//
// Output Buffer Handling
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, LimitedOutputCapacity) {
  // With capacity=1, passthrough should still work (one segment per call).
  TestLimitedOutputCapacity(segmenter(),
                            iree_make_cstring_view("Hello, World!"),
                            /*expected_segments=*/{"Hello, World!"});
}

//===----------------------------------------------------------------------===//
// Unicode Handling
// Passthrough preserves all bytes, including multi-byte UTF-8.
//===----------------------------------------------------------------------===//

TEST_F(PassthroughSegmenterTest, UnicodePreserved) {
  // Multi-byte UTF-8 sequences are preserved.
  auto segments =
      ProcessAndFinalize(segmenter(), iree_make_cstring_view("こんにちは世界"),
                         /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "こんにちは世界");
}

TEST_F(PassthroughSegmenterTest, MixedAsciiAndUnicode) {
  // Mixed ASCII and Unicode preserved as single segment.
  auto segments =
      ProcessAndFinalize(segmenter(), iree_make_cstring_view("Hello 世界!"),
                         /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "Hello 世界!");
}

TEST_F(PassthroughSegmenterTest, EmojiPreserved) {
  // Emoji (4-byte UTF-8) preserved.
  auto segments =
      ProcessAndFinalize(segmenter(), iree_make_cstring_view("Hello 🌍!"),
                         /*expect_pending_after_process=*/false);
  ASSERT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "Hello 🌍!");
}

}  // namespace
}  // namespace iree::tokenizer
