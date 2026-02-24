// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/strip.h"

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedNormalizer;
using testing::ScopedNormalizerState;
using testing::TestLimitedOutputCapacity;
using testing::TestWithAllChunkSizes;

//===----------------------------------------------------------------------===//
// Test fixture for strip normalizer tests.
//===----------------------------------------------------------------------===//

class StripNormalizerTest : public ::testing::Test {
 protected:
  // Creates a strip normalizer with the given settings.
  ScopedNormalizer CreateStrip(bool strip_left, bool strip_right) {
    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    IREE_EXPECT_OK(iree_tokenizer_normalizer_strip_allocate(
        strip_left, strip_right, iree_allocator_system(), &raw_normalizer));
    return ScopedNormalizer(raw_normalizer);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, CreateAndDestroy) {
  auto normalizer = CreateStrip(true, true);
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(StripNormalizerTest, CreateStripLeftOnly) {
  auto normalizer = CreateStrip(true, false);
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(StripNormalizerTest, CreateStripRightOnly) {
  auto normalizer = CreateStrip(false, true);
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(StripNormalizerTest, CreateStripNeither) {
  auto normalizer = CreateStrip(false, false);
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(StripNormalizerTest, StateSizeIsReasonable) {
  auto normalizer = CreateStrip(true, true);
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer.get());
  EXPECT_GT(state_size, 0u);
  // State includes 256-byte pending whitespace buffer for internal buffering
  // (avoids carryover overflow in sequence normalizers).
  EXPECT_LE(state_size, 512u);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, EmptyInput) {
  auto normalizer = CreateStrip(true, true);
  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_TRUE(result.empty());
}

TEST_F(StripNormalizerTest, NoWhitespaceUnchanged) {
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "hello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripNeitherPassesThrough) {
  auto normalizer = CreateStrip(false, false);
  TestWithAllChunkSizes(normalizer.get(), "  hello  ", "  hello  ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// strip_left Behavior
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, StripLeftSpaces) {
  auto normalizer = CreateStrip(true, false);
  TestWithAllChunkSizes(normalizer.get(), "   hello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripLeftTabs) {
  auto normalizer = CreateStrip(true, false);
  TestWithAllChunkSizes(normalizer.get(), "\t\thello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripLeftNewlines) {
  auto normalizer = CreateStrip(true, false);
  TestWithAllChunkSizes(normalizer.get(), "\n\r\nhello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripLeftMixedWhitespace) {
  auto normalizer = CreateStrip(true, false);
  TestWithAllChunkSizes(normalizer.get(), " \t\n\r hello", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripLeftPreservesTrailing) {
  auto normalizer = CreateStrip(true, false);
  TestWithAllChunkSizes(normalizer.get(), "  hello  ", "hello  ",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripLeftPreservesMiddle) {
  auto normalizer = CreateStrip(true, false);
  TestWithAllChunkSizes(normalizer.get(), "  hello world  ", "hello world  ",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// strip_right Behavior
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, StripRightSpaces) {
  auto normalizer = CreateStrip(false, true);
  TestWithAllChunkSizes(normalizer.get(), "hello   ", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripRightTabs) {
  auto normalizer = CreateStrip(false, true);
  TestWithAllChunkSizes(normalizer.get(), "hello\t\t", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripRightNewlines) {
  auto normalizer = CreateStrip(false, true);
  TestWithAllChunkSizes(normalizer.get(), "hello\n\r\n", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripRightMixedWhitespace) {
  auto normalizer = CreateStrip(false, true);
  TestWithAllChunkSizes(normalizer.get(), "hello \t\n\r ", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripRightPreservesLeading) {
  auto normalizer = CreateStrip(false, true);
  TestWithAllChunkSizes(normalizer.get(), "  hello  ", "  hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripRightPreservesMiddle) {
  auto normalizer = CreateStrip(false, true);
  TestWithAllChunkSizes(normalizer.get(), "  hello world  ", "  hello world",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// strip_left + strip_right Together
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, StripBothSides) {
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "  hello  ", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripBothWithMixedWhitespace) {
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "\t\n hello world \r\n",
                        "hello world", /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripBothPreservesMiddle) {
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "  hello   world  ", "hello   world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, AllWhitespaceBecomesEmpty) {
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "   \t\n  ", "",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Unicode Whitespace (Non-ASCII)
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, StripNBSP) {
  // No-Break Space (U+00A0) - 2 bytes UTF-8: 0xC2 0xA0.
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "\xC2\xA0hello\xC2\xA0", "hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripIdeographicSpace) {
  // Ideographic Space (U+3000) - 3 bytes UTF-8: 0xE3 0x80 0x80.
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "\xE3\x80\x80hello\xE3\x80\x80",
                        "hello", /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripEmSpace) {
  // Em Space (U+2003) - 3 bytes UTF-8: 0xE2 0x80 0x83.
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "\xE2\x80\x83hello\xE2\x80\x83",
                        "hello", /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, StripMixedAsciiAndUnicodeWhitespace) {
  // Mix of space, NBSP, tab, ideographic space.
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(),
                        " \xC2\xA0\t\xE3\x80\x80hello \xC2\xA0\n", "hello",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Non-whitespace Unicode Preserved
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, Utf8ContentPreserved) {
  // Strip spaces around UTF-8 content (café).
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "  caf\xC3\xA9  ", "caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, EmojiPreserved) {
  // 🎉 is U+1F389 (4-byte UTF-8).
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "  \xF0\x9F\x8E\x89  ",
                        "\xF0\x9F\x8E\x89",
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, CjkPreserved) {
  // 你好 (U+4F60 U+597D).
  auto normalizer = CreateStrip(true, true);
  TestWithAllChunkSizes(normalizer.get(), "  \xE4\xBD\xA0\xE5\xA5\xBD  ",
                        "\xE4\xBD\xA0\xE5\xA5\xBD",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming Behavior
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, StreamingStripRight) {
  // Test that strip_right correctly handles trailing whitespace using internal
  // buffering across multiple process() calls.
  //
  // With internal buffering, whitespace that MIGHT be trailing IS consumed
  // and stored internally. When non-WS follows, the buffered WS is emitted.
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  // First call: "hello ".
  // The space is potentially trailing, consumed and buffered internally.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 6u);  // All consumed (including space).
  EXPECT_EQ(written, 5u);   // "hello" emitted, space buffered.
  EXPECT_EQ(std::string(output, written), "hello");

  // Second call: "world".
  // The buffered " " is now confirmed intermediate (non-WS follows).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("world"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 5u);  // "world" consumed.
  EXPECT_EQ(written, 6u);   // Buffered " " + "world" emitted.
  EXPECT_EQ(std::string(output, written), " world");

  // Third call: "  " (trailing whitespace).
  // Consumed and buffered internally.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("  "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 2u);  // All consumed (buffered internally).
  EXPECT_EQ(written, 0u);   // Nothing emitted yet.

  // Finalize - buffered whitespace discarded (confirmed trailing).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), iree_make_mutable_string_view(output, sizeof(output)),
      &written));
  EXPECT_EQ(written, 0u);
}

TEST_F(StripNormalizerTest, LimitedOutputCapacity) {
  auto normalizer = CreateStrip(true, true);
  TestLimitedOutputCapacity(normalizer.get(), "  hello world  ", "hello world");
}

// Tests for partial UTF-8 handling were removed because the normalizer
// interface contract guarantees callers provide input on codepoint boundaries.

//===----------------------------------------------------------------------===//
// State Reset
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, StateResetClearsBuffers) {
  auto normalizer = CreateStrip(true, true);
  ScopedNormalizerState state(normalizer.get());

  // First use with trailing whitespace.
  char output[64];
  iree_host_size_t consumed = 0;
  iree_host_size_t written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("  hello  "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // Reset state.
  state.Reset();

  // New input should strip leading whitespace again.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("  world  "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(std::string(output, written), "world");
}

//===----------------------------------------------------------------------===//
// Long Trailing Whitespace
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, LongTrailingWhitespaceBuffer) {
  // Test that long trailing whitespace works correctly even when it exceeds
  // internal buffer size (should flush and emit intermediate whitespace).
  auto normalizer = CreateStrip(false, true);

  std::string input = "hello";
  // Add 100 spaces - more than the 64-byte trailing buffer.
  input += std::string(100, ' ');
  input += "world";
  input += std::string(50, ' ');  // Trailing whitespace.

  // Expected: "hello" + 100 spaces + "world" (trailing 50 spaces stripped).
  std::string expected = "hello";
  expected += std::string(100, ' ');
  expected += "world";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Trailing Whitespace Exceeding Buffer Size
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, TrailingWhitespaceExceedsBuffer) {
  // Trailing whitespace longer than IREE_STRIP_TRAILING_BUFFER_SIZE (64)
  // should still be fully stripped.
  auto normalizer = CreateStrip(false, true);

  std::string input = "hello";
  input += std::string(200, ' ');  // 200 trailing spaces > 64 byte buffer.

  std::string expected = "hello";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, TrailingWhitespaceExactlyBufferSize) {
  // Edge case: exactly 64 bytes of trailing whitespace.
  auto normalizer = CreateStrip(false, true);

  std::string input = "hello";
  input += std::string(64, ' ');

  std::string expected = "hello";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, TrailingWhitespaceJustOverBuffer) {
  // Edge case: 65 bytes (one over buffer).
  auto normalizer = CreateStrip(false, true);

  std::string input = "hello";
  input += std::string(65, ' ');

  std::string expected = "hello";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, TrailingUnicodeWhitespaceExceedsBuffer) {
  // Unicode whitespace (3-byte ideographic space U+3000) exceeding buffer.
  auto normalizer = CreateStrip(false, true);

  std::string input = "hello";
  // 30 ideographic spaces = 90 bytes > 64 byte buffer.
  for (int i = 0; i < 30; ++i) {
    input += "\xE3\x80\x80";  // U+3000 ideographic space.
  }

  std::string expected = "hello";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, TrailingOverflowCapacity1) {
  // Process trailing whitespace with output capacity=1.
  // With lazy consumption, trailing whitespace is NOT consumed, so
  // consumed=0 is valid when the normalizer is waiting for more input
  // or finalize.
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  std::string input = "hi";
  input += std::string(100, ' ');  // Trailing whitespace.

  std::string result;
  char output_char;
  size_t position = 0;

  while (position < input.size()) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    iree_string_view_t remaining = {input.data() + position,
                                    input.size() - position};
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), remaining, iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) {
      result += output_char;
    }
    if (consumed == 0 && written == 0) {
      // Normalizer is waiting for more input or finalize.
      // With lazy consumption, trailing whitespace is not consumed.
      break;
    }
    position += consumed;
  }

  // Finalize - any unconsumed input (trailing whitespace) is discarded.
  while (true) {
    iree_host_size_t finalize_written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(&output_char, 1),
        &finalize_written));
    if (finalize_written > 0) {
      result += output_char;
    } else {
      break;
    }
  }

  EXPECT_EQ(result, "hi");
}

//===----------------------------------------------------------------------===//
// Internal Buffering
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, EagerConsumptionBuffersTrailingWhitespace) {
  // Verify that all input is consumed, but trailing whitespace is buffered
  // internally (not emitted until we know it's intermediate).
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello   "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  EXPECT_EQ(consumed, 8u);  // All input consumed (including trailing spaces).
  EXPECT_EQ(written, 5u);   // Only "hello" emitted; spaces buffered internally.
  EXPECT_EQ(std::string(output, written), "hello");
}

TEST_F(StripNormalizerTest, IntermediateWhitespaceEmitted) {
  // Whitespace followed by non-WS in same chunk should be emitted.
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello   world"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  EXPECT_EQ(consumed, 13u);  // All consumed.
  EXPECT_EQ(written, 13u);   // All emitted.
  EXPECT_EQ(std::string(output, written), "hello   world");
}

TEST_F(StripNormalizerTest, CrossChunkIntermediateWhitespace) {
  // Whitespace at end of chunk 1, non-WS at start of chunk 2.
  // With internal buffering, caller sends just the next chunk (no prepending).
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  // Chunk 1: "hello   " - all consumed, whitespace buffered internally.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello   "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 8u);  // All consumed.
  EXPECT_EQ(std::string(output, written),
            "hello");  // WS buffered, not emitted.

  // Chunk 2: "world" - buffered whitespace is now confirmed intermediate.
  // When we see 'w', we flush the buffered "   " then emit "world".
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("world"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 5u);  // "world" consumed.
  EXPECT_EQ(std::string(output, written),
            "   world");  // Buffered WS + new content.
}

TEST_F(StripNormalizerTest, VeryLargeTrailingWhitespace) {
  // 10KB of trailing whitespace - no buffer limit with lazy consumption!
  auto normalizer = CreateStrip(false, true);

  std::string input = "hello";
  input += std::string(10000, ' ');  // 10KB trailing.

  std::string expected = "hello";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(StripNormalizerTest, AllWhitespaceInputBufferedInternally) {
  // Input that is entirely whitespace is consumed and buffered internally.
  // Nothing is emitted until we know if it's intermediate (non-WS follows)
  // or trailing (finalize called).
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("     "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  EXPECT_EQ(consumed, 5u);  // All consumed (buffered internally).
  EXPECT_EQ(written, 0u);   // Nothing emitted yet.
}

//===----------------------------------------------------------------------===//
// SEGMENT_END Flag
//===----------------------------------------------------------------------===//

TEST_F(StripNormalizerTest, SegmentEndDiscardsBufferedWhitespace) {
  // With internal buffering, whitespace is always consumed.
  // Without SEGMENT_END, it's buffered (might be intermediate).
  // With SEGMENT_END, the buffer is cleared (confirmed trailing).
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  // Without SEGMENT_END: whitespace consumed and buffered.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("     "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 5u);  // All consumed (buffered internally).
  EXPECT_EQ(written, 0u);   // Nothing emitted yet.

  // With SEGMENT_END: whitespace consumed and buffer cleared (trailing).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("     "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &consumed, &written));
  EXPECT_EQ(consumed, 5u);  // All consumed.
  EXPECT_EQ(written, 0u);   // Nothing emitted (trailing whitespace stripped).
}

TEST_F(StripNormalizerTest, SegmentEndWithMixedContent) {
  // "hello   " with SEGMENT_END: trailing spaces consumed but not emitted.
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello   "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &consumed, &written));
  EXPECT_EQ(consumed, 8u);  // All 8 bytes consumed.
  EXPECT_EQ(written, 5u);   // Only "hello" emitted.
  EXPECT_EQ(std::string(output, written), "hello");
}

TEST_F(StripNormalizerTest, SegmentEndResetsAtStart) {
  // After SEGMENT_END, at_start resets to true. Leading whitespace in the next
  // segment should be stripped by strip_left.
  auto normalizer =
      CreateStrip(true, true);  // Both strip_left and strip_right.
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  // Segment 1: "hello " - emits "hello", trailing space consumed by
  // SEGMENT_END.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("hello "),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &consumed, &written));
  EXPECT_EQ(consumed, 6u);
  EXPECT_EQ(std::string(output, written), "hello");

  // Segment 2: "  world" - at_start was reset, so leading spaces are stripped.
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("  world"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, 7u);
  EXPECT_EQ(std::string(output, written), "world");
}

TEST_F(StripNormalizerTest, SegmentEndWithUnicodeWhitespace) {
  // Unicode whitespace (U+3000, ideographic space) at segment end.
  auto normalizer = CreateStrip(false, true);
  ScopedNormalizerState state(normalizer.get());

  char output[64];
  iree_host_size_t consumed = 0, written = 0;

  // U+3000 is E3 80 80 in UTF-8 (ideographic space, 3 bytes).
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SV("A\xe3\x80\x80"),
      iree_make_mutable_string_view(output, sizeof(output)),
      IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END, &consumed, &written));
  EXPECT_EQ(consumed, 4u);  // 'A' + 3-byte Unicode space.
  EXPECT_EQ(written, 1u);   // Only 'A' emitted.
  EXPECT_EQ(output[0], 'A');
}

}  // namespace
}  // namespace iree::tokenizer
