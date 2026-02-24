// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/metaspace.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"

namespace iree::tokenizer::testing {
namespace {

// U+2581 LOWER ONE EIGHTH BLOCK (default Metaspace delimiter).
// UTF-8 encoding: 0xE2 0x96 0x81 (3 bytes).
constexpr const char* kDelimiter = "\xE2\x96\x81";

// Helper to create input strings with the delimiter.
std::string D(const std::string& str) { return std::string(kDelimiter) + str; }

//===----------------------------------------------------------------------===//
// 1. SMOKETEST: Lifecycle
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, CreateAndDestroy) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
      0, /*split_enabled=*/true, iree_allocator_system(), &segmenter));
  ASSERT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST(MetaspaceSegmenterTest, StateSizeIsReasonable) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter.get());
  // State should be small (< 128 bytes) for stack allocation.
  EXPECT_GT(state_size, 0u);
  EXPECT_LT(state_size, 128u);
}

//===----------------------------------------------------------------------===//
// 2. SMOKETEST: No-ops
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, EmptyInput) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  auto segments = ProcessAndFinalize(segmenter.get(), iree_string_view_empty(),
                                     /*expect_pending_after_process=*/false);
  EXPECT_TRUE(segments.empty());
}

TEST(MetaspaceSegmenterTest, ZeroCapacityOutput) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  TestZeroCapacityOutput(segmenter.get(), IREE_SVL("hello"));
}

//===----------------------------------------------------------------------===//
// 3. SMOKETEST: Edge Cases
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, OnlyDelimiters) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Three consecutive delimiters should produce three segments.
  std::string input =
      std::string(kDelimiter) + kDelimiter + kDelimiter;  // "▁▁▁"
  std::vector<std::string> expected = {kDelimiter, kDelimiter, kDelimiter};

  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        expected, /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, SingleNonDelimiterChar) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("a"), {"a"},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// 4. CORE: MergedWithNext Behavior
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, NoDelimiter) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello"), {"hello"},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, StartsWithDelimiter) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁hello" -> ["▁hello"]
  std::string input = D("hello");
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {input}, /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, EndsWithDelimiter) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "hello▁" -> ["hello", "▁"]
  std::string input = std::string("hello") + kDelimiter;
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {"hello", kDelimiter},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, MiddleDelimiter) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "hello▁world" -> ["hello", "▁world"]
  std::string input = std::string("hello") + kDelimiter + "world";
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {"hello", D("world")},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, ConsecutiveDelimiters) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁▁" -> ["▁", "▁"]
  std::string input = std::string(kDelimiter) + kDelimiter;
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {kDelimiter, kDelimiter},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, MultipleSegments) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁Hey▁friend▁" -> ["▁Hey", "▁friend", "▁"]
  std::string input = D("Hey") + kDelimiter + "friend" + kDelimiter;
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {D("Hey"), D("friend"), kDelimiter},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, RealWorldLlama) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Simulating Llama-style input after normalization:
  // "▁Hello▁world▁!" -> ["▁Hello", "▁world", "▁!"]
  std::string input = D("Hello") + kDelimiter + "world" + kDelimiter + "!";
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {D("Hello"), D("world"), D("!")},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// 5. CORE: Streaming
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, HasPendingCorrectness) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }
  ScopedSegmenterState state(segmenter.get());

  // Initially no pending.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // After processing non-delimiter content, has pending.
  iree_tokenizer_segment_t segments[16];
  iree_tokenizer_segment_output_t output = {16, segments};
  iree_host_size_t consumed = 0, count = 0;

  iree_string_view_t input = IREE_SVL("hello");
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, output, &consumed, &count));
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // After finalize, no pending.
  // Pull-based: finalize receives the unconsumed portion.
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + consumed, input.size - consumed);
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining_input, output, &count));
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

TEST(MetaspaceSegmenterTest, FinalizeEmitsPending) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  auto segments = ProcessAndFinalize(segmenter.get(), IREE_SVL("hello"),
                                     /*expect_pending_after_process=*/true);
  EXPECT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], "hello");
}

TEST(MetaspaceSegmenterTest, ByteAtATime) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Process "▁Hi▁!" byte by byte.
  // Input: "▁Hi▁!" = kDelimiter + "Hi" + kDelimiter + "!"
  // With MergedWithNext: ["▁Hi", "▁!"] - each delimiter merges with following.
  std::string input = D("Hi") + kDelimiter + "!";
  auto segments = ProcessChunkedAndFinalize(
      segmenter.get(), iree_make_string_view(input.c_str(), input.size()),
      /*chunk_size=*/1, /*expect_pending_after_process=*/true);
  EXPECT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], D("Hi"));
  EXPECT_EQ(segments[1], D("!"));
}

//===----------------------------------------------------------------------===//
// 6. CORE: UTF-8 Chunk Boundary
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, DelimiterSplitAtChunkBoundary) {
  // Tests that the "rewind strategy" correctly handles chunk boundaries that
  // split the UTF-8 delimiter (▁ = \xE2\x96\x81). The test utility
  // TestWithAllChunkSizes will test all chunk sizes from 1 to 64 bytes,
  // naturally testing cases where the delimiter is split after 1 or 2 bytes.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "ab▁cd" - when chunked at position 3 ("ab\xE2"), the segmenter should
  // rewind and not consume the partial delimiter byte.
  std::string input =
      "ab\xE2\x96\x81"
      "cd";  // "ab▁cd"
  std::string delimiter = "\xE2\x96\x81";
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {"ab", delimiter + "cd"},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, FalsePositiveStart) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // 0xE2 0x80 0x99 is RIGHT SINGLE QUOTATION MARK ('), not our delimiter.
  // This should NOT be treated as a segment boundary.
  std::string input =
      "a\xE2\x80\x99"
      "b";  // "a'b"
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {input}, /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, MixedUTF8Content) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Unicode content with delimiters: "▁日本語▁test"
  std::string japanese = "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E";  // 日本語
  std::string input = D(japanese) + kDelimiter + "test";
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {D(japanese), D("test")},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// 7. CORE: Output Buffer
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, LimitedOutputCapacity) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "a▁b▁c" with capacity=1 should work correctly.
  std::string input = std::string("a") + kDelimiter + "b" + kDelimiter + "c";
  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {"a", D("b"), D("c")});
}

//===----------------------------------------------------------------------===//
// 8. COMPONENT-SPECIFIC: Configuration
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, SplitDisabled) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/false, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // With split disabled, entire input is one segment.
  std::string input = D("Hey") + kDelimiter + "friend" + kDelimiter;
  auto segments = ProcessAndFinalize(
      segmenter.get(), iree_make_string_view(input.c_str(), input.size()),
      /*expect_pending_after_process=*/true);
  EXPECT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], input);
}

TEST(MetaspaceSegmenterTest, DefaultReplacementChar) {
  // 0 means use default (U+2581).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
      0, /*split_enabled=*/true, iree_allocator_system(), &segmenter));
  ASSERT_NE(segmenter, nullptr);

  ScopedSegmenter scoped(segmenter);
  std::string input = D("test");
  auto segments = ProcessAndFinalize(
      segmenter, iree_make_string_view(input.c_str(), input.size()),
      /*expect_pending_after_process=*/true);
  EXPECT_EQ(segments.size(), 1u);
  EXPECT_EQ(segments[0], input);
}

// Test case from cross-validated review: pending bytes that are NOT a
// delimiter. This verifies bytes aren't dropped when 0xE2 (start of ▁) is
// followed by non-delimiter continuation bytes.
TEST(MetaspaceSegmenterTest, FalsePositiveDelimiterByte) {
  // Tests that a lone 0xE2 byte (first byte of our delimiter ▁) followed by
  // non-matching continuation bytes is treated as regular content, not split.
  // This is similar to FalsePositiveStart but with a shorter false positive.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "A\xE2BC" - the \xE2 is followed by 'B' (0x42), not 0x96.
  // This is NOT a valid UTF-8 sequence, but should still be preserved as one
  // segment with no splits.
  std::string input =
      "A\xE2"
      "BC";  // 4 bytes total
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {input}, /*expect_pending_after_process=*/true);
}

// Cross-validation test: multiple consecutive false-positive delimiter starts.
// Tests 0xE2 appearing multiple times but never forming the actual delimiter.
TEST(MetaspaceSegmenterTest, MultipleFalsePositives) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Input with multiple 0xE2 bytes that are NOT delimiters (not followed by
  // 0x96 0x81). 0xE2 0x80 0x99 = RIGHT SINGLE QUOTATION MARK (') 0xE2 0x80 0x9C
  // = LEFT DOUBLE QUOTATION MARK (")
  std::string input =
      "a\xE2\x80\x99"
      "b\xE2\x80\x9C"
      "c";  // "a'b"c"
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {input}, /*expect_pending_after_process=*/true);
}

// Cross-validation test: delimiter immediately after false positive.
TEST(MetaspaceSegmenterTest, FalsePositiveThenRealDelimiter) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "a'▁b" - curly quote (0xE2 0x80 0x99) followed by real delimiter (0xE2 0x96
  // 0x81)
  std::string curly_quote = "\xE2\x80\x99";
  std::string input = "a" + curly_quote + kDelimiter + "b";
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {"a" + curly_quote, D("b")},
                        /*expect_pending_after_process=*/true);
}

// Cross-validation test: capacity exhaustion and resumption.
TEST(MetaspaceSegmenterTest, CapacityExhaustionResumption) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Input with 4 segments: "▁a▁b▁c▁d"
  std::string input = D("a") + D("b") + D("c") + D("d");

  // Use TestLimitedOutputCapacity which processes with capacity=1
  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {D("a"), D("b"), D("c"), D("d")});
}

TEST(MetaspaceSegmenterTest, CustomReplacementChar) {
  // Use ASCII underscore instead of U+2581.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
      '_', /*split_enabled=*/true, iree_allocator_system(), &segmenter));
  ASSERT_NE(segmenter, nullptr);

  ScopedSegmenter scoped(segmenter);

  // "hello_world" with underscore as delimiter.
  auto segments = ProcessAndFinalize(segmenter, IREE_SVL("hello_world"),
                                     /*expect_pending_after_process=*/true);
  EXPECT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "_world");
}

TEST(MetaspaceSegmenterTest, InvalidReplacementChar) {
  // Invalid Unicode codepoint should fail.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_segmenter_metaspace_allocate(
          0x110000,  // Beyond Unicode max (U+10FFFF).
          /*split_enabled=*/true, iree_allocator_system(), &segmenter));
}

//===----------------------------------------------------------------------===//
// STREAMING OFFSETS: Verify absolute offset reconstruction matches HuggingFace
//===----------------------------------------------------------------------===//

// Single process + finalize with multi-byte delimiter (▁ = 3 bytes).
// Process emits "ab" when it hits the delimiter, then finalize emits "▁cd".
//
// HuggingFace ground truth:
//   Metaspace(replacement="▁",
//   prepend_scheme="never").pre_tokenize_str("ab▁cd") → [("ab", (0, 2)),
//   ("▁cd", (2, 7))]  (byte offsets)
TEST(MetaspaceSegmenterTest, StreamingOffsets_ProcessAndFinalize) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }
  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // "ab▁cd" = 7 bytes (2 + 3 + 2).
  std::string input = std::string("ab") + kDelimiter + "cd";
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view(input.c_str(), input.size()),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 2u);
  ASSERT_EQ(segment_count, 1u);
  // Verify absolute: "ab" at [0, 2).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 2u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize with remaining "▁cd" (5 bytes).
  iree_host_size_t position = consumed;
  std::string remaining = std::string(kDelimiter) + "cd";
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view(remaining.c_str(), remaining.size()),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: ("▁cd", (2, 7)).
  EXPECT_EQ(position + segments[0].start, 2u);
  EXPECT_EQ(position + segments[0].end, 7u);
}

// Partial delimiter at chunk boundary forces rewind (consumed=0).
// When the chunk ends mid-delimiter, the segmenter can't determine if it's
// the real delimiter or a false positive (another 0xE2-starting character).
// The rewind leaves all bytes for finalize.
//
// HuggingFace ground truth:
//   Metaspace(replacement="▁",
//   prepend_scheme="never").pre_tokenize_str("ab▁cd") → [("ab", (0, 2)),
//   ("▁cd", (2, 7))]  (byte offsets)
TEST(MetaspaceSegmenterTest, StreamingOffsets_PartialDelimiterRewind) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }
  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Feed only "ab\xE2" (3 bytes) — the chunk ends with the first byte of ▁.
  // The segmenter rewinds: consumed=0 because it can't confirm the delimiter.
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view("ab\xE2", 3),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 0u);
  EXPECT_EQ(segment_count, 0u);

  // Finalize with full input "ab▁cd" (7 bytes) since nothing was consumed.
  iree_host_size_t position = consumed;
  std::string full_input = std::string("ab") + kDelimiter + "cd";
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view(full_input.c_str(), full_input.size()),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 2u);
  // Verify absolute reconstruction matches HF: ("ab", (0, 2)), ("▁cd", (2, 7)).
  EXPECT_EQ(position + segments[0].start, 0u);
  EXPECT_EQ(position + segments[0].end, 2u);
  EXPECT_EQ(position + segments[1].start, 2u);
  EXPECT_EQ(position + segments[1].end, 7u);
}

// Multiple delimiters with process emitting multiple segments.
// The trailing segment is left for finalize.
//
// HuggingFace ground truth:
//   Metaspace(replacement="▁",
//   prepend_scheme="never").pre_tokenize_str("a▁b▁c") → [("a", (0, 1)), ("▁b",
//   (1, 5)), ("▁c", (5, 9))]  (byte offsets)
TEST(MetaspaceSegmenterTest, StreamingOffsets_MultiChunkDelimiters) {
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }
  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // "a▁b▁c" = 9 bytes (1+3+1+3+1). Process emits "a" and "▁b", leaves "▁c"
  // pending.
  std::string input = std::string("a") + kDelimiter + "b" + kDelimiter + "c";
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view(input.c_str(), input.size()),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 5u);
  ASSERT_EQ(segment_count, 2u);
  // Verify absolute: "a" at [0, 1) and "▁b" at [1, 5).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 1u);
  EXPECT_EQ(segments[1].start, 1u);
  EXPECT_EQ(segments[1].end, 5u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize with remaining "▁c" (4 bytes).
  iree_host_size_t position = consumed;
  std::string remaining = std::string(kDelimiter) + "c";
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view(remaining.c_str(), remaining.size()),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Verify absolute reconstruction matches HF: ("▁c", (5, 9)).
  EXPECT_EQ(position + segments[0].start, 5u);
  EXPECT_EQ(position + segments[0].end, 9u);
}

//===----------------------------------------------------------------------===//
// CONSECUTIVE DELIMITER EDGE CASES
// These tests verify correct handling of consecutive delimiters which is
// relevant to whitespace tokenization failures where normalized whitespace
// produces multiple consecutive ▁ characters.
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, ConsecutiveDelimiters_FourInARow) {
  // Four consecutive delimiters should produce four separate segments.
  // This is critical for models like ALBERT/XLNet where "    " (4 spaces)
  // normalizes to "▁▁▁▁" and each delimiter starts a new segment.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  std::string input =
      std::string(kDelimiter) + kDelimiter + kDelimiter + kDelimiter;  // "▁▁▁▁"
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {kDelimiter, kDelimiter, kDelimiter, kDelimiter},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, ConsecutiveDelimiters_ThenWord) {
  // Multiple consecutive delimiters followed by a word.
  // Tests that the word properly merges with the preceding delimiter.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁▁▁word" -> ["▁", "▁", "▁word"]
  std::string input =
      std::string(kDelimiter) + kDelimiter + kDelimiter + "word";
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {kDelimiter, kDelimiter, D("word")},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, WordThenConsecutiveDelimiters) {
  // Word followed by multiple consecutive delimiters.
  // Relevant to inputs like "word   " which normalize to "▁word▁▁▁".
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁word▁▁▁" -> ["▁word", "▁", "▁", "▁"]
  std::string input = D("word") + kDelimiter + kDelimiter + kDelimiter;
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {D("word"), kDelimiter, kDelimiter, kDelimiter},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, InterleaveWordAndConsecutiveDelimiters) {
  // Tests the "  multiple   spaces  " pattern that fails in smoketests.
  // After normalization, this becomes "▁▁multiple▁▁▁spaces▁▁".
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁▁word▁▁▁other▁▁" (simulating "  word   other  " after normalization)
  std::string input = std::string(kDelimiter) + kDelimiter + "word" +
                      kDelimiter + kDelimiter + kDelimiter + "other" +
                      kDelimiter + kDelimiter;
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {kDelimiter, D("word"), kDelimiter, kDelimiter,
                         D("other"), kDelimiter, kDelimiter},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// MIXED WHITESPACE NORMALIZATION SCENARIOS
// These tests simulate inputs that would be produced by Replace normalizer
// transforming tabs and newlines into spaces (then prepended with ▁).
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, SimulatedTabNormalization) {
  // Simulates input after normalizing "word\tword" where \t becomes ▁.
  // The segmenter receives "▁word▁word" (with initial ▁ from prepend).
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // "▁word▁word" -> ["▁word", "▁word"]
  std::string input = D("word") + D("word");
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {D("word"), D("word")},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, SimulatedNewlineNormalization) {
  // Simulates input after normalizing "line1\nline2\nline3" where each \n
  // becomes ▁. With prepend, this becomes "▁line1▁line2▁line3".
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  std::string input = D("line1") + D("line2") + D("line3");
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {D("line1"), D("line2"), D("line3")},
                        /*expect_pending_after_process=*/true);
}

TEST(MetaspaceSegmenterTest, SimulatedMixedWhitespaceNormalization) {
  // Simulates "  multiple   spaces   and\ttabs\nand\nnewlines  " after
  // normalization, which would become a sequence of ▁ and text.
  // This matches the actual failing smoketest input pattern.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }

  // Simplified version: "▁▁multiple▁▁▁spaces▁and▁tabs▁"
  std::string input = std::string(kDelimiter) + D("multiple") + kDelimiter +
                      kDelimiter + D("spaces") + D("and") + D("tabs") +
                      kDelimiter;
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {kDelimiter, D("multiple"), kDelimiter, kDelimiter,
                         D("spaces"), D("and"), D("tabs"), kDelimiter},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// SEGMENT BOUNDARY VERIFICATION
// These tests verify that segment boundaries are correctly reported, which
// is important for the downstream model (Unigram/BPE) to receive proper inputs.
//===----------------------------------------------------------------------===//

TEST(MetaspaceSegmenterTest, SegmentBoundaries_SingleDelimiter) {
  // Verify exact byte offsets for a simple case with single delimiter.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }
  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // "word▁next" = 11 bytes (4 + 3 + 4)
  std::string input = std::string("word") + kDelimiter + "next";
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view(input.c_str(), input.size()),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  // Should emit "word" and leave "▁next" pending.
  EXPECT_EQ(consumed, 4u);
  ASSERT_EQ(segment_count, 1u);
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 4u);

  // Finalize to get the remaining segment.
  iree_host_size_t position = consumed;
  std::string remaining = std::string(kDelimiter) + "next";
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view(remaining.c_str(), remaining.size()),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // "▁next" starts at byte 4 (after "word") and ends at byte 11 (4 + 3 + 4).
  EXPECT_EQ(position + segments[0].start, 4u);
  EXPECT_EQ(position + segments[0].end, 11u);
}

TEST(MetaspaceSegmenterTest, SegmentBoundaries_ConsecutiveDelimiters) {
  // Verify exact byte offsets when multiple consecutive delimiters appear.
  // Each ▁ is 3 bytes.
  ScopedSegmenter segmenter;
  {
    iree_tokenizer_segmenter_t* raw = nullptr;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw));
    segmenter = ScopedSegmenter(raw);
  }
  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // "▁▁▁" = 9 bytes (3 + 3 + 3)
  std::string input = std::string(kDelimiter) + kDelimiter + kDelimiter;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view(input.c_str(), input.size()),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  // Should emit first two ▁ segments, leaving the third pending.
  EXPECT_EQ(consumed, 6u);
  ASSERT_EQ(segment_count, 2u);
  // First ▁: [0, 3)
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 3u);
  // Second ▁: [3, 6)
  EXPECT_EQ(segments[1].start, 3u);
  EXPECT_EQ(segments[1].end, 6u);

  // Finalize to get the third ▁.
  iree_host_size_t position = consumed;
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view(kDelimiter, 3),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 1u);
  // Third ▁: [6, 9)
  EXPECT_EQ(position + segments[0].start, 6u);
  EXPECT_EQ(position + segments[0].end, 9u);
}

}  // namespace
}  // namespace iree::tokenizer::testing
