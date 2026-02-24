// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/sequence.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter/metaspace.h"
#include "iree/tokenizer/segmenter/punctuation.h"
#include "iree/tokenizer/segmenter/segmenter_test_util.h"
#include "iree/tokenizer/segmenter/split.h"
#include "iree/tokenizer/segmenter/whitespace.h"

namespace iree::tokenizer::testing {
namespace {

// U+2581 LOWER ONE EIGHTH BLOCK (Metaspace delimiter).
constexpr const char* kMetaspace = "\xE2\x96\x81";

//===----------------------------------------------------------------------===//
// Test Fixtures
//===----------------------------------------------------------------------===//

class SequenceSegmenterTest : public ::testing::Test {
 protected:
  // Helper to create a sequence from a list of segmenters.
  // Takes ownership of the ScopedSegmenters: on success they are released to
  // the sequence, on failure they are cleaned up automatically.
  ScopedSegmenter CreateSequence(std::vector<ScopedSegmenter> children) {
    std::vector<iree_tokenizer_segmenter_t*> raw_children;
    raw_children.reserve(children.size());
    for (const auto& child : children) {
      raw_children.push_back(child.get());
    }

    iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
    iree_status_t status = iree_tokenizer_segmenter_sequence_allocate(
        raw_children.data(), raw_children.size(), iree_allocator_system(),
        &raw_segmenter);
    if (!iree_status_is_ok(status)) {
      return ScopedSegmenter(nullptr);
    }

    // Success: release all children to sequence ownership.
    for (auto& child : children) {
      child.release();
    }
    return ScopedSegmenter(raw_segmenter);
  }

  // Creates a metaspace segmenter with splitting enabled.
  ScopedSegmenter CreateMetaspace() {
    iree_tokenizer_segmenter_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_segmenter_metaspace_allocate(
        0, /*split_enabled=*/true, iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedSegmenter(nullptr);
    }
    return ScopedSegmenter(raw);
  }

  // Creates a whitespace segmenter.
  ScopedSegmenter CreateWhitespace() {
    iree_tokenizer_segmenter_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_segmenter_whitespace_allocate(
        iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedSegmenter(nullptr);
    }
    return ScopedSegmenter(raw);
  }

  // Helper to build metaspace-delimited string.
  static std::string M(const std::string& s) { return kMetaspace + s; }

  // Creates a Punctuation segmenter with the given behavior.
  ScopedSegmenter CreatePunctuation(
      iree_tokenizer_regex_split_behavior_t behavior =
          IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS) {
    iree_tokenizer_segmenter_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_segmenter_punctuation_allocate(
        behavior, iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return ScopedSegmenter(nullptr);
    }
    return ScopedSegmenter(raw);
  }

  // Creates a Split segmenter from a regex pattern.
  ScopedSegmenter CreateSplit(const char* pattern,
                              iree_tokenizer_regex_split_behavior_t behavior =
                                  IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED) {
    iree_tokenizer_regex_compile_error_t error = {0};
    iree_tokenizer_regex_dfa_t dfa;
    uint8_t* dfa_storage = nullptr;
    iree_status_t status = iree_tokenizer_regex_compile_and_load(
        iree_make_cstring_view(pattern),
        IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
        &dfa, &dfa_storage, &error);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return ScopedSegmenter(nullptr);
    }

    iree_tokenizer_segmenter_t* raw = nullptr;
    status = iree_tokenizer_segmenter_split_allocate(
        dfa, dfa_storage, behavior, /*invert=*/false, iree_allocator_system(),
        &raw);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), dfa_storage);
      iree_status_ignore(status);
      return ScopedSegmenter(nullptr);
    }
    // Ownership of dfa_storage transferred to segmenter.
    return ScopedSegmenter(raw);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, CreateAndDestroy) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  ASSERT_NE(children[0].get(), nullptr);
  ASSERT_NE(children[1].get(), nullptr);

  auto segmenter = CreateSequence(std::move(children));
  EXPECT_NE(segmenter.get(), nullptr);
}

TEST_F(SequenceSegmenterTest, CreateWithManyChildren) {
  std::vector<ScopedSegmenter> children;
  for (int i = 0; i < 4; ++i) {
    children.push_back(CreateWhitespace());
    ASSERT_NE(children.back().get(), nullptr);
  }

  auto segmenter = CreateSequence(std::move(children));
  EXPECT_NE(segmenter.get(), nullptr);
}

TEST_F(SequenceSegmenterTest, StateSizeIncludesChildren) {
  // Get child state sizes before transferring ownership.
  ScopedSegmenter child1 = CreateWhitespace();
  ScopedSegmenter child2 = CreateMetaspace();
  ASSERT_NE(child1.get(), nullptr);
  ASSERT_NE(child2.get(), nullptr);

  iree_host_size_t child1_state_size =
      iree_tokenizer_segmenter_state_size(child1.get());
  iree_host_size_t child2_state_size =
      iree_tokenizer_segmenter_state_size(child2.get());

  std::vector<ScopedSegmenter> children;
  children.push_back(std::move(child1));
  children.push_back(std::move(child2));
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  iree_host_size_t state_size =
      iree_tokenizer_segmenter_state_size(segmenter.get());

  // Sequence state must be at least base + both child states.
  EXPECT_GE(state_size, child1_state_size + child2_state_size);
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, RejectsExcessiveDepth) {
  std::vector<ScopedSegmenter> children;
  std::vector<iree_tokenizer_segmenter_t*> raw_pointers;

  for (int i = 0; i < IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH + 1; ++i) {
    ScopedSegmenter child = CreateWhitespace();
    ASSERT_NE(child.get(), nullptr);
    raw_pointers.push_back(child.get());
    children.push_back(std::move(child));
  }

  iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
  iree_status_t status = iree_tokenizer_segmenter_sequence_allocate(
      raw_pointers.data(), raw_pointers.size(), iree_allocator_system(),
      &raw_segmenter);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceSegmenterTest, RejectsTooFewChildren) {
  ScopedSegmenter child = CreateWhitespace();
  ASSERT_NE(child.get(), nullptr);

  iree_tokenizer_segmenter_t* raw_child = child.get();
  iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
  iree_status_t status = iree_tokenizer_segmenter_sequence_allocate(
      &raw_child, 1, iree_allocator_system(), &raw_segmenter);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceSegmenterTest, RejectsZeroChildren) {
  iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
  iree_status_t status = iree_tokenizer_segmenter_sequence_allocate(
      nullptr, 0, iree_allocator_system(), &raw_segmenter);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceSegmenterTest, RejectsNullChildren) {
  ScopedSegmenter child1 = CreateWhitespace();
  ASSERT_NE(child1.get(), nullptr);

  std::vector<iree_tokenizer_segmenter_t*> raw_children = {child1.get(),
                                                           nullptr};

  iree_tokenizer_segmenter_t* raw_segmenter = nullptr;
  iree_status_t status = iree_tokenizer_segmenter_sequence_allocate(
      raw_children.data(), raw_children.size(), iree_allocator_system(),
      &raw_segmenter);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

//===----------------------------------------------------------------------===//
// No-ops
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, EmptyInput) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  auto segments = ProcessAndFinalize(segmenter.get(), iree_string_view_empty(),
                                     /*expect_pending_after_process=*/false);
  EXPECT_TRUE(segments.empty());
}

TEST_F(SequenceSegmenterTest, ZeroCapacityOutput) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  TestZeroCapacityOutput(segmenter.get(), IREE_SVL("hello"));
}

//===----------------------------------------------------------------------===//
// Whitespace -> Metaspace Chain
// This is a realistic chain: first split by whitespace, then by metaspace.
// Input: "▁hello▁world ▁foo▁bar"
// After whitespace: ["▁hello▁world", "▁foo▁bar"]
// After metaspace on each: ["▁hello", "▁world", "▁foo", "▁bar"]
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, WhitespaceThenMetaspace) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Build input: "▁hello▁world ▁foo▁bar"
  std::string input = M("hello") + M("world") + " " + M("foo") + M("bar");

  // Whitespace splits into ["▁hello▁world", "▁foo▁bar"]
  // Metaspace further splits each: ["▁hello", "▁world", "▁foo", "▁bar"]
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {M("hello"), M("world"), M("foo"), M("bar")},
                        /*expect_pending_after_process=*/true);
}

TEST_F(SequenceSegmenterTest, WhitespaceThenMetaspaceSimple) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Simpler test: "▁hello ▁world" (one metaspace token per whitespace word)
  std::string input = M("hello") + " " + M("world");

  // Whitespace: ["▁hello", "▁world"]
  // Metaspace on each: ["▁hello"], ["▁world"] (no further splits)
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {M("hello"), M("world")},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Two Whitespace Segmenters
// Double whitespace is identity - whitespace segments have no internal spaces.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, TwoWhitespaceIdentity) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateWhitespace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // First whitespace: ["hello", "world", "test"]
  // Second whitespace on each word: same (no spaces in words)
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL("hello world test"),
                        {"hello", "world", "test"},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Two Metaspace Segmenters
// Double metaspace is identity - each segment starts with exactly one ▁.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, TwoMetaspaceIdentity) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateMetaspace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input: "▁hello▁world"
  std::string input = M("hello") + M("world");

  // First metaspace: ["▁hello", "▁world"]
  // Second metaspace on each: same (one ▁ per segment)
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {M("hello"), M("world")},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Output Capacity: Limited Capacity Stress Test
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, LimitedOutputCapacity) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input with multiple segments to stress capacity handling.
  std::string input = M("hello") + M("world") + " " + M("foo") + M("bar");

  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {M("hello"), M("world"), M("foo"), M("bar")});
}

//===----------------------------------------------------------------------===//
// Has Pending: Verify Pending State Tracking
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, HasPendingTracking) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  // Initially no pending.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Process input without trailing whitespace - last word is pending.
  iree_tokenizer_segment_t segments[8];
  auto output = iree_tokenizer_make_segment_output(segments, 8);
  iree_host_size_t consumed = 0, count = 0;
  iree_string_view_t input = IREE_SVL("hello world");
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, output, &consumed, &count));

  // "world" has no trailing whitespace, so whitespace child has pending.
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // After finalize, no pending.
  // Pull-based: finalize receives the unconsumed portion.
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + consumed, input.size - consumed);
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining_input, output, &count));
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()));
}

//===----------------------------------------------------------------------===//
// Backpressure: Process with Capacity=1
// Verifies consumed=0 until entire expansion tree for a root segment is
// emitted.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, ProcessBackpressureCapacity1) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  // Input: "▁a▁b ▁c" - one whitespace word "▁a▁b" that splits into 2 metaspace
  // tokens, followed by another whitespace word "▁c".
  std::string input = M("a") + M("b") + " " + M("c");
  iree_string_view_t input_sv =
      iree_make_string_view(input.data(), input.size());

  iree_tokenizer_segment_t segment;
  auto output = iree_tokenizer_make_segment_output(&segment, 1);

  std::vector<std::string> results;
  iree_host_size_t total_consumed = 0;

  // Process with capacity=1 repeatedly.
  while (total_consumed < input.size()) {
    iree_string_view_t remaining = iree_make_string_view(
        input_sv.data + total_consumed, input_sv.size - total_consumed);
    iree_host_size_t consumed = 0, count = 0;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
        state.get(), remaining, output, &consumed, &count));

    if (count > 0) {
      results.push_back(std::string(remaining.data + segment.start,
                                    segment.end - segment.start));
    }
    total_consumed += consumed;

    // Safety: if no progress, break.
    if (consumed == 0 && count == 0) break;
  }

  // Finalize to get any remaining segments.
  iree_string_view_t remaining = iree_make_string_view(
      input_sv.data + total_consumed, input_sv.size - total_consumed);
  while (true) {
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
        state.get(), remaining, output, &count));
    if (count == 0) break;
    results.push_back(std::string(remaining.data + segment.start,
                                  segment.end - segment.start));
  }

  // Should get all 3 segments in order.
  ASSERT_EQ(results.size(), 3);
  EXPECT_EQ(results[0], M("a"));
  EXPECT_EQ(results[1], M("b"));
  EXPECT_EQ(results[2], M("c"));
}

//===----------------------------------------------------------------------===//
// Offset Continuity: Finalize with Capacity=1
// Verifies segment offsets are correct across multiple finalize() calls.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, FinalizeOffsetContinuity) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  // Input with no trailing whitespace so everything goes to finalize.
  // "▁A▁B▁C" - three metaspace segments from one whitespace word.
  std::string input = M("A") + M("B") + M("C");
  iree_string_view_t input_sv =
      iree_make_string_view(input.data(), input.size());

  // Process first - should consume nothing (no trailing whitespace).
  iree_tokenizer_segment_t segments[8];
  auto output = iree_tokenizer_make_segment_output(segments, 8);
  iree_host_size_t consumed = 0, count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input_sv, output, &consumed, &count));
  EXPECT_EQ(consumed, 0);
  EXPECT_EQ(count, 0);

  // Now finalize with capacity=1, verifying offsets.
  iree_tokenizer_segment_t segment;
  auto single_output = iree_tokenizer_make_segment_output(&segment, 1);

  struct Expected {
    std::string text;
    size_t start;
    size_t end;
  };
  // Each ▁X is 4 bytes (3-byte metaspace + 1-byte letter).
  std::vector<Expected> expected = {
      {M("A"), 0, 4},
      {M("B"), 4, 8},
      {M("C"), 8, 12},
  };

  for (size_t i = 0; i < expected.size(); ++i) {
    count = 0;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
        state.get(), input_sv, single_output, &count));
    ASSERT_EQ(count, 1) << "Iteration " << i;

    std::string text(input_sv.data + segment.start,
                     segment.end - segment.start);
    EXPECT_EQ(text, expected[i].text) << "Iteration " << i;
    EXPECT_EQ(segment.start, expected[i].start)
        << "Start offset mismatch for " << expected[i].text;
    EXPECT_EQ(segment.end, expected[i].end)
        << "End offset mismatch for " << expected[i].text;
  }

  // Should have no more segments.
  count = 99;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), input_sv, single_output, &count));
  EXPECT_EQ(count, 0);
}

//===----------------------------------------------------------------------===//
// Deep Chain: 3-Level Recursion
// Tests whitespace -> metaspace -> metaspace chain with skip_count logic.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, ThreeChildChain) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  children.push_back(CreateMetaspace());  // Identity (second metaspace).
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input: "▁a▁b ▁c"
  std::string input = M("a") + M("b") + " " + M("c");

  // Chain behavior:
  // Whitespace: ["▁a▁b", "▁c"]
  // Metaspace on "▁a▁b": ["▁a", "▁b"]
  // Metaspace on "▁c": ["▁c"]
  // Second metaspace is identity.
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {M("a"), M("b"), M("c")},
                        /*expect_pending_after_process=*/true);
}

TEST_F(SequenceSegmenterTest, ThreeChildChainCapacity1) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  children.push_back(CreateMetaspace());  // Identity.
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input: "▁a▁b"
  std::string input = M("a") + M("b");

  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {M("a"), M("b")});
}

//===----------------------------------------------------------------------===//
// Pathological: All Whitespace
// Input with only whitespace produces no segments.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, PathologicalAllWhitespace) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  auto segments = ProcessAndFinalize(segmenter.get(), IREE_SVL("     "),
                                     /*expect_pending_after_process=*/false);
  EXPECT_TRUE(segments.empty());
}

//===----------------------------------------------------------------------===//
// Pathological: Consecutive Metaspace Delimiters
// Multiple consecutive ▁ characters.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, PathologicalConsecutiveMetaspace) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input: "▁▁▁" (three consecutive metaspace characters, no other content).
  // This is treated as one whitespace word "▁▁▁".
  // Metaspace splits on ▁, so we get three empty-ish segments.
  std::string input = std::string(kMetaspace) + kMetaspace + kMetaspace;

  // Metaspace behavior: each ▁ starts a new segment.
  // With split_enabled, consecutive ▁ produces segments that are just ▁.
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {kMetaspace, kMetaspace, kMetaspace},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Pathological: Mixed Delimiters
// Whitespace between metaspace tokens.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, PathologicalMixedDelimiters) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input: " ▁a ▁b " (leading/trailing whitespace, single chars).
  std::string input = " " + M("a") + " " + M("b") + " ";

  // Whitespace: ["▁a", "▁b"] (skips leading/trailing/inter-word whitespace)
  // Metaspace on each: ["▁a"], ["▁b"] (single tokens, no further splits)
  TestWithAllChunkSizes(
      segmenter.get(), iree_make_string_view(input.c_str(), input.size()),
      {M("a"), M("b")},
      /*expect_pending_after_process=*/false);  // Trailing whitespace = no
                                                // pending.
}

//===----------------------------------------------------------------------===//
// Streaming: UTF-8 Delimiter Split Across Chunks
// Verifies the metaspace child handles partial UTF-8 correctly.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, Utf8DelimiterSplitAcrossChunks) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Input: "▁hello▁world" where the ▁ between hello and world
  // will be split across chunks at various points.
  std::string input = M("hello") + M("world");

  // TestWithAllChunkSizes exercises chunk sizes 1,2,3,5,8,13,16,32,64
  // which will split the 3-byte ▁ delimiter at different points.
  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {M("hello"), M("world")},
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// Consumption Granularity: Verify Consumed Tracking
// Tests that consumed advances only after full expansion tree is emitted.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, ConsumptionGranularityMultiExpand) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  // Input: "▁a▁b▁c " - one whitespace word that expands to 3 metaspace tokens,
  // with trailing space so whitespace can emit during process().
  std::string input = M("a") + M("b") + M("c") + " ";
  iree_string_view_t input_sv =
      iree_make_string_view(input.data(), input.size());

  iree_tokenizer_segment_t segment;
  auto output = iree_tokenizer_make_segment_output(&segment, 1);

  std::vector<std::string> results;
  std::vector<iree_host_size_t> consumed_values;

  // Call process repeatedly with capacity=1.
  iree_host_size_t total_consumed = 0;
  while (total_consumed < input.size()) {
    iree_string_view_t remaining = iree_make_string_view(
        input_sv.data + total_consumed, input_sv.size - total_consumed);
    iree_host_size_t consumed = 0, count = 0;
    IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
        state.get(), remaining, output, &consumed, &count));

    if (count > 0) {
      results.push_back(std::string(remaining.data + segment.start,
                                    segment.end - segment.start));
    }
    consumed_values.push_back(consumed);

    if (consumed > 0) {
      total_consumed += consumed;
    } else if (count == 0) {
      break;  // No progress.
    }
  }

  // Should get all 3 segments.
  ASSERT_EQ(results.size(), 3);
  EXPECT_EQ(results[0], M("a"));
  EXPECT_EQ(results[1], M("b"));
  EXPECT_EQ(results[2], M("c"));

  // Key assertion: consumed should be 0 for the first two outputs,
  // then jump to the full word length on the third.
  // (Implementation may vary, but must be consistent.)
  // The important thing is total_consumed equals input.size() at the end.
  EXPECT_EQ(total_consumed, input.size());
}

//===----------------------------------------------------------------------===//
// Streaming Offsets: Verify Absolute Offset Reconstruction Matches HuggingFace
//===----------------------------------------------------------------------===//

// Cross-chunk streaming through the full chain. Process emits the first
// whitespace word's metaspace expansions, finalize emits the second word's.
//
// Input: "▁a▁b ▁c▁d" (17 bytes)
// Whitespace splits at the space into: ["▁a▁b", "▁c▁d"]
// Metaspace expands each: ["▁a", "▁b"] + ["▁c", "▁d"]
//
// HuggingFace ground truth:
//   Sequence([WhitespaceSplit(), Metaspace(replacement="▁",
//   prepend_scheme="never")]).pre_tokenize_str("▁a▁b ▁c▁d")
//   → [("▁a", (0, 4)), ("▁b", (4, 8)), ("▁c", (9, 13)), ("▁d", (13, 17))]
//   (byte offsets)
TEST_F(SequenceSegmenterTest, StreamingOffsets_CrossChunkThroughChain) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[8];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // "▁a▁b ▁c▁d" = 17 bytes: (3+1)+(3+1)+1+(3+1)+(3+1).
  std::string input = M("a") + M("b") + " " + M("c") + M("d");

  // Process full input. The space after "▁b" terminates the first whitespace
  // word, which is expanded through metaspace into "▁a" and "▁b".
  // The second word "▁c▁d" is pending in whitespace (no trailing terminator).
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), iree_make_string_view(input.c_str(), input.size()),
      iree_tokenizer_make_segment_output(segments, 8), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 9u);
  ASSERT_EQ(segment_count, 2u);
  // Verify absolute: "▁a" at [0, 4), "▁b" at [4, 8).
  EXPECT_EQ(segments[0].start, 0u);
  EXPECT_EQ(segments[0].end, 4u);
  EXPECT_EQ(segments[1].start, 4u);
  EXPECT_EQ(segments[1].end, 8u);
  EXPECT_TRUE(iree_tokenizer_segmenter_state_has_pending(state.get()));

  // Finalize with remaining "▁c▁d" (8 bytes).
  iree_host_size_t position = consumed;
  std::string remaining = M("c") + M("d");
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), iree_make_string_view(remaining.c_str(), remaining.size()),
      iree_tokenizer_make_segment_output(segments, 8), &finalize_count));

  ASSERT_EQ(finalize_count, 2u);
  // Verify absolute reconstruction matches HF: ("▁c", (9, 13)), ("▁d", (13,
  // 17)).
  EXPECT_EQ(position + segments[0].start, 9u);
  EXPECT_EQ(position + segments[0].end, 13u);
  EXPECT_EQ(position + segments[1].start, 13u);
  EXPECT_EQ(position + segments[1].end, 17u);
}

//===----------------------------------------------------------------------===//
// Multi-Split Chain (GPT-2 Style Pretokenization)
//===----------------------------------------------------------------------===//

// Tests chaining multiple Split segmenters, similar to GPT-2/DeepSeek configs.
// The chain processes input through multiple regex-based splits in sequence.
// Each Split either matches (producing isolated segments) or passes through
// the input as gap segments for the next child to process.
TEST_F(SequenceSegmenterTest, SplitChainSpacePunctuation) {
  // Chain: Split(digits) -> Split(punctuation pattern)
  // First split matches digit sequences, second handles punctuation.
  ScopedSegmenter split_digits = CreateSplit("\\p{N}{1,3}");
  ASSERT_NE(split_digits.get(), nullptr);

  // Pattern matches: optional space + punctuation/symbols.
  ScopedSegmenter split_main =
      CreateSplit(" ?[\\p{P}\\p{S}]+|\\s+(?!\\S)|\\s+");
  ASSERT_NE(split_main.get(), nullptr);

  std::vector<ScopedSegmenter> children;
  children.push_back(std::move(split_digits));
  children.push_back(std::move(split_main));
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // " <" matches ` ?[\p{P}\p{S}]+` as a single segment (space + less-than).
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(" <"), {" <"},
                        /*expect_pending_after_process=*/false);
}

// Tests the full GPT-2-style regex pattern through a Split chain.
TEST_F(SequenceSegmenterTest, SplitChainFullGpt2Pattern) {
  ScopedSegmenter split_digits = CreateSplit("\\p{N}{1,3}");
  ASSERT_NE(split_digits.get(), nullptr);

  // Full GPT-2 style pattern with multiple alternatives.
  ScopedSegmenter split_main = CreateSplit(
      "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
      "|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
      "| ?[\\p{P}\\p{S}]+[\\r\\n]*"
      "|\\s*[\\r\\n]+"
      "|\\s+(?!\\S)"
      "|\\s+");
  ASSERT_NE(split_main.get(), nullptr);

  std::vector<ScopedSegmenter> children;
  children.push_back(std::move(split_digits));
  children.push_back(std::move(split_main));
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // " <" matches ` ?[\p{P}\p{S}]+` as a single segment.
  TestWithAllChunkSizes(segmenter.get(), IREE_SVL(" <"), {" <"},
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Punctuation + Split Interaction
//===----------------------------------------------------------------------===//

// Tests that Punctuation followed by Split in a Sequence correctly propagates
// segment boundaries through finalize. Punctuation(CONTIGUOUS) creates segment
// boundaries at punctuation characters, and the resulting segments must have
// valid bounds when passed to subsequent children in the chain.
TEST_F(SequenceSegmenterTest, PunctuationThenSplit) {
  ScopedSegmenter punctuation = CreatePunctuation();
  ASSERT_NE(punctuation.get(), nullptr);

  ScopedSegmenter split = CreateSplit("\\s+");
  ASSERT_NE(split.get(), nullptr);

  std::vector<ScopedSegmenter> children;
  children.push_back(std::move(punctuation));
  children.push_back(std::move(split));

  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  iree_tokenizer_segment_t segments[32];
  auto output = iree_tokenizer_make_segment_output(segments, 32);
  iree_host_size_t consumed = 0;
  iree_host_size_t count = 0;

  // Input with punctuation characters that create segment boundaries.
  iree_string_view_t input = IREE_SVL("Hello, world!");

  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, output, &consumed, &count));

  iree_string_view_t remaining =
      iree_make_string_view(input.data + consumed, input.size - consumed);
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining, output, &finalize_count));

  // Verify segments were produced.
  EXPECT_GT(count + finalize_count, 0u);
}

//===----------------------------------------------------------------------===//
// Offset Consistency with Leading Whitespace
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, LeadingWhitespaceDoesNotAffectLaterOffsets) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // Test with varying amounts of leading whitespace.
  // The key assertion: segment content and offsets should be consistent
  // regardless of leading whitespace amount.
  std::vector<int> leading_spaces = {0, 1, 2, 3, 5, 10};

  for (int num_spaces : leading_spaces) {
    std::string leading(num_spaces, ' ');
    // Input: [leading spaces]▁hello▁world
    std::string input = leading + M("hello") + M("world");

    auto segments = ProcessAndFinalize(
        segmenter.get(), iree_make_string_view(input.c_str(), input.size()),
        /*expect_pending_after_process=*/true);

    ASSERT_EQ(segments.size(), 2)
        << "With " << num_spaces << " leading spaces, expected 2 segments";
    EXPECT_EQ(segments[0], M("hello"))
        << "With " << num_spaces << " leading spaces, first segment wrong";
    EXPECT_EQ(segments[1], M("world"))
        << "With " << num_spaces << " leading spaces, second segment wrong";
  }
}

TEST_F(SequenceSegmenterTest, ManySegmentsWithLeadingWhitespace) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // ALBERT-style input pattern: leading spaces, multiple words with
  // inter-word whitespace. Tests offset consistency with many segments.
  //
  // Input: "  ▁hello▁world ▁foo▁bar ▁baz"
  std::string input = "  " + M("hello") + M("world") + " " + M("foo") +
                      M("bar") + " " + M("baz");

  auto segments = ProcessAndFinalize(
      segmenter.get(), iree_make_string_view(input.c_str(), input.size()),
      /*expect_pending_after_process=*/true);

  // Expected segments after WhitespaceSplit -> Metaspace:
  // "▁hello", "▁world", "▁foo", "▁bar", "▁baz"
  ASSERT_EQ(segments.size(), 5);
  EXPECT_EQ(segments[0], M("hello"));
  EXPECT_EQ(segments[1], M("world"));
  EXPECT_EQ(segments[2], M("foo"));
  EXPECT_EQ(segments[3], M("bar"));
  EXPECT_EQ(segments[4], M("baz"));
}

//===----------------------------------------------------------------------===//
// No ASCII Whitespace (ALBERT-style post-normalization)
// When normalizer replaces \s+ -> ▁, there is no ASCII whitespace left.
// WhitespaceSplit produces one segment (the entire text), and Metaspace
// must split it into individual words via finalize.
//===----------------------------------------------------------------------===//

TEST_F(SequenceSegmenterTest, NoAsciiWhitespace_MetaspaceSplitsAll) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  // This is the ALBERT-normalized form of:
  //   "  multiple   spaces   and\ttabs\nand\nnewlines  "
  // After strip trailing + regex \s+ -> ▁:
  //   "▁multiple▁spaces▁and▁tabs▁and▁newlines"
  // There is NO ASCII whitespace - only ▁ (U+2581).
  std::string input = M("multiple") + M("spaces") + M("and") + M("tabs") +
                      M("and") + M("newlines");

  // WhitespaceSplit sees no ASCII whitespace, produces 1 segment (the whole
  // input). Metaspace splits on ▁, producing 6 segments.
  auto segments = ProcessAndFinalize(
      segmenter.get(), iree_make_string_view(input.c_str(), input.size()),
      /*expect_pending_after_process=*/true);

  ASSERT_EQ(segments.size(), 6);
  EXPECT_EQ(segments[0], M("multiple"));
  EXPECT_EQ(segments[1], M("spaces"));
  EXPECT_EQ(segments[2], M("and"));
  EXPECT_EQ(segments[3], M("tabs"));
  EXPECT_EQ(segments[4], M("and"));
  EXPECT_EQ(segments[5], M("newlines"));
}

// Same test but with various chunk sizes to stress streaming.
TEST_F(SequenceSegmenterTest, NoAsciiWhitespace_MetaspaceSplitsAll_AllChunks) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  std::string input = M("multiple") + M("spaces") + M("and") + M("tabs") +
                      M("and") + M("newlines");

  TestWithAllChunkSizes(segmenter.get(),
                        iree_make_string_view(input.c_str(), input.size()),
                        {M("multiple"), M("spaces"), M("and"), M("tabs"),
                         M("and"), M("newlines")},
                        /*expect_pending_after_process=*/true);
}

// Test with limited output capacity to verify offset correctness.
TEST_F(SequenceSegmenterTest, NoAsciiWhitespace_LimitedCapacity) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  std::string input = M("multiple") + M("spaces") + M("and") + M("tabs") +
                      M("and") + M("newlines");

  TestLimitedOutputCapacity(segmenter.get(),
                            iree_make_string_view(input.c_str(), input.size()),
                            {M("multiple"), M("spaces"), M("and"), M("tabs"),
                             M("and"), M("newlines")});
}

TEST_F(SequenceSegmenterTest, VerifySegmentOffsetsWithLeadingWhitespace) {
  std::vector<ScopedSegmenter> children;
  children.push_back(CreateWhitespace());
  children.push_back(CreateMetaspace());
  auto segmenter = CreateSequence(std::move(children));
  ASSERT_NE(segmenter.get(), nullptr);

  ScopedSegmenterState state(segmenter.get());

  // Input: "  ▁A▁B ▁C" (2 leading spaces, two whitespace words)
  // After WhitespaceSplit: ["▁A▁B", "▁C"]
  // After Metaspace expansion: ["▁A", "▁B", "▁C"]
  std::string input = "  " + M("A") + M("B") + " " + M("C");
  iree_string_view_t input_sv =
      iree_make_string_view(input.c_str(), input.size());

  iree_tokenizer_segment_t segments[8];
  auto output = iree_tokenizer_make_segment_output(segments, 8);
  iree_host_size_t consumed = 0;
  iree_host_size_t count = 0;

  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input_sv, output, &consumed, &count));

  // Process should emit ▁A and ▁B (from first whitespace word "▁A▁B").
  // The second word "▁C" has no trailing whitespace so it's pending.
  ASSERT_EQ(count, 2) << "Expected 2 segments from process()";

  // Verify actual text at segment offsets.
  for (iree_host_size_t i = 0; i < count; ++i) {
    std::string actual(input_sv.data + segments[i].start,
                       segments[i].end - segments[i].start);
    if (i == 0) {
      EXPECT_EQ(actual, M("A"))
          << "Segment 0 offset mismatch: start=" << segments[0].start
          << ", end=" << segments[0].end;
    } else if (i == 1) {
      EXPECT_EQ(actual, M("B"))
          << "Segment 1 offset mismatch: start=" << segments[1].start
          << ", end=" << segments[1].end;
    }
  }

  // Finalize to get remaining segment.
  iree_string_view_t remaining =
      iree_make_string_view(input_sv.data + consumed, input_sv.size - consumed);
  iree_host_size_t finalize_count = 0;
  IREE_ASSERT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining, output, &finalize_count));

  ASSERT_EQ(finalize_count, 1) << "Expected 1 segment from finalize()";

  // Verify finalize segment content.
  std::string finalize_actual(remaining.data + segments[0].start,
                              segments[0].end - segments[0].start);
  EXPECT_EQ(finalize_actual, M("C"))
      << "Finalize segment offset mismatch: start=" << segments[0].start
      << ", end=" << segments[0].end;
}

}  // namespace
}  // namespace iree::tokenizer::testing
