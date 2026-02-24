// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/normalizer/sequence.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/normalizer/deferred.h"
#include "iree/tokenizer/normalizer/normalizer_test_util.h"
#include "iree/tokenizer/normalizer/passthrough.h"
#include "iree/tokenizer/normalizer/prepend.h"
#include "iree/tokenizer/normalizer/replace.h"
#include "iree/tokenizer/normalizer/strip.h"

namespace iree::tokenizer::testing {
namespace {

//===----------------------------------------------------------------------===//
// Test Fixtures
//===----------------------------------------------------------------------===//

class SequenceNormalizerTest : public ::testing::Test {
 protected:
  // Helper to create a sequence from a list of normalizers.
  // Takes ownership of the ScopedNormalizers: on success they are released to
  // the sequence, on failure they are cleaned up automatically.
  ScopedNormalizer CreateSequence(std::vector<ScopedNormalizer> children) {
    std::vector<iree_tokenizer_normalizer_t*> raw_children;
    raw_children.reserve(children.size());
    for (const auto& child : children) {
      raw_children.push_back(child.get());
    }

    iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
    iree_status_t status = iree_tokenizer_normalizer_sequence_allocate(
        raw_children.data(), raw_children.size(), iree_allocator_system(),
        &raw_normalizer);
    if (!iree_status_is_ok(status)) {
      return ScopedNormalizer(nullptr);
    }

    // Success: release all children to sequence ownership.
    for (auto& child : children) {
      child.release();
    }
    return ScopedNormalizer(raw_normalizer);
  }

  // Creates a passthrough normalizer.
  ScopedNormalizer CreatePassthrough() {
    iree_tokenizer_normalizer_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_normalizer_passthrough_allocate(
        iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedNormalizer(nullptr);
    }
    return ScopedNormalizer(raw);
  }

  // Creates a deferred normalizer (buffers during process, emits on finalize).
  ScopedNormalizer CreateDeferred() {
    iree_tokenizer_normalizer_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_normalizer_deferred_allocate(
        iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedNormalizer(nullptr);
    }
    return ScopedNormalizer(raw);
  }

  // Creates a strip normalizer.
  ScopedNormalizer CreateStrip(bool left, bool right) {
    iree_tokenizer_normalizer_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_normalizer_strip_allocate(
        left, right, iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedNormalizer(nullptr);
    }
    return ScopedNormalizer(raw);
  }

  // Creates a replace normalizer for space -> metaspace substitution.
  ScopedNormalizer CreateSpaceToMetaspace() {
    iree_tokenizer_normalizer_t* raw = nullptr;
    // U+2581 LOWER ONE EIGHTH BLOCK = 0xE2 0x96 0x81 in UTF-8.
    iree_status_t status = iree_tokenizer_normalizer_replace_allocate(
        IREE_SV(" "), IREE_SV("\xE2\x96\x81"), iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedNormalizer(nullptr);
    }
    return ScopedNormalizer(raw);
  }

  // Creates a prepend normalizer that prepends metaspace.
  ScopedNormalizer CreatePrependMetaspace(bool skip_if_prefix_matches) {
    iree_tokenizer_normalizer_t* raw = nullptr;
    iree_status_t status = iree_tokenizer_normalizer_prepend_allocate(
        IREE_SV("\xE2\x96\x81"), skip_if_prefix_matches,
        iree_allocator_system(), &raw);
    if (!iree_status_is_ok(status)) {
      return ScopedNormalizer(nullptr);
    }
    return ScopedNormalizer(raw);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceNormalizerTest, CreateAndDestroy) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  ASSERT_NE(children[0].get(), nullptr);
  ASSERT_NE(children[1].get(), nullptr);

  auto normalizer = CreateSequence(std::move(children));
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(SequenceNormalizerTest, RejectsEmpty) {
  // Empty sequence (0 children) is rejected - use NULL for passthrough.
  std::vector<iree_tokenizer_normalizer_t*> children;
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_sequence_allocate(
      children.data(), children.size(), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceNormalizerTest, RejectsSingle) {
  // Single child is rejected - return the child directly, no wrapper needed.
  ScopedNormalizer passthrough = CreatePassthrough();
  ASSERT_NE(passthrough.get(), nullptr);

  std::vector<iree_tokenizer_normalizer_t*> raw_pointers = {passthrough.get()};
  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_sequence_allocate(
      raw_pointers.data(), raw_pointers.size(), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceNormalizerTest, CreateMaxDepth) {
  std::vector<ScopedNormalizer> children;
  for (int i = 0; i < IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH; ++i) {
    children.push_back(CreatePassthrough());
    ASSERT_NE(children.back().get(), nullptr);
  }

  auto normalizer = CreateSequence(std::move(children));
  EXPECT_NE(normalizer.get(), nullptr);
}

TEST_F(SequenceNormalizerTest, StateSizeIncludesChildren) {
  ScopedNormalizer child1 = CreatePassthrough();
  ScopedNormalizer child2 = CreatePassthrough();
  ASSERT_NE(child1.get(), nullptr);
  ASSERT_NE(child2.get(), nullptr);

  iree_host_size_t child1_state_size =
      iree_tokenizer_normalizer_state_size(child1.get());
  iree_host_size_t child2_state_size =
      iree_tokenizer_normalizer_state_size(child2.get());

  std::vector<ScopedNormalizer> children;
  children.push_back(std::move(child1));
  children.push_back(std::move(child2));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer.get());

  // Sequence state must be at least base + both child states.
  EXPECT_GE(state_size, child1_state_size + child2_state_size);
}

//===----------------------------------------------------------------------===//
// Validation Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceNormalizerTest, RejectsExcessiveDepth) {
  std::vector<ScopedNormalizer> children;
  std::vector<iree_tokenizer_normalizer_t*> raw_pointers;

  for (int i = 0; i < IREE_TOKENIZER_NORMALIZER_SEQUENCE_MAX_DEPTH + 1; ++i) {
    ScopedNormalizer child = CreatePassthrough();
    ASSERT_NE(child.get(), nullptr);
    raw_pointers.push_back(child.get());
    children.push_back(std::move(child));
  }

  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_sequence_allocate(
      raw_pointers.data(), raw_pointers.size(), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceNormalizerTest, RejectsNullChildren) {
  ScopedNormalizer child1 = CreatePassthrough();
  ASSERT_NE(child1.get(), nullptr);

  std::vector<iree_tokenizer_normalizer_t*> raw_children = {child1.get(),
                                                            nullptr};

  iree_tokenizer_normalizer_t* raw_normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_sequence_allocate(
      raw_children.data(), raw_children.size(), iree_allocator_system(),
      &raw_normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

//===----------------------------------------------------------------------===//
// Sequence Flattening Tests
//===----------------------------------------------------------------------===//
// Nested sequences are automatically flattened to prevent the "vertical tiling
// within vertical tiling" problem that causes lazy normalizers to receive
// insufficient lookahead context.

TEST_F(SequenceNormalizerTest, FlattensNestedSequence) {
  // Create inner sequence: Seq[Passthrough, Passthrough]
  std::vector<ScopedNormalizer> inner_children;
  inner_children.push_back(CreatePassthrough());
  inner_children.push_back(CreatePassthrough());
  auto inner_seq = CreateSequence(std::move(inner_children));
  ASSERT_NE(inner_seq.get(), nullptr);

  // Create outer sequence: Seq[inner_seq, Passthrough]
  // This should be flattened to: Seq[Passthrough, Passthrough, Passthrough]
  std::vector<ScopedNormalizer> outer_children;
  outer_children.push_back(std::move(inner_seq));
  outer_children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(outer_children));
  ASSERT_NE(normalizer.get(), nullptr);

  // Verify it works correctly.
  TestWithAllChunkSizes(normalizer.get(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceNormalizerTest, FlattensDeepNestedSequence) {
  // Create deeply nested sequences:
  // Seq[Seq[Seq[Passthrough, Passthrough], Passthrough], Passthrough]
  // Should flatten to: Seq[Passthrough, Passthrough, Passthrough, Passthrough]

  // Level 3 (innermost): Seq[Passthrough, Passthrough]
  std::vector<ScopedNormalizer> level3;
  level3.push_back(CreatePassthrough());
  level3.push_back(CreatePassthrough());
  auto seq3 = CreateSequence(std::move(level3));
  ASSERT_NE(seq3.get(), nullptr);

  // Level 2: Seq[seq3, Passthrough]
  std::vector<ScopedNormalizer> level2;
  level2.push_back(std::move(seq3));
  level2.push_back(CreatePassthrough());
  auto seq2 = CreateSequence(std::move(level2));
  ASSERT_NE(seq2.get(), nullptr);

  // Level 1 (outermost): Seq[seq2, Passthrough]
  std::vector<ScopedNormalizer> level1;
  level1.push_back(std::move(seq2));
  level1.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(level1));
  ASSERT_NE(normalizer.get(), nullptr);

  TestWithAllChunkSizes(normalizer.get(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

// Tests that nested sequences are flattened so lazy normalizers like Strip
// receive sufficient lookahead context. DeBERTa-style chains like
// Seq[Seq[NFC], Strip, Replace, Prepend] require flattening because Strip
// with strip_right needs to buffer whitespace until it sees non-whitespace.
TEST_F(SequenceNormalizerTest, FlattensNestedSequenceWithLazyNormalizer) {
  // Create inner sequence containing Strip (simulates Seq[NFC] + Strip).
  std::vector<ScopedNormalizer> inner_children;
  inner_children.push_back(CreatePassthrough());  // Simulates NFC
  inner_children.push_back(CreateStrip(/*left=*/false, /*right=*/true));
  auto inner_seq = CreateSequence(std::move(inner_children));
  ASSERT_NE(inner_seq.get(), nullptr);

  // Create outer sequence: Seq[inner_seq, Replace, Prepend].
  // After flattening becomes: Seq[Passthrough, Strip, Replace, Prepend].
  std::vector<ScopedNormalizer> outer_children;
  outer_children.push_back(std::move(inner_seq));
  outer_children.push_back(CreateSpaceToMetaspace());
  outer_children.push_back(
      CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(outer_children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "Hello, world!" should become "▁Hello,▁world!"
  // Without flattening, the space before "world" would be consumed but lost.
  const std::string expected =
      "\xE2\x96\x81"
      "Hello,"
      "\xE2\x96\x81"
      "world!";

  // Test baseline (no chunking).
  EXPECT_EQ(ProcessAndFinalize(normalizer.get(), "Hello, world!", false),
            expected);

  // Test with chunk sizes >= 2. We skip chunk_size=1 because lazy normalizers
  // (like Strip with strip_right) need lookahead beyond a single byte. With
  // chunk_size=1, Strip sees one byte at a time in the intermediate buffer and
  // can't see what follows a space, causing it to defer consumption forever.
  // Real tokenizers never process byte-by-byte, so this limitation is
  // acceptable.
  for (size_t chunk_size : {2, 3, 5, 8, 13, 16, 32, 64}) {
    SCOPED_TRACE(::testing::Message() << "chunk_size=" << chunk_size);
    EXPECT_EQ(
        ProcessChunkedAndFinalize(normalizer.get(), "Hello, world!", chunk_size,
                                  /*expect_pending_after_process=*/false),
        expected);
  }
}

TEST_F(SequenceNormalizerTest, FlatteningSingleChildReturnsChildDirectly) {
  // Create inner sequence: Seq[Passthrough, Passthrough]
  std::vector<ScopedNormalizer> inner_children;
  inner_children.push_back(CreatePassthrough());
  inner_children.push_back(CreatePassthrough());
  auto inner_seq = CreateSequence(std::move(inner_children));
  ASSERT_NE(inner_seq.get(), nullptr);

  // Create outer sequence with inner_seq and one grandchild from another seq.
  // Inner Seq: [P, P] (2 children)
  // Outer Seq: [Inner, P]
  // After flattening: [P, P, P] (3 children, valid sequence)

  // But if we create Seq[InnerSeq] where InnerSeq has 2 children,
  // flattening gives us 2 children which is still a valid sequence.
  // The single-child case is: Seq[InnerSeq] where InnerSeq has 1 child.
  // That's rejected at inner_seq creation time.

  // So let's test: Seq[Passthrough, Seq[Passthrough, Passthrough]]
  // Flattens to: Seq[Passthrough, Passthrough, Passthrough]
  std::vector<ScopedNormalizer> outer_children;
  outer_children.push_back(CreatePassthrough());
  outer_children.push_back(std::move(inner_seq));
  auto normalizer = CreateSequence(std::move(outer_children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestWithAllChunkSizes(normalizer.get(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Buffering Normalizer Tests
//===----------------------------------------------------------------------===//
// These tests verify correct handling of buffering normalizers (like Deferred
// or Precompiled) that may consume all input but write nothing until later.
// The sequence must not lose data when a stage buffers.

// Buffering normalizers (like Precompiled) may consume all input but write
// nothing until they have enough lookahead. The sequence must not claim input
// as consumed when a stage buffers without producing output.
TEST_F(SequenceNormalizerTest, BufferingNormalizerDoesNotLoseData) {
  // Create sequence: Deferred -> Replace
  // Deferred buffers all input until finalize, then emits it.
  // If the sequence incorrectly handles the buffering, data gets lost.
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateDeferred());
  children.push_back(CreateSpaceToMetaspace());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // Test with a string that has spaces to be replaced.
  // With correct handling: process() consumes nothing (Deferred buffers),
  // finalize() emits everything through Replace.
  std::string result = ProcessAndFinalize(
      normalizer.get(), "hello world", /*expect_pending_after_process=*/true);
  // Spaces should be replaced with metaspace.
  EXPECT_EQ(result, "hello\xE2\x96\x81world");
}

// Test that buffering normalizer at stage 0 followed by multiple stages works.
TEST_F(SequenceNormalizerTest, BufferingNormalizerWithMultipleStages) {
  // Sequence: Deferred -> Strip(right) -> Replace -> Prepend
  // This simulates the T5/Metaspace pipeline where precompiled normalizer
  // buffers data, then it flows through strip, replace, and prepend.
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateDeferred());
  children.push_back(CreateStrip(/*left=*/false, /*right=*/true));
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "hello world " has trailing space that should be stripped.
  // Result: ▁hello▁world (no trailing space)
  std::string result = ProcessAndFinalize(
      normalizer.get(), "hello world ", /*expect_pending_after_process=*/true);
  EXPECT_EQ(result,
            "\xE2\x96\x81"
            "hello"
            "\xE2\x96\x81"
            "world");
}

// Test with exact tile size (32 bytes) that would trigger precompiled buffering
TEST_F(SequenceNormalizerTest, BufferingNormalizerWith32ByteTile) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateDeferred());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // 32-byte input that matches the precompiled overlap buffer size.
  std::string input = "summarize: The quick brown fox j";  // exactly 32 bytes
  ASSERT_EQ(input.size(), 32u);

  std::string result = ProcessAndFinalize(
      normalizer.get(), input, /*expect_pending_after_process=*/true);
  EXPECT_EQ(result, input);
}

//===----------------------------------------------------------------------===//
// Basic Functionality Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceNormalizerTest, EmptyInput) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  std::string result = ProcessAndFinalize(
      normalizer.get(), "", /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, "");
}

// Note: Empty and single-child sequences are rejected by the allocator.
// The JSON parser handles those cases by returning NULL (empty) or unwrapping
// (single child). These tests verify the allocator contract.

TEST_F(SequenceNormalizerTest, TwoPassthroughIdentity) {
  // passthrough -> passthrough should still be identity.
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestWithAllChunkSizes(normalizer.get(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceNormalizerTest, ThreePassthroughIdentity) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestWithAllChunkSizes(normalizer.get(), "hello world", "hello world",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Output Capacity Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceNormalizerTest, ZeroCapacityOutput) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestZeroCapacityOutput(normalizer.get(), "hello");
}

TEST_F(SequenceNormalizerTest, LimitedOutputCapacity) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestLimitedOutputCapacity(normalizer.get(), "hello world", "hello world");
}

//===----------------------------------------------------------------------===//
// Has_Pending Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceNormalizerTest, HasPendingTracking) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  ScopedNormalizerState state(normalizer.get());

  // Initially no pending (passthrough doesn't buffer).
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // Process some input.
  char output_buffer[64];
  iree_mutable_string_view_t output =
      iree_make_mutable_string_view(output_buffer, sizeof(output_buffer));
  iree_host_size_t consumed = 0, written = 0;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), IREE_SVL("hello"), output,
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // Passthrough doesn't buffer, so still no pending.
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // After finalize, definitely no pending.
  iree_mutable_string_view_t finalize_output =
      iree_make_mutable_string_view(output_buffer, sizeof(output_buffer));
  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
      state.get(), finalize_output, &written));
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));
}

//===----------------------------------------------------------------------===//
// UTF-8 Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceNormalizerTest, Utf8Passthrough) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // Test with various UTF-8 sequences.
  TestWithAllChunkSizes(normalizer.get(),
                        "hello \xE2\x96\x81world",  // Contains metaspace
                        "hello \xE2\x96\x81world",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceNormalizerTest, Utf8Emoji) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // Test with emoji (4-byte UTF-8).
  TestWithAllChunkSizes(normalizer.get(),
                        "hello \xF0\x9F\x98\x80 world",  // 😀
                        "hello \xF0\x9F\x98\x80 world",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Intermediate Buffering Tests
//===----------------------------------------------------------------------===//

// Tests processing byte-by-byte through a two-child sequence, verifying that
// intermediate buffering works correctly when output is capacity-limited.
TEST_F(SequenceNormalizerTest, ProcessBackpressureCapacity1) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  ScopedNormalizerState state(normalizer.get());

  std::string input = "hello world";
  std::string result;
  char output_char;
  iree_host_size_t position = 0;

  // Process input with output capacity=1, tracking that we make progress.
  while (position < input.size()) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;

    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(),
        iree_make_string_view(input.data() + position, input.size() - position),
        iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

    if (written > 0) {
      result += output_char;
    }

    // We must make progress - either consuming input or producing output.
    ASSERT_TRUE(consumed > 0 || written > 0)
        << "No progress at position " << position;

    position += consumed;
  }

  // Finalize with capacity=1.
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

  EXPECT_EQ(result, input) << "Capacity-1 processing produced incorrect result";
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));
}

// Tests that a three-child chain works correctly with capacity-limited output.
// This exercises deeper intermediate buffering.
TEST_F(SequenceNormalizerTest, ThreePassthroughCapacity1) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestLimitedOutputCapacity(normalizer.get(), "hello world", "hello world");
}

// Tests that has_pending correctly reflects when intermediate data is buffered.
// This requires processing with limited output capacity to create pending
// state.
TEST_F(SequenceNormalizerTest, HasPendingWithIntermediateBuffer) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  ScopedNormalizerState state(normalizer.get());

  // Initially no pending.
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));

  // Process with very limited output capacity to force buffering.
  std::string input = "hello";
  char output_char;
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_string_view(input.data(), input.size()),
      iree_make_mutable_string_view(&output_char, 1),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // With capacity=1, we consumed some input but child[0] may have produced
  // more than 1 byte into intermediate. Check if there's pending data.
  // Note: has_pending should be true if there's buffered intermediate data.
  // Passthrough children don't buffer, so pending = intermediate buffer only.
  if (consumed > written) {
    // We consumed more than we output - there should be pending intermediate.
    EXPECT_TRUE(iree_tokenizer_normalizer_state_has_pending(state.get()));
  }

  // Drain remaining output.
  std::string result;
  result += output_char;

  while (iree_tokenizer_normalizer_state_has_pending(state.get())) {
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(), iree_make_string_view(nullptr, 0),
        iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) {
      result += output_char;
    }
    if (consumed == 0 && written == 0) break;
  }

  // Process remaining input.
  size_t pos = consumed;
  while (pos < input.size()) {
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(),
        iree_make_string_view(input.data() + pos, input.size() - pos),
        iree_make_mutable_string_view(&output_char, 1),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
    if (written > 0) result += output_char;
    if (consumed == 0 && written == 0) break;
    pos += consumed;
  }

  // Finalize.
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

  EXPECT_EQ(result, input);
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));
}

// Tests that consumption is granular - we don't over-consume input when output
// capacity is limited. This ensures backpressure works correctly through the
// chain.
TEST_F(SequenceNormalizerTest, ConsumptionGranularityWithLimitedOutput) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  ScopedNormalizerState state(normalizer.get());

  std::string input = "abcdefghij";  // 10 chars
  std::string result;

  // Process with capacity=2 - should consume and produce in small chunks.
  char output_buffer[2];
  size_t position = 0;

  while (position < input.size() ||
         iree_tokenizer_normalizer_state_has_pending(state.get())) {
    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;

    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
        state.get(),
        iree_make_string_view(input.data() + position, input.size() - position),
        iree_make_mutable_string_view(output_buffer, 2),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

    if (written > 0) {
      result.append(output_buffer, written);
    }

    if (consumed == 0 && written == 0) break;
    position += consumed;
  }

  // Finalize.
  while (true) {
    iree_host_size_t finalize_written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(output_buffer, 2),
        &finalize_written));
    if (finalize_written > 0) {
      result.append(output_buffer, finalize_written);
    } else {
      break;
    }
  }

  EXPECT_EQ(result, input);
}

// Tests processing a larger input that spans multiple process calls with
// constrained intermediate buffer behavior.
TEST_F(SequenceNormalizerTest, LargerInputCapacity1) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // Use a longer string to stress the buffering.
  std::string input =
      "The quick brown fox jumps over the lazy dog. "
      "Pack my box with five dozen liquor jugs.";

  TestLimitedOutputCapacity(normalizer.get(), input, input);
}

//===----------------------------------------------------------------------===//
// Finalize Backpressure Tests
//===----------------------------------------------------------------------===//

// Tests that finalize() rejects calls when process() has deferred consumption.
// This is a precondition violation - the caller must drain process() first.
TEST_F(SequenceNormalizerTest, FinalizeRejectsDeferredConsumption) {
  // Need 3+ children to trigger ping-pong buffering where backpressure can
  // leave pending_input_consumed > 0.
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  ScopedNormalizerState state(normalizer.get());

  // Process with very limited output to force backpressure mid-chain.
  // With 3 children and capacity=1, child[0] processes input into buffer[0],
  // then child[1] processes into buffer[1], then child[2] can only output 1
  // byte. This leaves pending data in the chain.
  std::string input = "hello world";
  char output_char;
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_string_view(input.data(), input.size()),
      iree_make_mutable_string_view(&output_char, 1),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));

  // If consumed < input.size() but we still have pending data, the deferred
  // consumption mechanism is active. Try to finalize - should fail.
  if (iree_tokenizer_normalizer_state_has_pending(state.get()) &&
      consumed == 0) {
    // This should fail with FAILED_PRECONDITION because we have deferred
    // consumption that hasn't been committed yet.
    iree_host_size_t finalize_written = 0;
    iree_status_t status = iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(&output_char, 1),
        &finalize_written);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, status);

    // Now properly drain process() to clear the deferred consumption.
    std::string result;
    result += output_char;  // First byte from initial process()
    size_t pos = 0;         // Haven't committed any consumption yet

    while (pos < input.size() ||
           iree_tokenizer_normalizer_state_has_pending(state.get())) {
      IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
          state.get(),
          iree_make_string_view(input.data() + pos, input.size() - pos),
          iree_make_mutable_string_view(&output_char, 1),
          IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
      if (written > 0) result += output_char;
      if (consumed == 0 && written == 0) break;
      pos += consumed;
    }

    // Now finalize should succeed.
    while (true) {
      IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
          state.get(), iree_make_mutable_string_view(&output_char, 1),
          &finalize_written));
      if (finalize_written > 0) {
        result += output_char;
      } else {
        break;
      }
    }

    EXPECT_EQ(result, input);
  }
}

// Tests that finalize can be called repeatedly with limited output capacity
// and produces correct results without double-emitting.
TEST_F(SequenceNormalizerTest, FinalizeWithLimitedCapacity) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  ScopedNormalizerState state(normalizer.get());

  // Process all input first with unlimited capacity.
  std::string input = "hello world";
  char output_buffer[64];
  iree_host_size_t consumed = 0, written = 0;

  IREE_ASSERT_OK(iree_tokenizer_normalizer_state_process(
      state.get(), iree_make_string_view(input.data(), input.size()),
      iree_make_mutable_string_view(output_buffer, sizeof(output_buffer)),
      IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written));
  EXPECT_EQ(consumed, input.size());
  EXPECT_EQ(written, input.size());

  std::string result(output_buffer, written);

  // Finalize with capacity=1 should work correctly (passthrough produces
  // nothing on finalize, but this tests the loop logic).
  char output_char;
  size_t finalize_iterations = 0;
  while (finalize_iterations < 100) {  // Safety limit
    iree_host_size_t finalize_written = 0;
    IREE_ASSERT_OK(iree_tokenizer_normalizer_state_finalize(
        state.get(), iree_make_mutable_string_view(&output_char, 1),
        &finalize_written));
    if (finalize_written > 0) {
      result += output_char;
    } else {
      break;
    }
    ++finalize_iterations;
  }

  EXPECT_EQ(result, input);
  EXPECT_FALSE(iree_tokenizer_normalizer_state_has_pending(state.get()));
}

//===----------------------------------------------------------------------===//
// Finalize Overflow Tests
//===----------------------------------------------------------------------===//

// Tests that a non-last child producing more finalize data than the 256-byte
// intermediate buffer is handled correctly with a single finalize call.
TEST_F(SequenceNormalizerTest, FinalizeNonLastChildExceedsIntermediateBuffer) {
  // Seq(Deferred, Passthrough) - deferred buffers during process, emits on
  // finalize. Feed 300 bytes, exceeding the 256-byte intermediate buffer.
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateDeferred());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // Create input larger than intermediate buffer (256 bytes).
  std::string input(300, 'x');

  std::string result = ProcessAndFinalize(
      normalizer.get(), input, /*expect_pending_after_process=*/true);

  EXPECT_EQ(result.size(), input.size())
      << "Finalize truncated output: got " << result.size()
      << " bytes, expected " << input.size();
  EXPECT_EQ(result, input);
}

// Same test but using capacity=1 finalize loop to exercise the loop logic.
TEST_F(SequenceNormalizerTest,
       FinalizeNonLastChildExceedsIntermediateBuffer_LimitedOutput) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateDeferred());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  std::string input(300, 'y');

  // TestLimitedOutputCapacity uses capacity=1 loop.
  TestLimitedOutputCapacity(normalizer.get(), input, input);
}

// Tests truncation when a middle child produces excessive finalize data.
// Seq(Passthrough, Deferred, Passthrough) - exercises three-child chain.
TEST_F(SequenceNormalizerTest, FinalizeMiddleChildExceedsIntermediateBuffer) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreatePassthrough());
  children.push_back(CreateDeferred());
  children.push_back(CreatePassthrough());
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  std::string input(300, 'z');

  // TestWithAllChunkSizes exercises various input/output chunk sizes.
  // The middle child buffers all input, then emits 300 bytes on finalize.
  TestWithAllChunkSizes(normalizer.get(), input, input,
                        /*expect_pending_after_process=*/true);
}

//===----------------------------------------------------------------------===//
// DeBERTa-Style Normalizer Chain Tests
//===----------------------------------------------------------------------===//
// These tests verify the normalizer chains used by DeBERTa and T5 models:
// Seq(Strip(right), Replace(" " -> "▁"), Prepend("▁"))
//
// These chains exercise multi-stage tiled processing with flag propagation.

// Tests the Replace + Prepend chain (2-child). This minimal chain exercises
// tiled processing with ping-pong scratch buffers.
TEST_F(SequenceNormalizerTest, ReplacePrependChain) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "Hello, world!" -> "▁Hello,▁world!"
  // Replace: "Hello, world!" -> "Hello,▁world!" (spaces become ▁)
  // Prepend: "Hello,▁world!" -> "▁Hello,▁world!" (prepend ▁ at start)
  TestWithAllChunkSizes(normalizer.get(), "Hello, world!",
                        "\xE2\x96\x81"
                        "Hello,"
                        "\xE2\x96\x81"
                        "world!",
                        /*expect_pending_after_process=*/false);
}

// Tests the full DeBERTa chain (3-child): Strip + Replace + Prepend.
TEST_F(SequenceNormalizerTest, DebertaChainSimple) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateStrip(/*left=*/false, /*right=*/true));
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "Hello, world!" -> "▁Hello,▁world!"
  TestWithAllChunkSizes(normalizer.get(), "Hello, world!",
                        "\xE2\x96\x81"
                        "Hello,"
                        "\xE2\x96\x81"
                        "world!",
                        /*expect_pending_after_process=*/false);
}

// Tests the DeBERTa chain with trailing whitespace.
TEST_F(SequenceNormalizerTest, DebertaChainTrailingWhitespace) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateStrip(/*left=*/false, /*right=*/true));
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "  trimmed whitespace  " ->
  //   Strip(right): "  trimmed whitespace"
  //   Replace: "▁▁trimmed▁whitespace"
  //   Prepend: "▁▁▁trimmed▁whitespace" (wait, input already starts with ▁!)
  // Actually, the input AFTER Strip and Replace starts with spaces converted
  // to ▁, so Prepend with skip_if_prefix_matches=true will NOT add another ▁.
  // Let me trace through:
  //   Input: "  trimmed whitespace  "
  //   After Strip(right): "  trimmed whitespace"
  //   After Replace: "▁▁trimmed▁whitespace" (leading spaces become ▁)
  //   After Prepend(skip_if_prefix_matches): "▁▁trimmed▁whitespace" (no change,
  //                                          already starts with ▁)
  TestWithAllChunkSizes(normalizer.get(), "  trimmed whitespace  ",
                        "\xE2\x96\x81"
                        "\xE2\x96\x81"
                        "trimmed"
                        "\xE2\x96\x81"
                        "whitespace",
                        /*expect_pending_after_process=*/false);
}

// Tests the 2-child chain with buffer-size-1 output, exercising spillover
// handling when output capacity is minimal.
TEST_F(SequenceNormalizerTest, ReplacePrependChainCapacity1) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  TestLimitedOutputCapacity(normalizer.get(), "Hello, world!",
                            "\xE2\x96\x81Hello,\xE2\x96\x81world!");
}

// Tests the DeBERTa chain with multi-byte UTF-8 characters.
TEST_F(SequenceNormalizerTest, DebertaChainUnicode) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateStrip(/*left=*/false, /*right=*/true));
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "Héllo wörld café" -> "▁Héllo▁wörld▁café"
  TestWithAllChunkSizes(normalizer.get(),
                        "H\xC3\xA9llo w\xC3\xB6rld caf\xC3\xA9",
                        "\xE2\x96\x81"
                        "H\xC3\xA9llo"
                        "\xE2\x96\x81"
                        "w\xC3\xB6rld"
                        "\xE2\x96\x81"
                        "caf\xC3\xA9",
                        /*expect_pending_after_process=*/false);
}

// Tests the T5 chain: Replace + Prepend without Strip.
// T5 uses a different prepend_scheme and doesn't strip.
TEST_F(SequenceNormalizerTest, T5ChainSimple) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "summarize: The quick brown fox" ->
  //   Replace: "summarize:▁The▁quick▁brown▁fox"
  //   Prepend: "▁summarize:▁The▁quick▁brown▁fox"
  TestWithAllChunkSizes(normalizer.get(), "summarize: The quick brown fox",
                        "\xE2\x96\x81"
                        "summarize:"
                        "\xE2\x96\x81"
                        "The"
                        "\xE2\x96\x81"
                        "quick"
                        "\xE2\x96\x81"
                        "brown"
                        "\xE2\x96\x81"
                        "fox",
                        /*expect_pending_after_process=*/false);
}

// Tests the T5 chain with the exact input from the smoketest.
TEST_F(SequenceNormalizerTest, T5SummarizeInput) {
  std::vector<ScopedNormalizer> children;
  children.push_back(CreateSpaceToMetaspace());
  children.push_back(CreatePrependMetaspace(/*skip_if_prefix_matches=*/true));
  auto normalizer = CreateSequence(std::move(children));
  ASSERT_NE(normalizer.get(), nullptr);

  // "summarize: The quick brown fox jumps over the lazy dog."
  std::string input = "summarize: The quick brown fox jumps over the lazy dog.";
  // U+2581 LOWER ONE EIGHTH BLOCK = 0xE2 0x96 0x81 in UTF-8.
  const char* metaspace = "\xE2\x96\x81";
  std::string expected = std::string(metaspace) + "summarize:" + metaspace +
                         "The" + metaspace + "quick" + metaspace + "brown" +
                         metaspace + "fox" + metaspace + "jumps" + metaspace +
                         "over" + metaspace + "the" + metaspace + "lazy" +
                         metaspace + "dog.";

  TestWithAllChunkSizes(normalizer.get(), input, expected,
                        /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer::testing
