// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/decoder/sequence.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/decoder/byte_fallback.h"
#include "iree/tokenizer/decoder/decoder_test_util.h"
#include "iree/tokenizer/decoder/metaspace.h"
#include "iree/tokenizer/decoder/passthrough.h"
#include "iree/tokenizer/decoder/replace.h"
#include "iree/tokenizer/decoder/strip.h"

namespace iree::tokenizer {
namespace {

using testing::ProcessAndFinalize;
using testing::ScopedDecoder;
using testing::ScopedDecoderState;
using testing::TestWithAllBatchSizes;
using testing::ToStringViews;

//===----------------------------------------------------------------------===//
// Test fixtures
//===----------------------------------------------------------------------===//

class SequenceDecoderTest : public ::testing::Test {
 protected:
  // Helper to create a sequence from a list of decoders.
  // Takes ownership of the ScopedDecoders: on success they are released to the
  // sequence, on failure they are cleaned up automatically.
  ScopedDecoder CreateSequence(std::vector<ScopedDecoder> children) {
    // Extract raw pointers for the C API.
    std::vector<iree_tokenizer_decoder_t*> raw_children;
    raw_children.reserve(children.size());
    for (const auto& child : children) {
      raw_children.push_back(child.get());
    }

    iree_tokenizer_decoder_t* raw_decoder = nullptr;
    iree_status_t status = iree_tokenizer_decoder_sequence_allocate(
        raw_children.data(), raw_children.size(), iree_allocator_system(),
        &raw_decoder);
    if (!iree_status_is_ok(status)) {
      // Children are cleaned up when vector goes out of scope.
      return ScopedDecoder(nullptr);
    }

    // Success: release all children to sequence ownership.
    for (auto& child : children) {
      child.release();
    }
    return ScopedDecoder(raw_decoder);
  }
};

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

TEST_F(SequenceDecoderTest, RejectsEmptySequence) {
  // Empty sequence (0 children) is rejected - use NULL for passthrough.
  std::vector<iree_tokenizer_decoder_t*> children;
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  iree_status_t status = iree_tokenizer_decoder_sequence_allocate(
      children.data(), children.size(), iree_allocator_system(), &raw_decoder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceDecoderTest, RejectsSingleChild) {
  // Single child is rejected - return the child directly, no wrapper needed.
  ScopedDecoder passthrough;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), passthrough.put()));

  std::vector<iree_tokenizer_decoder_t*> raw_pointers = {passthrough.get()};
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  iree_status_t status = iree_tokenizer_decoder_sequence_allocate(
      raw_pointers.data(), raw_pointers.size(), iree_allocator_system(),
      &raw_decoder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST_F(SequenceDecoderTest, CreateWithMultipleChildren) {
  ScopedDecoder byte_fallback;
  IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), byte_fallback.put()));

  ScopedDecoder metaspace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
      0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS,
      iree_allocator_system(), metaspace.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(byte_fallback));
  children.push_back(std::move(metaspace));
  auto decoder = CreateSequence(std::move(children));
  EXPECT_NE(decoder.get(), nullptr);
}

TEST_F(SequenceDecoderTest, StateSizeIncludesChildren) {
  ScopedDecoder passthrough1;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), passthrough1.put()));
  ScopedDecoder passthrough2;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), passthrough2.put()));

  // Get child state sizes before transferring ownership.
  iree_host_size_t child1_state_size =
      iree_tokenizer_decoder_state_size(passthrough1.get());
  iree_host_size_t child2_state_size =
      iree_tokenizer_decoder_state_size(passthrough2.get());

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(passthrough1));
  children.push_back(std::move(passthrough2));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  iree_host_size_t state_size =
      iree_tokenizer_decoder_state_size(decoder.get());

  // Sequence state must be at least base + both child states.
  EXPECT_GE(state_size, child1_state_size + child2_state_size);
}

TEST_F(SequenceDecoderTest, RejectsExcessiveDepth) {
  // Create more than MAX_DEPTH children.
  std::vector<ScopedDecoder> children;
  std::vector<iree_tokenizer_decoder_t*> raw_pointers;

  for (int i = 0; i < IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH + 1; ++i) {
    ScopedDecoder passthrough;
    IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
        iree_allocator_system(), passthrough.put()));
    raw_pointers.push_back(passthrough.get());
    children.push_back(std::move(passthrough));
  }

  // Call the C API directly to test the error path (CreateSequence would
  // release children on failure, but we want to verify the error).
  iree_tokenizer_decoder_t* raw_decoder = nullptr;
  iree_status_t status = iree_tokenizer_decoder_sequence_allocate(
      raw_pointers.data(), raw_pointers.size(), iree_allocator_system(),
      &raw_decoder);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  // children vector cleans up all decoders on scope exit.
}

//===----------------------------------------------------------------------===//
// Minimum 2 Children Required
//===----------------------------------------------------------------------===//

// Note: Empty and single-child sequences are rejected by the allocator.
// The JSON parser handles those cases by returning NULL (empty) or unwrapping
// (single child). These tests verify the allocator contract.

//===----------------------------------------------------------------------===//
// Two Decoders Chained
//===----------------------------------------------------------------------===//

TEST_F(SequenceDecoderTest, ByteFallbackThenMetaspace) {
  // This is the real-world Mistral/TinyLlama decoder chain pattern.
  ScopedDecoder byte_fallback;
  IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), byte_fallback.put()));

  ScopedDecoder metaspace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
      0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS,
      iree_allocator_system(), metaspace.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(byte_fallback));
  children.push_back(std::move(metaspace));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Input: byte tokens for ▁ (E2 96 81) followed by "Hello"
  // ByteFallback: "<0xE2><0x96><0x81>Hello" -> "▁Hello" (8 bytes)
  // Metaspace: "▁Hello" -> "Hello" (strips leading metaspace)
  std::vector<std::string> tokens = {"<0xE2>", "<0x96>", "<0x81>", "Hello"};
  TestWithAllBatchSizes(decoder.get(), tokens, "Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceDecoderTest, ByteFallbackThenMetaspaceMultipleWords) {
  ScopedDecoder byte_fallback;
  IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), byte_fallback.put()));

  ScopedDecoder metaspace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
      0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS,
      iree_allocator_system(), metaspace.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(byte_fallback));
  children.push_back(std::move(metaspace));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // "▁Hello▁World" encoded as byte tokens + text
  // ▁ = E2 96 81
  std::vector<std::string> tokens = {"<0xE2>", "<0x96>", "<0x81>", "Hello",
                                     "<0xE2>", "<0x96>", "<0x81>", "World"};

  TestWithAllBatchSizes(decoder.get(), tokens, "Hello World",
                        /*expect_pending_after_process=*/false);
}

//===----------------------------------------------------------------------===//
// Streaming with Pending Data
//===----------------------------------------------------------------------===//

TEST_F(SequenceDecoderTest, StreamingWithPendingFromFirstDecoder) {
  ScopedDecoder byte_fallback;
  IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), byte_fallback.put()));

  ScopedDecoder metaspace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
      0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS,
      iree_allocator_system(), metaspace.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(byte_fallback));
  children.push_back(std::move(metaspace));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  ScopedDecoderState state(decoder.get());

  // Send incomplete byte sequence.
  std::vector<std::string> tokens1 = {"<0xE2>"};
  auto views1 = ToStringViews(tokens1);
  iree_tokenizer_string_list_t list1 = {views1.size(), views1.data()};

  char output[64];
  iree_host_size_t strings_consumed = 0;
  iree_host_size_t bytes_written = 0;
  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), list1, iree_make_mutable_string_view(output, sizeof(output)),
      &strings_consumed, &bytes_written));

  EXPECT_EQ(strings_consumed, 1u);
  EXPECT_EQ(bytes_written, 0u);  // Waiting for more bytes.
  EXPECT_TRUE(iree_tokenizer_decoder_state_has_pending(state.get()));

  // Complete the sequence.
  std::vector<std::string> tokens2 = {"<0x96>", "<0x81>", "Hello"};
  auto views2 = ToStringViews(tokens2);
  iree_tokenizer_string_list_t list2 = {views2.size(), views2.data()};

  IREE_ASSERT_OK(iree_tokenizer_decoder_state_process(
      state.get(), list2, iree_make_mutable_string_view(output, sizeof(output)),
      &strings_consumed, &bytes_written));

  EXPECT_EQ(strings_consumed, 3u);
  // "▁Hello" after ByteFallback, then "Hello" after Metaspace strips ▁.
  EXPECT_EQ(bytes_written, 5u);
  EXPECT_EQ(std::string(output, bytes_written), "Hello");
}

//===----------------------------------------------------------------------===//
// HuggingFace Compatibility
//===----------------------------------------------------------------------===//

TEST_F(SequenceDecoderTest, HF_MistralPattern) {
  // Mistral uses: Replace(▁→ ) → ByteFallback → Fuse → Strip
  // We don't have Replace/Fuse/Strip yet, but ByteFallback → Metaspace
  // approximates the core functionality.
  ScopedDecoder byte_fallback;
  IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), byte_fallback.put()));

  ScopedDecoder metaspace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_metaspace_allocate(
      0, IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS,
      iree_allocator_system(), metaspace.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(byte_fallback));
  children.push_back(std::move(metaspace));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Simulated Mistral token sequence for "Hello World"
  std::vector<std::string> tokens = {"<0xE2>", "<0x96>", "<0x81>", "Hello",
                                     "<0xE2>", "<0x96>", "<0x81>", "World"};

  std::string result =
      ProcessAndFinalize(decoder.get(), tokens,
                         /*expect_pending_after_process=*/false);
  EXPECT_EQ(result, "Hello World");
}

//===----------------------------------------------------------------------===//
// Memory Overlap Tests
//===----------------------------------------------------------------------===//
// These tests verify that the sequence decoder correctly handles overlapping
// memory regions. The sequence decoder processes in-place where output of one
// decoder is input to the next, using the same scratch buffer. This requires
// memmove (not memcpy) in all decoders that copy from input to output.

TEST_F(SequenceDecoderTest, OverlapReplaceShrinking) {
  // Replace decoder shrinks data: "▁" (3 bytes) → " " (1 byte).
  // This shifts all following data left, requiring memmove.
  static const std::string kMetaspace = "\xE2\x96\x81";

  ScopedDecoder replace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_replace_allocate(
      iree_make_cstring_view(kMetaspace.c_str()), iree_make_cstring_view(" "),
      iree_allocator_system(), replace.put()));

  ScopedDecoder passthrough;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), passthrough.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(replace));
  children.push_back(std::move(passthrough));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Input: "Hello▁World▁Test" (20 bytes)
  // After replace: "Hello World Test" (16 bytes)
  // Passthrough just verifies no corruption.
  std::vector<std::string> tokens = {"Hello" + kMetaspace + "World" +
                                     kMetaspace + "Test"};

  TestWithAllBatchSizes(decoder.get(), tokens, "Hello World Test",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceDecoderTest, OverlapMultipleShrinkingDecoders) {
  // Chain two shrinking decoders to stress test overlap handling.
  static const std::string kMetaspace = "\xE2\x96\x81";

  // First: Replace "▁" (3 bytes) → " " (1 byte).
  ScopedDecoder replace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_replace_allocate(
      iree_make_cstring_view(kMetaspace.c_str()), iree_make_cstring_view(" "),
      iree_allocator_system(), replace.put()));

  // Second: Strip leading " " from each token (removes 1 byte per space).
  ScopedDecoder strip;
  IREE_ASSERT_OK(iree_tokenizer_decoder_strip_allocate(
      iree_make_cstring_view(" "), /*start_count=*/1, /*stop_count=*/0,
      iree_allocator_system(), strip.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(replace));
  children.push_back(std::move(strip));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Input: " ▁Hello" (has leading space + metaspace)
  // After replace: "  Hello" (space + space + Hello)
  // After strip (removes 1 leading space): " Hello"
  // Note: strip only removes 1 occurrence per call, not all.
  std::vector<std::string> tokens = {" " + kMetaspace + "Hello"};

  TestWithAllBatchSizes(decoder.get(), tokens, " Hello",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceDecoderTest, OverlapWithLargeDataReduction) {
  // Replace a long pattern with a short one many times.
  // This creates significant memory movement with overlap.
  ScopedDecoder replace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_replace_allocate(
      iree_make_cstring_view("XXXX"), iree_make_cstring_view("Y"),
      iree_allocator_system(), replace.put()));

  ScopedDecoder passthrough;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), passthrough.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(replace));
  children.push_back(std::move(passthrough));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Input: "XXXXAXXXXBXXXXCXXXXD" (20 bytes)
  // After replace: "YAYBYCYD" (8 bytes)
  // After passthrough: "YAYBYCYD" (unchanged)
  std::vector<std::string> tokens = {"XXXXAXXXXBXXXXCXXXXD"};

  TestWithAllBatchSizes(decoder.get(), tokens, "YAYBYCYD",
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceDecoderTest, OverlapPassthroughPreservesData) {
  // Multiple passthrough decoders should preserve data exactly.
  // This verifies that passthrough's memmove doesn't corrupt data.
  ScopedDecoder pass1;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), pass1.put()));

  ScopedDecoder pass2;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), pass2.put()));

  ScopedDecoder pass3;
  IREE_ASSERT_OK(iree_tokenizer_decoder_passthrough_allocate(
      iree_allocator_system(), pass3.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(pass1));
  children.push_back(std::move(pass2));
  children.push_back(std::move(pass3));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Test with binary data that could expose corruption.
  std::vector<std::string> tokens = {std::string("\x00\x01\x02\x03", 4),
                                     "middle", std::string("\xFE\xFF", 2)};

  std::string expected = std::string("\x00\x01\x02\x03", 4) + "middle" +
                         std::string("\xFE\xFF", 2);
  TestWithAllBatchSizes(decoder.get(), tokens, expected,
                        /*expect_pending_after_process=*/false);
}

TEST_F(SequenceDecoderTest, OverlapByteFallbackAndReplace) {
  // Real-world pattern: ByteFallback converts byte tokens, then Replace
  // converts metaspace to space. Data shrinks twice.
  ScopedDecoder byte_fallback;
  IREE_ASSERT_OK(iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), byte_fallback.put()));

  static const std::string kMetaspace = "\xE2\x96\x81";
  ScopedDecoder replace;
  IREE_ASSERT_OK(iree_tokenizer_decoder_replace_allocate(
      iree_make_cstring_view(kMetaspace.c_str()), iree_make_cstring_view(" "),
      iree_allocator_system(), replace.put()));

  std::vector<ScopedDecoder> children;
  children.push_back(std::move(byte_fallback));
  children.push_back(std::move(replace));
  auto decoder = CreateSequence(std::move(children));
  ASSERT_NE(decoder.get(), nullptr);

  // Input: byte tokens for "▁" + "Hello" + byte tokens for "▁" + "World"
  // ByteFallback: outputs "▁Hello▁World"
  // Replace: outputs " Hello World"
  std::vector<std::string> tokens = {"<0xE2>", "<0x96>", "<0x81>", "Hello",
                                     "<0xE2>", "<0x96>", "<0x81>", "World"};

  TestWithAllBatchSizes(decoder.get(), tokens, " Hello World",
                        /*expect_pending_after_process=*/false);
}

}  // namespace
}  // namespace iree::tokenizer
