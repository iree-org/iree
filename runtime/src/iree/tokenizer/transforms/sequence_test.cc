// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/transforms/transform.h"

namespace {

// Callback that accumulates segments into a vector.
struct EncodeContext {
  std::vector<std::string>* segments;
};

static iree_status_t AccumulateSegments(void* user_data,
                                        iree_string_view_list_t segments) {
  auto* context = static_cast<EncodeContext*>(user_data);
  for (size_t i = 0; i < segments.count; ++i) {
    context->segments->push_back(
        std::string(segments.values[i].data, segments.values[i].size));
  }
  return iree_ok_status();
}

class SequenceTransformTest : public ::testing::Test {
 protected:
  void SetUp() override { allocator_ = iree_allocator_system(); }

  std::vector<std::string> Encode(
      const iree_tokenizer_text_transform_t* transform, const char* text) {
    std::vector<std::string> segments;
    EncodeContext context = {&segments};
    iree_status_t status = iree_tokenizer_text_transform_encode(
        NULL, transform, IREE_SV(text), AccumulateSegments, &context);
    IREE_EXPECT_OK(status);
    return segments;
  }

  std::string Decode(const iree_tokenizer_text_transform_t* transform,
                     const char* text) {
    char decoded[1024];
    iree_host_size_t decoded_size = 0;
    iree_status_t status = iree_tokenizer_text_transform_decode(
        transform, IREE_SV(text), decoded, sizeof(decoded), &decoded_size);
    IREE_EXPECT_OK(status);
    return std::string(decoded, decoded_size);
  }

  iree_allocator_t allocator_;
};

TEST_F(SequenceTransformTest, EmptySequence) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      nullptr, 0, allocator_, &transform));

  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "hello world");

  auto decoded = Decode(&transform, "hello world");
  EXPECT_EQ(decoded, "hello world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, SingleTransformBert) {
  iree_tokenizer_text_transform_t bert;
  iree_tokenizer_text_transform_initialize_bert(&bert);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &bert, 1, allocator_, &transform));
  // bert ownership transferred - do not deinitialize

  auto segments = Encode(&transform, "hello, world!");
  // BERT splits on punctuation and whitespace.
  EXPECT_GE(segments.size(), 3u);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, SingleTransformMetaspace) {
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &metaspace);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &transform));
  // metaspace ownership transferred

  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");  // ▁hello
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");  // ▁world

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, TwoTransformsMetaspaceThenWhitespace) {
  // Metaspace replaces spaces with ▁, then Whitespace would split.
  // But Metaspace with SPLIT already splits, so Whitespace is a no-op.
  iree_tokenizer_text_transform_t children[2];
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_DEFAULT, &children[0]);
  iree_tokenizer_text_transform_initialize_whitespace(&children[1]);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &transform));
  // children ownership transferred

  // Metaspace without SPLIT produces a single segment with ▁.
  // Whitespace then does nothing (no actual whitespace in the output).
  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\xE2\x96\x81world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, RoundTripWithMetaspace) {
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_DEFAULT, &metaspace);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &transform));

  const char* original = "hello world";
  auto segments = Encode(&transform, original);
  ASSERT_EQ(segments.size(), 1);

  // Decode the encoded output.
  auto decoded = Decode(&transform, segments[0].c_str());
  EXPECT_EQ(decoded, original);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, DecodeReversesOrder) {
  // Two transforms: Metaspace (replaces space with ▁) then simulated chain.
  // Decode should apply in reverse: last transform decode first.
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_DEFAULT, &metaspace);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &transform));

  // Decode: ▁hello▁world -> hello world (metaspace decode strips leading ▁).
  auto decoded = Decode(&transform, "\xE2\x96\x81hello\xE2\x96\x81world");
  EXPECT_EQ(decoded, "hello world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Nested Sequence Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceTransformTest, NestedSequenceBasic) {
  // Create inner sequence with BERT.
  iree_tokenizer_text_transform_t bert;
  iree_tokenizer_text_transform_initialize_bert(&bert);

  iree_tokenizer_text_transform_t inner_sequence;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &bert, 1, allocator_, &inner_sequence));

  // Create outer sequence containing the inner sequence.
  iree_tokenizer_text_transform_t outer_sequence;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &inner_sequence, 1, allocator_, &outer_sequence));

  // Verify it works - BERT splits on punctuation.
  auto segments = Encode(&outer_sequence, "hello, world!");
  EXPECT_GE(segments.size(), 3u);

  iree_tokenizer_text_transform_deinitialize(&outer_sequence);
}

TEST_F(SequenceTransformTest, NestedSequenceDeepThreeLevels) {
  // Create level 3 (innermost): Whitespace.
  iree_tokenizer_text_transform_t whitespace;
  iree_tokenizer_text_transform_initialize_whitespace(&whitespace);

  iree_tokenizer_text_transform_t level3;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &whitespace, 1, allocator_, &level3));

  // Create level 2: wraps level 3.
  iree_tokenizer_text_transform_t level2;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &level3, 1, allocator_, &level2));

  // Create level 1 (outermost): wraps level 2.
  iree_tokenizer_text_transform_t level1;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &level2, 1, allocator_, &level1));

  // Whitespace splits on spaces, even through 3 levels of nesting.
  auto segments = Encode(&level1, "a b c");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "a");
  EXPECT_EQ(segments[1], "b");
  EXPECT_EQ(segments[2], "c");

  iree_tokenizer_text_transform_deinitialize(&level1);
}

TEST_F(SequenceTransformTest, NestedSequenceMixedWithSimpleTransforms) {
  // Inner sequence: Metaspace (no split).
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_DEFAULT, &metaspace);

  iree_tokenizer_text_transform_t inner_seq;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &inner_seq));

  // Outer sequence: [inner_seq, Whitespace].
  iree_tokenizer_text_transform_t children[2];
  children[0] = inner_seq;
  iree_tokenizer_text_transform_initialize_whitespace(&children[1]);

  iree_tokenizer_text_transform_t outer;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &outer));

  // Metaspace replaces spaces with ▁, then no whitespace remains to split.
  auto segments = Encode(&outer, "hello world");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello\xE2\x96\x81world");

  iree_tokenizer_text_transform_deinitialize(&outer);
}

//===----------------------------------------------------------------------===//
// Degenerate Case Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceTransformTest, SequenceOfNoneTransforms) {
  // Create array of NONE transforms.
  iree_tokenizer_text_transform_t children[3];
  iree_tokenizer_text_transform_initialize_none(&children[0]);
  iree_tokenizer_text_transform_initialize_none(&children[1]);
  iree_tokenizer_text_transform_initialize_none(&children[2]);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 3, allocator_, &transform));

  // Three NONEs in sequence should be passthrough.
  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "hello world");

  auto decoded = Decode(&transform, "hello world");
  EXPECT_EQ(decoded, "hello world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NestedEmptySequence) {
  // Create empty inner sequence.
  iree_tokenizer_text_transform_t inner;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      nullptr, 0, allocator_, &inner));

  // Wrap in outer sequence.
  iree_tokenizer_text_transform_t outer;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &inner, 1, allocator_, &outer));

  // Empty sequence becomes NONE, so this is Sequence[NONE] = passthrough.
  auto segments = Encode(&outer, "test");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "test");

  iree_tokenizer_text_transform_deinitialize(&outer);
}

TEST_F(SequenceTransformTest, MultipleNestedEmptySequences) {
  // Create several empty sequences and nest them.
  iree_tokenizer_text_transform_t empty1, empty2;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      nullptr, 0, allocator_, &empty1));
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      nullptr, 0, allocator_, &empty2));

  iree_tokenizer_text_transform_t children[2] = {empty1, empty2};
  iree_tokenizer_text_transform_t outer;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &outer));

  // Sequence[NONE, NONE] = passthrough.
  auto segments = Encode(&outer, "test");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "test");

  iree_tokenizer_text_transform_deinitialize(&outer);
}

TEST_F(SequenceTransformTest, SingleChildSequenceChain) {
  // Chain of single-child sequences: Seq[Seq[Seq[Whitespace]]].
  iree_tokenizer_text_transform_t ws;
  iree_tokenizer_text_transform_initialize_whitespace(&ws);

  iree_tokenizer_text_transform_t seq1;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &ws, 1, allocator_, &seq1));

  iree_tokenizer_text_transform_t seq2;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &seq1, 1, allocator_, &seq2));

  iree_tokenizer_text_transform_t seq3;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &seq2, 1, allocator_, &seq3));

  // Should still split on whitespace.
  auto segments = Encode(&seq3, "a b c d");
  ASSERT_EQ(segments.size(), 4);

  iree_tokenizer_text_transform_deinitialize(&seq3);
}

//===----------------------------------------------------------------------===//
// Multi-Child and Realistic Combination Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceTransformTest, AllTransformTypesInSequence) {
  // Sequence with all transform types (in a sensible order).
  // ByteLevel -> BERT -> Whitespace.
  iree_tokenizer_text_transform_t children[3];
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &children[0]);
  iree_tokenizer_text_transform_initialize_bert(&children[1]);
  iree_tokenizer_text_transform_initialize_whitespace(&children[2]);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 3, allocator_, &transform));

  // Just verify it runs without crashing.
  auto segments = Encode(&transform, "hello, world!");
  EXPECT_GE(segments.size(), 1u);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, RealisticGPT2Style) {
  // GPT-2 uses ByteLevel with add_prefix_space.
  iree_tokenizer_text_transform_t byte_level;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE, &byte_level);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &byte_level, 1, allocator_, &transform));

  // ByteLevel with prefix space adds Ġ at start.
  auto segments = Encode(&transform, "hello");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xC4\xA0hello");  // Ġhello

  // Round-trip.
  auto decoded = Decode(&transform, segments[0].c_str());
  EXPECT_EQ(decoded, "hello");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, RealisticLlamaStyle) {
  // Llama uses Metaspace with ▁ replacement.
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &metaspace);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &transform));

  auto segments = Encode(&transform, "Hello world");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81Hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, ManyChildrenSequence) {
  // Sequence with many transforms (stress test).
  constexpr size_t kCount = 10;
  iree_tokenizer_text_transform_t children[kCount];
  for (size_t i = 0; i < kCount; ++i) {
    iree_tokenizer_text_transform_initialize_none(&children[i]);
  }
  // Make the last one actually do something.
  iree_tokenizer_text_transform_initialize_whitespace(&children[kCount - 1]);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, kCount, allocator_, &transform));

  // Nine NONEs then Whitespace = splits on whitespace.
  auto segments = Encode(&transform, "a b c");
  ASSERT_EQ(segments.size(), 3);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Move Semantics Verification Tests
//===----------------------------------------------------------------------===//

TEST_F(SequenceTransformTest, MoveSemanticsSingleChild) {
  // Verify move semantics: source is zeroed after initialize_sequence.
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &metaspace);

  iree_tokenizer_text_transform_t seq;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &seq));

  // Source should be zeroed (moved from).
  EXPECT_EQ(metaspace.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);

  // Sequence should work.
  auto segments = Encode(&seq, "hello world");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");

  iree_tokenizer_text_transform_deinitialize(&seq);
}

TEST_F(SequenceTransformTest, MoveSemanticsNestedSequence) {
  // Create nested sequence, verify sources are zeroed.
  iree_tokenizer_text_transform_t whitespace;
  iree_tokenizer_text_transform_initialize_whitespace(&whitespace);

  iree_tokenizer_text_transform_t inner;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &whitespace, 1, allocator_, &inner));
  EXPECT_EQ(whitespace.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);

  iree_tokenizer_text_transform_t outer;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &inner, 1, allocator_, &outer));
  EXPECT_EQ(inner.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);

  // Outer should work.
  auto segments = Encode(&outer, "a b c");
  ASSERT_EQ(segments.size(), 3);

  iree_tokenizer_text_transform_deinitialize(&outer);
}

TEST_F(SequenceTransformTest, MoveSemanticsMultipleChildren) {
  // Create sequence with multiple children, verify all sources are zeroed.
  iree_tokenizer_text_transform_t children[3];
  iree_tokenizer_text_transform_initialize_bert(&children[0]);
  iree_tokenizer_text_transform_initialize_none(&children[1]);
  iree_tokenizer_text_transform_initialize_whitespace(&children[2]);

  iree_tokenizer_text_transform_t seq;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 3, allocator_, &seq));

  // All sources should be zeroed.
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(children[i].type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);
  }

  // Sequence should work.
  auto segments = Encode(&seq, "hello, world!");
  EXPECT_GE(segments.size(), 1u);

  iree_tokenizer_text_transform_deinitialize(&seq);
}

TEST_F(SequenceTransformTest, MovedFromChildDeinitIsSafe) {
  // Verify that deinitializing a moved-from child is safe (doesn't crash).
  // This is important because users might accidentally try to clean up children
  // after passing them to initialize_sequence.
  iree_tokenizer_text_transform_t bert;
  iree_tokenizer_text_transform_initialize_bert(&bert);

  iree_tokenizer_text_transform_t seq;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &bert, 1, allocator_, &seq));

  // After move, bert should be zeroed (type = NONE).
  EXPECT_EQ(bert.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);

  // Deinitializing the moved-from child should be safe (no crash, no
  // double-free). This is safe because the move zeroes the source, making it a
  // valid NONE transform.
  iree_tokenizer_text_transform_deinitialize(&bert);

  // Sequence should still work correctly.
  auto segments = Encode(&seq, "hello, world!");
  EXPECT_GE(segments.size(), 3u);

  iree_tokenizer_text_transform_deinitialize(&seq);
}

TEST_F(SequenceTransformTest, MovedFromChildBehavesAsNone) {
  // Verify that a moved-from transform behaves as NONE (passthrough).
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &metaspace);

  iree_tokenizer_text_transform_t seq;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &seq));

  // The moved-from metaspace should now behave as NONE.
  EXPECT_EQ(metaspace.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);

  // Using the moved-from transform should work as passthrough.
  auto segments = Encode(&metaspace, "hello world");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "hello world");  // No metaspace transformation.

  // Decode should also be passthrough.
  auto decoded = Decode(&metaspace, "hello world");
  EXPECT_EQ(decoded, "hello world");

  iree_tokenizer_text_transform_deinitialize(&seq);
}

//===----------------------------------------------------------------------===//
// Round-Trip Tests with Complex Structures
//===----------------------------------------------------------------------===//

TEST_F(SequenceTransformTest, RoundTripNestedSequence) {
  // Create nested sequence with Metaspace.
  iree_tokenizer_text_transform_t metaspace;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_DEFAULT, &metaspace);

  iree_tokenizer_text_transform_t inner;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &metaspace, 1, allocator_, &inner));

  iree_tokenizer_text_transform_t outer;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &inner, 1, allocator_, &outer));

  const char* original = "hello world test";
  auto segments = Encode(&outer, original);
  ASSERT_EQ(segments.size(), 1);

  auto decoded = Decode(&outer, segments[0].c_str());
  EXPECT_EQ(decoded, original);

  iree_tokenizer_text_transform_deinitialize(&outer);
}

TEST_F(SequenceTransformTest, RoundTripByteLevelSequence) {
  iree_tokenizer_text_transform_t byte_level;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE, &byte_level);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &byte_level, 1, allocator_, &transform));

  // Test with various characters including spaces and special chars.
  const char* original = "Hello World! Test 123";
  auto segments = Encode(&transform, original);
  ASSERT_EQ(segments.size(), 1);

  auto decoded = Decode(&transform, segments[0].c_str());
  EXPECT_EQ(decoded, original);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, RoundTripMultiChildSequence) {
  // Sequence with ByteLevel (no prefix) - should round-trip.
  iree_tokenizer_text_transform_t children[2];
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &children[0]);
  iree_tokenizer_text_transform_initialize_none(&children[1]);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &transform));

  const char* original = "hello world";
  auto segments = Encode(&transform, original);
  ASSERT_EQ(segments.size(), 1);

  auto decoded = Decode(&transform, segments[0].c_str());
  EXPECT_EQ(decoded, original);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Normalizer Integration Tests (No JSON)
//===----------------------------------------------------------------------===//

TEST_F(SequenceTransformTest, NormalizerWithWhitespaceTransform) {
  // Set up whitespace transform with lowercase normalizer directly.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_whitespace(&transform);
  iree_tokenizer_normalizer_initialize_lowercase(&transform.normalizer);

  // Verify normalizer is applied per-segment.
  auto segments = Encode(&transform, "HELLO WORLD");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NormalizerWithBertTransform) {
  // BERT transform with strip_accents normalizer.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_bert(&transform);
  iree_tokenizer_normalizer_initialize_strip_accents(&transform.normalizer);

  // BERT splits on punctuation, normalizer strips accents.
  auto segments = Encode(&transform, "café résumé");
  ASSERT_GE(segments.size(), 2u);
  EXPECT_EQ(segments[0], "cafe");
  EXPECT_EQ(segments[1], "resume");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NormalizerNoneHasNoEffect) {
  // Verify NONE normalizer doesn't modify segments.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_whitespace(&transform);
  // normalizer is already NONE by default, but be explicit.
  iree_tokenizer_normalizer_initialize_none(&transform.normalizer);

  auto segments = Encode(&transform, "HELLO WORLD");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "HELLO");  // Unchanged.
  EXPECT_EQ(segments[1], "WORLD");  // Unchanged.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NormalizerWithSequenceTransform) {
  // Sequence transform with normalizer.
  iree_tokenizer_text_transform_t whitespace;
  iree_tokenizer_text_transform_initialize_whitespace(&whitespace);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &whitespace, 1, allocator_, &transform));

  // Add normalizer after sequence is created.
  iree_tokenizer_normalizer_initialize_lowercase(&transform.normalizer);

  auto segments = Encode(&transform, "HELLO WORLD TEST");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
  EXPECT_EQ(segments[2], "test");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NormalizerWithNestedSequence) {
  // Nested sequence with normalizer on outer.
  iree_tokenizer_text_transform_t whitespace;
  iree_tokenizer_text_transform_initialize_whitespace(&whitespace);

  iree_tokenizer_text_transform_t inner;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &whitespace, 1, allocator_, &inner));

  iree_tokenizer_text_transform_t outer;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      &inner, 1, allocator_, &outer));

  // Add BERT normalizer (lowercase + strip accents).
  iree_tokenizer_normalizer_initialize_bert(
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE |
          IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS,
      &outer.normalizer);

  auto segments = Encode(&outer, "CAFÉ RÉSUMÉ");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "cafe");
  EXPECT_EQ(segments[1], "resume");

  iree_tokenizer_text_transform_deinitialize(&outer);
}

TEST_F(SequenceTransformTest, NormalizerSequenceWithTransform) {
  // Transform with sequence normalizer (strip accents then lowercase).
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_whitespace(&transform);

  // Create sequence normalizer.
  iree_tokenizer_normalizer_t norm_children[2];
  iree_tokenizer_normalizer_initialize_strip_accents(&norm_children[0]);
  iree_tokenizer_normalizer_initialize_lowercase(&norm_children[1]);
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_sequence(
      norm_children, 2, allocator_, &transform.normalizer));

  auto segments = Encode(&transform, "CAFÉ NAÏVE");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "cafe");
  EXPECT_EQ(segments[1], "naive");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NormalizerWithMetaspaceTransform) {
  // Metaspace transform (splits on ▁) with lowercase normalizer.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &transform);
  iree_tokenizer_normalizer_initialize_lowercase(&transform.normalizer);

  auto segments = Encode(&transform, "HELLO WORLD");
  ASSERT_EQ(segments.size(), 2);
  // Metaspace prepends ▁, normalizer lowercases.
  EXPECT_EQ(segments[0], "\xE2\x96\x81hello");
  EXPECT_EQ(segments[1], "\xE2\x96\x81world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NormalizerWithMultiChildSequence) {
  // Multi-child sequence with normalizer.
  iree_tokenizer_text_transform_t children[2];
  iree_tokenizer_text_transform_initialize_metaspace(
      0x2581, IREE_TOKENIZER_PREPEND_ALWAYS,
      IREE_TOKENIZER_METASPACE_FLAG_SPLIT, &children[0]);
  iree_tokenizer_text_transform_initialize_whitespace(&children[1]);

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &transform));

  // Add normalizer: strip accents.
  iree_tokenizer_normalizer_initialize_strip_accents(&transform.normalizer);

  // Input has accents; should be stripped.
  auto segments = Encode(&transform, "café résumé");
  ASSERT_GE(segments.size(), 2u);
  // Each segment should have accents stripped.
  for (const auto& seg : segments) {
    EXPECT_EQ(seg.find("é"), std::string::npos);  // No accented chars.
  }

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NoneTransformWithNormalizer) {
  // NONE transform with Prepend normalizer (TinyLlama style).
  // This is the key case: TinyLlama has pre_tokenizer=null (NONE transform)
  // but has a normalizer chain. The NONE transform must apply the normalizer.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_none(&transform);
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_prepend(
      IREE_SV("\xE2\x96\x81"), &transform.normalizer));  // ▁ (U+2581)

  // Verify normalizer is applied even with NONE transform.
  auto segments = Encode(&transform, "Hello");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81Hello");  // Prepended ▁

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, NoneTransformWithSequenceNormalizer) {
  // NONE transform with TinyLlama-style normalizer chain:
  // Prepend("▁") + Replace(" ", "▁")
  // This is the complete TinyLlama normalizer configuration.
  iree_tokenizer_text_transform_t transform;
  iree_tokenizer_text_transform_initialize_none(&transform);

  // Build sequence normalizer: Prepend("▁"), Replace(" ", "▁").
  iree_tokenizer_normalizer_t norm_children[2];
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_prepend(
      IREE_SV("\xE2\x96\x81"), &norm_children[0]));
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_replace(
      IREE_SV(" "), IREE_SV("\xE2\x96\x81"), &norm_children[1]));
  IREE_ASSERT_OK(iree_tokenizer_normalizer_initialize_sequence(
      norm_children, 2, allocator_, &transform.normalizer));

  // "Hello World" -> "▁Hello World" (Prepend) -> "▁Hello▁World" (Replace).
  auto segments = Encode(&transform, "Hello World");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "\xE2\x96\x81Hello\xE2\x96\x81World");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// GPT-2 Style: Sequence[Split(GPT2), ByteLevel]
//===----------------------------------------------------------------------===//

// This is the critical test for GPT-2 style tokenizers.
// The Split transform produces ["hello", " world"], then ByteLevel maps bytes.
// The space in " world" becomes Ġ (0xC4 0xA0), giving "Ġworld".
TEST_F(SequenceTransformTest, GPT2StyleSplitThenByteLevel) {
  // Create Split transform with GPT-2 pattern.
  iree_tokenizer_text_transform_t split;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_split(
      IREE_SV("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
              "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"),
      IREE_TOKENIZER_REGEX_SPLIT_REMOVED,  // We use invert=true.
      true,                                // invert = true (emit matches).
      allocator_, &split));

  // Create ByteLevel transform.
  iree_tokenizer_text_transform_t byte_level;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &byte_level);

  // Create Sequence[Split, ByteLevel].
  iree_tokenizer_text_transform_t children[2] = {split, byte_level};
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &transform));

  // "hello world" should produce:
  // 1. Split: ["hello", " world"]
  // 2. ByteLevel on each: ["hello", "Ġworld"]
  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "\xC4\xA0world");  // Ġworld (space mapped to 0x120).

  iree_tokenizer_text_transform_deinitialize(&transform);
}

// Test with behavior=1 (ISOLATED), invert=false - the configuration that
// ByteLevel pretokenizer with use_regex=true should use.
// ISOLATED emits gaps then matches. Since GPT-2 pattern matches everything,
// gaps are empty and we get just matches.
TEST_F(SequenceTransformTest, GPT2StyleIsolatedBehavior) {
  iree_tokenizer_text_transform_t split;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_split(
      IREE_SV("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
              "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"),
      IREE_TOKENIZER_REGEX_SPLIT_ISOLATED,  // Gaps then matches.
      false,                                // invert = false.
      allocator_, &split));

  iree_tokenizer_text_transform_t byte_level;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &byte_level);

  iree_tokenizer_text_transform_t children[2] = {split, byte_level};
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &transform));

  // "hello world" should produce:
  // 1. Split: ["hello", " world"] (ISOLATED with no gaps)
  // 2. ByteLevel on each: ["hello", "Ġworld"]
  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "\xC4\xA0world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(SequenceTransformTest, GPT2StyleMixedContent) {
  iree_tokenizer_text_transform_t split;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_split(
      IREE_SV("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
              "?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"),
      IREE_TOKENIZER_REGEX_SPLIT_REMOVED, true, allocator_, &split));

  iree_tokenizer_text_transform_t byte_level;
  iree_tokenizer_text_transform_initialize_byte_level(
      IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT, &byte_level);

  iree_tokenizer_text_transform_t children[2] = {split, byte_level};
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_initialize_sequence(
      children, 2, allocator_, &transform));

  // "Hello, world!" should produce:
  // 1. Split: ["Hello", ",", " world", "!"]
  // 2. ByteLevel: ["Hello", ",", "Ġworld", "!"]
  auto segments = Encode(&transform, "Hello, world!");
  ASSERT_EQ(segments.size(), 4u);
  EXPECT_EQ(segments[0], "Hello");
  EXPECT_EQ(segments[1], ",");
  EXPECT_EQ(segments[2], "\xC4\xA0world");  // Ġworld.
  EXPECT_EQ(segments[3], "!");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

}  // namespace
