// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/transform_json.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/transforms/transform.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

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

class TransformJsonTest : public ::testing::Test {
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

  iree_allocator_t allocator_;
};

//===----------------------------------------------------------------------===//
// parse_json tests (root tokenizer.json)
//===----------------------------------------------------------------------===//

TEST_F(TransformJsonTest, MissingPreTokenizer) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV("{\"vocab\": {}}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);
  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, NullPreTokenizer) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV("{\"pre_tokenizer\": null}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);
  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// parse_pretokenizer tests
//===----------------------------------------------------------------------===//

TEST_F(TransformJsonTest, ParseBertPreTokenizer) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"BertPreTokenizer\"}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_BERT);

  // Verify it works: BERT splits on punctuation.
  auto segments = Encode(&transform, "hello, world!");
  EXPECT_GE(segments.size(), 3u);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseWhitespace) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Whitespace\"}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE);

  auto segments = Encode(&transform, "hello world foo");
  ASSERT_EQ(segments.size(), 3);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "world");
  EXPECT_EQ(segments[2], "foo");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseWhitespaceSplitAlias) {
  // WhitespaceSplit is an alias for Whitespace (HuggingFace naming).
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"WhitespaceSplit\"}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseByteLevelDefault) {
  // use_regex=true (default) creates Sequence[Split(GPT2), ByteLevel].
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"ByteLevel\"}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE);
  EXPECT_EQ(transform.config.sequence.count, 2u);
  EXPECT_EQ(transform.config.sequence.children[0].type,
            IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT);
  EXPECT_EQ(transform.config.sequence.children[1].type,
            IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseByteLevelNoRegex) {
  // use_regex=false creates plain ByteLevel (no GPT-2 regex splitting).
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"ByteLevel\", \"use_regex\": false}"), allocator_,
      &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL);
  EXPECT_EQ(transform.config.byte_level.flags,
            IREE_TOKENIZER_BYTE_LEVEL_FLAG_DEFAULT);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseByteLevelWithPrefixSpace) {
  // use_regex=true (default) with add_prefix_space creates sequence.
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"ByteLevel\", \"add_prefix_space\": true}"),
      allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE);
  EXPECT_EQ(transform.config.sequence.count, 2u);
  // ByteLevel is the second child with ADD_PREFIX_SPACE flag.
  EXPECT_EQ(transform.config.sequence.children[1].type,
            IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL);
  EXPECT_EQ(transform.config.sequence.children[1].config.byte_level.flags,
            IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseByteLevelWithPrefixSpaceNoRegex) {
  // use_regex=false with add_prefix_space creates plain ByteLevel.
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"ByteLevel\", \"add_prefix_space\": true, "
              "\"use_regex\": false}"),
      allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL);
  EXPECT_EQ(transform.config.byte_level.flags,
            IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

// Verifies the GPT-2 ByteLevel pretokenizer (use_regex=true) correctly
// keeps " world" as a single segment before byte-level encoding.
// GPT-2 pattern: ' ?\p{L}+' matches optional-space-plus-letters.
TEST_F(TransformJsonTest, ByteLevelHelloWorldSegmentation) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"ByteLevel\"}"), allocator_, &transform));

  // The GPT-2 pattern splits "hello world" into ["hello", " world"].
  // ByteLevel then maps space (0x20) to Ġ (U+0120, UTF-8: C4 A0).
  // Result should be: ["hello", "Ġworld"]
  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 2u);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "\xC4\xA0world");  // Ġworld

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseMetaspaceDefault) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Metaspace\"}"), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE);
  EXPECT_EQ(transform.config.metaspace.replacement, 0x2581u);
  EXPECT_EQ(transform.config.metaspace.prepend_scheme,
            IREE_TOKENIZER_PREPEND_ALWAYS);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseMetaspaceWithOptions) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Metaspace\", \"replacement\": \"_\", "
              "\"prepend_scheme\": \"never\", \"split\": true}"),
      allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE);
  EXPECT_EQ(transform.config.metaspace.replacement, '_');
  EXPECT_EQ(transform.config.metaspace.prepend_scheme,
            IREE_TOKENIZER_PREPEND_NEVER);
  EXPECT_EQ(transform.config.metaspace.flags,
            IREE_TOKENIZER_METASPACE_FLAG_SPLIT);

  // Verify behavior: splits on underscore, no prepend on first word.
  auto segments = Encode(&transform, "hello world");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");
  EXPECT_EQ(segments[1], "_world");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseMetaspacePrependFirst) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Metaspace\", \"prepend_scheme\": \"first\"}"),
      allocator_, &transform));

  EXPECT_EQ(transform.config.metaspace.prepend_scheme,
            IREE_TOKENIZER_PREPEND_FIRST);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Sequence parsing tests
//===----------------------------------------------------------------------===//

TEST_F(TransformJsonTest, ParseEmptySequence) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Sequence\", \"pretokenizers\": []}"), allocator_,
      &transform));

  // Empty sequence becomes NONE transform.
  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_NONE);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseSequenceWithSingleChild) {
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Sequence\", \"pretokenizers\": "
              "[{\"type\": \"BertPreTokenizer\"}]}"),
      allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE);
  EXPECT_EQ(transform.config.sequence.count, 1u);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ParseSequenceWithMultipleChildren) {
  // When ByteLevel appears inside an explicit Sequence, use_regex=false is the
  // realistic case - users control the sequence structure manually. If they
  // want regex splitting, they'd include Split explicitly.
  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Sequence\", \"pretokenizers\": ["
              "{\"type\": \"ByteLevel\", \"add_prefix_space\": true, "
              "\"use_regex\": false},"
              "{\"type\": \"Whitespace\"}"
              "]}"),
      allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE);
  EXPECT_EQ(transform.config.sequence.count, 2u);

  // Verify child types.
  EXPECT_EQ(transform.config.sequence.children[0].type,
            IREE_TOKENIZER_TEXT_TRANSFORM_BYTE_LEVEL);
  EXPECT_EQ(transform.config.sequence.children[0].config.byte_level.flags,
            IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE);
  EXPECT_EQ(transform.config.sequence.children[1].type,
            IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Full tokenizer.json parsing tests
//===----------------------------------------------------------------------===//

TEST_F(TransformJsonTest, ParseFullTokenizerJson) {
  const char* json = R"({
    "version": "1.0",
    "pre_tokenizer": {
      "type": "Metaspace",
      "replacement": "\u2581",
      "prepend_scheme": "always"
    },
    "model": {
      "type": "BPE"
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_METASPACE);
  EXPECT_EQ(transform.config.metaspace.replacement, 0x2581u);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

//===----------------------------------------------------------------------===//
// Error handling tests
//===----------------------------------------------------------------------===//

TEST_F(TransformJsonTest, UnknownTypeReturnsUnimplemented) {
  iree_tokenizer_text_transform_t transform;
  iree_status_t status = iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"UnknownType\"}"), allocator_, &transform);

  EXPECT_THAT(Status(std::move(status)), StatusIs(StatusCode::kUnimplemented));
}

TEST_F(TransformJsonTest, MissingTypeReturnsError) {
  iree_tokenizer_text_transform_t transform;
  iree_status_t status = iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"foo\": \"bar\"}"), allocator_, &transform);

  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_free(status);
}

TEST_F(TransformJsonTest, InvalidPrependSchemeReturnsError) {
  iree_tokenizer_text_transform_t transform;
  iree_status_t status = iree_tokenizer_text_transform_parse_pretokenizer(
      IREE_SV("{\"type\": \"Metaspace\", \"prepend_scheme\": \"invalid\"}"),
      allocator_, &transform);

  EXPECT_THAT(Status(std::move(status)),
              StatusIs(StatusCode::kInvalidArgument));
}

//===----------------------------------------------------------------------===//
// Normalizer Integration Tests
//===----------------------------------------------------------------------===//

TEST_F(TransformJsonTest, ParseNormalizerFromJson) {
  // JSON with both pre_tokenizer and normalizer.
  const char* json = R"({
    "normalizer": {
      "type": "Lowercase"
    },
    "pre_tokenizer": {
      "type": "Whitespace"
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_WHITESPACE);
  EXPECT_EQ(transform.normalizer.type, IREE_TOKENIZER_NORMALIZER_LOWERCASE);

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, NormalizerAppliedDuringEncode) {
  // Whitespace transform with lowercase normalizer.
  const char* json = R"({
    "normalizer": {
      "type": "Lowercase"
    },
    "pre_tokenizer": {
      "type": "Whitespace"
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  // Verify normalizer is applied per-segment.
  auto segments = Encode(&transform, "HELLO WORLD");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");  // Normalized to lowercase.
  EXPECT_EQ(segments[1], "world");  // Normalized to lowercase.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, NormalizerNoneHasZeroOverhead) {
  // When normalizer is null, should have no effect.
  const char* json = R"({
    "normalizer": null,
    "pre_tokenizer": {
      "type": "Whitespace"
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.normalizer.type, IREE_TOKENIZER_NORMALIZER_NONE);

  // Text should pass through unchanged.
  auto segments = Encode(&transform, "HELLO WORLD");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "HELLO");
  EXPECT_EQ(segments[1], "WORLD");

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, BertNormalizerWithBertTransform) {
  // Full BERT configuration: normalizer + pre_tokenizer.
  const char* json = R"({
    "normalizer": {
      "type": "BertNormalizer",
      "lowercase": true,
      "strip_accents": true
    },
    "pre_tokenizer": {
      "type": "BertPreTokenizer"
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_BERT);
  EXPECT_EQ(transform.normalizer.type, IREE_TOKENIZER_NORMALIZER_BERT);

  // Verify BERT normalizer: lowercase + strip accents.
  auto segments = Encode(&transform, "Café");
  ASSERT_GE(segments.size(), 1u);
  EXPECT_EQ(segments[0], "cafe");  // Lowercased and accents stripped.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, SequenceNormalizerWithTransform) {
  // Sequence normalizer (strip accents then lowercase).
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "StripAccents"},
        {"type": "Lowercase"}
      ]
    },
    "pre_tokenizer": {
      "type": "Whitespace"
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.normalizer.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);

  // Verify sequence normalizer is applied.
  auto segments = Encode(&transform, "CAFÉ RÉSUMÉ");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "cafe");    // Strip accents + lowercase.
  EXPECT_EQ(segments[1], "resume");  // Strip accents + lowercase.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ClipStyleNormalizerWithSplit) {
  // CLIP-style configuration:
  // - Sequence normalizer: [NFC, Replace(\s+ -> " "), Lowercase]
  // - Split pre-tokenizer with regex pattern and invert=true
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "NFC"},
        {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
        {"type": "Lowercase"}
      ]
    },
    "pre_tokenizer": {
      "type": "Split",
      "pattern": {"Regex": "[a-z]+"},
      "behavior": "Isolated",
      "invert": true
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_SPLIT);
  EXPECT_EQ(transform.normalizer.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);

  // Input: "HELLO   WORLD" (uppercase with extra spaces)
  // After normalizer:
  //   1. NFC: no change
  //   2. Replace \s+ -> " ": "HELLO WORLD"
  //   3. Lowercase: "hello world"
  // After Split (invert=true, matches [a-z]+):
  //   Output: ["hello", "world"] (whitespace not matched, discarded)
  auto segments = Encode(&transform, "HELLO   WORLD");
  ASSERT_EQ(segments.size(), 2);
  EXPECT_EQ(segments[0], "hello");  // Lowercased.
  EXPECT_EQ(segments[1], "world");  // Lowercased.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ClipStyleSequenceWithSplitAndByteLevel) {
  // Full CLIP-style configuration with Sequence pre-tokenizer:
  // - Sequence normalizer: [NFC, Replace(\s+ -> " "), Lowercase]
  // - Sequence pre-tokenizer: [Split, ByteLevel]
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "NFC"},
        {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
        {"type": "Lowercase"}
      ]
    },
    "pre_tokenizer": {
      "type": "Sequence",
      "pretokenizers": [
        {
          "type": "Split",
          "pattern": {"Regex": "[a-z]+"},
          "behavior": "Isolated",
          "invert": true
        },
        {
          "type": "ByteLevel",
          "add_prefix_space": false
        }
      ]
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  EXPECT_EQ(transform.type, IREE_TOKENIZER_TEXT_TRANSFORM_SEQUENCE);
  EXPECT_EQ(transform.normalizer.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);

  // Input: "HELLO" (uppercase)
  // Flow:
  //   1. Normalizer applied by Split: "hello"
  //   2. Split emits: ["hello"]
  //   3. ByteLevel maps bytes: ["hello"] (ASCII maps directly)
  // Output should be lowercase.
  auto segments = Encode(&transform, "HELLO");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "hello");  // Lowercased.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

TEST_F(TransformJsonTest, ClipStyleWithUnicodeLetterPattern) {
  // Test with a pattern that matches both upper and lower case (like CLIP's
  // [\p{L}]+) to ensure normalization is applied.
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "NFC"},
        {"type": "Replace", "pattern": {"Regex": "\\s+"}, "content": " "},
        {"type": "Lowercase"}
      ]
    },
    "pre_tokenizer": {
      "type": "Sequence",
      "pretokenizers": [
        {
          "type": "Split",
          "pattern": {"Regex": "[A-Za-z]+"},
          "behavior": "Isolated",
          "invert": true
        },
        {
          "type": "ByteLevel",
          "add_prefix_space": false
        }
      ]
    }
  })";

  iree_tokenizer_text_transform_t transform;
  IREE_ASSERT_OK(iree_tokenizer_text_transform_parse_json(
      IREE_SV(json), allocator_, &transform));

  // Input: "HELLO" (uppercase)
  // Pattern [A-Za-z]+ matches both cases.
  // If normalization works: "hello" (lowercased before split)
  // If normalization fails: "HELLO" (uppercase)
  auto segments = Encode(&transform, "HELLO");
  ASSERT_EQ(segments.size(), 1);
  EXPECT_EQ(segments[0], "hello");  // Must be lowercased.

  iree_tokenizer_text_transform_deinitialize(&transform);
}

}  // namespace
