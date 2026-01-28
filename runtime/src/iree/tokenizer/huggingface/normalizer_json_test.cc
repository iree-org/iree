// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/normalizer_json.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class NormalizerJsonTest : public ::testing::Test {
 protected:
  std::string Apply(const iree_tokenizer_normalizer_t* normalizer,
                    const char* input) {
    char buffer[1024];
    iree_host_size_t length = 0;
    iree_status_t status = iree_tokenizer_normalizer_apply(
        normalizer, IREE_SV(input), buffer, sizeof(buffer), &length);
    IREE_EXPECT_OK(status);
    return std::string(buffer, length);
  }
};

//===----------------------------------------------------------------------===//
// BertNormalizer JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, BertNormalizerDefault) {
  // Standard BERT normalizer with all defaults (all true).
  const char* json = R"({
    "normalizer": {
      "type": "BertNormalizer",
      "clean_text": true,
      "handle_chinese_chars": true,
      "strip_accents": null,
      "lowercase": true
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_BERT);
  EXPECT_EQ(Apply(&norm, "Hello WORLD"), "hello world");
  EXPECT_EQ(Apply(&norm, "Café"), "cafe");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, BertNormalizerNoLowercase) {
  const char* json = R"({
    "normalizer": {
      "type": "BertNormalizer",
      "lowercase": false,
      "strip_accents": false
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(Apply(&norm, "Hello WORLD"), "Hello WORLD");
  EXPECT_EQ(Apply(&norm, "Café"), "Café");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Lowercase JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, Lowercase) {
  const char* json = R"({
    "normalizer": {
      "type": "Lowercase"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_LOWERCASE);
  EXPECT_EQ(Apply(&norm, "Hello WORLD"), "hello world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// StripAccents JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, StripAccents) {
  const char* json = R"({
    "normalizer": {
      "type": "StripAccents"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS);
  EXPECT_EQ(Apply(&norm, "café"), "cafe");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// NFD JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, NFD) {
  const char* json = R"({
    "normalizer": {
      "type": "NFD"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  // NFD maps to StripAccents for our purposes.
  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP_ACCENTS);

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Prepend JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, Prepend) {
  const char* json = R"({
    "normalizer": {
      "type": "Prepend",
      "prepend": "\u2581"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_PREPEND);
  EXPECT_EQ(Apply(&norm, "Hello"), "\xE2\x96\x81Hello");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, PrependSpace) {
  const char* json = R"({
    "normalizer": {
      "type": "Prepend",
      "prepend": " "
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(Apply(&norm, "Hello"), " Hello");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Strip JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, StripDefaults) {
  // Default: both strip_left and strip_right are true.
  const char* json = R"({
    "normalizer": {
      "type": "Strip"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP);
  EXPECT_EQ(Apply(&norm, "  hello world  "), "hello world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, StripLeftOnly) {
  const char* json = R"({
    "normalizer": {
      "type": "Strip",
      "strip_left": true,
      "strip_right": false
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP);
  EXPECT_EQ(Apply(&norm, "  hello world  "), "hello world  ");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, StripRightOnly) {
  const char* json = R"({
    "normalizer": {
      "type": "Strip",
      "strip_left": false,
      "strip_right": true
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP);
  EXPECT_EQ(Apply(&norm, "  hello world  "), "  hello world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, StripBothFalse) {
  const char* json = R"({
    "normalizer": {
      "type": "Strip",
      "strip_left": false,
      "strip_right": false
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_STRIP);
  // Both false = passthrough.
  EXPECT_EQ(Apply(&norm, "  hello world  "), "  hello world  ");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Replace JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, ReplaceWithStringPattern) {
  const char* json = R"({
    "normalizer": {
      "type": "Replace",
      "pattern": {"String": " "},
      "content": "\u2581"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_REPLACE);
  EXPECT_EQ(Apply(&norm, "Hello World"), "Hello\xE2\x96\x81World");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, ReplaceWithRegexPattern) {
  const char* json = R"({
    "normalizer": {
      "type": "Replace",
      "pattern": {"Regex": "\\s+"},
      "content": " "
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX);
  // Collapse whitespace: "a   b\t\tc" -> "a b c".
  EXPECT_EQ(Apply(&norm, "a   b\t\tc"), "a b c");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, ReplaceWithRegexPatternDigits) {
  const char* json = R"({
    "normalizer": {
      "type": "Replace",
      "pattern": {"Regex": "[0-9]+"},
      "content": "#"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_REPLACE_REGEX);
  EXPECT_EQ(Apply(&norm, "abc123def456"), "abc#def#");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, ReplaceWithRegexPatternDelete) {
  // Empty replacement (delete matches).
  const char* json = R"({
    "normalizer": {
      "type": "Replace",
      "pattern": {"Regex": "\\s+"},
      "content": ""
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(Apply(&norm, "Hello World"), "HelloWorld");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Sequence JSON Parsing
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, Sequence) {
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "StripAccents"},
        {"type": "Lowercase"}
      ]
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);
  EXPECT_EQ(Apply(&norm, "CAFÉ"), "cafe");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, SequenceWithPrepend) {
  // Sequence with Prepend and Lowercase.
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "Prepend", "prepend": "\u2581"},
        {"type": "Lowercase"}
      ]
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);
  // "Hello World" -> "▁Hello World" -> "▁hello world"
  EXPECT_EQ(Apply(&norm, "Hello World"), "\xE2\x96\x81hello world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, SequenceTinyLlamaStyle) {
  // TinyLlama-style: Prepend ▁, then Replace space with ▁.
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "Prepend", "prepend": "\u2581"},
        {"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}
      ]
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);
  // "Hello World" -> "▁Hello World" -> "▁Hello▁World"
  EXPECT_EQ(Apply(&norm, "Hello World"), "\xE2\x96\x81Hello\xE2\x96\x81World");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, SequenceWithStrip) {
  // DeBERTa v3 style: Strip -> Precompiled -> Replace.
  // We test just Strip in a Sequence since we don't have Precompiled JSON.
  const char* json = R"({
    "normalizer": {
      "type": "Sequence",
      "normalizers": [
        {"type": "Strip", "strip_left": true, "strip_right": true},
        {"type": "Lowercase"}
      ]
    }
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_SEQUENCE);
  // "  HELLO WORLD  " -> "HELLO WORLD" -> "hello world"
  EXPECT_EQ(Apply(&norm, "  HELLO WORLD  "), "hello world");

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Null/Missing Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, NullNormalizer) {
  const char* json = R"({
    "normalizer": null
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_NONE);

  iree_tokenizer_normalizer_deinitialize(&norm);
}

TEST_F(NormalizerJsonTest, MissingNormalizer) {
  const char* json = R"({
    "model": {}
  })";

  iree_tokenizer_normalizer_t norm;
  IREE_ASSERT_OK(iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm));

  EXPECT_EQ(norm.type, IREE_TOKENIZER_NORMALIZER_NONE);

  iree_tokenizer_normalizer_deinitialize(&norm);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, UnsupportedType) {
  const char* json = R"({
    "normalizer": {
      "type": "SomeUnknownNormalizer"
    }
  })";

  iree_tokenizer_normalizer_t norm;
  iree_status_t status = iree_tokenizer_normalizer_parse_json(
      IREE_SV(json), iree_allocator_system(), &norm);

  EXPECT_TRUE(iree_status_is_unimplemented(status));
  iree_status_free(status);
}

}  // namespace
