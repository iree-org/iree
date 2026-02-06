// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/segmenter_json.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SegmenterJsonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    flags_ = IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_NONE;
  }
  iree_allocator_t allocator_ = iree_allocator_system();
  iree_tokenizer_huggingface_pre_tokenizer_flags_t flags_;
};

//===----------------------------------------------------------------------===//
// Null / Passthrough
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, NullPreTokenizer) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV("null"), allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

//===----------------------------------------------------------------------===//
// BertPreTokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, BertPreTokenizer) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"BertPreTokenizer"})"), allocator_, &segmenter,
      &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT));
  iree_tokenizer_segmenter_free(segmenter);
}

//===----------------------------------------------------------------------===//
// Punctuation Pre-Tokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, PunctuationDefault) {
  // Default behavior is Isolated.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Punctuation"})"), allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, PunctuationExplicitIsolated) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Punctuation","behavior":"Isolated"})"), allocator_,
      &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, PunctuationRemoved) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Punctuation","behavior":"Removed"})"), allocator_,
      &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, PunctuationMergedWithPrevious) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Punctuation","behavior":"MergedWithPrevious"})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, PunctuationMergedWithNext) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Punctuation","behavior":"MergedWithNext"})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, PunctuationContiguous) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Punctuation","behavior":"Contiguous"})"), allocator_,
      &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, PunctuationUnknownBehavior) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Punctuation","behavior":"Unknown"})"), allocator_,
          &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

//===----------------------------------------------------------------------===//
// Metaspace Pre-Tokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, MetaspaceMissingReplacementError) {
  // replacement is required (char, no serde default in HF).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Metaspace"})"), allocator_, &segmenter, &flags_));
}

TEST_F(SegmenterJsonTest, MetaspaceWithReplacement) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Metaspace","replacement":"\u2581"})"), allocator_,
      &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, MetaspaceSplitFalse) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Metaspace","replacement":"\u2581","split":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, MetaspaceSplitTrue) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Metaspace","replacement":"\u2581","split":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, MetaspaceFullConfig) {
  // All fields (Mistral-style config). prepend_scheme="first" sets PREPEND,
  // and all Metaspace configs set REPLACE.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Metaspace","replacement":"\u2581",)"
              R"("prepend_scheme":"first","split":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_PREPEND));
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, MetaspacePrependNever) {
  // prepend_scheme="never" does not set the prepend flag, but REPLACE is
  // always set for any Metaspace pre_tokenizer.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Metaspace","replacement":"\u2581",)"
              R"("prepend_scheme":"never","split":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_FALSE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_PREPEND));
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE));
  iree_tokenizer_segmenter_free(segmenter);
}

//===----------------------------------------------------------------------===//
// ByteLevel Pre-Tokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, ByteLevelMissingFieldsError) {
  // add_prefix_space and trim_offsets are required (no serde default in HF).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"ByteLevel"})"), allocator_, &segmenter, &flags_));
}

TEST_F(SegmenterJsonTest, ByteLevelDefault) {
  // Default: use_regex=true, creates Split segmenter with GPT-2 pattern.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL));
  EXPECT_FALSE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, ByteLevelUseRegexFalse) {
  // use_regex=false means passthrough (no segmenter).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, ByteLevelAddPrefixSpace) {
  // add_prefix_space=true sets the ADD_PREFIX_SPACE flag.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"ByteLevel","add_prefix_space":true,)"
              R"("trim_offsets":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL));
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, ByteLevelFullConfig) {
  // All fields (GPT-2 style). add_prefix_space=false does not set that flag.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL));
  EXPECT_FALSE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_segmenter_free(segmenter);
}

//===----------------------------------------------------------------------===//
// Split Pre-Tokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, SplitBasicRegex) {
  // Simple whitespace-splitting regex with Removed behavior.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"Removed","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitBehaviorIsolated) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"Isolated","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitBehaviorMergedWithPrevious) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"MergedWithPrevious","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitBehaviorMergedWithNext) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"MergedWithNext","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitBehaviorContiguous) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"Contiguous","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitInvertTrue) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"Isolated","invert":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitUnicodeRegex) {
  // Unicode property escape requiring JSON unescaping (\\p -> \p).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\p{L}+"},)"
              R"("behavior":"Isolated","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitLlama3Pattern) {
  // Llama-3 style pattern with multiple alternations and Unicode properties.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(
          R"({"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
          R"([^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+)"
          R"([\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},)"
          R"("behavior":"Isolated","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitMissingInvertError) {
  // invert is required (no serde default in HF).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
                  R"("behavior":"Removed"})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SplitMissingPattern) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Split","behavior":"Isolated","invert":false})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SplitMissingBehavior) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(
              R"({"type":"Split","pattern":{"Regex":"\\s+"},"invert":false})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SplitUnknownBehavior) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
                  R"("behavior":"Unknown","invert":false})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SplitInvalidRegex) {
  // Unmatched parenthesis is an invalid regex.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Split","pattern":{"Regex":"(abc"},)"
                  R"("behavior":"Isolated","invert":false})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SplitStringPatternBasic) {
  // Literal string pattern: split on space delimiter.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":" "},)"
              R"("behavior":"Isolated","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternRemoved) {
  // String pattern with Removed behavior.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":","},)"
              R"("behavior":"Removed","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternMergedWithPrevious) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":":"},)"
              R"("behavior":"MergedWithPrevious","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternMergedWithNext) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":"-"},)"
              R"("behavior":"MergedWithNext","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternContiguous) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":"\n"},)"
              R"("behavior":"Contiguous","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternInvert) {
  // String pattern with invert=true.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":" "},)"
              R"("behavior":"Removed","invert":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternMultiByte) {
  // Multi-byte UTF-8 pattern (U+2581 LOWER ONE EIGHTH BLOCK, 3 bytes).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":"\u2581"},)"
              R"("behavior":"Isolated","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternMultiChar) {
  // Multi-character pattern.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":"<|endoftext|>"},)"
              R"("behavior":"Isolated","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternEscapedChars) {
  // JSON escaped characters (tab and backslash).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Split","pattern":{"String":"\t"},)"
              R"("behavior":"Removed","invert":false})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SplitStringPatternEmpty) {
  // Empty string pattern should be rejected.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Split","pattern":{"String":""},)"
                  R"("behavior":"Isolated","invert":false})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SplitEmptyPatternObject) {
  // Pattern object with neither Regex nor String.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_tokenizer_huggingface_parse_segmenter(
                            IREE_SV(R"({"type":"Split","pattern":{},)"
                                    R"("behavior":"Isolated","invert":false})"),
                            allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

//===----------------------------------------------------------------------===//
// Sequence Pre-Tokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, SequenceTwoChildren) {
  // Two Metaspace children -> produces a Sequence segmenter.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true},)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SequenceMixedChildren) {
  // Split + Metaspace children (real-world Llama-3 pattern).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"Split","pattern":{"Regex":"\\s+"},)"
              R"("behavior":"Isolated","invert":false},)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SequenceSingleChildUnwrapped) {
  // Single non-NULL child -> returned directly without Sequence wrapper.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SequenceEmptyArray) {
  // Empty pretokenizers array -> passthrough (NULL).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[]})"), allocator_,
      &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SequenceAllChildrenNull) {
  // All children parse to NULL (ByteLevel with use_regex=false) -> NULL result.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false},)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SequenceNullChildrenSkipped) {
  // Mix of NULL and non-NULL children. Only the non-NULL child survives,
  // so the single-child optimization applies (returned directly).
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false},)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true},)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SequenceNullChildrenWithTwoNonNull) {
  // Two non-NULL children among NULL ones -> creates Sequence.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false},)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true},)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":false},)"
              R"({"type":"Metaspace","replacement":"\u2581","split":false}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SequenceMissingPretokenizers) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Sequence"})"), allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SequenceChildParseError) {
  // Child with unsupported type should propagate the error and clean up
  // any already-parsed children.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
                  R"({"type":"Metaspace","replacement":"\u2581","split":true},)"
                  R"({"type":"Unsupported"}]})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SequenceTooManyChildren) {
  // Exceed IREE_TOKENIZER_SEGMENTER_SEQUENCE_MAX_DEPTH (8) children.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_segmenter(
          IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"},)"
                  R"({"type":"Metaspace","replacement":"\u2581"}]})"),
          allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, SequenceWithSplitChild) {
  // Split + ByteLevel (common pattern in modern tokenizers like cl100k_base).
  // ByteLevel flag must propagate through Sequence.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"Split","pattern":{"Regex":"\\p{L}+"},)"
              R"("behavior":"Isolated","invert":false},)"
              R"({"type":"ByteLevel","add_prefix_space":false,)"
              R"("trim_offsets":false,"use_regex":true}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, SequenceWithoutByteLevelNoFlag) {
  // Sequence without ByteLevel does not set the BYTE_LEVEL flag.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true},)"
              R"({"type":"Metaspace","replacement":"\u2581","split":true}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_FALSE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_BYTE_LEVEL));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, MetaspacePrependDefault) {
  // Default prepend_scheme is "always", which sets the METASPACE_PREPEND flag.
  // METASPACE_REPLACE is always set for any Metaspace pre_tokenizer.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Metaspace","replacement":"\u2581","split":true})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_PREPEND));
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_METASPACE_REPLACE));
  iree_tokenizer_segmenter_free(segmenter);
}

//===----------------------------------------------------------------------===//
// Whitespace Pre-Tokenizer
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, WhitespaceBasic) {
  // Whitespace pre_tokenizer: split on whitespace, discard whitespace.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Whitespace"})"), allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, WhitespaceSplitBasic) {
  // WhitespaceSplit is equivalent to Whitespace.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"WhitespaceSplit"})"), allocator_, &segmenter,
      &flags_));
  EXPECT_NE(segmenter, nullptr);
  EXPECT_TRUE(iree_any_bit_set(
      flags_, IREE_TOKENIZER_HUGGINGFACE_PRE_TOKENIZER_FLAG_WORD_LEVEL_SPLIT));
  iree_tokenizer_segmenter_free(segmenter);
}

TEST_F(SegmenterJsonTest, WhitespaceInSequence) {
  // Whitespace can be used as a child in Sequence.
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_segmenter(
      IREE_SV(R"({"type":"Sequence","pretokenizers":[)"
              R"({"type":"Whitespace"},)"
              R"({"type":"Punctuation"}]})"),
      allocator_, &segmenter, &flags_));
  EXPECT_NE(segmenter, nullptr);
  iree_tokenizer_segmenter_free(segmenter);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(SegmenterJsonTest, UnsupportedPreTokenizerType) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_tokenizer_huggingface_parse_segmenter(
                            IREE_SV(R"({"type":"SomeUnimplementedType"})"),
                            allocator_, &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

TEST_F(SegmenterJsonTest, MissingTypeField) {
  iree_tokenizer_segmenter_t* segmenter = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        iree_tokenizer_huggingface_parse_segmenter(
                            IREE_SV(R"({"not_type":"ByteLevel"})"), allocator_,
                            &segmenter, &flags_));
  EXPECT_EQ(segmenter, nullptr);
}

}  // namespace
