// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/normalizer_json.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class NormalizerJsonTest : public ::testing::Test {
 protected:
  iree_allocator_t allocator_ = iree_allocator_system();
};

//===----------------------------------------------------------------------===//
// Null / Passthrough
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, NullNormalizer) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV("null"), allocator_, &normalizer));
  EXPECT_EQ(normalizer, nullptr);
}

//===----------------------------------------------------------------------===//
// Lowercase
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, LowercaseNormalizer) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Lowercase"})"), allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// NFC
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, NFCNormalizer) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"NFC"})"), allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// Strip
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, StripNormalizerBoth) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Strip","strip_left":true,"strip_right":true})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, StripNormalizerLeftOnly) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Strip","strip_left":true,"strip_right":false})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, StripNormalizerMissingFieldsError) {
  // strip_left and strip_right are required (no serde default in HF).
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Strip"})"), allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(NormalizerJsonTest, StripNormalizerBothFalse) {
  // Explicit false values are valid.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Strip","strip_left":false,"strip_right":false})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// Prepend
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, PrependNormalizerAscii) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Prepend","prepend":"X"})"), allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, PrependNormalizerUnicode) {
  // U+2581 "▁" used by SentencePiece.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Prepend","prepend":"\u2581"})"), allocator_,
      &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// StripAccents
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, StripAccentsNormalizer) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"StripAccents"})"), allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// BertNormalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, BertNormalizerMissingFieldsError) {
  // clean_text, handle_chinese_chars, lowercase are required.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"BertNormalizer"})"), allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
}

TEST_F(NormalizerJsonTest, BertNormalizerAllTrue) {
  // All fields explicitly set to true.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"BertNormalizer","clean_text":true,)"
              R"("handle_chinese_chars":true,"strip_accents":null,)"
              R"("lowercase":true})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, BertNormalizerAllFalse) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"BertNormalizer","clean_text":false,)"
              R"("handle_chinese_chars":false,"strip_accents":false,)"
              R"("lowercase":false})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, BertNormalizerStripAccentsNull) {
  // strip_accents=null means same as lowercase (true here).
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"BertNormalizer","clean_text":true,)"
              R"("handle_chinese_chars":true,"strip_accents":null,)"
              R"("lowercase":true})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// Sequence
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, SequenceNormalizerMultipleTypes) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Sequence","normalizers":[)"
              R"({"type":"Lowercase"},)"
              R"({"type":"Strip","strip_left":true,"strip_right":true})"
              R"(]})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, SequenceNormalizerEmpty) {
  // Empty sequence compacts to NULL (passthrough).
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Sequence","normalizers":[]})"), allocator_,
      &normalizer));
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, SequenceNormalizerSingleChild) {
  // Single-child sequence unwraps to the child directly.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Sequence","normalizers":[{"type":"Lowercase"}]})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, UnsupportedNormalizerError) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"SomeUnknownNormalizer"})"), allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, MissingTypeField) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_normalizer(
          IREE_SV(R"({"not_type":"Lowercase"})"), allocator_, &normalizer));
  EXPECT_EQ(normalizer, nullptr);
}

//===----------------------------------------------------------------------===//
// Precompiled Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, PrecompiledNormalizerEmptyTrie) {
  // Base64 of {0, 0, 0, 0} (trie_size=0) = "AAAAAA=="
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Precompiled","precompiled_charsmap":"AAAAAA=="})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, PrecompiledNormalizerMissingCharsmap) {
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Precompiled"})"), allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, PrecompiledNormalizerInvalidBase64) {
  // "!!!" is not valid base64.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Precompiled","precompiled_charsmap":"!!!"})"),
      allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, PrecompiledNormalizerTruncatedData) {
  // Base64 of {1, 2} = "AQI=" - only 2 bytes, need at least 4 for trie_size.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Precompiled","precompiled_charsmap":"AQI="})"),
      allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, PrecompiledNormalizerEmptyBase64) {
  // Empty base64 string is treated as passthrough (returns NULL normalizer).
  // This matches HuggingFace behavior where empty charsmap means no
  // normalization.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Precompiled","precompiled_charsmap":""})"),
      allocator_, &normalizer));
  EXPECT_EQ(normalizer, nullptr);  // NULL = passthrough/identity.
}

TEST_F(NormalizerJsonTest, PrecompiledNormalizerInvalidTrieSize) {
  // Base64 of {5, 0, 0, 0, 0, 0, 0, 0, 0} - trie_size=5 not multiple of 4.
  // This is "BQAAAAAAAAAAA==" in base64.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(
          R"({"type":"Precompiled","precompiled_charsmap":"BQAAAAAAAAAAA=="})"),
      allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

//===----------------------------------------------------------------------===//
// Replace Normalizer
//===----------------------------------------------------------------------===//

TEST_F(NormalizerJsonTest, ReplaceNormalizerStringPattern) {
  // Basic string replacement: space -> underscore.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"String":" "},"content":"_"})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerUnicodePattern) {
  // Unicode pattern: U+2581 "▁" -> space.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(
          R"({"type":"Replace","pattern":{"String":"\u2581"},"content":" "})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerEmptyContent) {
  // Empty content means deletion.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"String":"X"},"content":""})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerMultiBytePattern) {
  // Multi-byte pattern: "XY" -> "Z".
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"String":"XY"},"content":"Z"})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerEmptyStringPattern) {
  // Empty String pattern is invalid - must have at least one character.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"String":""},"content":"X"})"),
      allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerRegexPattern) {
  // Regex patterns normalize whitespace (CLIP tokenizer pattern).
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"Regex":"\\s+"},"content":" "})"),
      allocator_, &normalizer));
  EXPECT_NE(normalizer, nullptr);
  iree_tokenizer_normalizer_free(normalizer);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerMissingPattern) {
  // Missing pattern field.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","content":"X"})"), allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerMissingContent) {
  // Missing content field.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"String":"X"}})"), allocator_,
      &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerInvalidPatternObject) {
  // Pattern object with neither String nor Regex.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{},"content":"X"})"), allocator_,
      &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerUnknownField) {
  // Unknown field should error (strict validation).
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"String":"X"},"content":"Y",)"
              R"("unknown_field":true})"),
      allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

TEST_F(NormalizerJsonTest, ReplaceNormalizerUnknownPatternType) {
  // Unknown pattern type in pattern object.
  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      IREE_SV(R"({"type":"Replace","pattern":{"Unknown":"X"},"content":"Y"})"),
      allocator_, &normalizer);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
  EXPECT_EQ(normalizer, nullptr);
}

}  // namespace
