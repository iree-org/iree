// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/postprocessor_json.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace tokenizer {
namespace {

//===----------------------------------------------------------------------===//
// BertProcessing
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, BertProcessing) {
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      IREE_SV(
          R"({"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]})"),
      &pp));

  // Single: [CLS] $A [SEP]
  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 1);
  EXPECT_EQ(pp.single.token_ids[0], 101);  // [CLS]
  EXPECT_EQ(pp.single.token_ids[1], 102);  // [SEP]
  EXPECT_EQ(pp.single.type_ids[0], 0);
  EXPECT_EQ(pp.single.type_ids[1], 0);
  EXPECT_EQ(pp.single.sequence_a_type_id, 0);

  // Pair: [CLS] $A [SEP] $B [SEP]
  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));
  EXPECT_EQ(pp.pair.prefix_count, 1);
  EXPECT_EQ(pp.pair.infix_count, 1);
  EXPECT_EQ(pp.pair.suffix_count, 1);
  EXPECT_EQ(pp.pair.token_ids[0], 101);  // [CLS] prefix
  EXPECT_EQ(pp.pair.token_ids[1], 102);  // [SEP] infix
  EXPECT_EQ(pp.pair.token_ids[2], 102);  // [SEP] suffix
  EXPECT_EQ(pp.pair.type_ids[0], 0);     // [CLS] type_id
  EXPECT_EQ(pp.pair.type_ids[1], 0);     // infix [SEP] type_id
  EXPECT_EQ(pp.pair.type_ids[2], 1);     // suffix [SEP] type_id
  EXPECT_EQ(pp.pair.sequence_a_type_id, 0);
  EXPECT_EQ(pp.pair.sequence_b_type_id, 1);

  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, BertProcessingUnknownKey) {
  iree_tokenizer_postprocessor_t pp;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_postprocessor(
          IREE_SV(
              R"({"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101],"extra":true})"),
          &pp));
}

//===----------------------------------------------------------------------===//
// RobertaProcessing
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, RobertaProcessing) {
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      IREE_SV(
          R"({"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":false,"add_prefix_space":false})"),
      &pp));

  // Single: <s> $A </s>
  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.suffix_count, 1);
  EXPECT_EQ(pp.single.token_ids[0], 0);  // <s>
  EXPECT_EQ(pp.single.token_ids[1], 2);  // </s>

  // Pair: <s> $A </s></s> $B </s>
  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));
  EXPECT_EQ(pp.pair.prefix_count, 1);
  EXPECT_EQ(pp.pair.infix_count, 2);
  EXPECT_EQ(pp.pair.suffix_count, 1);
  EXPECT_EQ(pp.pair.token_ids[0], 0);  // <s> prefix
  EXPECT_EQ(pp.pair.token_ids[1], 2);  // </s> first infix
  EXPECT_EQ(pp.pair.token_ids[2], 2);  // </s> second infix
  EXPECT_EQ(pp.pair.token_ids[3], 2);  // </s> suffix
  EXPECT_EQ(pp.pair.sequence_a_type_id, 0);
  EXPECT_EQ(pp.pair.sequence_b_type_id, 0);

  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, RobertaProcessingTrimOffsets) {
  // trim_offsets=true is now implemented.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      IREE_SV(
          R"({"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],"trim_offsets":true,"add_prefix_space":false})"),
      &pp));
  EXPECT_TRUE(iree_any_bit_set(pp.flags,
                               IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, RobertaProcessingMissingFieldsError) {
  // Missing trim_offsets and add_prefix_space should fail (required, no serde
  // default in HF).
  iree_tokenizer_postprocessor_t pp;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_postprocessor(
          IREE_SV(
              R"({"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0]})"),
          &pp));
}

//===----------------------------------------------------------------------===//
// ByteLevel
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, ByteLevel) {
  // GPT-2 style: trim_offsets=false, add_prefix_space=false.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      IREE_SV(
          R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true})"),
      &pp));

  // ByteLevel adds no special tokens.
  EXPECT_EQ(pp.single.prefix_count, 0);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 0);
  EXPECT_FALSE(iree_tokenizer_postprocessor_supports_pair(&pp));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, ByteLevelTrimOffsets) {
  // trim_offsets=true is now implemented.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      IREE_SV(
          R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true})"),
      &pp));
  EXPECT_TRUE(iree_any_bit_set(pp.flags,
                               IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, ByteLevelAddPrefixSpace) {
  // add_prefix_space=true is now implemented.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      IREE_SV(
          R"({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":false})"),
      &pp));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_TRUE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// TemplateProcessing
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, TemplateProcessingBert) {
  // BERT-style template.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}},
          {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "pair": [
          {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}},
          {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
          {"Sequence": {"id": "B", "type_id": 1}},
          {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
        ],
        "special_tokens": {
          "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
          "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
        }
      })"),
                                                                &pp));

  // Single: [CLS] $A [SEP]
  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 1);
  EXPECT_EQ(pp.single.token_ids[0], 101);
  EXPECT_EQ(pp.single.token_ids[1], 102);
  EXPECT_EQ(pp.single.sequence_a_type_id, 0);

  // Pair: [CLS] $A [SEP] $B [SEP]
  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));
  EXPECT_EQ(pp.pair.prefix_count, 1);
  EXPECT_EQ(pp.pair.infix_count, 1);
  EXPECT_EQ(pp.pair.suffix_count, 1);
  EXPECT_EQ(pp.pair.token_ids[0], 101);  // [CLS]
  EXPECT_EQ(pp.pair.token_ids[1], 102);  // [SEP] infix
  EXPECT_EQ(pp.pair.token_ids[2], 102);  // [SEP] suffix
  EXPECT_EQ(pp.pair.type_ids[0], 0);
  EXPECT_EQ(pp.pair.type_ids[1], 0);
  EXPECT_EQ(pp.pair.type_ids[2], 1);
  EXPECT_EQ(pp.pair.sequence_a_type_id, 0);
  EXPECT_EQ(pp.pair.sequence_b_type_id, 1);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, TemplateProcessingLlama) {
  // LLaMA 2 style: <bos> $A (prefix only, no suffix).
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "<s>", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
          {"SpecialToken": {"id": "<s>", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}},
          {"SpecialToken": {"id": "<s>", "type_id": 1}},
          {"Sequence": {"id": "B", "type_id": 1}}
        ],
        "special_tokens": {
          "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        }
      })"),
                                                                &pp));

  // Single: <bos> $A
  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 0);
  EXPECT_EQ(pp.single.token_ids[0], 1);

  // Pair: <bos> $A <bos>:1 $B:1
  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));
  EXPECT_EQ(pp.pair.prefix_count, 1);
  EXPECT_EQ(pp.pair.infix_count, 1);
  EXPECT_EQ(pp.pair.suffix_count, 0);
  EXPECT_EQ(pp.pair.token_ids[0], 1);  // <bos> prefix
  EXPECT_EQ(pp.pair.token_ids[1], 1);  // <bos> infix
  EXPECT_EQ(pp.pair.type_ids[1], 1);   // infix type_id
  EXPECT_EQ(pp.pair.sequence_a_type_id, 0);
  EXPECT_EQ(pp.pair.sequence_b_type_id, 1);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, TemplateProcessingMultiTokenSpecial) {
  // A special token that expands to multiple IDs.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "PREFIX", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
          {"SpecialToken": {"id": "PREFIX", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}},
          {"Sequence": {"id": "B", "type_id": 1}}
        ],
        "special_tokens": {
          "PREFIX": {"id": "PREFIX", "ids": [50258, 50259, 50360], "tokens": ["<|sot|>", "<|en|>", "<|transcribe|>"]}
        }
      })"),
                                                                &pp));

  // Single: multi-token prefix expands to 3 tokens.
  EXPECT_EQ(pp.single.prefix_count, 3);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 0);
  EXPECT_EQ(pp.single.token_ids[0], 50258);
  EXPECT_EQ(pp.single.token_ids[1], 50259);
  EXPECT_EQ(pp.single.token_ids[2], 50360);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, TemplateProcessingMaxPiecesExceeded) {
  // A template that exceeds MAX_PIECES.
  iree_tokenizer_postprocessor_t pp;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "BIG", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
          {"SpecialToken": {"id": "BIG", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}},
          {"Sequence": {"id": "B", "type_id": 0}}
        ],
        "special_tokens": {
          "BIG": {"id": "BIG", "ids": [1,2,3,4,5,6,7,8], "tokens": ["a","b","c","d","e","f","g","h"]}
        }
      })"),
                                                     &pp));
}

TEST(PostprocessorJson, TemplateProcessingMissingSpecialToken) {
  // Reference a special token not in the map.
  // The lookup of "MISSING" in the empty special_tokens map should fail.
  iree_tokenizer_postprocessor_t pp;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "MISSING", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}}
        ],
        "pair": [
          {"Sequence": {"id": "A", "type_id": 0}},
          {"Sequence": {"id": "B", "type_id": 0}}
        ],
        "special_tokens": {}
      })"),
                                                     &pp));
}

//===----------------------------------------------------------------------===//
// Sequence
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, SequenceByteLevelPlusTemplate) {
  // LLaMA 3 style: ByteLevel + TemplateProcessing.
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "Sequence",
        "processors": [
          {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": true
          },
          {
            "type": "TemplateProcessing",
            "single": [
              {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}}
            ],
            "pair": [
              {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
              {"Sequence": {"id": "A", "type_id": 0}},
              {"Sequence": {"id": "B", "type_id": 0}}
            ],
            "special_tokens": {
              "<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [128000], "tokens": ["<|begin_of_text|>"]}
            }
          }
        ]
      })"),
                                                                &pp));

  // Template from the TemplateProcessing child.
  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.token_ids[0], 128000);
  EXPECT_EQ(pp.single.suffix_count, 0);

  // ByteLevel flags merged.
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJson, SequenceByteLevelOnly) {
  // A Sequence with only ByteLevel (no template).
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(IREE_SV(R"({
        "type": "Sequence",
        "processors": [
          {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false}
        ]
      })"),
                                                                &pp));

  EXPECT_EQ(pp.single.prefix_count, 0);
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Null / Missing
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, NullValue) {
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(
      iree_tokenizer_huggingface_parse_postprocessor(IREE_SV("null"), &pp));

  EXPECT_EQ(pp.single.prefix_count, 0);
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
}

TEST(PostprocessorJson, EmptyString) {
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_huggingface_parse_postprocessor(
      iree_string_view_empty(), &pp));

  EXPECT_EQ(pp.single.prefix_count, 0);
}

//===----------------------------------------------------------------------===//
// Unsupported
//===----------------------------------------------------------------------===//

TEST(PostprocessorJson, UnsupportedType) {
  iree_tokenizer_postprocessor_t pp;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_tokenizer_huggingface_parse_postprocessor(
                            IREE_SV(R"({"type":"FutureProcessing"})"), &pp));
}

}  // namespace
}  // namespace tokenizer
}  // namespace iree
