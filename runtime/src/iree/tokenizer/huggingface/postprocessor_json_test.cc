// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/huggingface/postprocessor_json.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tokenizer/postprocessor.h"

namespace {

//===----------------------------------------------------------------------===//
// Null Post-Processor Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorJsonTest, NullPostProcessor) {
  const char* json = R"({"post_processor": null})";
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_NONE);
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJsonTest, MissingPostProcessor) {
  const char* json = R"({"version": "1.0"})";
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_NONE);
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// TemplateProcessing Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorJsonTest, TemplateProcessingSingleOnly) {
  // BERT-style: [CLS] $A [SEP]
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  EXPECT_EQ(pp.config.template_.single_count, 3);
  EXPECT_EQ(pp.config.template_.pair_count, 0);

  // Verify single template pieces.
  auto* pieces = pp.config.template_.templates;
  EXPECT_EQ(pieces[0].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[0].token_id, 101);  // [CLS]
  EXPECT_EQ(pieces[1].type, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A);
  EXPECT_EQ(pieces[2].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[2].token_id, 102);  // [SEP]

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJsonTest, TemplateProcessingWithPair) {
  // Full BERT: [CLS] $A [SEP] $B [SEP]
  const char* json = R"({
    "post_processor": {
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
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  EXPECT_EQ(pp.config.template_.single_count, 3);
  EXPECT_EQ(pp.config.template_.pair_count, 5);

  // Verify pair template pieces (starts at index 3).
  auto* pieces = pp.config.template_.templates;
  EXPECT_EQ(pieces[3].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[3].token_id, 101);  // [CLS]
  EXPECT_EQ(pieces[4].type, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A);
  EXPECT_EQ(pieces[5].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[5].token_id, 102);  // [SEP]
  EXPECT_EQ(pieces[6].type, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B);
  EXPECT_EQ(pieces[6].type_id, 1);  // Type ID for B
  EXPECT_EQ(pieces[7].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[7].token_id, 102);  // [SEP]
  EXPECT_EQ(pieces[7].type_id, 1);     // Type ID for second segment

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJsonTest, LlamaStyleBosOnly) {
  // LLaMA 2 style: <bos> $A (no EOS)
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<s>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  EXPECT_EQ(pp.config.template_.single_count, 2);
  EXPECT_EQ(pp.config.template_.pair_count, 0);

  auto* pieces = pp.config.template_.templates;
  EXPECT_EQ(pieces[0].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[0].token_id, 1);  // <s>
  EXPECT_EQ(pieces[1].type, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// RobertaProcessing Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorJsonTest, RobertaProcessing) {
  const char* json = R"({
    "post_processor": {
      "type": "RobertaProcessing",
      "sep": ["</s>", 2],
      "cls": ["<s>", 0],
      "trim_offsets": true,
      "add_prefix_space": true
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_ROBERTA);
  EXPECT_EQ(pp.config.roberta.cls_id, 0);
  EXPECT_EQ(pp.config.roberta.sep_id, 2);
  EXPECT_TRUE(pp.config.roberta.flags &
              IREE_TOKENIZER_ROBERTA_FLAG_ADD_PREFIX_SPACE);
  EXPECT_TRUE(pp.config.roberta.flags &
              IREE_TOKENIZER_ROBERTA_FLAG_TRIM_OFFSETS);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// ByteLevel Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorJsonTest, ByteLevel) {
  const char* json = R"({
    "post_processor": {
      "type": "ByteLevel",
      "add_prefix_space": true,
      "trim_offsets": false
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

// Real-world config from BART, Llama 3.x, CLIP, GPT-2, Qwen, etc.
// use_regex is an encoding-only parameter (we support it in the pre-tokenizer).
// The postprocessor should accept it since we fully support it in encoding.
TEST(PostprocessorJsonTest, ByteLevelWithUseRegex) {
  const char* json = R"({
    "post_processor": {
      "type": "ByteLevel",
      "add_prefix_space": true,
      "trim_offsets": true,
      "use_regex": true
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

// use_regex=false should also be accepted.
TEST(PostprocessorJsonTest, ByteLevelWithUseRegexFalse) {
  const char* json = R"({
    "post_processor": {
      "type": "ByteLevel",
      "use_regex": false
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Sequence Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorJsonTest, SequenceLlama3Style) {
  // Llama 3 style: Sequence of ByteLevel + TemplateProcessing
  const char* json = R"({
    "post_processor": {
      "type": "Sequence",
      "processors": [
        {
          "type": "ByteLevel",
          "add_prefix_space": true,
          "trim_offsets": false
        },
        {
          "type": "TemplateProcessing",
          "single": [
            {"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
          ],
          "pair": null,
          "special_tokens": {
            "<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [128000], "tokens": ["<|begin_of_text|>"]}
          }
        }
      ]
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_SEQUENCE);
  EXPECT_EQ(pp.config.sequence.count, 2);

  // First child: ByteLevel
  EXPECT_EQ(pp.config.sequence.children[0].type,
            IREE_TOKENIZER_POSTPROCESSOR_BYTE_LEVEL);

  // Second child: TemplateProcessing
  EXPECT_EQ(pp.config.sequence.children[1].type,
            IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  EXPECT_EQ(pp.config.sequence.children[1].config.template_.single_count, 2);

  auto* pieces = pp.config.sequence.children[1].config.template_.templates;
  EXPECT_EQ(pieces[0].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[0].token_id, 128000);  // <|begin_of_text|>

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorJsonTest, SequenceEmpty) {
  const char* json = R"({
    "post_processor": {
      "type": "Sequence",
      "processors": []
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  // Empty sequence should become NONE.
  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_NONE);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Multi-Prefix Token Tests (Whisper-style)
//===----------------------------------------------------------------------===//

// Whisper uses multiple special tokens before $A:
//   <|startoftranscript|> <|notimestamps|> $A <|endoftext|>
// This tests that all prefix tokens are correctly parsed.
TEST(PostprocessorJsonTest, WhisperStyleMultiPrefixTokens) {
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|startoftranscript|>", "type_id": 0}},
        {"SpecialToken": {"id": "<|notimestamps|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "<|startoftranscript|>": {"id": "<|startoftranscript|>", "ids": [50258], "tokens": ["<|startoftranscript|>"]},
        "<|notimestamps|>": {"id": "<|notimestamps|>", "ids": [50364], "tokens": ["<|notimestamps|>"]},
        "<|endoftext|>": {"id": "<|endoftext|>", "ids": [50257], "tokens": ["<|endoftext|>"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  EXPECT_EQ(pp.config.template_.single_count, 4);

  auto* pieces = pp.config.template_.templates;
  EXPECT_EQ(pieces[0].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[0].token_id, 50258);  // <|startoftranscript|>
  EXPECT_EQ(pieces[1].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[1].token_id, 50364);  // <|notimestamps|>
  EXPECT_EQ(pieces[2].type, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A);
  EXPECT_EQ(pieces[3].type, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL);
  EXPECT_EQ(pieces[3].token_id, 50257);  // <|endoftext|>

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Apply Tests - Verify postprocessor token emission
//===----------------------------------------------------------------------===//

// Base fixture for postprocessor tests that need to capture callback output.
// Provides member vectors for collecting emitted tokens and encoded text
// segments, with static callbacks that cast user_data back to the fixture.
class PostprocessorTestBase : public ::testing::Test {
 protected:
  std::vector<int32_t> emitted_tokens;
  std::vector<std::string> encoded_texts;

  static iree_status_t EncodeTextCallback(void* user_data,
                                          iree_string_view_t text) {
    auto* self = static_cast<PostprocessorTestBase*>(user_data);
    self->encoded_texts.push_back(std::string(text.data, text.size));
    return iree_ok_status();
  }

  static iree_status_t EmitTokenCallback(void* user_data, int32_t token_id) {
    auto* self = static_cast<PostprocessorTestBase*>(user_data);
    self->emitted_tokens.push_back(token_id);
    return iree_ok_status();
  }
};

class PostprocessorApplyTest : public PostprocessorTestBase {};
class PostprocessorEmitTest : public PostprocessorTestBase {};

// Tests that apply_single correctly emits all template tokens.
TEST_F(PostprocessorApplyTest, BertStyleSingleTokens) {
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello world"), EncodeTextCallback, EmitTokenCallback,
      this));

  // Should emit: [CLS]=101, text, [SEP]=102
  EXPECT_EQ(emitted_tokens.size(), 2);
  EXPECT_EQ(emitted_tokens[0], 101);  // [CLS]
  EXPECT_EQ(emitted_tokens[1], 102);  // [SEP]
  EXPECT_EQ(encoded_texts.size(), 1);
  EXPECT_EQ(encoded_texts[0], "hello world");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

// Tests Whisper-style multi-prefix token emission.
// Expected output: <|startoftranscript|>=50258, <|notimestamps|>=50364, text,
// <|endoftext|>=50257
TEST_F(PostprocessorApplyTest, WhisperStyleMultiPrefixTokens) {
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|startoftranscript|>", "type_id": 0}},
        {"SpecialToken": {"id": "<|notimestamps|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "<|startoftranscript|>": {"id": "<|startoftranscript|>", "ids": [50258], "tokens": ["<|startoftranscript|>"]},
        "<|notimestamps|>": {"id": "<|notimestamps|>", "ids": [50364], "tokens": ["<|notimestamps|>"]},
        "<|endoftext|>": {"id": "<|endoftext|>", "ids": [50257], "tokens": ["<|endoftext|>"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("Hello, world!"), EncodeTextCallback, EmitTokenCallback,
      this));

  // Should emit: 50258, 50364, text, 50257
  ASSERT_EQ(emitted_tokens.size(), 3);
  EXPECT_EQ(emitted_tokens[0], 50258);  // <|startoftranscript|>
  EXPECT_EQ(emitted_tokens[1], 50364);  // <|notimestamps|>
  EXPECT_EQ(emitted_tokens[2], 50257);  // <|endoftext|>
  ASSERT_EQ(encoded_texts.size(), 1);
  EXPECT_EQ(encoded_texts[0], "Hello, world!");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Emit Prefix/Suffix Tests
//===----------------------------------------------------------------------===//

// Tests emit_prefix with Whisper-style multi-token prefix.
// This is the code path used by the actual streaming tokenizer (not
// apply_single).
TEST_F(PostprocessorEmitTest, WhisperMultiPrefixEmit) {
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "<|startoftranscript|>", "type_id": 0}},
        {"SpecialToken": {"id": "<|notimestamps|>", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "<|endoftext|>", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "<|startoftranscript|>": {"id": "<|startoftranscript|>", "ids": [50258], "tokens": ["<|startoftranscript|>"]},
        "<|notimestamps|>": {"id": "<|notimestamps|>", "ids": [50364], "tokens": ["<|notimestamps|>"]},
        "<|endoftext|>": {"id": "<|endoftext|>", "ids": [50257], "tokens": ["<|endoftext|>"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  // Test emit_prefix - should emit BOTH special tokens before $A.
  IREE_ASSERT_OK(
      iree_tokenizer_postprocessor_emit_prefix(&pp, EmitTokenCallback, this));

  ASSERT_EQ(emitted_tokens.size(), 2)
      << "emit_prefix should emit both <|startoftranscript|> and "
         "<|notimestamps|>";
  EXPECT_EQ(emitted_tokens[0], 50258);  // <|startoftranscript|>
  EXPECT_EQ(emitted_tokens[1], 50364);  // <|notimestamps|>

  // Test emit_suffix - should emit only <|endoftext|>.
  emitted_tokens.clear();
  IREE_ASSERT_OK(
      iree_tokenizer_postprocessor_emit_suffix(&pp, EmitTokenCallback, this));

  ASSERT_EQ(emitted_tokens.size(), 1);
  EXPECT_EQ(emitted_tokens[0], 50257);  // <|endoftext|>

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

// Tests emit_prefix/suffix with BERT-style single prefix/suffix.
TEST_F(PostprocessorEmitTest, BertSinglePrefixSuffixEmit) {
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
        {"Sequence": {"id": "A", "type_id": 0}},
        {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
      ],
      "pair": null,
      "special_tokens": {
        "[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
        "[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
      }
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp));

  // Test emit_prefix - should emit [CLS].
  IREE_ASSERT_OK(
      iree_tokenizer_postprocessor_emit_prefix(&pp, EmitTokenCallback, this));

  ASSERT_EQ(emitted_tokens.size(), 1);
  EXPECT_EQ(emitted_tokens[0], 101);  // [CLS]

  // Test emit_suffix - should emit [SEP].
  emitted_tokens.clear();
  IREE_ASSERT_OK(
      iree_tokenizer_postprocessor_emit_suffix(&pp, EmitTokenCallback, this));

  ASSERT_EQ(emitted_tokens.size(), 1);
  EXPECT_EQ(emitted_tokens[0], 102);  // [SEP]

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Error Cases
//===----------------------------------------------------------------------===//

TEST(PostprocessorJsonTest, UnknownType) {
  const char* json = R"({
    "post_processor": {
      "type": "UnknownProcessor"
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  iree_status_t status = iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp);

  EXPECT_FALSE(iree_status_is_ok(status));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_UNIMPLEMENTED);
  iree_status_ignore(status);
}

TEST(PostprocessorJsonTest, MissingSpecialToken) {
  const char* json = R"({
    "post_processor": {
      "type": "TemplateProcessing",
      "single": [
        {"SpecialToken": {"id": "[UNKNOWN]", "type_id": 0}}
      ],
      "special_tokens": {}
    }
  })";

  iree_tokenizer_postprocessor_t pp;
  iree_status_t status = iree_tokenizer_postprocessor_parse_json(
      IREE_SV(json), iree_allocator_system(), &pp);

  EXPECT_FALSE(iree_status_is_ok(status));
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_ignore(status);
}

}  // namespace
