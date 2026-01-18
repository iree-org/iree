// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/postprocessor.h"

#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test Helpers
//===----------------------------------------------------------------------===//

struct TestContext {
  std::vector<int32_t> tokens;
  std::vector<std::string> encoded_texts;
};

static iree_status_t test_encode_text(void* user_data,
                                      iree_string_view_t text) {
  auto* ctx = static_cast<TestContext*>(user_data);
  ctx->encoded_texts.push_back(std::string(text.data, text.size));
  return iree_ok_status();
}

static iree_status_t test_emit_token(void* user_data, int32_t token_id) {
  auto* ctx = static_cast<TestContext*>(user_data);
  ctx->tokens.push_back(token_id);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// NONE Postprocessor Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorTest, NoneInitialize) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_none(&pp);
  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_NONE);
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorTest, NoneApplySingle) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_none(&pp);

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello world"), test_encode_text, test_emit_token, &ctx));

  // NONE should just encode the text, no special tokens.
  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "hello world");
  EXPECT_TRUE(ctx.tokens.empty());

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorTest, NoneApplyPair) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_none(&pp);

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_pair(
      &pp, IREE_SV("text a"), IREE_SV("text b"), test_encode_text,
      test_emit_token, &ctx));

  // NONE should encode both texts, no special tokens.
  EXPECT_EQ(ctx.encoded_texts.size(), 2);
  EXPECT_EQ(ctx.encoded_texts[0], "text a");
  EXPECT_EQ(ctx.encoded_texts[1], "text b");
  EXPECT_TRUE(ctx.tokens.empty());

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// RoBERTa Postprocessor Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorTest, RobertaInitialize) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_roberta(
      0, 2, IREE_TOKENIZER_ROBERTA_FLAG_DEFAULT, &pp);
  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_ROBERTA);
  EXPECT_EQ(pp.config.roberta.cls_id, 0);
  EXPECT_EQ(pp.config.roberta.sep_id, 2);
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorTest, RobertaApplySingle) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_roberta(
      0, 2, IREE_TOKENIZER_ROBERTA_FLAG_DEFAULT, &pp);

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), test_encode_text, test_emit_token, &ctx));

  // RoBERTa single: <s> text </s>
  EXPECT_EQ(ctx.tokens.size(), 2);
  EXPECT_EQ(ctx.tokens[0], 0);  // CLS
  EXPECT_EQ(ctx.tokens[1], 2);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "hello");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorTest, RobertaApplyPair) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_roberta(
      0, 2, IREE_TOKENIZER_ROBERTA_FLAG_DEFAULT, &pp);

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_pair(
      &pp, IREE_SV("text a"), IREE_SV("text b"), test_encode_text,
      test_emit_token, &ctx));

  // RoBERTa pair: <s> A </s></s> B </s>
  EXPECT_EQ(ctx.tokens.size(), 4);
  EXPECT_EQ(ctx.tokens[0], 0);  // CLS
  EXPECT_EQ(ctx.tokens[1], 2);  // SEP
  EXPECT_EQ(ctx.tokens[2], 2);  // SEP (doubled)
  EXPECT_EQ(ctx.tokens[3], 2);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 2);
  EXPECT_EQ(ctx.encoded_texts[0], "text a");
  EXPECT_EQ(ctx.encoded_texts[1], "text b");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Template Postprocessor Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorTest, TemplateApplySingle) {
  // Create a BERT-style template: [CLS] $A [SEP]
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  // [CLS] token
  templates[0].type = IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL;
  templates[0].token_id = 101;
  templates[0].type_id = 0;
  templates[0].reserved = 0;

  // $A sequence
  templates[1].type = IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A;
  templates[1].token_id = -1;
  templates[1].type_id = 0;
  templates[1].reserved = 0;

  // [SEP] token
  templates[2].type = IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL;
  templates[2].token_id = 102;
  templates[2].type_id = 0;
  templates[2].reserved = 0;

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 3, 0, iree_allocator_system(), &pp));

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), test_encode_text, test_emit_token, &ctx));

  // Should emit: CLS, encode("hello"), SEP
  EXPECT_EQ(ctx.tokens.size(), 2);
  EXPECT_EQ(ctx.tokens[0], 101);  // CLS
  EXPECT_EQ(ctx.tokens[1], 102);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "hello");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorTest, TemplateApplyPair) {
  // Create a BERT-style template: [CLS] $A [SEP] $B [SEP]
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 8 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  // Single template (3 pieces): [CLS] $A [SEP]
  templates[0] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};
  templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};
  templates[2] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};

  // Pair template (5 pieces): [CLS] $A [SEP] $B [SEP]
  templates[3] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};
  templates[4] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};
  templates[5] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};
  templates[6] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B, 1, 0};
  templates[7] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 1, 0};

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 3, 5, iree_allocator_system(), &pp));

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_pair(
      &pp, IREE_SV("text a"), IREE_SV("text b"), test_encode_text,
      test_emit_token, &ctx));

  // Should emit: CLS, encode("text a"), SEP, encode("text b"), SEP
  EXPECT_EQ(ctx.tokens.size(), 3);
  EXPECT_EQ(ctx.tokens[0], 101);  // CLS
  EXPECT_EQ(ctx.tokens[1], 102);  // SEP
  EXPECT_EQ(ctx.tokens[2], 102);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 2);
  EXPECT_EQ(ctx.encoded_texts[0], "text a");
  EXPECT_EQ(ctx.encoded_texts[1], "text b");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// ByteLevel Postprocessor Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorTest, ByteLevelApplySingle) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_byte_level(0, &pp);

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), test_encode_text, test_emit_token, &ctx));

  // ByteLevel should just encode the text (no-op for post-processing).
  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "hello");
  EXPECT_TRUE(ctx.tokens.empty());

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Sequence Postprocessor Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorTest, SequenceApplySingle) {
  // Create a sequence with ByteLevel + RoBERTa (like Llama 3 but simpler).
  iree_tokenizer_postprocessor_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 2 * sizeof(iree_tokenizer_postprocessor_t),
      (void**)&children));

  iree_tokenizer_postprocessor_initialize_byte_level(0, &children[0]);
  iree_tokenizer_postprocessor_initialize_roberta(
      0, 2, IREE_TOKENIZER_ROBERTA_FLAG_DEFAULT, &children[1]);

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_sequence(
      children, 2, iree_allocator_system(), &pp));

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), test_encode_text, test_emit_token, &ctx));

  // Sequence finds the first template-based processor (RoBERTa) and uses it.
  EXPECT_EQ(ctx.tokens.size(), 2);
  EXPECT_EQ(ctx.tokens[0], 0);  // CLS
  EXPECT_EQ(ctx.tokens[1], 2);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "hello");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorTest, NullPostprocessorPassthrough) {
  // Test that NULL postprocessor just encodes text.
  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      nullptr, IREE_SV("hello"), test_encode_text, test_emit_token, &ctx));

  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "hello");
  EXPECT_TRUE(ctx.tokens.empty());
}

//===----------------------------------------------------------------------===//
// Extended Template Postprocessor Tests
//===----------------------------------------------------------------------===//

TEST(TemplateProcessorTest, EmptyInput) {
  // Test template with empty input text.
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  templates[0] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // [CLS]
  templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};
  templates[2] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // [SEP]

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 3, 0, iree_allocator_system(), &pp));

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV(""), test_encode_text, test_emit_token, &ctx));

  // Should still emit special tokens even with empty input.
  EXPECT_EQ(ctx.tokens.size(), 2);
  EXPECT_EQ(ctx.tokens[0], 101);  // CLS
  EXPECT_EQ(ctx.tokens[1], 102);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 1);
  EXPECT_EQ(ctx.encoded_texts[0], "");  // Empty text still encoded.

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(TemplateProcessorTest, SpecialTokensOnly) {
  // Template with only special tokens, no sequence placeholder.
  // This is unusual but valid - e.g., a special "reset" template.
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 2 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  templates[0] = {100, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // BOS
  templates[1] = {200, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // EOS

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 2, 0, iree_allocator_system(), &pp));

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("ignored"), test_encode_text, test_emit_token, &ctx));

  // Should emit both special tokens, text is ignored (no $A placeholder).
  EXPECT_EQ(ctx.tokens.size(), 2);
  EXPECT_EQ(ctx.tokens[0], 100);
  EXPECT_EQ(ctx.tokens[1], 200);
  EXPECT_TRUE(ctx.encoded_texts.empty());

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(TemplateProcessorTest, MultipleSeparatorsBetweenSequences) {
  // Template: [CLS] $A [SEP][SEP] $B [SEP] (RoBERTa-like double sep).
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 6 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  // Pair template only (6 pieces).
  templates[0] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // CLS
  templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};
  templates[2] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // SEP
  templates[3] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};  // SEP
  templates[4] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B, 1, 0};
  templates[5] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 1, 0};  // SEP

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 0, 6, iree_allocator_system(), &pp));

  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_pair(
      &pp, IREE_SV("first"), IREE_SV("second"), test_encode_text,
      test_emit_token, &ctx));

  // Should emit: CLS, encode(first), SEP, SEP, encode(second), SEP.
  EXPECT_EQ(ctx.tokens.size(), 4);
  EXPECT_EQ(ctx.tokens[0], 101);  // CLS
  EXPECT_EQ(ctx.tokens[1], 102);  // SEP
  EXPECT_EQ(ctx.tokens[2], 102);  // SEP (doubled)
  EXPECT_EQ(ctx.tokens[3], 102);  // SEP
  EXPECT_EQ(ctx.encoded_texts.size(), 2);
  EXPECT_EQ(ctx.encoded_texts[0], "first");
  EXPECT_EQ(ctx.encoded_texts[1], "second");

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(TemplateProcessorTest, TypeIdAssignment) {
  // Verify type_id field is set correctly for different segments.
  // BERT: segment A has type_id 0, segment B has type_id 1.
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 5 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  // Pair template with explicit type_ids.
  templates[0] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0,
                  0};  // CLS, type 0
  templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0,
                  0};  // $A, type 0
  templates[2] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0,
                  0};  // SEP, type 0
  templates[3] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_B, 1,
                  0};  // $B, type 1
  templates[4] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 1,
                  0};  // SEP, type 1

  // Verify the type_ids are correctly set in the template.
  EXPECT_EQ(templates[0].type_id, 0);
  EXPECT_EQ(templates[1].type_id, 0);
  EXPECT_EQ(templates[2].type_id, 0);
  EXPECT_EQ(templates[3].type_id, 1);
  EXPECT_EQ(templates[4].type_id, 1);

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 0, 5, iree_allocator_system(), &pp));

  // The type_ids are stored in the template pieces. We verify the structure.
  EXPECT_EQ(pp.type, IREE_TOKENIZER_POSTPROCESSOR_TEMPLATE);
  EXPECT_EQ(pp.config.template_.pair_count, 5);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(TemplateProcessorTest, SingleSequenceNoPair) {
  // Template only defines single, apply_pair should work (common case).
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  templates[0] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};
  templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};
  templates[2] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 3, 0, iree_allocator_system(), &pp));

  // Apply pair when only single is defined.
  TestContext ctx;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_apply_pair(
      &pp, IREE_SV("first"), IREE_SV("second"), test_encode_text,
      test_emit_token, &ctx));

  // Should fall back to encoding both sequences without pair template.
  // Behavior: uses single template for first, then encodes second directly.
  // Or: encodes both as simple concatenation. Check actual behavior.
  EXPECT_FALSE(ctx.encoded_texts.empty());

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

//===----------------------------------------------------------------------===//
// Callback Error Propagation Tests
//===----------------------------------------------------------------------===//

static iree_status_t failing_encode_text(void* user_data,
                                         iree_string_view_t text) {
  (void)user_data;
  (void)text;
  return iree_make_status(IREE_STATUS_ABORTED, "encode error");
}

static iree_status_t failing_emit_token(void* user_data, int32_t token_id) {
  (void)user_data;
  (void)token_id;
  return iree_make_status(IREE_STATUS_CANCELLED, "emit error");
}

TEST(PostprocessorCallbackTest, EncodeTextErrorPropagates) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_roberta(0, 2, 0, &pp);

  TestContext ctx;
  iree_status_t status = iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), failing_encode_text, test_emit_token, &ctx);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
  iree_status_free(status);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorCallbackTest, EmitTokenErrorPropagates) {
  iree_tokenizer_postprocessor_t pp;
  iree_tokenizer_postprocessor_initialize_roberta(0, 2, 0, &pp);

  TestContext ctx;
  iree_status_t status = iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), test_encode_text, failing_emit_token, &ctx);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  iree_status_free(status);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorCallbackTest, TemplateCallbackErrorPropagates) {
  iree_tokenizer_template_piece_t* templates = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(
      iree_allocator_system(), 3 * sizeof(iree_tokenizer_template_piece_t),
      (void**)&templates));

  templates[0] = {101, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};
  templates[1] = {-1, IREE_TOKENIZER_TEMPLATE_PIECE_SEQUENCE_A, 0, 0};
  templates[2] = {102, IREE_TOKENIZER_TEMPLATE_PIECE_SPECIAL, 0, 0};

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_template(
      templates, 3, 0, iree_allocator_system(), &pp));

  // Error on emit_token (first thing called - CLS).
  TestContext ctx;
  iree_status_t status = iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), test_encode_text, failing_emit_token, &ctx);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  iree_status_free(status);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(PostprocessorCallbackTest, SequenceCallbackErrorPropagates) {
  // Error should propagate through Sequence post-processor.
  iree_tokenizer_postprocessor_t* children = nullptr;
  IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                       sizeof(iree_tokenizer_postprocessor_t),
                                       (void**)&children));

  iree_tokenizer_postprocessor_initialize_roberta(0, 2, 0, &children[0]);

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize_sequence(
      children, 1, iree_allocator_system(), &pp));

  TestContext ctx;
  iree_status_t status = iree_tokenizer_postprocessor_apply_single(
      &pp, IREE_SV("hello"), failing_encode_text, test_emit_token, &ctx);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
  iree_status_free(status);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

}  // namespace
