// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/postprocessor.h"

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace tokenizer {
namespace {

// Helper to build a template with given prefix/infix/suffix IDs.
iree_tokenizer_postprocessor_template_t MakeTemplate(
    std::initializer_list<int32_t> prefix_ids,
    std::initializer_list<int32_t> infix_ids,
    std::initializer_list<int32_t> suffix_ids,
    std::initializer_list<uint8_t> prefix_type_ids = {},
    std::initializer_list<uint8_t> infix_type_ids = {},
    std::initializer_list<uint8_t> suffix_type_ids = {},
    uint8_t sequence_a_type_id = 0, uint8_t sequence_b_type_id = 0) {
  iree_tokenizer_postprocessor_template_t t = {};
  t.prefix_count = static_cast<uint8_t>(prefix_ids.size());
  t.infix_count = static_cast<uint8_t>(infix_ids.size());
  t.suffix_count = static_cast<uint8_t>(suffix_ids.size());
  t.sequence_a_type_id = sequence_a_type_id;
  t.sequence_b_type_id = sequence_b_type_id;

  size_t i = 0;
  for (int32_t id : prefix_ids) t.token_ids[i++] = id;
  for (int32_t id : infix_ids) t.token_ids[i++] = id;
  for (int32_t id : suffix_ids) t.token_ids[i++] = id;

  i = 0;
  if (prefix_type_ids.size() > 0) {
    for (uint8_t tid : prefix_type_ids) t.type_ids[i++] = tid;
  } else {
    i += t.prefix_count;
  }
  if (infix_type_ids.size() > 0) {
    for (uint8_t tid : infix_type_ids) t.type_ids[i++] = tid;
  } else {
    i += t.infix_count;
  }
  if (suffix_type_ids.size() > 0) {
    for (uint8_t tid : suffix_type_ids) t.type_ids[i++] = tid;
  }

  return t;
}

// Helper to create a default postprocessor for encode state tests.
// Uses the given template as the single template, no pair.
iree_tokenizer_postprocessor_t MakePostprocessor(
    const iree_tokenizer_postprocessor_template_t& single,
    iree_tokenizer_postprocessor_flags_t flags =
        IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE) {
  iree_tokenizer_postprocessor_t pp = {};
  pp.single = single;
  pp.flags = flags;
  return pp;
}

TEST(PostprocessorTemplate, TotalCount) {
  iree_tokenizer_postprocessor_template_t t = {};
  t.prefix_count = 1;
  t.infix_count = 2;
  t.suffix_count = 1;
  EXPECT_EQ(iree_tokenizer_postprocessor_template_total_count(&t), 4);
}

TEST(PostprocessorTemplate, TotalCountEmpty) {
  iree_tokenizer_postprocessor_template_t t = {};
  EXPECT_EQ(iree_tokenizer_postprocessor_template_total_count(&t), 0);
}

TEST(Postprocessor, InitializeBertSingle) {
  // BERT single: [CLS] $A [SEP]
  auto single = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 1);
  EXPECT_EQ(pp.single.token_ids[0], 101);  // [CLS]
  EXPECT_EQ(pp.single.token_ids[1], 102);  // [SEP]
  EXPECT_EQ(pp.single.sequence_a_type_id, 0);
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_tokenizer_postprocessor_supports_pair(&pp));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, InitializeBertPair) {
  // BERT single: [CLS] $A [SEP]
  auto single = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  // BERT pair: [CLS] $A [SEP] $B [SEP]
  auto pair = MakeTemplate(
      /*prefix=*/{101}, /*infix=*/{102}, /*suffix=*/{102},
      /*prefix_type_ids=*/{0}, /*infix_type_ids=*/{0},
      /*suffix_type_ids=*/{1},
      /*sequence_a_type_id=*/0, /*sequence_b_type_id=*/1);

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, &pair, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));

  // Pair template.
  EXPECT_EQ(pp.pair.prefix_count, 1);
  EXPECT_EQ(pp.pair.infix_count, 1);
  EXPECT_EQ(pp.pair.suffix_count, 1);
  EXPECT_EQ(pp.pair.token_ids[0], 101);  // [CLS] prefix
  EXPECT_EQ(pp.pair.token_ids[1], 102);  // [SEP] infix
  EXPECT_EQ(pp.pair.token_ids[2], 102);  // [SEP] suffix
  EXPECT_EQ(pp.pair.type_ids[0], 0);     // [CLS] type=0
  EXPECT_EQ(pp.pair.type_ids[1], 0);     // [SEP] infix type=0
  EXPECT_EQ(pp.pair.type_ids[2], 1);     // [SEP] suffix type=1
  EXPECT_EQ(pp.pair.sequence_a_type_id, 0);
  EXPECT_EQ(pp.pair.sequence_b_type_id, 1);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, InitializeLlama) {
  // LLaMA 2 single: <bos> $A
  auto single = MakeTemplate(/*prefix=*/{1}, /*infix=*/{}, /*suffix=*/{});

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_EQ(pp.single.prefix_count, 1);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 0);
  EXPECT_EQ(pp.single.token_ids[0], 1);  // <bos>

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, InitializeRobertaPair) {
  // RoBERTa single: <s> $A </s>
  auto single = MakeTemplate(/*prefix=*/{0}, /*infix=*/{}, /*suffix=*/{2});
  // RoBERTa pair: <s> $A </s></s> $B </s>
  auto pair = MakeTemplate(
      /*prefix=*/{0}, /*infix=*/{2, 2}, /*suffix=*/{2},
      /*prefix_type_ids=*/{0}, /*infix_type_ids=*/{0, 0},
      /*suffix_type_ids=*/{0},
      /*sequence_a_type_id=*/0, /*sequence_b_type_id=*/0);

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, &pair, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));
  EXPECT_EQ(pp.pair.infix_count, 2);
  EXPECT_EQ(pp.pair.token_ids[1], 2);  // </s> first infix
  EXPECT_EQ(pp.pair.token_ids[2], 2);  // </s> second infix

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, InitializeWhisper) {
  // Whisper: <|startoftranscript|><|en|><|transcribe|> $A <|endoftext|>
  auto single = MakeTemplate(/*prefix=*/{50258, 50259, 50360},
                             /*infix=*/{}, /*suffix=*/{50257});

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_EQ(pp.single.prefix_count, 3);
  EXPECT_EQ(pp.single.suffix_count, 1);
  EXPECT_EQ(pp.single.token_ids[0], 50258);
  EXPECT_EQ(pp.single.token_ids[1], 50259);
  EXPECT_EQ(pp.single.token_ids[2], 50360);
  EXPECT_EQ(pp.single.token_ids[3], 50257);

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, InitializeByteLevelOnly) {
  // ByteLevel: no special tokens, no offset trimming.
  iree_tokenizer_postprocessor_template_t empty = {};

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &empty, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_EQ(pp.single.prefix_count, 0);
  EXPECT_EQ(pp.single.infix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 0);
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  EXPECT_FALSE(iree_tokenizer_postprocessor_supports_pair(&pp));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, TrimOffsetsStored) {
  // TRIM_OFFSETS flag is accepted and stored.
  iree_tokenizer_postprocessor_template_t empty = {};
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &empty, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS,
      &pp));
  EXPECT_TRUE(iree_any_bit_set(pp.flags,
                               IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, AddPrefixSpaceStored) {
  // ADD_PREFIX_SPACE flag is accepted and stored.
  iree_tokenizer_postprocessor_template_t empty = {};
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &empty, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE,
      &pp));
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_TRUE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, ValidateMaxPiecesExceeded) {
  // Build a template that exceeds MAX_PIECES (7).
  iree_tokenizer_postprocessor_template_t too_many = {};
  too_many.prefix_count = 4;
  too_many.infix_count = 2;
  too_many.suffix_count = 2;  // Total = 8 > 7.

  iree_tokenizer_postprocessor_t pp;
  iree_status_t status = iree_tokenizer_postprocessor_initialize(
      &too_many, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(Postprocessor, ValidatePairMaxPiecesExceeded) {
  iree_tokenizer_postprocessor_template_t single = {};
  single.prefix_count = 1;
  single.suffix_count = 1;

  iree_tokenizer_postprocessor_template_t pair_too_many = {};
  pair_too_many.prefix_count = 3;
  pair_too_many.infix_count = 3;
  pair_too_many.suffix_count = 3;  // Total = 9 > 7.

  iree_tokenizer_postprocessor_t pp;
  iree_status_t status = iree_tokenizer_postprocessor_initialize(
      &single, &pair_too_many, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, status);
}

TEST(Postprocessor, DeinitializeZeros) {
  auto single = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, /*pair=*/NULL, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  iree_tokenizer_postprocessor_deinitialize(&pp);

  // After deinitialize, struct is zeroed.
  EXPECT_EQ(pp.single.prefix_count, 0);
  EXPECT_EQ(pp.single.suffix_count, 0);
  EXPECT_FALSE(iree_any_bit_set(
      pp.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
}

TEST(Postprocessor, SupportsPairWithTypeIdsOnly) {
  // A pair template with no special tokens but different type_ids still counts
  // as pair-capable (needed for models that only differentiate by type_id).
  iree_tokenizer_postprocessor_template_t single = {};
  iree_tokenizer_postprocessor_template_t pair = {};
  pair.sequence_a_type_id = 0;
  pair.sequence_b_type_id = 1;

  iree_tokenizer_postprocessor_t pp;
  IREE_ASSERT_OK(iree_tokenizer_postprocessor_initialize(
      &single, &pair, IREE_TOKENIZER_POSTPROCESSOR_FLAG_NONE, &pp));

  EXPECT_TRUE(iree_tokenizer_postprocessor_supports_pair(&pp));

  iree_tokenizer_postprocessor_deinitialize(&pp);
}

TEST(Postprocessor, DeinitializeNull) {
  // Should not crash.
  iree_tokenizer_postprocessor_deinitialize(NULL);
}

//===----------------------------------------------------------------------===//
// Encode State Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorEncodeState, InitializeEmptyTemplate) {
  iree_tokenizer_postprocessor_template_t empty = {};
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(empty);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &empty, &state);

  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_IDLE);
  EXPECT_EQ(state.active_template, nullptr);
}

TEST(PostprocessorEncodeState, InitializeWithPrefix) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX);
  EXPECT_EQ(state.position, 0);
  EXPECT_EQ(state.active_template, &tmpl);
}

TEST(PostprocessorEncodeState, InitializeSuffixOnly) {
  // Template with no prefix but has suffix — starts in SEQUENCE_A.
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A);
}

TEST(PostprocessorEncodeState, EmitPrefix) {
  // BERT: [CLS] $A [SEP] — prefix={101}, suffix={102}
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[8] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 8);

  iree_host_size_t emitted =
      iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);
  EXPECT_EQ(emitted, 1u);
  EXPECT_EQ(token_ids[0], 101);
  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A);
}

TEST(PostprocessorEncodeState, EmitPrefixMultiple) {
  // Whisper: 3 prefix tokens.
  auto tmpl = MakeTemplate(/*prefix=*/{50258, 50259, 50360},
                           /*infix=*/{}, /*suffix=*/{50257});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[8] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 8);

  iree_host_size_t emitted =
      iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);
  EXPECT_EQ(emitted, 3u);
  EXPECT_EQ(token_ids[0], 50258);
  EXPECT_EQ(token_ids[1], 50259);
  EXPECT_EQ(token_ids[2], 50360);
  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A);
}

TEST(PostprocessorEncodeState, EmitPrefixPartialCapacity) {
  // 3 prefix tokens but only capacity for 2 — phase stays PREFIX.
  auto tmpl = MakeTemplate(/*prefix=*/{50258, 50259, 50360},
                           /*infix=*/{}, /*suffix=*/{50257});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[2] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 2);

  iree_host_size_t emitted =
      iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);
  EXPECT_EQ(emitted, 2u);
  EXPECT_EQ(token_ids[0], 50258);
  EXPECT_EQ(token_ids[1], 50259);
  // Phase stays PREFIX — not all tokens emitted.
  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX);
  EXPECT_EQ(state.position, 2);

  // Resume with more capacity.
  iree_tokenizer_token_id_t token_ids2[4] = {};
  iree_tokenizer_token_output_t output2 =
      iree_tokenizer_make_token_output(token_ids2, NULL, NULL, 4);
  emitted = iree_tokenizer_postprocessor_emit_prefix(&state, output2, 0);
  EXPECT_EQ(emitted, 1u);
  EXPECT_EQ(token_ids2[0], 50360);
  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A);
}

TEST(PostprocessorEncodeState, EmitPrefixNoOpWhenNotInPrefixPhase) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  // Advance past prefix.
  iree_tokenizer_token_id_t token_ids[8] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 8);
  iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);
  ASSERT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A);

  // Second call is a no-op.
  iree_host_size_t emitted =
      iree_tokenizer_postprocessor_emit_prefix(&state, output, 1);
  EXPECT_EQ(emitted, 0u);
}

TEST(PostprocessorEncodeState, EmitSuffix) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[8] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 8);

  // Emit prefix, then transition to suffix.
  iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);
  iree_tokenizer_postprocessor_begin_suffix(&state);
  ASSERT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_SUFFIX);

  iree_host_size_t emitted =
      iree_tokenizer_postprocessor_emit_suffix(&state, output, 1);
  EXPECT_EQ(emitted, 1u);
  EXPECT_EQ(token_ids[1], 102);
  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_DONE);
}

TEST(PostprocessorEncodeState, BeginSuffixNoSuffix) {
  // Template with prefix only — begin_suffix transitions directly to DONE.
  auto tmpl = MakeTemplate(/*prefix=*/{1}, /*infix=*/{}, /*suffix=*/{});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  // Skip prefix.
  state.phase = IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A;
  iree_tokenizer_postprocessor_begin_suffix(&state);
  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_DONE);
}

TEST(PostprocessorEncodeState, AssignTypeIds) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102},
                           /*prefix_type_ids=*/{}, /*infix_type_ids=*/{},
                           /*suffix_type_ids=*/{},
                           /*sequence_a_type_id=*/0);
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  // Advance to SEQUENCE_A.
  state.phase = IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A;

  uint8_t type_ids[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  iree_tokenizer_token_id_t token_ids[4] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 4);

  iree_tokenizer_postprocessor_assign_type_ids(&state, output, 0, 3);
  EXPECT_EQ(type_ids[0], 0);
  EXPECT_EQ(type_ids[1], 0);
  EXPECT_EQ(type_ids[2], 0);
  EXPECT_EQ(type_ids[3], 0xFF);  // Untouched.
}

TEST(PostprocessorEncodeState, AssignTypeIdsNonZero) {
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{},
                           /*prefix_type_ids=*/{}, /*infix_type_ids=*/{},
                           /*suffix_type_ids=*/{},
                           /*sequence_a_type_id=*/2);
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);
  state.phase = IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A;
  state.active_template = &tmpl;

  uint8_t type_ids[3] = {};
  iree_tokenizer_token_id_t token_ids[3] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 3);

  iree_tokenizer_postprocessor_assign_type_ids(&state, output, 1, 2);
  EXPECT_EQ(type_ids[0], 0);  // Untouched.
  EXPECT_EQ(type_ids[1], 2);
  EXPECT_EQ(type_ids[2], 2);
}

TEST(PostprocessorEncodeState, AssignTypeIdsNoOpWhenNotSequencePhase) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);
  // State is in PREFIX phase.

  uint8_t type_ids[4] = {0xFF, 0xFF, 0xFF, 0xFF};
  iree_tokenizer_token_id_t token_ids[4] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 4);

  iree_tokenizer_postprocessor_assign_type_ids(&state, output, 0, 3);
  // All untouched — wrong phase.
  EXPECT_EQ(type_ids[0], 0xFF);
  EXPECT_EQ(type_ids[1], 0xFF);
  EXPECT_EQ(type_ids[2], 0xFF);
}

TEST(PostprocessorEncodeState, AssignTypeIdsNoOpNullTypeIds) {
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{},
                           /*prefix_type_ids=*/{}, /*infix_type_ids=*/{},
                           /*suffix_type_ids=*/{},
                           /*sequence_a_type_id=*/1);
  iree_tokenizer_postprocessor_encode_state_t state = {};
  state.phase = IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A;
  state.active_template = &tmpl;

  iree_tokenizer_token_id_t token_ids[4] = {};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 4);

  // Should not crash.
  iree_tokenizer_postprocessor_assign_type_ids(&state, output, 0, 3);
}

TEST(PostprocessorEncodeState, FullBertFlow) {
  // BERT: [CLS]=101 $A [SEP]=102, sequence_a_type_id=0
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102},
                           /*prefix_type_ids=*/{0}, /*infix_type_ids=*/{},
                           /*suffix_type_ids=*/{0},
                           /*sequence_a_type_id=*/0);
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[8] = {};
  uint8_t type_ids[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 8);

  iree_host_size_t offset = 0;

  // Prefix: [CLS]
  offset += iree_tokenizer_postprocessor_emit_prefix(&state, output, offset);
  EXPECT_EQ(offset, 1u);

  // Simulate model producing 3 tokens at positions 1,2,3.
  token_ids[1] = 1000;
  token_ids[2] = 1001;
  token_ids[3] = 1002;
  iree_tokenizer_postprocessor_assign_type_ids(&state, output, 1, 3);
  offset += 3;

  // Suffix: [SEP]
  iree_tokenizer_postprocessor_begin_suffix(&state);
  offset += iree_tokenizer_postprocessor_emit_suffix(&state, output, offset);
  EXPECT_EQ(offset, 5u);

  // Verify final output: [CLS] 1000 1001 1002 [SEP]
  EXPECT_EQ(token_ids[0], 101);
  EXPECT_EQ(token_ids[1], 1000);
  EXPECT_EQ(token_ids[2], 1001);
  EXPECT_EQ(token_ids[3], 1002);
  EXPECT_EQ(token_ids[4], 102);

  // Type IDs: all 0 (prefix type=0, sequence_a=0, suffix type=0).
  EXPECT_EQ(type_ids[0], 0);
  EXPECT_EQ(type_ids[1], 0);
  EXPECT_EQ(type_ids[2], 0);
  EXPECT_EQ(type_ids[3], 0);
  EXPECT_EQ(type_ids[4], 0);

  EXPECT_EQ(state.phase, IREE_TOKENIZER_POSTPROCESSOR_PHASE_DONE);
}

TEST(PostprocessorEncodeState, EmitPrefixWithOffsets) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[4] = {};
  iree_tokenizer_offset_t offsets[4] = {{99, 99}, {99, 99}, {99, 99}, {99, 99}};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, offsets, NULL, 4);

  iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);
  // Special tokens get zero-length offsets.
  EXPECT_EQ(offsets[0].start, 0u);
  EXPECT_EQ(offsets[0].end, 0u);
  // Other positions untouched.
  EXPECT_EQ(offsets[1].start, 99u);
}

//===----------------------------------------------------------------------===//
// Offset Trimming State Tests
//===----------------------------------------------------------------------===//

TEST(PostprocessorEncodeState, TrimOffsetsFlagsCached) {
  auto tmpl = MakeTemplate(/*prefix=*/{101}, /*infix=*/{}, /*suffix=*/{102});
  iree_tokenizer_postprocessor_encode_state_t state;

  // Test with no flags.
  auto pp1 = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp1, &tmpl, &state);
  EXPECT_FALSE(iree_any_bit_set(
      state.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      state.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  EXPECT_FALSE(state.first_model_token_trimmed);

  // Test with TRIM_OFFSETS only.
  auto pp2 =
      MakePostprocessor(tmpl, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp2, &tmpl, &state);
  EXPECT_TRUE(iree_any_bit_set(state.flags,
                               IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_FALSE(iree_any_bit_set(
      state.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  EXPECT_FALSE(state.first_model_token_trimmed);

  // Test with both flags.
  auto pp3 = MakePostprocessor(
      tmpl, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS |
                IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp3, &tmpl, &state);
  EXPECT_TRUE(iree_any_bit_set(state.flags,
                               IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_TRUE(iree_any_bit_set(
      state.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
  EXPECT_FALSE(state.first_model_token_trimmed);

  // Test with ADD_PREFIX_SPACE only.
  auto pp4 = MakePostprocessor(
      tmpl, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE);
  iree_tokenizer_postprocessor_encode_state_initialize(&pp4, &tmpl, &state);
  EXPECT_FALSE(iree_any_bit_set(
      state.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS));
  EXPECT_TRUE(iree_any_bit_set(
      state.flags, IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE));
}

TEST(PostprocessorTrimOffsets, NoOpWhenDisabled) {
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{});
  auto pp = MakePostprocessor(tmpl);
  iree_tokenizer_postprocessor_encode_state_t state;
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[2] = {100, 200};
  iree_tokenizer_offset_t offsets[2] = {{0, 5}, {5, 10}};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, offsets, NULL, 2);

  // With TRIM_OFFSETS unset, offsets should be unchanged (even with NULL
  // vocab).
  iree_tokenizer_postprocessor_trim_token_offsets(&state, /*vocab=*/NULL,
                                                  output, 0, 2);
  EXPECT_EQ(offsets[0].start, 0u);
  EXPECT_EQ(offsets[0].end, 5u);
  EXPECT_EQ(offsets[1].start, 5u);
  EXPECT_EQ(offsets[1].end, 10u);
}

TEST(PostprocessorTrimOffsets, NoOpWhenNoOffsets) {
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{});
  auto pp =
      MakePostprocessor(tmpl, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS);
  iree_tokenizer_postprocessor_encode_state_t state;
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[2] = {100, 200};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, NULL, NULL, 2);

  // Should not crash with NULL token_offsets.
  iree_tokenizer_postprocessor_trim_token_offsets(&state, /*vocab=*/NULL,
                                                  output, 0, 2);
}

TEST(PostprocessorTrimOffsets, NoOpWhenNoVocab) {
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{});
  auto pp =
      MakePostprocessor(tmpl, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS);
  iree_tokenizer_postprocessor_encode_state_t state;
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  iree_tokenizer_token_id_t token_ids[2] = {100, 200};
  iree_tokenizer_offset_t offsets[2] = {{0, 5}, {5, 10}};
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(token_ids, offsets, NULL, 2);

  // With NULL vocab, offsets should be unchanged (early return).
  iree_tokenizer_postprocessor_trim_token_offsets(&state, /*vocab=*/NULL,
                                                  output, 0, 2);
  EXPECT_EQ(offsets[0].start, 0u);
  EXPECT_EQ(offsets[0].end, 5u);
}

TEST(PostprocessorTrimOffsets, FirstTokenTrackedAcrossCalls) {
  // This test verifies the state tracking logic by manually setting the flag.
  // With NULL vocab, the function returns early without modifying state, so we
  // test the flag semantics directly rather than through the function.
  auto tmpl = MakeTemplate(/*prefix=*/{}, /*infix=*/{}, /*suffix=*/{});
  auto pp = MakePostprocessor(
      tmpl, IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS |
                IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE);
  iree_tokenizer_postprocessor_encode_state_t state;
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);

  // Initial state: first_model_token_trimmed is false.
  EXPECT_FALSE(state.first_model_token_trimmed);

  // Simulate first batch processing (what the function would do with real
  // vocab). After processing any tokens, first_model_token_trimmed becomes
  // true.
  state.first_model_token_trimmed = true;
  EXPECT_TRUE(state.first_model_token_trimmed);

  // Verify that re-initializing resets the flag (important for encode resets).
  iree_tokenizer_postprocessor_encode_state_initialize(&pp, &tmpl, &state);
  EXPECT_FALSE(state.first_model_token_trimmed);
}

// Full trim_offsets testing with actual token text lookup is performed in
// tokenizer_huggingface_test.cc where real vocabularies are available.

}  // namespace
}  // namespace tokenizer
}  // namespace iree
