// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/model/model_test_util.h"

namespace iree::tokenizer::testing {

namespace {

// Maximum tokens we expect from a single encode/finalize call.
// Must be large enough to hold outputs from long-segment tests (>2048 bytes
// with 1-byte-per-token worst case).
constexpr size_t kMaxTokens = 4096;

}  // namespace

EncodeResult EncodeWithOffsetsAndFinalize(
    iree_tokenizer_model_t* model, const std::string& text,
    const std::vector<iree_tokenizer_segment_t>& segments,
    bool expect_pending_after_encode) {
  ScopedModelState state(model);

  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};

  iree_tokenizer_token_id_t tokens[kMaxTokens];
  iree_tokenizer_offset_t offsets[kMaxTokens];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  // Encode all segments with offset tracking.
  IREE_EXPECT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer,
      iree_tokenizer_make_segment_list(segments.data(), segments.size()),
      iree_tokenizer_make_token_output(tokens, offsets, NULL, kMaxTokens),
      &segments_consumed, &token_count));

  if (token_count < kMaxTokens) {
    EXPECT_EQ(segments_consumed, segments.size())
        << "Expected all segments to be consumed when output has capacity";
  }

  bool has_pending = iree_tokenizer_model_state_has_pending(state.get());
  EXPECT_EQ(has_pending, expect_pending_after_encode)
      << "has_pending() after encode() should be "
      << (expect_pending_after_encode ? "true" : "false") << " for text: \""
      << text << "\"";

  EncodeResult result;
  result.tokens.assign(tokens, tokens + token_count);
  result.offsets.assign(offsets, offsets + token_count);

  // Finalize.
  iree_tokenizer_token_id_t finalize_tokens[kMaxTokens];
  iree_tokenizer_offset_t finalize_offsets[kMaxTokens];
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_model_state_finalize(
      state.get(),
      iree_tokenizer_make_token_output(finalize_tokens, finalize_offsets, NULL,
                                       kMaxTokens),
      &finalize_count));

  for (size_t i = 0; i < finalize_count; ++i) {
    result.tokens.push_back(finalize_tokens[i]);
    result.offsets.push_back(finalize_offsets[i]);
  }

  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

EncodeResult EncodeWithOffsetsAndFinalize(iree_tokenizer_model_t* model,
                                          const std::string& text,
                                          bool expect_pending_after_encode) {
  std::vector<iree_tokenizer_segment_t> segments = {
      {0, static_cast<iree_host_size_t>(text.size())}};
  return EncodeWithOffsetsAndFinalize(model, text, segments,
                                      expect_pending_after_encode);
}

std::vector<iree_tokenizer_token_id_t> EncodeAndFinalize(
    iree_tokenizer_model_t* model, const std::string& text,
    const std::vector<iree_tokenizer_segment_t>& segments,
    bool expect_pending_after_encode) {
  ScopedModelState state(model);

  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};

  iree_tokenizer_token_id_t tokens[kMaxTokens];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  // Encode all segments.
  IREE_EXPECT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer,
      iree_tokenizer_make_segment_list(segments.data(), segments.size()),
      iree_tokenizer_make_token_output(tokens, NULL, NULL, kMaxTokens),
      &segments_consumed, &token_count));

  // Verify all segments were consumed (unless output was full).
  if (token_count < kMaxTokens) {
    EXPECT_EQ(segments_consumed, segments.size())
        << "Expected all segments to be consumed when output has capacity";
  }

  // Verify has_pending() after encode.
  bool has_pending = iree_tokenizer_model_state_has_pending(state.get());
  EXPECT_EQ(has_pending, expect_pending_after_encode)
      << "has_pending() after encode() should be "
      << (expect_pending_after_encode ? "true" : "false") << " for text: \""
      << text << "\"";

  // Collect tokens from encode().
  std::vector<iree_tokenizer_token_id_t> result(tokens, tokens + token_count);

  // Always call finalize().
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_model_state_finalize(
      state.get(),
      iree_tokenizer_make_token_output(tokens, NULL, NULL, kMaxTokens),
      &finalize_count));

  // Add any tokens from finalize().
  for (size_t i = 0; i < finalize_count; ++i) {
    result.push_back(tokens[i]);
  }

  // Verify has_pending() is false after finalize.
  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

std::vector<iree_tokenizer_token_id_t> EncodeAndFinalize(
    iree_tokenizer_model_t* model, const std::string& text,
    bool expect_pending_after_encode) {
  // Create single segment covering entire text.
  std::vector<iree_tokenizer_segment_t> segments = {
      {0, static_cast<iree_host_size_t>(text.size())}};
  return EncodeAndFinalize(model, text, segments, expect_pending_after_encode);
}

void TestEncode(iree_tokenizer_model_t* model, const std::string& text,
                const std::vector<iree_tokenizer_segment_t>& segments,
                const std::vector<iree_tokenizer_token_id_t>& expected_tokens,
                bool expect_pending_after_encode) {
  auto actual =
      EncodeAndFinalize(model, text, segments, expect_pending_after_encode);
  EXPECT_EQ(actual, expected_tokens)
      << "Encoding \"" << text << "\" produced unexpected tokens";
}

void TestEncode(iree_tokenizer_model_t* model, const std::string& text,
                const std::vector<iree_tokenizer_token_id_t>& expected_tokens,
                bool expect_pending_after_encode) {
  std::vector<iree_tokenizer_segment_t> segments = {
      {0, static_cast<iree_host_size_t>(text.size())}};
  TestEncode(model, text, segments, expected_tokens,
             expect_pending_after_encode);
}

void TestMultipleEncodeCalls(
    iree_tokenizer_model_t* model, const std::string& text1,
    const std::vector<iree_tokenizer_segment_t>& segments1,
    const std::vector<iree_tokenizer_token_id_t>& expected1,
    const std::string& text2,
    const std::vector<iree_tokenizer_segment_t>& segments2,
    const std::vector<iree_tokenizer_token_id_t>& expected2) {
  ScopedModelState state(model);

  iree_tokenizer_token_id_t tokens[kMaxTokens];
  iree_host_size_t segments_consumed = 0;
  iree_host_size_t token_count = 0;

  // First encode call.
  iree_const_byte_span_t buffer1 = {
      reinterpret_cast<const uint8_t*>(text1.data()), text1.size()};
  IREE_EXPECT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer1,
      iree_tokenizer_make_segment_list(segments1.data(), segments1.size()),
      iree_tokenizer_make_token_output(tokens, NULL, NULL, kMaxTokens),
      &segments_consumed, &token_count));

  std::vector<iree_tokenizer_token_id_t> result1(tokens, tokens + token_count);
  EXPECT_EQ(result1, expected1) << "First encode call failed";

  // Second encode call with independent segment list.
  iree_const_byte_span_t buffer2 = {
      reinterpret_cast<const uint8_t*>(text2.data()), text2.size()};
  IREE_EXPECT_OK(iree_tokenizer_model_state_encode(
      state.get(), buffer2,
      iree_tokenizer_make_segment_list(segments2.data(), segments2.size()),
      iree_tokenizer_make_token_output(tokens, NULL, NULL, kMaxTokens),
      &segments_consumed, &token_count));

  std::vector<iree_tokenizer_token_id_t> result2(tokens, tokens + token_count);
  EXPECT_EQ(result2, expected2) << "Second encode call failed";

  // Finalize and verify no pending.
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_model_state_finalize(
      state.get(),
      iree_tokenizer_make_token_output(tokens, NULL, NULL, kMaxTokens),
      &finalize_count));

  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";
}

void TestLimitedOutputCapacity(
    iree_tokenizer_model_t* model, const std::string& text,
    const std::vector<iree_tokenizer_segment_t>& segments,
    const std::vector<iree_tokenizer_token_id_t>& expected_tokens) {
  ScopedModelState state(model);

  iree_const_byte_span_t buffer = {
      reinterpret_cast<const uint8_t*>(text.data()), text.size()};

  std::vector<iree_tokenizer_token_id_t> result;
  iree_tokenizer_token_id_t token;

  // Process with capacity=1, one token at a time.
  size_t segment_index = 0;
  while (segment_index < segments.size()) {
    // Create segment list from remaining segments.
    std::vector<iree_tokenizer_segment_t> remaining_segments(
        segments.begin() + segment_index, segments.end());

    iree_host_size_t segments_consumed = 0;
    iree_host_size_t token_count = 0;

    IREE_EXPECT_OK(iree_tokenizer_model_state_encode(
        state.get(), buffer,
        iree_tokenizer_make_segment_list(remaining_segments.data(),
                                         remaining_segments.size()),
        iree_tokenizer_make_token_output(&token, NULL, NULL, 1),
        &segments_consumed, &token_count));

    if (token_count > 0) {
      result.push_back(token);
    }

    // If we consumed segments, advance.
    if (segments_consumed > 0) {
      segment_index += segments_consumed;
    } else if (token_count == 0) {
      // No progress - should not happen.
      ADD_FAILURE() << "No progress with capacity=1 at segment "
                    << segment_index;
      break;
    }
    // If token_count > 0 but segments_consumed == 0, we're mid-segment.
  }

  // Finalize with capacity=1.
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_model_state_finalize(
      state.get(), iree_tokenizer_make_token_output(&token, NULL, NULL, 1),
      &finalize_count));

  if (finalize_count > 0) {
    result.push_back(token);
  }

  EXPECT_EQ(result, expected_tokens)
      << "Limited capacity (1) produced different tokens";

  EXPECT_FALSE(iree_tokenizer_model_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";
}

}  // namespace iree::tokenizer::testing
