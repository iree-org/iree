// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Unigram (SentencePiece) model encoding.
//
// Tests the Viterbi dynamic programming against:
// - Score-based token selection (varied float scores)
// - Varied vocabulary sizes (from minimal to large)
// - Edge cases in Viterbi path finding (unreachable segments, ties)
// - UNK handling and fusion
// - Byte fallback (<0xXX> tokens)
// - Different flag combinations
// - Segments larger than max_token_length (chunked Viterbi)
//
// The fuzzer builds a Unigram model from fuzzed vocab/score data, then encodes
// fuzzed input text through it. This tests both model construction and the
// Viterbi encode path with adversarial configurations.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cmath>
#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/model/unigram.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

// Maximum limits to prevent OOM during fuzzing.
static constexpr iree_host_size_t kMaxTokens = 128;
static constexpr iree_host_size_t kMaxTokenLength = 32;
static constexpr iree_host_size_t kMaxInputLength = 512;

// Parses a single byte as token length, reading up to |max_length| bytes.
// Returns bytes consumed (1 + actual_length).
static iree_host_size_t parse_token(const uint8_t* data, iree_host_size_t size,
                                    iree_host_size_t max_length,
                                    iree_string_view_t* out_token) {
  if (size == 0) {
    *out_token = iree_string_view_empty();
    return 0;
  }
  iree_host_size_t length = data[0];
  if (length > max_length) length = max_length;
  if (length > size - 1) length = size - 1;
  *out_token =
      iree_make_string_view(reinterpret_cast<const char*>(data + 1), length);
  return 1 + length;
}

// Parses a 32-bit float from 4 bytes. Sanitizes NaN/Inf to finite values.
static float parse_score(const uint8_t* data, iree_host_size_t size) {
  if (size < 4) return -1.0f;
  uint32_t bits = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
                  ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
  float value;
  memcpy(&value, &bits, sizeof(value));
  // Sanitize: NaN/Inf -> finite negative value.
  if (!std::isfinite(value)) {
    value = -10.0f;
  }
  // Clamp to reasonable range for scores.
  if (value > 0.0f) value = 0.0f;
  if (value < -100.0f) value = -100.0f;
  return value;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Need at least: flags (1) + token_count (1) + unk_id (1) + input_len (1)
  if (size < 4) return 0;

  iree_host_size_t pos = 0;

  //===--------------------------------------------------------------------===//
  // Phase 1: Parse fuzz input structure
  //===--------------------------------------------------------------------===//

  // Byte 0: Unigram flags (masked to valid range).
  iree_tokenizer_unigram_flags_t flags = data[pos++] & 0x03;

  // Byte 1: Token count.
  iree_host_size_t token_count = data[pos++];
  if (token_count > kMaxTokens) token_count = kMaxTokens;
  if (token_count == 0) return 0;  // Need at least one token.

  // Byte 2: UNK token ID (0xFF = no UNK).
  uint8_t unk_id_byte = data[pos++];
  iree_tokenizer_token_id_t unk_token_id =
      (unk_id_byte == 0xFF) ? IREE_TOKENIZER_TOKEN_ID_INVALID
                            : (int32_t)(unk_id_byte % token_count);

  // Byte 3: Input text length.
  iree_host_size_t input_length = data[pos++];
  if (input_length > kMaxInputLength) input_length = kMaxInputLength;

  //===--------------------------------------------------------------------===//
  // Phase 2: Build vocabulary from fuzzed tokens with scores
  //===--------------------------------------------------------------------===//

  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_status_t status = iree_tokenizer_vocab_builder_allocate(
      token_count, iree_allocator_system(), &builder);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  // Parse and add tokens with scores.
  iree_host_size_t tokens_added = 0;
  float unk_score = -10.0f;

  for (iree_host_size_t i = 0; i < token_count && pos < size; ++i) {
    iree_string_view_t token_text;
    iree_host_size_t consumed =
        parse_token(data + pos, size - pos, kMaxTokenLength, &token_text);
    if (consumed == 0) break;
    pos += consumed;

    // Parse score (4 bytes).
    float score = parse_score(data + pos, size - pos);
    if (pos + 4 <= size) pos += 4;

    // Empty tokens are invalid, skip them.
    if (token_text.size == 0) continue;

    status = iree_tokenizer_vocab_builder_add_token(
        builder, token_text, score, IREE_TOKENIZER_TOKEN_ATTR_NONE);
    if (iree_status_is_ok(status)) {
      // Track UNK score if this is the UNK token.
      if ((int32_t)tokens_added == unk_token_id) {
        unk_score = score;
      }
      ++tokens_added;
    } else {
      // Duplicate token or other error - continue.
      iree_status_ignore(status);
    }
  }

  if (tokens_added == 0) {
    iree_tokenizer_vocab_builder_free(builder);
    return 0;
  }

  // Adjust unk_token_id if tokens were skipped.
  if (unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID &&
      unk_token_id >= (int32_t)tokens_added) {
    unk_token_id = IREE_TOKENIZER_TOKEN_ID_INVALID;
  }

  // Set UNK special token if valid.
  if (unk_token_id != IREE_TOKENIZER_TOKEN_ID_INVALID) {
    status = iree_tokenizer_vocab_builder_set_special_token(
        builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, unk_token_id);
    iree_status_ignore(status);
  }

  // Build the vocabulary.
  iree_tokenizer_vocab_t* vocab = NULL;
  status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 3: Create Unigram model
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_t* model = NULL;
  status = iree_tokenizer_unigram_model_allocate(
      vocab, unk_token_id, unk_score, flags, iree_allocator_system(), &model);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_free(vocab);
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 4: Encode fuzzed input text
  //===--------------------------------------------------------------------===//

  // Get remaining bytes as input text (up to input_length).
  iree_host_size_t available_input = size - pos;
  if (available_input > input_length) available_input = input_length;
  const char* input_data = reinterpret_cast<const char*>(data + pos);

  // Allocate state storage.
  iree_host_size_t state_size = iree_tokenizer_model_state_size(model);
  void* state_storage = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                 &state_storage);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_model_free(model);
    iree_tokenizer_vocab_free(vocab);
    iree_status_ignore(status);
    return 0;
  }

  iree_tokenizer_model_state_t* state = NULL;
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), state_storage);
    iree_tokenizer_model_free(model);
    iree_tokenizer_vocab_free(vocab);
    iree_status_ignore(status);
    return 0;
  }

  // Output buffer.
  iree_tokenizer_token_id_t tokens[256];
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 256);

  //===--------------------------------------------------------------------===//
  // Test 1: Single segment encoding
  //===--------------------------------------------------------------------===//

  if (available_input > 0) {
    iree_tokenizer_segment_t segment = {0, (uint32_t)available_input};
    iree_tokenizer_segment_list_t segments =
        iree_tokenizer_make_segment_list(&segment, 1);
    iree_const_byte_span_t transform_buffer = iree_make_const_byte_span(
        reinterpret_cast<const uint8_t*>(input_data), available_input);

    iree_host_size_t segments_consumed = 0;
    iree_host_size_t token_count = 0;
    status = iree_tokenizer_model_state_encode(
        state, transform_buffer, segments, output, &segments_consumed,
        &token_count);
    iree_status_ignore(status);

    // Finalize.
    iree_host_size_t final_count = 0;
    status = iree_tokenizer_model_state_finalize(state, output, &final_count);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Multiple small segments
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input >= 4) {
    // Create segments of varying sizes.
    iree_tokenizer_segment_t segments_array[32];
    iree_host_size_t segment_count = 0;
    iree_host_size_t offset = 0;

    while (offset < available_input && segment_count < 32) {
      iree_host_size_t seg_size = (segment_count % 4) + 1;  // 1-4 bytes.
      if (offset + seg_size > available_input) {
        seg_size = available_input - offset;
      }
      segments_array[segment_count].start = (uint32_t)offset;
      segments_array[segment_count].end = (uint32_t)(offset + seg_size);
      offset += seg_size;
      ++segment_count;
    }

    iree_tokenizer_segment_list_t segments =
        iree_tokenizer_make_segment_list(segments_array, segment_count);
    iree_const_byte_span_t transform_buffer = iree_make_const_byte_span(
        reinterpret_cast<const uint8_t*>(input_data), available_input);

    iree_host_size_t segments_consumed = 0;
    iree_host_size_t token_count = 0;
    status = iree_tokenizer_model_state_encode(
        state, transform_buffer, segments, output, &segments_consumed,
        &token_count);
    iree_status_ignore(status);

    iree_host_size_t final_count = 0;
    status = iree_tokenizer_model_state_finalize(state, output, &final_count);
    iree_status_ignore(status);
  } else {
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Limited output capacity
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input > 0) {
    // Output buffer of size 1.
    iree_tokenizer_token_id_t single_token[1];
    iree_tokenizer_token_output_t small_output =
        iree_tokenizer_make_token_output(single_token, NULL, NULL, 1);

    iree_tokenizer_segment_t segment = {0, (uint32_t)available_input};
    iree_tokenizer_segment_list_t segments =
        iree_tokenizer_make_segment_list(&segment, 1);
    iree_const_byte_span_t transform_buffer = iree_make_const_byte_span(
        reinterpret_cast<const uint8_t*>(input_data), available_input);

    // Keep encoding until done or hit iteration limit.
    iree_host_size_t total_segments_consumed = 0;
    for (int iter = 0; iter < 500 && total_segments_consumed < 1; ++iter) {
      iree_host_size_t segments_consumed = 0;
      iree_host_size_t token_count = 0;
      status = iree_tokenizer_model_state_encode(
          state, transform_buffer, segments, small_output, &segments_consumed,
          &token_count);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }
      total_segments_consumed += segments_consumed;
      if (segments_consumed == 0 && token_count == 0) {
        break;
      }
    }

    // Finalize with small output.
    for (int iter = 0;
         iter < 100 && iree_tokenizer_model_state_has_pending(state); ++iter) {
      iree_host_size_t final_count = 0;
      status = iree_tokenizer_model_state_finalize(state, small_output,
                                                   &final_count);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }
      if (final_count == 0) break;
    }
  } else {
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Cleanup
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  iree_allocator_free(iree_allocator_system(), state_storage);
  iree_tokenizer_model_free(model);
  iree_tokenizer_vocab_free(vocab);

  return 0;
}
