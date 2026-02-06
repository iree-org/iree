// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for WordPiece model encoding.
//
// Tests the greedy longest-match tokenization against:
// - Continuation prefix handling (varied prefixes, including empty)
// - max_input_chars_per_word limits (words exceeding limit -> [UNK])
// - Unknown token fallback paths (unmatchable substrings)
// - Pre-computation semantics (partial word failure -> whole word [UNK])
// - Varied vocabulary sizes with mixed prefixed/unprefixed tokens
// - Limited output capacity with pending token resume
//
// The fuzzer builds a WordPiece model from fuzzed vocab data, then encodes
// fuzzed input text through it. This tests both model construction and the
// greedy matching encode path with adversarial configurations.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/model/wordpiece.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

// Maximum limits to prevent OOM during fuzzing.
static constexpr iree_host_size_t kMaxTokens = 128;
static constexpr iree_host_size_t kMaxTokenLength = 32;
static constexpr iree_host_size_t kMaxPrefixLength = 16;
static constexpr iree_host_size_t kMaxInputLength = 512;
static constexpr iree_host_size_t kMaxInputCharsPerWord = 200;

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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Need at least: prefix_len (1) + max_chars (1) + token_count (1) +
  //                unk_index (1) + input_len (1) = 5 bytes
  if (size < 5) return 0;

  iree_host_size_t position = 0;

  //===--------------------------------------------------------------------===//
  // Phase 1: Parse fuzz input structure
  //===--------------------------------------------------------------------===//

  // Byte 0: Prefix length (0-16).
  iree_host_size_t prefix_length = data[position++] % (kMaxPrefixLength + 1);

  // Byte 1: max_input_chars_per_word (1-200, 0 treated as 1).
  iree_host_size_t max_input_chars_per_word = data[position++];
  if (max_input_chars_per_word == 0) max_input_chars_per_word = 1;
  if (max_input_chars_per_word > kMaxInputCharsPerWord) {
    max_input_chars_per_word = kMaxInputCharsPerWord;
  }

  // Byte 2: Token count (need at least 1 for UNK).
  iree_host_size_t token_count = data[position++];
  if (token_count > kMaxTokens) token_count = kMaxTokens;
  if (token_count == 0) return 0;

  // Byte 3: UNK token index within vocab.
  uint8_t unk_index_byte = data[position++];
  iree_host_size_t unk_index = unk_index_byte % token_count;

  // Byte 4: Input text length.
  iree_host_size_t input_length = data[position++];
  if (input_length > kMaxInputLength) input_length = kMaxInputLength;

  //===--------------------------------------------------------------------===//
  // Phase 2: Parse continuation prefix
  //===--------------------------------------------------------------------===//

  char prefix_buffer[kMaxPrefixLength];
  iree_host_size_t actual_prefix_length = prefix_length;
  if (position + prefix_length > size) {
    actual_prefix_length = size - position;
  }
  if (actual_prefix_length > 0) {
    memcpy(prefix_buffer, data + position, actual_prefix_length);
    position += actual_prefix_length;
  }
  iree_string_view_t continuing_subword_prefix =
      iree_make_string_view(prefix_buffer, actual_prefix_length);

  //===--------------------------------------------------------------------===//
  // Phase 3: Build vocabulary from fuzzed tokens
  //===--------------------------------------------------------------------===//

  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_status_t status = iree_tokenizer_vocab_builder_allocate(
      token_count, iree_allocator_system(), &builder);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  // Parse and add tokens.
  // Some tokens will naturally have the prefix (if present in fuzz data),
  // simulating the "##word" vs "word" distinction.
  iree_host_size_t tokens_added = 0;
  for (iree_host_size_t i = 0; i < token_count && position < size; ++i) {
    iree_string_view_t token_text;
    iree_host_size_t consumed = parse_token(data + position, size - position,
                                            kMaxTokenLength, &token_text);
    if (consumed == 0) break;
    position += consumed;

    // Empty tokens are invalid, skip them.
    if (token_text.size == 0) continue;

    status = iree_tokenizer_vocab_builder_add_token(
        builder, token_text, 0.0f, IREE_TOKENIZER_TOKEN_ATTR_NONE);
    if (iree_status_is_ok(status)) {
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

  // Adjust unk_index if tokens were skipped.
  if (unk_index >= tokens_added) {
    unk_index = 0;
  }

  // Set UNK special token (required for WordPiece).
  status = iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK,
      (iree_tokenizer_token_id_t)unk_index);
  iree_status_ignore(status);

  // Build the vocabulary.
  iree_tokenizer_vocab_t* vocab = NULL;
  status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 4: Create WordPiece model
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_t* model = NULL;
  status = iree_tokenizer_wordpiece_model_allocate(
      vocab, continuing_subword_prefix, max_input_chars_per_word,
      IREE_TOKENIZER_WORDPIECE_FLAG_NONE, iree_allocator_system(), &model);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_vocab_free(vocab);
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 5: Encode fuzzed input text
  //===--------------------------------------------------------------------===//

  // Get remaining bytes as input text (up to input_length).
  iree_host_size_t available_input = size - position;
  if (available_input > input_length) available_input = input_length;
  const char* input_data = reinterpret_cast<const char*>(data + position);

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
  // Test 1: Single segment encoding (entire input as one word)
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
  // Test 2: Multiple small segments (simulates word boundaries)
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input >= 2) {
    // Create segments of varying sizes (1-4 bytes each).
    iree_tokenizer_segment_t segments_array[64];
    iree_host_size_t segment_count = 0;
    iree_host_size_t offset = 0;

    while (offset < available_input && segment_count < 64) {
      iree_host_size_t segment_size = (segment_count % 4) + 1;
      if (offset + segment_size > available_input) {
        segment_size = available_input - offset;
      }
      segments_array[segment_count].start = (uint32_t)offset;
      segments_array[segment_count].end = (uint32_t)(offset + segment_size);
      offset += segment_size;
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
  // Test 3: Limited output capacity (stress pending token resume)
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
    for (int iteration = 0; iteration < 500 && total_segments_consumed < 1;
         ++iteration) {
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

    // Finalize with small output (may need multiple calls).
    for (int iteration = 0;
         iteration < 100 && iree_tokenizer_model_state_has_pending(state);
         ++iteration) {
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
  // Test 4: Long word triggering max_input_chars_per_word
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input > 0) {
    // Use entire input as one segment - tests character counting and UNK path.
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
