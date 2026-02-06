// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for BPE model encoding with adversarial inputs.
//
// Tests the BPE model against:
// - Varied merge patterns (deep chains, wide fanout, cycles)
// - Pathological merge sequences (overlapping, conflicting priorities)
// - Tokens at chunk boundaries (partial segment processing)
// - Invalid and edge-case token IDs in merge rules
// - Very long tokens and very short segments
// - All BPE flag combinations
//
// The fuzzer builds a BPE model from fuzzed vocab/merge data, then encodes
// fuzzed input text through it. This tests both model construction and the
// encode path with adversarial configurations.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

// Maximum limits to prevent OOM during fuzzing.
static constexpr iree_host_size_t kMaxTokens = 256;
static constexpr iree_host_size_t kMaxMerges = 512;
static constexpr iree_host_size_t kMaxTokenLength = 64;
static constexpr iree_host_size_t kMaxInputLength = 4096;

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

// Parses a 16-bit little-endian value. Returns 2 if successful, 0 if not
// enough data.
static iree_host_size_t parse_u16(const uint8_t* data, iree_host_size_t size,
                                  uint16_t* out_value) {
  if (size < 2) return 0;
  *out_value = (uint16_t)data[0] | ((uint16_t)data[1] << 8);
  return 2;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Need at least: flags (1) + token_count (1) + merge_count (1) + input_len
  // (2)
  if (size < 5) return 0;

  iree_host_size_t pos = 0;

  //===--------------------------------------------------------------------===//
  // Phase 1: Parse fuzz input structure
  //===--------------------------------------------------------------------===//

  // Byte 0: BPE flags (masked to valid range).
  iree_tokenizer_bpe_flags_t flags = data[pos++] & 0x1F;

  // Byte 1: Token count.
  iree_host_size_t token_count = data[pos++];
  if (token_count > kMaxTokens) token_count = kMaxTokens;
  if (token_count == 0) return 0;  // Need at least one token.

  // Byte 2: Merge count.
  iree_host_size_t merge_count = data[pos++];
  if (merge_count > kMaxMerges) merge_count = kMaxMerges;

  // Bytes 3-4: Input text length.
  uint16_t input_length = 0;
  iree_host_size_t consumed = parse_u16(data + pos, size - pos, &input_length);
  if (consumed == 0) return 0;
  pos += consumed;
  if (input_length > kMaxInputLength) input_length = kMaxInputLength;

  //===--------------------------------------------------------------------===//
  // Phase 2: Build vocabulary from fuzzed tokens
  //===--------------------------------------------------------------------===//

  iree_tokenizer_vocab_builder_t* builder = NULL;
  iree_status_t status = iree_tokenizer_vocab_builder_allocate(
      token_count, iree_allocator_system(), &builder);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  // Parse and add tokens.
  iree_host_size_t tokens_added = 0;
  for (iree_host_size_t i = 0; i < token_count && pos < size; ++i) {
    iree_string_view_t token_text;
    consumed =
        parse_token(data + pos, size - pos, kMaxTokenLength, &token_text);
    if (consumed == 0) break;
    pos += consumed;

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

  // Parse and add merge rules.
  // Each merge is 4 bytes: left_id (2) + right_id (2).
  for (iree_host_size_t i = 0; i < merge_count && pos + 4 <= size; ++i) {
    uint16_t left_id = 0, right_id = 0;
    consumed = parse_u16(data + pos, size - pos, &left_id);
    pos += consumed;
    consumed = parse_u16(data + pos, size - pos, &right_id);
    pos += consumed;

    // Clamp IDs to valid token range.
    left_id = left_id % tokens_added;
    right_id = right_id % tokens_added;

    status = iree_tokenizer_vocab_builder_add_merge(builder, left_id, right_id);
    // Ignore errors (duplicate merges, etc.).
    iree_status_ignore(status);
  }

  // Set UNK token if we have tokens.
  status = iree_tokenizer_vocab_builder_set_special_token(
      builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0);
  iree_status_ignore(status);

  // Build the vocabulary.
  iree_tokenizer_vocab_t* vocab = NULL;
  status = iree_tokenizer_vocab_builder_build(builder, &vocab);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 3: Create BPE model
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_t* model = NULL;
  status = iree_tokenizer_bpe_model_allocate(vocab, flags,
                                             iree_allocator_system(), &model);
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
  iree_tokenizer_token_id_t tokens[512];
  iree_tokenizer_token_output_t output =
      iree_tokenizer_make_token_output(tokens, NULL, NULL, 512);

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
  // Test 2: Multiple small segments (chunk boundary stress)
  //===--------------------------------------------------------------------===//

  // Reinitialize state.
  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input >= 4) {
    // Create segments of varying sizes: 1, 2, 1, 2, ...
    iree_tokenizer_segment_t segments_array[64];
    iree_host_size_t segment_count = 0;
    iree_host_size_t offset = 0;

    while (offset < available_input && segment_count < 64) {
      iree_host_size_t seg_size = (segment_count % 2 == 0) ? 1 : 2;
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
  // Test 3: Partial segment processing (streaming simulation)
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input >= 8) {
    // Simulate partial segment: first half, then second half.
    iree_const_byte_span_t transform_buffer = iree_make_const_byte_span(
        reinterpret_cast<const uint8_t*>(input_data), available_input);

    // First half as partial segment.
    iree_host_size_t half = available_input / 2;
    iree_tokenizer_segment_t seg1 = {0, (uint32_t)half};
    iree_tokenizer_segment_list_t segments1 = {1, &seg1, true};  // Partial.

    iree_host_size_t segments_consumed = 0;
    iree_host_size_t token_count = 0;
    status = iree_tokenizer_model_state_encode(
        state, transform_buffer, segments1, output, &segments_consumed,
        &token_count);
    iree_status_ignore(status);

    // Reclaim what we can.
    (void)iree_tokenizer_model_state_reclaim(state);

    // Second half (complete).
    iree_tokenizer_segment_t seg2 = {(uint32_t)half, (uint32_t)available_input};
    iree_tokenizer_segment_list_t segments2 = {1, &seg2, false};

    status = iree_tokenizer_model_state_encode(
        state, transform_buffer, segments2, output, &segments_consumed,
        &token_count);
    iree_status_ignore(status);

    iree_host_size_t final_count = 0;
    status = iree_tokenizer_model_state_finalize(state, output, &final_count);
    iree_status_ignore(status);
  } else {
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Test 4: Limited output capacity (stress output buffering)
  //===--------------------------------------------------------------------===//

  iree_tokenizer_model_state_deinitialize(state);
  status = iree_tokenizer_model_state_initialize(model, state_storage, &state);
  if (iree_status_is_ok(status) && available_input > 0) {
    // Output buffer of size 1 to force multiple encode calls.
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
    for (int iter = 0; iter < 1000 && total_segments_consumed < 1; ++iter) {
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
        // No progress.
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
