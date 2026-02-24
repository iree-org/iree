// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for WordPiece decoder.
//
// Tests the WordPiece decoder's robustness against:
// - Tokens with ## prefix (continuation subwords)
// - Tokens without prefix (word starts)
// - Empty tokens and prefix-only tokens ("##")
// - Various prefix patterns
// - Multi-token sequences with mixed prefixed/unprefixed tokens
// - Cleanup mode (punctuation spacing, contractions)
// - Limited output capacity with resume handling
// - Output expansion from space prepending
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder/wordpiece.h"

// Maximum limits to prevent OOM during fuzzing.
static constexpr iree_host_size_t kMaxTokens = 64;
static constexpr iree_host_size_t kMaxTokenLength = 32;
static constexpr iree_host_size_t kMaxOutputSize = 2048;

// Test with a specific configuration.
static void test_with_config(const uint8_t* data, size_t size,
                             iree_string_view_t prefix, bool cleanup) {
  if (size < 2) return;

  iree_host_size_t pos = 0;

  // Parse token list from fuzz input.
  iree_host_size_t token_count = data[pos++];
  if (token_count > kMaxTokens) token_count = kMaxTokens;
  if (token_count == 0) return;

  iree_string_view_t tokens[kMaxTokens];
  iree_host_size_t actual_token_count = 0;

  for (iree_host_size_t i = 0; i < token_count && pos < size; ++i) {
    iree_host_size_t length = data[pos++];
    if (length > kMaxTokenLength) length = kMaxTokenLength;
    if (pos + length > size) length = size - pos;

    tokens[actual_token_count] = iree_make_string_view(
        reinterpret_cast<const char*>(data + pos), length);
    pos += length;
    ++actual_token_count;
  }

  if (actual_token_count == 0) return;

  // Allocate decoder.
  iree_tokenizer_decoder_wordpiece_config_t config = {prefix, cleanup};
  iree_tokenizer_decoder_t* decoder = NULL;
  iree_status_t status = iree_tokenizer_decoder_wordpiece_allocate(
      config, iree_allocator_system(), &decoder);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }

  // Allocate state.
  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder);
  void* state_storage = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                 &state_storage);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_tokenizer_decoder_free(decoder);
    return;
  }

  char output_buffer[kMaxOutputSize];

  // Test 1: Full output capacity.
  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status =
        iree_tokenizer_decoder_state_initialize(decoder, state_storage, &state);
    if (iree_status_is_ok(status)) {
      iree_tokenizer_string_list_t token_list = {actual_token_count, tokens};
      iree_mutable_string_view_t output = {output_buffer, kMaxOutputSize};

      iree_host_size_t strings_consumed = 0;
      iree_host_size_t bytes_written = 0;
      status = iree_tokenizer_decoder_state_process(
          state, token_list, output, &strings_consumed, &bytes_written);
      iree_status_ignore(status);

      iree_host_size_t final_written = 0;
      status =
          iree_tokenizer_decoder_state_finalize(state, output, &final_written);
      iree_status_ignore(status);

      iree_tokenizer_decoder_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }
  }

  // Test 2: Limited output capacity.
  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status =
        iree_tokenizer_decoder_state_initialize(decoder, state_storage, &state);
    if (iree_status_is_ok(status)) {
      iree_host_size_t small_capacity = (data[0] % 8) + 1;
      iree_mutable_string_view_t small_output = {output_buffer, small_capacity};

      iree_tokenizer_string_list_t remaining = {actual_token_count, tokens};
      iree_host_size_t total_consumed = 0;

      for (int iter = 0; iter < 500 && total_consumed < actual_token_count;
           ++iter) {
        iree_host_size_t strings_consumed = 0;
        iree_host_size_t bytes_written = 0;
        status = iree_tokenizer_decoder_state_process(
            state, remaining, small_output, &strings_consumed, &bytes_written);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }

        total_consumed += strings_consumed;
        remaining.values += strings_consumed;
        remaining.count -= strings_consumed;

        if (strings_consumed == 0 && bytes_written == 0 &&
            remaining.count > 0) {
          continue;
        }
      }

      for (int iter = 0;
           iter < 100 && iree_tokenizer_decoder_state_has_pending(state);
           ++iter) {
        iree_host_size_t final_written = 0;
        status = iree_tokenizer_decoder_state_finalize(state, small_output,
                                                       &final_written);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }
        if (final_written == 0) break;
      }

      iree_tokenizer_decoder_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }
  }

  // Test 3: One token at a time.
  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status =
        iree_tokenizer_decoder_state_initialize(decoder, state_storage, &state);
    if (iree_status_is_ok(status)) {
      iree_mutable_string_view_t output = {output_buffer, kMaxOutputSize};

      for (iree_host_size_t i = 0; i < actual_token_count; ++i) {
        iree_tokenizer_string_list_t single = {1, &tokens[i]};
        iree_host_size_t strings_consumed = 0;
        iree_host_size_t bytes_written = 0;
        status = iree_tokenizer_decoder_state_process(
            state, single, output, &strings_consumed, &bytes_written);
        iree_status_ignore(status);
      }

      iree_host_size_t final_written = 0;
      status =
          iree_tokenizer_decoder_state_finalize(state, output, &final_written);
      iree_status_ignore(status);

      iree_tokenizer_decoder_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }
  }

  iree_allocator_free(iree_allocator_system(), state_storage);
  iree_tokenizer_decoder_free(decoder);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return 0;

  // Test with standard "##" prefix.
  test_with_config(data, size, iree_make_cstring_view("##"), false);
  test_with_config(data, size, iree_make_cstring_view("##"), true);

  // Test with alternative prefixes based on fuzz input.
  if (size > 3) {
    char prefix_buf[4] = {0};
    prefix_buf[0] = (char)data[0];
    iree_string_view_t custom_prefix = iree_make_string_view(prefix_buf, 1);
    test_with_config(data + 1, size - 1, custom_prefix, data[0] & 1);
  }

  return 0;
}
