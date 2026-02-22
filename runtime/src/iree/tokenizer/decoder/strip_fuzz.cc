// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Strip decoder.
//
// Tests the strip decoder's robustness against:
// - Various content strings to strip
// - Different start_count values
// - Tokens that start with the strip content
// - Tokens that don't start with the strip content
// - Mixed sequences
// - Limited output capacity
//
// Note: stop_count > 0 requires buffering which is not yet supported.
// This fuzzer only tests start stripping (stop_count = 0).
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder/strip.h"

static constexpr iree_host_size_t kMaxTokens = 64;
static constexpr iree_host_size_t kMaxTokenLength = 32;
static constexpr iree_host_size_t kMaxOutputSize = 2048;

static void test_with_config(const uint8_t* data, size_t size,
                             iree_string_view_t content,
                             iree_host_size_t start_count) {
  if (size < 2) return;

  iree_host_size_t pos = 0;

  // Parse token list.
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

  // Allocate decoder (stop_count = 0 for streaming compatibility).
  iree_tokenizer_decoder_t* decoder = NULL;
  iree_status_t status = iree_tokenizer_decoder_strip_allocate(
      content, start_count, 0, iree_allocator_system(), &decoder);
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

  // Test with full capacity.
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

  // Test one token at a time.
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

  // Test with standard space stripping (HuggingFace default).
  test_with_config(data, size, iree_make_cstring_view(" "), 1);

  // Test with various start_count values.
  test_with_config(data, size, iree_make_cstring_view(" "), 0);
  test_with_config(data, size, iree_make_cstring_view(" "), 3);

  // Test with fuzz-derived content.
  if (size > 2) {
    size_t content_len = (data[0] % 4) + 1;  // 1-4 bytes.
    if (content_len <= 8 && content_len < size - 1) {
      iree_string_view_t content = iree_make_string_view(
          reinterpret_cast<const char*>(data + 1), content_len);
      iree_host_size_t start_count = data[0] % 5;
      test_with_config(data + 1 + content_len, size - 1 - content_len, content,
                       start_count);
    }
  }

  return 0;
}
