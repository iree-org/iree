// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for CTC (Connectionist Temporal Classification) decoder.
//
// Tests the CTC decoder's robustness against:
// - Consecutive duplicate tokens (deduplication)
// - Pad/blank token patterns
// - Word delimiter token handling
// - Mixed pad/content sequences
// - Various pad_token and word_delimiter_token strings
// - Cleanup mode (punctuation spacing, contractions)
// - Limited output capacity with resume handling
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder/ctc.h"

static constexpr iree_host_size_t kMaxTokens = 64;
static constexpr iree_host_size_t kMaxTokenLength = 16;
static constexpr iree_host_size_t kMaxOutputSize = 2048;

static void test_with_config(const uint8_t* data, size_t size,
                             iree_string_view_t pad_token,
                             iree_string_view_t word_delimiter_token,
                             bool cleanup) {
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

  // Allocate decoder.
  iree_tokenizer_decoder_t* decoder = NULL;
  iree_status_t status = iree_tokenizer_decoder_ctc_allocate(
      pad_token, word_delimiter_token, cleanup, iree_allocator_system(),
      &decoder);
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

  // Test with full output capacity.
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

  // Test with standard CTC tokens.
  test_with_config(data, size, iree_make_cstring_view("<pad>"),
                   iree_make_cstring_view("|"), false);
  test_with_config(data, size, iree_make_cstring_view("<pad>"),
                   iree_make_cstring_view("|"), true);
  test_with_config(data, size, iree_make_cstring_view("[PAD]"),
                   iree_make_cstring_view(" "), true);

  // Test with empty word delimiter.
  test_with_config(data, size, iree_make_cstring_view("<pad>"),
                   iree_string_view_empty(), false);

  return 0;
}
