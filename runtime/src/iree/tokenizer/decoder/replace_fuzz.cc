// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Replace decoder.
//
// Tests the replace decoder's robustness against:
// - Pattern/content replacement pairs
// - Multi-byte patterns (e.g., ▁ -> space)
// - Overlapping pattern occurrences
// - Pattern at token boundaries
// - Empty content (deletion)
// - Various token sequences
// - Limited output capacity
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder/replace.h"

static constexpr iree_host_size_t kMaxTokens = 64;
static constexpr iree_host_size_t kMaxTokenLength = 32;
static constexpr iree_host_size_t kMaxOutputSize = 2048;

static void test_with_pattern(const uint8_t* data, size_t size,
                              iree_string_view_t pattern,
                              iree_string_view_t content) {
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
  iree_status_t status = iree_tokenizer_decoder_replace_allocate(
      pattern, content, iree_allocator_system(), &decoder);
  if (!iree_status_is_ok(status)) {
    // Pattern validation may fail (empty pattern, expanding replacement).
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
  if (size < 4) return 0;

  // Test with standard metaspace replacement (▁ -> space).
  // ▁ = U+2581 = E2 96 81 in UTF-8.
  test_with_pattern(data, size, iree_make_cstring_view("\xE2\x96\x81"),
                    iree_make_cstring_view(" "));

  // Test with fuzz-derived pattern/content.
  size_t pattern_len = (data[0] % 4) + 1;  // 1-4 bytes.
  if (pattern_len > size - 1) return 0;
  size_t content_len = data[1] % pattern_len;  // Must be <= pattern_len.

  iree_string_view_t pattern = iree_make_string_view(
      reinterpret_cast<const char*>(data + 2), pattern_len);
  iree_string_view_t content = iree_make_string_view(
      reinterpret_cast<const char*>(data + 2), content_len);

  if (2 + pattern_len < size) {
    test_with_pattern(data + 2 + pattern_len, size - 2 - pattern_len, pattern,
                      content);
  }

  return 0;
}
