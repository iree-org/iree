// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Metaspace decoder.
//
// Tests the metaspace-to-space replacement against:
// - Various replacement codepoints (default U+2581, custom codepoints)
// - All prepend_scheme modes (ALWAYS, NEVER, FIRST)
// - Metaspace at token boundaries (start, middle, end)
// - Multiple consecutive metaspaces
// - Partial metaspace sequences (incomplete UTF-8)
// - Mixed metaspace and regular content
// - Limited output capacity
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder/metaspace.h"

// Maximum limits to prevent OOM during fuzzing.
static constexpr iree_host_size_t kMaxTokens = 64;
static constexpr iree_host_size_t kMaxTokenLength = 255;
static constexpr iree_host_size_t kMaxOutputSize = 2048;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) return 0;

  iree_host_size_t pos = 0;

  //===--------------------------------------------------------------------===//
  // Phase 1: Parse decoder configuration
  //===--------------------------------------------------------------------===//

  // Byte 0: Prepend scheme (0-2).
  iree_tokenizer_decoder_metaspace_prepend_scheme_t prepend_scheme =
      (iree_tokenizer_decoder_metaspace_prepend_scheme_t)(data[pos++] % 3);

  // Bytes 1-2: Replacement codepoint (0 = default U+2581).
  uint32_t replacement_codepoint =
      (uint32_t)data[pos] | ((uint32_t)data[pos + 1] << 8);
  pos += 2;

  // Sanitize codepoint: 0 means default, otherwise clamp to valid Unicode.
  if (replacement_codepoint > 0x10FFFF) {
    replacement_codepoint = replacement_codepoint % 0x10FFFF;
  }
  // Avoid surrogates (0xD800-0xDFFF).
  if (replacement_codepoint >= 0xD800 && replacement_codepoint <= 0xDFFF) {
    replacement_codepoint = 0;  // Use default.
  }

  // Byte 3: Token count.
  iree_host_size_t token_count = data[pos++];
  if (token_count > kMaxTokens) token_count = kMaxTokens;
  if (token_count == 0) return 0;

  //===--------------------------------------------------------------------===//
  // Phase 2: Create decoder
  //===--------------------------------------------------------------------===//

  iree_tokenizer_decoder_t* decoder = NULL;
  iree_status_t status = iree_tokenizer_decoder_metaspace_allocate(
      replacement_codepoint, prepend_scheme, iree_allocator_system(), &decoder);
  if (!iree_status_is_ok(status)) {
    // Invalid codepoint - expected for some fuzz inputs.
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 3: Parse token list
  //===--------------------------------------------------------------------===//

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

  if (actual_token_count == 0) {
    iree_tokenizer_decoder_free(decoder);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Phase 4: Allocate state and output buffer
  //===--------------------------------------------------------------------===//

  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(decoder);
  void* state_storage = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                 &state_storage);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_decoder_free(decoder);
    iree_status_ignore(status);
    return 0;
  }

  char output_buffer[kMaxOutputSize];

  //===--------------------------------------------------------------------===//
  // Test 1: Full output capacity
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status =
        iree_tokenizer_decoder_state_initialize(decoder, state_storage, &state);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), state_storage);
      iree_tokenizer_decoder_free(decoder);
      iree_status_ignore(status);
      return 0;
    }

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
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Limited output capacity
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status =
        iree_tokenizer_decoder_state_initialize(decoder, state_storage, &state);
    if (iree_status_is_ok(status)) {
      // Use varying small capacity based on fuzz input.
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

      iree_tokenizer_decoder_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 3: One token at a time
  //===--------------------------------------------------------------------===//

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

      iree_tokenizer_decoder_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }
  }

  //===--------------------------------------------------------------------===//
  // Cleanup
  //===--------------------------------------------------------------------===//

  iree_allocator_free(iree_allocator_system(), state_storage);
  iree_tokenizer_decoder_free(decoder);

  return 0;
}
