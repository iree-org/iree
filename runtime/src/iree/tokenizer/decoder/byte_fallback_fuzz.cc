// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for ByteFallback decoder.
//
// Tests the byte fallback decoder's robustness against:
// - Valid <0xHH> patterns reconstructing valid UTF-8
// - Invalid hex patterns (<0xGG>, <0x>, <0x1>)
// - Partial <0x patterns at token boundaries
// - Byte sequences that form invalid UTF-8 when decoded
// - Mixed passthrough tokens and byte tokens
// - Multi-token sequences with pending UTF-8 state
// - Limited output capacity with resume handling
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>

#include "iree/base/api.h"
#include "iree/tokenizer/decoder/byte_fallback.h"

// Maximum limits to prevent OOM during fuzzing.
static constexpr iree_host_size_t kMaxTokens = 64;
static constexpr iree_host_size_t kMaxTokenLength = 32;
static constexpr iree_host_size_t kMaxOutputSize = 2048;

// Global decoder instance built once at fuzzer startup.
static iree_tokenizer_decoder_t* g_decoder = NULL;

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  (void)argc;
  (void)argv;
  iree_status_t status = iree_tokenizer_decoder_byte_fallback_allocate(
      iree_allocator_system(), &g_decoder);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_decoder == NULL || size < 2) return 0;

  iree_host_size_t pos = 0;

  //===--------------------------------------------------------------------===//
  // Phase 1: Parse token list from fuzz input
  //===--------------------------------------------------------------------===//

  // Byte 0: Number of tokens.
  iree_host_size_t token_count = data[pos++];
  if (token_count > kMaxTokens) token_count = kMaxTokens;
  if (token_count == 0) return 0;

  // Parse tokens: [len:1][text:len]...
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

  if (actual_token_count == 0) return 0;

  //===--------------------------------------------------------------------===//
  // Phase 2: Allocate state and output buffer
  //===--------------------------------------------------------------------===//

  iree_host_size_t state_size = iree_tokenizer_decoder_state_size(g_decoder);
  void* state_storage = NULL;
  iree_status_t status = iree_allocator_malloc(iree_allocator_system(),
                                               state_size, &state_storage);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  char output_buffer[kMaxOutputSize];

  //===--------------------------------------------------------------------===//
  // Test 1: Full output capacity
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status = iree_tokenizer_decoder_state_initialize(g_decoder, state_storage,
                                                     &state);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), state_storage);
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

    // Finalize.
    iree_host_size_t final_written = 0;
    status =
        iree_tokenizer_decoder_state_finalize(state, output, &final_written);
    iree_status_ignore(status);

    iree_tokenizer_decoder_state_deinitialize(state);
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Limited output capacity (stress resume handling)
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status = iree_tokenizer_decoder_state_initialize(g_decoder, state_storage,
                                                     &state);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), state_storage);
      iree_status_ignore(status);
      return 0;
    }

    // Use first data byte (mod 8) + 1 as output capacity (1-8 bytes).
    iree_host_size_t small_capacity = (data[0] % 8) + 1;
    iree_mutable_string_view_t small_output = {output_buffer, small_capacity};

    iree_tokenizer_string_list_t remaining = {actual_token_count, tokens};
    iree_host_size_t total_consumed = 0;

    // Keep processing until all tokens consumed or hit iteration limit.
    for (int iter = 0; iter < 1000 && total_consumed < actual_token_count;
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

      // If no progress and still have tokens, output buffer is full.
      if (strings_consumed == 0 && bytes_written == 0 && remaining.count > 0) {
        continue;
      }
    }

    // Finalize with limited capacity.
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
  }

  //===--------------------------------------------------------------------===//
  // Test 3: One token at a time (stress cross-token pending state)
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_decoder_state_t* state = NULL;
    status = iree_tokenizer_decoder_state_initialize(g_decoder, state_storage,
                                                     &state);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), state_storage);
      iree_status_ignore(status);
      return 0;
    }

    iree_mutable_string_view_t output = {output_buffer, kMaxOutputSize};

    // Process each token individually.
    for (iree_host_size_t i = 0; i < actual_token_count; ++i) {
      iree_tokenizer_string_list_t single = {1, &tokens[i]};
      iree_host_size_t strings_consumed = 0;
      iree_host_size_t bytes_written = 0;
      status = iree_tokenizer_decoder_state_process(
          state, single, output, &strings_consumed, &bytes_written);
      iree_status_ignore(status);
    }

    // Finalize.
    iree_host_size_t final_written = 0;
    status =
        iree_tokenizer_decoder_state_finalize(state, output, &final_written);
    iree_status_ignore(status);

    iree_tokenizer_decoder_state_deinitialize(state);
  }

  //===--------------------------------------------------------------------===//
  // Cleanup
  //===--------------------------------------------------------------------===//

  iree_allocator_free(iree_allocator_system(), state_storage);
  return 0;
}
