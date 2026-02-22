// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for special token matching with adversarial inputs.
//
// Tests the special token matcher against:
// - Arbitrary byte sequences that may partially match tokens
// - Streaming continuation with varying chunk sizes
// - Rapid match/no-match alternation
// - Very long inputs
// - Inputs that stress the bucket lookup and prefix matching
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/special_tokens.h"

// Global special tokens collection built once at fuzzer startup.
static iree_tokenizer_special_tokens_t g_special_tokens;

// Build a special tokens collection with various patterns for fuzzing.
static iree_status_t build_test_special_tokens(void) {
  iree_tokenizer_special_tokens_builder_t builder;
  iree_tokenizer_special_tokens_builder_initialize(iree_allocator_system(),
                                                   &builder);

  // GPT-style tokens (start with '<').
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|endoftext|>"), 50256,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|startoftext|>"), 50257,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|pad|>"), 50258,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|im_start|>"), 50259,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<|im_end|>"), 50260,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  // BERT-style tokens (start with '[').
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[CLS]"), 101, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[SEP]"), 102, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[MASK]"), 103,
      IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[PAD]"), 0, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("[UNK]"), 100, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  // Llama-style tokens (also start with '<').
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<s>"), 1, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("</s>"), 2, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  // Short tokens that could cause greedy matching issues.
  IREE_RETURN_IF_ERROR(iree_tokenizer_special_tokens_builder_add(
      &builder, IREE_SV("<"), 200, IREE_TOKENIZER_SPECIAL_TOKEN_FLAG_NONE));

  iree_status_t status = iree_tokenizer_special_tokens_builder_build(
      &builder, iree_allocator_system(), &g_special_tokens);
  iree_tokenizer_special_tokens_builder_deinitialize(&builder);
  return status;
}

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  (void)argc;
  (void)argv;
  iree_status_t status = build_test_special_tokens();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size == 0) return 0;

  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  //===--------------------------------------------------------------------===//
  // Test 1: One-shot matching at each position
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_special_tokens_encode_state_t state;
    for (size_t i = 0; i < size; ++i) {
      iree_tokenizer_special_tokens_encode_state_initialize(&state);
      iree_host_size_t length = 0;
      iree_tokenizer_token_id_t id = 0;
      iree_string_view_t remaining =
          iree_make_string_view(input.data + i, size - i);
      iree_tokenizer_special_tokens_match(&g_special_tokens, remaining, &length,
                                          &id, &state);
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Streaming with byte-by-byte input
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_special_tokens_encode_state_t state;
    iree_tokenizer_special_tokens_encode_state_initialize(&state);

    size_t offset = 0;
    while (offset < size) {
      iree_host_size_t length = 0;
      iree_tokenizer_token_id_t id = 0;
      iree_string_view_t chunk = iree_make_string_view(input.data + offset, 1);

      auto result = iree_tokenizer_special_tokens_match(
          &g_special_tokens, chunk, &length, &id, &state);

      if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
        offset += length;
        // State is cleared on match, start fresh.
      } else if (result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
        offset += 1;  // Byte consumed, tracked in state.
      } else {
        // NO_MATCH: if we had a partial, drain it.
        if (state.match_position > 0) {
          uint8_t partial_buffer[256];
          iree_tokenizer_special_tokens_encode_state_get_partial(
              &state, &g_special_tokens, partial_buffer);
          iree_tokenizer_special_tokens_encode_state_clear_partial(&state);
          // Don't advance offset - retry this byte fresh.
        } else {
          offset += 1;  // Skip this byte.
        }
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Streaming with varying chunk sizes (fuzz-derived)
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_special_tokens_encode_state_t state;
    iree_tokenizer_special_tokens_encode_state_initialize(&state);

    size_t offset = 0;
    size_t iteration = 0;
    while (offset < size) {
      // Use input bytes to determine chunk size (1-16 bytes).
      size_t chunk_size = (data[offset % size] % 16) + 1;
      if (chunk_size > size - offset) chunk_size = size - offset;

      iree_host_size_t length = 0;
      iree_tokenizer_token_id_t id = 0;
      iree_string_view_t chunk =
          iree_make_string_view(input.data + offset, chunk_size);

      auto result = iree_tokenizer_special_tokens_match(
          &g_special_tokens, chunk, &length, &id, &state);

      if (result == IREE_TOKENIZER_SPECIAL_TOKENS_MATCHED) {
        offset += length;
      } else if (result == IREE_TOKENIZER_SPECIAL_TOKENS_NEED_MORE) {
        offset += chunk_size;
      } else {
        if (state.match_position > 0) {
          uint8_t partial_buffer[256];
          iree_tokenizer_special_tokens_encode_state_get_partial(
              &state, &g_special_tokens, partial_buffer);
          iree_tokenizer_special_tokens_encode_state_clear_partial(&state);
        } else {
          offset += 1;
        }
      }

      // Prevent infinite loops.
      if (++iteration > size * 2) break;
    }
  }

  //===--------------------------------------------------------------------===//
  // Test 4: safe_prefix_length scanning
  //===--------------------------------------------------------------------===//

  {
    for (size_t i = 0; i < size; ++i) {
      iree_string_view_t remaining =
          iree_make_string_view(input.data + i, size - i);
      iree_tokenizer_special_tokens_safe_prefix_length(&g_special_tokens,
                                                       remaining);
    }
  }

  return 0;
}
