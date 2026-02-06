// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for tokenizer decoding with adversarial token sequences.
//
// Tests the decoding pipeline against:
// - Invalid token IDs (negative, out of range)
// - Very long token sequences
// - Sequences with many special tokens
// - Token ID patterns that stress decoder state
// - Rapid switching between valid/invalid IDs
//
// Uses fuzzing_util for tokenizer loading (dummy or real via --tokenizer_json).
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/testing/fuzzing_util.h"
#include "iree/tokenizer/tokenizer.h"

// Global tokenizer instance built once at fuzzer startup.
static iree_tokenizer_t* g_tokenizer = NULL;
static int32_t g_vocab_size = 0;

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  iree_status_t status = iree_tokenizer_fuzz_load_or_build(
      argc, argv, &g_tokenizer, &g_vocab_size);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_tokenizer == NULL || size < 4) return 0;

  // Interpret fuzz data as a sequence of token IDs.
  iree_host_size_t token_count = size / sizeof(int32_t);
  if (token_count == 0) return 0;
  if (token_count > 4096) token_count = 4096;  // Limit to avoid OOM.

  const int32_t* token_ids = reinterpret_cast<const int32_t*>(data);
  iree_tokenizer_token_id_list_t tokens =
      iree_tokenizer_make_token_id_list(token_ids, token_count);

  //===--------------------------------------------------------------------===//
  // Test 1: One-shot decode
  //===--------------------------------------------------------------------===//

  {
    char output[16384];
    iree_mutable_string_view_t text_output = {output, sizeof(output)};
    iree_host_size_t text_length = 0;
    iree_status_t status = iree_tokenizer_decode(
        g_tokenizer, tokens, IREE_TOKENIZER_DECODE_FLAG_NONE, text_output,
        iree_allocator_system(), &text_length);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Test 2: One-shot decode with skip_special_tokens
  //===--------------------------------------------------------------------===//

  {
    char output[16384];
    iree_mutable_string_view_t text_output = {output, sizeof(output)};
    iree_host_size_t text_length = 0;
    iree_status_t status = iree_tokenizer_decode(
        g_tokenizer, tokens, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        text_output, iree_allocator_system(), &text_length);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Streaming decode with varying chunk sizes
  //===--------------------------------------------------------------------===//

  {
    iree_host_size_t state_size = 0;
    iree_status_t status =
        iree_tokenizer_decode_state_calculate_size(g_tokenizer, &state_size);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return 0;
    }

    void* state_storage = NULL;
    status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                   &state_storage);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return 0;
    }

    iree_tokenizer_decode_state_t* state = NULL;
    status = iree_tokenizer_decode_state_initialize(
        g_tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
        iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage),
                            state_size),
        &state);

    if (iree_status_is_ok(status)) {
      char output[4096];
      iree_mutable_string_view_t text_output = {output, sizeof(output)};

      // Feed tokens in varying chunk sizes.
      iree_host_size_t offset = 0;
      while (offset < token_count) {
        iree_host_size_t chunk_count = ((offset % 4) + 1) * 2;
        if (chunk_count > token_count - offset)
          chunk_count = token_count - offset;

        iree_tokenizer_token_id_list_t chunk =
            iree_tokenizer_make_token_id_list(token_ids + offset, chunk_count);

        iree_host_size_t tokens_consumed = 0;
        iree_host_size_t text_length = 0;
        status = iree_tokenizer_decode_state_feed(
            state, chunk, text_output, &tokens_consumed, &text_length);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }
        offset += tokens_consumed;
        if (tokens_consumed == 0 && text_length == 0) {
          break;  // No progress.
        }
      }

      // Finalize.
      iree_host_size_t final_length = 0;
      status = iree_tokenizer_decode_state_finalize(state, text_output,
                                                    &final_length);
      iree_status_ignore(status);

      iree_tokenizer_decode_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }

    iree_allocator_free(iree_allocator_system(), state_storage);
  }

  return 0;
}
