// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for tokenizer encoding with adversarial inputs.
//
// Tests the encoding pipeline against:
// - Very long input strings
// - Inputs with all unknown characters
// - Every Unicode category
// - Rapid whitespace/non-whitespace alternation
// - Invalid UTF-8 sequences
// - Inputs that stress ring buffer wraparound
// - Pathological patterns for BPE merging
//
// Uses fuzzing_util for tokenizer loading (dummy or real via --tokenizer_json).
// Supports --track_offsets for offset tracking coverage.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/testing/fuzzing_util.h"
#include "iree/tokenizer/tokenizer.h"

// Global tokenizer instance built once at fuzzer startup.
static iree_tokenizer_t* g_tokenizer = NULL;

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  iree_status_t status =
      iree_tokenizer_fuzz_load_or_build(argc, argv, &g_tokenizer, NULL);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_tokenizer == NULL) return 0;

  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  iree_tokenizer_encode_flags_t flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  if (iree_tokenizer_fuzz_track_offsets()) {
    flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
  }

  //===--------------------------------------------------------------------===//
  // Test 1: One-shot encode
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_token_id_t tokens[8192];
    iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
        tokens, NULL, NULL, IREE_ARRAYSIZE(tokens));
    iree_host_size_t token_count = 0;
    iree_status_t status =
        iree_tokenizer_encode(g_tokenizer, input, flags, output,
                              iree_allocator_system(), &token_count);
    // RESOURCE_EXHAUSTED is expected for very large inputs.
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Streaming encode with varying chunk sizes
  //===--------------------------------------------------------------------===//

  {
    iree_host_size_t state_size = 0;
    iree_status_t status =
        iree_tokenizer_encode_state_calculate_size(g_tokenizer, &state_size);
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

    iree_host_size_t buffer_size =
        iree_tokenizer_transform_buffer_recommended_size(size);
    void* transform_buffer = NULL;
    status = iree_allocator_malloc(iree_allocator_system(), buffer_size,
                                   &transform_buffer);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), state_storage);
      iree_status_ignore(status);
      return 0;
    }

    iree_tokenizer_encode_state_t* state = NULL;
    status = iree_tokenizer_encode_state_initialize(
        g_tokenizer,
        iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage),
                            state_size),
        iree_make_byte_span(reinterpret_cast<uint8_t*>(transform_buffer),
                            buffer_size),
        iree_tokenizer_offset_run_list_empty(),
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &state);

    if (iree_status_is_ok(status)) {
      iree_tokenizer_token_id_t tokens[1024];
      iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
          tokens, NULL, NULL, IREE_ARRAYSIZE(tokens));

      // Feed input in varying chunk sizes to stress edge cases.
      iree_host_size_t offset = 0;
      while (offset < size) {
        iree_host_size_t chunk_size = ((offset % 5) == 0)   ? 1
                                      : ((offset % 5) == 1) ? 7
                                      : ((offset % 5) == 2) ? 64
                                      : ((offset % 5) == 3) ? 256
                                                            : 512;
        if (chunk_size > size - offset) chunk_size = size - offset;

        iree_string_view_t chunk =
            iree_make_string_view(input.data + offset, chunk_size);

        iree_host_size_t bytes_consumed = 0;
        iree_host_size_t token_count = 0;
        status = iree_tokenizer_encode_state_feed(
            state, chunk, output, &bytes_consumed, &token_count);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }
        offset += bytes_consumed;
        if (bytes_consumed == 0 && token_count == 0) {
          break;  // No progress.
        }
      }

      // Finalize.
      iree_host_size_t final_count = 0;
      status =
          iree_tokenizer_encode_state_finalize(state, output, &final_count);
      iree_status_ignore(status);

      iree_tokenizer_encode_state_deinitialize(state);
    } else {
      iree_status_ignore(status);
    }

    iree_allocator_free(iree_allocator_system(), transform_buffer);
    iree_allocator_free(iree_allocator_system(), state_storage);
  }

  return 0;
}
