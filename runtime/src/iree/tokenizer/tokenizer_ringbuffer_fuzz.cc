// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for ring buffer wraparound edge cases in streaming encoding.
//
// Tests the ring buffer management in tokenizer encode against:
// - Alternating small/large chunks that cause wraparound
// - Chunks that fill the buffer exactly
// - Partial token emission followed by compaction
// - Many small chunks accumulating
// - Pathological patterns that stress buffer boundaries
//
// The key insight is that tokenizers use a ring buffer for streaming to avoid
// copying data. Edge cases occur when:
// - A segment spans the wraparound point
// - The buffer fills up mid-segment
// - Compaction must move live data while preserving segment boundaries
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

// Global tokenizer instance.
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
  if (g_tokenizer == NULL || size < 2) return 0;

  // First byte controls buffer size (powers of 2 from 256 to 16KB).
  uint8_t size_selector = data[0] % 7;  // 0-6
  iree_host_size_t buffer_sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
  iree_host_size_t buffer_size = buffer_sizes[size_selector];
  data++;
  size--;

  // Remaining bytes are the input to encode.
  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  //===--------------------------------------------------------------------===//
  // Test: Streaming encode with controlled buffer size
  //===--------------------------------------------------------------------===//

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

  void* transform_buffer = NULL;
  status = iree_allocator_malloc(iree_allocator_system(), buffer_size,
                                 &transform_buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), state_storage);
    iree_status_ignore(status);
    return 0;
  }

  iree_tokenizer_encode_flags_t encode_flags =
      IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
  if (iree_tokenizer_fuzz_track_offsets()) {
    encode_flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
  }

  iree_tokenizer_encode_state_t* state = NULL;
  status = iree_tokenizer_encode_state_initialize(
      g_tokenizer,
      iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage),
                          state_size),
      iree_make_byte_span(reinterpret_cast<uint8_t*>(transform_buffer),
                          buffer_size),
      iree_tokenizer_offset_run_list_empty(), encode_flags, &state);

  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), transform_buffer);
    iree_allocator_free(iree_allocator_system(), state_storage);
    iree_status_ignore(status);
    return 0;
  }

  iree_tokenizer_token_id_t tokens[2048];
  iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
      tokens, NULL, NULL, IREE_ARRAYSIZE(tokens));

  // Feed input using chunk sizes derived from the input itself.
  // This creates data-dependent access patterns that stress ring buffer edges.
  iree_host_size_t offset = 0;
  iree_host_size_t chunk_index = 0;
  while (offset < size) {
    // Derive chunk size from input data to create varied patterns.
    iree_host_size_t chunk_base =
        (chunk_index < size) ? (data[chunk_index % size] & 0x7F) + 1 : 64;
    // Scale to range 1-512 bytes.
    iree_host_size_t chunk_size = ((chunk_base % 64) + 1) * 8;
    if (chunk_size > size - offset) chunk_size = size - offset;

    iree_string_view_t chunk =
        iree_make_string_view(input.data + offset, chunk_size);

    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t token_count = 0;
    status = iree_tokenizer_encode_state_feed(state, chunk, output,
                                              &bytes_consumed, &token_count);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }

    offset += bytes_consumed;
    ++chunk_index;

    // Prevent infinite loops.
    if (bytes_consumed == 0 && token_count == 0) {
      break;
    }
    if (chunk_index > 10000) {
      break;  // Safety limit.
    }
  }

  // Finalize to flush any remaining data.
  iree_host_size_t final_count = 0;
  status = iree_tokenizer_encode_state_finalize(state, output, &final_count);
  iree_status_ignore(status);

  iree_tokenizer_encode_state_deinitialize(state);
  iree_allocator_free(iree_allocator_system(), transform_buffer);
  iree_allocator_free(iree_allocator_system(), state_storage);

  return 0;
}
