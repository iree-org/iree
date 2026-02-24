// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for replace normalizer streaming behavior.
//
// Exercises the multi-byte replace normalizer with:
// - Varied pattern sizes (1-32 bytes derived from input)
// - Varied content sizes (0-32 bytes derived from input)
// - Randomized input chunk sizes (1-32 bytes)
// - Randomized output buffer sizes (1-64 bytes)
// - Cross-chunk boundary pattern matching (overlap buffer stress)
// - Output buffer exhaustion (pending buffer stress)
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer/replace.h"

// Minimum input size needed to derive pattern, content, and have some input.
static constexpr size_t kMinInputSize = 4;

// Maximum sizes matching the replace normalizer limits.
static constexpr size_t kMaxPatternSize = 32;
static constexpr size_t kMaxContentSize = 32;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < kMinInputSize) return 0;

  // First two bytes control pattern and content sizes.
  uint8_t pattern_size_byte = data[0];
  uint8_t content_size_byte = data[1];
  uint8_t mode_byte = data[2];
  data += 3;
  size -= 3;

  // Derive pattern size (1-32 bytes, never 0).
  size_t pattern_size = (pattern_size_byte % kMaxPatternSize) + 1;
  if (pattern_size > size) {
    pattern_size = size > 0 ? size : 1;
  }

  // Extract pattern from input.
  const char* pattern_data = reinterpret_cast<const char*>(data);
  data += pattern_size;
  size -= pattern_size;

  if (size < 1) return 0;

  // Derive content size (0-32 bytes, can be 0 for deletion).
  size_t content_size = content_size_byte % (kMaxContentSize + 1);
  if (content_size > size) {
    content_size = size;
  }

  // Extract content from input.
  const char* content_data = reinterpret_cast<const char*>(data);
  data += content_size;
  size -= content_size;

  // Remaining data is the input to normalize.
  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Create replace normalizer.
  iree_string_view_t pattern =
      iree_make_string_view(pattern_data, pattern_size);
  iree_string_view_t content =
      iree_make_string_view(content_data, content_size);

  iree_tokenizer_normalizer_t* normalizer = NULL;
  iree_status_t status = iree_tokenizer_normalizer_replace_allocate(
      pattern, content, iree_allocator_system(), &normalizer);

  if (!iree_status_is_ok(status)) {
    // Allocation can fail for invalid pattern/content (e.g., empty pattern).
    iree_status_ignore(status);
    return 0;
  }

  // Allocate state.
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer);
  void* state_buffer = NULL;
  status =
      iree_allocator_malloc(iree_allocator_system(), state_size, &state_buffer);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_normalizer_free(normalizer);
    iree_status_ignore(status);
    return 0;
  }

  iree_tokenizer_normalizer_state_t* state = NULL;
  status = iree_tokenizer_normalizer_state_initialize(normalizer, state_buffer,
                                                      &state);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), state_buffer);
    iree_tokenizer_normalizer_free(normalizer);
    iree_status_ignore(status);
    return 0;
  }

  // Output buffer size controlled by mode.
  // Small sizes stress the pending buffer draining logic.
  size_t output_size = (mode_byte & 0x07) == 0   ? 1      // Minimum.
                       : (mode_byte & 0x07) == 1 ? 2      // Tiny.
                       : (mode_byte & 0x07) == 2 ? 4      // Very small.
                       : (mode_byte & 0x07) == 3 ? 8      // Small.
                       : (mode_byte & 0x07) == 4 ? 16     // Medium.
                       : (mode_byte & 0x07) == 5 ? 64     // Normal.
                       : (mode_byte & 0x07) == 6 ? 256    // Large.
                                                 : 4096;  // Very large.

  char* output = (char*)malloc(output_size);
  if (!output) {
    iree_allocator_free(iree_allocator_system(), state_buffer);
    iree_tokenizer_normalizer_free(normalizer);
    return 0;
  }

  if ((mode_byte & 0x80) == 0) {
    //===------------------------------------------------------------------===//
    // Mode A: Streaming with varied chunk sizes
    //===------------------------------------------------------------------===//
    size_t offset = 0;
    size_t chunk_index = 0;

    while (offset < size) {
      // Data-dependent chunk size (1-32 bytes).
      size_t chunk_base =
          (chunk_index < size) ? (data[chunk_index % size] & 0x1F) : 8;
      size_t chunk_size = chunk_base + 1;
      if (chunk_size > size - offset) chunk_size = size - offset;

      iree_string_view_t chunk =
          iree_make_string_view(input.data + offset, chunk_size);
      iree_mutable_string_view_t out_view =
          iree_make_mutable_string_view(output, output_size);

      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t bytes_written = 0;

      status = iree_tokenizer_normalizer_state_process(
          state, chunk, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
          &bytes_consumed, &bytes_written);

      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }

      // Advance by consumed amount.
      offset += bytes_consumed;
      ++chunk_index;

      // Prevent infinite loops.
      if (bytes_consumed == 0 && bytes_written == 0 && chunk_size > 0) {
        // Output buffer might be too small to make progress.
        // This shouldn't happen with well-formed normalizers but bail safely.
        break;
      }
      if (chunk_index > 100000) {
        break;  // Iteration limit.
      }
    }

    // Finalize - may need multiple calls with small output buffer.
    for (int finalize_iter = 0; finalize_iter < 1000; ++finalize_iter) {
      iree_mutable_string_view_t final_view =
          iree_make_mutable_string_view(output, output_size);
      iree_host_size_t final_written = 0;
      status = iree_tokenizer_normalizer_state_finalize(state, final_view,
                                                        &final_written);
      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        break;
      }
      if (final_written == 0) {
        break;  // Nothing more to drain.
      }
    }

  } else {
    //===------------------------------------------------------------------===//
    // Mode B: One-shot process + finalize
    //===------------------------------------------------------------------===//
    iree_mutable_string_view_t out_view =
        iree_make_mutable_string_view(output, output_size);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t bytes_written = 0;

    status = iree_tokenizer_normalizer_state_process(
        state, input, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
        &bytes_consumed, &bytes_written);
    iree_status_ignore(status);

    iree_mutable_string_view_t final_view =
        iree_make_mutable_string_view(output, output_size);
    iree_host_size_t final_written = 0;
    status = iree_tokenizer_normalizer_state_finalize(state, final_view,
                                                      &final_written);
    iree_status_ignore(status);
  }

  free(output);
  iree_allocator_free(iree_allocator_system(), state_buffer);
  iree_tokenizer_normalizer_free(normalizer);

  return 0;
}
