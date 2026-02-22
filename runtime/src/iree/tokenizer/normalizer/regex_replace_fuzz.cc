// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for regex replace normalizer streaming behavior.
//
// Once a valid regex compiles, this fuzzer fans out to run many executions
// with different output buffer sizes and chunk strategies. This amortizes the
// expensive regex compilation (~80% of runtime) across many streaming tests.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <string>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer/regex_replace.h"

// Minimum input size needed to derive pattern, content, and have some input.
static constexpr size_t kMinInputSize = 4;

// Maximum sizes matching the regex replace normalizer limits.
static constexpr size_t kMaxPatternSize = 64;
static constexpr size_t kMaxContentSize =
    IREE_TOKENIZER_REGEX_REPLACE_MAX_CONTENT;

// Output buffer sizes to test (stress different code paths).
static constexpr size_t kOutputSizes[] = {1, 2, 4, 8, 16, 64, 256, 4096};
static constexpr size_t kNumOutputSizes =
    sizeof(kOutputSizes) / sizeof(kOutputSizes[0]);

// Runs one normalization pass with the given parameters.
// Returns true if execution completed without error.
static bool RunNormalization(iree_tokenizer_normalizer_t* normalizer,
                             iree_string_view_t input, size_t output_size,
                             size_t chunk_size,  // 0 = one-shot mode
                             const uint8_t* chunk_variation_data,
                             size_t chunk_variation_size) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer);

  // Stack-allocate state for small sizes, heap for large.
  uint8_t stack_buffer[512];
  void* state_buffer = nullptr;
  bool heap_allocated = false;

  if (state_size <= sizeof(stack_buffer)) {
    state_buffer = stack_buffer;
  } else {
    if (!iree_status_is_ok(iree_allocator_malloc(iree_allocator_system(),
                                                 state_size, &state_buffer))) {
      return false;
    }
    heap_allocated = true;
  }

  iree_tokenizer_normalizer_state_t* state = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_state_initialize(
      normalizer, state_buffer, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    if (heap_allocated)
      iree_allocator_free(iree_allocator_system(), state_buffer);
    return false;
  }

  // Allocate output buffer.
  char* output = (char*)malloc(output_size);
  if (!output) {
    if (heap_allocated)
      iree_allocator_free(iree_allocator_system(), state_buffer);
    return false;
  }

  bool success = true;

  if (chunk_size == 0) {
    // One-shot mode.
    iree_mutable_string_view_t out_view =
        iree_make_mutable_string_view(output, output_size);
    iree_host_size_t bytes_consumed = 0;
    iree_host_size_t bytes_written = 0;

    status = iree_tokenizer_normalizer_state_process(
        state, input, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
        &bytes_consumed, &bytes_written);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      success = false;
    }
  } else {
    // Streaming mode with varied chunk sizes.
    size_t offset = 0;
    size_t chunk_index = 0;

    while (offset < input.size) {
      // Vary chunk size based on input data for coverage.
      size_t actual_chunk = chunk_size;
      if (chunk_variation_data && chunk_variation_size > 0) {
        uint8_t variation =
            chunk_variation_data[chunk_index % chunk_variation_size];
        actual_chunk = (variation & 0x1F) + 1;  // 1-32 bytes
      }
      if (actual_chunk > input.size - offset) {
        actual_chunk = input.size - offset;
      }

      iree_string_view_t chunk =
          iree_make_string_view(input.data + offset, actual_chunk);
      iree_mutable_string_view_t out_view =
          iree_make_mutable_string_view(output, output_size);

      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t bytes_written = 0;

      status = iree_tokenizer_normalizer_state_process(
          state, chunk, out_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
          &bytes_consumed, &bytes_written);

      if (!iree_status_is_ok(status)) {
        iree_status_ignore(status);
        success = false;
        break;
      }

      offset += bytes_consumed;
      ++chunk_index;

      // Prevent infinite loops.
      if (bytes_consumed == 0 && bytes_written == 0 && actual_chunk > 0) {
        break;
      }
      if (chunk_index > 100000) {
        break;
      }
    }
  }

  // Finalize - may need multiple calls with small output buffer.
  if (success) {
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
        break;
      }
    }
  }

  free(output);
  if (heap_allocated) {
    iree_allocator_free(iree_allocator_system(), state_buffer);
  }

  return success;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < kMinInputSize) return 0;

  // First two bytes control pattern and content sizes.
  uint8_t pattern_size_byte = data[0];
  uint8_t content_size_byte = data[1];
  data += 2;
  size -= 2;

  // Derive pattern size (1-64 bytes, never 0).
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

  // Create regex replace normalizer (expensive - ~80% of runtime).
  iree_string_view_t pattern =
      iree_make_string_view(pattern_data, pattern_size);
  iree_string_view_t content =
      iree_make_string_view(content_data, content_size);

  iree_tokenizer_normalizer_t* normalizer = nullptr;
  iree_status_t status = iree_tokenizer_normalizer_regex_replace_allocate(
      pattern, content, iree_allocator_system(), &normalizer);

  if (!iree_status_is_ok(status)) {
    // Invalid regex - expected, just skip.
    iree_status_ignore(status);
    return 0;
  }

  // Fan out: run many executions with the compiled regex to amortize cost.
  // Test each output buffer size with multiple chunk strategies.
  for (size_t i = 0; i < kNumOutputSizes; ++i) {
    size_t output_size = kOutputSizes[i];

    // One-shot mode.
    RunNormalization(normalizer, input, output_size, 0, nullptr, 0);

    // Fixed chunk sizes: 1, 2, 4, 8, 16 bytes.
    for (size_t chunk = 1; chunk <= 16; chunk *= 2) {
      RunNormalization(normalizer, input, output_size, chunk, nullptr, 0);
    }

    // Variable chunk sizes driven by input data.
    RunNormalization(normalizer, input, output_size, 1, data, size);
  }

  iree_tokenizer_normalizer_free(normalizer);

  return 0;
}
