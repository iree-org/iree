// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Strip normalizer.
//
// Tests the strip normalizer's robustness against:
// - ASCII whitespace (space, tab, LF, VT, FF, CR)
// - Unicode whitespace (NBSP, En/Em spaces, ideographic space)
// - Mixed leading/trailing whitespace
// - All-whitespace input
// - No-whitespace input
// - Lazy consumption edge cases for strip_right
// - Streaming with various chunk sizes
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/normalizer/strip.h"

static void process_with_chunk_size(iree_tokenizer_normalizer_t* normalizer,
                                    const uint8_t* data, size_t size,
                                    size_t chunk_size) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer);
  if (state_size == 0 || state_size > 64 * 1024) return;

  void* state_buffer = malloc(state_size);
  if (!state_buffer) return;

  iree_tokenizer_normalizer_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_normalizer_state_initialize(
      normalizer, state_buffer, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    free(state_buffer);
    return;
  }

  size_t output_capacity = size + 64;
  if (output_capacity > 64 * 1024) output_capacity = 64 * 1024;
  char* output = (char*)malloc(output_capacity);
  if (!output) {
    iree_tokenizer_normalizer_state_deinitialize(state);
    free(state_buffer);
    return;
  }

  size_t offset = 0;
  size_t total_written = 0;
  size_t stall_count = 0;
  while (offset < size) {
    size_t remaining = size - offset;
    size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;

    iree_string_view_t input_chunk = iree_make_string_view(
        reinterpret_cast<const char*>(data + offset), this_chunk);

    if (total_written >= output_capacity) break;
    iree_mutable_string_view_t output_view = iree_make_mutable_string_view(
        output + total_written, output_capacity - total_written);

    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    status = iree_tokenizer_normalizer_state_process(
        state, input_chunk, output_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
        &consumed, &written);

    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }

    total_written += written;

    if (consumed == 0) {
      if (written == 0) break;
      if (++stall_count > 16) break;
    } else {
      stall_count = 0;
    }
    offset += consumed;
  }

  if (total_written < output_capacity) {
    iree_host_size_t final_written = 0;
    status = iree_tokenizer_normalizer_state_finalize(
        state,
        iree_make_mutable_string_view(output + total_written,
                                      output_capacity - total_written),
        &final_written);
    iree_status_ignore(status);
  }

  iree_tokenizer_normalizer_state_deinitialize(state);
  free(output);
  free(state_buffer);
}

static void test_with_config(const uint8_t* data, size_t size, bool strip_left,
                             bool strip_right) {
  iree_tokenizer_normalizer_t* normalizer = NULL;
  iree_status_t status = iree_tokenizer_normalizer_strip_allocate(
      strip_left, strip_right, iree_allocator_system(), &normalizer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }

  process_with_chunk_size(normalizer, data, size, 1);
  process_with_chunk_size(normalizer, data, size, 7);
  process_with_chunk_size(normalizer, data, size, 64);
  process_with_chunk_size(normalizer, data, size, size);

  iree_tokenizer_normalizer_free(normalizer);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > 16 * 1024) size = 16 * 1024;

  // Test all combinations of strip_left and strip_right.
  test_with_config(data, size, false, false);
  test_with_config(data, size, true, false);
  test_with_config(data, size, false, true);
  test_with_config(data, size, true, true);

  return 0;
}
