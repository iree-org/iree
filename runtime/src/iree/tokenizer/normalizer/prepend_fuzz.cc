// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Prepend normalizer.
//
// Tests the prepend normalizer's robustness against:
// - Various prepend string lengths (1-16 bytes)
// - Multi-byte UTF-8 prepend strings (e.g., ▁)
// - Empty input (should produce empty output)
// - skip_if_prefix_matches mode
// - Input that starts with the prepend string
// - Output buffer capacity stress
// - Streaming with various chunk sizes
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/normalizer/prepend.h"

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

  // Prepend adds bytes, so output can be larger than input.
  size_t output_capacity = size + 32;
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

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 2) return 0;
  if (size > 16 * 1024) size = 16 * 1024;

  // Use first byte to determine prepend string length (1-8).
  size_t prepend_len = (data[0] % 8) + 1;
  if (prepend_len > size - 1) prepend_len = size - 1;

  iree_string_view_t prepend_string = iree_make_string_view(
      reinterpret_cast<const char*>(data + 1), prepend_len);

  const uint8_t* input_data = data + 1 + prepend_len;
  size_t input_size = size - 1 - prepend_len;

  // Test with skip_if_prefix_matches = false.
  {
    iree_tokenizer_normalizer_t* normalizer = NULL;
    iree_status_t status = iree_tokenizer_normalizer_prepend_allocate(
        prepend_string, false, iree_allocator_system(), &normalizer);
    if (iree_status_is_ok(status)) {
      process_with_chunk_size(normalizer, input_data, input_size, 1);
      process_with_chunk_size(normalizer, input_data, input_size, 7);
      process_with_chunk_size(normalizer, input_data, input_size, input_size);
      iree_tokenizer_normalizer_free(normalizer);
    } else {
      iree_status_ignore(status);
    }
  }

  // Test with skip_if_prefix_matches = true.
  {
    iree_tokenizer_normalizer_t* normalizer = NULL;
    iree_status_t status = iree_tokenizer_normalizer_prepend_allocate(
        prepend_string, true, iree_allocator_system(), &normalizer);
    if (iree_status_is_ok(status)) {
      process_with_chunk_size(normalizer, input_data, input_size, 1);
      process_with_chunk_size(normalizer, input_data, input_size, 7);
      process_with_chunk_size(normalizer, input_data, input_size, input_size);
      iree_tokenizer_normalizer_free(normalizer);
    } else {
      iree_status_ignore(status);
    }
  }

  return 0;
}
