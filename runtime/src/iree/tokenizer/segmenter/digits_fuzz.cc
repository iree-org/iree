// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Digits segmenter.
//
// Tests the digits segmenter's robustness against:
// - ASCII digit sequences (0-9)
// - Mixed digit and non-digit text
// - Long runs of digits
// - Long runs of non-digits
// - Alternating digits/non-digits
// - Invalid UTF-8 sequences
// - Chunk boundary handling
// - Both individual_digits=true and individual_digits=false modes
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/segmenter/digits.h"

static void process_with_chunk_size(const uint8_t* data, size_t size,
                                    size_t chunk_size, bool individual_digits) {
  iree_tokenizer_segmenter_t* segmenter = NULL;
  iree_status_t status = iree_tokenizer_segmenter_digits_allocate(
      individual_digits, iree_allocator_system(), &segmenter);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }

  iree_host_size_t state_size = iree_tokenizer_segmenter_state_size(segmenter);
  if (state_size == 0 || state_size > 64 * 1024) {
    iree_tokenizer_segmenter_free(segmenter);
    return;
  }

  void* state_buffer = malloc(state_size);
  if (!state_buffer) {
    iree_tokenizer_segmenter_free(segmenter);
    return;
  }

  iree_tokenizer_segmenter_state_t* state = NULL;
  status = iree_tokenizer_segmenter_state_initialize(segmenter, state_buffer,
                                                     &state);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    free(state_buffer);
    iree_tokenizer_segmenter_free(segmenter);
    return;
  }

  iree_tokenizer_segment_t segments[64];
  iree_tokenizer_segment_output_t output =
      iree_tokenizer_make_segment_output(segments, 64);

  size_t offset = 0;
  size_t current_chunk_size = chunk_size;
  while (offset < size) {
    size_t remaining = size - offset;
    size_t this_chunk =
        remaining < current_chunk_size ? remaining : current_chunk_size;

    iree_string_view_t input_chunk = iree_make_string_view(
        reinterpret_cast<const char*>(data + offset), this_chunk);

    iree_host_size_t consumed = 0;
    iree_host_size_t segment_count = 0;
    status = iree_tokenizer_segmenter_state_process(state, input_chunk, output,
                                                    &consumed, &segment_count);

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_ignore(status);
      output = iree_tokenizer_make_segment_output(segments, 64);
    } else if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }

    if (consumed == 0 && segment_count == 0) {
      if (current_chunk_size >= remaining) break;
      current_chunk_size = current_chunk_size * 2 < remaining
                               ? current_chunk_size * 2
                               : remaining;
      continue;
    }
    current_chunk_size = chunk_size;
    offset += consumed;
  }

  iree_string_view_t remaining_input = iree_make_string_view(
      reinterpret_cast<const char*>(data + offset), size - offset);
  iree_host_size_t final_segment_count = 0;
  status = iree_tokenizer_segmenter_state_finalize(
      state, remaining_input, output, &final_segment_count);
  iree_status_ignore(status);

  iree_tokenizer_segmenter_state_deinitialize(state);
  free(state_buffer);
  iree_tokenizer_segmenter_free(segmenter);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > 16 * 1024) size = 16 * 1024;

  // Test individual_digits=true mode.
  process_with_chunk_size(data, size, 1, /*individual_digits=*/true);
  process_with_chunk_size(data, size, 5, /*individual_digits=*/true);
  process_with_chunk_size(data, size, 64, /*individual_digits=*/true);
  process_with_chunk_size(data, size, size, /*individual_digits=*/true);

  // Test individual_digits=false mode.
  process_with_chunk_size(data, size, 1, /*individual_digits=*/false);
  process_with_chunk_size(data, size, 5, /*individual_digits=*/false);
  process_with_chunk_size(data, size, 64, /*individual_digits=*/false);
  process_with_chunk_size(data, size, size, /*individual_digits=*/false);

  return 0;
}
