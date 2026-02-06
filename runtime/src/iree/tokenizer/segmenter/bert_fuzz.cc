// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for BERT segmenter (BertPreTokenizer).
//
// Tests the BERT segmenter's robustness against:
// - Invalid UTF-8 sequences
// - Chunk boundaries splitting multi-byte codepoints
// - Long runs of punctuation or whitespace
// - Mixed Unicode whitespace characters
// - Unicode punctuation categories
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/segmenter/bert.h"

// Process fuzz input through BERT segmenter with streaming chunks.
static void process_with_chunk_size(const uint8_t* data, size_t size,
                                    size_t chunk_size) {
  iree_tokenizer_segmenter_t* segmenter = NULL;
  iree_status_t status = iree_tokenizer_segmenter_bert_allocate(
      iree_allocator_system(), &segmenter);
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

  // Output buffer for segments.
  iree_tokenizer_segment_t segments[64];
  iree_tokenizer_segment_output_t output =
      iree_tokenizer_make_segment_output(segments, 64);

  // Process in chunks.
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
      // Output buffer full - reset to continue.
      output = iree_tokenizer_make_segment_output(segments, 64);
    } else if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }

    if (consumed == 0 && segment_count == 0) {
      // Stalled - expand chunk for multi-byte codepoints.
      if (current_chunk_size >= remaining) {
        break;
      }
      current_chunk_size = current_chunk_size * 2 < remaining
                               ? current_chunk_size * 2
                               : remaining;
      continue;
    }
    current_chunk_size = chunk_size;
    offset += consumed;
  }

  // Finalize with remaining input.
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
  if (size > 16 * 1024) {
    size = 16 * 1024;
  }

  // Test with different chunk sizes.
  process_with_chunk_size(data, size, 1);     // Byte at a time.
  process_with_chunk_size(data, size, 5);     // Small prime.
  process_with_chunk_size(data, size, size);  // All at once.

  return 0;
}
