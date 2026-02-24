// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for NFD (Unicode Normalization Form D) normalizer.
//
// Tests the NFD normalizer's robustness against:
// - Invalid UTF-8 sequences (overlong, truncated, surrogate pairs)
// - Combining character sequences (mark ordering, reordering by CCC)
// - Long combining sequences (exceeding 32-codepoint stream-safe limit)
// - Hangul syllable decomposition (LV and LVT forms)
// - Canonical decomposition edge cases (e.g. precomposed accented chars)
//
// NFD performs canonical decomposition only (no compatibility decompositions),
// so unlike NFKD, ligatures (ﬁ), fractions (½), and fullwidth chars pass
// through unchanged. The maximum expansion per codepoint is smaller than NFKD.
//
// The fuzzer exercises the normalizer using the streaming process/finalize
// pattern with various chunk sizes to stress state management, particularly
// combining sequence buffering across chunk boundaries.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/normalizer/nfd.h"

// Process fuzz input through NFD normalizer with streaming chunks.
static void process_with_chunk_size(iree_tokenizer_normalizer_t* normalizer,
                                    const uint8_t* data, size_t size,
                                    size_t chunk_size) {
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer);
  if (state_size == 0 || state_size > 64 * 1024) {
    return;  // Sanity check.
  }

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

  // Output buffer sized relative to input.
  // NFD canonical decomposition has smaller expansion than NFKD but can still
  // expand significantly (e.g. Hangul syllables decompose to 2-3 Jamo).
  size_t output_capacity = size * 4 + 128;
  if (output_capacity > 128 * 1024) {
    output_capacity = 128 * 1024;  // Cap to prevent OOM.
  }
  char* output = (char*)malloc(output_capacity);
  if (!output) {
    iree_tokenizer_normalizer_state_deinitialize(state);
    free(state_buffer);
    return;
  }

  // Process in chunks to stress state management.
  size_t offset = 0;
  size_t total_written = 0;
  size_t stall_count = 0;
  while (offset < size) {
    size_t remaining = size - offset;
    size_t this_chunk = remaining < chunk_size ? remaining : chunk_size;

    iree_string_view_t input_chunk = iree_make_string_view(
        reinterpret_cast<const char*>(data + offset), this_chunk);

    // Advance output pointer to accumulate normalized text.
    if (total_written >= output_capacity) break;
    iree_mutable_string_view_t output_view = iree_make_mutable_string_view(
        output + total_written, output_capacity - total_written);

    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    status = iree_tokenizer_normalizer_state_process(
        state, input_chunk, output_view, IREE_TOKENIZER_NORMALIZER_FLAG_NONE,
        &consumed, &written);

    if (iree_status_is_resource_exhausted(status)) {
      // Combining sequence too long or output full - expected behavior.
      iree_status_ignore(status);
      break;
    } else if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }

    total_written += written;

    if (consumed == 0) {
      if (written == 0) {
        break;  // No progress.
      }
      if (++stall_count > 16) {
        break;  // Abort stalled processing.
      }
    } else {
      stall_count = 0;
    }
    offset += consumed;
  }

  // Finalize to flush any buffered combining sequence.
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
  // Cap input size to prevent excessive memory/time usage.
  if (size > 16 * 1024) {
    size = 16 * 1024;
  }

  // Allocate NFD normalizer.
  iree_tokenizer_normalizer_t* normalizer = NULL;
  iree_status_t status = iree_tokenizer_normalizer_nfd_allocate(
      iree_allocator_system(), &normalizer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  // Test with different chunk sizes to stress streaming state.
  // 1: byte-at-a-time, catches mid-codepoint splits.
  // 7: prime, misaligns with all tile/codepoint boundaries.
  // 64: matches internal tile size.
  // 256: multi-tile processing.
  // 512: larger multi-tile processing.
  // size: all at once, exercises bulk fast path.
  process_with_chunk_size(normalizer, data, size, 1);
  process_with_chunk_size(normalizer, data, size, 7);
  process_with_chunk_size(normalizer, data, size, 64);
  process_with_chunk_size(normalizer, data, size, 256);
  process_with_chunk_size(normalizer, data, size, 512);
  process_with_chunk_size(normalizer, data, size, size);

  iree_tokenizer_normalizer_free(normalizer);
  return 0;
}
