// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for BERT normalizer.
//
// Tests the BERT normalizer's robustness against:
// - Invalid UTF-8 sequences
// - Control characters (clean_text removes these)
// - CJK characters (handle_chinese_chars adds spaces)
// - Combining marks (strip_accents removes after NFD)
// - Unicode case folding edge cases (lowercase)
// - Output expansion from CJK spacing and NFD decomposition
//
// Tests multiple flag combinations to cover all code paths.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/normalizer.h"
#include "iree/tokenizer/normalizer/bert.h"

// Process fuzz input through BERT normalizer with streaming chunks.
static void process_with_flags(iree_tokenizer_bert_normalizer_flags_t flags,
                               const uint8_t* data, size_t size,
                               size_t chunk_size) {
  iree_tokenizer_normalizer_t* normalizer = NULL;
  iree_status_t status = iree_tokenizer_normalizer_bert_allocate(
      flags, iree_allocator_system(), &normalizer);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return;
  }

  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer);
  if (state_size == 0 || state_size > 64 * 1024) {
    iree_tokenizer_normalizer_free(normalizer);
    return;
  }

  void* state_buffer = malloc(state_size);
  if (!state_buffer) {
    iree_tokenizer_normalizer_free(normalizer);
    return;
  }

  iree_tokenizer_normalizer_state_t* state = NULL;
  status = iree_tokenizer_normalizer_state_initialize(normalizer, state_buffer,
                                                      &state);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    free(state_buffer);
    iree_tokenizer_normalizer_free(normalizer);
    return;
  }

  // Output buffer sized for expansion (CJK spacing can 3x output).
  size_t output_capacity = size * 4 + 64;
  if (output_capacity > 64 * 1024) {
    output_capacity = 64 * 1024;
  }
  char* output = (char*)calloc(output_capacity, 1);  // Zero for cleaner MSAN.
  if (!output) {
    iree_tokenizer_normalizer_state_deinitialize(state);
    free(state_buffer);
    iree_tokenizer_normalizer_free(normalizer);
    return;
  }

  // Process in chunks to stress state management.
  size_t offset = 0;
  size_t total_written = 0;
  size_t current_chunk_size = chunk_size;
  while (offset < size) {
    size_t remaining = size - offset;
    size_t this_chunk =
        remaining < current_chunk_size ? remaining : current_chunk_size;

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

    if (iree_status_is_resource_exhausted(status)) {
      iree_status_ignore(status);
      break;
    } else if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      break;
    }

    total_written += written;

    if (consumed == 0) {
      if (written == 0) {
        // Stalled - expand chunk to include more bytes (e.g., for multi-byte
        // UTF-8 sequences that need more data to process).
        if (current_chunk_size >= remaining) {
          break;  // Already providing all remaining data, truly stuck.
        }
        current_chunk_size = current_chunk_size * 2 < remaining
                                 ? current_chunk_size * 2
                                 : remaining;
        continue;  // Retry with larger chunk.
      }
    } else {
      current_chunk_size = chunk_size;  // Reset to original chunk size.
    }
    offset += consumed;
  }

  // Finalize to flush buffered output.
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
  iree_tokenizer_normalizer_free(normalizer);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Cap input size to prevent excessive memory/time usage.
  if (size > 16 * 1024) {
    size = 16 * 1024;
  }

  // Test with different flag combinations to cover all code paths.
  // Use first byte of input to select flags (if available).
  iree_tokenizer_bert_normalizer_flags_t base_flags =
      IREE_TOKENIZER_BERT_NORMALIZER_FLAG_DEFAULT;
  if (size > 0) {
    // Extract flag bits from first byte to vary configuration.
    uint8_t flag_byte = data[0];
    base_flags = 0;
    if (flag_byte & 0x01)
      base_flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_CLEAN_TEXT;
    if (flag_byte & 0x02)
      base_flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_HANDLE_CHINESE_CHARS;
    if (flag_byte & 0x04)
      base_flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_STRIP_ACCENTS;
    if (flag_byte & 0x08)
      base_flags |= IREE_TOKENIZER_BERT_NORMALIZER_FLAG_LOWERCASE;
    // Skip the flag byte for actual input.
    data++;
    size--;
  }

  if (size == 0) return 0;

  // Test with different chunk sizes.
  process_with_flags(base_flags, data, size, 1);     // Byte at a time.
  process_with_flags(base_flags, data, size, 11);    // Prime size.
  process_with_flags(base_flags, data, size, size);  // All at once.

  return 0;
}
