// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Split segmenter.
//
// Tests the regex-based split segmenter's robustness against:
// - Various regex patterns
// - All split behaviors (ISOLATED, REMOVED, MERGED_*, CONTIGUOUS)
// - Invert mode (pattern matches tokens vs delimiters)
// - Pattern matches at chunk boundaries
// - Empty segments from consecutive delimiters
// - Invalid UTF-8 in input
// - Streaming with various chunk sizes
//
// Note: This fuzzer tests with a fixed set of safe regex patterns to avoid
// spending fuzz time on regex compilation failures. The regex_fuzz target
// covers regex compilation edge cases separately.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "iree/base/api.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/segmenter.h"
#include "iree/tokenizer/segmenter/split.h"

// Pre-compiled patterns for testing (compiled once at startup).
static iree_tokenizer_segmenter_t* g_segmenters[10] = {NULL};
static size_t g_segmenter_count = 0;

// Helper to compile a pattern and create a segmenter.
static iree_tokenizer_segmenter_t* compile_segmenter(
    const char* pattern, iree_tokenizer_regex_split_behavior_t behavior,
    bool invert) {
  // Compile pattern to binary DFA data.
  uint8_t* dfa_data = NULL;
  iree_host_size_t dfa_size = 0;
  iree_tokenizer_regex_compile_error_t error = {0};
  iree_status_t status = iree_tokenizer_regex_compile(
      iree_make_cstring_view(pattern),
      IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE, iree_allocator_system(),
      &dfa_data, &dfa_size, &error);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return NULL;
  }

  // Load the DFA from binary data.
  iree_tokenizer_regex_dfa_t dfa = {0};
  status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(dfa_data, dfa_size), &dfa);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_allocator_free(iree_allocator_system(), dfa_data);
    return NULL;
  }

  // Create segmenter (takes ownership of dfa_data on success).
  iree_tokenizer_segmenter_t* segmenter = NULL;
  status = iree_tokenizer_segmenter_split_allocate(
      dfa, dfa_data, behavior, invert, iree_allocator_system(), &segmenter);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_allocator_free(iree_allocator_system(), dfa_data);
    return NULL;
  }

  return segmenter;
}

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  (void)argc;
  (void)argv;

  // Compile a variety of useful patterns with different behaviors.
  // Pattern 1: Whitespace splitting (common use case).
  g_segmenters[g_segmenter_count++] =
      compile_segmenter("\\s+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_REMOVED, false);

  // Pattern 2: Word matching (GPT-2 style, invert mode).
  g_segmenters[g_segmenter_count++] = compile_segmenter(
      "[a-zA-Z]+", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED, true);

  // Pattern 3: Punctuation as delimiter.
  g_segmenters[g_segmenter_count++] = compile_segmenter(
      "[.,!?;:]", IREE_TOKENIZER_UTIL_REGEX_SPLIT_ISOLATED, false);

  // Pattern 4: Merged with previous.
  g_segmenters[g_segmenter_count++] = compile_segmenter(
      " ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_PREVIOUS, false);

  // Pattern 5: Merged with next.
  g_segmenters[g_segmenter_count++] = compile_segmenter(
      " ", IREE_TOKENIZER_UTIL_REGEX_SPLIT_MERGED_WITH_NEXT, false);

  // Pattern 6: Contiguous.
  g_segmenters[g_segmenter_count++] =
      compile_segmenter("-", IREE_TOKENIZER_UTIL_REGEX_SPLIT_CONTIGUOUS, false);

  return 0;
}

static void process_with_chunk_size(iree_tokenizer_segmenter_t* segmenter,
                                    const uint8_t* data, size_t size,
                                    size_t chunk_size) {
  if (!segmenter) return;

  iree_host_size_t state_size = iree_tokenizer_segmenter_state_size(segmenter);
  if (state_size == 0 || state_size > 64 * 1024) return;

  void* state_buffer = malloc(state_size);
  if (!state_buffer) return;

  iree_tokenizer_segmenter_state_t* state = NULL;
  iree_status_t status = iree_tokenizer_segmenter_state_initialize(
      segmenter, state_buffer, &state);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    free(state_buffer);
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
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > 16 * 1024) size = 16 * 1024;

  // Test each pre-compiled segmenter with different chunk sizes.
  for (size_t i = 0; i < g_segmenter_count; ++i) {
    if (g_segmenters[i]) {
      process_with_chunk_size(g_segmenters[i], data, size, 1);
      process_with_chunk_size(g_segmenters[i], data, size, 7);
      process_with_chunk_size(g_segmenters[i], data, size, size);
    }
  }

  return 0;
}
