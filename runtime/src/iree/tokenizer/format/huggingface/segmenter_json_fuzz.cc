// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for HuggingFace segmenter (pre_tokenizer) JSON parsing.
//
// Tests the segmenter parser's robustness against:
// - Malformed JSON syntax
// - Unknown pre_tokenizer types
// - Invalid regex patterns (for Split/ByteLevel)
// - Deeply nested Sequence pre-tokenizers
// - Missing required fields
// - Type mismatches
// - Invalid parameters (invalid codepoints, empty patterns)
// - All segmenter types: ByteLevel, Metaspace, Split, Sequence, Punctuation
//
// When parsing succeeds, the segmenter is exercised against a corpus of
// interesting strings to test the execution path.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/segmenter_json.h"
#include "iree/tokenizer/segmenter.h"

// Long string for buffer boundary testing (2KB of mixed content).
// Forces multiple process() calls when output buffer fills.
static char kLongInput[2048];
static bool kLongInputInitialized = false;

static void initialize_long_input(void) {
  if (kLongInputInitialized) return;
  // Pattern: "word1 word2, word3. word4! " repeated to fill buffer.
  const char* pattern = "alpha beta, gamma. delta! ";
  size_t pattern_length = strlen(pattern);
  size_t position = 0;
  while (position + pattern_length < sizeof(kLongInput)) {
    memcpy(kLongInput + position, pattern, pattern_length);
    position += pattern_length;
  }
  kLongInput[position] = '\0';
  kLongInputInitialized = true;
}

// Test corpus for exercising parsed segmenters.
// Covers: empty, ASCII, whitespace, Unicode, CJK, punctuation-heavy, emoji.
static const char* kTestInputs[] = {
    "",                                  // Empty string.
    "hello world",                       // Simple ASCII with space.
    "hello",                             // Single word.
    "hello  world",                      // Multiple spaces.
    "   hello   ",                       // Leading/trailing whitespace.
    "\t\n\r",                            // Tab, newline, carriage return.
    "Hello, World! How are you?",        // Punctuation.
    "can't won't don't",                 // Contractions.
    "test@example.com",                  // Email-like.
    "https://example.com/path?query=1",  // URL-like.
    "Caf\xC3\xA9 na\xC3\xAFve",          // UTF-8 accented.
    "\xE4\xB8\xAD\xE6\x96\x87\xE6\xB5\x8B\xE8\xAF\x95",  // Chinese: 中文测试.
    "\xF0\x9F\x98\x80\xF0\x9F\x8E\x89",                  // Emoji: 😀🎉.
    "a.b.c.d.e",                                         // Lots of punctuation.
    "123-456-7890",      // Numbers with hyphens.
    "\xC0\xC1\xF5\xF6",  // Invalid UTF-8.
    kLongInput,          // Long input for buffer stress.
};
static const size_t kNumTestInputs =
    sizeof(kTestInputs) / sizeof(kTestInputs[0]);

// Exercises a segmenter against test inputs using process/finalize pattern.
// Uses a loop to handle cases where the output buffer fills up mid-input.
static void exercise_segmenter(iree_tokenizer_segmenter_t* segmenter) {
  // Initialize long input buffer on first call.
  initialize_long_input();

  iree_host_size_t state_size = iree_tokenizer_segmenter_state_size(segmenter);
  if (state_size == 0 || state_size > 64 * 1024) {
    return;  // Sanity check.
  }

  void* state_buffer = malloc(state_size);
  if (!state_buffer) return;

  // Small output buffer to force multiple process() calls on long inputs.
  iree_tokenizer_segment_t segments[64];
  iree_tokenizer_segment_output_t output =
      iree_tokenizer_make_segment_output(segments, 64);

  for (size_t i = 0; i < kNumTestInputs; ++i) {
    iree_tokenizer_segmenter_state_t* state = NULL;
    iree_status_t status = iree_tokenizer_segmenter_state_initialize(
        segmenter, state_buffer, &state);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    const char* input = kTestInputs[i];
    size_t input_length = strlen(input);

    // Loop until all input is consumed, handling output buffer full cases.
    iree_host_size_t total_consumed = 0;
    while (total_consumed < input_length) {
      iree_string_view_t chunk = iree_make_string_view(
          input + total_consumed, input_length - total_consumed);

      iree_host_size_t consumed = 0;
      iree_host_size_t segment_count = 0;
      status = iree_tokenizer_segmenter_state_process(
          state, chunk, output, &consumed, &segment_count);

      if (iree_status_is_resource_exhausted(status)) {
        // Output buffer full - continue processing remaining input.
        iree_status_ignore(status);
      } else if (!iree_status_is_ok(status)) {
        // Genuine failure in segmenter logic.
        iree_status_ignore(status);
        break;
      }

      // Prevent infinite loops if segmenter refuses to make progress.
      if (consumed == 0 && segment_count == 0) {
        break;
      }
      total_consumed += consumed;
    }

    // Finalize with remaining input (bytes not consumed by process loop).
    iree_string_view_t remaining = iree_make_string_view(
        input + total_consumed,
        input_length > total_consumed ? input_length - total_consumed : 0);
    iree_host_size_t final_segment_count = 0;
    status = iree_tokenizer_segmenter_state_finalize(state, remaining, output,
                                                     &final_segment_count);
    iree_status_ignore(status);

    iree_tokenizer_segmenter_state_deinitialize(state);
  }

  free(state_buffer);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Treat fuzz input as pre_tokenizer JSON.
  iree_string_view_t segmenter_json =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Try parsing the JSON as a segmenter.
  iree_tokenizer_segmenter_t* segmenter = NULL;
  iree_tokenizer_huggingface_pre_tokenizer_flags_t flags = 0;
  iree_status_t status = iree_tokenizer_huggingface_parse_segmenter(
      segmenter_json, iree_allocator_system(), &segmenter, &flags);

  if (iree_status_is_ok(status) && segmenter != NULL) {
    // Successfully parsed - exercise the segmenter.
    exercise_segmenter(segmenter);
  }

  // Always ignore status.
  iree_status_ignore(status);

  // Free unconditionally - null-safe.
  iree_tokenizer_segmenter_free(segmenter);

  return 0;
}
