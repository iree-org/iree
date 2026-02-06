// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for HuggingFace normalizer JSON parsing.
//
// Tests the normalizer parser's robustness against:
// - Malformed JSON syntax
// - Unknown normalizer types
// - Invalid normalizer configurations
// - Deeply nested Sequence normalizers (stack overflow risk)
// - Missing required fields
// - Type mismatches (string where object expected, etc.)
// - Invalid parameters (empty patterns, out-of-range values)
// - null input for passthrough normalizer
// - All normalizer types: BERT, NFC, Lowercase, Strip, StripAccents,
//   Prepend, Replace, Precompiled, Sequence
//
// When parsing succeeds, the normalizer is exercised against a corpus of
// interesting strings to test the execution path, not just construction.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/format/huggingface/normalizer_json.h"
#include "iree/tokenizer/normalizer.h"

// Test corpus for exercising parsed normalizers.
// Covers: empty, ASCII, whitespace, Unicode, combining marks, emoji, CJK.
static const char* kTestInputs[] = {
    "",                          // Empty string.
    "hello world",               // Simple ASCII.
    "   \t\n  ",                 // Whitespace only.
    "HELLO WORLD",               // Uppercase ASCII.
    "Caf\xC3\xA9",               // UTF-8: café (with é).
    "na\xC3\xAFve",              // UTF-8: naïve (with ï).
    "e\xCC\x81",                 // Combining acute accent (e + ́).
    "\xE2\x80\x83\xE2\x80\x83",  // Em spaces (Unicode whitespace).
    "\xE4\xB8\xAD\xE6\x96\x87",  // Chinese: 中文.
    "\xF0\x9F\x98\x80",          // Emoji: 😀.
    "\xC0\xC1\xF5\xF6",          // Invalid UTF-8 sequences.
    "a\x00"
    "b",  // Embedded NUL.
};
static const size_t kNumTestInputs =
    sizeof(kTestInputs) / sizeof(kTestInputs[0]);

// Exercises a normalizer against test inputs.
static void exercise_normalizer(iree_tokenizer_normalizer_t* normalizer) {
  // Allocate state once, reuse for all inputs.
  iree_host_size_t state_size =
      iree_tokenizer_normalizer_state_size(normalizer);
  if (state_size == 0 || state_size > 64 * 1024) {
    return;  // Sanity check.
  }

  void* state_buffer = malloc(state_size);
  if (!state_buffer) return;

  char output[256];

  for (size_t i = 0; i < kNumTestInputs; ++i) {
    iree_tokenizer_normalizer_state_t* state = NULL;
    iree_status_t status = iree_tokenizer_normalizer_state_initialize(
        normalizer, state_buffer, &state);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }

    // Process the test input.
    const char* input = kTestInputs[i];
    size_t input_len = strlen(input);
    // Handle the embedded NUL case specially.
    if (i == kNumTestInputs - 1) {
      input_len = 3;  // "a\0b"
    }

    iree_host_size_t consumed = 0;
    iree_host_size_t written = 0;
    status = iree_tokenizer_normalizer_state_process(
        state, iree_make_string_view(input, input_len),
        iree_make_mutable_string_view(output, sizeof(output)),
        IREE_TOKENIZER_NORMALIZER_FLAG_NONE, &consumed, &written);
    iree_status_ignore(status);

    // Finalize.
    iree_host_size_t final_written = 0;
    status = iree_tokenizer_normalizer_state_finalize(
        state, iree_make_mutable_string_view(output, sizeof(output)),
        &final_written);
    iree_status_ignore(status);

    iree_tokenizer_normalizer_state_deinitialize(state);
  }

  free(state_buffer);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  // Treat fuzz input as normalizer JSON.
  iree_string_view_t normalizer_json =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Try parsing the JSON as a normalizer.
  iree_tokenizer_normalizer_t* normalizer = NULL;
  iree_status_t status = iree_tokenizer_huggingface_parse_normalizer(
      normalizer_json, iree_allocator_system(), &normalizer);

  if (iree_status_is_ok(status) && normalizer != NULL) {
    // Successfully parsed - exercise the normalizer with test inputs.
    exercise_normalizer(normalizer);
  }

  // Always ignore status - most fuzzed inputs will fail parsing.
  iree_status_ignore(status);

  // Free unconditionally - IREE free functions are null-safe.
  // This catches implementation bugs where errors don't clean up partial state.
  iree_tokenizer_normalizer_free(normalizer);

  return 0;
}
