// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for regex compilation and execution.
//
// Tests the regex engine's robustness against:
// - Invalid regex syntax (unclosed brackets, invalid escapes)
// - Patterns that could cause exponential state explosion
// - Unicode edge cases (invalid UTF-8, surrogate pairs)
// - Very long patterns and inputs
// - Edge cases in lookahead assertions
// - Patterns with many alternations/groups
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/regex/compile.h"
#include "iree/tokenizer/regex/exec.h"

// Match callback that does nothing but count.
static iree_status_t count_callback(void* user_data,
                                    iree_tokenizer_regex_match_t match) {
  (void)user_data;
  (void)match;
  return iree_ok_status();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 4) return 0;

  // First 2 bytes encode pattern length as a fraction of remaining data.
  // This ensures we always have valid bounds.
  uint16_t pattern_frac = (uint16_t)(data[0] | (data[1] << 8));
  data += 2;
  size -= 2;

  // Calculate pattern length (max half the remaining data, min 1 byte).
  iree_host_size_t max_pattern_len = size / 2;
  if (max_pattern_len == 0) max_pattern_len = size;
  iree_host_size_t pattern_len = pattern_frac % (max_pattern_len + 1);
  if (pattern_len == 0 && size > 0) pattern_len = 1;

  iree_string_view_t pattern =
      iree_make_string_view(reinterpret_cast<const char*>(data), pattern_len);
  iree_string_view_t input = iree_make_string_view(
      reinterpret_cast<const char*>(data + pattern_len), size - pattern_len);

  //===--------------------------------------------------------------------===//
  // Test 1: Compile pattern (may fail gracefully)
  //===--------------------------------------------------------------------===//

  uint8_t* dfa_data = NULL;
  iree_host_size_t dfa_size = 0;
  iree_tokenizer_regex_compile_error_t compile_error = {0};
  iree_status_t status = iree_tokenizer_regex_compile(
      pattern, IREE_TOKENIZER_UTIL_REGEX_COMPILE_FLAG_NONE,
      iree_allocator_system(), &dfa_data, &dfa_size, &compile_error);

  if (!iree_status_is_ok(status)) {
    // Compilation failed - expected for malformed patterns.
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Load and validate DFA
  //===--------------------------------------------------------------------===//

  iree_tokenizer_regex_dfa_t dfa;
  status = iree_tokenizer_regex_dfa_load(
      iree_make_const_byte_span(dfa_data, dfa_size), &dfa);
  if (!iree_status_is_ok(status)) {
    // Load failed - internal error, should not happen.
    iree_status_ignore(status);
    iree_allocator_free(iree_allocator_system(), dfa_data);
    return 0;
  }

  // Validate the DFA for internal consistency.
  status = iree_tokenizer_regex_dfa_validate(&dfa);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    iree_allocator_free(iree_allocator_system(), dfa_data);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Execute DFA on input (streaming mode)
  //===--------------------------------------------------------------------===//

  iree_tokenizer_regex_exec_state_t exec_state;
  iree_tokenizer_regex_exec_initialize(&exec_state, &dfa);

  // Feed input in chunks to stress streaming edge cases.
  iree_host_size_t offset = 0;
  while (offset < input.size) {
    // Vary chunk sizes to stress boundary handling.
    iree_host_size_t chunk_size = ((offset % 7) + 1) * 16;
    if (chunk_size > input.size - offset) chunk_size = input.size - offset;

    iree_string_view_t chunk =
        iree_make_string_view(input.data + offset, chunk_size);
    status = iree_tokenizer_regex_exec_feed(&dfa, &exec_state, chunk, offset,
                                            NULL, count_callback, NULL);
    if (!iree_status_is_ok(status)) {
      // Feed failed - may happen for pathological patterns.
      iree_status_ignore(status);
      iree_allocator_free(iree_allocator_system(), dfa_data);
      return 0;
    }
    offset += chunk_size;
  }

  status = iree_tokenizer_regex_exec_finalize(&dfa, &exec_state, input.size,
                                              count_callback, NULL);
  iree_status_ignore(status);

  //===--------------------------------------------------------------------===//
  // Test 4: Execute DFA in one-shot mode
  //===--------------------------------------------------------------------===//

  status = iree_tokenizer_regex_exec(&dfa, input, count_callback, NULL);
  iree_status_ignore(status);

  //===--------------------------------------------------------------------===//
  // Test 5: Count matches
  //===--------------------------------------------------------------------===//

  iree_host_size_t match_count = 0;
  status = iree_tokenizer_regex_count_matches(&dfa, input, &match_count);
  iree_status_ignore(status);

  // Clean up.
  iree_allocator_free(iree_allocator_system(), dfa_data);

  return 0;
}
