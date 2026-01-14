// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for Unicode utilities: UTF-8 decoding/validation, category
// classification, case folding, and composition. Includes invariant assertions
// that crash on consistency violations.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/internal/unicode.h"

// Invariant assertion that crashes on failure.
// We use __builtin_trap() to get a clean crash for the fuzzer to detect.
#define FUZZ_ASSERT(condition) \
  do {                         \
    if (!(condition)) {        \
      __builtin_trap();        \
    }                          \
  } while (0)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  // Test UTF-8 validation and counting.
  (void)iree_unicode_utf8_validate(input);
  (void)iree_unicode_utf8_codepoint_count(input);

  // Test incomplete tail detection.
  (void)iree_unicode_utf8_incomplete_tail_length(input.data, input.size);

  // Decode all codepoints and test classification/transformation functions.
  iree_host_size_t position = 0;
  while (position < input.size) {
    uint32_t codepoint = iree_unicode_utf8_decode(input, &position);

    // Category classification.
    (void)iree_unicode_category(codepoint);
    (void)iree_unicode_is_letter(codepoint);
    (void)iree_unicode_is_mark(codepoint);
    (void)iree_unicode_is_number(codepoint);
    (void)iree_unicode_is_punctuation(codepoint);
    (void)iree_unicode_is_symbol(codepoint);
    (void)iree_unicode_is_separator(codepoint);
    (void)iree_unicode_is_other(codepoint);
    (void)iree_unicode_is_whitespace(codepoint);
    (void)iree_unicode_is_control(codepoint);
    (void)iree_unicode_is_cjk(codepoint);
    (void)iree_unicode_is_hiragana(codepoint);
    (void)iree_unicode_is_katakana(codepoint);
    (void)iree_unicode_is_hangul(codepoint);

    // Case folding.
    (void)iree_unicode_to_lower(codepoint);
    (void)iree_unicode_to_upper(codepoint);

    // NFD decomposition.
    (void)iree_unicode_nfd_base(codepoint);

    // Canonical Combining Class.
    (void)iree_unicode_ccc(codepoint);

    // UTF-8 encoding (roundtrip test).
    char encode_buffer[4];
    (void)iree_unicode_utf8_encode(codepoint, encode_buffer);
    (void)iree_unicode_utf8_encoded_length(codepoint);
  }

  //===--------------------------------------------------------------------===//
  // Direct codepoint testing (raw byte interpretation)
  //===--------------------------------------------------------------------===//
  // Interpret every 4 bytes as a raw uint32_t codepoint to test the full
  // codepoint space including invalid ranges (>0x10FFFF, surrogates).
  // This exercises table lookup binary search with boundary values.
  for (size_t i = 0; i + 4 <= size; i += 4) {
    uint32_t codepoint = (static_cast<uint32_t>(data[i]) << 24) |
                         (static_cast<uint32_t>(data[i + 1]) << 16) |
                         (static_cast<uint32_t>(data[i + 2]) << 8) |
                         static_cast<uint32_t>(data[i + 3]);

    // Test all classification functions on arbitrary codepoint values.
    iree_unicode_category_t category = iree_unicode_category(codepoint);
    bool is_letter = iree_unicode_is_letter(codepoint);
    bool is_mark = iree_unicode_is_mark(codepoint);
    bool is_number = iree_unicode_is_number(codepoint);
    bool is_punctuation = iree_unicode_is_punctuation(codepoint);
    bool is_symbol = iree_unicode_is_symbol(codepoint);
    bool is_separator = iree_unicode_is_separator(codepoint);
    bool is_other = iree_unicode_is_other(codepoint);
    (void)iree_unicode_is_whitespace(codepoint);
    (void)iree_unicode_is_control(codepoint);
    (void)iree_unicode_is_cjk(codepoint);
    (void)iree_unicode_is_hiragana(codepoint);
    (void)iree_unicode_is_katakana(codepoint);
    (void)iree_unicode_is_hangul(codepoint);

    // Invariant: category classification consistency.
    // If is_X returns true, the corresponding category bit must be set.
    if (is_letter) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_LETTER) != 0);
    }
    if (is_mark) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_MARK) != 0);
    }
    if (is_number) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_NUMBER) != 0);
    }
    if (is_punctuation) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_PUNCTUATION) != 0);
    }
    if (is_symbol) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_SYMBOL) != 0);
    }
    if (is_separator) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_SEPARATOR) != 0);
    }
    if (is_other) {
      FUZZ_ASSERT((category & IREE_UNICODE_CATEGORY_OTHER) != 0);
    }

    // Test case folding and NFD.
    uint32_t lower = iree_unicode_to_lower(codepoint);
    uint32_t upper = iree_unicode_to_upper(codepoint);
    uint32_t nfd = iree_unicode_nfd_base(codepoint);
    (void)iree_unicode_ccc(codepoint);

    // Invariant: case folding idempotency.
    // Applying the same case operation twice should yield the same result.
    FUZZ_ASSERT(iree_unicode_to_lower(lower) == lower);
    FUZZ_ASSERT(iree_unicode_to_upper(upper) == upper);

    // Note: NFD decomposition may be multi-level (e.g., ẳ → ạ → a), so
    // nfd_base is NOT necessarily idempotent. Instead, verify it converges
    // to a fixed point within a reasonable number of steps.
    uint32_t nfd_current = nfd;
    for (int depth = 0; depth < 10; ++depth) {
      uint32_t nfd_next = iree_unicode_nfd_base(nfd_current);
      if (nfd_next == nfd_current) break;  // Reached fixed point.
      nfd_current = nfd_next;
    }
    // After at most 10 iterations, we must have reached a fixed point.
    FUZZ_ASSERT(iree_unicode_nfd_base(nfd_current) == nfd_current);

    // Invariant: encode/decode roundtrip for valid codepoints.
    int encoded_length = iree_unicode_utf8_encoded_length(codepoint);
    if (encoded_length > 0) {
      char encode_buffer[4];
      int actual_length = iree_unicode_utf8_encode(codepoint, encode_buffer);

      // Invariant: encoded_length and encode must agree.
      FUZZ_ASSERT(encoded_length == actual_length);

      // Decode what we just encoded and verify roundtrip.
      iree_string_view_t encoded = iree_make_string_view(
          encode_buffer, static_cast<iree_host_size_t>(actual_length));
      iree_host_size_t decode_position = 0;
      uint32_t decoded = iree_unicode_utf8_decode(encoded, &decode_position);

      // Invariant: roundtrip must recover the original codepoint.
      FUZZ_ASSERT(decoded == codepoint);
      FUZZ_ASSERT(decode_position ==
                  static_cast<iree_host_size_t>(actual_length));
    }
  }

  //===--------------------------------------------------------------------===//
  // Composition testing with status verification
  //===--------------------------------------------------------------------===//
  // Test composition on valid UTF-8 sequences, verifying status codes.
  if (iree_unicode_utf8_validate(input)) {
    // Allocate output buffer (composition can only shrink).
    char* compose_buffer = new char[size + 1];
    iree_host_size_t out_length = 0;
    iree_status_t status =
        iree_unicode_compose(input, compose_buffer, size + 1, &out_length);

    // Status must be OK or RESOURCE_EXHAUSTED (for very long combining seqs).
    // Any other status indicates a bug in the compose function.
    FUZZ_ASSERT(iree_status_is_ok(status) ||
                iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED);

    if (iree_status_is_ok(status)) {
      // Invariant: output length must not exceed input length.
      // Composition can only shrink (combining base + mark -> precomposed).
      FUZZ_ASSERT(out_length <= input.size);

      // The output should also be valid UTF-8.
      iree_string_view_t output =
          iree_make_string_view(compose_buffer, out_length);
      FUZZ_ASSERT(iree_unicode_utf8_validate(output));
    } else {
      iree_status_ignore(status);
    }
    delete[] compose_buffer;
  }

  // Test pairwise composition with interpreted codepoints.
  if (size >= 8) {
    uint32_t base = (static_cast<uint32_t>(data[0]) << 24) |
                    (static_cast<uint32_t>(data[1]) << 16) |
                    (static_cast<uint32_t>(data[2]) << 8) |
                    static_cast<uint32_t>(data[3]);
    uint32_t combining = (static_cast<uint32_t>(data[4]) << 24) |
                         (static_cast<uint32_t>(data[5]) << 16) |
                         (static_cast<uint32_t>(data[6]) << 8) |
                         static_cast<uint32_t>(data[7]);
    (void)iree_unicode_compose_pair(base, combining);
  }

  return 0;
}
