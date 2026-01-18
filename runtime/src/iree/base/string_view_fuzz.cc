// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for string parsing utilities: integer/float parsing, device size
// parsing with units, bitfield parsing, pattern matching, and hex byte parsing.
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

// Sample bitfield mapping table for fuzzing iree_bitfield_parse.
// Uses realistic flag names similar to actual IREE usage.
static const iree_bitfield_string_mapping_t kTestBitfieldMappings[] = {
    {0x7, IREE_SVL("ALL")},        // Combined flag (A|B|C).
    {0x1, IREE_SVL("READ")},       // Bit 0.
    {0x2, IREE_SVL("WRITE")},      // Bit 1.
    {0x4, IREE_SVL("EXECUTE")},    // Bit 2.
    {0x8, IREE_SVL("DISCARD")},    // Bit 3.
    {0x10, IREE_SVL("MAPPABLE")},  // Bit 4.
    {0x20, IREE_SVL("COHERENT")},  // Bit 5.
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  //===--------------------------------------------------------------------===//
  // Integer parsing (signed and unsigned, various bases)
  //===--------------------------------------------------------------------===//

  {
    int32_t value_i32 = 0;
    (void)iree_string_view_atoi_int32(input, &value_i32);
    (void)iree_string_view_atoi_int32_base(input, 10, &value_i32);
    (void)iree_string_view_atoi_int32_base(input, 16, &value_i32);
    (void)iree_string_view_atoi_int32_base(input, 2, &value_i32);
  }

  {
    uint32_t value_u32 = 0;
    (void)iree_string_view_atoi_uint32(input, &value_u32);
    (void)iree_string_view_atoi_uint32_base(input, 10, &value_u32);
    (void)iree_string_view_atoi_uint32_base(input, 16, &value_u32);
    (void)iree_string_view_atoi_uint32_base(input, 2, &value_u32);
  }

  {
    int64_t value_i64 = 0;
    (void)iree_string_view_atoi_int64(input, &value_i64);
    (void)iree_string_view_atoi_int64_base(input, 10, &value_i64);
    (void)iree_string_view_atoi_int64_base(input, 16, &value_i64);
    (void)iree_string_view_atoi_int64_base(input, 2, &value_i64);
  }

  {
    uint64_t value_u64 = 0;
    (void)iree_string_view_atoi_uint64(input, &value_u64);
    (void)iree_string_view_atoi_uint64_base(input, 10, &value_u64);
    (void)iree_string_view_atoi_uint64_base(input, 16, &value_u64);
    (void)iree_string_view_atoi_uint64_base(input, 2, &value_u64);
  }

  //===--------------------------------------------------------------------===//
  // Floating point parsing
  //===--------------------------------------------------------------------===//

  {
    float value_f32 = 0.0f;
    (void)iree_string_view_atof(input, &value_f32);
  }

  {
    double value_f64 = 0.0;
    (void)iree_string_view_atod(input, &value_f64);
  }

  //===--------------------------------------------------------------------===//
  // Device size parsing with units (e.g., "1kb", "2mib", "3gb")
  //===--------------------------------------------------------------------===//

  {
    iree_device_size_t device_size = 0;
    iree_status_t status =
        iree_string_view_parse_device_size(input, &device_size);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Bitfield parsing
  //===--------------------------------------------------------------------===//

  {
    uint32_t bitfield_value = 0;
    iree_status_t status =
        iree_bitfield_parse(input, IREE_ARRAYSIZE(kTestBitfieldMappings),
                            kTestBitfieldMappings, &bitfield_value);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Pattern matching (wildcard patterns with * and ?)
  //===--------------------------------------------------------------------===//

  // Use the first half as value and second half as pattern.
  if (size >= 2) {
    size_t mid = size / 2;
    iree_string_view_t value =
        iree_make_string_view(reinterpret_cast<const char*>(data), mid);
    iree_string_view_t pattern = iree_make_string_view(
        reinterpret_cast<const char*>(data + mid), size - mid);
    (void)iree_string_view_match_pattern(value, pattern);
  }

  // Also test pattern matching with specific patterns that stress recursion.
  (void)iree_string_view_match_pattern(input, IREE_SV("*"));
  (void)iree_string_view_match_pattern(input, IREE_SV("?*?"));
  (void)iree_string_view_match_pattern(input, IREE_SV("***"));

  //===--------------------------------------------------------------------===//
  // Hex byte parsing
  //===--------------------------------------------------------------------===//

  // Parse up to 64 bytes of hex data.
  {
    uint8_t hex_buffer[64] = {0};
    (void)iree_string_view_parse_hex_bytes(input, sizeof(hex_buffer),
                                           hex_buffer);
  }

  // Try parsing various sizes to test boundary conditions.
  for (size_t parse_size = 1; parse_size <= 8; ++parse_size) {
    uint8_t small_buffer[8] = {0};
    (void)iree_string_view_parse_hex_bytes(input, parse_size, small_buffer);
  }

  //===--------------------------------------------------------------------===//
  // String view operations that process the data
  //===--------------------------------------------------------------------===//

  (void)iree_string_view_trim(input);

  // Split operations with various split characters.
  {
    iree_string_view_t lhs, rhs;
    (void)iree_string_view_split(input, '|', &lhs, &rhs);
    (void)iree_string_view_split(input, '=', &lhs, &rhs);
    (void)iree_string_view_split(input, ',', &lhs, &rhs);
    (void)iree_string_view_split(input, ':', &lhs, &rhs);
  }

  // Find operations.
  if (size > 0) {
    char search_char = static_cast<char>(data[0]);
    (void)iree_string_view_find_char(input, search_char, 0);

    if (size > 1) {
      iree_string_view_t search_set =
          iree_make_string_view(reinterpret_cast<const char*>(data), size / 2);
      (void)iree_string_view_find_first_of(input, search_set, 0);
      (void)iree_string_view_find_last_of(input, search_set, SIZE_MAX);
    }
  }

  // Comparison operations.
  if (size >= 2) {
    size_t mid = size / 2;
    iree_string_view_t left =
        iree_make_string_view(reinterpret_cast<const char*>(data), mid);
    iree_string_view_t right = iree_make_string_view(
        reinterpret_cast<const char*>(data + mid), size - mid);
    (void)iree_string_view_equal(left, right);
    (void)iree_string_view_equal_case(left, right);
    (void)iree_string_view_compare(left, right);
    (void)iree_string_view_starts_with(left, right);
    (void)iree_string_view_ends_with(left, right);
  }

  return 0;
}
