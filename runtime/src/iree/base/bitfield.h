// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_BITFIELD_H_
#define IREE_BASE_BITFIELD_H_

#include "iree/base/attributes.h"
#include "iree/base/string_builder.h"
#include "iree/base/string_view.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Bitfield utilities
//===----------------------------------------------------------------------===//

// Returns true if any bit from |rhs| is set in |lhs|.
#define iree_any_bit_set(lhs, rhs) (((lhs) & (rhs)) != 0)
// Returns true iff all bits from |rhs| are set in |lhs|.
#define iree_all_bits_set(lhs, rhs) (((lhs) & (rhs)) == (rhs))

// Maps bits within a bitfield to a string literal.
typedef struct iree_bitfield_string_mapping_t {
  uint32_t bits;
  iree_string_view_t string;
} iree_bitfield_string_mapping_t;

// Parses the bitfield |value| from a string.
// The provided |mappings| table is used for string lookup. Unknown values
// result in a failure.
//
// Usage:
//  // Static mapping table:
//  static const iree_bitfield_string_mapping_t my_bitfield_mappings[] = {
//    {MY_BITFIELD_ALL, IREE_SVL("ALL")},  // combined flags first
//    {MY_BITFIELD_A,   IREE_SVL("A")},
//    {MY_BITFIELD_B,   IREE_SVL("B")},
//    {MY_BITFIELD_C,   IREE_SVL("C")},
//  };
//
//  // Produces the bits MY_BITFIELD_A|MY_BITFIELD_B:
//  uint32_t value_ab = 0;
//  IREE_RETURN_IF_ERROR(iree_bitfield_parse(
//      IREE_SV("A|B"),
//      IREE_ARRAYSIZE(my_bitfield_mappings), my_bitfield_mappings,
//      &value_ab));
IREE_API_EXPORT iree_status_t iree_bitfield_parse(
    iree_string_view_t value, iree_host_size_t mapping_count,
    const iree_bitfield_string_mapping_t* mappings, uint32_t* out_value);

// Appends the formatted contents of the given bitfield |value|.
// Processes values in the order of the mapping table provided and will only
// use each bit once. Use this to prioritize combined flags over split ones.
//
// Usage:
//  // Static mapping table:
//  static const iree_bitfield_string_mapping_t my_bitfield_mappings[] = {
//    {MY_BITFIELD_ALL, IREE_SVL("ALL")},  // combined flags first
//    {MY_BITFIELD_A,   IREE_SVL("A")},
//    {MY_BITFIELD_B,   IREE_SVL("B")},
//    {MY_BITFIELD_C,   IREE_SVL("C")},
//  };
//
//  // Produces the string "A|B":
//  IREE_RETURN_IF_ERROR(iree_bitfield_format(
//      MY_BITFIELD_A | MY_BITFIELD_B,
//      IREE_ARRAYSIZE(my_bitfield_mappings), my_bitfield_mappings,
//      &string_builder));
//
//  // Produces the string "ALL":
//  IREE_RETURN_IF_ERROR(iree_bitfield_format(
//      MY_BITFIELD_A | MY_BITFIELD_B | MY_BITFIELD_C,
//      IREE_ARRAYSIZE(my_bitfield_mappings), my_bitfield_mappings,
//      &string_builder));
IREE_API_EXPORT iree_status_t
iree_bitfield_format(uint32_t value, iree_host_size_t mapping_count,
                     const iree_bitfield_string_mapping_t* mappings,
                     iree_string_builder_t* string_builder);

// Stack storage for iree_bitfield_format_inline temporary strings.
typedef struct iree_bitfield_string_temp_t {
  char buffer[128];
} iree_bitfield_string_temp_t;

// Appends the formatted contents of the given bitfield value.
// As with iree_bitfield_format only the storage for the formatted string is
// allocated inline on the stack.
//
// Usage:
//  // Produces the string "A|B":
//  iree_bitfield_string_temp_t temp;
//  iree_string_view_t my_str = iree_bitfield_format_inline(
//      MY_BITFIELD_A | MY_BITFIELD_B,
//      IREE_ARRAYSIZE(my_bitfield_mappings), my_bitfield_mappings,
//      &temp);
IREE_API_EXPORT iree_string_view_t
iree_bitfield_format_inline(uint32_t value, iree_host_size_t mapping_count,
                            const iree_bitfield_string_mapping_t* mappings,
                            iree_bitfield_string_temp_t* out_temp);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_BASE_BITFIELD_H_
