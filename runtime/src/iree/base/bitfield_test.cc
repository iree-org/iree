// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

enum my_bitfield_e {
  MY_BITFIELD_NONE = 0,
  MY_BITFIELD_A = 1 << 0,
  MY_BITFIELD_B = 1 << 1,
  MY_BITFIELD_ALL = MY_BITFIELD_A | MY_BITFIELD_B,
};
typedef uint32_t my_bitfield_t;

template <size_t mapping_count>
std::string FormatBitfieldValue(
    uint32_t value,
    const iree_bitfield_string_mapping_t (&mappings)[mapping_count]) {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_bitfield_format_inline(value, mappings, mapping_count, &temp);
  return std::string(sv.data, sv.size);
}

// Tests general usage.
TEST(BitfieldTest, FormatBitfieldValue) {
  static const iree_bitfield_string_mapping_t mappings[] = {
      {MY_BITFIELD_A, IREE_SV("A")},
      {MY_BITFIELD_B, IREE_SV("B")},
  };
  EXPECT_EQ("", FormatBitfieldValue(MY_BITFIELD_NONE, mappings));
  EXPECT_EQ("A", FormatBitfieldValue(MY_BITFIELD_A, mappings));
  EXPECT_EQ("A|B",
            FormatBitfieldValue(MY_BITFIELD_A | MY_BITFIELD_B, mappings));
}

// Tests that empty mapping tables are fine.
TEST(BitfieldTest, FormatBitfieldValueEmpty) {
  static const iree_bitfield_string_mapping_t mappings[1] = {
      {0, IREE_SV("UNUSED")},
  };
  iree_bitfield_string_temp_t temp;
  auto sv = iree_bitfield_format_inline(MY_BITFIELD_NONE, mappings, 0, &temp);
  EXPECT_TRUE(iree_string_view_is_empty(sv));
}

// Tests that values not found in the mappings are still displayed.
TEST(BitfieldTest, FormatBitfieldValueUnhandledValues) {
  EXPECT_EQ("A|2h", FormatBitfieldValue(MY_BITFIELD_A | MY_BITFIELD_B,
                                        {
                                            {MY_BITFIELD_A, IREE_SV("A")},
                                        }));
}

// Tests priority order in the mapping table.
TEST(BitfieldTest, FormatBitfieldValuePriority) {
  // No priority, will do separate.
  EXPECT_EQ("A|B", FormatBitfieldValue(MY_BITFIELD_A | MY_BITFIELD_B,
                                       {
                                           {MY_BITFIELD_A, IREE_SV("A")},
                                           {MY_BITFIELD_B, IREE_SV("B")},
                                           {MY_BITFIELD_ALL, IREE_SV("ALL")},
                                       }));

  // Priority on the combined flag, use that instead.
  EXPECT_EQ("ALL", FormatBitfieldValue(MY_BITFIELD_A | MY_BITFIELD_B,
                                       {
                                           {MY_BITFIELD_ALL, IREE_SV("ALL")},
                                           {MY_BITFIELD_A, IREE_SV("A")},
                                           {MY_BITFIELD_B, IREE_SV("B")},
                                       }));
}

}  // namespace
}  // namespace iree
