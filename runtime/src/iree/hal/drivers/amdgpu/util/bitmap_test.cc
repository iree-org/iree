// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/bitmap.h"

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

TEST(BitmapTest, CalculateWords) {
  static_assert(IREE_HAL_AMDGPU_BITMAP_BITS_PER_WORD == 64,
                "assumes 64-bit words");
  EXPECT_EQ(iree_hal_amdgpu_bitmap_calculate_words(0), 0);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_calculate_words(1), 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_calculate_words(63), 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_calculate_words(64), 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_calculate_words(65), 2);
}

// Tests that a NULL storage pointer is allowed (as we shouldn't touch it).
TEST(BitmapTest, Empty) {
  iree_hal_amdgpu_bitmap_t bitmap = {0, NULL};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_set_all(bitmap);           // no-op
  iree_hal_amdgpu_bitmap_reset_all(bitmap);         // no-op
  iree_hal_amdgpu_bitmap_set_span(bitmap, 0, 0);    // no-op
  iree_hal_amdgpu_bitmap_reset_span(bitmap, 0, 0);  // no-op
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), 0);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, 0), 0);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(bitmap, 0, 0), 0);
}

TEST(BitmapTest, Test) {
  uint64_t words[] = {
      0ull | 0b1010,
      0ull | 0b1,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 1));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 2));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 3));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 0));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 1));
}

TEST(BitmapTest, Set63) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 63));
  iree_hal_amdgpu_bitmap_set(bitmap, 63);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 63));
  EXPECT_EQ(words[0], 0b1ull << 63);
  EXPECT_EQ(words[1], 0ull);
}

TEST(BitmapTest, Set64) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 64));
  iree_hal_amdgpu_bitmap_set(bitmap, 64);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 1ull);
}

TEST(BitmapTest, Set65) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 65));
  iree_hal_amdgpu_bitmap_set(bitmap, 65);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 65));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 0b10ull);
}

TEST(BitmapTest, Set73) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 73));
  iree_hal_amdgpu_bitmap_set(bitmap, 73);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 73));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 1ull << (73 - 64));
}

TEST(BitmapTest, SetPreserve) {
  uint64_t words[] = {
      0ull | (1ull << 2),
      0ull | (1ull << 3),
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 2));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 3));
  iree_hal_amdgpu_bitmap_set(bitmap, 0);
  iree_hal_amdgpu_bitmap_set(bitmap, 64 + 1);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 2));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 1));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 3));
  EXPECT_EQ(words[0], 0b101ull);
  EXPECT_EQ(words[1], 0b1010ull);
}

TEST(BitmapTest, SetSpanPrefix) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_set_span(bitmap, 0, 64 + 10 - 1);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], ~0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b0111111111ull);  // note tail bits are undefined
}

TEST(BitmapTest, SetSpanSuffix) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_set_span(bitmap, 64 + 10 - 1, 1);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b1000000000ull);  // note tail bits are undefined
}

TEST(BitmapTest, SetSpanSplit) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_set_span(bitmap, 63, 2);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], 0b1ull << 63);
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b1ull);  // note tail bits are undefined
}

TEST(BitmapTest, SetAll) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_set_all(bitmap);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], ~0x0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b1111111111ull);  // note tail bits are undefined
}

TEST(BitmapTest, Reset0) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  iree_hal_amdgpu_bitmap_set(bitmap, 0);
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  iree_hal_amdgpu_bitmap_reset(bitmap, 0);
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 0ull);
}

TEST(BitmapTest, Reset63) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  iree_hal_amdgpu_bitmap_set(bitmap, 63);
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 63));
  iree_hal_amdgpu_bitmap_reset(bitmap, 63);
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 63));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 0ull);
}

TEST(BitmapTest, Reset64) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 64));
  iree_hal_amdgpu_bitmap_set(bitmap, 64);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 1ull);
}

TEST(BitmapTest, Reset65) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 65));
  iree_hal_amdgpu_bitmap_set(bitmap, 65);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 65));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 0b10ull);
}

TEST(BitmapTest, Reset73) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 73));
  iree_hal_amdgpu_bitmap_set(bitmap, 73);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 73));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1], 1ull << (73 - 64));
}

TEST(BitmapTest, ResetPreserve) {
  uint64_t words[] = {
      0b101ull,
      0b1010ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 2));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 1));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 3));
  iree_hal_amdgpu_bitmap_reset(bitmap, 0);
  iree_hal_amdgpu_bitmap_reset(bitmap, 64 + 1);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 0));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 2));
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 1));
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_test(bitmap, 64 + 3));
  EXPECT_EQ(words[0], 0b100ull);
  EXPECT_EQ(words[1], 0b1000ull);
}

TEST(BitmapTest, ResetSpanPrefix) {
  uint64_t words[] = {
      ~0ull,
      ~0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_reset_span(bitmap, 0, 64 + 10 - 1);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b1000000000ull);  // note tail bits are undefined
}

TEST(BitmapTest, ResetSpanSuffix) {
  uint64_t words[] = {
      ~0ull,
      ~0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_reset_span(bitmap, 64 + 10 - 1, 1);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], ~0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b0111111111ull);  // note tail bits are undefined
}

TEST(BitmapTest, ResetSpanSplit) {
  uint64_t words[] = {
      ~0ull,
      ~0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_reset_span(bitmap, 63, 2);
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], ~(0b1ull << 63));
  EXPECT_EQ(words[1] & 0b1111111111ull,
            0b1111111110ull);  // note tail bits are undefined
}

TEST(BitmapTest, ResetSpanAll) {
  uint64_t words[] = {
      ~0ull,
      ~0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_reset_span(bitmap, 0, bitmap.bit_count);
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull, 0ull);  // note tail bits are undefined
}

TEST(BitmapTest, ResetAll) {
  uint64_t words[] = {
      ~0ull,
      ~0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  iree_hal_amdgpu_bitmap_reset_all(bitmap);
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(words[0], 0ull);
  EXPECT_EQ(words[1] & 0b1111111111ull, 0ull);  // note tail bits are undefined
}

TEST(BitmapTest, FindEmpty) {
  uint64_t words[] = {
      0ull,
      0ull,
  };
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), bitmap.bit_count);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, 0), 0);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, 10), 10);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(bitmap, 10,
                                                         bitmap.bit_count - 10),
            10);
}

TEST(BitmapTest, Find0) {
  uint64_t words[] = {
      0ull | 0b1,
      0ull,
  };
  const int bit_index = 0;
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), bit_index);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, bit_index),
            bit_index + 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(
                bitmap, bit_index, bitmap.bit_count - bit_index - 1),
            bit_index + 1);
}

TEST(BitmapTest, Find63) {
  uint64_t words[] = {
      0ull | (1ull << 63),
      0ull,
  };
  const int bit_index = 63;
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), bit_index);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, bit_index),
            bit_index + 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(
                bitmap, bit_index, bitmap.bit_count - bit_index - 1),
            bit_index + 1);
}

TEST(BitmapTest, Find64) {
  uint64_t words[] = {
      0ull,
      0ull | 0b1,
  };
  const int bit_index = 64;
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), bit_index);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, bit_index),
            bit_index + 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(
                bitmap, bit_index, bitmap.bit_count - bit_index - 1),
            bit_index + 1);
}

TEST(BitmapTest, Find67) {
  uint64_t words[] = {
      0ull,
      0ull | (0b1 << 3),
  };
  const int bit_index = 64 + 3;
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_FALSE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), bit_index);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, bit_index),
            bit_index + 1);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(
                bitmap, bit_index, bitmap.bit_count - bit_index - 1),
            bit_index + 1);
}

TEST(BitmapTest, Find73) {
  uint64_t words[] = {
      0ull,
      0ull | (0b1 << 10),
  };
  const int bit_index = 64 + 10;
  iree_hal_amdgpu_bitmap_t bitmap = {64 + 10, words};
  EXPECT_TRUE(iree_hal_amdgpu_bitmap_empty(bitmap));
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_set(bitmap, 0), bit_index);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset(bitmap, bit_index),
            bitmap.bit_count);
  EXPECT_EQ(iree_hal_amdgpu_bitmap_find_first_unset_span(bitmap, bit_index, 0),
            bitmap.bit_count);
}

}  // namespace
}  // namespace iree::hal::amdgpu
