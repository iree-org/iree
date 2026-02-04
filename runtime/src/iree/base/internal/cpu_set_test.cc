// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/cpu_set.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(CpuSet, Generalities) {
  const iree_allocator_t allocator = iree_allocator_system();
  const std::vector<iree_host_size_t> bit_counts = {0,  1,  56,   57,
                                                    64, 65, 1000, 1024};
  iree_host_size_t bit_counts_count = bit_counts.size();
  std::vector<iree_cpu_set_t> cpu_sets(bit_counts_count);

  for (iree_host_size_t i = 0; i < bit_counts_count; ++i) {
    IREE_EXPECT_OK(
        iree_cpu_set_allocate(allocator, bit_counts[i], &cpu_sets[i]));
    EXPECT_EQ(iree_cpu_set_is_inline(cpu_sets[i]), (bit_counts[i] <= 56));
    EXPECT_EQ(iree_cpu_set_get_bit_size(cpu_sets[i]), bit_counts[i]);
    for (iree_host_size_t b = 0; b < bit_counts[i]; ++b) {
      EXPECT_EQ(b, iree_cpu_set_population_count(cpu_sets[i]));
      EXPECT_EQ(false, iree_cpu_set_get_bit(cpu_sets[i], b));
      iree_cpu_set_set_bit(&cpu_sets[i], b);
      EXPECT_EQ(true, iree_cpu_set_get_bit(cpu_sets[i], b));
      EXPECT_EQ(b + 1, iree_cpu_set_population_count(cpu_sets[i]));
    }
    for (iree_host_size_t b = 0; b < bit_counts[i]; ++b) {
      EXPECT_EQ(bit_counts[i] - b, iree_cpu_set_population_count(cpu_sets[i]));
      EXPECT_EQ(true, iree_cpu_set_get_bit(cpu_sets[i], b));
      iree_cpu_set_clear_bit(&cpu_sets[i], b);
      EXPECT_EQ(false, iree_cpu_set_get_bit(cpu_sets[i], b));
      EXPECT_EQ(bit_counts[i] - b - 1,
                iree_cpu_set_population_count(cpu_sets[i]));
    }
    EXPECT_EQ(iree_cpu_set_get_const_words(&cpu_sets[i]),
              iree_cpu_set_get_mutable_words(&cpu_sets[i]));

    for (iree_host_size_t b = 1; b < bit_counts[i]; b += 2) {
      iree_cpu_set_set_bit(&cpu_sets[i], b);
    }
    EXPECT_EQ(bit_counts[i] / 2, iree_cpu_set_population_count(cpu_sets[i]));
    iree_cpu_set_clear(&cpu_sets[i]);
    EXPECT_EQ(0, iree_cpu_set_population_count(cpu_sets[i]));
  }

  for (iree_host_size_t i = 0; i < bit_counts_count; ++i) {
    for (iree_host_size_t j = 0; j < bit_counts_count; ++j) {
      EXPECT_EQ(i == j, iree_cpu_set_equal(cpu_sets[i], cpu_sets[j]));
    }
  }

  for (iree_host_size_t i = 0; i < bit_counts_count; ++i) {
    iree_cpu_set_free(allocator, &cpu_sets[i]);
  }
}

TEST(CpuSet, InlineExample) {
  const iree_allocator_t allocator = iree_allocator_system();
  iree_cpu_set_t s1, s2;
  IREE_EXPECT_OK(iree_cpu_set_allocate(allocator, 56, &s1));
  IREE_EXPECT_OK(iree_cpu_set_allocate(allocator, 56, &s2));
  EXPECT_EQ(iree_cpu_set_get_bit_size(s1), 56);
  EXPECT_TRUE(iree_cpu_set_is_inline(s1));
  EXPECT_TRUE(iree_cpu_set_equal(s1, s2));
  EXPECT_EQ(0, iree_cpu_set_population_count(s1));
  iree_cpu_set_set_bit(&s1, 55);
  EXPECT_EQ(1, iree_cpu_set_population_count(s1));
  EXPECT_TRUE(iree_cpu_set_get_const_words(&s1)[0] & (1ull << (55)));
  EXPECT_FALSE(iree_cpu_set_equal(s1, s2));
  iree_cpu_set_get_mutable_words(&s2)[0] |= 1ull << 55;
  EXPECT_TRUE(iree_cpu_set_get_bit(s2, 55));
  EXPECT_TRUE(iree_cpu_set_equal(s1, s2));
  iree_cpu_set_free(allocator, &s1);
  iree_cpu_set_free(allocator, &s2);
}

TEST(CpuSet, OutOfLineExample) {
  const iree_allocator_t allocator = iree_allocator_system();
  iree_cpu_set_t s1, s2;
  IREE_EXPECT_OK(iree_cpu_set_allocate(allocator, 200, &s1));
  IREE_EXPECT_OK(iree_cpu_set_allocate(allocator, 200, &s2));
  EXPECT_EQ(iree_cpu_set_get_bit_size(s1), 200);
  EXPECT_FALSE(iree_cpu_set_is_inline(s1));
  EXPECT_TRUE(iree_cpu_set_equal(s1, s2));
  EXPECT_EQ(0, iree_cpu_set_population_count(s1));
  iree_cpu_set_set_bit(&s1, 199);
  EXPECT_EQ(1, iree_cpu_set_population_count(s1));
  EXPECT_TRUE(iree_cpu_set_get_const_words(&s1)[199 / 64] &
              (1ull << (199 % 64)));
  EXPECT_FALSE(iree_cpu_set_equal(s1, s2));
  iree_cpu_set_get_mutable_words(&s2)[199 / 64] |= 1ull << (199 % 64);
  EXPECT_TRUE(iree_cpu_set_get_bit(s2, 199));
  EXPECT_TRUE(iree_cpu_set_equal(s1, s2));
  iree_cpu_set_free(allocator, &s1);
  iree_cpu_set_free(allocator, &s2);
}

}  // namespace
