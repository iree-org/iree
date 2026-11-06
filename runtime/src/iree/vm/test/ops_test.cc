// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ops.h requires the VM buffer declarations from api.h.
// clang-format off
#include "iree/vm/api.h"

#include "iree/vm/ops.h"
// clang-format on

#include <limits>

#include "iree/testing/gtest.h"

namespace {

template <typename T>
void CheckFloatComparisons(int32_t (*const (&compare)[8])(T, T)) {
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  struct Case {
    T lhs;
    T rhs;
    // EQ ordered/unordered, NE ordered/unordered, LT ordered/unordered,
    // LTE ordered/unordered.
    int32_t expected[8];
  };
  const Case cases[] = {
      {1, 1, {1, 1, 0, 0, 0, 0, 1, 1}},
      {1, 2, {0, 0, 1, 1, 1, 1, 1, 1}},
      {2, 1, {0, 0, 1, 1, 0, 0, 0, 0}},
      {-T(0), T(0), {1, 1, 0, 0, 0, 0, 1, 1}},
      {inf, inf, {1, 1, 0, 0, 0, 0, 1, 1}},
      {-inf, inf, {0, 0, 1, 1, 1, 1, 1, 1}},
      {nan, 1, {0, 1, 0, 1, 0, 1, 0, 1}},
      {1, nan, {0, 1, 0, 1, 0, 1, 0, 1}},
      {nan, nan, {0, 1, 0, 1, 0, 1, 0, 1}},
  };
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    for (size_t j = 0; j < 8; ++j) {
      EXPECT_EQ(compare[j](cases[i].lhs, cases[i].rhs), cases[i].expected[j])
          << "case " << i << ", predicate " << j;
    }
  }
}

TEST(VMOpsTest, Float32Comparisons) {
  int32_t (*const compare[])(float, float) = {
      vm_cmp_eq_f32o, vm_cmp_eq_f32u, vm_cmp_ne_f32o,  vm_cmp_ne_f32u,
      vm_cmp_lt_f32o, vm_cmp_lt_f32u, vm_cmp_lte_f32o, vm_cmp_lte_f32u};
  CheckFloatComparisons(compare);
}

TEST(VMOpsTest, Float64Comparisons) {
  int32_t (*const compare[])(double, double) = {
      vm_cmp_eq_f64o, vm_cmp_eq_f64u, vm_cmp_ne_f64o,  vm_cmp_ne_f64u,
      vm_cmp_lt_f64o, vm_cmp_lt_f64u, vm_cmp_lte_f64o, vm_cmp_lte_f64u};
  CheckFloatComparisons(compare);
}

}  // namespace
