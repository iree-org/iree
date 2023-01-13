// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <list>
#include <string>
#include <tuple>
#include <vector>

#include "iree/compiler/Utils/ADTExtras.h"
#include "iree/testing/gtest.h"

using ::testing::ElementsAre;

using mlir::iree_compiler::enumerate_zip_equal;

namespace {

TEST(ADTExtras, EnumerateZipEqualVector) {
  llvm::SmallVector<int> ints = {0, 1, 2};
  std::vector<char> chars = {'a', 'b', 'c'};

  using Values = std::tuple<size_t, int, char>;
  EXPECT_THAT(
      enumerate_zip_equal(ints, chars),
      ElementsAre(Values(0, 0, 'a'), Values(1, 1, 'b'), Values(2, 2, 'c')));

  for (auto [i, a, b] : enumerate_zip_equal(ints, chars)) {
    EXPECT_LT(i, size_t{3});
    EXPECT_LT(a, 3);
    EXPECT_LT(b, 'd');
    a = -1;
    b = 'x';
  }

  EXPECT_THAT(
      enumerate_zip_equal(ints, chars),
      ElementsAre(Values(0, -1, 'x'), Values(1, -1, 'x'), Values(2, -1, 'x')));
}

TEST(ADTExtras, EnumerateZipEqualVectorBool) {
  // `std::vector<bool>` is interesting because exposes elements via a custom
  // reference wrapper type.
  std::vector<bool> bools1 = {true, false, true};
  std::vector<bool> bools2 = bools1;

  using Values = std::tuple<size_t, bool, bool>;
  EXPECT_THAT(enumerate_zip_equal(bools1, bools2),
              ElementsAre(Values(0, true, true), Values(1, false, false),
                          Values(2, true, true)));

  for (auto [i, a, b] : enumerate_zip_equal(bools1, bools2)) {
    EXPECT_LT(i, size_t{3});
    EXPECT_EQ(a, b);
    // Check that we can modify the elements behind the reference wrapper type.
    a = true;
    b = false;
  }

  EXPECT_THAT(enumerate_zip_equal(bools1, bools2),
              ElementsAre(Values(0, true, false), Values(1, true, false),
                          Values(2, true, false)));
}

TEST(ADTExtras, EnumerateZipEqualEmpty) {
  std::list<int> a;
  EXPECT_THAT(enumerate_zip_equal(a, a, a), ElementsAre());

  for (auto it : enumerate_zip_equal(a, a, a)) {
    (void)it;
    FAIL() << "This loop body should not execute";
  }
}
}  // namespace
