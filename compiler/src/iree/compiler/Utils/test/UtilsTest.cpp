// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "iree/compiler/Utils/Permutation.h"

using namespace mlir::iree_compiler;
using namespace testing;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

TEST(Permutation, MakeMovePermutation) {
  EXPECT_THAT(makeMovePermutation(1, 0, 0), ElementsAre(0));
  EXPECT_THAT(makeMovePermutation(2, 0, 1), ElementsAre(1, 0));
  EXPECT_THAT(makeMovePermutation(5, 1, 3), ElementsAre(0, 2, 3, 1, 4));
  EXPECT_THAT(makeMovePermutation(3, 1, 2), ElementsAre(0, 2, 1));
  EXPECT_THAT(makeMovePermutation(3, 2, 0), ElementsAre(2, 0, 1));
}
