// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree_pjrt/common/tensor_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

struct BroadcastTestData {
  BroadcastTestData(std::vector<int64_t> ishape, std::vector<int64_t> bshape,
           std::vector<int64_t> perms, int64_t esize = 4) {
    assert(ishape.size() == bshape.size());
    assert(ishape.size() == perms.size());

    element_size = esize;
    input_shape = ishape;
    broadcast_shape = bshape;
    permutation = perms;

    output_strides.resize(perms.size());
    output_shape.resize(perms.size());

    for (int i = 0, s = perms.size(); i < s; ++i) {
      output_shape[i] = broadcast_shape[perms[i]];
    }

    std::vector<int64_t> input_strides(ishape.size(), 0);
    int64_t stride = esize;
    for (int i = ishape.size() - 1; i >= 0; --i) {
      if (ishape[i] == 1) continue;
      input_strides[i] = stride;
      stride = stride * ishape[i];
    }

    for (int i = 0, s = perms.size(); i < s; ++i) {
      output_shape[i] = broadcast_shape[perms[i]];
    }

    for (int i = 0, s = perms.size(); i < s; ++i) {
      output_strides[i] = input_strides[perms[i]];
    }
  }

  void run_compute() {
    computed_input_shape.resize(input_shape.size());
    computed_permutation.resize(input_shape.size());

    iree::pjrt::computeBroadcastArgs(permutation.size(), element_size,
                                     output_strides.data(), output_shape.data(),
                                     computed_input_shape.data(),
                                     computed_permutation.data());
  }

  int64_t element_size;

  std::vector<int64_t> input_shape;
  std::vector<int64_t> broadcast_shape;
  std::vector<int64_t> permutation;

  std::vector<int64_t> output_shape;
  std::vector<int64_t> output_strides;

  std::vector<int64_t> computed_input_shape;
  std::vector<int64_t> computed_permutation;
};

TEST(BroadcastComputeTest, IdentityTest) {
  struct BroadcastTestData testdata({2, 3, 4}, {2, 3, 4}, {0, 1, 2}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(2, 3, 4));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(48, 16, 4));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(0, 1, 2));
}

TEST(BroadcastComputeTest, RotateLeftTest) {
  struct BroadcastTestData testdata({2, 3, 4}, {2, 3, 4}, {1, 2, 0}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(3, 4, 2));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(16, 4, 48));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(1, 2, 0));
}

TEST(BroadcastComputeTest, RotateRightTest) {
  struct BroadcastTestData testdata({2, 3, 4}, {2, 3, 4}, {2, 0, 1}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(4, 2, 3));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(4, 48, 16));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(2, 0, 1));
}

TEST(BroadcastComputeTest, PermutateTest) {
  struct BroadcastTestData testdata({2, 3, 4}, {2, 3, 4}, {2, 1, 0}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(4, 3, 2));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(4, 16, 48));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 3, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(2, 1, 0));
}

TEST(BroadcastComputeTest, BroadcastFirstTest) {
  struct BroadcastTestData testdata({1, 3, 4}, {2, 3, 4}, {0, 1, 2}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(2, 3, 4));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(0, 16, 4));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(1, 3, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(0, 1, 2));
}

TEST(BroadcastComputeTest, BroadcastMiddleTest) {
  struct BroadcastTestData testdata({2, 1, 4}, {2, 3, 4}, {0, 1, 2}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(2, 3, 4));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(16, 0, 4));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 1, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(0, 1, 2));
}

TEST(BroadcastComputeTest, BroadcastLastTest) {
  struct BroadcastTestData testdata({2, 3, 1}, {2, 3, 4}, {0, 1, 2}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(2, 3, 4));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(12, 4, 0));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 3, 1));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(0, 1, 2));
}

TEST(BroadcastComputeTest, PermutateBroadcastFirstTest) {
  struct BroadcastTestData testdata({1, 3, 4}, {2, 3, 4}, {0, 2, 1}, 4);

  ASSERT_THAT(testdata.output_shape, testing::ElementsAre(2, 4, 3));
  ASSERT_THAT(testdata.output_strides, testing::ElementsAre(0, 4, 16));
  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(1, 3, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(0, 2, 1));
}

TEST(BroadcastComputeTest, PermutateBroadcastMidTest) {
  struct BroadcastTestData testdata({2, 1, 4}, {2, 3, 4}, {2, 1, 0}, 4);

  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 1, 4));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(2, 1, 0));
}

TEST(BroadcastComputeTest, PermutateBroadcastLastTest) {
  struct BroadcastTestData testdata({2, 3, 1}, {2, 3, 4}, {1, 0, 2}, 4);

  testdata.run_compute();
  EXPECT_THAT(testdata.computed_input_shape, testing::ElementsAre(2, 3, 1));
  EXPECT_THAT(testdata.computed_permutation, testing::ElementsAre(1, 0, 2));
}
