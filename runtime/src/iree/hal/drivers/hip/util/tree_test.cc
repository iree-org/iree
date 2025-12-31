// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/hip/util/tree.h"

#include "iree/testing/gtest.h"

class RedBlackTreeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_allocator_t allocator = iree_allocator_system();
    iree_hal_hip_util_tree_initialize(allocator, sizeof(int), initial_cache,
                                      1024, &tree_);
  }

  void TearDown() override { iree_hal_hip_util_tree_deinitialize(&tree_); }

  iree_hal_hip_util_tree_t tree_;
  uint8_t initial_cache[1024];
};

TEST_F(RedBlackTreeTest, initialize) {
  EXPECT_EQ(iree_hal_hip_util_tree_size(&tree_), 0);
}

TEST_F(RedBlackTreeTest, insert) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  EXPECT_EQ(iree_hal_hip_util_tree_insert(&tree_, 10, &node), iree_ok_status());
  EXPECT_EQ(iree_hal_hip_util_tree_size(&tree_), 1);
  EXPECT_EQ(iree_hal_hip_util_tree_node_get_key(node), 10);
}

TEST_F(RedBlackTreeTest, get) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  EXPECT_NE(iree_hal_hip_util_tree_get(&tree_, 10), nullptr);
  EXPECT_EQ(iree_hal_hip_util_tree_get(&tree_, 20), nullptr);
}

TEST_F(RedBlackTreeTest, delete) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  iree_hal_hip_util_tree_erase(&tree_, node);
  EXPECT_EQ(iree_hal_hip_util_tree_get(&tree_, 10), nullptr);
  EXPECT_EQ(iree_hal_hip_util_tree_size(&tree_), 0);
}

TEST_F(RedBlackTreeTest, walk) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  static_cast<int*>(iree_hal_hip_util_tree_node_get_value(node))[0] = 10;
  iree_hal_hip_util_tree_insert(&tree_, 20, &node);
  static_cast<int*>(iree_hal_hip_util_tree_node_get_value(node))[0] = 20;
  iree_hal_hip_util_tree_insert(&tree_, 30, &node);
  static_cast<int*>(iree_hal_hip_util_tree_node_get_value(node))[0] = 30;

  int sum = 0;
  auto callback = [](iree_hal_hip_util_tree_node_t* node,
                     void* user_data) -> bool {
    int* sum = static_cast<int*>(user_data);
    EXPECT_EQ(*static_cast<int*>(iree_hal_hip_util_tree_node_get_value(node)),
              iree_hal_hip_util_tree_node_get_key(node));
    *sum += *static_cast<int*>(iree_hal_hip_util_tree_node_get_value(node));
    return true;
  };
  iree_hal_hip_util_tree_walk(&tree_, IREE_TREE_WALK_PREORDER, callback, &sum);
  EXPECT_EQ(sum, 60);
}

TEST_F(RedBlackTreeTest, boundary_conditions) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  iree_hal_hip_util_tree_insert(&tree_, 20, &node);
  iree_hal_hip_util_tree_insert(&tree_, 30, &node);

  EXPECT_EQ(
      iree_hal_hip_util_tree_node_get_key(iree_hal_hip_util_tree_first(&tree_)),
      10);
  EXPECT_EQ(
      iree_hal_hip_util_tree_node_get_key(iree_hal_hip_util_tree_last(&tree_)),
      30);
  EXPECT_EQ(iree_hal_hip_util_tree_node_get_key(
                iree_hal_hip_util_tree_lower_bound(&tree_, 15)),
            20);
  EXPECT_EQ(iree_hal_hip_util_tree_node_get_key(
                iree_hal_hip_util_tree_upper_bound(&tree_, 15)),
            20);
}

TEST_F(RedBlackTreeTest, move_node) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  iree_hal_hip_util_tree_move_node(&tree_, node, 20);
  EXPECT_EQ(iree_hal_hip_util_tree_get(&tree_, 10), nullptr);
  EXPECT_NE(iree_hal_hip_util_tree_get(&tree_, 20), nullptr);
}

TEST_F(RedBlackTreeTest, in_order_iterators) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  iree_hal_hip_util_tree_insert(&tree_, 20, &node);
  iree_hal_hip_util_tree_insert(&tree_, 30, &node);

  std::vector<int> keys;
  for (iree_hal_hip_util_tree_node_t* node =
           iree_hal_hip_util_tree_first(&tree_);
       node != nullptr; node = iree_hal_hip_util_tree_node_next(node)) {
    keys.push_back(iree_hal_hip_util_tree_node_get_key(node));
  }

  EXPECT_EQ(keys.size(), 3);
  EXPECT_EQ(keys[0], 10);
  EXPECT_EQ(keys[1], 20);
  EXPECT_EQ(keys[2], 30);
}

TEST_F(RedBlackTreeTest, in_order_iterators_last) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  iree_hal_hip_util_tree_insert(&tree_, 20, &node);
  iree_hal_hip_util_tree_insert(&tree_, 30, &node);

  std::vector<int> keys;
  for (iree_hal_hip_util_tree_node_t* node =
           iree_hal_hip_util_tree_last(&tree_);
       node != nullptr; node = iree_hal_hip_util_tree_node_prev(node)) {
    keys.push_back(iree_hal_hip_util_tree_node_get_key(node));
  }

  EXPECT_EQ(keys.size(), 3);
  EXPECT_EQ(keys[0], 30);
  EXPECT_EQ(keys[1], 20);
  EXPECT_EQ(keys[2], 10);
}

class RedBlackTreeWalkTest
    : public RedBlackTreeTest,
      public ::testing::WithParamInterface<iree_hal_hip_util_tree_walk_type_t> {
};

TEST_P(RedBlackTreeWalkTest, walk) {
  iree_hal_hip_util_tree_node_t* node = NULL;
  iree_hal_hip_util_tree_insert(&tree_, 10, &node);
  iree_hal_hip_util_tree_insert(&tree_, 20, &node);
  iree_hal_hip_util_tree_insert(&tree_, 30, &node);

  std::vector<int> keys;
  auto callback = [](iree_hal_hip_util_tree_node_t* node,
                     void* user_data) -> bool {
    auto* keys = static_cast<std::vector<int>*>(user_data);
    keys->push_back(iree_hal_hip_util_tree_node_get_key(node));
    return true;
  };
  iree_hal_hip_util_tree_walk(&tree_, GetParam(), callback, &keys);

  if (GetParam() == IREE_TREE_WALK_INORDER) {
    EXPECT_EQ(keys.size(), 3);
    EXPECT_EQ(keys[0], 10);
    EXPECT_EQ(keys[1], 20);
    EXPECT_EQ(keys[2], 30);
  } else if (GetParam() == IREE_TREE_WALK_PREORDER) {
    EXPECT_EQ(keys.size(), 3);
    EXPECT_EQ(keys[0], 20);  // Assuming 20 is the root after balancing
    EXPECT_EQ(keys[1], 10);
    EXPECT_EQ(keys[2], 30);
  } else if (GetParam() == IREE_TREE_WALK_POSTORDER) {
    EXPECT_EQ(keys.size(), 3);
    EXPECT_EQ(keys[0], 10);
    EXPECT_EQ(keys[1], 30);
    EXPECT_EQ(keys[2], 20);  // Assuming 20 is the root after balancing
  }
}

INSTANTIATE_TEST_SUITE_P(WalkTypes, RedBlackTreeWalkTest,
                         ::testing::Values(IREE_TREE_WALK_PREORDER,
                                           IREE_TREE_WALK_INORDER,
                                           IREE_TREE_WALK_POSTORDER));
