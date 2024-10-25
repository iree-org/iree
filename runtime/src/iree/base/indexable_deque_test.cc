// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/indexable_deque.h"

#include "iree/testing/gtest.h"

IREE_TYPED_INDEXABLE_QUEUE_WRAPPER(test_queue, int32_t, 4);

class IndexableDequeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    test_queue_initialize(&queue_, iree_allocator_system());
  }

  void TearDown() override { test_queue_deinitialize(&queue_); }

  test_queue_indexable_queue_t queue_;
};

TEST_F(IndexableDequeTest, initialize) {
  EXPECT_EQ(queue_.element_count, 0);
  EXPECT_EQ(queue_.capacity, 4);
  EXPECT_EQ(queue_.element_size, sizeof(int));
}

TEST_F(IndexableDequeTest, push_back) {
  int value = 42;
  IREE_CHECK_OK(test_queue_push_back(&queue_, value));
  EXPECT_EQ(queue_.element_count, 1);
  EXPECT_EQ(test_queue_at(&queue_, 0), value);
}

TEST_F(IndexableDequeTest, push_back_and_expand) {
  for (int i = 0; i < 5; ++i) {
    IREE_CHECK_OK(test_queue_push_back(&queue_, i));
  }
  EXPECT_EQ(queue_.element_count, 5);
  EXPECT_GT(queue_.capacity, 4);
}

TEST_F(IndexableDequeTest, pop_front) {
  IREE_CHECK_OK(test_queue_push_back(&queue_, 12));
  IREE_CHECK_OK(test_queue_push_back(&queue_, 15));
  test_queue_pop_front(&queue_, 1);
  EXPECT_EQ(queue_.element_count, 1);
  EXPECT_EQ(test_queue_at(&queue_, 0), 15);
}

TEST_F(IndexableDequeTest, at) {
  int value1 = 42;
  int value2 = 84;
  IREE_CHECK_OK(test_queue_push_back(&queue_, value1));
  IREE_CHECK_OK(test_queue_push_back(&queue_, value2));
  EXPECT_EQ(test_queue_at(&queue_, 0), value1);
  EXPECT_EQ(test_queue_at(&queue_, 1), value2);
}

TEST_F(IndexableDequeTest, cycle_around_queue) {
  for (int i = 0; i < 4; ++i) {
    IREE_CHECK_OK(test_queue_push_back(&queue_, i));
  }
  for (int i = 0; i < 4; ++i) {
    test_queue_pop_front(&queue_, 1);
    IREE_CHECK_OK(test_queue_push_back(&queue_, i + 4));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(test_queue_at(&queue_, i), i + 4);
  }
  EXPECT_EQ(queue_.element_count, 4);
}

TEST_F(IndexableDequeTest, cycle_around_queue_twice) {
  for (int i = 0; i < 4; ++i) {
    IREE_CHECK_OK(test_queue_push_back(&queue_, i));
  }
  for (int i = 0; i < 8; ++i) {
    test_queue_pop_front(&queue_, 1);
    IREE_CHECK_OK(test_queue_push_back(&queue_, i + 4));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(test_queue_at(&queue_, i), i + 8);
  }
  EXPECT_EQ(queue_.element_count, 4);
}

TEST_F(IndexableDequeTest, allocate_twice) {
  for (int i = 0; i < 8; ++i) {
    IREE_CHECK_OK(test_queue_push_back(&queue_, i));
  }
  EXPECT_EQ(queue_.element_count, 8);
  EXPECT_GT(queue_.capacity, 4);

  for (int i = 8; i < 16; ++i) {
    IREE_CHECK_OK(test_queue_push_back(&queue_, i));
  }
  EXPECT_EQ(queue_.element_count, 16);
  EXPECT_GT(queue_.capacity, 8);
}

TEST_F(IndexableDequeTest, no_reallocation_when_capacity_sufficient) {
  size_t initial_capacity = queue_.capacity;
  for (int i = 0; i < initial_capacity; ++i) {
    IREE_CHECK_OK(test_queue_push_back(&queue_, i));
  }
  EXPECT_EQ(queue_.capacity, initial_capacity);
}
