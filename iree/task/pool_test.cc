// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/pool.h"

#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct iree_test_task_t {
  iree_task_t base;
  uint8_t payload[32];
} iree_test_task_t;

TEST(PoolTest, Lifetime) {
  iree_task_pool_t pool;
  IREE_ASSERT_OK(iree_task_pool_initialize(
      iree_allocator_system(), sizeof(iree_test_task_t), 32, &pool));
  iree_task_pool_deinitialize(&pool);
}

TEST(PoolTest, AcquireRelease) {
  // Start with 2 preallocated tasks so we can test both acquiring existing and
  // growing to allocate new tasks.
  iree_task_pool_t pool;
  IREE_ASSERT_OK(iree_task_pool_initialize(iree_allocator_system(),
                                           sizeof(iree_test_task_t), 2, &pool));

  // Acquire 4 tasks (so we test both the initial size and allocated tasks).
  iree_test_task_t* tasks[4] = {NULL, NULL, NULL, NULL};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    IREE_ASSERT_OK(iree_task_pool_acquire(&pool, (iree_task_t**)&tasks[i]));
    EXPECT_TRUE(tasks[i] != NULL);
  }

  // Release all tasks back to the pool.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    iree_task_pool_release(&pool, (iree_task_t*)tasks[i]);
  }

  // Acquire all tasks again to make sure we put them back in correctly.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    IREE_ASSERT_OK(iree_task_pool_acquire(&pool, (iree_task_t**)&tasks[i]));
    EXPECT_TRUE(tasks[i] != NULL);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    iree_task_pool_release(&pool, (iree_task_t*)tasks[i]);
  }

  iree_task_pool_deinitialize(&pool);
}

TEST(PoolTest, Trim) {
  // Start with 2 preallocated tasks so we can test both acquiring existing and
  // growing to allocate new tasks.
  iree_task_pool_t pool;
  IREE_ASSERT_OK(iree_task_pool_initialize(iree_allocator_system(),
                                           sizeof(iree_test_task_t), 2, &pool));

  // Acquire and release some tasks.
  iree_test_task_t* tasks[8] = {NULL, NULL, NULL, NULL};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    IREE_ASSERT_OK(iree_task_pool_acquire(&pool, (iree_task_t**)&tasks[i]));
    EXPECT_TRUE(tasks[i] != NULL);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    iree_task_pool_release(&pool, (iree_task_t*)tasks[i]);
  }

  // Trim to shrink the pool memory.
  // NOTE: trimming is only supported when there are no outstanding tasks.
  iree_task_pool_trim(&pool);

  // Acquire again to make sure we can reallocate the pool.
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    IREE_ASSERT_OK(iree_task_pool_acquire(&pool, (iree_task_t**)&tasks[i]));
    EXPECT_TRUE(tasks[i] != NULL);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(tasks); ++i) {
    iree_task_pool_release(&pool, (iree_task_t*)tasks[i]);
  }

  iree_task_pool_deinitialize(&pool);
}

}  // namespace
