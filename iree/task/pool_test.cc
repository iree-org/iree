// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/task/pool.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct {
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
