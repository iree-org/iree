// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <vector>

#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class GridCoverage {
 public:
  explicit GridCoverage(const uint32_t workgroup_count[3])
      : workgroup_count_(workgroup_count[0] * workgroup_count[1] *
                         workgroup_count[2]),
        storage_(new iree_atomic_int32_t[workgroup_count_]) {
    for (iree_host_size_t i = 0; i < workgroup_count_; ++i) {
      storage_[i] = IREE_ATOMIC_VAR_INIT(0);
    }
  }

  bool Verify() {
    fflush(stdout);
    for (iree_host_size_t i = 0; i < workgroup_count_; ++i) {
      if (iree_atomic_load_int32(&storage_[i], iree_memory_order_seq_cst) !=
          1) {
        return false;
      }
    }
    return true;
  }

  static iree_status_t Tile(uintptr_t user_context,
                            const iree_task_tile_context_t* tile_context,
                            iree_task_submission_t* pending_submission) {
    GridCoverage* coverage = reinterpret_cast<GridCoverage*>(user_context);
    uint32_t slot =
        tile_context->workgroup_xyz[2] * (tile_context->workgroup_count[1] *
                                          tile_context->workgroup_count[0]) +
        tile_context->workgroup_xyz[1] * tile_context->workgroup_count[0] +
        tile_context->workgroup_xyz[0];
    iree_atomic_fetch_add_int32(&coverage->storage_[slot], 1,
                                iree_memory_order_seq_cst);

    // Useful when testing large grids:
    // printf("%u, %u, %u\n", tile_context->workgroup_xyz[0],
    //        tile_context->workgroup_xyz[1], tile_context->workgroup_xyz[2]);

    return iree_ok_status();
  }

 private:
  size_t workgroup_count_;
  std::unique_ptr<iree_atomic_int32_t[]> storage_;
};

class TaskDispatchTest : public TaskTest {
 public:
  void DispatchAndVerifyGrid(const uint32_t workgroup_size[3],
                             const uint32_t workgroup_count[3],
                             uint32_t dispatch_flags) {
    GridCoverage coverage(workgroup_count);
    iree_task_dispatch_t task;
    iree_task_dispatch_initialize(&scope_,
                                  iree_task_make_dispatch_closure(
                                      GridCoverage::Tile, (uintptr_t)&coverage),
                                  workgroup_size, workgroup_count, &task);
    task.header.flags |= dispatch_flags;
    IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
    EXPECT_TRUE(coverage.Verify());
  }
};

TEST_F(TaskDispatchTest, Issue000Sharded) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {0, 0, 0};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, 0);
}

TEST_F(TaskDispatchTest, Issue000Sliced) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {0, 0, 0};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount,
                        IREE_TASK_FLAG_DISPATCH_SLICED);
}

TEST_F(TaskDispatchTest, Issue120Sharded) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {1, 2, 0};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, 0);
}

TEST_F(TaskDispatchTest, Issue120Sliced) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {1, 2, 0};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount,
                        IREE_TASK_FLAG_DISPATCH_SLICED);
}

TEST_F(TaskDispatchTest, Issue111Sharded) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {1, 1, 1};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, 0);
}

TEST_F(TaskDispatchTest, Issue111Sliced) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {1, 1, 1};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount,
                        IREE_TASK_FLAG_DISPATCH_SLICED);
}

TEST_F(TaskDispatchTest, Issue345Sharded) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {3, 4, 5};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, 0);
}

TEST_F(TaskDispatchTest, Issue345Sliced) {
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {3, 4, 5};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount,
                        IREE_TASK_FLAG_DISPATCH_SLICED);
}

TEST_F(TaskDispatchTest, IssueIndirect) {
  static const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  static const uint32_t kWorkgroupCount[3] = {3, 4, 5};
  uint32_t indirect_workgroup_count[3] = {0, 0, 0};
  GridCoverage coverage(kWorkgroupCount);

  iree_task_call_t calculate_task;
  iree_task_call_initialize(
      &scope_,
      iree_task_make_call_closure(
          [](uintptr_t user_context, iree_task_t* task,
             iree_task_submission_t* pending_submission) {
            uint32_t* indirect_workgroup_count_ptr = (uint32_t*)user_context;
            for (size_t i = 0; i < IREE_ARRAYSIZE(kWorkgroupCount); ++i) {
              indirect_workgroup_count_ptr[i] = kWorkgroupCount[i];
            }
            return iree_ok_status();
          },
          (uintptr_t)indirect_workgroup_count),
      &calculate_task);

  iree_task_dispatch_t dispatch_task;
  iree_task_dispatch_initialize_indirect(
      &scope_,
      iree_task_make_dispatch_closure(GridCoverage::Tile, (uintptr_t)&coverage),
      kWorkgroupSize, indirect_workgroup_count, &dispatch_task);
  iree_task_set_completion_task(&calculate_task.header, &dispatch_task.header);

  IREE_ASSERT_OK(
      SubmitTasksAndWaitIdle(&calculate_task.header, &dispatch_task.header));
  EXPECT_TRUE(coverage.Verify());
}

}  // namespace
