// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstdio>
#include <memory>

#include "iree/base/api.h"
#include "iree/task/submission.h"
#include "iree/task/task.h"
#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using iree::Status;
using iree::StatusCode;
using iree::testing::status::StatusIs;

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
      if (iree_atomic_load(&storage_[i], iree_memory_order_seq_cst) != 1) {
        return false;
      }
    }
    return true;
  }

  static iree_status_t Tile(void* user_context,
                            const iree_task_tile_context_t* tile_context,
                            iree_task_submission_t* pending_submission) {
    GridCoverage* coverage = reinterpret_cast<GridCoverage*>(user_context);
    uint32_t slot =
        tile_context->workgroup_xyz[2] * (tile_context->workgroup_count[1] *
                                          tile_context->workgroup_count[0]) +
        tile_context->workgroup_xyz[1] * tile_context->workgroup_count[0] +
        tile_context->workgroup_xyz[0];
    iree_atomic_fetch_add(&coverage->storage_[slot], 1,
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
    IREE_TRACE_SCOPE();
    GridCoverage coverage(workgroup_count);
    iree_task_dispatch_t task;
    iree_task_dispatch_initialize(
        &scope_,
        iree_task_make_dispatch_closure(GridCoverage::Tile, (void*)&coverage),
        workgroup_size, workgroup_count, &task);
    task.header.flags |= dispatch_flags;
    IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
    EXPECT_TRUE(coverage.Verify());
  }
};

TEST_F(TaskDispatchTest, Issue000) {
  IREE_TRACE_SCOPE();
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {0, 0, 0};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, IREE_TASK_FLAG_NONE);
}

TEST_F(TaskDispatchTest, Issue120) {
  IREE_TRACE_SCOPE();
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {1, 2, 0};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, IREE_TASK_FLAG_NONE);
}

TEST_F(TaskDispatchTest, Issue111) {
  IREE_TRACE_SCOPE();
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {1, 1, 1};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, IREE_TASK_FLAG_NONE);
}

TEST_F(TaskDispatchTest, Issue345) {
  IREE_TRACE_SCOPE();
  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {3, 4, 5};
  DispatchAndVerifyGrid(kWorkgroupSize, kWorkgroupCount, IREE_TASK_FLAG_NONE);
}

TEST_F(TaskDispatchTest, IssueIndirect) {
  IREE_TRACE_SCOPE();

  static const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  static const uint32_t kWorkgroupCount[3] = {3, 4, 5};
  uint32_t indirect_workgroup_count[3] = {0, 0, 0};
  GridCoverage coverage(kWorkgroupCount);

  iree_task_call_t calculate_task;
  iree_task_call_initialize(
      &scope_,
      iree_task_make_call_closure(
          [](void* user_context, iree_task_t* task,
             iree_task_submission_t* pending_submission) {
            IREE_TRACE_SCOPE();
            uint32_t* indirect_workgroup_count_ptr = (uint32_t*)user_context;
            for (size_t i = 0; i < IREE_ARRAYSIZE(kWorkgroupCount); ++i) {
              indirect_workgroup_count_ptr[i] = kWorkgroupCount[i];
            }
            return iree_ok_status();
          },
          (void*)indirect_workgroup_count),
      &calculate_task);

  iree_task_dispatch_t dispatch_task;
  iree_task_dispatch_initialize_indirect(
      &scope_,
      iree_task_make_dispatch_closure(GridCoverage::Tile, (void*)&coverage),
      kWorkgroupSize, indirect_workgroup_count, &dispatch_task);
  iree_task_set_completion_task(&calculate_task.header, &dispatch_task.header);

  IREE_ASSERT_OK(
      SubmitTasksAndWaitIdle(&calculate_task.header, &dispatch_task.header));
  EXPECT_TRUE(coverage.Verify());
}

TEST_F(TaskDispatchTest, IssueFailure) {
  IREE_TRACE_SCOPE();

  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {64, 1, 1};

  auto tile = [](void* user_context,
                 const iree_task_tile_context_t* tile_context,
                 iree_task_submission_t* pending_submission) -> iree_status_t {
    IREE_TRACE_SCOPE();
    return tile_context->workgroup_xyz[0] == 32
               ? iree_make_status(IREE_STATUS_DATA_LOSS, "whoops!")
               : iree_ok_status();
  };

  iree_task_dispatch_t task;
  iree_task_dispatch_initialize(&scope_,
                                iree_task_make_dispatch_closure(tile, NULL),
                                kWorkgroupSize, kWorkgroupCount, &task);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDataLoss));
}

TEST_F(TaskDispatchTest, IssueFailureChained) {
  IREE_TRACE_SCOPE();

  const uint32_t kWorkgroupSize[3] = {1, 1, 1};
  const uint32_t kWorkgroupCount[3] = {64, 1, 1};

  auto tile = [](void* user_context,
                 const iree_task_tile_context_t* tile_context,
                 iree_task_submission_t* pending_submission) -> iree_status_t {
    return tile_context->workgroup_xyz[0] == 32
               ? iree_make_status(IREE_STATUS_DATA_LOSS, "whoops!")
               : iree_ok_status();
  };

  iree_task_dispatch_t dispatch_task;
  iree_task_dispatch_initialize(
      &scope_, iree_task_make_dispatch_closure(tile, NULL), kWorkgroupSize,
      kWorkgroupCount, &dispatch_task);

  int did_call = 0;
  iree_task_call_t call_task;
  iree_task_call_initialize(&scope_,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE();
                                  int* did_call_ptr = (int*)user_context;
                                  ++(*did_call_ptr);
                                  return iree_ok_status();
                                },
                                &did_call),
                            &call_task);
  iree_task_set_completion_task(&dispatch_task.header, &call_task.header);

  IREE_ASSERT_OK(
      SubmitTasksAndWaitIdle(&dispatch_task.header, &call_task.header));
  EXPECT_EQ(0, did_call);
  EXPECT_THAT(Status(iree_task_scope_consume_status(&scope_)),
              StatusIs(StatusCode::kDataLoss));
}

}  // namespace
