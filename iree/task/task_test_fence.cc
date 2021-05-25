// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TaskFenceTest : public TaskTest {};

TEST_F(TaskFenceTest, IssueChained) {
  iree_task_fence_t task_a;
  iree_task_fence_initialize(&scope_, &task_a);

  iree_task_fence_t task_b;
  iree_task_fence_initialize(&scope_, &task_b);
  iree_task_set_completion_task(&task_a.header, &task_b.header);

  iree_task_fence_t task_c;
  iree_task_fence_initialize(&scope_, &task_c);
  iree_task_set_completion_task(&task_b.header, &task_c.header);

  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task_a.header, &task_c.header));
}

}  // namespace
