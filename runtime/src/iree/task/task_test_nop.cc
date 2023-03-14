// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/task.h"
#include "iree/task/testing/task_test.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class TaskNopTest : public TaskTest {};

TEST_F(TaskNopTest, Issue) {
  IREE_TRACE_SCOPE();
  iree_task_nop_t task;
  iree_task_nop_initialize(&scope_, &task);
  IREE_ASSERT_OK(SubmitTasksAndWaitIdle(&task.header, &task.header));
}

}  // namespace
