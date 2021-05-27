// Copyright 2021 Google LLC
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
