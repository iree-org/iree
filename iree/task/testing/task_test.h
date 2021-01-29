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

// NOTE: the best kind of synchronization is no synchronization; always try to
// design your algorithm so that you don't need anything from this file :)
// See https://travisdowns.github.io/blog/2020/07/06/concurrency-costs.html

#ifndef IREE_TASK_TESTING_TASK_TEST_H_
#define IREE_TASK_TESTING_TASK_TEST_H_

#include <memory>

#include "iree/task/executor.h"
#include "iree/task/scope.h"
#include "iree/task/task.h"
#include "iree/task/topology.h"
#include "iree/testing/status_matchers.h"

class TaskTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(8, &topology);
    IREE_ASSERT_OK(iree_task_executor_create(IREE_TASK_SCHEDULING_MODE_RESERVED,
                                             &topology, iree_allocator_system(),
                                             &executor_));
    iree_task_topology_deinitialize(&topology);

    iree_task_scope_initialize(iree_make_cstring_view("scope"), &scope_);
  }

  virtual void TearDown() {
    iree_task_scope_deinitialize(&scope_);

    iree_task_executor_release(executor_);
  }

  // Submits a sequence of tasks with |head_task| at the head and |tail_task| at
  // the tail (they can be the same).
  iree_status_t SubmitTasksAndWaitIdle(iree_task_t* head_task,
                                       iree_task_t* tail_task) {
    iree_task_fence_t* fence = NULL;
    IREE_RETURN_IF_ERROR(
        iree_task_executor_acquire_fence(executor_, &scope_, &fence));
    iree_task_set_completion_task(tail_task, &fence->header);

    iree_task_submission_t submission;
    iree_task_submission_initialize(&submission);
    iree_task_submission_enqueue(&submission, head_task);
    iree_task_executor_submit(executor_, &submission);
    iree_task_executor_flush(executor_);
    return iree_task_scope_wait_idle(&scope_, IREE_TIME_INFINITE_FUTURE);
  }

  // Submits a DAG of tasks with |tail_task| at the tail (used just for idle
  // detection).
  iree_status_t SubmitAndWaitIdle(iree_task_submission_t* submission,
                                  iree_task_t* tail_task) {
    iree_task_fence_t* fence = NULL;
    IREE_RETURN_IF_ERROR(
        iree_task_executor_acquire_fence(executor_, &scope_, &fence));
    iree_task_set_completion_task(tail_task, &fence->header);

    iree_task_executor_submit(executor_, submission);
    iree_task_executor_flush(executor_);
    return iree_task_scope_wait_idle(&scope_, IREE_TIME_INFINITE_FUTURE);
  }

  iree_task_executor_t* executor_ = NULL;
  iree_task_scope_t scope_;
};

#endif  // IREE_TASK_TESTING_TASK_TEST_H_
