// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/task/executor.h"

#include <cstddef>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Tests that an executor can be created and destroyed repeatedly without
// running out of system resources. Since all systems are different there's no
// guarantee this will fail but it does give ASAN/TSAN some nice stuff to chew
// on.
TEST(ExecutorTest, Lifetime) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(/*group_count=*/4, &topology);

  for (int i = 0; i < 100; ++i) {
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_local_memory_size = 64 * 1024;
    iree_task_executor_t* executor = NULL;
    IREE_ASSERT_OK(iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor));
    // -- idle --
    iree_task_executor_release(executor);
  }

  iree_task_topology_deinitialize(&topology);
}

// Tests lifetime when issuing submissions before exiting.
// This tries to catch races in shutdown with pending work.
TEST(ExecutorTest, LifetimeStress) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(/*group_count=*/4, &topology);

  for (int i = 0; i < 100; ++i) {
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_local_memory_size = 64 * 1024;
    iree_task_executor_t* executor = NULL;
    IREE_ASSERT_OK(iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor));
    iree_task_scope_t scope;
    iree_task_scope_initialize(iree_make_cstring_view("scope"),
                               IREE_TASK_SCOPE_FLAG_NONE, &scope);

    static std::atomic<int> received_value = {0};
    iree_task_call_t call;
    iree_task_call_initialize(
        &scope,
        iree_task_make_call_closure(
            [](void* user_context, iree_task_t* task,
               iree_task_submission_t* pending_submission) {
              received_value = (int)(uintptr_t)user_context;
              return iree_ok_status();
            },
            (void*)(uintptr_t)i),
        &call);

    iree_task_fence_t* fence = NULL;
    IREE_ASSERT_OK(iree_task_executor_acquire_fence(executor, &scope, &fence));
    iree_task_set_completion_task(&call.header, &fence->header);

    iree_task_submission_t submission;
    iree_task_submission_initialize(&submission);
    iree_task_submission_enqueue(&submission, &call.header);
    iree_task_executor_submit(executor, &submission);
    iree_task_executor_flush(executor);
    IREE_ASSERT_OK(
        iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE));

    EXPECT_EQ(received_value, i) << "call did not correlate to loop";

    iree_task_scope_deinitialize(&scope);
    iree_task_executor_release(executor);
  }

  iree_task_topology_deinitialize(&topology);
}

// Tests heavily serialized submission to an executor.
// This puts pressure on the overheads involved in spilling up threads.
TEST(ExecutorTest, SubmissionStress) {
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_local_memory_size = 64 * 1024;
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(/*group_count=*/4, &topology);
  iree_task_executor_t* executor = NULL;
  IREE_ASSERT_OK(iree_task_executor_create(options, &topology,
                                           iree_allocator_system(), &executor));
  iree_task_scope_t scope;
  iree_task_scope_initialize(iree_make_cstring_view("scope"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope);

  for (int i = 0; i < 1000; ++i) {
    static std::atomic<int> received_value = {0};
    iree_task_call_t call;
    iree_task_call_initialize(
        &scope,
        iree_task_make_call_closure(
            [](void* user_context, iree_task_t* task,
               iree_task_submission_t* pending_submission) {
              received_value = (int)(uintptr_t)user_context;
              return iree_ok_status();
            },
            (void*)(uintptr_t)i),
        &call);

    iree_task_fence_t* fence = NULL;
    IREE_ASSERT_OK(iree_task_executor_acquire_fence(executor, &scope, &fence));
    iree_task_set_completion_task(&call.header, &fence->header);

    iree_task_submission_t submission;
    iree_task_submission_initialize(&submission);
    iree_task_submission_enqueue(&submission, &call.header);
    iree_task_executor_submit(executor, &submission);
    iree_task_executor_flush(executor);
    IREE_ASSERT_OK(
        iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE));

    EXPECT_EQ(received_value, i) << "call did not correlate to loop";
  }

  iree_task_scope_deinitialize(&scope);
  iree_task_executor_release(executor);
  iree_task_topology_deinitialize(&topology);
}

}  // namespace
