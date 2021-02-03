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

#include "iree/task/executor.h"

#include <thread>

#include "iree/base/internal/prng.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static thread_local volatile uint64_t xxx = 0;

static void simulate_work(const iree_task_tile_context_t* tile_context) {
  iree_prng_splitmix64_state_t state;
  iree_prng_splitmix64_initialize(xxx, &state);
  bool slow = false;  // tile_context->workgroup_xyz[0] % 3 == 1;
  if (tile_context->workgroup_xyz[0] == 128 ||
      tile_context->workgroup_xyz[0] == 1023) {
    // Introduce big variance to highlight work stealing.
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  for (int i = 0; i < 256 * 1024; ++i) {
    uint64_t value = iree_prng_splitmix64_next(&state);
    xxx += value;
    if (slow) {
      for (int j = 0; j < 4; ++j) {
        value = iree_prng_splitmix64_next(&state);
        xxx += value;
      }
    }
  }
}

TEST(ExecutorTest, Any) {
  IREE_TRACE_SCOPE0("ExecutorTest::Any");

  iree_allocator_t allocator = iree_allocator_system();

  iree_task_topology_t topology;
#if 1
  iree_task_topology_initialize_from_physical_cores(
      /*max_core_count=*/6, &topology);
#elif 0
  iree_task_topology_initialize_from_unique_l2_cache_groups(
      /*max_group_count=*/6, &topology);
#else
  iree_task_topology_initialize_from_group_count(/*group_count=*/6, &topology);
#endif

  iree_task_executor_t* executor = NULL;
  iree_task_scheduling_mode_t scheduling_mode =
      IREE_TASK_SCHEDULING_MODE_RESERVED;
  IREE_CHECK_OK(iree_task_executor_create(scheduling_mode, &topology, allocator,
                                          &executor));
  iree_task_topology_deinitialize(&topology);

  //
  iree_task_scope_t scope_a;
  iree_task_scope_initialize(iree_make_cstring_view("a"), &scope_a);

  //
  iree_task_call_t call0;
  iree_task_call_initialize(&scope_a,
                            iree_task_make_call_closure(
                                [](uintptr_t user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE0("call0");
                                  EXPECT_EQ(0, user_context);
                                  return iree_ok_status();
                                },
                                0),
                            &call0);

  const uint32_t workgroup_size_0[3] = {256, 1, 1};
  const uint32_t workgroup_count_0[3] = {32, 4, 2};
  iree_task_dispatch_t dispatch0;
  iree_task_dispatch_initialize(
      &scope_a,
      iree_task_make_dispatch_closure(
          [](uintptr_t user_context,
             const iree_task_tile_context_t* tile_context,
             iree_task_submission_t* pending_submission) {
            IREE_TRACE_SCOPE0("tile0");
            EXPECT_EQ(0, user_context);
            simulate_work(tile_context);
            iree_atomic_fetch_add_int32(&tile_context->statistics->reserved, 1,
                                        iree_memory_order_relaxed);
            return iree_ok_status();
          },
          0),
      workgroup_size_0, workgroup_count_0, &dispatch0);
  // dispatch0.header.flags |= IREE_TASK_FLAG_DISPATCH_SLICED;

  const uint32_t workgroup_size_1[3] = {128, 1, 1};
  const uint32_t workgroup_count_1[3] = {16, 2, 1};
  iree_task_dispatch_t dispatch1;
  iree_task_dispatch_initialize(
      &scope_a,
      iree_task_make_dispatch_closure(
          [](uintptr_t user_context,
             const iree_task_tile_context_t* tile_context,
             iree_task_submission_t* pending_submission) {
            IREE_TRACE_SCOPE0("tile1");
            EXPECT_EQ(0, user_context);
            simulate_work(tile_context);
            iree_atomic_fetch_add_int32(&tile_context->statistics->reserved, 1,
                                        iree_memory_order_relaxed);
            return iree_ok_status();
          },
          0),
      workgroup_size_1, workgroup_count_1, &dispatch1);
  dispatch1.header.flags |= IREE_TASK_FLAG_DISPATCH_SLICED;

  //
  iree_task_call_t call1;
  iree_task_call_initialize(&scope_a,
                            iree_task_make_call_closure(
                                [](uintptr_t user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE0("call1");
                                  EXPECT_EQ(1, user_context);
                                  return iree_ok_status();
                                },
                                1),
                            &call1);

#if 1
  // no barrier between dispatches; fanout
  iree_task_t* barrier0_tasks[2] = {&dispatch0.header, &dispatch1.header};
  iree_task_barrier_t barrier0;
  iree_task_barrier_initialize(&scope_a, IREE_ARRAYSIZE(barrier0_tasks),
                               barrier0_tasks, &barrier0);
  iree_task_set_completion_task(&call0.header, &barrier0.header);
  iree_task_set_completion_task(&dispatch0.header, &call1.header);
  iree_task_set_completion_task(&dispatch1.header, &call1.header);
#else
  // barrier between dispatches
  iree_task_set_completion_task(&call0.header, &dispatch0.header);
  iree_task_set_completion_task(&dispatch0.header, &dispatch1.header);
  iree_task_set_completion_task(&dispatch1.header, &call1.header);
#endif

  // fence
  iree_task_fence_t* fence0 = NULL;
  IREE_CHECK_OK(iree_task_executor_acquire_fence(executor, &scope_a, &fence0));
  iree_task_set_completion_task(&call1.header, &fence0->header);

  //
  iree_task_submission_t sub0;
  iree_task_submission_initialize(&sub0);
  iree_task_submission_enqueue(&sub0, &call0.header);
  iree_task_executor_submit(executor, &sub0);

  //
  // iree_task_submission_t sub1;
  // iree_task_submission_initialize(&sub1);
  // IREE_CHECK_OK(iree_task_executor_submit(executor, &sub1));

  iree_task_executor_flush(executor);

  IREE_CHECK_OK(iree_task_scope_wait_idle(&scope_a, IREE_TIME_INFINITE_FUTURE));

  iree_task_scope_deinitialize(&scope_a);
  iree_task_executor_release(executor);
}

}  // namespace
