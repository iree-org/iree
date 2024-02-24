// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>

#include "iree/base/internal/prng.h"
#include "iree/task/executor.h"

// TODO(benvanik): clean this up into a reasonable demo; it's currently staging
// area for testing executor behavior across different platforms and topologies.

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

extern "C" int main(int argc, char* argv[]) {
  IREE_TRACE_APP_ENTER();
  IREE_TRACE_SCOPE_NAMED("ExecutorTest::Any");

  iree_allocator_t allocator = iree_allocator_system();

  iree_task_topology_t topology;
#if 1
  IREE_CHECK_OK(iree_task_topology_initialize_from_physical_cores(
      IREE_TASK_TOPOLOGY_NODE_ID_ANY, IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY,
      /*max_core_count=*/6, &topology));
#else
  iree_task_topology_initialize_from_group_count(/*group_count=*/6, &topology);
#endif

  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  options.worker_local_memory_size = 0;  // 64 * 1024;
  iree_task_executor_t* executor = NULL;
  IREE_CHECK_OK(
      iree_task_executor_create(options, &topology, allocator, &executor));
  iree_task_topology_deinitialize(&topology);

  //
  iree_task_scope_t scope_a;
  iree_task_scope_initialize(iree_make_cstring_view("a"),
                             IREE_TASK_SCOPE_FLAG_NONE, &scope_a);

  //
  iree_task_call_t call0;
  iree_task_call_initialize(&scope_a,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE_NAMED("call0");
                                  IREE_ASSERT_EQ(0, user_context);
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
          [](void* user_context, const iree_task_tile_context_t* tile_context,
             iree_task_submission_t* pending_submission) {
            IREE_TRACE_SCOPE_NAMED("tile0");
            IREE_ASSERT_EQ(0, user_context);
            simulate_work(tile_context);
            iree_atomic_fetch_add_int32(&tile_context->statistics->reserved, 1,
                                        iree_memory_order_relaxed);
            return iree_ok_status();
          },
          0),
      workgroup_size_0, workgroup_count_0, &dispatch0);

  const uint32_t workgroup_size_1[3] = {128, 1, 1};
  const uint32_t workgroup_count_1[3] = {16, 2, 1};
  iree_task_dispatch_t dispatch1;
  iree_task_dispatch_initialize(
      &scope_a,
      iree_task_make_dispatch_closure(
          [](void* user_context, const iree_task_tile_context_t* tile_context,
             iree_task_submission_t* pending_submission) {
            IREE_TRACE_SCOPE_NAMED("tile1");
            IREE_ASSERT_EQ(0, user_context);
            simulate_work(tile_context);
            iree_atomic_fetch_add_int32(&tile_context->statistics->reserved, 1,
                                        iree_memory_order_relaxed);
            return iree_ok_status();
          },
          0),
      workgroup_size_1, workgroup_count_1, &dispatch1);

  //
  iree_task_call_t call1;
  iree_task_call_initialize(&scope_a,
                            iree_task_make_call_closure(
                                [](void* user_context, iree_task_t* task,
                                   iree_task_submission_t* pending_submission) {
                                  IREE_TRACE_SCOPE_NAMED("call1");
                                  IREE_ASSERT_EQ((void*)1, user_context);
                                  return iree_ok_status();
                                },
                                (void*)1),
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
  IREE_TRACE_APP_EXIT(0);
  return 0;
}

}  // namespace
