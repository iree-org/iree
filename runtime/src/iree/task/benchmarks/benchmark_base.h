// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Base infrastructure for task system benchmarks.
//
// Provides executor lifecycle management and workload helpers for
// microbenchmarking dispatch overhead, wake latency, work-stealing
// effectiveness, and scheduling scalability.
//
// Usage:
//   #include "iree/task/benchmarks/benchmark_base.h"
//
//   void BM_Something(::benchmark::State& state) {
//     TaskBenchmarkContext context(state.range(0));
//     for (auto _ : state) {
//       // submit work, measure
//     }
//   }
//   BENCHMARK(BM_Something)->Range(1, 32);

#ifndef IREE_TASK_BENCHMARKS_BENCHMARK_BASE_H_
#define IREE_TASK_BENCHMARKS_BENCHMARK_BASE_H_

#include <atomic>
#include <chrono>
#include <cstring>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/task/executor.h"
#include "iree/task/scope.h"
#include "iree/task/task.h"
#include "iree/task/topology.h"

namespace iree::task::benchmarks {

//===----------------------------------------------------------------------===//
// Task benchmark context
//===----------------------------------------------------------------------===//

// Manages executor lifetime for benchmarks.
// Creates an executor with configurable worker count and provides helpers
// for submitting work and waiting for completion.
struct TaskBenchmarkContext {
  iree_task_executor_t* executor = nullptr;
  iree_task_scope_t scope;
  iree_host_size_t worker_count = 0;

  // Creates an executor with |num_workers| workers.
  // Returns false and calls state.SkipWithError() on failure.
  bool Initialize(iree_host_size_t num_workers, ::benchmark::State& state) {
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_local_memory_size = 64 * 1024;

    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(num_workers, &topology);

    iree_status_t status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor);
    iree_task_topology_deinitialize(&topology);

    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Executor creation failed");
      iree_status_ignore(status);
      return false;
    }

    iree_task_scope_initialize(iree_make_cstring_view("bench"),
                               IREE_TASK_SCOPE_FLAG_NONE, &scope);
    worker_count = num_workers;
    return true;
  }

  // Creates an executor with |num_workers| and optional spin duration.
  bool Initialize(iree_host_size_t num_workers, iree_duration_t spin_ns,
                  ::benchmark::State& state) {
    iree_task_executor_options_t options;
    iree_task_executor_options_initialize(&options);
    options.worker_local_memory_size = 64 * 1024;
    options.worker_spin_ns = spin_ns;

    iree_task_topology_t topology;
    iree_task_topology_initialize_from_group_count(num_workers, &topology);

    iree_status_t status = iree_task_executor_create(
        options, &topology, iree_allocator_system(), &executor);
    iree_task_topology_deinitialize(&topology);

    if (!iree_status_is_ok(status)) {
      state.SkipWithError("Executor creation failed");
      iree_status_ignore(status);
      return false;
    }

    iree_task_scope_initialize(iree_make_cstring_view("bench"),
                               IREE_TASK_SCOPE_FLAG_NONE, &scope);
    worker_count = num_workers;
    return true;
  }

  ~TaskBenchmarkContext() {
    if (executor) {
      // Wait for any in-flight tasks to complete before shutdown. This
      // handles cases where a benchmark loop breaks early (e.g., on timeout)
      // while tasks referencing stack-allocated memory are still in-flight.
      iree_status_ignore(
          iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE));
      // Release the executor first (shuts down all workers) to ensure no
      // worker threads are still touching the scope during cleanup.
      iree_task_executor_release(executor);
      iree_task_scope_deinitialize(&scope);
    }
  }

  // Submit a task DAG and wait for completion.
  // |tail_task| is the last task in the DAG (a fence is appended after it).
  // Returns false if the submission or wait fails.
  bool SubmitAndWait(iree_task_submission_t* submission,
                     iree_task_t* tail_task) {
    iree_task_fence_t* fence = nullptr;
    iree_status_t status =
        iree_task_executor_acquire_fence(executor, &scope, &fence);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    }
    iree_task_set_completion_task(tail_task, &fence->header);

    iree_task_executor_submit(executor, submission);
    iree_task_executor_flush(executor);
    status = iree_task_scope_wait_idle(&scope, IREE_TIME_INFINITE_FUTURE);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return false;
    }
    return true;
  }

  // Submit a single head→tail chain and wait.
  bool SubmitChainAndWait(iree_task_t* head_task, iree_task_t* tail_task) {
    iree_task_submission_t submission;
    iree_task_submission_initialize(&submission);
    iree_task_submission_enqueue(&submission, head_task);
    return SubmitAndWait(&submission, tail_task);
  }

  // Non-copyable.
  TaskBenchmarkContext(const TaskBenchmarkContext&) = delete;
  TaskBenchmarkContext& operator=(const TaskBenchmarkContext&) = delete;
  TaskBenchmarkContext() = default;
};

//===----------------------------------------------------------------------===//
// Workload helpers
//===----------------------------------------------------------------------===//

// Dispatch closure that does zero work. Measures pure scheduling overhead.
static iree_status_t dispatch_closure_noop(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  (void)user_context;
  (void)tile_context;
  (void)pending_submission;
  return iree_ok_status();
}

// Dispatch closure that touches memory to prevent the compiler from
// eliminating the dispatch entirely. Does minimal work (~10ns per tile).
static thread_local volatile uint64_t benchmark_sink = 0;
static iree_status_t dispatch_closure_trivial(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  (void)user_context;
  (void)pending_submission;
  benchmark_sink += tile_context->workgroup_xyz[0];
  return iree_ok_status();
}

// Dispatch closure that does configurable work. The |user_context| is
// interpreted as the number of iterations of a simple arithmetic loop.
static iree_status_t dispatch_closure_work(
    void* user_context, const iree_task_tile_context_t* tile_context,
    iree_task_submission_t* pending_submission) {
  (void)pending_submission;
  uintptr_t iterations = (uintptr_t)user_context;
  volatile uint64_t accumulator = tile_context->workgroup_xyz[0];
  for (uintptr_t i = 0; i < iterations; ++i) {
    accumulator = accumulator * 6364136223846793005ULL + 1442695040888963407ULL;
  }
  benchmark_sink += accumulator;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dispatch chain builder
//===----------------------------------------------------------------------===//

// Maximum number of dispatches in a chain for stack allocation.
static constexpr int kMaxChainLength = 256;

// Builds a linear chain of dispatches with direct completion edges:
//   dispatch[0] → dispatch[1] → ... → dispatch[N-1]
//
// Each dispatch's completion_task points directly to the next dispatch.
// This is the minimal-overhead chain structure, measuring pure dispatch
// scheduling cost without barrier task overhead.
//
// |dispatch_count|: Number of dispatches in the chain.
// |tile_count|: Number of tiles per dispatch (workgroup_count[0]).
// |closure|: The dispatch closure to execute per tile.
// |dispatches|: Caller-allocated array of dispatch_count dispatches.
// |out_head|: Set to the first task in the chain.
// |out_tail|: Set to the last task in the chain.
static void BuildDispatchChain(iree_task_scope_t* scope, int dispatch_count,
                               uint32_t tile_count,
                               iree_task_dispatch_closure_t closure,
                               iree_task_dispatch_t* dispatches,
                               iree_task_t** out_head, iree_task_t** out_tail) {
  const uint32_t workgroup_size[3] = {1, 1, 1};
  const uint32_t workgroup_count[3] = {tile_count, 1, 1};

  for (int i = 0; i < dispatch_count; ++i) {
    iree_task_dispatch_initialize(scope, closure, workgroup_size,
                                  workgroup_count, &dispatches[i]);
  }

  // Chain dispatches directly: dispatch[i].completion → dispatch[i+1].
  for (int i = 0; i < dispatch_count - 1; ++i) {
    iree_task_set_completion_task(&dispatches[i].header,
                                  &dispatches[i + 1].header);
  }

  *out_head = &dispatches[0].header;
  *out_tail = &dispatches[dispatch_count - 1].header;
}

// Builds a linear chain of dispatches with barrier tasks between them:
//   dispatch[0] → barrier[0] → dispatch[1] → barrier[1] → ... → dispatch[N-1]
//
// This models the realistic HAL command buffer pattern where barriers
// separate dispatches. Measures dispatch + barrier scheduling overhead.
//
// |dispatch_count|: Number of dispatches in the chain.
// |tile_count|: Number of tiles per dispatch (workgroup_count[0]).
// |closure|: The dispatch closure to execute per tile.
// |dispatches|: Caller-allocated array of dispatch_count dispatches.
// |barriers|: Caller-allocated array of (dispatch_count - 1) barriers.
// |out_head|: Set to the first task in the chain.
// |out_tail|: Set to the last task in the chain.
static void BuildDispatchChainWithBarriers(
    iree_task_scope_t* scope, int dispatch_count, uint32_t tile_count,
    iree_task_dispatch_closure_t closure, iree_task_dispatch_t* dispatches,
    iree_task_barrier_t* barriers, iree_task_t** out_head,
    iree_task_t** out_tail) {
  const uint32_t workgroup_size[3] = {1, 1, 1};
  const uint32_t workgroup_count[3] = {tile_count, 1, 1};

  for (int i = 0; i < dispatch_count; ++i) {
    iree_task_dispatch_initialize(scope, closure, workgroup_size,
                                  workgroup_count, &dispatches[i]);
  }

  // Chain dispatches with barriers: dispatch[i] → barrier[i] → dispatch[i+1].
  for (int i = 0; i < dispatch_count - 1; ++i) {
    iree_task_barrier_initialize_empty(scope, &barriers[i]);
    iree_task_set_completion_task(&dispatches[i].header, &barriers[i].header);
    iree_task_set_completion_task(&barriers[i].header,
                                  &dispatches[i + 1].header);
  }

  *out_head = &dispatches[0].header;
  *out_tail = &dispatches[dispatch_count - 1].header;
}

}  // namespace iree::task::benchmarks

#endif  // IREE_TASK_BENCHMARKS_BENCHMARK_BASE_H_
