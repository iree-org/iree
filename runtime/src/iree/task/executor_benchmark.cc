// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for the task executor's wake and drain paths.
//
// These measure the end-to-end latency of scheduling processes and having
// workers pick them up, which is the critical path for dispatch latency.
// Key scenarios:
//   - wake_budget == 1 wake: single worker wakes to drain an immediate process.
//   - wake_budget > 1 wake: N workers wake to cooperatively drain a compute
//   process.
//   - Concurrent activation: multiple processes activate simultaneously.

#include <atomic>
#include <thread>

#include "benchmark/benchmark.h"
#include "iree/base/api.h"
#include "iree/task/executor.h"
#include "iree/task/process.h"
#include "iree/task/topology.h"

namespace {

// Skip benchmarks that request more workers than the machine has cores.
// Creating N threads on M << N cores produces meaningless contention noise
// and can timeout on small CI runners.
static bool ShouldSkipWorkerCount(benchmark::State& state, int worker_count) {
  int available = static_cast<int>(std::thread::hardware_concurrency());
  if (available > 0 && worker_count > available) {
    state.SkipWithMessage("worker_count exceeds available cores");
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static iree_task_executor_t* CreateExecutor(iree_host_size_t worker_count) {
  iree_task_topology_t topology;
  iree_task_topology_initialize_from_group_count(worker_count, &topology);
  iree_task_executor_options_t options;
  iree_task_executor_options_initialize(&options);
  iree_task_executor_t* executor = NULL;
  IREE_CHECK_OK(iree_task_executor_create(options, &topology,
                                          iree_allocator_system(), &executor));
  iree_task_topology_deinitialize(&topology);
  return executor;
}

// Drain function for wake_budget == 1 benchmarks: completes on first call.
// Does not signal the main thread — release_fn handles that (the worker
// still accesses process fields after drain() returns).
static iree_status_t instant_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* result) {
  result->did_work = true;
  result->completed = true;
  return iree_ok_status();
}

// Release callback for wake_budget == 1 benchmarks. After this fires, no worker
// accesses the process again — safe to reinitialize for the next iteration.
// user_data points to std::atomic<bool>.
static void budget1_bench_release(iree_task_process_t* process) {
  auto* released = reinterpret_cast<std::atomic<bool>*>(process->user_data);
  released->store(true, std::memory_order_release);
}

// Context for wake_budget > 1 benchmarks with cooperative multi-worker
// draining.
struct ComputeBenchmarkContext {
  std::atomic<int32_t> tiles_remaining;
  // Signaled by completion_fn (eager — may precede full release).
  std::atomic<bool> completed{false};
  // Signaled by release_fn (all drainers have exited — safe to reuse).
  std::atomic<bool> released{false};
};

// Drain function for wake_budget > 1 benchmarks: each worker claims tiles
// atomically.
static iree_status_t compute_bench_drain(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* result) {
  auto* context =
      reinterpret_cast<ComputeBenchmarkContext*>(process->user_data);

  // Claim a tile.
  int32_t remaining =
      context->tiles_remaining.fetch_sub(1, std::memory_order_acq_rel);
  if (remaining <= 0) {
    context->tiles_remaining.fetch_add(1, std::memory_order_relaxed);
    result->did_work = false;
    result->completed = true;
    return iree_ok_status();
  }

  result->did_work = true;
  result->completed =
      (context->tiles_remaining.load(std::memory_order_relaxed) <= 0);
  return iree_ok_status();
}

// Completion callback: fires eagerly when the first worker observes completion.
// For wake_budget > 1, other workers may still be inside drain() — do NOT free
// resources here that drain() accesses. Use release_fn for that.
static void compute_bench_completion(iree_task_process_t* process,
                                     iree_status_t status) {
  auto* context =
      reinterpret_cast<ComputeBenchmarkContext*>(process->user_data);
  context->completed.store(true, std::memory_order_release);
  iree_status_free(status);
}

// Release callback: fires when all active drainers have exited the slot.
// After this, no worker accesses the process — safe to reinitialize.
static void compute_bench_release(iree_task_process_t* process) {
  auto* context =
      reinterpret_cast<ComputeBenchmarkContext*>(process->user_data);
  context->released.store(true, std::memory_order_release);
}

// Spins until condition is true.
template <typename Fn>
static void SpinUntil(Fn&& condition) {
  while (!condition()) {
    iree_thread_yield();
  }
}

//===----------------------------------------------------------------------===//
// wake_budget == 1 wake benchmarks
//===----------------------------------------------------------------------===//

// Measures the round-trip latency of scheduling a wake_budget == 1 process and
// having a single worker wake up, drain it, and complete it. This is the fast
// path for queue management, host callbacks, and retire/signal operations.
void BM_WakeSingleWorker(benchmark::State& state) {
  iree_task_executor_t* executor = CreateExecutor(1);

  // Declared outside the loop: the worker still accesses process fields after
  // drain() returns (schedule_state, release_fn, completion). release_fn
  // signals when the process is safe to reinitialize.
  std::atomic<bool> released{false};
  iree_task_process_t process;

  for (auto _ : state) {
    released.store(false, std::memory_order_relaxed);
    iree_task_process_initialize(instant_drain, /*suspend_count=*/0,
                                 /*wake_budget=*/1, &process);
    process.release_fn = budget1_bench_release;
    process.user_data = &released;
    iree_task_executor_schedule_process(executor, &process);
    SpinUntil([&] { return released.load(std::memory_order_acquire); });
  }

  iree_task_executor_release(executor);
}
BENCHMARK(BM_WakeSingleWorker)->UseRealTime();

//===----------------------------------------------------------------------===//
// wake_budget > 1 wake benchmarks (wake tree)
//===----------------------------------------------------------------------===//

// Measures round-trip latency for a cold-start wake_budget > 1 process:
// schedule, wake N workers, cooperatively drain all tiles, complete, and
// release. Workers start idle (sleeping) at the beginning of each iteration, so
// this captures the full cost including futex wake.
//
// Parameter: number of workers (and budget).
void BM_WakeAllWorkers(benchmark::State& state) {
  const int worker_count = state.range(0);
  if (ShouldSkipWorkerCount(state, worker_count)) return;
  iree_task_executor_t* executor = CreateExecutor(worker_count);

  // Give workers enough tiles that they all get work to do.
  const int tiles_per_worker = 100;

  ComputeBenchmarkContext context;
  iree_task_process_t process;

  for (auto _ : state) {
    context.tiles_remaining.store(worker_count * tiles_per_worker);
    context.completed.store(false);
    context.released.store(false);

    iree_task_process_initialize(compute_bench_drain, /*suspend_count=*/0,
                                 /*wake_budget=*/worker_count, &process);
    process.completion_fn = compute_bench_completion;
    process.release_fn = compute_bench_release;
    process.user_data = &context;

    iree_task_executor_schedule_process(executor, &process);

    // Wait for full release before reusing process memory. For wake_budget > 1,
    // this waits for all active drainers to exit the compute slot.
    SpinUntil([&] { return context.released.load(std::memory_order_acquire); });
  }

  state.SetItemsProcessed(state.iterations() * worker_count);
  iree_task_executor_release(executor);
}
BENCHMARK(BM_WakeAllWorkers)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->UseRealTime();

// Measures wake latency when workers are already active (warm start).
// After the first iteration warms up all workers, subsequent iterations
// measure the overhead of the wake tree when workers loop back quickly.
void BM_WakeWarmWorkers(benchmark::State& state) {
  const int worker_count = state.range(0);
  if (ShouldSkipWorkerCount(state, worker_count)) return;
  iree_task_executor_t* executor = CreateExecutor(worker_count);

  const int tiles_per_worker = 10;

  ComputeBenchmarkContext context;
  iree_task_process_t process;

  for (auto _ : state) {
    context.tiles_remaining.store(worker_count * tiles_per_worker);
    context.completed.store(false);
    context.released.store(false);

    iree_task_process_initialize(compute_bench_drain, /*suspend_count=*/0,
                                 /*wake_budget=*/worker_count, &process);
    process.completion_fn = compute_bench_completion;
    process.release_fn = compute_bench_release;
    process.user_data = &context;

    iree_task_executor_schedule_process(executor, &process);
    SpinUntil([&] { return context.released.load(std::memory_order_acquire); });
  }

  state.SetItemsProcessed(state.iterations() * worker_count);
  iree_task_executor_release(executor);
}
BENCHMARK(BM_WakeWarmWorkers)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->UseRealTime();

//===----------------------------------------------------------------------===//
// Concurrent activation benchmarks
//===----------------------------------------------------------------------===//

// Two compute processes activate simultaneously, each requesting half the
// workers. Measures the wake tree's ability to merge concurrent activations
// via the shared desired_wake counter.
void BM_ConcurrentActivation(benchmark::State& state) {
  const int worker_count = state.range(0);
  if (ShouldSkipWorkerCount(state, worker_count)) return;
  iree_task_executor_t* executor = CreateExecutor(worker_count);

  const int budget_per_process = worker_count / 2;
  if (budget_per_process < 1) {
    state.SkipWithMessage("need at least 2 workers");
    iree_task_executor_release(executor);
    return;
  }
  const int tiles_per_process = 100;

  ComputeBenchmarkContext context_a, context_b;
  iree_task_process_t process_a, process_b;

  for (auto _ : state) {
    context_a.tiles_remaining.store(tiles_per_process);
    context_a.completed.store(false);
    context_a.released.store(false);
    context_b.tiles_remaining.store(tiles_per_process);
    context_b.completed.store(false);
    context_b.released.store(false);

    iree_task_process_initialize(compute_bench_drain, 0, budget_per_process,
                                 &process_a);
    iree_task_process_initialize(compute_bench_drain, 0, budget_per_process,
                                 &process_b);
    process_a.completion_fn = compute_bench_completion;
    process_a.release_fn = compute_bench_release;
    process_a.user_data = &context_a;
    process_b.completion_fn = compute_bench_completion;
    process_b.release_fn = compute_bench_release;
    process_b.user_data = &context_b;

    // Schedule both simultaneously.
    iree_task_executor_schedule_process(executor, &process_a);
    iree_task_executor_schedule_process(executor, &process_b);

    // Wait for both processes to fully release before reusing memory.
    SpinUntil([&] {
      return context_a.released.load(std::memory_order_acquire) &&
             context_b.released.load(std::memory_order_acquire);
    });
  }

  state.SetItemsProcessed(state.iterations() * 2);
  iree_task_executor_release(executor);
}
BENCHMARK(BM_ConcurrentActivation)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->UseRealTime();

//===----------------------------------------------------------------------===//
// wake_budget == 1 throughput benchmark
//===----------------------------------------------------------------------===//

// Measures the throughput of scheduling and completing many wake_budget == 1
// processes sequentially (one at a time). This is the steady-state path for
// queue management operations in the local_task driver.
void BM_SequentialProcessThroughput(benchmark::State& state) {
  iree_task_executor_t* executor = CreateExecutor(1);

  std::atomic<bool> released{false};
  iree_task_process_t process;

  for (auto _ : state) {
    released.store(false, std::memory_order_relaxed);
    iree_task_process_initialize(instant_drain, 0, 1, &process);
    process.release_fn = budget1_bench_release;
    process.user_data = &released;
    iree_task_executor_schedule_process(executor, &process);
    SpinUntil([&] { return released.load(std::memory_order_acquire); });
  }

  state.SetItemsProcessed(state.iterations());
  iree_task_executor_release(executor);
}
BENCHMARK(BM_SequentialProcessThroughput)->UseRealTime();

}  // namespace
