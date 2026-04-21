// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for process scheduling through the executor. These exercise the full
// stack: process → immediate list → worker drain → completion → dependent
// activation. Unlike process_test.cc (which tests the process type in
// isolation), these tests verify that the executor correctly picks up, drains,
// and completes processes via real worker threads.

#include "iree/task/executor.h"

#include <atomic>
#include <chrono>
#include <thread>

#include "iree/base/api.h"
#include "iree/task/process.h"
#include "iree/task/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

//===----------------------------------------------------------------------===//
// Test infrastructure
//===----------------------------------------------------------------------===//

// Creates a simple executor with the given number of workers.
// Caller must release with iree_task_executor_release.
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

// Context for a test process that completes after a fixed number of drain
// calls. Thread-safe: drain_count is atomic.
struct CountingProcessContext {
  std::atomic<int32_t> drain_count{0};
  int32_t drains_until_complete;
  std::atomic<bool> completed{false};
  iree_status_code_t completion_status_code = IREE_STATUS_OK;
};

static iree_status_t counting_drain(iree_task_process_t* process,
                                    uint32_t worker_index,
                                    iree_task_process_drain_result_t* result) {
  auto* context = reinterpret_cast<CountingProcessContext*>(process->user_data);
  int32_t count = context->drain_count.fetch_add(1, std::memory_order_relaxed);
  result->did_work = true;
  result->completed = (count + 1 >= context->drains_until_complete);
  return iree_ok_status();
}

static void counting_completion(iree_task_process_t* process,
                                iree_status_t status) {
  auto* context = reinterpret_cast<CountingProcessContext*>(process->user_data);
  context->completion_status_code = iree_status_code(status);
  context->completed.store(true, std::memory_order_release);
  iree_status_ignore(status);
}

// Drain function that always returns an error on the first call.
// Uses CountingProcessContext: completes immediately with DATA_LOSS.
static iree_status_t failing_drain(iree_task_process_t* process,
                                   uint32_t worker_index,
                                   iree_task_process_drain_result_t* result) {
  result->did_work = true;
  result->completed = true;
  return iree_make_status(IREE_STATUS_DATA_LOSS, "oops");
}

// Context for a process that sleeps until woken, then completes on the next
// drain. Used to test the sleeping/re-wake protocol.
struct SleepingProcessContext {
  std::atomic<bool> ready{false};
  std::atomic<bool> completed{false};
  std::atomic<int32_t> drain_count{0};
  // Set by the drain function when it returns did_work=false, indicating the
  // worker is about to enter the sleep protocol. Tests can spin on this
  // instead of using a fixed sleep_for.
  std::atomic<bool> entered_sleep{false};
};

static iree_status_t sleeping_drain(iree_task_process_t* process,
                                    uint32_t worker_index,
                                    iree_task_process_drain_result_t* result) {
  auto* context = reinterpret_cast<SleepingProcessContext*>(process->user_data);
  context->drain_count.fetch_add(1, std::memory_order_relaxed);
  if (context->ready.load(std::memory_order_acquire)) {
    result->did_work = true;
    result->completed = true;
  } else {
    result->did_work = false;
    result->completed = false;
    context->entered_sleep.store(true, std::memory_order_release);
  }
  return iree_ok_status();
}

static void sleeping_completion(iree_task_process_t* process,
                                iree_status_t status) {
  auto* context = reinterpret_cast<SleepingProcessContext*>(process->user_data);
  context->completed.store(true, std::memory_order_release);
  iree_status_ignore(status);
}

// Spins until a condition is true, with a timeout to prevent hangs.
template <typename Fn>
static bool SpinUntil(Fn&& condition, std::chrono::milliseconds timeout_ms =
                                          std::chrono::milliseconds(5000)) {
  auto deadline = std::chrono::steady_clock::now() + timeout_ms;
  while (!condition()) {
    if (std::chrono::steady_clock::now() > deadline) return false;
    std::this_thread::yield();
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Basic scheduling tests
//===----------------------------------------------------------------------===//

TEST(ExecutorProcessTest, SingleProcessCompletesImmediately) {
  iree_task_executor_t* executor = CreateExecutor(1);

  CountingProcessContext context;
  context.drains_until_complete = 1;

  iree_task_process_t process;
  iree_task_process_initialize(counting_drain, /*suspend_count=*/0,
                               /*wake_budget=*/1, &process);
  process.completion_fn = counting_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "process did not complete within timeout";
  EXPECT_EQ(context.drain_count.load(), 1);
  EXPECT_EQ(context.completion_status_code, IREE_STATUS_OK);

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, MultiDrainProcess) {
  iree_task_executor_t* executor = CreateExecutor(1);

  CountingProcessContext context;
  context.drains_until_complete = 5;

  iree_task_process_t process;
  iree_task_process_initialize(counting_drain, 0, 1, &process);
  process.completion_fn = counting_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "process did not complete within timeout";
  EXPECT_EQ(context.drain_count.load(), 5);

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, DependencyChain) {
  iree_task_executor_t* executor = CreateExecutor(2);

  // A → B → C. Each completes after 1 drain.
  CountingProcessContext context_a, context_b, context_c;
  context_a.drains_until_complete = 1;
  context_b.drains_until_complete = 1;
  context_c.drains_until_complete = 1;

  iree_task_process_t a, b, c;
  iree_task_process_initialize(counting_drain, /*suspend_count=*/0, 1, &a);
  iree_task_process_initialize(counting_drain, /*suspend_count=*/1, 1, &b);
  iree_task_process_initialize(counting_drain, /*suspend_count=*/1, 1, &c);
  a.completion_fn = counting_completion;
  a.user_data = &context_a;
  b.completion_fn = counting_completion;
  b.user_data = &context_b;
  c.completion_fn = counting_completion;
  c.user_data = &context_c;

  // Wire dependencies.
  iree_task_process_t* a_deps[] = {&b};
  a.dependents = a_deps;
  a.dependent_count = 1;
  iree_task_process_t* b_deps[] = {&c};
  b.dependents = b_deps;
  b.dependent_count = 1;

  // Only schedule A — B and C are suspended, waiting for A to complete.
  iree_task_executor_schedule_process(executor, &a);

  ASSERT_TRUE(SpinUntil([&] { return context_c.completed.load(); }))
      << "chain did not complete within timeout";
  EXPECT_TRUE(context_a.completed.load());
  EXPECT_TRUE(context_b.completed.load());
  EXPECT_TRUE(context_c.completed.load());

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, DiamondDependency) {
  iree_task_executor_t* executor = CreateExecutor(2);

  //     A
  //    / \       .
  //   B   C
  //    \ /       .
  //     D
  CountingProcessContext context_a, context_b, context_c, context_d;
  context_a.drains_until_complete = 1;
  context_b.drains_until_complete = 1;
  context_c.drains_until_complete = 1;
  context_d.drains_until_complete = 1;

  iree_task_process_t a, b, c, d;
  iree_task_process_initialize(counting_drain, 0, 1, &a);
  iree_task_process_initialize(counting_drain, 1, 1, &b);
  iree_task_process_initialize(counting_drain, 1, 1, &c);
  iree_task_process_initialize(counting_drain, 2, 1, &d);
  a.completion_fn = counting_completion;
  a.user_data = &context_a;
  b.completion_fn = counting_completion;
  b.user_data = &context_b;
  c.completion_fn = counting_completion;
  c.user_data = &context_c;
  d.completion_fn = counting_completion;
  d.user_data = &context_d;

  iree_task_process_t* a_deps[] = {&b, &c};
  a.dependents = a_deps;
  a.dependent_count = 2;
  iree_task_process_t* b_deps[] = {&d};
  b.dependents = b_deps;
  b.dependent_count = 1;
  iree_task_process_t* c_deps[] = {&d};
  c.dependents = c_deps;
  c.dependent_count = 1;

  iree_task_executor_schedule_process(executor, &a);

  ASSERT_TRUE(SpinUntil([&] { return context_d.completed.load(); }))
      << "diamond did not complete within timeout";
  EXPECT_TRUE(context_a.completed.load());
  EXPECT_TRUE(context_b.completed.load());
  EXPECT_TRUE(context_c.completed.load());

  iree_task_executor_release(executor);
}

//===----------------------------------------------------------------------===//
// Sleeping and re-wake tests
//===----------------------------------------------------------------------===//

TEST(ExecutorProcessTest, SleepAndRewake) {
  iree_task_executor_t* executor = CreateExecutor(1);

  SleepingProcessContext context;

  iree_task_process_t process;
  iree_task_process_initialize(sleeping_drain, 0, 1, &process);
  process.completion_fn = sleeping_completion;
  process.user_data = &context;

  // Schedule the process. It will drain once, return did_work=false, and the
  // worker will enter the sleep protocol.
  iree_task_executor_schedule_process(executor, &process);

  // Wait for the drain function to signal it returned did_work=false. At this
  // point the worker is in the sleep protocol (checking needs_drain,
  // transitioning to IDLE, etc.). This is a deterministic signal — no
  // sleep_for needed.
  ASSERT_TRUE(SpinUntil([&] { return context.entered_sleep.load(); }))
      << "process never entered sleep";

  // The process should NOT be completed — it's sleeping.
  EXPECT_FALSE(context.completed.load());

  // Wake it up: set ready flag and notify the executor.
  context.ready.store(true, std::memory_order_release);
  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "process did not complete after re-wake";

  iree_task_executor_release(executor);
}

//===----------------------------------------------------------------------===//
// Error propagation tests
//===----------------------------------------------------------------------===//

TEST(ExecutorProcessTest, DrainErrorDeliveredToCompletion) {
  iree_task_executor_t* executor = CreateExecutor(1);

  CountingProcessContext context;
  context.drains_until_complete = 1;

  iree_task_process_t process;
  iree_task_process_initialize(failing_drain, 0, 1, &process);
  process.completion_fn = counting_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "process did not complete within timeout";
  EXPECT_EQ(context.completion_status_code, IREE_STATUS_DATA_LOSS);

  iree_task_executor_release(executor);
}

//===----------------------------------------------------------------------===//
// Concurrent scheduling tests
//===----------------------------------------------------------------------===//

TEST(ExecutorProcessTest, ManyIndependentProcesses) {
  iree_task_executor_t* executor = CreateExecutor(4);

  static constexpr int kProcessCount = 64;
  CountingProcessContext contexts[kProcessCount];
  iree_task_process_t processes[kProcessCount];

  for (int i = 0; i < kProcessCount; ++i) {
    contexts[i].drains_until_complete = 1;
    iree_task_process_initialize(counting_drain, 0, 1, &processes[i]);
    processes[i].completion_fn = counting_completion;
    processes[i].user_data = &contexts[i];
  }

  // Schedule all at once.
  for (int i = 0; i < kProcessCount; ++i) {
    iree_task_executor_schedule_process(executor, &processes[i]);
  }

  // Wait for all to complete.
  ASSERT_TRUE(SpinUntil([&] {
    for (int i = 0; i < kProcessCount; ++i) {
      if (!contexts[i].completed.load()) return false;
    }
    return true;
  })) << "not all processes completed within timeout";

  for (int i = 0; i < kProcessCount; ++i) {
    EXPECT_EQ(contexts[i].drain_count.load(), 1) << "process " << i;
  }

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ConcurrentScheduleFromMultipleThreads) {
  iree_task_executor_t* executor = CreateExecutor(4);

  static constexpr int kThreadCount = 8;
  static constexpr int kProcessesPerThread = 16;
  static constexpr int kTotalProcesses = kThreadCount * kProcessesPerThread;

  CountingProcessContext contexts[kTotalProcesses];
  iree_task_process_t processes[kTotalProcesses];
  for (int i = 0; i < kTotalProcesses; ++i) {
    contexts[i].drains_until_complete = 1;
    iree_task_process_initialize(counting_drain, 0, 1, &processes[i]);
    processes[i].completion_fn = counting_completion;
    processes[i].user_data = &contexts[i];
  }

  // Multiple threads schedule processes concurrently.
  std::thread threads[kThreadCount];
  for (int t = 0; t < kThreadCount; ++t) {
    threads[t] = std::thread([&, t]() {
      for (int i = 0; i < kProcessesPerThread; ++i) {
        iree_task_executor_schedule_process(
            executor, &processes[t * kProcessesPerThread + i]);
      }
    });
  }
  for (auto& t : threads) t.join();

  ASSERT_TRUE(SpinUntil([&] {
    for (int i = 0; i < kTotalProcesses; ++i) {
      if (!contexts[i].completed.load()) return false;
    }
    return true;
  })) << "not all processes completed within timeout";

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, RepeatedSleepWakeCycles) {
  // Exercises the sleeping protocol repeatedly to stress the Dekker-style
  // race between schedule_state and needs_drain.
  //
  // Each cycle uses a separate process instance because the worker may still
  // be inside drain_process (storing schedule_state, reading dependent_count)
  // when the completion callback fires. Reusing the same instance would race
  // with re-initialization.
  iree_task_executor_t* executor = CreateExecutor(2);

  static constexpr int kCycles = 50;
  SleepingProcessContext contexts[kCycles];
  iree_task_process_t processes[kCycles];

  for (int cycle = 0; cycle < kCycles; ++cycle) {
    iree_task_process_initialize(sleeping_drain, 0, 1, &processes[cycle]);
    processes[cycle].completion_fn = sleeping_completion;
    processes[cycle].user_data = &contexts[cycle];

    iree_task_executor_schedule_process(executor, &processes[cycle]);

    // Wait for sleep entry.
    ASSERT_TRUE(SpinUntil([&] { return contexts[cycle].entered_sleep.load(); }))
        << "cycle " << cycle << ": never entered sleep";

    // Wake and complete.
    contexts[cycle].ready.store(true, std::memory_order_release);
    iree_task_executor_schedule_process(executor, &processes[cycle]);

    ASSERT_TRUE(SpinUntil([&] { return contexts[cycle].completed.load(); }))
        << "cycle " << cycle << ": did not complete after re-wake";
  }

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ConcurrentSleepWakeFromMultipleThreads) {
  // Multiple threads each own a sleeping process and race against each other
  // to schedule/re-wake. Stresses the immediate list push + wake_workers
  // under contention.
  iree_task_executor_t* executor = CreateExecutor(4);

  static constexpr int kThreadCount = 8;
  std::atomic<int> completed_count{0};
  std::thread threads[kThreadCount];

  SleepingProcessContext contexts[kThreadCount];
  iree_task_process_t processes[kThreadCount];
  for (int t = 0; t < kThreadCount; ++t) {
    iree_task_process_initialize(sleeping_drain, 0, 1, &processes[t]);
    processes[t].completion_fn = sleeping_completion;
    processes[t].user_data = &contexts[t];
  }

  for (int t = 0; t < kThreadCount; ++t) {
    threads[t] = std::thread([&, t]() {
      iree_task_executor_schedule_process(executor, &processes[t]);

      // Wait for sleep.
      SpinUntil([&]() { return contexts[t].entered_sleep.load(); });

      // Wake and complete.
      contexts[t].ready.store(true, std::memory_order_release);
      iree_task_executor_schedule_process(executor, &processes[t]);

      SpinUntil([&]() { return contexts[t].completed.load(); });
      completed_count.fetch_add(1, std::memory_order_relaxed);
    });
  }
  for (auto& t : threads) t.join();

  EXPECT_EQ(completed_count.load(), kThreadCount);
  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, DependencyChainWithMultipleDrains) {
  // A → B → C where each process requires multiple drains. Exercises
  // dependent activation with non-trivial process lifetimes.
  iree_task_executor_t* executor = CreateExecutor(2);

  CountingProcessContext context_a, context_b, context_c;
  context_a.drains_until_complete = 10;
  context_b.drains_until_complete = 5;
  context_c.drains_until_complete = 3;

  iree_task_process_t a, b, c;
  iree_task_process_initialize(counting_drain, 0, 1, &a);
  iree_task_process_initialize(counting_drain, 1, 1, &b);
  iree_task_process_initialize(counting_drain, 1, 1, &c);
  a.completion_fn = counting_completion;
  a.user_data = &context_a;
  b.completion_fn = counting_completion;
  b.user_data = &context_b;
  c.completion_fn = counting_completion;
  c.user_data = &context_c;

  iree_task_process_t* a_deps[] = {&b};
  a.dependents = a_deps;
  a.dependent_count = 1;
  iree_task_process_t* b_deps[] = {&c};
  b.dependents = b_deps;
  b.dependent_count = 1;

  iree_task_executor_schedule_process(executor, &a);

  ASSERT_TRUE(SpinUntil([&] { return context_c.completed.load(); }))
      << "chain did not complete";
  EXPECT_EQ(context_a.drain_count.load(), 10);
  EXPECT_EQ(context_b.drain_count.load(), 5);
  EXPECT_EQ(context_c.drain_count.load(), 3);

  iree_task_executor_release(executor);
}

//===----------------------------------------------------------------------===//
// Compute slot tests (wake_budget > 1)
//===----------------------------------------------------------------------===//

// Context for a compute process that simulates parallel tile work.
// Multiple workers drain concurrently, each atomically claiming tiles.
struct ComputeProcessContext {
  std::atomic<int32_t> tiles_remaining;
  std::atomic<int32_t> tiles_completed{0};
  std::atomic<int32_t> active_drainers{0};
  std::atomic<bool> completed{false};
  iree_status_code_t completion_status_code = IREE_STATUS_OK;
  // Track which workers participated. Atomic because the completion callback
  // fires eagerly (first worker to observe terminal state) — other workers
  // may still be writing their entries when the main thread reads.
  std::atomic<bool> worker_participated[IREE_TASK_EXECUTOR_MAX_WORKER_COUNT] =
      {};
};

static iree_status_t compute_drain(iree_task_process_t* process,
                                   uint32_t worker_index,
                                   iree_task_process_drain_result_t* result) {
  auto* context = reinterpret_cast<ComputeProcessContext*>(process->user_data);
  context->active_drainers.fetch_add(1, std::memory_order_acq_rel);

  // Record this worker's participation.
  context->worker_participated[worker_index].store(true,
                                                   std::memory_order_relaxed);

  // Try to claim a tile.
  int32_t remaining =
      context->tiles_remaining.fetch_sub(1, std::memory_order_acq_rel);
  if (remaining <= 0) {
    // No tiles left — undo the decrement and report completion.
    context->tiles_remaining.fetch_add(1, std::memory_order_relaxed);
    int32_t active_drainers =
        context->active_drainers.fetch_sub(1, std::memory_order_acq_rel) - 1;
    result->did_work = false;
    result->completed =
        context->tiles_remaining.load(std::memory_order_acquire) <= 0 &&
        active_drainers == 0;
    return iree_ok_status();
  }

  // "Execute" the tile.
  context->tiles_completed.fetch_add(1, std::memory_order_relaxed);
  int32_t active_drainers =
      context->active_drainers.fetch_sub(1, std::memory_order_acq_rel) - 1;
  result->did_work = true;
  result->completed =
      context->tiles_remaining.load(std::memory_order_acquire) <= 0 &&
      active_drainers == 0;
  return iree_ok_status();
}

static void compute_completion(iree_task_process_t* process,
                               iree_status_t status) {
  auto* context = reinterpret_cast<ComputeProcessContext*>(process->user_data);
  context->completion_status_code = iree_status_code(status);
  context->completed.store(true, std::memory_order_release);
  iree_status_ignore(status);
}

// Context for a persistent wake_budget > 1 process that repeatedly goes idle
// and is rescheduled with one more unit of work. This mirrors the executor
// contract local-task relies on for its long-lived compute process without
// involving any HAL queue state.
struct RepeatedComputeWakeContext {
  std::atomic<int32_t> pending_work{0};
  std::atomic<int32_t> processed_work{0};
  std::atomic<int32_t> active_drainers{0};
  std::atomic<bool> shutdown{false};
  std::atomic<bool> completed{false};
};

static iree_status_t repeated_compute_wake_drain(
    iree_task_process_t* process, uint32_t worker_index,
    iree_task_process_drain_result_t* result) {
  (void)worker_index;
  auto* context =
      reinterpret_cast<RepeatedComputeWakeContext*>(process->user_data);
  context->active_drainers.fetch_add(1, std::memory_order_acq_rel);

  bool did_work = false;
  while (true) {
    int32_t pending_work =
        context->pending_work.load(std::memory_order_acquire);
    if (pending_work <= 0) break;
    if (context->pending_work.compare_exchange_weak(
            pending_work, pending_work - 1, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      context->processed_work.fetch_add(1, std::memory_order_relaxed);
      did_work = true;
      break;
    }
  }

  int32_t active_drainers =
      context->active_drainers.fetch_sub(1, std::memory_order_acq_rel) - 1;
  result->did_work = did_work;
  result->completed =
      context->shutdown.load(std::memory_order_acquire) &&
      context->pending_work.load(std::memory_order_acquire) <= 0 &&
      active_drainers == 0;
  return iree_ok_status();
}

static void repeated_compute_wake_completion(iree_task_process_t* process,
                                             iree_status_t status) {
  auto* context =
      reinterpret_cast<RepeatedComputeWakeContext*>(process->user_data);
  context->completed.store(true, std::memory_order_release);
  iree_status_ignore(status);
}

TEST(ExecutorProcessTest, ComputeSlotSingleProcess) {
  iree_task_executor_t* executor = CreateExecutor(4);

  ComputeProcessContext context;
  context.tiles_remaining.store(100);

  iree_task_process_t process;
  iree_task_process_initialize(compute_drain, /*suspend_count=*/0,
                               /*wake_budget=*/4, &process);
  process.completion_fn = compute_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "compute process did not complete within timeout";
  EXPECT_EQ(context.tiles_completed.load(), 100);
  EXPECT_EQ(context.completion_status_code, IREE_STATUS_OK);

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ComputeSlotMultipleWorkerParticipation) {
  // Use enough tiles that multiple workers should participate.
  iree_task_executor_t* executor = CreateExecutor(4);

  ComputeProcessContext context;
  context.tiles_remaining.store(10000);

  iree_task_process_t process;
  iree_task_process_initialize(compute_drain, 0, /*wake_budget=*/4, &process);
  process.completion_fn = compute_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "compute process did not complete within timeout";
  EXPECT_EQ(context.tiles_completed.load(), 10000);

  // With 10000 tiles and 4 workers, we expect multiple workers participated.
  // This is not strictly guaranteed (one worker could theoretically drain all
  // tiles before others wake up), but with 10000 tiles it's very likely.
  int participating_workers = 0;
  for (iree_host_size_t i = 0; i < IREE_TASK_EXECUTOR_MAX_WORKER_COUNT; ++i) {
    if (context.worker_participated[i].load(std::memory_order_relaxed)) {
      ++participating_workers;
    }
  }
  EXPECT_GE(participating_workers, 1);

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ComputeSlotMultipleConcurrentProcesses) {
  iree_task_executor_t* executor = CreateExecutor(4);

  static constexpr int kProcessCount = 8;
  ComputeProcessContext contexts[kProcessCount];
  iree_task_process_t processes[kProcessCount];

  for (int i = 0; i < kProcessCount; ++i) {
    contexts[i].tiles_remaining.store(50);
    iree_task_process_initialize(compute_drain, 0, /*wake_budget=*/2,
                                 &processes[i]);
    processes[i].completion_fn = compute_completion;
    processes[i].user_data = &contexts[i];
  }

  for (int i = 0; i < kProcessCount; ++i) {
    iree_task_executor_schedule_process(executor, &processes[i]);
  }

  ASSERT_TRUE(SpinUntil([&] {
    for (int i = 0; i < kProcessCount; ++i) {
      if (!contexts[i].completed.load()) return false;
    }
    return true;
  })) << "not all compute processes completed within timeout";

  for (int i = 0; i < kProcessCount; ++i) {
    EXPECT_EQ(contexts[i].tiles_completed.load(), 50) << "process " << i;
  }

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ComputeSlotWithDependencyChain) {
  // A(budget=4) → B(budget=2) → C(budget=1, immediate list).
  // Exercises compute slot → compute slot → immediate list handoff.
  iree_task_executor_t* executor = CreateExecutor(4);

  ComputeProcessContext context_a, context_b;
  CountingProcessContext context_c;
  context_a.tiles_remaining.store(200);
  context_b.tiles_remaining.store(100);
  context_c.drains_until_complete = 1;

  iree_task_process_t a, b, c;
  iree_task_process_initialize(compute_drain, 0, /*wake_budget=*/4, &a);
  iree_task_process_initialize(compute_drain, 1, /*wake_budget=*/2, &b);
  iree_task_process_initialize(counting_drain, 1, /*wake_budget=*/1, &c);

  a.completion_fn = compute_completion;
  a.user_data = &context_a;
  b.completion_fn = compute_completion;
  b.user_data = &context_b;
  c.completion_fn = counting_completion;
  c.user_data = &context_c;

  // Wire dependencies.
  iree_task_process_t* a_deps[] = {&b};
  a.dependents = a_deps;
  a.dependent_count = 1;
  iree_task_process_t* b_deps[] = {&c};
  b.dependents = b_deps;
  b.dependent_count = 1;

  iree_task_executor_schedule_process(executor, &a);

  ASSERT_TRUE(SpinUntil([&] { return context_c.completed.load(); }))
      << "chain did not complete within timeout";
  EXPECT_EQ(context_a.tiles_completed.load(), 200);
  EXPECT_EQ(context_b.tiles_completed.load(), 100);
  EXPECT_TRUE(context_a.completed.load());
  EXPECT_TRUE(context_b.completed.load());

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ComputeSlotErrorPropagation) {
  iree_task_executor_t* executor = CreateExecutor(4);

  ComputeProcessContext context;
  context.tiles_remaining.store(100);
  // Intentionally not used — we'll use the failing drain instead.
  (void)context;

  iree_task_process_t process;
  iree_task_process_initialize(failing_drain, 0, /*wake_budget=*/4, &process);
  process.completion_fn = compute_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] { return context.completed.load(); }))
      << "compute process did not complete within timeout";
  EXPECT_EQ(context.completion_status_code, IREE_STATUS_DATA_LOSS);

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ComputeSlotRepeatedSleepWakeCycles) {
  // Exercises a non-terminal wake_budget > 1 process that repeatedly goes idle
  // and is rescheduled with one unit of new work. This is the generic
  // process-level contract local-task's persistent compute process depends on.
  // Unlike the wake_budget == 1 path, a wake_budget > 1 process may transition
  // to IDLE after a final did_work=true drain when no additional drain was
  // requested, so this test waits on schedule_state rather than expecting a
  // final did_work=false drain.
  iree_task_executor_t* executor = CreateExecutor(4);

  RepeatedComputeWakeContext context;
  iree_task_process_t process;
  iree_task_process_initialize(repeated_compute_wake_drain,
                               /*suspend_count=*/0, /*wake_budget=*/4,
                               &process);
  process.completion_fn = repeated_compute_wake_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  static constexpr int kCycles = 500;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    ASSERT_TRUE(SpinUntil([&] {
      return iree_atomic_load(&process.schedule_state,
                              iree_memory_order_acquire) ==
             IREE_TASK_PROCESS_SCHEDULE_IDLE;
    })) << "cycle "
        << cycle << ": process never went idle";

    context.pending_work.fetch_add(1, std::memory_order_release);
    iree_task_executor_schedule_process(executor, &process);

    ASSERT_TRUE(SpinUntil([&] {
      return context.processed_work.load(std::memory_order_acquire) > cycle;
    })) << "cycle "
        << cycle << ": process never consumed rescheduled work";
  }

  ASSERT_TRUE(SpinUntil([&] {
    return iree_atomic_load(&process.schedule_state,
                            iree_memory_order_acquire) ==
           IREE_TASK_PROCESS_SCHEDULE_IDLE;
  })) << "process never went idle after final work item";

  context.shutdown.store(true, std::memory_order_release);
  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] {
    return context.completed.load(std::memory_order_acquire);
  })) << "process did not complete after shutdown";
  EXPECT_EQ(context.processed_work.load(std::memory_order_acquire), kCycles);

  iree_task_executor_release(executor);
}

TEST(ExecutorProcessTest, ComputeSlotConcurrentScheduleWhileDraining) {
  // External schedule_process calls can race a wake_budget > 1 process's final
  // drainer as it decides whether to release the compute slot and transition
  // the process to IDLE. Stress that handoff by repeatedly publishing one unit
  // of process-local work from a producer thread while worker threads are
  // concurrently draining the same long-lived process.
  iree_task_executor_t* executor = CreateExecutor(4);

  RepeatedComputeWakeContext context;
  iree_task_process_t process;
  iree_task_process_initialize(repeated_compute_wake_drain,
                               /*suspend_count=*/0, /*wake_budget=*/4,
                               &process);
  process.completion_fn = repeated_compute_wake_completion;
  process.user_data = &context;

  iree_task_executor_schedule_process(executor, &process);

  static constexpr int kWorkItems = 20000;
  std::thread producer([&]() {
    for (int i = 0; i < kWorkItems; ++i) {
      context.pending_work.fetch_add(1, std::memory_order_release);
      iree_task_executor_schedule_process(executor, &process);
    }
  });
  producer.join();

  ASSERT_TRUE(SpinUntil([&] {
    return context.processed_work.load(std::memory_order_acquire) == kWorkItems;
  })) << "process stranded with producer work still pending";
  EXPECT_EQ(context.pending_work.load(std::memory_order_acquire), 0);

  context.shutdown.store(true, std::memory_order_release);
  iree_task_executor_schedule_process(executor, &process);

  ASSERT_TRUE(SpinUntil([&] {
    return context.completed.load(std::memory_order_acquire);
  })) << "process did not complete after shutdown";
  EXPECT_EQ(context.processed_work.load(std::memory_order_acquire), kWorkItems);

  iree_task_executor_release(executor);
}

}  // namespace
