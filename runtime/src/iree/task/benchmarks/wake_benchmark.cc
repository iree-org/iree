// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Thread wake latency benchmarks for the task system.
//
// These benchmarks measure the cost of waking parked worker threads,
// which dominates dispatch barrier overhead in real workloads.
//
// Benchmarks:
//
//   WakeLatency/Cold/{workers}:
//     Force all workers to park (100ms idle), then submit a single
//     dispatch. Measures worst-case wake latency using manual timing
//     from submit to completion.
//
//   WakeLatency/Warm/{gap_us}/{workers}:
//     Submit dispatches with controlled gaps between them. Measures
//     how inter-dispatch gap duration affects wake overhead. Short gaps
//     (1-10us) may catch workers still spinning; long gaps (1ms+)
//     guarantee they've parked.
//
//   WakeLatency/WithSpin/{spin_us}/{workers}:
//     Cold wake latency with different spin durations. Directly
//     measures the latency benefit of spinning vs. the cost.

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "benchmark/benchmark.h"
#include "iree/task/benchmarks/benchmark_base.h"

namespace iree::task::benchmarks {

//===----------------------------------------------------------------------===//
// Cold wake latency
//===----------------------------------------------------------------------===//

// Measures worst-case wake latency: all workers parked → submit → complete.
// Uses manual timing to capture the exact submit-to-completion window.
//
// Args: [0] = worker_count
static void BM_WakeLatencyCold(::benchmark::State& state) {
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(0));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_trivial, nullptr);

  // Use enough tiles to engage all workers.
  const uint32_t tile_count = static_cast<uint32_t>(worker_count) * 4;
  iree_task_dispatch_t dispatch;
  const uint32_t workgroup_size[3] = {1, 1, 1};
  const uint32_t workgroup_count[3] = {tile_count, 1, 1};

  for (auto _ : state) {
    // Let workers park. 10ms is plenty — futex_wait enters within ~10us,
    // and even the most aggressive spin duration is 100us.
    state.PauseTiming();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    state.ResumeTiming();

    iree_task_dispatch_initialize(&context.scope, closure, workgroup_size,
                                  workgroup_count, &dispatch);

    auto start = std::chrono::high_resolution_clock::now();

    if (!context.SubmitChainAndWait(&dispatch.header, &dispatch.header)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["tiles"] = tile_count;
}

//===----------------------------------------------------------------------===//
// Warm wake with controlled gaps
//===----------------------------------------------------------------------===//

// Measures wake latency with controlled inter-dispatch gaps.
// Submits a burst of dispatches with a sleep between each.
//
// Args: [0] = gap_us, [1] = worker_count
static void BM_WakeLatencyWarm(::benchmark::State& state) {
  const int64_t gap_us = state.range(0);
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(1));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_trivial, nullptr);

  const uint32_t tile_count = static_cast<uint32_t>(worker_count) * 4;
  iree_task_dispatch_t dispatch;
  const uint32_t workgroup_size[3] = {1, 1, 1};
  const uint32_t workgroup_count[3] = {tile_count, 1, 1};

  for (auto _ : state) {
    // Inter-dispatch gap.
    if (gap_us > 0) {
      state.PauseTiming();
      std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
      state.ResumeTiming();
    }

    iree_task_dispatch_initialize(&context.scope, closure, workgroup_size,
                                  workgroup_count, &dispatch);

    auto start = std::chrono::high_resolution_clock::now();

    if (!context.SubmitChainAndWait(&dispatch.header, &dispatch.header)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.counters["gap_us"] = static_cast<double>(gap_us);
  state.counters["workers"] = static_cast<double>(worker_count);
}

//===----------------------------------------------------------------------===//
// Wake latency with spin
//===----------------------------------------------------------------------===//

// Measures cold wake latency with different spin durations.
// This is the key experiment for adaptive spinning: what's the latency
// improvement per microsecond of spin budget?
//
// Args: [0] = spin_us, [1] = worker_count
static void BM_WakeLatencyWithSpin(::benchmark::State& state) {
  const iree_duration_t spin_ns = state.range(0) * 1000;  // us → ns
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(1));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, spin_ns, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_trivial, nullptr);

  const uint32_t tile_count = static_cast<uint32_t>(worker_count) * 4;
  iree_task_dispatch_t dispatch;
  const uint32_t workgroup_size[3] = {1, 1, 1};
  const uint32_t workgroup_count[3] = {tile_count, 1, 1};

  for (auto _ : state) {
    // Let workers potentially park (but spin may keep them alive).
    state.PauseTiming();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    state.ResumeTiming();

    iree_task_dispatch_initialize(&context.scope, closure, workgroup_size,
                                  workgroup_count, &dispatch);

    auto start = std::chrono::high_resolution_clock::now();

    if (!context.SubmitChainAndWait(&dispatch.header, &dispatch.header)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.counters["spin_us"] = state.range(0);
  state.counters["workers"] = static_cast<double>(worker_count);
}

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

static void RegisterWakeBenchmarks() {
  // Cold wake latency scaling.
  for (int workers : {1, 2, 4, 8, 16}) {
    std::string name = "WakeLatency/Cold/" + std::to_string(workers) + "w";
    ::benchmark::RegisterBenchmark(
        name.c_str(),
        [](::benchmark::State& state) { BM_WakeLatencyCold(state); })
        ->Args({workers})
        ->Unit(::benchmark::kMicrosecond)
        ->UseManualTime();
  }

  // Warm wake with varying gaps.
  for (int gap_us : {0, 1, 10, 50, 100, 500, 1000, 10000}) {
    for (int workers : {4, 8}) {
      std::string name = "WakeLatency/Warm/gap" + std::to_string(gap_us) +
                         "us/" + std::to_string(workers) + "w";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_WakeLatencyWarm(state); })
          ->Args({gap_us, workers})
          ->Unit(::benchmark::kMicrosecond)
          ->UseManualTime();
    }
  }

  // Wake latency with spin — the adaptive spinning experiment.
  for (int spin_us : {0, 1, 5, 10, 25, 50, 100, 500, 1000}) {
    for (int workers : {4, 8}) {
      std::string name = "WakeLatency/WithSpin/spin" + std::to_string(spin_us) +
                         "us/" + std::to_string(workers) + "w";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_WakeLatencyWithSpin(state); })
          ->Args({spin_us, workers})
          ->Unit(::benchmark::kMicrosecond)
          ->UseManualTime();
    }
  }
}

static bool wake_benchmarks_registered_ = (RegisterWakeBenchmarks(), true);

}  // namespace iree::task::benchmarks
