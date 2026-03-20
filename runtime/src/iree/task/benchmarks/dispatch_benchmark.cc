// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Dispatch overhead benchmarks for the task system.
//
// These benchmarks measure the pure scheduling cost of dispatching work
// through the task system, isolating overhead from useful computation.
//
// Benchmarks:
//
//   DispatchChain/Noop/{dispatches}x{tiles}/{workers}:
//     Chain of dispatches with zero-work tiles. Measures scheduling overhead
//     per dispatch barrier as a function of tile count and worker count.
//     This is the number we want to minimize: it's the tax on every model
//     layer boundary.
//
//   DispatchChain/Trivial/{dispatches}x{tiles}/{workers}:
//     Same chain but with trivial work per tile (~10ns). Ensures dispatch
//     closures actually fire and measures any overhead from closure invocation.
//
//   DispatchChainWithBarriers/{dispatches}x{tiles}/{workers}:
//     Chain with explicit barrier tasks between dispatches (matches real HAL
//     command buffer patterns). Shows the additional cost of barrier tasks.
//
//   SingleDispatch/{tiles}/{workers}:
//     Single dispatch with varying tile counts. Isolates dispatch fan-out
//     and shard allocation cost from chain/barrier overhead.
//
//   DispatchOverheadPerTile/{tiles}/{workers}:
//     Single dispatch measuring per-tile overhead. Reports time/tile to show
//     how tile count affects amortization of fixed dispatch costs.

#include <cstring>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/task/benchmarks/benchmark_base.h"

namespace iree::task::benchmarks {

//===----------------------------------------------------------------------===//
// Dispatch chain benchmarks
//===----------------------------------------------------------------------===//

// Measures dispatch chain throughput with noop tiles.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count
static void BM_DispatchChainNoop(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_noop, nullptr);

  // Stack-allocate dispatch array. For large chains, heap-allocate.
  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  // Report dispatches/second and per-dispatch overhead.
  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
}

// Measures dispatch chain throughput with trivial work per tile.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count
static void BM_DispatchChainTrivial(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_trivial, nullptr);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
}

// Measures dispatch chain with explicit barrier tasks between dispatches.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count
static void BM_DispatchChainWithBarriers(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_noop, nullptr);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);
  std::vector<iree_task_barrier_t> barriers(
      dispatch_count > 1 ? dispatch_count - 1 : 0);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChainWithBarriers(&context.scope, dispatch_count, tile_count,
                                   closure, dispatches.data(), barriers.data(),
                                   &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
}

//===----------------------------------------------------------------------===//
// Single dispatch benchmarks
//===----------------------------------------------------------------------===//

// Measures single dispatch overhead as a function of tile count.
// Isolates dispatch fan-out and shard allocation cost.
//
// Args: [0] = tile_count, [1] = worker_count
static void BM_SingleDispatch(::benchmark::State& state) {
  const uint32_t tile_count = static_cast<uint32_t>(state.range(0));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(1));

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_noop, nullptr);

  iree_task_dispatch_t dispatch;
  const uint32_t workgroup_size[3] = {1, 1, 1};
  const uint32_t workgroup_count[3] = {tile_count, 1, 1};

  for (auto _ : state) {
    iree_task_dispatch_initialize(&context.scope, closure, workgroup_size,
                                  workgroup_count, &dispatch);
    if (!context.SubmitChainAndWait(&dispatch.header, &dispatch.header)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(tile_count));
  state.counters["tiles"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
}

//===----------------------------------------------------------------------===//
// Spin delay comparison benchmarks
//===----------------------------------------------------------------------===//

// Measures dispatch chain latency with different worker spin durations.
// This directly measures the effect of spinning on barrier overhead.
//
// Args: [0] = dispatch_count, [1] = tile_count, [2] = worker_count,
//       [3] = spin_us
static void BM_DispatchChainWithSpin(::benchmark::State& state) {
  const int dispatch_count = static_cast<int>(state.range(0));
  const uint32_t tile_count = static_cast<uint32_t>(state.range(1));
  const iree_host_size_t worker_count =
      static_cast<iree_host_size_t>(state.range(2));
  const iree_duration_t spin_ns = state.range(3) * 1000;  // us → ns

  TaskBenchmarkContext context;
  if (!context.Initialize(worker_count, spin_ns, state)) return;

  iree_task_dispatch_closure_t closure =
      iree_task_make_dispatch_closure(dispatch_closure_noop, nullptr);

  std::vector<iree_task_dispatch_t> dispatches(dispatch_count);

  for (auto _ : state) {
    iree_task_t* head = nullptr;
    iree_task_t* tail = nullptr;
    BuildDispatchChain(&context.scope, dispatch_count, tile_count, closure,
                       dispatches.data(), &head, &tail);

    if (!context.SubmitChainAndWait(head, tail)) {
      state.SkipWithError("Submit/wait failed");
      break;
    }
  }

  state.SetItemsProcessed(state.iterations() *
                          static_cast<int64_t>(dispatch_count));
  state.counters["dispatches"] = dispatch_count;
  state.counters["tiles_per_dispatch"] = tile_count;
  state.counters["workers"] = static_cast<double>(worker_count);
  state.counters["spin_us"] = state.range(3);
}

//===----------------------------------------------------------------------===//
// Benchmark registration
//===----------------------------------------------------------------------===//

// Helper to generate a descriptive name for multi-arg benchmarks.
static void RegisterDispatchChainBenchmarks() {
  // Dispatch chain with noop tiles.
  // Sweep: dispatch_count × tile_count × worker_count.
  for (int dispatches : {1, 4, 16, 64}) {
    for (int tiles : {1, 8, 32, 128, 1024}) {
      for (int workers : {1, 2, 4, 8}) {
        std::string name = "DispatchChain/Noop/" + std::to_string(dispatches) +
                           "x" + std::to_string(tiles) + "/" +
                           std::to_string(workers) + "w";
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [dispatches, tiles, workers](::benchmark::State& state) {
              state.SetLabel(std::to_string(dispatches) + " dispatches, " +
                             std::to_string(tiles) + " tiles/dispatch, " +
                             std::to_string(workers) + " workers");
              // Manually set ranges for the benchmark functions.
              // We use the lambda capture instead of state.range() here,
              // but still call the standard Args-based function.
              BM_DispatchChainNoop(state);
            })
            ->Args({dispatches, tiles, workers})
            ->Unit(::benchmark::kMicrosecond)
            ->MeasureProcessCPUTime()
            ->UseRealTime();
      }
    }
  }

  // Dispatch chain with trivial work — subset of configurations.
  for (int dispatches : {16, 64}) {
    for (int tiles : {32, 128}) {
      for (int workers : {1, 4, 8}) {
        std::string name =
            "DispatchChain/Trivial/" + std::to_string(dispatches) + "x" +
            std::to_string(tiles) + "/" + std::to_string(workers) + "w";
        ::benchmark::RegisterBenchmark(
            name.c_str(),
            [](::benchmark::State& state) { BM_DispatchChainTrivial(state); })
            ->Args({dispatches, tiles, workers})
            ->Unit(::benchmark::kMicrosecond)
            ->MeasureProcessCPUTime()
            ->UseRealTime();
      }
    }
  }

  // Dispatch chain with barriers — compare against no-barrier version.
  for (int dispatches : {16, 64}) {
    for (int tiles : {32, 128}) {
      for (int workers : {1, 4, 8}) {
        std::string name =
            "DispatchChainWithBarriers/" + std::to_string(dispatches) + "x" +
            std::to_string(tiles) + "/" + std::to_string(workers) + "w";
        ::benchmark::RegisterBenchmark(name.c_str(),
                                       [](::benchmark::State& state) {
                                         BM_DispatchChainWithBarriers(state);
                                       })
            ->Args({dispatches, tiles, workers})
            ->Unit(::benchmark::kMicrosecond)
            ->MeasureProcessCPUTime()
            ->UseRealTime();
      }
    }
  }

  // Single dispatch scalability.
  for (int tiles : {1, 4, 16, 64, 256, 1024, 4096}) {
    for (int workers : {1, 2, 4, 8}) {
      std::string name = "SingleDispatch/" + std::to_string(tiles) + "t/" +
                         std::to_string(workers) + "w";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_SingleDispatch(state); })
          ->Args({tiles, workers})
          ->Unit(::benchmark::kMicrosecond)
          ->MeasureProcessCPUTime()
          ->UseRealTime();
    }
  }

  // Spin delay comparison — key experiment for adaptive spinning design.
  // Fixed workload (64 dispatches × 128 tiles) with varying spin durations.
  for (int workers : {4, 8}) {
    for (int spin_us : {0, 1, 5, 10, 25, 50, 100}) {
      std::string name = "DispatchChainWithSpin/64x128/" +
                         std::to_string(workers) + "w/spin" +
                         std::to_string(spin_us) + "us";
      ::benchmark::RegisterBenchmark(
          name.c_str(),
          [](::benchmark::State& state) { BM_DispatchChainWithSpin(state); })
          ->Args({64, 128, workers, spin_us})
          ->Unit(::benchmark::kMicrosecond)
          ->MeasureProcessCPUTime()
          ->UseRealTime();
    }
  }
}

// Static registration at load time.
static bool dispatch_benchmarks_registered_ =
    (RegisterDispatchChainBenchmarks(), true);

}  // namespace iree::task::benchmarks
