// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Microbenchmarks for frontier operations (compare, merge, is_satisfied).
//
// These operations are on the latency-critical path: network frontier
// propagation → tracker advance → waiter dispatch. The merge fast path
// (same-axis-set epoch-max) executes on every network frontier update in
// steady state. Understanding the cost at various entry counts is essential
// for validating that frontier propagation stays within the "invisible"
// latency budget relative to network RTT.
//
// Build/run:
//   iree-bazel-run //runtime/src/iree/async:frontier_benchmark
//
// With detailed stats:
//   iree-bazel-run //runtime/src/iree/async:frontier_benchmark -- \
//     --benchmark_counters_tabular=true
//
// Key things to look for in results:
//   - Fast-path vs slow-path gap: quantifies the value of same-axis detection.
//   - Path 2 vs path 3 gap: quantifies the cost of entry movement.
//   - Entry count scaling: should be linear. Non-linear growth indicates
//     cache effects or branch prediction degradation.
//   - "Update" vs "NoChange" merge: isolates the store cost from comparison.

#include <cstdint>
#include <cstring>

#include "benchmark/benchmark.h"
#include "iree/async/frontier.h"

namespace {

//===----------------------------------------------------------------------===//
// Frontier construction helpers
//===----------------------------------------------------------------------===//

// Maximum entry count we benchmark. 128 entries × 16 bytes = 2KB of entry data,
// well within L1 cache but large enough to show scaling behavior.
static constexpr int kMaxEntries = 128;

// Stack storage for a frontier with up to kMaxEntries entries.
struct FrontierStorage {
  alignas(16) uint8_t data[sizeof(iree_async_frontier_t) +
                           kMaxEntries * sizeof(iree_async_frontier_entry_t)];

  iree_async_frontier_t* frontier() {
    return reinterpret_cast<iree_async_frontier_t*>(data);
  }
};

// Populates a frontier with N entries using sequential device indices on the
// given machine. Axes are sorted (device 0, 1, 2, ...) and epochs start at
// base_epoch and increment by epoch_stride.
//
// This gives deterministic, sorted, non-trivial values that the compiler
// cannot constant-fold through a function call boundary.
static void PopulateFrontier(iree_async_frontier_t* frontier, int entry_count,
                             uint8_t machine, uint64_t base_epoch,
                             uint64_t epoch_stride) {
  iree_async_frontier_initialize(frontier, static_cast<uint8_t>(entry_count));
  for (int i = 0; i < entry_count; ++i) {
    frontier->entries[i].axis =
        iree_async_axis_make_queue(1, machine, static_cast<uint8_t>(i), 0);
    frontier->entries[i].epoch = base_epoch + i * epoch_stride;
  }
}

// Populates a frontier with even-indexed device axes (0, 2, 4, ...).
// Used for interleaved merge benchmarks.
static void PopulateFrontierEvenAxes(iree_async_frontier_t* frontier,
                                     int entry_count, uint8_t machine,
                                     uint64_t base_epoch) {
  iree_async_frontier_initialize(frontier, static_cast<uint8_t>(entry_count));
  for (int i = 0; i < entry_count; ++i) {
    frontier->entries[i].axis =
        iree_async_axis_make_queue(1, machine, static_cast<uint8_t>(i * 2), 0);
    frontier->entries[i].epoch = base_epoch + i;
  }
}

// Populates a frontier with odd-indexed device axes (1, 3, 5, ...).
// Used for interleaved merge benchmarks.
static void PopulateFrontierOddAxes(iree_async_frontier_t* frontier,
                                    int entry_count, uint8_t machine,
                                    uint64_t base_epoch) {
  iree_async_frontier_initialize(frontier, static_cast<uint8_t>(entry_count));
  for (int i = 0; i < entry_count; ++i) {
    frontier->entries[i].axis = iree_async_axis_make_queue(
        1, machine, static_cast<uint8_t>(i * 2 + 1), 0);
    frontier->entries[i].epoch = base_epoch + i;
  }
}

// Populates a frontier with a half-overlap pattern relative to a sequential
// frontier. The first half of entries share axes with a sequential frontier
// (devices 0..N/2-1), the second half are unique (devices N..N+N/2-1).
static void PopulateFrontierHalfOverlap(iree_async_frontier_t* frontier,
                                        int entry_count, uint8_t machine,
                                        int base_sequential_count,
                                        uint64_t base_epoch) {
  iree_async_frontier_initialize(frontier, static_cast<uint8_t>(entry_count));
  int overlap_count = entry_count / 2;
  int unique_count = entry_count - overlap_count;
  // Overlapping entries: same devices as the first half of the sequential set.
  for (int i = 0; i < overlap_count; ++i) {
    frontier->entries[i].axis =
        iree_async_axis_make_queue(1, machine, static_cast<uint8_t>(i), 0);
    frontier->entries[i].epoch = base_epoch + i;
  }
  // Unique entries: devices beyond the sequential set's range.
  for (int i = 0; i < unique_count; ++i) {
    frontier->entries[overlap_count + i].axis = iree_async_axis_make_queue(
        1, machine, static_cast<uint8_t>(base_sequential_count + i), 0);
    frontier->entries[overlap_count + i].epoch = base_epoch + overlap_count + i;
  }
}

// Saves a copy of a frontier's entries for reset between iterations.
struct FrontierSnapshot {
  uint8_t entry_count;
  iree_async_frontier_entry_t entries[kMaxEntries];

  void capture(const iree_async_frontier_t* frontier) {
    entry_count = frontier->entry_count;
    memcpy(entries, frontier->entries,
           entry_count * sizeof(iree_async_frontier_entry_t));
  }

  void restore(iree_async_frontier_t* frontier) const {
    frontier->entry_count = entry_count;
    memcpy(frontier->entries, entries,
           entry_count * sizeof(iree_async_frontier_entry_t));
  }
};

//===----------------------------------------------------------------------===//
// Compare benchmarks
//===----------------------------------------------------------------------===//

// Fast path: identical axis sets, all epochs equal (result: EQUAL).
// This measures the cost of the axis-matching check + epoch comparison loop
// with perfect branch prediction (no direction changes).
static void BM_Compare_FastPath_Equal(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_a, storage_b;
  PopulateFrontier(storage_a.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_b.frontier(), entry_count, 0, 100, 10);
  for (auto _ : state) {
    auto result =
        iree_async_frontier_compare(storage_a.frontier(), storage_b.frontier());
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Compare_FastPath_Equal)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

// Fast path: identical axis sets, a dominates b (result: AFTER).
// All of a's epochs are higher. Measures the same-axes fast path where the
// branch (epoch_a > epoch_b) is always taken.
static void BM_Compare_FastPath_ADominates(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_a, storage_b;
  PopulateFrontier(storage_a.frontier(), entry_count, 0, 200, 10);
  PopulateFrontier(storage_b.frontier(), entry_count, 0, 100, 10);
  for (auto _ : state) {
    auto result =
        iree_async_frontier_compare(storage_a.frontier(), storage_b.frontier());
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Compare_FastPath_ADominates)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

// Fast path: identical axis sets, concurrent (a ahead on evens, b on odds).
// This is the worst case for the fast path because it cannot early-exit until
// it has seen both a less and a greater epoch.
static void BM_Compare_FastPath_Concurrent(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_a, storage_b;
  PopulateFrontier(storage_a.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_b.frontier(), entry_count, 0, 100, 10);
  // Make a ahead on even indices, b ahead on odd indices.
  for (int i = 0; i < entry_count; ++i) {
    if (i % 2 == 0) {
      storage_a.frontier()->entries[i].epoch += 50;
    } else {
      storage_b.frontier()->entries[i].epoch += 50;
    }
  }
  for (auto _ : state) {
    auto result =
        iree_async_frontier_compare(storage_a.frontier(), storage_b.frontier());
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Compare_FastPath_Concurrent)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

// Slow path: completely disjoint axis sets (no overlap).
// a has machine=0 axes, b has machine=1 axes. Every axis in each frontier is
// unique to it, so the result is always CONCURRENT. Measures the merge-scan
// with the 3-way branch always taking the < or > path (never ==).
static void BM_Compare_SlowPath_Disjoint(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_a, storage_b;
  PopulateFrontier(storage_a.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_b.frontier(), entry_count, 1, 100, 10);
  for (auto _ : state) {
    auto result =
        iree_async_frontier_compare(storage_a.frontier(), storage_b.frontier());
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Compare_SlowPath_Disjoint)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

// Slow path: interleaved axis sets (a has even devices, b has odd devices).
// The merge-scan alternates between a-advance and b-advance on every step.
// This gives unpredictable branching (alternating < and > comparisons).
static void BM_Compare_SlowPath_Interleaved(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_a, storage_b;
  PopulateFrontierEvenAxes(storage_a.frontier(), entry_count, 0, 100);
  PopulateFrontierOddAxes(storage_b.frontier(), entry_count, 0, 100);
  for (auto _ : state) {
    auto result =
        iree_async_frontier_compare(storage_a.frontier(), storage_b.frontier());
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Compare_SlowPath_Interleaved)->Arg(4)->Arg(16)->Arg(64);

// Slow path: half overlap (50% shared axes, 50% unique to each).
// This is the realistic "topology changed slightly" case.
static void BM_Compare_SlowPath_HalfOverlap(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_a, storage_b;
  PopulateFrontier(storage_a.frontier(), entry_count, 0, 100, 10);
  PopulateFrontierHalfOverlap(storage_b.frontier(), entry_count, 0, entry_count,
                              100);
  for (auto _ : state) {
    auto result =
        iree_async_frontier_compare(storage_a.frontier(), storage_b.frontier());
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Compare_SlowPath_HalfOverlap)->Arg(4)->Arg(16)->Arg(64);

//===----------------------------------------------------------------------===//
// Merge benchmarks: path 2 (same axis set, epoch-max in place)
//===----------------------------------------------------------------------===//

// Path 2: source has higher epochs on all entries (all stores execute).
// This is the steady-state hot path: same topology, epochs advancing.
// We reset target epochs between iterations to ensure stores actually happen.
static void BM_Merge_EpochMax_AllUpdate(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_source.frontier(), entry_count, 0, 200, 10);
  // Save target state for reset.
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(entry_count),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_EpochMax_AllUpdate)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

// Path 2: source has lower epochs on all entries (no stores, comparison only).
// This measures the pure comparison cost without any memory writes.
// Realistic scenario: receiving a redundant/old frontier update.
static void BM_Merge_EpochMax_NoneUpdate(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 200, 10);
  PopulateFrontier(storage_source.frontier(), entry_count, 0, 100, 10);
  for (auto _ : state) {
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(entry_count),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_EpochMax_NoneUpdate)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

// Path 2: half of entries update (alternating higher/lower epochs).
// Tests branch prediction with unpredictable update pattern.
static void BM_Merge_EpochMax_HalfUpdate(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_source.frontier(), entry_count, 0, 100, 10);
  // Source is ahead on even indices, behind on odd indices.
  for (int i = 0; i < entry_count; ++i) {
    if (i % 2 == 0) {
      storage_source.frontier()->entries[i].epoch += 50;
    } else {
      storage_source.frontier()->entries[i].epoch -= 50;
    }
  }
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(entry_count),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_EpochMax_HalfUpdate)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

//===----------------------------------------------------------------------===//
// Merge benchmarks: path 3 (different axis sets, right-to-left merge)
//===----------------------------------------------------------------------===//

// Path 3: source axes entirely after target axes (append pattern).
// All source entries go at the end, target entries don't move.
// This is the cheapest path-3 case: no target entry shifting.
static void BM_Merge_RightToLeft_Append(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  const int capacity = entry_count * 2;
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_source.frontier(), entry_count, 1, 100, 10);
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(capacity),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_RightToLeft_Append)->Arg(4)->Arg(16)->Arg(64);

// Path 3: source axes entirely before target axes (prepend pattern).
// All target entries shift right by source_count positions.
// This is the most expensive path-3 case: maximum target movement.
static void BM_Merge_RightToLeft_Prepend(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  const int capacity = entry_count * 2;
  FrontierStorage storage_target, storage_source;
  // Target on machine=1 (higher axis values), source on machine=0 (lower).
  PopulateFrontier(storage_target.frontier(), entry_count, 1, 100, 10);
  PopulateFrontier(storage_source.frontier(), entry_count, 0, 100, 10);
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(capacity),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_RightToLeft_Prepend)->Arg(4)->Arg(16)->Arg(64);

// Path 3: interleaved axes (target has even devices, source has odd).
// Every source entry inserts between two target entries.
// This is the worst case for branch prediction in the merge scan.
static void BM_Merge_RightToLeft_Interleaved(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  const int capacity = entry_count * 2;
  FrontierStorage storage_target, storage_source;
  PopulateFrontierEvenAxes(storage_target.frontier(), entry_count, 0, 100);
  PopulateFrontierOddAxes(storage_source.frontier(), entry_count, 0, 100);
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(capacity),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_RightToLeft_Interleaved)->Arg(4)->Arg(16)->Arg(64);

// Path 3: half overlap (50% shared axes get epoch-max, 50% new axes inserted).
// Realistic case: two machines with partially shared topology.
static void BM_Merge_RightToLeft_HalfOverlap(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  // Worst case: entry_count shared + entry_count/2 new = 1.5× original.
  const int capacity = entry_count + entry_count;
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 100, 10);
  PopulateFrontierHalfOverlap(storage_source.frontier(), entry_count, 0,
                              entry_count, 200);
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(capacity),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_RightToLeft_HalfOverlap)->Arg(4)->Arg(16)->Arg(64);

//===----------------------------------------------------------------------===//
// Merge benchmarks: path selection overhead
//===----------------------------------------------------------------------===//

// Measures path 1 (source empty). This is the O(1) early exit.
// Establishes the baseline measurement overhead (function call + branch).
static void BM_Merge_SourceEmpty(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 100, 10);
  iree_async_frontier_initialize(storage_source.frontier(), 0);
  for (auto _ : state) {
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(entry_count),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_SourceEmpty)->Arg(4)->Arg(64);

// Measures the fast-path detection overhead: same entry count but DIFFERENT
// axes, forcing fall-through to path 3. Compares to
// BM_Merge_EpochMax_NoneUpdate which has matching axes (stays in path 2). The
// difference is the cost of the axis-matching scan that discovers mismatch +
// path 3 execution.
static void BM_Merge_FastPathMiss(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  const int capacity = entry_count * 2;
  FrontierStorage storage_target, storage_source;
  PopulateFrontier(storage_target.frontier(), entry_count, 0, 100, 10);
  // Same entry count, but different axes (machine=1 vs machine=0).
  // The fast-path check compares axes, finds mismatch on entry 0, falls
  // through.
  PopulateFrontier(storage_source.frontier(), entry_count, 1, 100, 10);
  FrontierSnapshot snapshot;
  snapshot.capture(storage_target.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_target.frontier());
    iree_async_frontier_merge(storage_target.frontier(),
                              static_cast<uint8_t>(capacity),
                              storage_source.frontier());
    benchmark::DoNotOptimize(storage_target.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Merge_FastPathMiss)->Arg(4)->Arg(16)->Arg(64);

//===----------------------------------------------------------------------===//
// IsSatisfied benchmarks
//===----------------------------------------------------------------------===//

// All entries satisfied (full scan, no early exit).
// This is the common case in steady state: the frontier has been reached.
static void BM_IsSatisfied_AllSatisfied(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_frontier;
  PopulateFrontier(storage_frontier.frontier(), entry_count, 0, 100, 10);
  // Current epochs are ahead of the frontier on all axes.
  iree_async_frontier_entry_t current_epochs[kMaxEntries];
  for (int i = 0; i < entry_count; ++i) {
    current_epochs[i].axis =
        iree_async_axis_make_queue(1, 0, static_cast<uint8_t>(i), 0);
    current_epochs[i].epoch = 200 + i * 10;
  }
  for (auto _ : state) {
    bool result = iree_async_frontier_is_satisfied(storage_frontier.frontier(),
                                                   current_epochs, entry_count);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IsSatisfied_AllSatisfied)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128);

// First entry not satisfied (immediate early exit).
// Best case: one comparison then return false.
static void BM_IsSatisfied_FirstFails(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_frontier;
  PopulateFrontier(storage_frontier.frontier(), entry_count, 0, 100, 10);
  // Current epochs: first axis is behind, rest are ahead.
  iree_async_frontier_entry_t current_epochs[kMaxEntries];
  for (int i = 0; i < entry_count; ++i) {
    current_epochs[i].axis =
        iree_async_axis_make_queue(1, 0, static_cast<uint8_t>(i), 0);
    current_epochs[i].epoch = (i == 0) ? 50 : 200 + i * 10;
  }
  for (auto _ : state) {
    bool result = iree_async_frontier_is_satisfied(storage_frontier.frontier(),
                                                   current_epochs, entry_count);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IsSatisfied_FirstFails)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

// Last entry not satisfied (almost-full scan then early exit).
// Worst case that still exits early: scans N-1 entries before finding failure.
static void BM_IsSatisfied_LastFails(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_frontier;
  PopulateFrontier(storage_frontier.frontier(), entry_count, 0, 100, 10);
  // Current epochs: all ahead except the last axis.
  iree_async_frontier_entry_t current_epochs[kMaxEntries];
  for (int i = 0; i < entry_count; ++i) {
    current_epochs[i].axis =
        iree_async_axis_make_queue(1, 0, static_cast<uint8_t>(i), 0);
    current_epochs[i].epoch = (i == entry_count - 1) ? 50 : 200 + i * 10;
  }
  for (auto _ : state) {
    bool result = iree_async_frontier_is_satisfied(storage_frontier.frontier(),
                                                   current_epochs, entry_count);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IsSatisfied_LastFails)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

// Midpoint failure: scans exactly half the entries before finding failure.
// Measures the average-case early exit behavior.
static void BM_IsSatisfied_MidpointFails(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_frontier;
  PopulateFrontier(storage_frontier.frontier(), entry_count, 0, 100, 10);
  // Current epochs: ahead on first half, behind on midpoint.
  iree_async_frontier_entry_t current_epochs[kMaxEntries];
  int midpoint = entry_count / 2;
  for (int i = 0; i < entry_count; ++i) {
    current_epochs[i].axis =
        iree_async_axis_make_queue(1, 0, static_cast<uint8_t>(i), 0);
    current_epochs[i].epoch = (i == midpoint) ? 50 : 200 + i * 10;
  }
  for (auto _ : state) {
    bool result = iree_async_frontier_is_satisfied(storage_frontier.frontier(),
                                                   current_epochs, entry_count);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IsSatisfied_MidpointFails)->Arg(16)->Arg(64)->Arg(128);

// Current epochs have extra axes beyond what the frontier references.
// The is_satisfied scan must skip past irrelevant axes in current_epochs.
// Measures the cost of the inner while-loop that advances past smaller axes.
static void BM_IsSatisfied_SparseCurrentEpochs(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_frontier;
  // Frontier references even-indexed axes only.
  PopulateFrontierEvenAxes(storage_frontier.frontier(), entry_count, 0, 100);
  // Current epochs include ALL axes (even + odd), so the scan skips odds.
  iree_async_frontier_entry_t current_epochs[kMaxEntries * 2];
  int current_count = entry_count * 2;
  for (int i = 0; i < current_count; ++i) {
    current_epochs[i].axis =
        iree_async_axis_make_queue(1, 0, static_cast<uint8_t>(i), 0);
    current_epochs[i].epoch = 200 + i;
  }
  for (auto _ : state) {
    bool result = iree_async_frontier_is_satisfied(
        storage_frontier.frontier(), current_epochs, current_count);
    benchmark::DoNotOptimize(result);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_IsSatisfied_SparseCurrentEpochs)->Arg(4)->Arg(16)->Arg(64);

//===----------------------------------------------------------------------===//
// Validate benchmarks
//===----------------------------------------------------------------------===//

// Valid frontier (full scan of both loops).
// Establishes the cost of a debug-path validation pass.
static void BM_Validate(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage;
  PopulateFrontier(storage.frontier(), entry_count, 0, 100, 10);
  for (auto _ : state) {
    iree_status_t status = iree_async_frontier_validate(storage.frontier());
    benchmark::DoNotOptimize(status);
    // status is ok_status (NULL pointer), no free needed.
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Validate)->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(128);

//===----------------------------------------------------------------------===//
// End-to-end scenario benchmarks
//===----------------------------------------------------------------------===//

// Simulates the steady-state frontier propagation hot path:
// receive frontier → compare to local → merge if newer.
// This is what happens on every network frontier update.
static void BM_Scenario_NetworkUpdate(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_local, storage_remote;
  PopulateFrontier(storage_local.frontier(), entry_count, 0, 100, 10);
  PopulateFrontier(storage_remote.frontier(), entry_count, 0, 110, 10);
  FrontierSnapshot snapshot;
  snapshot.capture(storage_local.frontier());
  for (auto _ : state) {
    snapshot.restore(storage_local.frontier());
    // Compare to see if update is newer.
    auto comparison = iree_async_frontier_compare(storage_local.frontier(),
                                                  storage_remote.frontier());
    benchmark::DoNotOptimize(comparison);
    // If remote is ahead, merge it in.
    if (comparison == IREE_ASYNC_FRONTIER_BEFORE) {
      iree_async_frontier_merge(storage_local.frontier(),
                                static_cast<uint8_t>(entry_count),
                                storage_remote.frontier());
    }
    benchmark::DoNotOptimize(storage_local.frontier()->entries[0].epoch);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scenario_NetworkUpdate)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

// Simulates checking if a pending operation can execute:
// a waiter's frontier is checked against the tracker's current state.
// This happens on every advance() call for every pending waiter.
static void BM_Scenario_WaiterCheck(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  FrontierStorage storage_waiter;
  PopulateFrontier(storage_waiter.frontier(), entry_count, 0, 100, 10);
  // Tracker state: most axes have advanced past the waiter's requirements,
  // but we vary which one is "last to arrive" to avoid perfect prediction.
  iree_async_frontier_entry_t tracker_epochs[kMaxEntries];
  for (int i = 0; i < entry_count; ++i) {
    tracker_epochs[i].axis =
        iree_async_axis_make_queue(1, 0, static_cast<uint8_t>(i), 0);
    // All epochs at or above the frontier's targets.
    tracker_epochs[i].epoch = 100 + i * 10;
  }
  for (auto _ : state) {
    bool satisfied = iree_async_frontier_is_satisfied(
        storage_waiter.frontier(), tracker_epochs, entry_count);
    benchmark::DoNotOptimize(satisfied);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scenario_WaiterCheck)->Arg(1)->Arg(4)->Arg(8)->Arg(16)->Arg(32);

}  // namespace
