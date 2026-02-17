// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Microbenchmarks for frontier tracker operations.
//
// The frontier tracker is on the critical path for async dispatch:
//   GPU completes -> semaphore signal -> tracker advance -> waiter dispatch
//
// Key operations to measure:
//   - advance() with no waiters: baseline CAS + mutex overhead
//   - advance() with unaffected waiters: cost of scanning past irrelevant
//   waiters
//   - advance() that dispatches: full satisfaction check + callback invocation
//   - wait() immediate: already-satisfied fast path
//   - wait() pending: insertion into waiter list
//   - cancel_wait(): removal from waiter list
//
// Build/run:
//   iree-bazel-run //runtime/src/iree/async:frontier_tracker_benchmark
//
// With detailed stats:
//   iree-bazel-run //runtime/src/iree/async:frontier_tracker_benchmark -- \
//     --benchmark_counters_tabular=true

#include <cstdint>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Creates a queue axis for benchmarking.
static iree_async_axis_t Axis(uint8_t device) {
  return iree_async_axis_make_queue(1, 0, device, 0);
}

// Maximum axis table capacity for benchmarks.
static constexpr int kMaxAxes = 64;

// RAII wrapper for tracker lifecycle.
class TrackerFixture {
 public:
  explicit TrackerFixture(uint32_t capacity) : entries_(capacity) {
    iree_status_t status = iree_async_frontier_tracker_initialize(
        &tracker_, entries_.data(), capacity, iree_allocator_system());
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      std::abort();
    }
  }

  ~TrackerFixture() { iree_async_frontier_tracker_deinitialize(&tracker_); }

  iree_async_frontier_tracker_t* tracker() { return &tracker_; }

  void AddAxis(iree_async_axis_t axis) {
    iree_async_axis_table_add(&tracker_.axis_table, axis, nullptr);
  }

  void ResetEpochs() {
    for (uint32_t i = 0; i < tracker_.axis_table.count; ++i) {
      iree_atomic_store(&tracker_.axis_table.entries[i].current_epoch, 0,
                        iree_memory_order_release);
    }
  }

 private:
  std::vector<iree_async_axis_table_entry_t> entries_;
  iree_async_frontier_tracker_t tracker_;
};

// Storage for a frontier with up to N entries.
template <int N>
struct FrontierStorage {
  alignas(16) uint8_t data[sizeof(iree_async_frontier_t) +
                           N * sizeof(iree_async_frontier_entry_t)];

  iree_async_frontier_t* frontier() {
    return reinterpret_cast<iree_async_frontier_t*>(data);
  }

  void Initialize(int entry_count) {
    iree_async_frontier_initialize(frontier(),
                                   static_cast<uint8_t>(entry_count));
  }

  void SetEntry(int index, iree_async_axis_t axis, uint64_t epoch) {
    frontier()->entries[index].axis = axis;
    frontier()->entries[index].epoch = epoch;
  }
};

// Minimal callback that just prevents optimization.
static void MinimalCallback(void* user_data, iree_status_t status) {
  benchmark::DoNotOptimize(status);
  // Note: status is iree_ok_status() which is NULL, so no free needed.
}

//===----------------------------------------------------------------------===//
// Baseline benchmarks (to understand overhead)
//===----------------------------------------------------------------------===//

// Just mutex lock/unlock to measure the futex overhead.
static void BM_Baseline_MutexLockUnlock(benchmark::State& state) {
  iree_slim_mutex_t mutex;
  iree_slim_mutex_initialize(&mutex);
  for (auto _ : state) {
    iree_slim_mutex_lock(&mutex);
    benchmark::ClobberMemory();
    iree_slim_mutex_unlock(&mutex);
  }
  iree_slim_mutex_deinitialize(&mutex);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Baseline_MutexLockUnlock);

// Callback invocation overhead.
static void BM_Baseline_CallbackInvoke(benchmark::State& state) {
  for (auto _ : state) {
    MinimalCallback(nullptr, iree_ok_status());
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Baseline_CallbackInvoke);

// Function pointer call overhead (indirect call).
static void BM_Baseline_IndirectCall(benchmark::State& state) {
  iree_async_frontier_waiter_fn_t callback = MinimalCallback;
  void* user_data = nullptr;
  for (auto _ : state) {
    callback(user_data, iree_ok_status());
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Baseline_IndirectCall);

//===----------------------------------------------------------------------===//
// Advance benchmarks
//===----------------------------------------------------------------------===//

// Advance with no waiters: pure CAS + mutex overhead.
// This is the hot path when axis advances happen but no one is waiting.
static void BM_Advance_NoWaiters(benchmark::State& state) {
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  uint64_t epoch = 1;
  for (auto _ : state) {
    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), epoch);
    benchmark::DoNotOptimize(dispatched);
    ++epoch;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Advance_NoWaiters);

// Advance with waiters that don't reference the advancing axis.
// Measures the cost of the quick-check optimization (skip waiters that don't
// care about this axis).
static void BM_Advance_UnaffectedWaiters(benchmark::State& state) {
  const int waiter_count = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);

  // Axis 0 is what we advance; axes 1..N are what waiters wait on.
  fixture.AddAxis(Axis(0));
  for (int i = 1; i <= waiter_count; ++i) {
    fixture.AddAxis(Axis(static_cast<uint8_t>(i)));
  }

  // Create waiters that all wait on Axis(1), not Axis(0).
  std::vector<FrontierStorage<1>> frontier_storage(waiter_count);
  std::vector<iree_async_frontier_waiter_t> waiters(waiter_count);
  for (int i = 0; i < waiter_count; ++i) {
    frontier_storage[i].Initialize(1);
    frontier_storage[i].SetEntry(0, Axis(1), 1000000);  // Never satisfied.
    iree_async_frontier_tracker_wait(fixture.tracker(),
                                     frontier_storage[i].frontier(),
                                     MinimalCallback, nullptr, &waiters[i]);
  }

  uint64_t epoch = 1;
  for (auto _ : state) {
    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), epoch);
    benchmark::DoNotOptimize(dispatched);
    ++epoch;
  }

  // Cleanup: cancel all waiters.
  for (int i = 0; i < waiter_count; ++i) {
    iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiters[i]);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Advance_UnaffectedWaiters)
    ->Arg(1)
    ->Arg(4)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256);

// Advance that dispatches a single waiter.
// Measures the full dispatch path: CAS + satisfaction check + callback.
// NOTE: Uses PauseTiming which adds syscall overhead. See
// BM_Advance_DispatchOne_Clean.
static void BM_Advance_DispatchOne(benchmark::State& state) {
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  FrontierStorage<1> frontier_storage;
  frontier_storage.Initialize(1);
  frontier_storage.SetEntry(0, Axis(0), 1);

  iree_async_frontier_waiter_t waiter;
  uint64_t epoch = 1;
  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetEpochs();
    iree_async_frontier_tracker_wait(fixture.tracker(),
                                     frontier_storage.frontier(),
                                     MinimalCallback, nullptr, &waiter);
    state.ResumeTiming();

    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), epoch);
    benchmark::DoNotOptimize(dispatched);
    ++epoch;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Advance_DispatchOne);

// Clean measurement of dispatch path only (no wait overhead in timing).
// Pre-registers waiters, then measures just the advance+dispatch.
static void BM_Advance_DispatchOnly(benchmark::State& state) {
  const int kBatchSize = 1000;
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  // All waiters want epoch 1.
  std::vector<FrontierStorage<1>> frontiers(kBatchSize);
  std::vector<iree_async_frontier_waiter_t> waiters(kBatchSize);
  for (int i = 0; i < kBatchSize; ++i) {
    frontiers[i].Initialize(1);
    frontiers[i].SetEntry(0, Axis(0), 1);
  }

  for (auto _ : state) {
    state.PauseTiming();
    // Reset and register all waiters (not timed).
    fixture.ResetEpochs();
    for (int i = 0; i < kBatchSize; ++i) {
      iree_async_frontier_tracker_wait(fixture.tracker(),
                                       frontiers[i].frontier(), MinimalCallback,
                                       nullptr, &waiters[i]);
    }
    state.ResumeTiming();

    // This single advance dispatches all 1000 waiters.
    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 1);
    benchmark::DoNotOptimize(dispatched);
  }
  // Report per-waiter cost.
  state.SetItemsProcessed(state.iterations() * kBatchSize);
}
BENCHMARK(BM_Advance_DispatchOnly);

// Advance that dispatches multiple waiters on the same axis.
// Measures scaling behavior when many waiters are satisfied at once.
static void BM_Advance_DispatchMany(benchmark::State& state) {
  const int waiter_count = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  std::vector<FrontierStorage<1>> frontier_storage(waiter_count);
  std::vector<iree_async_frontier_waiter_t> waiters(waiter_count);
  for (int i = 0; i < waiter_count; ++i) {
    frontier_storage[i].Initialize(1);
    frontier_storage[i].SetEntry(0, Axis(0), 1);
  }

  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetEpochs();
    for (int i = 0; i < waiter_count; ++i) {
      iree_async_frontier_tracker_wait(fixture.tracker(),
                                       frontier_storage[i].frontier(),
                                       MinimalCallback, nullptr, &waiters[i]);
    }
    state.ResumeTiming();

    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 1);
    benchmark::DoNotOptimize(dispatched);
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["dispatched/op"] = waiter_count;
}
BENCHMARK(BM_Advance_DispatchMany)->Arg(1)->Arg(4)->Arg(16)->Arg(64);

// Advance with multi-entry frontiers.
// Measures the cost of the full satisfaction check (O(frontier_entries) reads).
static void BM_Advance_MultiEntryFrontier(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);

  for (int i = 0; i < entry_count; ++i) {
    fixture.AddAxis(Axis(static_cast<uint8_t>(i)));
  }

  // Frontier requires all axes at epoch 1.
  FrontierStorage<64> frontier_storage;
  frontier_storage.Initialize(entry_count);
  for (int i = 0; i < entry_count; ++i) {
    frontier_storage.SetEntry(i, Axis(static_cast<uint8_t>(i)), 1);
  }

  iree_async_frontier_waiter_t waiter;
  uint64_t epoch = 1;
  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetEpochs();
    // Advance all axes except the last one.
    for (int i = 0; i < entry_count - 1; ++i) {
      iree_async_frontier_tracker_advance(fixture.tracker(),
                                          Axis(static_cast<uint8_t>(i)), epoch);
    }
    iree_async_frontier_tracker_wait(fixture.tracker(),
                                     frontier_storage.frontier(),
                                     MinimalCallback, nullptr, &waiter);
    state.ResumeTiming();

    // Advance the last axis to trigger satisfaction check.
    iree_host_size_t dispatched = iree_async_frontier_tracker_advance(
        fixture.tracker(), Axis(static_cast<uint8_t>(entry_count - 1)), epoch);
    benchmark::DoNotOptimize(dispatched);
    ++epoch;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Advance_MultiEntryFrontier)
    ->Arg(1)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16);

//===----------------------------------------------------------------------===//
// Wait benchmarks
//===----------------------------------------------------------------------===//

// Wait on already-satisfied frontier: immediate dispatch fast path.
static void BM_Wait_ImmediatelySatisfied(benchmark::State& state) {
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));
  iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), 100);

  FrontierStorage<1> frontier_storage;
  frontier_storage.Initialize(1);
  frontier_storage.SetEntry(0, Axis(0), 50);  // Already satisfied.

  iree_async_frontier_waiter_t waiter;
  for (auto _ : state) {
    iree_status_t status = iree_async_frontier_tracker_wait(
        fixture.tracker(), frontier_storage.frontier(), MinimalCallback,
        nullptr, &waiter);
    benchmark::DoNotOptimize(status);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Wait_ImmediatelySatisfied);

// Wait on not-yet-satisfied frontier: pending insertion path.
// Uses a batched approach to avoid allocating millions of waiters.
static void BM_Wait_Pending(benchmark::State& state) {
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  FrontierStorage<1> frontier_storage;
  frontier_storage.Initialize(1);
  frontier_storage.SetEntry(0, Axis(0), 1000000);  // Never satisfied.

  iree_async_frontier_waiter_t waiter;
  for (auto _ : state) {
    iree_status_t status = iree_async_frontier_tracker_wait(
        fixture.tracker(), frontier_storage.frontier(), MinimalCallback,
        nullptr, &waiter);
    benchmark::DoNotOptimize(status);

    state.PauseTiming();
    iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Wait_Pending);

// Wait with multi-entry frontier satisfaction check.
static void BM_Wait_MultiEntrySatisfied(benchmark::State& state) {
  const int entry_count = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);

  for (int i = 0; i < entry_count; ++i) {
    fixture.AddAxis(Axis(static_cast<uint8_t>(i)));
    iree_async_frontier_tracker_advance(fixture.tracker(),
                                        Axis(static_cast<uint8_t>(i)), 100);
  }

  FrontierStorage<64> frontier_storage;
  frontier_storage.Initialize(entry_count);
  for (int i = 0; i < entry_count; ++i) {
    frontier_storage.SetEntry(i, Axis(static_cast<uint8_t>(i)), 50);
  }

  iree_async_frontier_waiter_t waiter;
  for (auto _ : state) {
    iree_status_t status = iree_async_frontier_tracker_wait(
        fixture.tracker(), frontier_storage.frontier(), MinimalCallback,
        nullptr, &waiter);
    benchmark::DoNotOptimize(status);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Wait_MultiEntrySatisfied)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

//===----------------------------------------------------------------------===//
// Cancel benchmarks
//===----------------------------------------------------------------------===//

// Cancel a waiter at the head of the list (best case).
static void BM_Cancel_Head(benchmark::State& state) {
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  FrontierStorage<1> frontier_storage;
  frontier_storage.Initialize(1);
  frontier_storage.SetEntry(0, Axis(0), 1000000);

  iree_async_frontier_waiter_t waiter;
  for (auto _ : state) {
    state.PauseTiming();
    iree_async_frontier_tracker_wait(fixture.tracker(),
                                     frontier_storage.frontier(),
                                     MinimalCallback, nullptr, &waiter);
    state.ResumeTiming();

    iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiter);
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Cancel_Head);

// Cancel a waiter at the tail of the list (worst case: full scan).
static void BM_Cancel_Tail(benchmark::State& state) {
  const int list_length = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  std::vector<FrontierStorage<1>> frontier_storage(list_length);
  std::vector<iree_async_frontier_waiter_t> waiters(list_length);

  // Pre-populate the list.
  for (int i = 0; i < list_length; ++i) {
    frontier_storage[i].Initialize(1);
    frontier_storage[i].SetEntry(0, Axis(0), 1000000);
  }

  for (auto _ : state) {
    state.PauseTiming();
    for (int i = 0; i < list_length; ++i) {
      iree_async_frontier_tracker_wait(fixture.tracker(),
                                       frontier_storage[i].frontier(),
                                       MinimalCallback, nullptr, &waiters[i]);
    }
    state.ResumeTiming();

    // Cancel the first one added (now at the tail due to prepend).
    iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiters[0]);

    state.PauseTiming();
    // Cleanup the rest.
    for (int i = 1; i < list_length; ++i) {
      iree_async_frontier_tracker_cancel_wait(fixture.tracker(), &waiters[i]);
    }
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Cancel_Tail)->Arg(4)->Arg(16)->Arg(64)->Arg(256);

//===----------------------------------------------------------------------===//
// End-to-end scenario benchmarks
//===----------------------------------------------------------------------===//

// Simulates the GPU completion hot path:
// GPU signals semaphore -> tracker advance -> waiter dispatch -> submit next op
static void BM_Scenario_GPUCompletionDispatch(benchmark::State& state) {
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  FrontierStorage<1> frontier_storage;
  frontier_storage.Initialize(1);
  frontier_storage.SetEntry(0, Axis(0), 1);

  iree_async_frontier_waiter_t waiter;
  uint64_t epoch = 1;
  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetEpochs();
    state.ResumeTiming();

    // Register a waiter (simulates operation waiting for GPU).
    iree_async_frontier_tracker_wait(fixture.tracker(),
                                     frontier_storage.frontier(),
                                     MinimalCallback, nullptr, &waiter);

    // Advance (simulates GPU completion signal).
    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), epoch);
    benchmark::DoNotOptimize(dispatched);
    ++epoch;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scenario_GPUCompletionDispatch);

// Simulates fan-in pattern: one waiter waiting on multiple GPU queues.
static void BM_Scenario_FanIn(benchmark::State& state) {
  const int queue_count = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);

  for (int i = 0; i < queue_count; ++i) {
    fixture.AddAxis(Axis(static_cast<uint8_t>(i)));
  }

  FrontierStorage<64> frontier_storage;
  frontier_storage.Initialize(queue_count);
  for (int i = 0; i < queue_count; ++i) {
    frontier_storage.SetEntry(i, Axis(static_cast<uint8_t>(i)), 1);
  }

  iree_async_frontier_waiter_t waiter;
  uint64_t epoch = 1;
  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetEpochs();
    state.ResumeTiming();

    // Register waiter waiting on all queues.
    iree_async_frontier_tracker_wait(fixture.tracker(),
                                     frontier_storage.frontier(),
                                     MinimalCallback, nullptr, &waiter);

    // Simulate queues completing one by one.
    for (int i = 0; i < queue_count; ++i) {
      iree_async_frontier_tracker_advance(fixture.tracker(),
                                          Axis(static_cast<uint8_t>(i)), epoch);
    }
    ++epoch;
  }
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Scenario_FanIn)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

// Simulates fan-out pattern: multiple waiters on a single GPU queue.
static void BM_Scenario_FanOut(benchmark::State& state) {
  const int waiter_count = static_cast<int>(state.range(0));
  TrackerFixture fixture(kMaxAxes);
  fixture.AddAxis(Axis(0));

  std::vector<FrontierStorage<1>> frontier_storage(waiter_count);
  std::vector<iree_async_frontier_waiter_t> waiters(waiter_count);
  for (int i = 0; i < waiter_count; ++i) {
    frontier_storage[i].Initialize(1);
    frontier_storage[i].SetEntry(0, Axis(0), 1);
  }

  uint64_t epoch = 1;
  for (auto _ : state) {
    state.PauseTiming();
    fixture.ResetEpochs();
    state.ResumeTiming();

    // Register all waiters.
    for (int i = 0; i < waiter_count; ++i) {
      iree_async_frontier_tracker_wait(fixture.tracker(),
                                       frontier_storage[i].frontier(),
                                       MinimalCallback, nullptr, &waiters[i]);
    }

    // Single advance dispatches all waiters.
    iree_host_size_t dispatched =
        iree_async_frontier_tracker_advance(fixture.tracker(), Axis(0), epoch);
    benchmark::DoNotOptimize(dispatched);
    ++epoch;
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["dispatched/op"] = waiter_count;
}
BENCHMARK(BM_Scenario_FanOut)->Arg(1)->Arg(4)->Arg(16)->Arg(64);

}  // namespace
