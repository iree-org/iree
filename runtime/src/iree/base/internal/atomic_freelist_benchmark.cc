// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/atomic_freelist.h"
#include "iree/base/internal/memory.h"

namespace {

//==============================================================================
// Timing utilities
//==============================================================================

// Emulates work (like pointer manipulation) to create realistic contention.
void SpinDelay(int count, int* data) {
  for (size_t i = 0; i < count * 10; ++i) {
    ++(*data);
    benchmark::DoNotOptimize(*data);
  }
}

// Helper to manage slots array lifetime. Uses aligned allocation for atomics.
class SlotsArray {
 public:
  explicit SlotsArray(size_t count) : count_(count) {
    if (count > 0) {
      IREE_CHECK_OK(iree_aligned_alloc(alignof(iree_atomic_freelist_slot_t),
                                       count * sizeof(slots_[0]),
                                       (void**)&slots_));
    }
  }
  ~SlotsArray() {
    if (slots_) iree_aligned_free(slots_);
  }
  iree_atomic_freelist_slot_t* data() { return slots_; }
  size_t size() const { return count_; }

 private:
  iree_atomic_freelist_slot_t* slots_ = nullptr;
  size_t count_ = 0;
};

//==============================================================================
// Single-threaded benchmarks
//==============================================================================

// Single-threaded acquire/release cycle.
static void BM_AcquireRelease(benchmark::State& state) {
  const size_t count = static_cast<size_t>(state.range(0));
  SlotsArray slots(count);
  iree_atomic_freelist_t freelist;
  iree_status_ignore(
      iree_atomic_freelist_initialize(slots.data(), count, &freelist));

  // Pre-acquire one to benchmark the cycle.
  uint16_t index;
  iree_atomic_freelist_try_pop(&freelist, slots.data(), &index);

  for (auto _ : state) {
    iree_atomic_freelist_push(&freelist, slots.data(), index);
    iree_atomic_freelist_try_pop(&freelist, slots.data(), &index);
  }

  // Return the index.
  iree_atomic_freelist_push(&freelist, slots.data(), index);
  iree_atomic_freelist_deinitialize(&freelist);

  state.SetItemsProcessed(state.iterations() * 2);  // push + pop per iteration
}
BENCHMARK(BM_AcquireRelease)->Arg(64)->Arg(256)->Arg(1024);

// Acquire-only until exhaustion, then release all.
static void BM_AcquireOnly(benchmark::State& state) {
  const size_t count = static_cast<size_t>(state.range(0));
  SlotsArray slots(count);
  std::vector<uint16_t> acquired(count);
  iree_atomic_freelist_t freelist;

  for (auto _ : state) {
    state.PauseTiming();
    iree_status_ignore(
        iree_atomic_freelist_initialize(slots.data(), count, &freelist));
    state.ResumeTiming();

    // Acquire all.
    for (size_t i = 0; i < count; ++i) {
      iree_atomic_freelist_try_pop(&freelist, slots.data(), &acquired[i]);
    }

    state.PauseTiming();
    // Release all for next iteration.
    for (size_t i = 0; i < count; ++i) {
      iree_atomic_freelist_push(&freelist, slots.data(), acquired[i]);
    }
    state.ResumeTiming();
  }

  iree_atomic_freelist_deinitialize(&freelist);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}
BENCHMARK(BM_AcquireOnly)->Arg(64)->Arg(256)->Arg(1024);

// Release-only (push) from full acquisition.
static void BM_ReleaseOnly(benchmark::State& state) {
  const size_t count = static_cast<size_t>(state.range(0));
  SlotsArray slots(count);
  std::vector<uint16_t> acquired(count);
  iree_atomic_freelist_t freelist;

  for (auto _ : state) {
    state.PauseTiming();
    iree_status_ignore(
        iree_atomic_freelist_initialize(slots.data(), count, &freelist));
    // Acquire all.
    for (size_t i = 0; i < count; ++i) {
      iree_atomic_freelist_try_pop(&freelist, slots.data(), &acquired[i]);
    }
    state.ResumeTiming();

    // Release all (timed).
    for (size_t i = 0; i < count; ++i) {
      iree_atomic_freelist_push(&freelist, slots.data(), acquired[i]);
    }
  }

  iree_atomic_freelist_deinitialize(&freelist);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
}
BENCHMARK(BM_ReleaseOnly)->Arg(64)->Arg(256)->Arg(1024);

//==============================================================================
// Mutex-based freelist (comparison baseline)
//==============================================================================

class MutexFreelist {
 public:
  MutexFreelist(size_t count)
      : count_(count), slots_(count), available_(count) {
    for (size_t i = 0; i < count - 1; ++i) {
      slots_[i] = static_cast<uint16_t>(i + 1);
    }
    slots_[count - 1] = UINT16_MAX;
    head_ = 0;
  }

  bool try_pop(uint16_t* out_index) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (head_ == UINT16_MAX) return false;
    *out_index = head_;
    head_ = slots_[head_];
    --available_;
    return true;
  }

  void push(uint16_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    slots_[index] = head_;
    head_ = index;
    ++available_;
  }

 private:
  size_t count_ = 0;
  std::vector<uint16_t> slots_;
  uint16_t head_ = 0;
  size_t available_ = 0;
  std::mutex mutex_;
};

static void BM_MutexAcquireRelease(benchmark::State& state) {
  const size_t count = static_cast<size_t>(state.range(0));
  MutexFreelist freelist(count);

  // Pre-acquire one to benchmark the cycle.
  uint16_t index;
  freelist.try_pop(&index);

  for (auto _ : state) {
    freelist.push(index);
    freelist.try_pop(&index);
  }

  freelist.push(index);
  state.SetItemsProcessed(state.iterations() * 2);
}
BENCHMARK(BM_MutexAcquireRelease)->Arg(64)->Arg(256)->Arg(1024);

//==============================================================================
// Multi-threaded contention benchmarks
//==============================================================================

// Shared state for atomic freelist contention benchmark.
// Keyed by pool size to allow parameterization.
struct AtomicFreelistShared {
  iree_atomic_freelist_t freelist;
  iree_atomic_freelist_slot_t* slots;
  size_t pool_size;

  explicit AtomicFreelistShared(size_t pool_size_) : pool_size(pool_size_) {
    IREE_CHECK_OK(iree_aligned_alloc(alignof(iree_atomic_freelist_slot_t),
                                     pool_size * sizeof(slots[0]),
                                     (void**)&slots));
    iree_status_ignore(
        iree_atomic_freelist_initialize(slots, pool_size, &freelist));
  }
  ~AtomicFreelistShared() {
    iree_atomic_freelist_deinitialize(&freelist);
    if (slots) iree_aligned_free(slots);
  }
};

static std::map<size_t, std::unique_ptr<AtomicFreelistShared>>&
GetAtomicSharedMap() {
  static std::map<size_t, std::unique_ptr<AtomicFreelistShared>> instances;
  return instances;
}

static AtomicFreelistShared* GetAtomicShared(size_t pool_size) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  auto& instances = GetAtomicSharedMap();
  auto& ptr = instances[pool_size];
  if (!ptr) {
    ptr = std::make_unique<AtomicFreelistShared>(pool_size);
  }
  return ptr.get();
}

// Multi-threaded contention benchmark for atomic freelist.
// Args: [0] = pool size, [1] = hold time (SpinDelay count)
static void BM_FreelistContendedCycle(benchmark::State& state) {
  const size_t pool_size = static_cast<size_t>(state.range(0));
  const int hold_time = static_cast<int>(state.range(1));
  auto* shared = GetAtomicShared(pool_size);

  int local_work = 0;
  int hold_work = 0;
  uint64_t successful_cycles = 0;

  for (auto _ : state) {
    // Local work outside critical section (scaled by thread count to maintain
    // consistent contention ratio regardless of parallelism level).
    SpinDelay(100 * state.threads(), &local_work);

    uint16_t index;
    if (iree_atomic_freelist_try_pop(&shared->freelist, shared->slots,
                                     &index)) {
      // Hold time (work while item acquired). Uses thread-local variable.
      SpinDelay(hold_time, &hold_work);
      iree_atomic_freelist_push(&shared->freelist, shared->slots, index);
      ++successful_cycles;
    }
  }

  // Report throughput as successful pop+push cycles per second.
  state.counters["cycles/s"] = benchmark::Counter(
      static_cast<double>(successful_cycles), benchmark::Counter::kIsRate);
  state.counters["success_rate"] =
      benchmark::Counter(static_cast<double>(successful_cycles) /
                             static_cast<double>(state.iterations()),
                         benchmark::Counter::kAvgThreads);
}
BENCHMARK(BM_FreelistContendedCycle)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16)
    ->Threads(32)
    ->Threads(64)
    // Args: {pool_size, hold_time}
    ->Args({64, 50})
    ->Args({256, 50})
    ->Args({512, 50})
    ->Args({64, 200})
    ->Args({256, 200})
    ->Args({512, 200});

// Shared state for mutex freelist contention benchmark.
struct MutexFreelistShared {
  MutexFreelist freelist;

  explicit MutexFreelistShared(size_t pool_size) : freelist(pool_size) {}
};

static std::map<size_t, std::unique_ptr<MutexFreelistShared>>&
GetMutexSharedMap() {
  static std::map<size_t, std::unique_ptr<MutexFreelistShared>> instances;
  return instances;
}

static MutexFreelistShared* GetMutexShared(size_t pool_size) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  auto& instances = GetMutexSharedMap();
  auto& ptr = instances[pool_size];
  if (!ptr) {
    ptr = std::make_unique<MutexFreelistShared>(pool_size);
  }
  return ptr.get();
}

// Multi-threaded contention benchmark for mutex-based freelist.
// Provides direct comparison baseline against atomic freelist.
static void BM_MutexContendedCycle(benchmark::State& state) {
  const size_t pool_size = static_cast<size_t>(state.range(0));
  const int hold_time = static_cast<int>(state.range(1));
  auto* shared = GetMutexShared(pool_size);

  int local_work = 0;
  int hold_work = 0;
  uint64_t successful_cycles = 0;

  for (auto _ : state) {
    // Local work outside critical section (scaled by thread count).
    SpinDelay(100 * state.threads(), &local_work);

    uint16_t index;
    if (shared->freelist.try_pop(&index)) {
      // Hold time (work while item acquired). Uses thread-local variable.
      SpinDelay(hold_time, &hold_work);
      shared->freelist.push(index);
      ++successful_cycles;
    }
  }

  // Report throughput as successful pop+push cycles per second.
  state.counters["cycles/s"] = benchmark::Counter(
      static_cast<double>(successful_cycles), benchmark::Counter::kIsRate);
  state.counters["success_rate"] =
      benchmark::Counter(static_cast<double>(successful_cycles) /
                             static_cast<double>(state.iterations()),
                         benchmark::Counter::kAvgThreads);
}
BENCHMARK(BM_MutexContendedCycle)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16)
    ->Threads(32)
    ->Threads(64)
    // Args: {pool_size, hold_time}
    ->Args({64, 50})
    ->Args({256, 50})
    ->Args({512, 50})
    ->Args({64, 200})
    ->Args({256, 200})
    ->Args({512, 200});

//==============================================================================
// Saturation benchmark
//==============================================================================

// Measures behavior when threads exceed pool size.
// Reports success_rate counter showing what fraction of pop attempts succeed.
static void BM_FreelistSaturation(benchmark::State& state) {
  const size_t pool_size = static_cast<size_t>(state.range(0));
  auto* shared = GetAtomicShared(pool_size);

  uint64_t attempts = 0;
  uint64_t successes = 0;

  for (auto _ : state) {
    ++attempts;
    uint16_t index;
    if (iree_atomic_freelist_try_pop(&shared->freelist, shared->slots,
                                     &index)) {
      ++successes;
      // Minimal hold time (immediate push after pop).
      iree_atomic_freelist_push(&shared->freelist, shared->slots, index);
    }
  }

  // Report success rate and throughput.
  if (attempts > 0) {
    state.counters["success_rate"] = benchmark::Counter(
        static_cast<double>(successes) / static_cast<double>(attempts),
        benchmark::Counter::kAvgThreads);
    state.counters["cycles/s"] = benchmark::Counter(
        static_cast<double>(successes), benchmark::Counter::kIsRate);
  }
}
BENCHMARK(BM_FreelistSaturation)
    ->UseRealTime()
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8)
    ->Threads(16)
    ->Threads(32)
    ->Threads(64)
    ->Threads(128)
    ->Arg(64)
    ->Arg(256)
    ->Arg(512);

}  // namespace

BENCHMARK_MAIN();
