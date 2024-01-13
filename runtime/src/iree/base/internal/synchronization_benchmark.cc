// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !IREE_HAS_NOP_BENCHMARK_LIB

#include <cstddef>
#include <mutex>

#include "iree/base/internal/synchronization.h"
#include "iree/testing/benchmark_lib.h"

namespace {

//==============================================================================
// Inlined timing utils
//==============================================================================

void SpinDelay(int count, int* data) {
  // This emulates work we may be doing while holding the lock (like swapping
  // around some pointers).
  for (size_t i = 0; i < count * 10; ++i) {
    ++(*data);
    benchmark::DoNotOptimize(*data);
  }
}

//==============================================================================
// iree_mutex_t / iree_slim_mutex_t
//==============================================================================

void BM_Mutex(benchmark::State& state) {
  static iree_mutex_t* mu = ([]() -> iree_mutex_t* {
    auto mutex = new iree_mutex_t();
    iree_mutex_initialize(mutex);
    return mutex;
  })();
  for (auto _ : state) {
    iree_mutex_lock(mu);
    benchmark::DoNotOptimize(*mu);
    iree_mutex_unlock(mu);
  }
}
BENCHMARK(BM_Mutex)->UseRealTime()->Threads(1)->ThreadPerCpu();

template <typename MutexType>
class RaiiLocker;

template <>
class RaiiLocker<iree_mutex_t> {
 public:
  static void Initialize(iree_mutex_t* out_mu) {
    iree_mutex_initialize(out_mu);
  }
  static void Deinitialize(iree_mutex_t* mu) { iree_mutex_deinitialize(mu); }
  explicit RaiiLocker(iree_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)
      : mu_(mu) {
    iree_mutex_lock(mu_);
  }
  ~RaiiLocker() IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    iree_mutex_unlock(mu_);
  }

 private:
  iree_mutex_t* mu_;
};

template <>
class RaiiLocker<iree_slim_mutex_t> {
 public:
  static void Initialize(iree_slim_mutex_t* out_mu) {
    iree_slim_mutex_initialize(out_mu);
  }
  static void Deinitialize(iree_slim_mutex_t* mu) {
    iree_slim_mutex_deinitialize(mu);
  }
  explicit RaiiLocker(iree_slim_mutex_t* mu)
      IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)
      : mu_(mu) {
    iree_slim_mutex_lock(mu_);
  }
  ~RaiiLocker() IREE_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis) {
    iree_slim_mutex_unlock(mu_);
  }

 private:
  iree_slim_mutex_t* mu_;
};

template <>
class RaiiLocker<std::mutex> {
 public:
  static void Initialize(std::mutex* out_mu) {}
  static void Deinitialize(std::mutex* mu) {}
  explicit RaiiLocker(std::mutex* mu) : mu_(mu) { mu_->lock(); }
  ~RaiiLocker() { mu_->unlock(); }

 private:
  std::mutex* mu_;
};

template <typename MutexType>
void BM_CreateDelete(benchmark::State& state) {
  for (auto _ : state) {
    MutexType mu;
    RaiiLocker<MutexType>::Initialize(&mu);
    benchmark::DoNotOptimize(mu);
    RaiiLocker<MutexType>::Deinitialize(&mu);
  }
}

BENCHMARK_TEMPLATE(BM_CreateDelete, iree_mutex_t)->UseRealTime()->Threads(1);

BENCHMARK_TEMPLATE(BM_CreateDelete, iree_slim_mutex_t)
    ->UseRealTime()
    ->Threads(1);

BENCHMARK_TEMPLATE(BM_CreateDelete, std::mutex)->UseRealTime()->Threads(1);

template <typename MutexType>
void BM_Uncontended(benchmark::State& state) {
  MutexType mu;
  RaiiLocker<MutexType>::Initialize(&mu);
  int data = 0;
  int local = 0;
  for (auto _ : state) {
    // Here we model both local work outside of the critical section as well as
    // some work inside of the critical section. The idea is to capture some
    // more or less realisitic contention levels.
    // If contention is too low, the benchmark won't measure anything useful.
    // If contention is unrealistically high, the benchmark will favor
    // bad mutex implementations that block and otherwise distract threads
    // from the mutex and shared state for as much as possible.
    // To achieve this amount of local work is multiplied by number of threads
    // to keep ratio between local work and critical section approximately
    // equal regardless of number of threads.
    SpinDelay(100 * state.threads(), &local);
    RaiiLocker<MutexType> locker(&mu);
    SpinDelay(static_cast<int>(state.range(0)), &data);
  }
}

BENCHMARK_TEMPLATE(BM_Uncontended, iree_mutex_t)
    ->UseRealTime()
    ->Threads(1)
    ->Arg(50)
    ->Arg(200);

BENCHMARK_TEMPLATE(BM_Uncontended, iree_slim_mutex_t)
    ->UseRealTime()
    ->Threads(1)
    ->Arg(50)
    ->Arg(200);

BENCHMARK_TEMPLATE(BM_Uncontended, std::mutex)
    ->UseRealTime()
    ->Threads(1)
    ->Arg(50)
    ->Arg(200);

template <typename MutexType>
void BM_Contended(benchmark::State& state) {
  struct Shared {
    MutexType mu;
    int data = 0;
    Shared() { RaiiLocker<MutexType>::Initialize(&mu); }
  };
  static auto* shared = new Shared();
  int local = 0;
  for (auto _ : state) {
    // Here we model both local work outside of the critical section as well as
    // some work inside of the critical section. The idea is to capture some
    // more or less realisitic contention levels.
    // If contention is too low, the benchmark won't measure anything useful.
    // If contention is unrealistically high, the benchmark will favor
    // bad mutex implementations that block and otherwise distract threads
    // from the mutex and shared state for as much as possible.
    // To achieve this amount of local work is multiplied by number of threads
    // to keep ratio between local work and critical section approximately
    // equal regardless of number of threads.
    SpinDelay(100 * state.threads(), &local);
    RaiiLocker<MutexType> locker(&shared->mu);
    SpinDelay(static_cast<int>(state.range(0)), &shared->data);
  }
}

BENCHMARK_TEMPLATE(BM_Contended, iree_mutex_t)
    ->UseRealTime()
    // ThreadPerCpu poorly handles non-power-of-two CPU counts.
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(6)
    ->Threads(8)
    ->Threads(12)
    ->Threads(16)
    ->Threads(24)
    ->Threads(32)
    ->Threads(48)
    ->Threads(64)
    ->Threads(96)
    // Some empirically chosen amounts of work in critical section.
    // 1 is low contention, 200 is high contention and few values in between.
    ->Arg(50)
    ->Arg(200);

BENCHMARK_TEMPLATE(BM_Contended, iree_slim_mutex_t)
    ->UseRealTime()
    // ThreadPerCpu poorly handles non-power-of-two CPU counts.
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(6)
    ->Threads(8)
    ->Threads(12)
    ->Threads(16)
    ->Threads(24)
    ->Threads(32)
    ->Threads(48)
    ->Threads(64)
    ->Threads(96)
    // Some empirically chosen amounts of work in critical section.
    // 1 is low contention, 200 is high contention and few values in between.
    ->Arg(50)
    ->Arg(200);

BENCHMARK_TEMPLATE(BM_Contended, std::mutex)
    ->UseRealTime()
    // ThreadPerCpu poorly handles non-power-of-two CPU counts.
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(6)
    ->Threads(8)
    ->Threads(12)
    ->Threads(16)
    ->Threads(24)
    ->Threads(32)
    ->Threads(48)
    ->Threads(64)
    ->Threads(96)
    // Some empirically chosen amounts of work in critical section.
    // 1 is low contention, 200 is high contention and few values in between.
    ->Arg(50)
    ->Arg(200);

//==============================================================================
// iree_notification_t
//==============================================================================

// TODO(benvanik): benchmark this; it should in the worst case be as bad as
// mutex/futex (as that's what is used), but at the moment we don't really
// care beyond that.

}  // namespace

#endif  // !IREE_HAS_NOP_BENCHMARK_LIB
